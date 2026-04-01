"""SiamFC tracker — inference-only.

Implements the classic SiamFC tracking loop following zllrunning/SiameseX.PyTorch
(demo_utils/siamvggtracker.py):
  * template extraction at initialisation
  * multi-scale search + cosine-window response at each frame
  * linear-interpolation scale update and position refinement

Usage::

    from trackers.siamX.siamx_fc.siam_fc_tracker import SiamFCTracker

    tracker = SiamFCTracker(model_path='trackers/siamX/pretrained_weights/SiamFC.pth')
    tracker.initialize(0, first_frame, [x, y, w, h])  # top-left bbox
    for frame in subsequent_frames:
        bbox, score, scale_id = tracker.track(0, frame)
        # bbox -> [x, y, w, h] (top-left)
"""

from __future__ import annotations

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from .siamx_siamese import SiameseNet
from .parse_arguments import load_siamfc_params


# ---------------------------------------------------------------------------
# Position update helper (ported from SiameseX.PyTorch siamvggtracker.py)
# ---------------------------------------------------------------------------

def _update_target_position(
    pos_x: float,
    pos_y: float,
    score: np.ndarray,
    final_score_sz: int,
    tot_stride: int,
    search_sz: int,
    response_up: int,
    x_sz: float,
) -> Tuple[float, float]:
    """Map the peak of *score* back to image-pixel displacement and update pos.

    Faithfully reproduces the formula from
    ``demo_utils/siamvggtracker.py::_update_target_position``.
    """
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    pos_y = pos_y + disp_in_frame[0]
    pos_x = pos_x + disp_in_frame[1]
    return float(pos_x), float(pos_y)


class SiamFCTracker:
    """Single-object SiamFC tracker (dict-keyed state for multi-object support).

    Tracking logic faithfully ported from zllrunning/SiameseX.PyTorch
    ``demo_utils/siamvggtracker.py``.

    Args:
        model_path: Path to the pretrained ``SiamFC.pth`` weight file.
        device:     Torch device string (``'cuda'`` or ``'cpu'``).
        params_dir: Directory containing ``hyperparams.json`` /
                    ``design.json``.  ``None`` uses the bundled defaults.
        use_trt:    Pass ``True`` to enable TensorRT acceleration.
                    Requires TensorRT and PyCUDA; falls back to PyTorch
                    silently when unavailable.
    """

    # ------------------------------------------------------------------
    # Embedded parameter classes (fallback when JSON is absent)
    # ------------------------------------------------------------------

    class Params:
        class Hyper:
            scale_num        = 3
            scale_step       = 1.040
            scale_penalty    = 0.97
            scale_lr         = 0.59
            window_influence = 0.25
            response_up      = 8

        class Design:
            exemplar_sz    = 127
            search_sz      = 255
            score_sz       = 33
            tot_stride     = 4
            final_score_sz = 273
            context_amount = 0.5

    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        params_dir: Optional[str] = None,
        use_trt: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device_str = device

        # Load hyperparameters
        try:
            self.hyper, self.design = load_siamfc_params(params_dir)
        except Exception:
            self.hyper  = self.Params.Hyper()
            self.design = self.Params.Design()

        # Build siamese network
        self.net = SiameseNet(model_path=model_path, device=device,
                              use_trt=use_trt)

        # Normalised cosine (Hann) window — matches SiameseX.PyTorch penalty
        final_sz = self.design.final_score_sz
        hann_1d = np.hanning(final_sz)
        self._cos_window = np.outer(hann_1d, hann_1d).astype(np.float32)
        self._cos_window /= np.sum(self._cos_window)

        # Scale factors: step ** linspace(-ceil(n/2), ceil(n/2), n)
        # Matches SiameseX.PyTorch exactly (for n=3 gives [-2, 0, 2])
        num_s = self.hyper.scale_num
        step  = self.hyper.scale_step
        _half = math.ceil(num_s / 2)
        self._scale_factors = step ** np.linspace(-_half, _half, num_s)

        # Per-tracker state (indexed by _id)
        self._pos_y:    Dict[int, float] = {}
        self._pos_x:    Dict[int, float] = {}
        self._target_h: Dict[int, float] = {}
        self._target_w: Dict[int, float] = {}
        self._z_sz:     Dict[int, float] = {}
        self._x_sz:     Dict[int, float] = {}
        self._template: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, _id: int, im: np.ndarray, bbox: List[float]) -> None:
        """Initialise tracker *_id* on *im* using bounding box *bbox*.

        Args:
            _id:  Tracker identifier (use 0 for single-object tracking).
            im:   First frame as BGR uint8 ndarray (H, W, 3).
            bbox: Initial bounding box [x, y, w, h] in **top-left** format.
        """
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = x + w / 2
        cy = y + h / 2

        # Context size (matches SiameseX.PyTorch: context = design.context)
        p    = self.design.context_amount * (w + h)
        z_sz = math.sqrt((w + p) * (h + p))
        x_sz = float(self.design.search_sz) / self.design.exemplar_sz * z_sz

        self._pos_y[_id]    = cy
        self._pos_x[_id]    = cx
        self._target_h[_id] = h
        self._target_w[_id] = w
        self._z_sz[_id]     = z_sz
        self._x_sz[_id]     = x_sz

        template = self.net.get_template_z_new(cx, cy, z_sz, im, self.design)
        self._template[_id] = template

    def track(
        self,
        _id: int,
        im: np.ndarray,
    ) -> Tuple[List[float], float, int]:
        """Track object *_id* in frame *im*.

        Implements the tracking loop from SiameseX.PyTorch
        ``SiamVGGTracker.track()`` faithfully:
          1. Compute multi-scale response maps.
          2. Apply scale penalty to non-centre scales.
          3. Select best scale by max response.
          4. Update x_sz / target_w / target_h with linear interpolation.
          5. Normalise the selected score map and blend cosine window.
          6. Refine position using exact stride/response_up displacement formula.

        Args:
            _id: Tracker identifier (use 0 for single-object tracking).
            im:  Current frame as BGR uint8 ndarray (H, W, 3).

        Returns:
            (bbox, score, scale_id) where *bbox* is [x, y, w, h] top-left,
            *score* is the peak windowed response, and *scale_id* is the index
            of the selected scale in ``self._scale_factors``.
        """
        pos_y = self._pos_y[_id]
        pos_x = self._pos_x[_id]
        x_sz  = self._x_sz[_id]

        scaled_search_area = x_sz * self._scale_factors          # (scale_num,)
        scaled_target_w    = self._target_w[_id] * self._scale_factors
        scaled_target_h    = self._target_h[_id] * self._scale_factors

        # -- Get response maps for all scales --
        scores = self.net.get_scores_new(
            pos_x, pos_y,
            scaled_search_area.tolist(),
            self._template[_id],
            im,
            self.design,
            self.design.final_score_sz,
        )  # torch Tensor (scale_num, final_score_sz, final_score_sz)

        scores_np = scores.cpu().numpy()  # (scale_num, H, W)

        # -- Scale penalty (penalise all non-centre scales) --
        mid = len(self._scale_factors) // 2
        for i in range(len(self._scale_factors)):
            if i != mid:
                scores_np[i] *= self.hyper.scale_penalty

        new_scale_id = int(np.argmax(np.amax(scores_np, axis=(1, 2))))

        # -- Linear-interpolation scale update (SiameseX.PyTorch convention) --
        lr = self.hyper.scale_lr
        self._x_sz[_id] = (
            (1 - lr) * x_sz
            + lr * scaled_search_area[new_scale_id]
        )
        self._target_w[_id] = (
            (1 - lr) * self._target_w[_id]
            + lr * scaled_target_w[new_scale_id]
        )
        self._target_h[_id] = (
            (1 - lr) * self._target_h[_id]
            + lr * scaled_target_h[new_scale_id]
        )

        # -- Score normalisation + cosine-window blending --
        score = scores_np[new_scale_id].copy()
        score -= np.min(score)
        score_sum = np.sum(score)
        if score_sum > 0:
            score /= score_sum
        # When score_sum == 0 (degenerate response), the blending below
        # reduces to window_influence * cos_window, biasing toward the centre.
        score = (
            (1 - self.hyper.window_influence) * score
            + self.hyper.window_influence * self._cos_window
        )

        # -- Position refinement --
        pos_x, pos_y = _update_target_position(
            pos_x, pos_y, score,
            self.design.final_score_sz,
            self.design.tot_stride,
            self.design.search_sz,
            self.hyper.response_up,
            self._x_sz[_id],
        )
        self._pos_x[_id] = pos_x
        self._pos_y[_id] = pos_y

        bbox: List[float] = [
            pos_x - self._target_w[_id] / 2.0,
            pos_y - self._target_h[_id] / 2.0,
            self._target_w[_id],
            self._target_h[_id],
        ]
        peak_score = float(np.max(score))
        return bbox, peak_score, new_scale_id
