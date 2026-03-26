"""SiamFC tracker — inference-only.

Implements the classic SiamFC tracking loop:
  * template extraction at initialisation
  * multi-scale search + cosine-window response at each frame
  * scale-LR update and position refinement

Usage::

    from trackers.siamX.siamx_fc.siam_fc_tracker import SiamFCTracker

    tracker = SiamFCTracker(model_path='trackers/siamX/pretrained_weights/SiamFC.pth')
    tracker.initialize(0, first_frame, [x, y, w, h])  # top-left bbox
    for frame in subsequent_frames:
        bbox, score, scale_id = tracker.track(0, frame)
        # bbox -> [x, y, w, h] (top-left)

Adapted from deep_mdp's siamx_fc/siam_fc_tracker.py.
"""

from __future__ import annotations

import os
import math
import numpy as np
import torch

from .siamx_siamese import SiameseNet
from .parse_arguments import load_siamfc_params


class SiamFCTracker:
    """Single-object SiamFC tracker (dict-keyed for compatibility with deep_mdp
    multi-tracker interface).

    Args:
        model_path: Path to the pretrained ``SiamFC.pth`` weight file.
        device:     Torch device string (``'cuda'`` or ``'cpu'``).
        params_dir: Directory containing ``hyperparams.json`` /
                    ``design.json``.  ``None`` → use bundled defaults.
        use_trt:    Pass ``True`` to enable TensorRT acceleration for the
                    backbone.  Requires TensorRT and PyCUDA; silently falls
                    back to PyTorch when unavailable.
    """

    # ------------------------------------------------------------------
    # Embedded parameter classes (used as fallback when JSON is absent)
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

    def __init__(self, model_path: str | None = None,
                 device: str = 'cpu',
                 params_dir: str | None = None,
                 use_trt: bool = False):
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

        # Build cosine window (applied to the upsampled response map)
        final_sz = self.design.final_score_sz
        hann_1d = np.hanning(final_sz)
        self._cos_window = np.outer(hann_1d, hann_1d).astype(np.float32)

        # Build scale factors (sorted smallest → largest)
        num_s = self.hyper.scale_num
        step  = self.hyper.scale_step
        self._scale_factors = step ** np.arange(
            -(num_s // 2), num_s // 2 + 1
        )  # shape (scale_num,)

        # Per-tracker state (indexed by _id)
        self._pos_y:     dict[int, float] = {}
        self._pos_x:     dict[int, float] = {}
        self._target_h:  dict[int, float] = {}
        self._target_w:  dict[int, float] = {}
        self._z_sz:      dict[int, float] = {}
        self._x_sz:      dict[int, float] = {}
        self._template:  dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, _id: int, im: np.ndarray, bbox: list[float]) -> None:
        """Initialise tracker *_id* on *im* using bounding box *bbox*.

        Args:
            _id:  Tracker identifier (use 0 for single-object tracking).
            im:   First frame as BGR uint8 ndarray (H, W, 3).
            bbox: Initial bounding box [x, y, w, h] in **top-left** format.
        """
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = x + w / 2
        cy = y + h / 2

        # Context size
        p  = self.design.context_amount * (w + h)
        z_sz = math.sqrt((w + p) * (h + p))
        x_sz = z_sz * math.sqrt(self.design.search_sz / self.design.exemplar_sz)

        self._pos_y[_id]    = cy
        self._pos_x[_id]    = cx
        self._target_h[_id] = h
        self._target_w[_id] = w
        self._z_sz[_id]     = z_sz
        self._x_sz[_id]     = x_sz

        template = self.net.get_template_z_new(
            cx, cy, z_sz, im, self.design
        )
        self._template[_id] = template

    def track(self, _id: int, im: np.ndarray
              ) -> tuple[list[float], float, int]:
        """Track object *_id* in frame *im*.

        Args:
            _id: Tracker identifier (use 0 for single-object tracking).
            im:  Current frame as BGR uint8 ndarray (H, W, 3).

        Returns:
            (bbox, score, scale_id) where *bbox* is [x, y, w, h] top-left,
            *score* is the peak response value, and *scale_id* is the index
            of the selected scale in ``self._scale_factors``.
        """
        pos_y = self._pos_y[_id]
        pos_x = self._pos_x[_id]
        x_sz  = self._x_sz[_id]
        z_sz  = self._z_sz[_id]

        scaled_search = x_sz * self._scale_factors  # (scale_num,)

        scores = self.net.get_scores_new(
            pos_x, pos_y,
            scaled_search.tolist(),
            self._template[_id],
            im,
            self.design,
            self.design.final_score_sz,
        )  # (scale_num, final_sz, final_sz)

        scores_np = scores.cpu().numpy()  # (scale_num, H, H)

        # Apply scale penalty to off-scale responses
        scores_penalised = scores_np.copy()
        mid = len(self._scale_factors) // 2
        for i, sf in enumerate(self._scale_factors):
            if i != mid:
                scores_penalised[i] *= self.hyper.scale_penalty

        # Apply cosine window
        scores_windowed = (
            (1 - self.hyper.window_influence) * scores_penalised
            + self.hyper.window_influence * self._cos_window[np.newaxis]
        )

        # Best scale
        scale_id = int(np.argmax([s.max() for s in scores_windowed]))
        best_score_map = scores_windowed[scale_id]
        peak_score     = float(scores_penalised[scale_id].max())

        # Peak location in the upsampled score map
        final_sz = self.design.final_score_sz
        flat_idx = int(np.argmax(best_score_map))
        p_y = flat_idx // final_sz
        p_x = flat_idx  % final_sz

        # Convert peak to displacement from the crop centre
        displacement_x = (p_x - final_sz / 2) / final_sz
        displacement_y = (p_y - final_sz / 2) / final_sz

        # Scale displacement back to original image pixels
        # (the search crop covers scaled_search[scale_id] pixels)
        disp_x = displacement_x * scaled_search[scale_id]
        disp_y = displacement_y * scaled_search[scale_id]

        # Update position
        pos_x_new = pos_x + disp_x
        pos_y_new = pos_y + disp_y

        # Update target size with scale LR
        sf = self._scale_factors[scale_id]
        lr = self.hyper.scale_lr
        new_w = self._target_w[_id] * (sf ** lr)
        new_h = self._target_h[_id] * (sf ** lr)
        new_z_sz = z_sz * (sf ** lr)
        new_x_sz = new_z_sz * math.sqrt(self.design.search_sz / self.design.exemplar_sz)

        self._pos_x[_id]    = pos_x_new
        self._pos_y[_id]    = pos_y_new
        self._target_w[_id] = new_w
        self._target_h[_id] = new_h
        self._z_sz[_id]     = new_z_sz
        self._x_sz[_id]     = new_x_sz

        bbox = [
            pos_x_new - new_w / 2,
            pos_y_new - new_h / 2,
            new_w,
            new_h,
        ]
        return bbox, peak_score, scale_id
