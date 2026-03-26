"""SiamFCOmniTracker — SiamFC adapted for 360° omnidirectional tracking.

This module wraps the planar SiamFC tracker inside the 360VOT sphere-to-plane
projection pipeline:

  1. :meth:`initialize`: crop the first equirectangular frame around the
     ground-truth BFoV annotation → initialise SiamFC with the local view.
  2. :meth:`track`: crop the next frame centred on the last predicted BFoV
     (scaled up for context) → run SiamFC → project local bbox back to
     spherical BFoV.

Usage::

    from trackers.siamfc_omni import SiamFCOmniTracker
    from lib.utils import dict2Bfov
    import cv2

    tracker = SiamFCOmniTracker(
        model_path='trackers/siamX/pretrained_weights/SiamFC.pth'
    )

    first_frame = cv2.imread('frame_0001.jpg')
    init_bfov   = dict2Bfov({'clon': 10, 'clat': 5, 'fov_h': 20,
                              'fov_v': 15, 'rotation': 0})
    tracker.initialize(first_frame, init_bfov)

    for frame_path in frame_paths[1:]:
        frame = cv2.imread(frame_path)
        pred_bfov = tracker.track(frame)
        print(pred_bfov.todict())
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
from typing import Optional

# Allow running from the repo root without explicit install
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'lib')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from lib.omni import OmniImage
from lib.utils import Bfov, Bbox, scaleBFoV

from trackers.siamX.siamx_fc.siam_fc_tracker import SiamFCTracker


class SiamFCOmniTracker:
    """SiamFC tracker adapted for 360° omnidirectional tracking.

    Args:
        model_path:   Path to the ``SiamFC.pth`` pretrained weight file.
                      Pass ``None`` to run with random weights (testing only).
        img_w:        Width  of the equirectangular input image in pixels.
        img_h:        Height of the equirectangular input image in pixels.
        search_scale: Multiplier applied to the predicted BFoV when defining
                      the search region for the next frame.  Values around
                      2.0 give the tracker sufficient context to handle
                      moderate motion.
        crop_size:    Number of horizontal pixels in the local perspective
                      crop produced by :meth:`OmniImage.crop_bfov`.
        device:       Torch device for the SiamFC model (``'cuda'`` or
                      ``'cpu'``).
        use_trt:      Pass ``True`` to enable TensorRT acceleration for the
                      SiamFC backbone.  Requires TensorRT and PyCUDA; falls
                      back to PyTorch silently when unavailable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        img_w: int = 1920,
        img_h: int = 960,
        search_scale: float = 2.0,
        crop_size: int = 500,
        device: str = 'cpu',
        use_trt: bool = False,
    ):
        self.omni = OmniImage(img_w=img_w, img_h=img_h)
        self.search_scale = search_scale
        self.crop_size = crop_size
        self.tracker = SiamFCTracker(model_path=model_path, device=device,
                                     use_trt=use_trt)

        # State
        self._last_bfov: Optional[Bfov] = None
        self._last_u: Optional[np.ndarray] = None
        self._last_v: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def initialize(self, img: np.ndarray, bfov: Bfov) -> None:
        """Initialise the tracker on the first frame.

        Args:
            img:  Equirectangular panoramic frame (H, W, 3) BGR uint8.
            bfov: Ground-truth BFoV annotation for the first frame.
        """
        # Crop the panorama at the GT BFoV scaled up for context
        search_bfov = scaleBFoV(bfov, self.search_scale)
        crop, u_map, v_map = self.omni.crop_bfov(
            img, search_bfov, num_sample_h=self.crop_size
        )

        crop_h, crop_w = crop.shape[:2]

        # Target occupies the centre 1/search_scale portion of the crop
        target_w = crop_w / self.search_scale
        target_h = crop_h / self.search_scale
        cx = crop_w / 2.0
        cy = crop_h / 2.0

        # top-left bbox [x, y, w, h]
        init_bbox = [cx - target_w / 2, cy - target_h / 2, target_w, target_h]
        self.tracker.initialize(0, crop, init_bbox)

        self._last_bfov = bfov
        self._last_u = u_map
        self._last_v = v_map

    def track(self, img: np.ndarray) -> Bfov:
        """Track the object in a new frame.

        Args:
            img: Equirectangular panoramic frame (H, W, 3) BGR uint8.

        Returns:
            Predicted BFoV for this frame.
        """
        if self._last_bfov is None:
            raise RuntimeError("Call initialize() before track().")

        # Crop around the last predicted BFoV (scaled for search context)
        search_bfov = scaleBFoV(self._last_bfov, self.search_scale)
        crop, u_map, v_map = self.omni.crop_bfov(
            img, search_bfov, num_sample_h=self.crop_size
        )

        crop_h, crop_w = crop.shape[:2]

        # Recompute expected target size in this crop's coordinate system.
        # The GT/predicted BFoV occupies the central 1/search_scale portion.
        target_w = float(crop_w) / self.search_scale
        target_h = float(crop_h) / self.search_scale

        context = self.tracker.design.context_amount
        p = context * (target_w + target_h)
        z_sz = math.sqrt((target_w + p) * (target_h + p))
        x_sz = z_sz * math.sqrt(
            self.tracker.design.search_sz / self.tracker.design.exemplar_sz
        )

        # Reset the tracker's internal state to the centre of the new crop
        self.tracker._pos_x[0]    = crop_w / 2.0
        self.tracker._pos_y[0]    = crop_h / 2.0
        self.tracker._target_w[0] = target_w
        self.tracker._target_h[0] = target_h
        self.tracker._z_sz[0]     = z_sz
        self.tracker._x_sz[0]     = x_sz

        local_bbox, _score, _scale_id = self.tracker.track(0, crop)
        # local_bbox: [x, y, w, h] top-left

        x, y, w, h = local_bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Clamp to crop bounds to avoid out-of-range remap
        cx = float(np.clip(cx, 0, crop_w - 1))
        cy = float(np.clip(cy, 0, crop_h - 1))
        w  = float(np.clip(w, 1, crop_w))
        h  = float(np.clip(h, 1, crop_h))

        local_box = Bbox(cx, cy, w, h, rotation=0)

        # Project local bbox back to global spherical BFoV
        pred_bfov = self.omni.localBbox2Bfov(local_box, u_map, v_map,
                                              need_rotation=False)

        self._last_bfov = pred_bfov
        self._last_u = u_map
        self._last_v = v_map

        return pred_bfov
