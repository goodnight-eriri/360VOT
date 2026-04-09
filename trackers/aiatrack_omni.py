"""AiATrackOmniTracker — AiATrack adapted for 360° omnidirectional tracking.

This module wraps the planar AiATrack tracker (ECCV'22) inside the 360VOT
sphere-to-plane projection pipeline:

  1. :meth:`initialize`: crop the first equirectangular frame around the
     ground-truth BFoV annotation -> initialise AiATrack with the local view.
  2. :meth:`track`: crop the next frame centred on the last predicted BFoV
     (scaled up for context) -> run AiATrack -> project local bbox back to
     spherical BFoV.

AiATrack repository: https://github.com/Little-Podi/AiATrack

Usage::

    from trackers.aiatrack_omni import AiATrackOmniTracker
    from lib.utils import dict2Bfov
    import cv2

    tracker = AiATrackOmniTracker(
        aiatrack_path='/path/to/AiATrack',
        checkpoint='/path/to/AIATRACK_ep0500.pth.tar',
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
from typing import Optional

import cv2
import numpy as np

# Allow running from the repo root without explicit install
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'lib')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.omni import OmniImage
from lib.utils import Bfov, Bbox, scaleBFoV


class AiATrackOmniTracker:
    """AiATrack tracker adapted for 360° omnidirectional tracking.

    Args:
        aiatrack_path: Path to the cloned AiATrack repository root.  This
                       directory is inserted into ``sys.path`` so that
                       AiATrack's modules can be imported.
        yaml_name:     AiATrack config yaml name (without the ``.yaml``
                       extension), corresponding to a file inside
                       ``<aiatrack_path>/experiments/aiatrack/``.
                       Defaults to ``'baseline'``.
        checkpoint:    Path to the AiATrack checkpoint ``.pth.tar`` file.
        img_w:         Width of the equirectangular input image in pixels.
        img_h:         Height of the equirectangular input image in pixels.
        search_scale:  Multiplier applied to the predicted BFoV when defining
                       the search region for the next frame.  Values around
                       2.0 give the tracker sufficient context to handle
                       moderate motion.
        crop_size:     Number of horizontal pixels in the local perspective
                       crop produced by :meth:`OmniImage.crop_bfov`.
        device:        Torch device for the AiATrack model
                       (``'cuda'`` or ``'cpu'``).
        dataset_name:  Dataset name used to select AiATrack hyper-parameters
                       (default ``'360VOT'``).
    """

    def __init__(
        self,
        aiatrack_path: str,
        yaml_name: str = 'baseline',
        checkpoint: Optional[str] = None,
        img_w: int = 1920,
        img_h: int = 960,
        search_scale: float = 2.0,
        crop_size: int = 500,
        device: str = 'cuda',
        dataset_name: str = '360VOT',
    ):
        self.omni = OmniImage(img_w=img_w, img_h=img_h)
        self.search_scale = search_scale
        self.crop_size = crop_size

        # Insert AiATrack repo into sys.path so its modules are importable
        aiatrack_path = os.path.abspath(aiatrack_path)
        if aiatrack_path not in sys.path:
            sys.path.insert(0, aiatrack_path)

        # Import AiATrack parameter loader and instantiate the inner tracker
        from lib.test.parameter.aiatrack import parameters as aiatrack_parameters

        params = aiatrack_parameters(yaml_name)

        # Override the checkpoint path when the caller supplies one
        if checkpoint is not None:
            params.checkpoint = checkpoint

        # Override device
        import torch
        params.device = torch.device(device)

        from lib.test.tracker.aiatrack import AIATRACK
        self.tracker_inner = AIATRACK(params, dataset_name)

        # Cache AiATrack's expected search size (typically 320)
        self._search_size: int = int(params.search_size)
        # AiATrack's internal search_factor (typically 5.0)
        self._search_factor: float = float(params.search_factor)

        # State
        self._last_bfov: Optional[Bfov] = None
        self._last_u: Optional[np.ndarray] = None
        self._last_v: Optional[np.ndarray] = None
        # Last known target size in resized-crop pixels (for state override)
        self._target_w_resized: float = 0.0
        self._target_h_resized: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resize_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize a perspective crop to AiATrack's expected search_size."""
        sz = self._search_size
        return cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_LINEAR)

    def _target_bbox_in_resized(
        self, crop_w: int, crop_h: int, target_w: float, target_h: float
    ):
        """Return init_bbox [x1, y1, w, h] for the target centred in the
        resized crop, scaled to the resized image dimensions.

        The target occupies the centre ``1/search_scale`` fraction of the
        original crop.  After resizing to ``search_size x search_size`` we
        need to rescale the bbox accordingly.
        """
        sz = self._search_size
        scale_x = sz / crop_w
        scale_y = sz / crop_h

        tw = target_w * scale_x
        th = target_h * scale_y
        cx = sz / 2.0
        cy = sz / 2.0
        return [cx - tw / 2.0, cy - th / 2.0, tw, th]

    def _state_for_full_crop(self, target_w_res: float, target_h_res: float):
        """Compute the [x1, y1, w, h] state that AiATrack's internal
        ``sample_target`` must see so that it re-crops the *entire* provided
        resized image.

        AiATrack's ``sample_target`` takes a crop of side length::

            crop_sz = ceil(sqrt(w * h) * search_factor)

        where ``w``, ``h`` are taken from ``self.state``.  We want
        ``crop_sz == search_size``, so::

            sqrt(w * h) = search_size / search_factor

        We preserve the aspect ratio of the target and centre the bbox.
        """
        sz = self._search_size
        sf = self._search_factor
        desired_area = (sz / sf) ** 2  # w * h

        if target_w_res > 0 and target_h_res > 0:
            aspect = target_w_res / target_h_res
        else:
            aspect = 1.0

        tw = math.sqrt(desired_area * aspect)
        th = desired_area / tw

        cx = sz / 2.0
        cy = sz / 2.0
        return [cx - tw / 2.0, cy - th / 2.0, tw, th]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, img: np.ndarray, bfov: Bfov) -> None:
        """Initialise the tracker on the first frame.

        Args:
            img:  Equirectangular panoramic frame (H, W, 3) BGR uint8.
            bfov: Ground-truth BFoV annotation for the first frame.
        """
        # 1. Crop the panorama around the GT BFoV scaled up for context
        search_bfov = scaleBFoV(bfov, self.search_scale)
        crop, u_map, v_map = self.omni.crop_bfov(
            img, search_bfov, num_sample_h=self.crop_size
        )

        crop_h, crop_w = crop.shape[:2]

        # 2. Target occupies the centre 1/search_scale portion of the crop
        target_w = crop_w / self.search_scale
        target_h = crop_h / self.search_scale

        # 3. Resize the crop to AiATrack's expected search_size
        crop_resized = self._resize_crop(crop)

        # 4. Compute init_bbox in resized-crop coordinates
        init_bbox = self._target_bbox_in_resized(crop_w, crop_h, target_w, target_h)

        # Store target size in resized coords for subsequent state overrides
        sz = self._search_size
        scale_x = sz / crop_w
        scale_y = sz / crop_h
        self._target_w_resized = target_w * scale_x
        self._target_h_resized = target_h * scale_y

        # 5. AiATrack expects RGB; OpenCV returns BGR
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

        # 6. Initialise inner tracker
        self.tracker_inner.initialize(crop_rgb, {'init_bbox': init_bbox})

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

        # 1. Crop around the last predicted BFoV (scaled for search context)
        search_bfov = scaleBFoV(self._last_bfov, self.search_scale)
        crop, u_map, v_map = self.omni.crop_bfov(
            img, search_bfov, num_sample_h=self.crop_size
        )

        crop_h, crop_w = crop.shape[:2]

        # 2. Resize the crop to AiATrack's expected search_size
        crop_resized = self._resize_crop(crop)

        # 3. Override AiATrack's internal state so that its internal
        #    sample_target() with search_factor covers the whole resized image.
        self.tracker_inner.state = self._state_for_full_crop(
            self._target_w_resized, self._target_h_resized
        )

        # 4. AiATrack expects RGB; OpenCV returns BGR
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

        # 5. Run AiATrack; returns {'target_bbox': [x1, y1, w, h]}
        out = self.tracker_inner.track(crop_rgb)
        x1, y1, tw, th = out['target_bbox']

        # 6. Update target size in resized-crop coords for the next state override
        self._target_w_resized = float(tw)
        self._target_h_resized = float(th)

        sz = self._search_size
        cx_res = float(x1) + float(tw) / 2.0
        cy_res = float(y1) + float(th) / 2.0

        # 7. Map predicted bbox from resized-crop coords back to original crop coords
        scale_back_x = crop_w / sz
        scale_back_y = crop_h / sz

        cx = cx_res * scale_back_x
        cy = cy_res * scale_back_y
        w  = float(tw) * scale_back_x
        h  = float(th) * scale_back_y

        # Clamp to crop bounds to avoid out-of-range remap
        cx = float(np.clip(cx, 0, crop_w - 1))
        cy = float(np.clip(cy, 0, crop_h - 1))
        w  = float(np.clip(w, 1, crop_w))
        h  = float(np.clip(h, 1, crop_h))

        local_box = Bbox(cx, cy, w, h, rotation=0)

        # 8. Project local bbox back to global spherical BFoV
        pred_bfov = self.omni.localBbox2Bfov(local_box, u_map, v_map,
                                              need_rotation=False)

        self._last_bfov = pred_bfov
        self._last_u = u_map
        self._last_v = v_map

        return pred_bfov
