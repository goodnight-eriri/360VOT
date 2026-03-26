"""SiameseNet — thin wrapper around the SiamFC model providing the
``get_template_z_new`` / ``get_scores_new`` interface expected by
SiamFCTracker.

Ported from zllrunning/SiameseX.PyTorch ``demo_utils/siamese.py``.
Key adaptation: ``get_scores_new`` accepts a numpy BGR ndarray instead of
a filename string, since 360VOT passes crops as ndarrays.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .crops import extract_crops_z, extract_crops_x, image_to_tensor
from ..siamx_models.builder import SiamFC, build_siamfc


class SiameseNet:
    """Wraps :class:`SiamFC` and exposes the tracker-facing interface."""

    def __init__(self, model_path: str | None = None,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.model: SiamFC = build_siamfc(model_path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Feature extraction (used internally)
    # ------------------------------------------------------------------

    def branch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract backbone features from a pre-normalised tensor."""
        return self.model.features(tensor)

    # ------------------------------------------------------------------
    # Template extraction (called once during initialisation)
    # ------------------------------------------------------------------

    def get_template_z_new(self, pos_x: float, pos_y: float, z_sz: float,
                           image: np.ndarray,
                           design) -> torch.Tensor:
        """Extract template features from the first frame.

        Args:
            pos_x:  Horizontal centre of the target in *image*.
            pos_y:  Vertical centre of the target in *image*.
            z_sz:   Template crop size in pixels (before resizing to 127).
            image:  BGR uint8 frame.
            design: Design parameter object with ``exemplar_sz`` attribute.

        Returns:
            Template feature tensor of shape (1, C, Hz, Wz).
        """
        z_crop = extract_crops_z(
            image, pos_y, pos_x, z_sz, exemplar_sz=design.exemplar_sz
        )
        z_tensor = image_to_tensor(z_crop, self.device)
        with torch.no_grad():
            z_feat = self.branch(z_tensor)
        return z_feat

    # ------------------------------------------------------------------
    # Score computation (called once per frame after initialisation)
    # ------------------------------------------------------------------

    def get_scores_new(self, pos_x: float, pos_y: float,
                       scaled_search_area: list[float],
                       template_z: torch.Tensor,
                       image: np.ndarray,
                       design,
                       final_score_sz: int) -> torch.Tensor:
        """Compute response scores for each scaled search crop.

        Args:
            pos_x:              Current horizontal centre estimate.
            pos_y:              Current vertical centre estimate.
            scaled_search_area: List of search crop sizes (one per scale).
            template_z:         Template features from
                                :meth:`get_template_z_new`.
            image:              BGR uint8 current frame (numpy ndarray).
            design:             Design parameters (``search_sz`` attribute).
            final_score_sz:     Side length to upsample the response map to.

        Returns:
            Tensor of shape (n_scales, final_score_sz, final_score_sz).
        """
        num_scales = len(scaled_search_area)

        # Extract one search crop per scale
        x_crops = [
            extract_crops_x(
                image, pos_y, pos_x,
                x_sz=sa, scale_factors=[1.0],
                search_sz=design.search_sz,
            )[0]
            for sa in scaled_search_area
        ]

        x_tensor = torch.cat(
            [image_to_tensor(c, self.device) for c in x_crops],
            dim=0,
        )  # (n_scales, 3, search_sz, search_sz)

        # Expand template to match batch size
        z_batch = template_z.expand(num_scales, -1, -1, -1)

        with torch.no_grad():
            x_feat = self.branch(x_tensor)
            scores = self.model.head(z_batch, x_feat)  # (n_scales, 1, Hr, Wr)

        scores = scores.squeeze(1)  # (n_scales, Hr, Wr)

        # Upsample to final_score_sz × final_score_sz
        scores = F.interpolate(
            scores.unsqueeze(1),
            size=(final_score_sz, final_score_sz),
            mode='bicubic',
            align_corners=False,
        ).squeeze(1)  # (n_scales, final_score_sz, final_score_sz)

        return scores
