"""SiameseNet — thin wrapper around the SiamFC model providing the
``get_template_z_new`` / ``get_scores_new`` interface expected by
SiamFCTracker.

Ported from zllrunning/SiameseX.PyTorch ``demo_utils/siamese.py``.
Key adaptation: ``get_scores_new`` accepts a numpy BGR ndarray instead of
a filename string, since 360VOT passes crops as ndarrays.
"""

from __future__ import annotations

import os
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .crops import (
    extract_crops_z, extract_crops_x,
    image_to_tensor, image_to_tensor_batch,
)
from ..siamx_models.builder import SiamFC, build_siamfc

logger = logging.getLogger(__name__)


class SiameseNet:
    """Wraps :class:`SiamFC` and exposes the tracker-facing interface.

    Args:
        model_path: Path to the pretrained ``SiamFC.pth`` weight file.
        device:     Torch device string (``'cuda'`` or ``'cpu'``).
        use_trt:    When ``True``, export the backbone to ONNX (if needed),
                    build/load a TensorRT engine, and use it for inference.
                    Requires TensorRT and PyCUDA to be installed; falls back
                    silently to PyTorch when they are unavailable.
        onnx_dir:   Directory where ONNX / TRT engine files are stored.
                    Defaults to the same directory as *model_path*, or the
                    current working directory when *model_path* is ``None``.
    """

    def __init__(self, model_path: str | None = None,
                 device: str = 'cpu',
                 use_trt: bool = False,
                 onnx_dir: str | None = None):
        self.device = torch.device(device)
        self.model: SiamFC = build_siamfc(model_path)
        self.model.to(self.device)
        self.model.eval()

        self._trt_z: object | None = None  # TRTBackbone for template
        self._trt_x: object | None = None  # TRTBackbone for search
        self._use_trt = False

        if use_trt:
            self._init_trt(model_path, onnx_dir)

    # ------------------------------------------------------------------
    # TensorRT initialisation
    # ------------------------------------------------------------------

    def _init_trt(self, model_path: str | None, onnx_dir: str | None) -> None:
        """Try to set up TRT backends; silently fall back to PyTorch on error."""
        try:
            from .trt_backend import TRTBackbone, export_backbone_onnx

            if onnx_dir is None:
                if model_path is not None:
                    onnx_dir = os.path.dirname(os.path.abspath(model_path))
                else:
                    onnx_dir = os.getcwd()

            onnx_z = os.path.join(onnx_dir, 'siamfc_backbone_z.onnx')
            onnx_x = os.path.join(onnx_dir, 'siamfc_backbone_x.onnx')
            engine_z = os.path.join(onnx_dir, 'siamfc_backbone_z.trt')
            engine_x = os.path.join(onnx_dir, 'siamfc_backbone_x.trt')

            # Export ONNX files if missing
            if not os.path.exists(onnx_z) or not os.path.exists(onnx_x + ".search.onnx"):
                logger.info("Exporting SiamFC backbone to ONNX …")
                export_backbone_onnx(
                    self.model,
                    onnx_z,
                    template_shape=(1, 3, 127, 127),
                    search_shape=(3, 3, 255, 255),
                )
                # The search ONNX is created as <onnx_z>.search.onnx by the exporter
                onnx_x = onnx_z + ".search.onnx"
            else:
                onnx_x = onnx_z + ".search.onnx"

            self._trt_z = TRTBackbone(onnx_z, engine_z, fp16=True, max_batch=1)
            self._trt_x = TRTBackbone(onnx_x, engine_x, fp16=True, max_batch=4)
            self._use_trt = True
            logger.info("TensorRT backbone enabled.")

        except Exception as exc:
            logger.warning(
                "TensorRT initialisation failed (%s) — falling back to PyTorch.", exc
            )
            self._use_trt = False

    # ------------------------------------------------------------------
    # Feature extraction (used internally)
    # ------------------------------------------------------------------

    def branch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract backbone features from a pre-normalised tensor.

        When TRT is enabled the tensor is transferred to CPU as a numpy array,
        run through the TRT engine, and the result is returned as a CUDA tensor.
        """
        if self._use_trt:
            tensor_np = tensor.cpu().numpy()
            out_np = self._trt_x(tensor_np)  # numpy (N, C, H, W)
            return torch.from_numpy(out_np).to(self.device)
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

        if self._use_trt:
            z_np = z_tensor.cpu().numpy()
            out_np = self._trt_z(z_np)
            return torch.from_numpy(out_np).to(self.device)

        with torch.no_grad():
            z_feat = self.model.features(z_tensor)
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

        # Extract one search crop per scale — use fast OpenCV path by default
        x_crops = [
            extract_crops_x(
                image, pos_y, pos_x,
                x_sz=sa, scale_factors=[1.0],
                search_sz=design.search_sz,
            )[0]
            for sa in scaled_search_area
        ]

        # Batch tensor conversion — single .to(device) call
        x_tensor = image_to_tensor_batch(x_crops, self.device)

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
