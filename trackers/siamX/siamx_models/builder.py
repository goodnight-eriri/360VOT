"""SiamFC model builder.

SiamFC uses:
  - AlexNet backbone (no padding, produces 6x6 for z=127 and 22x22 for x=255)
  - XCorrHead (simple cross-correlation)

Expected pretrained weight file: ``trackers/siamX/pretrained_weights/SiamFC.pth``
"""

import torch
import torch.nn as nn

from .backbones import AlexNet
from .heads import XCorrHead
from .utils import load_pretrain


class SiamFC(nn.Module):
    """Siamese network for SiamFC tracking (AlexNet + cross-correlation)."""

    def __init__(self):
        super().__init__()
        self.backbone = AlexNet()
        self.head = XCorrHead()

    def template(self, z: torch.Tensor) -> torch.Tensor:
        """Extract template features from a 127x127 crop."""
        return self.backbone(z)

    def track(self, x: torch.Tensor) -> torch.Tensor:
        """Extract search features from a 255x255 crop."""
        return self.backbone(x)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute response map between template *z* and search *x*."""
        z_feat = self.template(z)
        x_feat = self.track(x)
        return self.head(z_feat, x_feat)


def build_siamfc(pretrain_path: str = None) -> SiamFC:
    """Build a SiamFC model, optionally loading pretrained weights.

    Args:
        pretrain_path: path to pretrained ``.pth`` file.  Pass ``None`` to
            skip weight loading (useful for testing the architecture).

    Returns:
        SiamFC model in eval mode.
    """
    model = SiamFC()
    if pretrain_path is not None:
        model = load_pretrain(model, pretrain_path)
    model.eval()
    return model
