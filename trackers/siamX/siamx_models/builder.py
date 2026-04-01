"""SiamFC model — ported from zllrunning/SiameseX.PyTorch (models/builder.py).

Architecture:
  - AlexNet backbone with ``self.feature`` (singular) Sequential attribute
  - SiamFC_ base: cross-correlation head
  - SiamFC child: adds ``0.001 * out + 0.0`` scale adjustment and kaiming
    weight initialisation

Weight loading:
  The pretrained ``SiamFC.pth`` from SiameseX.PyTorch stores keys that match
  the SiamFC model structure directly (``features.feature.*``).  An optional
  ``model.`` prefix (produced when saving from SiamVGGTracker) is stripped
  automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class AlexNet(nn.Module):
    """SiameseX.PyTorch AlexNet backbone for SiamFC.

    This matches the pretrained ``SiamFC.pth`` layout from
    zllrunning/SiameseX.PyTorch:

    - conv1 / conv2 / conv3 / conv4 / conv5
    - BatchNorm after conv1..conv4
    - groups=2 for conv2, conv4 and conv5
    """

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv2 (groups=2!)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),

            # conv4 (groups=2!)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),

            # conv5 (groups=2!)
            nn.Conv2d(384, 256, kernel_size=3, stride=1, groups=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)


# ---------------------------------------------------------------------------
# Siamese model
# ---------------------------------------------------------------------------

class SiamFC_(nn.Module):
    """SiamFC base: AlexNet features + fast cross-correlation head."""

    def __init__(self):
        super().__init__()
        self.features = AlexNet()

    def head(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Fast cross-correlation: (B,C,Hz,Wz) × (B,C,Hx,Wx) → (B,1,Hr,Wr)."""
        n, c, h, w = x.size()
        x_flat = x.view(1, n * c, h, w)
        out = F.conv2d(x_flat, z, groups=n)
        return out.permute(1, 0, 2, 3)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_feat = self.features(z)
        x_feat = self.features(x)
        return self.head(z_feat, x_feat)


class SiamFC(SiamFC_):
    """SiamFC with scale-adjusted response and kaiming weight initialisation."""

    def __init__(self):
        super().__init__()
        self._initialize_weights()

    def head(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = super().head(z, x)
        return 0.001 * out + 0.0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_siamfc(model_path: Optional[str] = None) -> SiamFC:
    """Build a :class:`SiamFC` model, optionally loading pretrained weights.

    The ``SiamFC.pth`` from SiameseX.PyTorch stores weights with keys that
    match the SiamFC structure directly (``features.feature.*``).  If the
    file was exported from a ``SiamVGGTracker`` wrapper (adding a ``model.``
    prefix), that prefix is stripped automatically.

    Args:
        model_path: path to ``SiamFC.pth``.  ``None`` → random weights.

    Returns:
        :class:`SiamFC` in eval mode.
    """
    model = SiamFC()
    if model_path is not None:
        pretrained = torch.load(model_path, map_location='cpu')
        # Unwrap 'state_dict' wrapper if present
        if isinstance(pretrained, dict) and 'state_dict' in pretrained:
            pretrained = pretrained['state_dict']
        # Strip 'model.' prefix added by SiamVGGTracker saving convention
        pretrained = {
            (k[len('model.'):] if k.startswith('model.') else k): v
            for k, v in pretrained.items()
        }
        # Strip 'module.' prefix from DataParallel if present
        pretrained = {k.replace('module.', ''): v for k, v in pretrained.items()}
        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        if missing:
            print(f'[siamX] Missing keys when loading {model_path}: {missing}')
        if unexpected:
            print(f'[siamX] Unexpected keys when loading {model_path}: {unexpected}')
    model.eval()
    return model
