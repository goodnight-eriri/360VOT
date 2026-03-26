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


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class AlexNet(nn.Module):
    """5-layer AlexNet backbone for SiamFC (no FC layers, no padding).

    Attribute name ``self.feature`` (singular) matches SiameseX.PyTorch so
    that state-dict keys align with the pretrained ``SiamFC.pth``.

    Input/output sizes (no padding):
      z = 127 × 127  →  6 × 6 × 256
      x = 255 × 255  → 22 × 22 × 256
    """

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # Conv1: 127→59 / 255→123
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # Pool1: 59→29 / 123→61
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2: 29→25 / 61→57
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Pool2: 25→12 / 57→28
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3: 12→10 / 28→26
            nn.Conv2d(256, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Conv4: 10→8 / 26→24
            nn.Conv2d(384, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Conv5: 8→6 / 24→22
            nn.Conv2d(384, 256, kernel_size=3),
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

def build_siamfc(model_path: str | None = None) -> SiamFC:
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
