"""Utility modules shared across siamX model components."""

import torch
import torch.nn as nn


class AdjustLayer(nn.Module):
    """1x1 conv adjustment layer to align feature-channel dimensions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if x.size(3) < 20:
            # Remove padding artefact for template branch
            pad_offset = 4
            x = x[:, :, pad_offset:-pad_offset, pad_offset:-pad_offset]
        return x
