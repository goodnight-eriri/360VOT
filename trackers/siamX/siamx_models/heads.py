"""Tracking heads for siamX models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def xcorr_fast(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Fast cross-correlation between template features *z* and search features *x*.

    Args:
        z: template feature map (B, C, Hz, Wz).
        x: search feature map  (B, C, Hx, Wx).

    Returns:
        Response map (B, 1, Hx-Hz+1, Wx-Wz+1).
    """
    batch = z.shape[0]
    # Reshape to treat each sample in the batch as a separate depthwise group
    z_flat = z.view(batch, -1, z.shape[-2], z.shape[-1])
    x_flat = x.view(1, batch * z.shape[1], x.shape[-2], x.shape[-1])
    out = F.conv2d(x_flat, z_flat, groups=batch)
    # out: (1, B, Hr, Wr) — reshape to (B, 1, Hr, Wr)
    out = out.permute(1, 0, 2, 3)
    return out


class XCorrHead(nn.Module):
    """Simple cross-correlation head used by SiamFC."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return xcorr_fast(z, x)
