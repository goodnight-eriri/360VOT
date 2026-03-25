"""Neck modules for siamX (identity pass-through used by SiamFC)."""

import torch
import torch.nn as nn


class IdentityNeck(nn.Module):
    """Identity neck — passes features unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
