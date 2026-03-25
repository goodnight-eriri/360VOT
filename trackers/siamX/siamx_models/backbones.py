"""
AlexNet-based backbone for SiamFC tracker.

Input sizes and feature map sizes:
  Template (z):  127x127 -> 6x6x256
  Search   (x):  255x255 -> 22x22x256

Architecture (no padding, 2 max-pool layers):
  Conv1 (11,s=2) -> BN -> ReLU -> MaxPool(3,s=2)
  Conv2 (5,s=1)  -> BN -> ReLU -> MaxPool(3,s=2)
  Conv3 (3,s=1)  -> BN -> ReLU
  Conv4 (3,s=1)  -> BN -> ReLU
  Conv5 (3,s=1)
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """Modified AlexNet backbone for SiamFC (no FC layers, no padding)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: 127->59, 255->123
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # Layer 2: 29->25, 61->57
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # Layer 3: 12->10, 28->26
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Layer 4: 10->8, 26->24
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Layer 5: 8->6, 24->22
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
