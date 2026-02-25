"""
Module 4: Feature Pyramid Network (FPN).

Top-down pathway with lateral connections to produce
multi-scale feature maps at a uniform channel dimension (256).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network that takes multi-scale backbone features
    and produces unified 256-channel feature maps at each scale.

    Input (from Swin-T backbone):
        F1: (B, 96,  64, 64)
        F2: (B, 192, 32, 32)
        F3: (B, 384, 16, 16)
        F4: (B, 768, 8,  8)

    Output:
        P1: (B, 256, 64, 64)  — 1/4 scale
        P2: (B, 256, 32, 32)  — 1/8 scale
        P3: (B, 256, 16, 16)  — 1/16 scale
        P4: (B, 256, 8,  8)   — 1/32 scale
    """

    def __init__(
        self,
        in_channels: list = None,
        out_channels: int = 256,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = [96, 192, 384, 768]

        # Lateral connections: 1×1 conv to project to out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])

        # Output convolutions: 3×3 conv to smooth after addition
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

    def forward(self, features: list) -> list:
        """
        Build feature pyramid from backbone features.

        Args:
            features: List of 4 feature tensors [F1, F2, F3, F4]
                      from finest to coarsest.

        Returns:
            List of 4 FPN outputs [P1, P2, P3, P4], all 256 channels.
        """
        assert len(features) == len(self.lateral_convs), (
            f"Expected {len(self.lateral_convs)} features, got {len(features)}"
        )

        # Step 1: Apply lateral convolutions (1×1)
        laterals = [
            lat_conv(feat)
            for lat_conv, feat in zip(self.lateral_convs, features)
        ]

        # Step 2: Top-down pathway — start from coarsest (P4)
        # Add upsampled coarser level to finer level
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample coarser level to match finer level's spatial size
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Step 3: Apply 3×3 output convolutions
        outputs = [
            out_conv(lateral)
            for out_conv, lateral in zip(self.output_convs, laterals)
        ]

        return outputs  # [P1, P2, P3, P4]
