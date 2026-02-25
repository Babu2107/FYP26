"""
Module 2: Dual-Stream Swin Transformer V2 Backbone.

Processes both original and horizontally-flipped CT images through a
shared-weight Swin-T V2 backbone to extract multi-scale feature maps.
"""

import torch
import torch.nn as nn
import timm


class DualStreamSwinV2(nn.Module):
    """
    Dual-stream Swin Transformer V2 backbone with shared weights.

    Takes original and flipped images, extracts 4-scale hierarchical
    features from each using the SAME backbone (shared weights).

    Output scales (for 256×256 input):
        F1: (B, 96,  64, 64)  — 1/4
        F2: (B, 192, 32, 32)  — 1/8
        F3: (B, 384, 16, 16)  — 1/16
        F4: (B, 768, 8,  8)   — 1/32
    """

    def __init__(
        self,
        model_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()

        # Create backbone with feature extraction mode
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )

        # Get output channel dimensions for each stage
        self.feature_dims = self.backbone.feature_info.channels()
        # Typically: [96, 192, 384, 768] for Swin-T

    def forward(
        self, x_orig: torch.Tensor, x_flip: torch.Tensor
    ) -> tuple:
        """
        Forward pass through shared backbone for both streams.

        Args:
            x_orig: Original images (B, C, H, W).
            x_flip: Horizontally flipped images (B, C, H, W).

        Returns:
            f_orig: List of 4 feature maps from original stream.
            f_flip: List of 4 feature maps from flipped stream.
        """
        f_orig = self.backbone(x_orig)
        f_flip = self.backbone(x_flip)
        return f_orig, f_flip

    def get_feature_dims(self) -> list:
        """Return channel dimensions at each feature scale."""
        return self.feature_dims


class ContextProjection(nn.Module):
    """
    Projects 2.5D multi-channel input (15ch) down to 3 channels
    for the Swin backbone that expects 3-channel input.

    Used when context_slices > 0 (2.5D mode).
    """

    def __init__(self, in_channels: int = 15, out_channels: int = 3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project multi-channel 2.5D input to 3 channels."""
        return self.proj(x)
