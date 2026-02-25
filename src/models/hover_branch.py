"""
Module 6: HoVer-Net-Inspired Distance Branch.

Predicts horizontal and vertical distance maps from each pixel to its
instance center, providing explicit boundary signals to separate
adjacent hemorrhage instances.
"""

import torch
import torch.nn as nn


class HoVerBranch(nn.Module):
    """
    HoVer distance map prediction branch.

    Takes per-pixel embeddings and predicts H/V distance maps.
    At inference, Sobel gradients of these maps reveal instance boundaries.
    """

    def __init__(self, in_channels: int = 256, mid_channels: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Separate heads for horizontal and vertical distance maps
        self.h_head = nn.Conv2d(mid_channels // 2, 1, kernel_size=1)
        self.v_head = nn.Conv2d(mid_channels // 2, 1, kernel_size=1)

    def forward(self, pixel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict H/V distance maps.

        Args:
            pixel_embeddings: Per-pixel features (B, C, H, W).

        Returns:
            Distance maps (B, 2, H, W) — channel 0 = horizontal, channel 1 = vertical.
        """
        features = self.encoder(pixel_embeddings)
        h_map = self.h_head(features)  # (B, 1, H, W)
        v_map = self.v_head(features)  # (B, 1, H, W)
        return torch.cat([h_map, v_map], dim=1)  # (B, 2, H, W)


def compute_hv_maps(instance_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute ground truth horizontal/vertical distance maps from instance masks.
    Used during preprocessing to create GT targets.

    For each instance, compute normalized distance from each pixel
    to the instance centroid. H-map = horizontal distance, V-map = vertical.

    Args:
        instance_mask: (H, W) integer tensor where each unique value > 0
                       is a separate instance.

    Returns:
        (2, H, W) tensor — channel 0 = h_map, channel 1 = v_map.
        Values normalized to [-1, 1] per instance.
    """
    H, W = instance_mask.shape
    h_map = torch.zeros(H, W, dtype=torch.float32)
    v_map = torch.zeros(H, W, dtype=torch.float32)

    instance_ids = torch.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]  # Remove background (0)

    for inst_id in instance_ids:
        mask = (instance_mask == inst_id)
        if mask.sum() == 0:
            continue

        # Find centroid
        ys, xs = torch.where(mask)
        cy = ys.float().mean()
        cx = xs.float().mean()

        # Compute distances from centroid
        h_dist = (xs.float() - cx)
        v_dist = (ys.float() - cy)

        # Normalize to [-1, 1]
        max_h = h_dist.abs().max().clamp(min=1.0)
        max_v = v_dist.abs().max().clamp(min=1.0)

        h_map[ys, xs] = h_dist / max_h
        v_map[ys, xs] = v_dist / max_v

    return torch.stack([h_map, v_map], dim=0)
