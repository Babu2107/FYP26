"""Dice Loss for segmentation mask overlap optimization."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Binary Dice Loss for per-mask segmentation.

    Dice = 2|P∩G| / (|P| + |G|)
    DiceLoss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted mask logits (B, H, W) or (B, N, H, W).
            target: Ground truth binary mask, same shape.

        Returns:
            Dice loss scalar.
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.flatten(1)
        target_flat = target.float().flatten(1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()
