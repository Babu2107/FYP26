"""MSE Loss for HoVer distance map regression."""

import torch
import torch.nn as nn


class HoVerLoss(nn.Module):
    """
    Masked MSE loss for horizontal/vertical distance maps.
    Only computes loss on hemorrhage pixels (ignores background).
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self, pred_hv: torch.Tensor, target_hv: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_hv: Predicted H/V maps (B, 2, H, W).
            target_hv: Ground truth H/V maps (B, 2, H, W).
            mask: Binary mask of hemorrhage pixels (B, H, W) or (B, 1, H, W).
                  Loss is only computed where mask > 0.

        Returns:
            Masked MSE loss scalar.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)

        mask = mask.expand_as(pred_hv).float()
        loss = self.mse(pred_hv, target_hv)
        masked_loss = loss * mask

        num_pixels = mask.sum().clamp(min=1.0)
        return masked_loss.sum() / num_pixels
