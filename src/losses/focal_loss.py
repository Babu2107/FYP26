"""Focal Loss for class-imbalanced classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for handling class imbalance.

    Reduces loss contribution from easy examples, focusing training
    on hard examples (e.g., small hemorrhages vs large background).

    L_focal = -α(1-p_t)^γ * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 reduction: str = "mean", no_object_weight: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.no_object_weight = no_object_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (B, C) or (B, N, C).
            targets: Ground truth class indices (B,) or (B, N).

        Returns:
            Focal loss scalar.
        """
        num_classes = inputs.shape[-1]

        # Build per-class weights: down-weight background/no-object (class 0)
        # so the model can't collapse to all-background predictions
        weight = torch.ones(num_classes, device=inputs.device)
        weight[0] = self.no_object_weight

        ce_loss = F.cross_entropy(
            inputs.reshape(-1, num_classes),
            targets.reshape(-1),
            weight=weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
