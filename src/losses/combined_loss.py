"""Combined Loss: weighted sum of all SymPanICH-Net v2 losses."""

import torch
import torch.nn as nn
from .focal_loss import FocalLoss
from .dice_loss import DiceLoss
from .hover_loss import HoVerLoss
from .contrastive_loss import ContrastiveLoss


class CombinedLoss(nn.Module):
    """
    Combined loss for SymPanICH-Net v2 training.

    L_total = w_cls * L_cls + w_dice * L_dice + w_mask_focal * L_mask_focal
            + w_hv * L_hv + w_deep * L_deep
            + w_contrastive * L_contrastive + w_text_cls * L_text_cls
    """

    def __init__(
        self,
        cls_weight: float = 2.0,
        dice_weight: float = 5.0,
        mask_focal_weight: float = 5.0,
        hv_weight: float = 1.0,
        deep_supervision_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        text_cls_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        super().__init__()

        # Loss components
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice_loss = DiceLoss()
        self.hover_loss = HoVerLoss()
        self.contrastive_loss = ContrastiveLoss()

        # Weights
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight
        self.mask_focal_weight = mask_focal_weight
        self.hv_weight = hv_weight
        self.deep_weight = deep_supervision_weight
        self.contrastive_weight = contrastive_weight
        self.text_cls_weight = text_cls_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        pred_hv: torch.Tensor,
        target_classes: torch.Tensor,
        target_masks: torch.Tensor,
        target_hv: torch.Tensor,
        hemorrhage_mask: torch.Tensor,
        pixel_embedding: torch.Tensor = None,
        text_embeddings: torch.Tensor = None,
        aux_outputs: list = None,
    ) -> dict:
        """
        Compute all losses.

        Args:
            pred_logits: (B, N, num_classes)
            pred_masks: (B, N, H, W)
            pred_hv: (B, 2, H, W)
            target_classes: (B, N) matched GT class per query
            target_masks: (B, N, H, W) matched GT masks
            target_hv: (B, 2, H, W) GT distance maps
            hemorrhage_mask: (B, H, W) binary mask of any hemorrhage
            pixel_embedding: (B, C, H, W) for contrastive loss
            text_embeddings: (6, C) for contrastive loss
            aux_outputs: list of intermediate predictions for deep supervision

        Returns:
            Dict with 'total' and individual loss components.
        """
        losses = {}

        # 1. Classification loss (Focal CE)
        losses['cls'] = self.cls_weight * self.focal_loss(pred_logits, target_classes)

        # 2. Mask Dice loss
        losses['dice'] = self.dice_weight * self.dice_loss(pred_masks, target_masks)

        # 3. Mask Focal loss (pixel-level)
        mask_focal = self.focal_loss(
            pred_masks.flatten(0, 1).unsqueeze(-1).expand(-1, -1, 2).flatten(0, 1),
            target_masks.flatten(0, 1).long().flatten(),
        )
        losses['mask_focal'] = self.mask_focal_weight * mask_focal

        # 4. HoVer distance loss
        losses['hv'] = self.hv_weight * self.hover_loss(pred_hv, target_hv, hemorrhage_mask)

        # 5. Contrastive loss
        if pixel_embedding is not None and text_embeddings is not None:
            losses['contrastive'] = self.contrastive_weight * self.contrastive_loss(
                pixel_embedding, text_embeddings, pred_masks.detach(), target_classes,
            )
        else:
            losses['contrastive'] = torch.tensor(0.0, device=pred_logits.device)

        # 6. Deep supervision (auxiliary losses from intermediate decoder layers)
        if aux_outputs is not None:
            deep_loss = torch.tensor(0.0, device=pred_logits.device)
            for aux in aux_outputs:
                deep_loss += self.focal_loss(aux['pred_logits'], target_classes)
                deep_loss += self.dice_loss(aux['pred_masks'], target_masks)
            losses['deep'] = self.deep_weight * deep_loss / max(len(aux_outputs), 1)
        else:
            losses['deep'] = torch.tensor(0.0, device=pred_logits.device)

        # Total
        losses['total'] = sum(losses.values())

        return losses
