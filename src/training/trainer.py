"""
PyTorch Lightning Training Module for SymPanICH-Net v2.

Handles training loop, validation, phased training, and logging.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from ..models.sympanich_net import SymPanICHNetV2
from ..losses.combined_loss import CombinedLoss
from ..utils.metrics import compute_dice, compute_iou


class SymPanICHNetModule(pl.LightningModule):
    """
    Lightning module wrapping SymPanICH-Net v2 with phased training.

    Training Phases:
        Phase 1 (ep 1-5):    Warmup — backbone + text encoder frozen
        Phase 2 (ep 6-60):   Full training — all unfrozen
        Phase 3 (ep 61-90):  Fine-tuning — lower LRs
        Phase 4 (ep 91-100): Report tuning — freeze segmentation
    """

    def __init__(
        self,
        # Model config
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        num_queries: int = 50,
        num_classes: int = 7,
        num_decoder_layers: int = 9,
        use_context: bool = True,
        text_descriptions_path: Optional[str] = None,
        # Training config
        base_lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        max_epochs: int = 100,
        warmup_epochs: int = 5,
        # Loss weights
        cls_weight: float = 2.0,
        dice_weight: float = 5.0,
        mask_focal_weight: float = 5.0,
        hv_weight: float = 1.0,
        contrastive_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build model
        self.model = SymPanICHNetV2(
            backbone_name=backbone_name,
            pretrained=pretrained,
            use_context=use_context,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            text_descriptions_path=text_descriptions_path,
        )

        # Build loss
        self.criterion = CombinedLoss(
            cls_weight=cls_weight,
            dice_weight=dice_weight,
            mask_focal_weight=mask_focal_weight,
            hv_weight=hv_weight,
            contrastive_weight=contrastive_weight,
        )

        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

    def forward(self, images, images_flipped):
        return self.model(images, images_flipped)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch['image']
        images_flipped = batch['image_flipped']
        gt_mask = batch['mask']
        gt_hv = batch['hv_maps']

        # Forward pass
        outputs = self.model(images, images_flipped)

        # Create dummy matched targets for now
        # (Full Hungarian matching will be added when dataset format is confirmed)
        B, N, C = outputs['pred_logits'].shape
        H, W = outputs['pred_masks'].shape[-2:]

        # Simple target creation: use semantic mask as supervision
        target_classes = torch.zeros(B, N, dtype=torch.long, device=images.device)
        target_masks = torch.zeros(B, N, H, W, device=images.device)

        # Assign first few queries to detected classes
        for b in range(B):
            unique_classes = gt_mask[b].unique()
            unique_classes = unique_classes[unique_classes > 0]  # Remove background
            for qi, cls_id in enumerate(unique_classes[:N]):
                target_classes[b, qi] = cls_id
                cls_mask = (gt_mask[b] == cls_id).float()
                # Resize GT mask to match prediction size
                cls_mask_resized = F.interpolate(
                    cls_mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="nearest"
                ).squeeze()
                target_masks[b, qi] = cls_mask_resized

        # Hemorrhage mask for HoVer loss
        hemorrhage_mask = (gt_mask > 0).float()
        hv_resized = F.interpolate(gt_hv, size=(H, W), mode="bilinear", align_corners=False)
        hem_resized = F.interpolate(
            hemorrhage_mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
        ).squeeze(1)

        # Compute losses
        losses = self.criterion(
            pred_logits=outputs['pred_logits'],
            pred_masks=outputs['pred_masks'],
            pred_hv=outputs['hv_maps'],
            target_classes=target_classes,
            target_masks=target_masks,
            target_hv=hv_resized,
            hemorrhage_mask=hem_resized,
            pixel_embedding=outputs['pixel_embedding'],
            text_embeddings=outputs['text_embeddings'],
            aux_outputs=outputs.get('aux_outputs'),
        )

        # Log losses
        for name, val in losses.items():
            self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=(name == 'total'))

        return losses['total']

    def validation_step(self, batch: dict, batch_idx: int):
        images = batch['image']
        images_flipped = batch['image_flipped']
        gt_mask = batch['mask']

        # Forward pass
        outputs = self.model(images, images_flipped)

        # Simple metric: semantic segmentation Dice
        pred_semantic = outputs['pred_logits'].argmax(dim=-1)  # (B, N)
        # Use the highest-confidence mask per class for evaluation
        pred_scores = outputs['pred_logits'].softmax(dim=-1)

        # Compute per-class dice on the semantic map
        B = images.shape[0]
        dice_scores = []
        for b in range(B):
            semantic_pred = torch.zeros_like(gt_mask[b])
            # Assign each pixel to the class of the highest-scoring query
            masks = outputs['pred_masks'][b].sigmoid()
            for q in range(outputs['pred_logits'].shape[1]):
                cls = pred_semantic[b, q].item()
                if cls > 0:
                    mask = F.interpolate(
                        masks[q:q+1].unsqueeze(0),
                        size=gt_mask[b].shape,
                        mode="bilinear", align_corners=False,
                    ).squeeze() > 0.5
                    semantic_pred[mask] = cls

            # Overall hemorrhage dice
            pred_hem = (semantic_pred > 0).float()
            gt_hem = (gt_mask[b] > 0).float()
            dice = compute_dice(pred_hem, gt_hem)
            dice_scores.append(dice)

        avg_dice = torch.stack(dice_scores).mean()
        self.log("val/dice", avg_dice, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self):
        """Handle phased training transitions."""
        epoch = self.current_epoch

        if epoch < self.warmup_epochs:
            # Phase 1: Freeze backbone + text encoder
            self.model.freeze_backbone()
            self.model.freeze_text_encoder()
        elif epoch == self.warmup_epochs:
            # Phase 2: Unfreeze all
            self.model.unfreeze_backbone()
            self.model.unfreeze_text_encoder()

    def configure_optimizers(self):
        # Differential learning rates
        param_groups = self.model.get_param_groups(self.base_lr)

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        # Poly scheduler with warmup
        def poly_scheduler(step):
            warmup = self.hparams.warmup_steps
            if step < warmup:
                return step / max(warmup, 1)
            total = self.trainer.estimated_stepping_batches
            progress = (step - warmup) / max(total - warmup, 1)
            return (1 - progress) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_scheduler)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
