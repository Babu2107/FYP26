"""
SymPanICH-Net v2: Full Model Assembly.

Assembles all 8 modules into the complete
Text-Guided Symmetry-Aware Panoptic Segmentation Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .backbone import DualStreamSwinV2, ContextProjection
from .symmetry import SymmetryModule
from .fpn import FPN
from .panoptic_head import PanopticHead
from .hover_branch import HoVerBranch
from .text_encoder import TextEncoder
from .cross_modal_attention import CrossModalAttention


class SymPanICHNetV2(nn.Module):
    """
    SymPanICH-Net v2: Text-Guided Symmetry-Aware Panoptic Segmentation.

    Pipeline:
        Input (3D CT) → Module 1 (Multi-Window 2.5D)
                       → Module 2 (Dual-Stream Swin-T V2)
                       → Module 3 (Symmetry Cross-Attention)
                       → Module 4 (FPN)
                       → Module 7 (Text Encoder) ─┐
                       → Module 5 (Panoptic Head + Text Fusion)
                       → Module 6 (HoVer Distance Branch)
                       → Module 8 (AI Clinical Report)
                       → Output (Panoptic Map + Report)
    """

    def __init__(
        self,
        # Backbone config
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        # Context config
        use_context: bool = True,
        context_channels: int = 15,
        # Model dims
        fpn_out_channels: int = 256,
        # Panoptic head config
        num_queries: int = 50,
        num_classes: int = 7,
        num_decoder_layers: int = 9,
        # Text encoder config
        text_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        text_embed_dim: int = 256,
        text_descriptions_path: Optional[str] = None,
        # Feature dims
        feature_dims: list = None,
    ):
        super().__init__()
        if feature_dims is None:
            feature_dims = [96, 192, 384, 768]

        self.use_context = use_context
        self.num_classes = num_classes

        # Module 1: 2.5D Context Projection (if using 2.5D input)
        if use_context:
            self.context_proj = ContextProjection(
                in_channels=context_channels,
                out_channels=3,
            )
        else:
            self.context_proj = None

        # Module 2: Dual-Stream Swin-T V2 Backbone
        self.backbone = DualStreamSwinV2(
            model_name=backbone_name,
            pretrained=pretrained,
            in_channels=3,
        )

        # Module 3: Symmetry-Aware Cross-Attention
        self.symmetry = SymmetryModule(
            feature_dims=feature_dims,
            num_heads=8,
            dropout=0.1,
        )

        # Module 4: Feature Pyramid Network
        self.fpn = FPN(
            in_channels=feature_dims,
            out_channels=fpn_out_channels,
        )

        # Module 5: Panoptic Head (includes Pixel Decoder + Transformer Decoder)
        self.panoptic_head = PanopticHead(
            hidden_dim=fpn_out_channels,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
        )

        # Module 6: HoVer Distance Branch
        self.hover_branch = HoVerBranch(
            in_channels=fpn_out_channels,
            mid_channels=128,
        )

        # Module 7: Text Encoder
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=text_embed_dim,
            descriptions_path=text_descriptions_path,
        )

        # Cross-Modal Text-Vision Attention (bridges Module 7 → Module 5)
        self.cross_modal = CrossModalAttention(
            visual_dim=fpn_out_channels,
            text_dim=text_embed_dim,
        )

    def forward(
        self,
        images: torch.Tensor,
        images_flipped: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all modules.

        Args:
            images: Original CT images (B, C, H, W).
                   C=3 for standard input, C=15 for 2.5D input.
            images_flipped: Horizontally flipped CT images (B, C, H, W).

        Returns:
            Dictionary with:
                - 'pred_logits': Class predictions (B, N, num_classes)
                - 'pred_masks': Mask predictions (B, N, H/4, W/4)
                - 'hv_maps': H/V distance maps (B, 2, H/4, W/4)
                - 'pixel_embedding': Per-pixel features (B, C, H/4, W/4)
                - 'text_embeddings': Text embeddings (6, embed_dim)
                - 'aux_outputs': Intermediate predictions for deep supervision
        """
        # Module 1: Project 2.5D context (15ch → 3ch) if needed
        if self.use_context and self.context_proj is not None:
            images = self.context_proj(images)
            images_flipped = self.context_proj(images_flipped)

        # Module 2: Extract features from both streams
        f_orig, f_flip = self.backbone(images, images_flipped)

        # Module 3: Symmetry-aware cross-attention
        f_sym = self.symmetry(f_orig, f_flip)

        # Module 4: Feature Pyramid Network
        fpn_features = self.fpn(f_sym)  # [P1, P2, P3, P4]

        # Module 7: Get text embeddings
        text_embeddings = self.text_encoder(device=images.device)

        # Module 5: Panoptic head
        panoptic_out = self.panoptic_head(fpn_features)

        # Apply cross-modal attention to pixel embeddings
        pixel_emb = panoptic_out['pixel_embedding']
        pixel_emb_enhanced = self.cross_modal(pixel_emb, text_embeddings)

        # Module 6: HoVer distance maps
        hv_maps = self.hover_branch(pixel_emb_enhanced)

        return {
            'pred_logits': panoptic_out['pred_logits'],
            'pred_masks': panoptic_out['pred_masks'],
            'hv_maps': hv_maps,
            'pixel_embedding': pixel_emb_enhanced,
            'text_embeddings': text_embeddings,
            'aux_outputs': panoptic_out['aux_outputs'],
        }

    def freeze_backbone(self):
        """Freeze backbone parameters (for Phase 1 warmup)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (for Phase 2+)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_text_encoder(self):
        """Freeze text encoder parameters."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def unfreeze_text_encoder(self):
        """Unfreeze text encoder for fine-tuning."""
        self.text_encoder.unfreeze()

    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """
        Get parameter groups with differential learning rates.

        Args:
            base_lr: Base learning rate for head parameters.

        Returns:
            List of param group dicts for optimizer.
        """
        backbone_params = list(self.backbone.parameters())
        text_params = list(self.text_encoder.parameters())
        head_params = [
            p for name, p in self.named_parameters()
            if not any(p is bp for bp in backbone_params)
            and not any(p is tp for tp in text_params)
        ]

        return [
            {"params": backbone_params, "lr": base_lr * 0.1, "name": "backbone"},
            {"params": text_params, "lr": base_lr * 0.05, "name": "text_encoder"},
            {"params": head_params, "lr": base_lr, "name": "heads"},
        ]
