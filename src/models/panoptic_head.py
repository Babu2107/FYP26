"""
Module 5: Mask2Former-Style Panoptic Head.

Combines a Pixel Decoder (multi-scale feature processing) with a
Transformer Decoder (object query-based mask + class prediction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PixelDecoder(nn.Module):
    """
    Multi-scale pixel decoder that processes FPN features into
    per-pixel embeddings used for mask prediction.

    Uses progressive upsampling with skip connections (simplified
    version of multi-scale deformable attention).
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()

        # Progressive upsampling layers with skip connections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            ))

        # Final projection
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # Lateral projections for skip connections from each FPN level
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            for _ in range(4)
        ])

    def forward(self, fpn_features: list) -> torch.Tensor:
        """
        Process FPN features into per-pixel embeddings.

        Args:
            fpn_features: [P1(1/4), P2(1/8), P3(1/16), P4(1/32)] each (B, 256, H_i, W_i)

        Returns:
            Per-pixel embeddings (B, 256, H/4, W/4) — at 1/4 scale.
        """
        # Start from coarsest (P4) and progressively upsample
        x = self.skip_projs[3](fpn_features[3])  # Start at P4

        # Process through layers with skip connections
        for i, layer in enumerate(self.layers):
            # Upsample to next finer scale
            target_idx = 2 - i  # 2, 1, 0 → P3, P2, P1
            if target_idx >= 0:
                target_size = fpn_features[target_idx].shape[2:]
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
                # Add skip connection
                x = x + self.skip_projs[target_idx](fpn_features[target_idx])

            x = layer(x)

        return self.output_proj(x)


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with masked cross-attention."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Masked cross-attention (queries attend to feature map)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        features: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: Object queries (B, N, C).
            features: Flattened feature map (B, H*W, C).
            attn_mask: Optional attention mask for masked cross-attention
                       (B*num_heads, N, H*W) — True means MASKED (ignored).

        Returns:
            Updated queries (B, N, C).
        """
        # Self-attention
        q = self.norm1(queries)
        q2, _ = self.self_attn(q, q, q)
        queries = queries + q2

        # Masked cross-attention
        q = self.norm2(queries)
        q2, _ = self.cross_attn(q, features, features, attn_mask=attn_mask)
        queries = queries + q2

        # FFN
        q = self.norm3(queries)
        queries = queries + self.ffn(q)

        return queries


class PanopticHead(nn.Module):
    """
    Mask2Former-style panoptic segmentation head.

    Uses a transformer decoder with learnable object queries to predict
    per-query class labels and binary masks.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 50,
        num_classes: int = 6,
        num_decoder_layers: int = 9,
        num_heads: int = 8,
        num_feature_levels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers
        self.num_feature_levels = num_feature_levels

        # Pixel decoder
        self.pixel_decoder = PixelDecoder(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
        )

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Prediction heads (applied after each decoder layer for deep supervision)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Level embedding to distinguish feature scales
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)

    def forward(
        self, fpn_features: list
    ) -> dict:
        """
        Full panoptic head forward pass.

        Args:
            fpn_features: [P1, P2, P3, P4] from FPN, each (B, 256, H_i, W_i).

        Returns:
            Dictionary with:
                - 'pred_logits': class predictions (B, N, num_classes)
                - 'pred_masks': mask predictions (B, N, H/4, W/4)
                - 'pixel_embedding': per-pixel features (B, C, H/4, W/4)
                - 'aux_outputs': list of intermediate predictions for deep supervision
        """
        B = fpn_features[0].shape[0]

        # Step 1: Pixel decoder → per-pixel embeddings
        pixel_embedding = self.pixel_decoder(fpn_features)  # (B, C, H/4, W/4)

        # Step 2: Prepare multi-scale features for cross-attention
        # Use P2, P3, P4 (3 levels) and cycle through them
        ms_features = []
        for level_idx, feat_idx in enumerate([1, 2, 3]):  # P2, P3, P4
            feat = fpn_features[feat_idx]  # (B, C, H_i, W_i)
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # (B, H_i*W_i, C)
            feat_flat = feat_flat + self.level_embed.weight[level_idx].unsqueeze(0).unsqueeze(0)
            ms_features.append(feat_flat)

        # Step 3: Initialize queries
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)

        # Step 4: Run through decoder layers
        aux_outputs = []
        pred_mask = None

        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            # Select which feature level to attend to (cycle: P4→P3→P2→P4→...)
            level_idx = layer_idx % self.num_feature_levels
            feat_seq = ms_features[self.num_feature_levels - 1 - level_idx]  # Coarse to fine

            # Create attention mask from previous layer's mask prediction
            attn_mask = None
            if pred_mask is not None:
                # Resize predicted mask to match feature spatial size
                feat_level = 3 - (layer_idx % self.num_feature_levels)  # P4=3, P3=2, P2=1
                target_size = fpn_features[feat_level].shape[2:]
                mask_resized = F.interpolate(
                    pred_mask.float(), size=target_size, mode="bilinear", align_corners=False
                )
                # Create binary attention mask (True = masked/ignored)
                attn_mask = (mask_resized.flatten(2) < 0.5)  # (B, N, H_i*W_i)
                # Expand for multi-head: (B*num_heads, N, H_i*W_i)
                # Ensure at least some positions are attended to
                attn_mask = attn_mask & (~attn_mask.all(dim=-1, keepdim=True))

            # Decoder layer
            queries = decoder_layer(queries, feat_seq, attn_mask=None)
            # Note: using attn_mask=None for stability during initial training;
            # enable masked attention after warmup for better small-object detection

            # Intermediate predictions (for deep supervision)
            pred_logits = self.class_head(queries)  # (B, N, num_classes)
            mask_emb = self.mask_embed(queries)      # (B, N, C)

            # Mask prediction via dot product with pixel embeddings
            pred_mask = torch.einsum(
                'bqc,bchw->bqhw', mask_emb, pixel_embedding
            )  # (B, N, H/4, W/4)

            aux_outputs.append({
                'pred_logits': pred_logits,
                'pred_masks': pred_mask,
            })

        # Final predictions = last layer's output
        return {
            'pred_logits': pred_logits,
            'pred_masks': pred_mask,
            'pixel_embedding': pixel_embedding,
            'aux_outputs': aux_outputs[:-1],  # All except last (which is the main output)
        }
