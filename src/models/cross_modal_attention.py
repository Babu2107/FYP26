"""
Cross-Modal Text-Vision Attention.

Fuses text embeddings with visual features so that each pixel can
attend to clinical descriptions of all hemorrhage types. This allows
the model to leverage textual clinical knowledge during segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between visual features and text embeddings.

    Each pixel (visual query) attends to ALL 6 hemorrhage type text
    descriptions (text keys/values). Pixels matching a description
    (e.g., "crescent-shaped hyperdense collection") will get high
    attention to the corresponding text embedding (subdural).

    Uses a learnable scaling parameter α (initialized to 0.1) with
    residual connection so the model starts with mostly visual features
    and gradually learns how much text guidance to use.
    """

    def __init__(
        self,
        visual_dim: int = 256,
        text_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(visual_dim, visual_dim)  # Visual → Q
        self.k_proj = nn.Linear(text_dim, visual_dim)     # Text → K
        self.v_proj = nn.Linear(text_dim, visual_dim)     # Text → V
        self.out_proj = nn.Linear(visual_dim, visual_dim)

        # Learnable scaling parameter (start small to avoid disrupting visual features)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Normalization
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.output_norm = nn.LayerNorm(visual_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply cross-modal text-vision attention.

        Args:
            visual_features: Per-pixel embeddings (B, C, H, W).
            text_embeddings: Text embeddings (num_types, C) — typically (6, 256).

        Returns:
            Text-enhanced visual features (B, C, H, W).
        """
        B, C, H, W = visual_features.shape
        N_text = text_embeddings.shape[0]  # 6

        # Reshape visual to sequence: (B, C, H, W) → (B, H*W, C)
        visual_seq = visual_features.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        visual_normed = self.visual_norm(visual_seq)

        # Expand text embeddings for batch: (6, C) → (B, 6, C)
        text_expanded = text_embeddings.unsqueeze(0).expand(B, -1, -1)

        # Multi-head projections
        Q = self.q_proj(visual_normed)  # (B, HW, C)
        K = self.k_proj(text_expanded)  # (B, 6, C)
        V = self.v_proj(text_expanded)  # (B, 6, C)

        # Reshape for multi-head attention
        Q = Q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, HW, d)
        K = K.view(B, N_text, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, 6, d)
        V = V.view(B, N_text, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, 6, d)

        # Attention scores: (B, heads, HW, d) × (B, heads, d, 6) → (B, heads, HW, 6)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention: (B, heads, HW, 6) × (B, heads, 6, d) → (B, heads, HW, d)
        text_guided = torch.matmul(attn_weights, V)
        text_guided = text_guided.transpose(1, 2).contiguous().view(B, H * W, C)
        text_guided = self.out_proj(text_guided)

        # Residual connection with learnable scaling
        output = visual_seq + self.alpha * text_guided
        output = self.output_norm(output)

        # Reshape back: (B, HW, C) → (B, C, H, W)
        return output.permute(0, 2, 1).view(B, C, H, W)
