"""
Module 3: Symmetry-Aware Cross-Attention (SACA).

The key novelty of SymPanICH-Net — uses cross-attention between
original and flipped brain features to learn which asymmetries
are clinically meaningful (hemorrhages) vs normal variation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetryCrossAttention(nn.Module):
    """
    Symmetry-Aware Cross-Attention for a single feature scale.

    Mechanism:
        1. Cross-attention: Q=F_orig, K=V=F_flip → discovers asymmetries
        2. Difference gating: g = σ(W·|F_orig - F_flip|)
        3. Gated fusion: F_sym = (1-g)·F_orig + g·CrossAttn(F_orig, F_flip)

    When g≈0 (symmetric/normal) → keep original features
    When g≈1 (asymmetric/hemorrhage) → use cross-attention output
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Gating mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, f_orig: torch.Tensor, f_flip: torch.Tensor) -> torch.Tensor:
        """
        Apply symmetry-aware cross-attention.

        Args:
            f_orig: Original features (B, C, H, W).
            f_flip: Flipped features (B, C, H, W).

        Returns:
            Symmetry-enhanced features (B, C, H, W).
        """
        B, C, H, W = f_orig.shape

        # Reshape to sequence: (B, C, H, W) → (B, H*W, C)
        orig_seq = f_orig.flatten(2).permute(0, 2, 1)  # (B, N, C)
        flip_seq = f_flip.flatten(2).permute(0, 2, 1)  # (B, N, C)

        orig_normed = self.norm1(orig_seq)
        flip_normed = self.norm2(flip_seq)

        # Multi-head cross-attention: Q=orig, K=V=flip
        Q = self.q_proj(orig_normed).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(flip_normed).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(flip_normed).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: (B, heads, N, head_dim) × (B, heads, head_dim, N) → (B, heads, N, N)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, C)  # (B, N, C)
        attn_out = self.out_proj(attn_out)

        # Difference gating: g = σ(W·|F_orig - F_flip|)
        diff = torch.abs(orig_seq - flip_seq)
        gate = self.gate_proj(diff)  # (B, N, C) → values in [0, 1]

        # Gated fusion: F_sym = (1-g)·F_orig + g·CrossAttn_output
        fused = (1 - gate) * orig_seq + gate * attn_out

        # Reshape back to spatial: (B, N, C) → (B, C, H, W)
        output = fused.permute(0, 2, 1).view(B, C, H, W)
        return output


class SymmetryModule(nn.Module):
    """
    Applies Symmetry-Aware Cross-Attention at each feature scale (F1-F4).
    """

    def __init__(
        self,
        feature_dims: list = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_dims is None:
            feature_dims = [96, 192, 384, 768]

        self.saca_layers = nn.ModuleList([
            SymmetryCrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
            for dim in feature_dims
        ])

    def forward(
        self, f_orig: list, f_flip: list
    ) -> list:
        """
        Apply SACA at each scale.

        Args:
            f_orig: List of 4 feature tensors from original stream.
            f_flip: List of 4 feature tensors from flipped stream.

        Returns:
            List of 4 symmetry-enhanced feature tensors.
        """
        f_sym = []
        for saca, fo, ff in zip(self.saca_layers, f_orig, f_flip):
            f_sym.append(saca(fo, ff))
        return f_sym
