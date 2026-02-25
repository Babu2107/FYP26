"""InfoNCE Contrastive Loss for text-visual alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for aligning visual features with text embeddings.

    For each detected hemorrhage region, pools visual features and computes
    similarity with all text embeddings. The matching type text is the positive,
    all others are negatives.

    L = -log(exp(sim(v_i, t_pos)/τ) / Σ_j exp(sim(v_i, t_j)/τ))
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        masks: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: Per-pixel embeddings (B, C, H, W).
            text_embeddings: Text embeddings (num_types, C) — (6, 256).
            masks: Predicted binary masks (B, N, H, W).
            class_labels: GT class for each query (B, N) — 0=bg, 1-6=types.

        Returns:
            Contrastive loss scalar.
        """
        B, C, H, W = visual_features.shape
        N = masks.shape[1]
        num_types = text_embeddings.shape[0]  # 6

        losses = []

        for b in range(B):
            for n in range(N):
                cls = int(class_labels[b, n].item())
                if cls == 0:  # Skip background queries
                    continue

                # Pool visual features using the mask
                mask = masks[b, n].detach()  # (H, W)
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()  # (H, W)
                mask_binary = (mask_resized > 0.5).float()

                if mask_binary.sum() < 1:
                    continue

                # Masked average pooling
                vis_feat = visual_features[b]  # (C, H, W)
                pooled = (vis_feat * mask_binary.unsqueeze(0)).sum(dim=[1, 2])
                pooled = pooled / mask_binary.sum().clamp(min=1)  # (C,)

                # Normalize
                pooled = F.normalize(pooled, dim=0)
                text_norm = F.normalize(text_embeddings, dim=1)  # (6, C)

                # Compute similarities
                sims = torch.matmul(text_norm, pooled) / self.temperature.abs()  # (6,)

                # Positive = matching class (cls-1 because text_embeddings is 0-indexed for types)
                target_idx = cls - 1  # Convert 1-6 → 0-5
                if target_idx < 0 or target_idx >= num_types:
                    continue

                # InfoNCE loss
                loss = F.cross_entropy(sims.unsqueeze(0), torch.tensor([target_idx], device=sims.device))
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=visual_features.device, requires_grad=True)

        return torch.stack(losses).mean()
