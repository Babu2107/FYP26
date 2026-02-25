"""
Panoptic Fusion: Post-processing to merge predictions into final panoptic map.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from typing import List, Dict, Optional


def panoptic_fusion(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    hv_maps: Optional[torch.Tensor] = None,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_area: int = 50,
    image_size: int = 256,
) -> dict:
    """
    Merge per-query predictions into a unified panoptic segmentation map.

    Steps:
        1. Filter by confidence threshold
        2. Threshold masks → binary
        3. Sort by confidence (descending)
        4. Greedy pixel assignment (highest confidence wins)
        5. Connected components to split disconnected regions
        6. Remove small instances (area < min_area)

    Args:
        pred_logits: Class predictions (N, num_classes).
        pred_masks: Mask predictions (N, H, W) — raw logits.
        hv_maps: Optional H/V distance maps (2, H, W).
        score_threshold: Minimum confidence to keep a prediction.
        mask_threshold: Threshold for binarizing masks.
        min_area: Minimum instance area in pixels.
        image_size: Target output size.

    Returns:
        Dictionary with:
            - 'panoptic_map': (H, W) integer map (class_id * 1000 + instance_id)
            - 'semantic_map': (H, W) class-only map
            - 'segments': list of segment info dicts
    """
    # Move to numpy
    if isinstance(pred_logits, torch.Tensor):
        pred_logits = pred_logits.detach().cpu()
        pred_masks = pred_masks.detach().cpu()

    # Get class predictions and scores
    scores, classes = pred_logits.softmax(dim=-1).max(dim=-1)  # (N,), (N,)
    scores = scores.numpy()
    classes = classes.numpy()

    # Resize masks to full resolution
    pred_masks_full = F.interpolate(
        pred_masks.unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).sigmoid().numpy()  # (N, H, W)

    # Filter by score and non-background
    valid = (scores > score_threshold) & (classes > 0)
    valid_indices = np.where(valid)[0]

    # Sort by confidence
    sorted_idx = valid_indices[np.argsort(-scores[valid_indices])]

    # Greedy assignment
    panoptic_map = np.zeros((image_size, image_size), dtype=np.int32)
    semantic_map = np.zeros((image_size, image_size), dtype=np.int32)
    occupied = np.zeros((image_size, image_size), dtype=bool)
    segments = []
    instance_id = 1

    for idx in sorted_idx:
        cls_id = int(classes[idx])
        score = float(scores[idx])
        mask = pred_masks_full[idx] > mask_threshold

        # Remove already occupied pixels
        mask = mask & ~occupied

        if mask.sum() < min_area:
            continue

        # Connected components to handle disconnected regions
        labeled, num_features = ndimage.label(mask)
        for comp_id in range(1, num_features + 1):
            comp_mask = labeled == comp_id

            if comp_mask.sum() < min_area:
                continue

            panoptic_map[comp_mask] = cls_id * 1000 + instance_id
            semantic_map[comp_mask] = cls_id
            occupied[comp_mask] = True

            # Compute centroid
            ys, xs = np.where(comp_mask)
            segments.append({
                'id': instance_id,
                'class_id': cls_id,
                'score': score,
                'area': int(comp_mask.sum()),
                'centroid': (float(ys.mean()), float(xs.mean())),
                'mask': comp_mask,
            })
            instance_id += 1

    return {
        'panoptic_map': panoptic_map,
        'semantic_map': semantic_map,
        'segments': segments,
    }
