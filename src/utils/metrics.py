"""
Evaluation Metrics for Panoptic Segmentation.

Includes Panoptic Quality (PQ), Segmentation Quality (SQ),
Recognition Quality (RQ), Dice Score, and IoU.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Dice coefficient for binary masks.

    Args:
        pred: Predicted binary mask (B, H, W) or (H, W).
        target: Ground truth binary mask, same shape as pred.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice score (scalar tensor).
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Intersection over Union for binary masks.

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.
        smooth: Smoothing factor.

    Returns:
        IoU score (scalar tensor).
    """
    pred = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_per_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 7,
    ignore_background: bool = True,
) -> Dict[int, float]:
    """
    Compute Dice score per class.

    Args:
        pred: Predicted class map (H, W) with integer labels.
        target: Ground truth class map (H, W).
        num_classes: Total number of classes.
        ignore_background: If True, skip class 0 (background).

    Returns:
        Dictionary mapping class_id → dice_score.
    """
    results = {}
    start = 1 if ignore_background else 0
    for c in range(start, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        if target_c.sum() == 0 and pred_c.sum() == 0:
            results[c] = 1.0  # Both empty = perfect
        elif target_c.sum() == 0:
            results[c] = 0.0  # False positive
        else:
            results[c] = compute_dice(pred_c, target_c).item()
    return results


def compute_panoptic_quality(
    pred_segments: list,
    gt_segments: list,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).

    PQ = SQ × RQ
    SQ = mean IoU of matched (TP) segments
    RQ = TP / (TP + 0.5*FP + 0.5*FN)

    Args:
        pred_segments: List of dicts with keys:
            - 'mask': binary mask (H, W) numpy array
            - 'class_id': integer class label
        gt_segments: Same format as pred_segments.
        iou_threshold: IoU threshold for matching (default: 0.5).

    Returns:
        Tuple of (PQ, SQ, RQ) as floats.
    """
    if len(gt_segments) == 0 and len(pred_segments) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_segments) == 0:
        return 0.0, 0.0, 0.0

    # Build IoU matrix between predictions and ground truth
    num_pred = len(pred_segments)
    num_gt = len(gt_segments)
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)

    for i, pred in enumerate(pred_segments):
        for j, gt in enumerate(gt_segments):
            if pred["class_id"] != gt["class_id"]:
                continue  # Only match same-class segments
            pred_mask = pred["mask"].astype(bool)
            gt_mask = gt["mask"].astype(bool)
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Greedy matching: match highest IoU pairs first
    matched_pred = set()
    matched_gt = set()
    tp_ious = []

    # Sort all (pred, gt) pairs by IoU descending
    pairs = []
    for i in range(num_pred):
        for j in range(num_gt):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for iou_val, i, j in pairs:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            tp_ious.append(iou_val)

    tp = len(tp_ious)
    fp = num_pred - tp
    fn = num_gt - tp

    sq = np.mean(tp_ious) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    return float(pq), float(sq), float(rq)
