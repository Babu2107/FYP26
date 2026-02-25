"""
Visualization Utilities for ICH Segmentation.
"""

import numpy as np
import torch
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Color map for hemorrhage types (RGB, 0-255)
CLASS_COLORS = {
    0: (0, 0, 0),         # Background — black
    1: (255, 0, 0),       # Intraventricular — red
    2: (0, 255, 0),       # Intraparenchymal — green
    3: (0, 0, 255),       # Subarachnoid — blue
    4: (255, 255, 0),     # Epidural — yellow
    5: (255, 0, 255),     # Subdural — magenta
    6: (128, 128, 128),   # Ambiguous — gray
}

CLASS_NAMES = [
    "Background", "Intraventricular", "Intraparenchymal",
    "Subarachnoid", "Epidural", "Subdural", "Ambiguous",
]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert class label mask to RGB colorized image.

    Args:
        mask: (H, W) integer array with class labels 0-6.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgb[mask == cls_id] = color
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay colorized mask on grayscale/RGB image.

    Args:
        image: (H, W) or (H, W, 3) image in [0, 1] range.
        mask: (H, W) class label mask.
        alpha: Overlay transparency.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] > 3:
        image = image[:, :, :3]

    image_uint8 = (image * 255).astype(np.uint8)
    mask_rgb = colorize_mask(mask)

    # Only blend where mask > 0
    foreground = mask > 0
    blended = image_uint8.copy()
    blended[foreground] = (
        (1 - alpha) * image_uint8[foreground] + alpha * mask_rgb[foreground]
    ).astype(np.uint8)

    return blended


def plot_prediction(
    image: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "SymPanICH-Net v2 Prediction",
):
    """
    Plot image with ground truth and prediction side by side.

    Args:
        image: (H, W) or (H, W, 3) input image.
        gt_mask: Optional (H, W) ground truth mask.
        pred_mask: Optional (H, W) predicted mask.
        save_path: Optional path to save the figure.
        title: Plot title.
    """
    if not HAS_MPL:
        print("matplotlib not available for visualization")
        return

    num_panels = 1
    if gt_mask is not None:
        num_panels += 1
    if pred_mask is not None:
        num_panels += 1

    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 6))
    if num_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Input image
    if image.ndim == 2:
        axes[panel_idx].imshow(image, cmap='gray')
    else:
        axes[panel_idx].imshow(image[:, :, :3] if image.shape[2] > 3 else image)
    axes[panel_idx].set_title("Input CT Slice")
    axes[panel_idx].axis("off")
    panel_idx += 1

    # Ground truth
    if gt_mask is not None:
        axes[panel_idx].imshow(overlay_mask(image if image.ndim == 3 else np.stack([image]*3, -1), gt_mask))
        axes[panel_idx].set_title("Ground Truth")
        axes[panel_idx].axis("off")
        panel_idx += 1

    # Prediction
    if pred_mask is not None:
        axes[panel_idx].imshow(overlay_mask(image if image.ndim == 3 else np.stack([image]*3, -1), pred_mask))
        axes[panel_idx].set_title("Prediction")
        axes[panel_idx].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255, label=CLASS_NAMES[i])
        for i in range(1, 7)
    ]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()
