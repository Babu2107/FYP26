"""
CT Windowing Utilities for Multi-Window Input Pipeline (Module 1).

Applies different Hounsfield Unit (HU) windows to CT slices to reveal
different anatomical structures relevant for hemorrhage detection.
"""

import numpy as np
import torch


# Standard CT window presets for ICH detection
WINDOW_PRESETS = {
    "brain":    {"center": 40,  "width": 80},
    "subdural": {"center": 75,  "width": 215},
    "bone":     {"center": 600, "width": 2800},
}


def apply_window(ct_slice: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply a CT window to a slice of Hounsfield Unit (HU) values.

    Clamps HU values to [center - width/2, center + width/2] and normalizes to [0, 1].

    Args:
        ct_slice: 2D numpy array of raw HU values (H, W).
        center: Window center in HU.
        width: Window width in HU.

    Returns:
        Windowed and normalized image in [0, 1] range, shape (H, W).
    """
    lower = center - width / 2.0
    upper = center + width / 2.0
    windowed = np.clip(ct_slice, lower, upper)
    normalized = (windowed - lower) / (upper - lower + 1e-8)
    return normalized.astype(np.float32)


def get_multi_window(ct_slice: np.ndarray) -> np.ndarray:
    """
    Create a 3-channel image from 3 CT windows (brain, subdural, bone).

    Args:
        ct_slice: 2D numpy array of raw HU values (H, W).

    Returns:
        3-channel image, shape (H, W, 3) with values in [0, 1].
        Channel 0: Brain window  (gray/white matter, large bleeds)
        Channel 1: Subdural window (subdural & epidural near skull)
        Channel 2: Bone window (skull fractures, calcifications)
    """
    brain = apply_window(ct_slice, **WINDOW_PRESETS["brain"])
    subdural = apply_window(ct_slice, **WINDOW_PRESETS["subdural"])
    bone = apply_window(ct_slice, **WINDOW_PRESETS["bone"])
    return np.stack([brain, subdural, bone], axis=-1)


def get_context_slices(
    volume: np.ndarray,
    target_idx: int,
    context: int = 2,
) -> np.ndarray:
    """
    Extract 2.5D context: target slice ± context neighboring slices.
    Each slice gets multi-window treatment → stacked along channel dim.

    Args:
        volume: 3D CT volume (H, W, num_slices) in HU values.
        target_idx: Index of the target slice.
        context: Number of neighboring slices on each side (default: 2).

    Returns:
        Multi-channel image (H, W, (2*context+1)*3) with all windows applied.
        For context=2: shape is (H, W, 15).
    """
    num_slices = volume.shape[2]
    channels = []

    for offset in range(-context, context + 1):
        idx = target_idx + offset
        # Pad with zeros for boundary slices
        if idx < 0 or idx >= num_slices:
            h, w = volume.shape[:2]
            channels.append(np.zeros((h, w, 3), dtype=np.float32))
        else:
            channels.append(get_multi_window(volume[:, :, idx]))

    # Stack: (H, W, (2*context+1)*3) → e.g., (H, W, 15) for context=2
    return np.concatenate(channels, axis=-1)


def multi_window_to_tensor(multi_window_image: np.ndarray) -> torch.Tensor:
    """
    Convert a multi-window numpy image to a PyTorch tensor.

    Args:
        multi_window_image: (H, W, C) numpy array.

    Returns:
        (C, H, W) float32 tensor.
    """
    return torch.from_numpy(multi_window_image.transpose(2, 0, 1)).float()
