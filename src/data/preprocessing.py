"""
Data Preprocessing: Multi-Window + 2.5D Pipeline (Module 1).

Handles NIfTI volume loading, windowing, context extraction, and resizing.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import cv2

try:
    import nibabel as nib
except ImportError:
    nib = None

from ..utils.ct_windows import get_multi_window, get_context_slices


def load_nifti(path: str) -> np.ndarray:
    """
    Load a NIfTI volume (.nii or .nii.gz).

    Args:
        path: Path to the NIfTI file.

    Returns:
        3D numpy array (H, W, num_slices).
    """
    if nib is None:
        raise ImportError("nibabel is required for NIfTI loading. Install: pip install nibabel")
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def resize_slice(image: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Resize a 2D image or multi-channel image to (size, size).

    Args:
        image: (H, W) or (H, W, C) numpy array.
        size: Target size.

    Returns:
        Resized image (size, size) or (size, size, C).
    """
    if image.ndim == 2:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess_volume(
    ct_path: str,
    mask_path: Optional[str] = None,
    image_size: int = 256,
    context_slices: int = 2,
) -> dict:
    """
    Preprocess a full CT volume into training-ready slices.

    Args:
        ct_path: Path to CT NIfTI volume.
        mask_path: Optional path to segmentation mask NIfTI.
        image_size: Target image size (default: 256).
        context_slices: Number of neighboring slices for 2.5D (default: 2).

    Returns:
        Dictionary with keys:
            - 'images': list of (C, H, W) numpy arrays (multi-window)
            - 'images_flipped': list of horizontally flipped versions
            - 'masks': list of (H, W) mask arrays (if mask_path provided)
            - 'num_slices': number of slices
    """
    volume = load_nifti(ct_path)

    mask_volume = None
    if mask_path is not None:
        mask_volume = load_nifti(mask_path)

    num_slices = volume.shape[2]
    results = {
        'images': [],
        'images_flipped': [],
        'masks': [],
        'num_slices': num_slices,
    }

    for idx in range(num_slices):
        # Multi-window with 2.5D context
        if context_slices > 0:
            multi_ch = get_context_slices(volume, idx, context=context_slices)
        else:
            multi_ch = get_multi_window(volume[:, :, idx])

        # Resize
        multi_ch = resize_slice(multi_ch, image_size)

        # Create flipped version
        multi_ch_flipped = np.flip(multi_ch, axis=1).copy()

        # Transpose to (C, H, W) for PyTorch
        results['images'].append(multi_ch.transpose(2, 0, 1).astype(np.float32))
        results['images_flipped'].append(multi_ch_flipped.transpose(2, 0, 1).astype(np.float32))

        # Process mask
        if mask_volume is not None:
            mask = mask_volume[:, :, idx]
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            results['masks'].append(mask.astype(np.int64))

    return results
