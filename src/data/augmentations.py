"""
Augmentation Pipeline for ICH Dataset.

Heavy augmentations critical for small dataset (81 patients).
Uses albumentations for efficient, composable transforms.

NOTE: HV maps are NOT augmented — they are recomputed from
the augmented mask in the dataset __getitem__ method.
"""

import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBU = True
except ImportError:
    HAS_ALBU = False

import torch


def get_train_transforms(image_size: int = 256):
    """
    Training augmentations — aggressive for small dataset.

    Only image and mask are augmented. HV maps are recomputed
    from the augmented mask afterward (in the dataset).
    """
    if not HAS_ALBU:
        return _fallback_transform(image_size)

    return A.Compose([
        # Geometric transforms (applied to both image and mask)
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.RandomScale(scale_limit=0.15, p=0.3),
        A.PadIfNeeded(
            min_height=int(image_size * 0.875),
            min_width=int(image_size * 0.875),
            border_mode=0,
            value=0,
            mask_value=0,
        ),
        A.RandomCrop(height=int(image_size * 0.875), width=int(image_size * 0.875), p=0.5),
        A.Resize(image_size, image_size),

        # Pixel transforms (applied to image ONLY, not mask)
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(
            max_holes=3, max_height=32, max_width=32,
            min_holes=1, min_height=16, min_width=16,
            fill_value=0, p=0.2,
        ),
        ToTensorV2(),
    ])
    # NOTE: No additional_targets for hv_map — it's recomputed after augmentation


def get_val_transforms(image_size: int = 256):
    """Validation transforms — just resize and convert."""
    if not HAS_ALBU:
        return _fallback_transform(image_size)

    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])


class _fallback_transform:
    """Simple fallback when albumentations is not installed."""

    def __init__(self, image_size: int = 256):
        self.size = image_size

    def __call__(self, image, mask=None, **kwargs):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.expand_dims(image, 0)
            elif image.ndim == 3 and image.shape[2] <= 15:
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()

        result = {'image': image}
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
            result['mask'] = mask

        return result
