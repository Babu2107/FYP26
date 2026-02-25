"""
Augmentation Pipeline for ICH Dataset.

Heavy augmentations critical for small dataset (81 patients).
Uses albumentations for efficient, composable transforms.
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

    All transforms are applied consistently to image, mask, and additional targets.
    """
    if not HAS_ALBU:
        return _fallback_transform(image_size)

    return A.Compose([
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.RandomScale(scale_limit=0.15, p=0.3),
        A.RandomCrop(height=int(image_size * 0.875), width=int(image_size * 0.875), p=0.5),
        A.Resize(image_size, image_size),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(
            max_holes=3, max_height=32, max_width=32,
            min_holes=1, min_height=16, min_width=16,
            fill_value=0, p=0.2,
        ),
        ToTensorV2(),
    ], additional_targets={
        'mask': 'mask',
        'instance_mask': 'mask',
        'hv_map': 'image',
    })


def get_val_transforms(image_size: int = 256):
    """Validation transforms — just resize and convert."""
    if not HAS_ALBU:
        return _fallback_transform(image_size)

    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ], additional_targets={
        'mask': 'mask',
        'instance_mask': 'mask',
        'hv_map': 'image',
    })


class _fallback_transform:
    """Simple fallback when albumentations is not installed."""

    def __init__(self, image_size: int = 256):
        self.size = image_size

    def __call__(self, image, mask=None, **kwargs):
        # Simple numpy → tensor conversion
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

        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                result[key] = torch.from_numpy(val)
            else:
                result[key] = val

        return result
