"""
ICH Dataset: PyTorch Dataset for Intracranial Hemorrhage CT data.

Flexible dataset that can handle multiple directory structures.
Provide your data path and it will auto-detect the format.
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Callable

from ..utils.ct_windows import get_multi_window, get_context_slices
from ..models.hover_branch import compute_hv_maps

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class ICHDataset(Dataset):
    """
    Dataset for ICH panoptic segmentation.

    Supports two modes:
    1. Raw mode: loads from NIfTI volumes on-the-fly
    2. Preprocessed mode: loads from pre-saved numpy arrays

    The dataset will be adapted once the actual data format is known.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 256,
        context_slices: int = 2,
        preprocessed: bool = False,
    ):
        """
        Args:
            data_dir: Root data directory.
            split: One of 'train', 'val', 'test'.
            transform: Augmentation transforms.
            image_size: Target image size.
            context_slices: Number of context slices for 2.5D.
            preprocessed: If True, load from preprocessed numpy files.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.context_slices = context_slices
        self.preprocessed = preprocessed

        # Discover data samples
        self.samples = self._discover_samples()

    def _discover_samples(self) -> list:
        """
        Auto-discover dataset samples based on directory structure.

        Supports:
        - Structure A: data_dir/patient_XXX/ct.nii + mask.nii
        - Structure B: data_dir/images/*.npy + data_dir/masks/*.npy
        - Structure C: data_dir/split_name/images/ + masks/

        Returns:
            List of sample dicts with paths.
        """
        samples = []

        # Try Structure A: NIfTI patient folders
        patient_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        nifti_found = False
        for pdir in patient_dirs:
            nifti_files = list(pdir.glob("*.nii")) + list(pdir.glob("*.nii.gz"))
            if nifti_files:
                nifti_found = True
                ct_files = [f for f in nifti_files if 'mask' not in f.stem.lower() and 'seg' not in f.stem.lower() and 'label' not in f.stem.lower()]
                mask_files = [f for f in nifti_files if 'mask' in f.stem.lower() or 'seg' in f.stem.lower() or 'label' in f.stem.lower()]

                ct_path = ct_files[0] if ct_files else nifti_files[0]
                mask_path = mask_files[0] if mask_files else None

                # Each slice in the volume becomes a sample
                if HAS_NIBABEL:
                    vol = nib.load(str(ct_path))
                    num_slices = vol.shape[2] if len(vol.shape) >= 3 else 1
                    for s in range(num_slices):
                        samples.append({
                            'ct_path': str(ct_path),
                            'mask_path': str(mask_path) if mask_path else None,
                            'slice_idx': s,
                            'patient_id': pdir.name,
                            'type': 'nifti',
                        })

        if nifti_found:
            return samples

        # Try Structure B: Preprocessed numpy files
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            split_dir = self.data_dir

        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"

        if image_dir.exists():
            for img_path in sorted(image_dir.glob("*.npy")):
                mask_path = mask_dir / img_path.name if mask_dir.exists() else None
                samples.append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_path) if mask_path and mask_path.exists() else None,
                    'type': 'numpy',
                })

        # Try Structure C: PNG/JPG images
        if not samples:
            for ext in ['*.png', '*.jpg', '*.npy']:
                for img_path in sorted(self.data_dir.glob(ext)):
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': None,
                        'type': 'image',
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training sample.

        Returns:
            Dictionary with keys:
                - 'image': (C, H, W) tensor
                - 'image_flipped': (C, H, W) tensor
                - 'mask': (H, W) tensor
                - 'hv_maps': (2, H, W) tensor
                - 'patient_id': string
                - 'slice_idx': int
        """
        sample_info = self.samples[idx]

        if sample_info['type'] == 'nifti':
            return self._load_nifti_sample(sample_info)
        elif sample_info['type'] == 'numpy':
            return self._load_numpy_sample(sample_info)
        else:
            return self._load_image_sample(sample_info)

    def _load_nifti_sample(self, info: dict) -> dict:
        """Load a slice from a NIfTI volume."""
        import nibabel as nib

        # Load volume (with caching potential)
        vol = nib.load(info['ct_path']).get_fdata().astype(np.float32)
        slice_idx = info['slice_idx']

        # Multi-window with 2.5D context
        if self.context_slices > 0:
            image = get_context_slices(vol, slice_idx, self.context_slices)
        else:
            image = get_multi_window(vol[:, :, slice_idx])

        # Resize
        import cv2
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Flipped version
        image_flipped = np.flip(image, axis=1).copy()

        # Load mask
        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        if info['mask_path']:
            mask_vol = nib.load(info['mask_path']).get_fdata()
            if slice_idx < mask_vol.shape[2]:
                mask = mask_vol[:, :, slice_idx]
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(np.int64)

        # Compute HV maps from mask (treating each connected component as instance)
        hv_maps = compute_hv_maps(torch.from_numpy(mask)).numpy()

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask, hv_map=hv_maps.transpose(1, 2, 0))
            image = transformed['image']
            mask = transformed['mask']
            if 'hv_map' in transformed:
                hv_maps = transformed['hv_map']
                if isinstance(hv_maps, torch.Tensor) and hv_maps.dim() == 3 and hv_maps.shape[2] == 2:
                    hv_maps = hv_maps.permute(2, 0, 1)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()
            hv_maps = torch.from_numpy(hv_maps).float()

        # Create flipped tensor
        if isinstance(image, torch.Tensor):
            image_flipped = torch.flip(image, dims=[2])
        else:
            image_flipped = torch.from_numpy(image_flipped.transpose(2, 0, 1)).float()

        return {
            'image': image,
            'image_flipped': image_flipped,
            'mask': mask,
            'hv_maps': hv_maps if isinstance(hv_maps, torch.Tensor) else torch.from_numpy(hv_maps).float(),
            'patient_id': info.get('patient_id', 'unknown'),
            'slice_idx': info.get('slice_idx', 0),
        }

    def _load_numpy_sample(self, info: dict) -> dict:
        """Load from preprocessed numpy files."""
        image = np.load(info['image_path']).astype(np.float32)

        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        if info['mask_path']:
            mask = np.load(info['mask_path']).astype(np.int64)

        hv_maps = compute_hv_maps(torch.from_numpy(mask)).numpy()

        if self.transform:
            if image.ndim == 3 and image.shape[0] <= 15:
                image = image.transpose(1, 2, 0)
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            if image.ndim == 3 and image.shape[2] <= 15:
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).long()

        image_flipped = torch.flip(image, dims=[2]) if isinstance(image, torch.Tensor) else image

        return {
            'image': image,
            'image_flipped': image_flipped,
            'mask': mask,
            'hv_maps': torch.from_numpy(hv_maps).float(),
            'patient_id': Path(info['image_path']).stem,
            'slice_idx': 0,
        }

    def _load_image_sample(self, info: dict) -> dict:
        """Load from standard image files (PNG/JPG)."""
        import cv2
        image = cv2.imread(info['image_path'], cv2.IMREAD_UNCHANGED)
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        image = cv2.resize(image, (self.image_size, self.image_size)).astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        hv_maps = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()

        image_flipped = torch.flip(image, dims=[2])

        return {
            'image': image,
            'image_flipped': image_flipped,
            'mask': mask,
            'hv_maps': torch.from_numpy(hv_maps).float(),
            'patient_id': Path(info['image_path']).stem,
            'slice_idx': 0,
        }
