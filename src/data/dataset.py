"""
ICH Dataset: PyTorch Dataset for the ct-ich PhysioNet Dataset.

Dataset structure:
    data/computed-tomography-.../
        ct_scans/        → 049.nii, 050.nii, ... 130.nii
        masks/           → 049.nii, 050.nii, ... 130.nii (binary masks)
        hemorrhage_diagnosis_raw_ct.csv  → per-slice type labels

CSV columns: PatientNumber, SliceNumber, Intraventricular, Intraparenchymal,
             Subarachnoid, Epidural, Subdural, No_Hemorrhage, Fracture_Yes_No

Masks are BINARY (any hemorrhage = 1). The hemorrhage TYPE comes from the CSV.
We combine them to create multi-class semantic masks:
    0 = background
    1 = intraventricular
    2 = intraparenchymal
    3 = subarachnoid
    4 = epidural
    5 = subdural
If multiple types on the same slice, the mask pixels get the FIRST type label
(since the binary mask doesn't distinguish types per-pixel).
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Callable, List
import cv2

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("WARNING: nibabel not installed. Install with: pip install nibabel")

from ..utils.ct_windows import get_multi_window, get_context_slices
from ..models.hover_branch import compute_hv_maps


# Type columns in the CSV, mapped to class IDs 1-5
TYPE_COLUMNS = {
    'Intraventricular': 1,
    'Intraparenchymal': 2,
    'Subarachnoid': 3,
    'Epidural': 4,
    'Subdural': 5,
}


class ICHDataset(Dataset):
    """
    Dataset for the ct-ich PhysioNet intracranial hemorrhage dataset.

    Each sample is a single 2D CT slice with:
    - Multi-window image (brain/subdural/bone → 3 channels, or 15ch with 2.5D)
    - Horizontally flipped copy
    - Multi-class semantic mask (0=bg, 1-5 = hemorrhage types)
    - H/V distance maps for instance separation
    - Per-slice type labels from CSV
    """

    def __init__(
        self,
        data_dir: str,
        csv_path: str = None,
        transform: Optional[Callable] = None,
        image_size: int = 256,
        context_slices: int = 2,
        patient_ids: Optional[List[int]] = None,
    ):
        """
        Args:
            data_dir: Path to the dataset root (the inner folder containing ct_scans/ and masks/).
            csv_path: Path to hemorrhage_diagnosis_raw_ct.csv (auto-detected if None).
            transform: Augmentation transforms.
            image_size: Target image size (default: 256).
            context_slices: Number of context slices for 2.5D (default: 2).
            patient_ids: If provided, only use these patient IDs (for splitting).
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        self.context_slices = context_slices

        # Auto-detect paths
        self.ct_dir = self._find_dir('ct_scans')
        self.mask_dir = self._find_dir('masks')
        self.csv_path = csv_path or self._find_csv()

        # Load CSV labels
        self.labels_df = pd.read_csv(self.csv_path)

        # Discover all samples (patient_id, slice_idx)
        self.samples = self._build_sample_list(patient_ids)

        # Cache for loaded volumes (avoid re-reading NIfTI per slice)
        self._volume_cache = {}
        self._mask_cache = {}

    def _find_dir(self, name: str) -> Path:
        """Recursively find a directory by name."""
        for root, dirs, files in os.walk(self.data_dir):
            if name in dirs:
                return Path(root) / name
        raise FileNotFoundError(f"Could not find '{name}' directory in {self.data_dir}")

    def _find_csv(self) -> str:
        """Find the hemorrhage CSV file."""
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if 'hemorrhage_diagnosis' in f and f.endswith('.csv'):
                    return os.path.join(root, f)
        raise FileNotFoundError(f"Could not find hemorrhage CSV in {self.data_dir}")

    def _build_sample_list(self, patient_ids: Optional[List[int]] = None) -> list:
        """Build list of (patient_id, slice_idx, types) tuples."""
        samples = []

        # Get available patient NIfTI files
        available_patients = set()
        for f in self.ct_dir.glob("*.nii*"):
            try:
                pid = int(f.stem.split('.')[0])
                available_patients.add(pid)
            except ValueError:
                continue

        # Filter by requested patient IDs
        if patient_ids is not None:
            available_patients = available_patients & set(patient_ids)

        # Group CSV rows by patient
        for pid in sorted(available_patients):
            patient_rows = self.labels_df[self.labels_df['PatientNumber'] == pid]
            ct_path = self._get_nifti_path(self.ct_dir, pid)
            mask_path = self._get_nifti_path(self.mask_dir, pid)

            if ct_path is None:
                continue

            # Get number of slices from NIfTI header
            if HAS_NIBABEL:
                vol = nib.load(str(ct_path))
                num_slices = vol.shape[2] if len(vol.shape) >= 3 else 1
            else:
                num_slices = len(patient_rows)

            for _, row in patient_rows.iterrows():
                slice_num = int(row['SliceNumber'])
                slice_idx = slice_num - 1  # Convert 1-indexed to 0-indexed

                if slice_idx >= num_slices:
                    continue

                # Get hemorrhage types for this slice
                types = []
                for col, cls_id in TYPE_COLUMNS.items():
                    if row.get(col, 0) == 1:
                        types.append(cls_id)

                samples.append({
                    'patient_id': pid,
                    'slice_idx': slice_idx,
                    'slice_num': slice_num,
                    'types': types,
                    'has_hemorrhage': len(types) > 0,
                    'has_fracture': row.get('Fracture_Yes_No', 0) == 1,
                    'ct_path': str(ct_path),
                    'mask_path': str(mask_path) if mask_path else None,
                })

        return samples

    def _get_nifti_path(self, directory: Path, patient_id: int) -> Optional[Path]:
        """Find the NIfTI file for a patient."""
        for ext in ['.nii', '.nii.gz']:
            path = directory / f"{patient_id:03d}{ext}"
            if path.exists():
                return path
            # Try without zero-padding
            path = directory / f"{patient_id}{ext}"
            if path.exists():
                return path
        return None

    def _load_volume(self, path: str) -> np.ndarray:
        """Load a NIfTI volume with caching."""
        if path not in self._volume_cache:
            vol = nib.load(path).get_fdata().astype(np.float32)
            self._volume_cache[path] = vol
            # Keep cache small (max 5 volumes)
            if len(self._volume_cache) > 5:
                oldest_key = next(iter(self._volume_cache))
                del self._volume_cache[oldest_key]
        return self._volume_cache[path]

    def get_all_patient_ids(self) -> List[int]:
        """Get sorted list of all patient IDs in the dataset."""
        return sorted(set(s['patient_id'] for s in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training sample.

        Returns:
            Dictionary with:
                - 'image': (C, H, W) tensor — 3ch or 15ch depending on context_slices
                - 'image_flipped': horizontally flipped copy
                - 'mask': (H, W) tensor — 0=bg, 1-5=hemorrhage types
                - 'hv_maps': (2, H, W) tensor — H/V distance maps
                - 'type_labels': (5,) tensor — binary flags per hemorrhage type
                - 'patient_id': int
                - 'slice_idx': int
                - 'has_hemorrhage': bool
        """
        info = self.samples[idx]

        # Load CT volume
        volume = self._load_volume(info['ct_path'])
        slice_idx = info['slice_idx']

        # ------- IMAGE -------
        if self.context_slices > 0:
            # 2.5D: stack neighboring slices with multi-window
            image = get_context_slices(volume, slice_idx, self.context_slices)
        else:
            # Single slice with multi-window (3 channels)
            image = get_multi_window(volume[:, :, slice_idx])

        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # ------- MASK -------
        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        if info['mask_path'] is not None and info['has_hemorrhage']:
            mask_volume = self._load_volume(info['mask_path'])
            if slice_idx < mask_volume.shape[2]:
                raw_mask = mask_volume[:, :, slice_idx]
                raw_mask = cv2.resize(raw_mask, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)

                # Convert binary mask to multi-class using CSV type labels
                # If multiple types, assign PRIORITY: first type in the list wins
                binary_mask = (raw_mask > 0)
                if info['types']:
                    # Use the primary (first) hemorrhage type as the class label
                    primary_type = info['types'][0]
                    mask[binary_mask] = primary_type

        # ------- HV MAPS -------
        # Create instance-like mask for HoVer (connected components)
        from scipy import ndimage
        instance_mask_np = np.zeros_like(mask, dtype=np.int32)
        if mask.max() > 0:
            labeled, num = ndimage.label(mask > 0)
            instance_mask_np = labeled

        hv_maps = compute_hv_maps(torch.from_numpy(instance_mask_np.astype(np.int64))).numpy()

        # ------- TYPE LABELS -------
        type_labels = np.zeros(5, dtype=np.float32)
        for t in info['types']:
            type_labels[t - 1] = 1.0  # Convert 1-5 to 0-4 index

        # ------- TRANSFORMS -------
        if self.transform:
            # Transpose HV maps for albumentations: (2, H, W) → (H, W, 2)
            hv_for_aug = hv_maps.transpose(1, 2, 0) if hv_maps.shape[0] == 2 else hv_maps

            transformed = self.transform(image=image, mask=mask, hv_map=hv_for_aug)
            image_t = transformed['image']
            mask_t = transformed['mask']
            hv_t = transformed.get('hv_map', hv_for_aug)

            # Handle tensor conversions
            if isinstance(image_t, torch.Tensor):
                image_tensor = image_t.float()
            else:
                image_tensor = torch.from_numpy(image_t.transpose(2, 0, 1) if image_t.ndim == 3 else image_t).float()

            if isinstance(mask_t, torch.Tensor):
                mask_tensor = mask_t.long()
            else:
                mask_tensor = torch.from_numpy(mask_t).long()

            if isinstance(hv_t, torch.Tensor):
                if hv_t.dim() == 3 and hv_t.shape[-1] == 2:
                    hv_tensor = hv_t.permute(2, 0, 1).float()
                else:
                    hv_tensor = hv_t.float()
            else:
                if hv_t.ndim == 3 and hv_t.shape[-1] == 2:
                    hv_tensor = torch.from_numpy(hv_t.transpose(2, 0, 1)).float()
                else:
                    hv_tensor = torch.from_numpy(hv_t).float()
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask).long()
            hv_tensor = torch.from_numpy(hv_maps).float()

        # Create flipped version
        image_flipped = torch.flip(image_tensor, dims=[2])

        return {
            'image': image_tensor,
            'image_flipped': image_flipped,
            'mask': mask_tensor,
            'hv_maps': hv_tensor,
            'type_labels': torch.from_numpy(type_labels),
            'patient_id': info['patient_id'],
            'slice_idx': info['slice_idx'],
            'has_hemorrhage': info['has_hemorrhage'],
        }
