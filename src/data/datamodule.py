"""
PyTorch Lightning DataModule for ICH Dataset.

Handles patient-level train/val/test splitting and data loading.
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from .dataset import ICHDataset
from .augmentations import get_train_transforms, get_val_transforms


class ICHDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for ICH panoptic segmentation.

    Splits data at PATIENT level (not slice level) to avoid data leakage.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        batch_size: int = 4,
        num_workers: int = 4,
        context_slices: int = 2,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_slices = context_slices
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        train_transform = get_train_transforms(self.image_size)
        val_transform = get_val_transforms(self.image_size)

        # Check if data has pre-defined splits
        data_path = Path(self.data_dir)
        has_splits = (data_path / "train").exists()

        if has_splits:
            if stage == "fit" or stage is None:
                self.train_dataset = ICHDataset(
                    str(data_path / "train"), split="train",
                    transform=train_transform, image_size=self.image_size,
                    context_slices=self.context_slices,
                )
                self.val_dataset = ICHDataset(
                    str(data_path / "val"), split="val",
                    transform=val_transform, image_size=self.image_size,
                    context_slices=self.context_slices,
                )
            if stage == "test" or stage is None:
                test_dir = data_path / "test"
                if not test_dir.exists():
                    test_dir = data_path / "val"
                self.test_dataset = ICHDataset(
                    str(test_dir), split="test",
                    transform=val_transform, image_size=self.image_size,
                    context_slices=self.context_slices,
                )
        else:
            # Auto-split at patient level
            full_dataset = ICHDataset(
                self.data_dir, split="all",
                transform=None, image_size=self.image_size,
                context_slices=self.context_slices,
            )

            # Group samples by patient
            patient_samples = {}
            for idx, sample in enumerate(full_dataset.samples):
                pid = sample.get('patient_id', f'p{idx}')
                if pid not in patient_samples:
                    patient_samples[pid] = []
                patient_samples[pid].append(idx)

            # Split patients
            patients = sorted(patient_samples.keys())
            n_train = int(len(patients) * self.train_ratio)
            n_val = int(len(patients) * self.val_ratio)

            train_patients = patients[:n_train]
            val_patients = patients[n_train:n_train + n_val]
            test_patients = patients[n_train + n_val:]

            # Create index lists
            train_indices = [i for p in train_patients for i in patient_samples[p]]
            val_indices = [i for p in val_patients for i in patient_samples[p]]
            test_indices = [i for p in test_patients for i in patient_samples[p]]

            if stage == "fit" or stage is None:
                self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
                self.train_dataset.dataset.transform = train_transform

                self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
                # Val uses val transform (we'll handle this in getitem)

            if stage == "test" or stage is None:
                self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return self.val_dataloader()
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )
