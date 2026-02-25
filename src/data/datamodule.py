"""
PyTorch Lightning DataModule for ct-ich PhysioNet Dataset.

Handles patient-level train/val/test splitting to prevent data leakage.
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

try:
    import pytorch_lightning as pl
except ImportError:
    try:
        import lightning as pl
    except ImportError:
        raise ImportError("Install pytorch-lightning or lightning")

from .dataset import ICHDataset
from .augmentations import get_train_transforms, get_val_transforms


class ICHDataModule(pl.LightningDataModule):
    """
    DataModule for ct-ich dataset with patient-level splitting.

    Split: 60% train / 20% val / 20% test (at patient level).
    75 patients → ~45 train / 15 val / 15 test
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
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Create train/val/test datasets with patient-level splits."""
        # First, create a temp dataset to discover all patients
        temp_dataset = ICHDataset(
            data_dir=self.data_dir,
            transform=None,
            image_size=self.image_size,
            context_slices=self.context_slices,
        )

        all_patients = temp_dataset.get_all_patient_ids()
        n_total = len(all_patients)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        # Deterministic split
        train_patients = all_patients[:n_train]
        val_patients = all_patients[n_train:n_train + n_val]
        test_patients = all_patients[n_train + n_val:]

        print(f"Dataset split: {n_total} patients total")
        print(f"  Train: {len(train_patients)} patients → {train_patients}")
        print(f"  Val:   {len(val_patients)} patients → {val_patients}")
        print(f"  Test:  {len(test_patients)} patients → {test_patients}")

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ICHDataset(
                data_dir=self.data_dir,
                transform=get_train_transforms(self.image_size),
                image_size=self.image_size,
                context_slices=self.context_slices,
                patient_ids=train_patients,
            )
            self.val_dataset = ICHDataset(
                data_dir=self.data_dir,
                transform=get_val_transforms(self.image_size),
                image_size=self.image_size,
                context_slices=self.context_slices,
                patient_ids=val_patients,
            )
            print(f"  Train samples: {len(self.train_dataset)}")
            print(f"  Val samples:   {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = ICHDataset(
                data_dir=self.data_dir,
                transform=get_val_transforms(self.image_size),
                image_size=self.image_size,
                context_slices=self.context_slices,
                patient_ids=test_patients,
            )
            print(f"  Test samples:  {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return self.val_dataloader()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
