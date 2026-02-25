"""
Training Script for SymPanICH-Net v2.

Usage:
    python scripts/train.py --data_dir data/raw --max_epochs 100
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.training.trainer import SymPanICHNetModule
from src.data.datamodule import ICHDataModule


def main():
    parser = argparse.ArgumentParser(description="Train SymPanICH-Net v2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--fast_dev_run", action="store_true", help="Quick test run")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"model": {}, "training": {}, "data": {}, "loss": {}}

    # Override with CLI args
    data_dir = args.data_dir or config.get("data", {}).get("data_dir", "data/raw")
    max_epochs = args.max_epochs or config.get("training", {}).get("max_epochs", 100)
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 4)

    # DataModule
    datamodule = ICHDataModule(
        data_dir=data_dir,
        image_size=config.get("data", {}).get("image_size", 256),
        batch_size=batch_size,
        num_workers=config.get("training", {}).get("num_workers", 4),
        context_slices=config.get("data", {}).get("context_slices", 2),
    )

    # Model
    model = SymPanICHNetModule(
        backbone_name=config.get("model", {}).get("backbone", {}).get("name", "swinv2_tiny_window8_256"),
        pretrained=config.get("model", {}).get("backbone", {}).get("pretrained", True),
        num_queries=config.get("model", {}).get("panoptic_head", {}).get("num_queries", 50),
        num_classes=config.get("data", {}).get("num_classes", 7),
        base_lr=config.get("training", {}).get("optimizer", {}).get("lr", 1e-4),
        weight_decay=config.get("training", {}).get("optimizer", {}).get("weight_decay", 0.05),
        max_epochs=max_epochs,
        cls_weight=config.get("loss", {}).get("cls_weight", 2.0),
        dice_weight=config.get("loss", {}).get("dice_weight", 5.0),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="sympanich-{epoch:02d}-{val/dice:.4f}",
            monitor="val/dice",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/dice",
            patience=config.get("training", {}).get("early_stopping", {}).get("patience", 15),
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = True  # Default CSV logger
    if args.wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=config.get("wandb", {}).get("project", "sympanich-net-v2"),
                log_model=True,
            )
        except ImportError:
            print("WARNING: wandb not installed, using CSV logger")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=args.gpus if torch.cuda.is_available() else "auto",
        precision=args.precision if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=config.get("training", {}).get("gradient_accumulation", 4),
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=config.get("wandb", {}).get("log_every_n_steps", 10),
    )

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)

    # Test
    if not args.fast_dev_run:
        trainer.test(model, datamodule=datamodule)

    print("\n✅ Training complete!")
    print(f"Best model: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
