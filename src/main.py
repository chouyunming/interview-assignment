"""
Main entry point for Dogs vs. Cats image classification training.
Orchestrates configuration, data loading, model setup, and training.
"""

import os
import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

import config
from dataset import DogCatDataset
from model import initialize_model
from train import train


def str_to_bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def create_transforms(image_size):
    """Create data augmentation transforms."""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])
    }


def create_dataloaders(cfg):
    """Create training and validation dataloaders."""
    transforms_dict = create_transforms(cfg.image_size)

    train_data = DogCatDataset(dir=cfg.train_path, transform=transforms_dict['train'])
    val_data = DogCatDataset(dir=cfg.val_path, transform=transforms_dict['val'])

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader


def setup_model(cfg):
    """Initialize model, criterion, and optimizer."""
    model, _ = initialize_model(
        cfg.model_name,
        config.NUM_CLASSES,
        feature_extract=cfg.feature_extract,
        use_pretrained=cfg.use_pretrained
    )
    model = model.to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    return model, criterion, optimizer


def setup_wandb(cfg, experiment_name):
    """Initialize Weights & Biases tracking with memory-efficient settings."""
    if cfg.wandb_enabled:
        wandb.init(
            project=cfg.wandb_project,
            name=experiment_name,
            config={
                "model": cfg.model_name,
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "optimizer": "Adam",
                "criterion": "CrossEntropyLoss",
                "pretrained": cfg.use_pretrained,
                "feature_extract": cfg.feature_extract,
                "device": cfg.device,
            },
            save_code=False,  # Don't upload source code to save memory
        )


def parse_args(args=None):
    """Parse command line arguments to override default config."""
    parser = argparse.ArgumentParser(
        description="Dogs vs. Cats Image Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
        Examples:
        python main.py --model vitb16 --epochs 10
        python main.py --wandb True --batch-size 64
        python main.py --pretrained False --feature-extract True
        """
    )

    # Paths
    parser.add_argument("--train-path", type=str, default=config.TRAIN_PATH,
                        help=f"Path to training data (default: {config.TRAIN_PATH})")
    parser.add_argument("--val-path", type=str, default=config.VAL_PATH,
                        help=f"Path to validation data (default: {config.VAL_PATH})")

    # Data Configuration
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size for training (default: {config.BATCH_SIZE})")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE,
                        help=f"Image size for preprocessing (default: {config.IMAGE_SIZE})")

    # Training Configuration
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"Number of training epochs (default: {config.EPOCHS})")
    parser.add_argument("--learning-rate", "--lr", type=float, default=config.LEARNING_RATE,
                        help=f"Learning rate (default: {config.LEARNING_RATE})")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME,
                        choices=["resnet", "vitb16"],
                        help=f"Model architecture (default: {config.MODEL_NAME})")
    parser.add_argument("--pretrained", type=str_to_bool, default=config.USE_PRETRAINED,
                        help=f"Use pretrained weights: True or False (default: {config.USE_PRETRAINED})")
    parser.add_argument("--feature-extract", type=str_to_bool, default=config.FEATURE_EXTRACT,
                        help=f"Fine-tune only final layer: True or False (default: {config.FEATURE_EXTRACT})")

    # W&B Configuration
    parser.add_argument("--wandb", type=str_to_bool, default=config.WANDB_ENABLED,
                        help=f"Enable Weights & Biases tracking: True or False (default: {config.WANDB_ENABLED})")
    parser.add_argument("--wandb-project", type=str, default=config.WANDB_PROJECT,
                        help=f"W&B project name (default: {config.WANDB_PROJECT})")

    # Device
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        choices=["cuda", "cpu"],
                        help=f"Device for training (default: {config.DEVICE})")

    cfg = parser.parse_args(args)

    # Convert attribute names to match config style
    cfg.model_name = cfg.model
    cfg.use_pretrained = cfg.pretrained
    cfg.wandb_enabled = cfg.wandb

    return cfg

def main():
    """Main training pipeline."""
    # Parse configuration from command line
    cfg = parse_args()

    # Create data loaders
    print("\n Loading data...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f" Train samples: {len(train_loader.dataset)}")
    print(f" Val samples: {len(val_loader.dataset)}")

    # Setup model
    print("\n  Setting up model...")
    model, criterion, optimizer = setup_model(cfg)
    print(f" Model: {cfg.model_name}")
    print(f" Device: {cfg.device}")

    # Setup wandb
    print("\n Setting up logging...")
    pretrained_suffix = "-pretrained" if cfg.use_pretrained else ""
    experiment_name = f"dogs-vs-cats-{cfg.model_name}{pretrained_suffix}"
    setup_wandb(cfg, experiment_name)
    if cfg.wandb_enabled:
        print(f" W&B project: {cfg.wandb_project}")
        print(f" Experiment: {experiment_name}")
    else:
        print(" W&B tracking disabled")

    # Create experiment-specific directories
    experiment_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, experiment_name)
    experiment_logs_dir = os.path.join(config.LOGS_DIR, experiment_name)

    # Train model
    print("\n🚀 Starting training...\n")
    try:
        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            cfg.epochs,
            cfg.device,
            wandb_enabled=cfg.wandb_enabled,
            checkpoint_dir=experiment_checkpoint_dir,
            logs_dir=experiment_logs_dir,
            iteration_interval=config.ITERATION
            )
    except Exception as e:
        print(f"An error occurred during training: {e}")

    finally:
        # --- W&B Finalization ---
        if cfg.wandb_enabled:
            wandb.finish()
            
    print("\n Training complete!")


if __name__ == "__main__":
    main()
