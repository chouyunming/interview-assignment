"""
Configuration constants for Dogs vs. Cats image classification.
"""

import torch

# ==================== Paths ====================
TRAIN_PATH = "./data/train"  # 8000 dogs & 8000 cats
VAL_PATH = "./data/val"      # 2000 dogs & 2000 cats
TEST_PATH = "./data/test"    # Test dataset

# ==================== Data Configuration ====================
NUM_CLASSES = 2
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Normalization values (ImageNet standard)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ==================== Training Configuration ====================
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ITERATION = 1000  # Print metrics every N iterations (batches)

# Model selection
MODEL_NAME = "resnet"  # Options: "resnet", "vitb16"
USE_PRETRAINED = True
FEATURE_EXTRACT = True  # If True, only fine-tune final layer

# ==================== Experiment Configuration ====================
EXPERIMENT_NAME = f"dogs-vs-cats-{MODEL_NAME}"  # Experiment name for tracking

# ==================== Weights & Biases Configuration ====================
WANDB_PROJECT = "dogs-vs-cats"
WANDB_ENABLED = False

# ==================== Checkpoint Configuration ====================
CHECKPOINT_DIR = "./ckpts"  # Directory to save best model checkpoint
LOGS_DIR = "./logs"  # Directory to save training logs and metrics
RESULT_DIR = "./result"  # Directory to save evaluation results
