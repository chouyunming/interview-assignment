# Dogs vs. Cats

PyTorch image classification model for distinguishing between dogs and cats. Designed to run on Google Colab with GPU support.

## Quick Start

### Local Setup (Editor Support Only)

For type checking and autocomplete in your editor:

```bash
pip3 install -r requirements.txt
```

### Training with GPU (Recommended)

Run training on [Google Colab](https://colab.research.google.com):

```python
!git clone https://github.com/chouyunming/interview-assignment.git
%cd interview-assignment
from src.model import initialize_model
from src.train import train
# ... see notebooks/ for complete examples
```

Or locally (CPU-only):

```bash
python src/main.py --model resnet --epochs 5 --batch-size 32
```

## Usage

### Command Line Training

```bash
python src/main.py [OPTIONS]
```

**Options:**
- `--model {resnet, vitb16}` — Model architecture (default: resnet)
- `--epochs N` — Training epochs (default: 5)
- `--batch-size N` — Batch size (default: 32)
- `--learning-rate LR` — Learning rate (default: 0.001)
- `--pretrained {True, False}` — Use pretrained weights (default: True)
- `--feature-extract {True, False}` — Freeze backbone, train only head (default: False)
- `--wandb {True, False}` — Enable Weights & Biases logging (default: False)
- `--device {cuda, cpu}` — Device (default: cuda if available, else cpu)

**Example:**
```bash
python src/main.py --model vitb16 --epochs 10 --batch-size 64 --wandb True
```

## Project Structure

```
.
├── src/
│   ├── main.py           — Entry point with argument parsing
│   ├── model.py          — Model initialization (ResNet, ViT)
│   ├── dataset.py        — Data loading and augmentation
│   ├── train.py          — Training loop and metrics
│   ├── evaluate.py       — Evaluation functions
│   └── config.py         — Configuration constants
├── notebooks/            — Jupyter notebooks (Colab-compatible)
├── data/                 — Training and validation data (ignored)
├── ckpts/                — Model checkpoints (ignored)
├── logs/                 — Training logs (ignored)
├── requirements.txt      — Python dependencies
└── README.md
```

## Dependencies

- PyTorch 2.0+
- torchvision
- timm (for Vision Transformer)
- wandb (for experiment tracking)
- numpy, Pillow, scikit-learn, matplotlib

## Adding Dependencies

```bash
pip3 install <package-name>
echo <package-name> >> requirements.txt
```

## Architecture

The codebase separates concerns for clean, reusable code:

- **`src/`** — Pure Python modules (no notebook code, no hardcoded paths)
  - Designed to work in any environment (local, Colab, Docker)
  - Handles core logic: models, data loading, training

- **`notebooks/`** — Jupyter notebooks for orchestration and visualization
  - Import from `src/` using relative imports
  - Handle I/O, experiment setup, GPU configuration
  - Compatible with Google Colab

## Logging

Training metrics are saved to:
- `logs/{experiment}/history.json` — Loss and accuracy per epoch
- `ckpts/{experiment}/best_model.pth` — Best model checkpoint
- Weights & Biases (if `--wandb True`)

## License

Interview assignment.
