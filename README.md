# Dogs vs. Cats

PyTorch image classification model for distinguishing between dogs and cats. Designed to run on Google Colab with GPU support.

## Data Preparation

Prepare your dataset in the following structure:

```
data/
├── train/
│   ├── cat.1.jpg
│   └── dog.1.jpg
└── val/
    ├── cat.2.jpg
    └── dog.2.jpg
```

Training expects image folders organized by class. Supported formats: JPG, PNG.

## Quick Start

### Local Setup (Editor Support Only)

**Option 1: Using pip**

```bash
pip install -r requirements.txt
```

**Option 2: Using conda** (Recommended)

```bash
conda env create -f environment.yml
conda activate dogs-vs-cats
```

### Training with GPU (Recommended)

Run training on [Google Colab](https://colab.research.google.com):

```python
!git clone https://github.com/chouyunming/interview-assignment.git
%cd interview-assignment
# ... see notebooks/ for complete examples
```

Or locally:

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

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py --checkpoint ckpts/experiment_name/best_model.pth
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

- **PyTorch 2.0+** — Deep learning framework
- **torchvision** — Image utilities and pretrained models
- **timm** — Vision Transformer and other model architectures
- **wandb** — Experiment tracking and visualization
- **scikit-learn** — Metrics (confusion matrix, classification report)
- **numpy, Pillow, matplotlib, seaborn** — Data processing and visualization

## Adding Dependencies

```bash
pip install <package-name>
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

