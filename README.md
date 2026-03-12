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

### Using Docker

```bash
docker build -t dogs-vs-cats .
docker run dogs-vs-cats python src/main.py --model resnet --epochs 5 --batch-size 32
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
- `--feature-extract {True, False}` — Freeze backbone, train only head (default: True)
- `--wandb {True, False}` — Enable Weights & Biases logging (default: False)
- `--device {cuda, cpu}` — Device (default: cuda if available, else cpu)

**Example:**
```bash
python src/main.py --model vitb16 --epochs 5 --pretrained True --wandb True
```

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py --experiment-name dogs-vs-cats-vitb16-pretrained
```

Results are saved to `result/{experiment-name}/`:
- `metrics.json` — Accuracy, precision, recall, and AUC
- `confusion_matrix.png` — Confusion matrix plot
- `roc_curve.png` — ROC curve plot

#### Metrics

The evaluation computes the following metrics:

*1.* $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

*2.* $$\text{Precision} = \frac{TP}{TP + FP}$$

*3.* $$\text{Recall} = \frac{TP}{TP + FN}$$

*4.* $$\text{AUC (Area Under the Receiver Operating Characteristic Curve)}$$
- Measures the probability that the model ranks a random positive example higher than a random negative example
- Ranges from 0 to 1, with 1.0 being perfect classification

**Legend:**
- $$\text{TP}$$ — True Positives (correctly predicted dogs)
- $$\text{TN}$$ — True Negatives (correctly predicted cats)
- $$\text{FP}$$ — False Positives (cats predicted as dogs)
- $$\text{FN}$$ — False Negatives (dogs predicted as cats)

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
├── result/               — Evaluation results (ignored)
├── Dockerfile            — Docker container setup
├── environment.yml       — Conda environment definition
├── requirements.txt      — Python dependencies (pip)
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
- **Weights & Biases** (if `--wandb True`) — View training runs and experiments:
  - [Dogs vs. Cats Wandb Dashboard](https://wandb.ai/chouyunming-national-chung-hsing-university/dogs-vs-cats?nw=nwuserchouyunming)

## License

Interview assignment.



