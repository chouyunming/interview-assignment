# Dogs vs. Cats

Image classification model for distinguishing between dogs and cats.

## Setup

**Local (editor support only):**
```bash
conda env create -f environment.yml
conda activate dogs-vs-cats
```

**Google Colab (GPU training):**
```python
!git clone <repo-url>
%cd interview-assignment
```
Then run notebooks from the `notebooks/` directory.

## Adding Dependencies

```bash
pip3 install <package-name>
echo <package-name> >> requirements.txt
```

## Project Structure

- `src/` - Core model and training code
  - `model.py` - Neural network architecture
  - `dataset.py` - Data loading and preprocessing
  - `train.py` - Training logic
  - `evaluate.py` - Evaluation metrics
- `notebooks/` - Jupyter notebooks (compatible with Colab)
- `data/` - Training and validation datasets
- `ckpts/` - Model checkpoints
