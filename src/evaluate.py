"""
Evaluation module for Dogs vs. Cats image classification.
Computes metrics, confusion matrix, and ROC curve.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
import seaborn as sns

from dataset import DogCatDataset
import config


def evaluate(model, test_loader, device, experiment_name, result_dir="./result"):
    """
    Evaluate model on test set and generate metrics and plots.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda or cpu)
        experiment_name: Name of experiment for result directory
        result_dir: Base directory to save results

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # Get predictions on test set
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (dog)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Create experiment-specific result directory
    exp_result_dir = os.path.join(result_dir, experiment_name)
    os.makedirs(exp_result_dir, exist_ok=True)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc_score),
    }

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc_score:.4f}")
    print("=" * 60 + "\n")

    # Save metrics to JSON
    metrics_path = os.path.join(exp_result_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, exp_result_dir)

    # Plot ROC curve
    plot_roc_curve(all_labels, all_probs, exp_result_dir)

    return metrics


def plot_confusion_matrix(cm, result_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Cat', 'Dog'])
    plt.yticks([0.5, 1.5], ['Cat', 'Dog'])
    plt.tight_layout()

    cm_path = os.path.join(result_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


def plot_roc_curve(labels, probs, result_dir):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(result_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {roc_path}")


def load_test_data(test_dir, image_size):
    """Load test dataset."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])

    test_data = DogCatDataset(dir=test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return test_loader


def load_checkpoint(checkpoint_path, model, device):
    """Load best model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model


def main():
    """Main evaluation pipeline."""
    import argparse
    from model import initialize_model

    parser = argparse.ArgumentParser(description="Evaluate Dogs vs. Cats model")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "vitb16"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default="./ckpts/dogs-vs-cats-resnet/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--test-dir", type=str, default=config.TEST_PATH,
                        help="Path to test dataset")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE,
                        help="Image size for preprocessing")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        choices=["cuda", "cpu"])

    args = parser.parse_args()

    # Load model
    print(f"\nLoading {args.model} model...")
    model, _ = initialize_model(args.model, config.NUM_CLASSES, use_pretrained=False)
    model = model.to(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, model, args.device)

    # Load test data
    print(f"Loading test data from {args.test_dir}...")
    test_loader = load_test_data(args.test_dir, args.image_size)
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Evaluate
    experiment_name = f"dogs-vs-cats-{args.model}"
    evaluate(model, test_loader, args.device, experiment_name)


if __name__ == "__main__":
    main()
