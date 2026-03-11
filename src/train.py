"""
Training loop for Dogs vs. Cats image classification.
"""

import os
import json
import torch
import copy
from tqdm import tqdm
import wandb


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, wandb_enabled=True, checkpoint_dir="./ckpts", logs_dir="./logs", iteration_interval=1000):
    """
    Train the model.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on (cuda or cpu)
        wandb_enabled: Whether to log to Weights & Biases
        checkpoint_dir: Directory to save best model checkpoint
        logs_dir: Directory to save training metrics JSON
        iteration_interval: Print metrics every N iterations

    Returns:
        Trained model with best validation weights
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize metrics tracking
    history = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Progress bar for epochs
    pbar_epoch = tqdm(range(num_epochs), desc='Epoch', unit='epoch')

    for epoch in pbar_epoch:
        epoch_metrics = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            sample_count = 0

            # Progress bar for batches
            pbar_batch = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} [{phase.upper()}]',
                            leave=False, unit='batch')

            for inputs, labels in pbar_batch:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                total_samples += batch_size
                sample_count += batch_size

                # Print every N iterations (samples)
                if sample_count % iteration_interval == 0:
                    running_loss_avg = running_loss / total_samples
                    running_acc = running_corrects.double() / total_samples
                    phase_label = phase.capitalize()
                    tqdm.write(f'[Epoch {epoch + 1}/{num_epochs}] Iteration {sample_count} -> {phase_label} Loss: {running_loss_avg:.4f}, Accuracy: {running_acc:.4f}')

                # Update batch progress bar
                pbar_batch.set_postfix({
                    'loss': f'{running_loss / total_samples:.4f}',
                    'acc': f'{running_corrects.double() / total_samples:.4f}'
                })

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            epoch_metrics[f'{phase}_loss'] = float(epoch_loss)
            epoch_metrics[f'{phase}_acc'] = float(epoch_acc)

            # Track metrics
            if phase == 'train':
                history["epochs"].append(epoch)
                history["train_loss"].append(float(epoch_loss))
                history["train_accuracy"].append(float(epoch_acc))
            else:
                history["val_loss"].append(float(epoch_loss))
                history["val_accuracy"].append(float(epoch_acc))

            # Log metrics to wandb
            if wandb_enabled:
                wandb.log({
                    f"{phase}/loss": float(epoch_loss),
                    f"{phase}/accuracy": float(epoch_acc),
                    "epoch": epoch
                })

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Update epoch progress bar with metrics
        pbar_epoch.set_postfix({
            'train_loss': f"{epoch_metrics['train_loss']:.4f}",
            'train_acc': f"{epoch_metrics['train_acc']:.4f}",
            'val_loss': f"{epoch_metrics['val_loss']:.4f}",
            'val_acc': f"{epoch_metrics['val_acc']:.4f}"
        })

    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Save training metrics to JSON
    os.makedirs(logs_dir, exist_ok=True)
    metrics_path = os.path.join(logs_dir, "history.json")
    with open(metrics_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Log final best accuracy to wandb
    if wandb_enabled:
        wandb.log({"best_val_accuracy": float(best_acc)})

    return model
