"""Reproducible training pipeline for PlantVillage MobileNetV3 classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from src.cropcare.training.data_utils import (
    DEFAULT_IMAGE_SIZE,
    PlantVillageDataset,
    create_balanced_subset,
    get_eval_transforms,
    get_train_transforms,
    load_class_mapping,
    parse_split_file,
    split_train_val_by_leaf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model(
    num_classes: int = 38,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Construct MobileNetV3-Small model with custom classification head.

    Args:
        num_classes: Number of target classes.
        pretrained: Whether to initialize with ImageNet-1K pretrained weights.
        dropout: Classifier dropout probability.

    Returns:
        Configured PyTorch nn.Module.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # In MobileNetV3-Small, classifier is:
    # (0): Linear(in_features=576, out_features=1024, bias=True)
    # (1): Hardswish()
    # (2): Dropout(p=0.2, inplace=True)
    # (3): Linear(in_features=1024, out_features=1000, bias=True)
    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout)
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation loader.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def run_training(
    data_root: str = "data/raw",
    split_dir: str = "data/plantvillage_metadata",
    output_model_path: str = "models/plantvillage_mobilenetv3.pth",
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_train_per_class: int = 60,
    max_val_per_class: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete reproducible training pipeline.

    Args:
        data_root: Directory containing extracted PlantVillage images.
        split_dir: Directory containing split files and metadata.
        output_model_path: Path where best checkpoint will be saved.
        epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        lr: Initial learning rate.
        max_train_per_class: Subsetting threshold for compute-aware training (default: 60/class).
        max_val_per_class: Validation subset threshold (default: 15/class).
        seed: Random seed for reproducibility.

    Returns:
        Summary dictionary with training metrics and checkpoint info.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Load class mapping
    class_mapping = load_class_mapping("models/class_mapping.json")
    class_to_idx = class_mapping["class_to_idx"]
    num_classes = class_mapping["num_classes"]

    # 2. Parse official training split
    train_split_path = os.path.join(split_dir, "splits", "color_train.txt")
    leaf_map_path = os.path.join(split_dir, "leaf_grouping", "leaf-map.json")

    logger.info("Parsing official training split records...")
    raw_train_records = parse_split_file(train_split_path, leaf_map_path=leaf_map_path, class_to_idx=class_to_idx)

    # 3. Partition train and validation respecting leaf grouping
    logger.info("Partitioning into leaf-aware train and validation sets...")
    full_train_records, full_val_records = split_train_val_by_leaf(raw_train_records, val_fraction=0.15, random_seed=seed)

    # 4. Apply stratified compute-aware subsetting if specified
    if max_train_per_class > 0:
        logger.info(f"Applying stratified subsetting: max {max_train_per_class} per class for train...")
        train_records = create_balanced_subset(full_train_records, max_per_class=max_train_per_class, random_seed=seed)
    else:
        train_records = full_train_records

    if max_val_per_class > 0:
        val_records = create_balanced_subset(full_val_records, max_per_class=max_val_per_class, random_seed=seed)
    else:
        val_records = full_val_records

    logger.info(f"Final training records: {len(train_records)}, validation records: {len(val_records)}")

    # 5. Build Datasets and DataLoaders
    train_dataset = PlantVillageDataset(train_records, data_root=data_root, transform=get_train_transforms())
    val_dataset = PlantVillageDataset(val_records, data_root=data_root, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 6. Build model, loss, optimizer, scheduler
    logger.info(f"Initializing MobileNetV3-Small backbone with {num_classes} classes...")
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: List[Dict[str, Any]] = []
    best_val_acc = 0.0
    best_epoch = -1

    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)

    # 7. Training loop
    start_time = time.time()
    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch}/{epochs} ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "lr": round(current_lr, 6),
                "duration_seconds": round(epoch_duration, 2),
            }
        )

        # Checkpoint saving: best validation accuracy
        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": "MobileNetV3-Small-PlantVillage",
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "image_size": DEFAULT_IMAGE_SIZE,
                "best_epoch": best_epoch,
                "best_val_accuracy": round(best_val_acc, 4),
                "training_history": history,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            torch.save(checkpoint, output_model_path)
            logger.info(f" Saved improved checkpoint to {output_model_path} (Val Acc: {val_acc*100:.2f}%)")

    total_duration = time.time() - start_time
    logger.info(f"Training completed in {total_duration:.1f}s. Best Epoch: {best_epoch} with Val Acc: {best_val_acc*100:.2f}%")

    return {
        "status": "success",
        "output_model_path": output_model_path,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4),
        "total_duration_seconds": round(total_duration, 2),
        "train_sample_count": len(train_records),
        "val_sample_count": len(val_records),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 on PlantVillage")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_train_per_class", type=int, default=60, help="Max train examples per class")
    parser.add_argument("--max_val_per_class", type=int, default=15, help="Max val examples per class")
    args = parser.parse_args()

    run_training(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
    )
