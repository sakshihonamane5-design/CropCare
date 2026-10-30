"""Held-out test set evaluation pipeline for PlantVillage classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cropcare.training.data_utils import (
    DEFAULT_IMAGE_SIZE,
    PlantVillageDataset,
    create_balanced_subset,
    get_eval_transforms,
    load_class_mapping,
    parse_split_file,
)
from src.cropcare.training.train_model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_for_eval(
    checkpoint_path: str = "models/plantvillage_mobilenetv3.pth",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model architecture and weights from checkpoint file.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        device: Device to place model on.

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", 38)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def run_evaluation(
    data_root: str = "data/raw",
    split_dir: str = "data/plantvillage_metadata",
    checkpoint_path: str = "models/plantvillage_mobilenetv3.pth",
    output_metrics_json: str = "reports/model_metrics.json",
    max_test_per_class: int = -1,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate model on the held-out official PlantVillage test split.

    Args:
        data_root: Directory containing raw extracted images.
        split_dir: Directory containing splits.
        checkpoint_path: Checkpoint to evaluate.
        output_metrics_json: Destination for JSON metrics report.
        max_test_per_class: If > 0, evaluates on a stratified subset of test set.
        batch_size: DataLoader batch size.

    Returns:
        Structured evaluation metrics dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint {checkpoint_path} on {device}...")
    model, checkpoint = load_model_for_eval(checkpoint_path, device=device)

    class_mapping = load_class_mapping("models/class_mapping.json")
    class_to_idx = class_mapping["class_to_idx"]
    classes = class_mapping["classes"]
    num_classes = class_mapping["num_classes"]

    test_split_path = os.path.join(split_dir, "splits", "color_test.txt")
    leaf_map_path = os.path.join(split_dir, "leaf_grouping", "leaf-map.json")

    logger.info(f"Parsing official held-out test split from {test_split_path}...")
    all_test_records = parse_split_file(test_split_path, leaf_map_path=leaf_map_path, class_to_idx=class_to_idx)

    if max_test_per_class > 0:
        logger.info(f"Subsetting test set to max {max_test_per_class} per class...")
        test_records = create_balanced_subset(all_test_records, max_per_class=max_test_per_class, random_seed=42)
    else:
        test_records = all_test_records

    logger.info(f"Evaluating {len(test_records)} test examples across {num_classes} classes...")

    test_dataset = PlantVillageDataset(test_records, data_root=data_root, transform=get_eval_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true: List[int] = []
    y_pred: List[int] = []
    latencies_ms: List[float] = []

    start_eval_time = time.time()

    for images, targets in test_loader:
        images = images.to(device)
        targets_list = targets.tolist()

        t0 = time.time()
        outputs = model(images)
        t1 = time.time()

        batch_time_ms = ((t1 - t0) * 1000.0) / images.size(0)
        latencies_ms.extend([batch_time_ms] * images.size(0))

        preds = outputs.argmax(dim=1).cpu().tolist()
        y_true.extend(targets_list)
        y_pred.extend(preds)

    total_eval_duration = time.time() - start_eval_time
    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    overall_accuracy = float(np.mean(y_true_np == y_pred_np))

    # Macro & Weighted metrics
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="weighted", zero_division=0
    )

    # Per-class metrics
    per_class_prec, per_class_rec, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=list(range(num_classes)), average=None, zero_division=0
    )

    per_class_metrics: Dict[str, Any] = {}
    for idx, cls_name in enumerate(classes):
        per_class_metrics[cls_name] = {
            "class_index": idx,
            "precision": round(float(per_class_prec[idx]), 4),
            "recall": round(float(per_class_rec[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_supp[idx]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes))).tolist()

    # Top class confusions (where true != pred)
    confusions: List[Dict[str, Any]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                confusions.append(
                    {
                        "true_class": classes[i],
                        "predicted_class": classes[j],
                        "count": cm[i][j],
                    }
                )
    confusions.sort(key=lambda x: x["count"], reverse=True)

    metrics_report = {
        "model_name": checkpoint.get("model_name", "MobileNetV3-Small-PlantVillage"),
        "checkpoint_path": checkpoint_path,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_device": str(device),
        "total_test_samples_evaluated": len(y_true),
        "official_test_split_total": len(all_test_records),
        "subset_evaluated": max_test_per_class > 0,
        "metrics": {
            "test_accuracy": round(overall_accuracy, 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_prec), 4),
            "weighted_recall": round(float(weighted_rec), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "mean_inference_latency_ms": round(avg_latency_ms, 2),
            "total_eval_duration_sec": round(total_eval_duration, 2),
        },
        "per_class_metrics": per_class_metrics,
        "top_class_confusions": confusions[:10],
        "confusion_matrix_shape": [num_classes, num_classes],
        "training_history": checkpoint.get("training_history", []),
    }

    os.makedirs(os.path.dirname(output_metrics_json) or ".", exist_ok=True)
    with open(output_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    logger.info(
        f"Evaluation finished: Accuracy = {overall_accuracy*100:.2f}%, "
        f"Macro F1 = {macro_f1:.4f}, Weighted F1 = {weighted_f1:.4f}, "
        f"Latency = {avg_latency_ms:.2f} ms/image"
    )
    logger.info(f"Metrics saved to {output_metrics_json}")

    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PlantVillage model")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--checkpoint", type=str, default="models/plantvillage_mobilenetv3.pth", help="Checkpoint path")
    parser.add_argument("--max_test_per_class", type=int, default=-1, help="Max test samples per class")
    args = parser.parse_args()

    run_evaluation(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        max_test_per_class=args.max_test_per_class,
    )
