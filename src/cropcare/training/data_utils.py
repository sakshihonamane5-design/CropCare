"""Dataset loading, parsing, and transformation utilities for PlantVillage."""

from __future__ import annotations

import collections
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters (required by torchvision pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224


def load_class_mapping(mapping_path: str = "models/class_mapping.json") -> Dict[str, Any]:
    """Load the deterministic class mapping configuration.

    Args:
        mapping_path: Path to class_mapping.json.

    Returns:
        Dictionary containing class list, class_to_idx, and class_details.
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping file not found at: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_split_file(
    split_file_path: str,
    leaf_map_path: Optional[str] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Parse an official PlantVillage split text file into record dictionaries.

    Format expected per line: `raw/color/<class_name>/<filename>`

    Args:
        split_file_path: Path to split text file (e.g. color_train.txt).
        leaf_map_path: Optional path to leaf-map.json for grouping metadata.
        class_to_idx: Optional mapping from class name to integer label index.

    Returns:
        List of structured record dictionaries.
    """
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    leaf_map = {}
    if leaf_map_path and os.path.exists(leaf_map_path):
        with open(leaf_map_path, "r", encoding="utf-8") as f:
            leaf_map = json.load(f)

    records = []
    with open(split_file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split("/")
        if len(parts) < 4:
            continue
        cls_name = parts[2]
        filename = parts[3]
        crop, disease = cls_name.split("___") if "___" in cls_name else (cls_name, "unknown")
        leaf_id = leaf_map.get(filename)

        record = {
            "relative_path": line,
            "class_name": cls_name,
            "crop": crop.replace("_", " "),
            "disease": disease.replace("_", " "),
            "filename": filename,
            "leaf_id": leaf_id,
            "label_idx": class_to_idx[cls_name] if class_to_idx and cls_name in class_to_idx else None,
        }
        records.append(record)

    return records


def split_train_val_by_leaf(
    train_records: List[Dict[str, Any]],
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create a leaf-aware validation split from training records.

    Respects `leaf_id` grouping so leaves from the same plant/session are not split
    across training and validation sets, preventing data leakage.

    Args:
        train_records: Parsed training records.
        val_fraction: Target fraction for validation set (default: 0.15).
        random_seed: Seed for reproducible shuffling.

    Returns:
        Tuple of (train_partition, val_partition).
    """
    rng = random.Random(random_seed)

    # Group by class first to preserve class balance
    by_class: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in train_records:
        by_class[r["class_name"]].append(r)

    train_out: List[Dict[str, Any]] = []
    val_out: List[Dict[str, Any]] = []

    for cls_name, cls_records in sorted(by_class.items()):
        # Group records by leaf_id (or individual filename if leaf_id is missing)
        by_leaf: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in cls_records:
            leaf_key = r["leaf_id"] if r.get("leaf_id") else r["filename"]
            by_leaf[leaf_key].append(r)

        unique_leaves = list(by_leaf.keys())
        rng.shuffle(unique_leaves)

        target_val_count = max(1, int(len(cls_records) * val_fraction))
        current_val_count = 0
        cls_val: List[Dict[str, Any]] = []
        cls_train: List[Dict[str, Any]] = []

        for leaf in unique_leaves:
            group = by_leaf[leaf]
            if current_val_count < target_val_count and len(cls_val) < len(cls_train):
                cls_val.extend(group)
                current_val_count += len(group)
            else:
                cls_train.extend(group)

        train_out.extend(cls_train)
        val_out.extend(cls_val)

    return train_out, val_out


def create_balanced_subset(
    records: List[Dict[str, Any]],
    max_per_class: int = 60,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create a reproducible, class-stratified subset for compute-aware training.

    Args:
        records: Input records list.
        max_per_class: Maximum examples to retain per class.
        random_seed: Random seed for reproducibility.

    Returns:
        Stratified subset of records.
    """
    rng = random.Random(random_seed)
    by_class: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_class[r["class_name"]].append(r)

    subset = []
    for cls_name in sorted(by_class.keys()):
        cls_samples = list(by_class[cls_name])
        rng.shuffle(cls_samples)
        subset.extend(cls_samples[:max_per_class])

    rng.shuffle(subset)
    return subset


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """Get training image transformations with mild, realistic data augmentations."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """Get evaluation/test/inference image transformations (deterministic)."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class PlantVillageDataset(Dataset):
    """PyTorch Dataset for PlantVillage image records."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        data_root: str,
        transform: Optional[Any] = None,
    ):
        """Initialize dataset.

        Args:
            records: List of record dictionaries.
            data_root: Root directory where images are extracted.
            transform: PyTorch image transformation pipeline.
        """
        self.records = records
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        record = self.records[idx]
        rel_path = record["relative_path"]

        # Support both 'raw/color/...' and nested directories
        full_path = os.path.join(self.data_root, rel_path)
        if not os.path.exists(full_path):
            # Check alternative nested paths (e.g. data_root/<rel_path> or data_root/raw/...)
            alt_path = os.path.join(self.data_root, os.path.basename(os.path.dirname(rel_path)), os.path.basename(rel_path))
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                raise FileNotFoundError(f"Image not found at expected path: {full_path}")

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to decode image at {full_path}: {e}")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        label_idx = record.get("label_idx", -1)
        return image_tensor, label_idx
