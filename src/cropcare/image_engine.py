"""Leaf Image Intelligence Module for CropCare.

This module provides plant leaf disease screening from uploaded photographs using
a MobileNetV3-Small transfer-learning model trained on the PlantVillage dataset.
It provides prediction of crop species, disease/condition status, calibrated confidence,
class probability distributions, crop-constrained filtering, image quality screening,
and explicit uncertainty indicators.

Scientific Disclaimers & Ethics:
1. PlantVillage is a laboratory/benchmark dataset; strong performance on it does not guarantee
   field-level agricultural diagnosis under open-farm microclimates.
2. Softmax confidence scores indicate model certainty relative to its training distribution,
   not an absolute, confirmed biological diagnosis.
3. High visual similarity between early-stage diseases requires cautious agronomic interpretation.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# Default model artifacts
DEFAULT_MODEL_PATH = "models/plantvillage_mobilenetv3.pth"
DEFAULT_CLASS_MAPPING_PATH = "models/class_mapping.json"
DEFAULT_CONFIDENCE_THRESHOLD = 0.50  # Calibrated threshold for uncertainty flagging
MIN_IMAGE_DIMENSION = 32
BLANK_IMAGE_STD_THRESHOLD = 5.0

# Singleton cache for loaded model and class mapping
_CACHED_MODEL: Optional[nn.Module] = None
_CACHED_METADATA: Optional[Dict[str, Any]] = None
_CACHED_DEVICE: Optional[torch.device] = None


def _get_empty_prediction_schema(
    model_name: str = "MobileNetV3-Small-PlantVillage",
    warning_msg: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an empty, fully initialized output dictionary complying with the contract."""
    warnings = [warning_msg] if warning_msg else []
    return {
        "predicted_class": None,
        "crop": None,
        "disease": None,
        "confidence": 0.0,
        "probabilities": {},
        "top_candidates": [],
        "crop_constrained": False,
        "uncertain": True,
        "model_name": model_name,
        "warnings": warnings,
    }


def _load_model_cached(
    model_path: str = DEFAULT_MODEL_PATH,
    mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
) -> Tuple[Optional[nn.Module], Optional[Dict[str, Any]], Optional[str]]:
    """Load model and mapping into memory, caching for subsequent inference calls."""
    global _CACHED_MODEL, _CACHED_METADATA, _CACHED_DEVICE

    if _CACHED_MODEL is not None and _CACHED_METADATA is not None:
        return _CACHED_MODEL, _CACHED_METADATA, None

    if not os.path.exists(model_path):
        return None, None, f"Model checkpoint not found at: {model_path}. Please train or download model weights."

    if not os.path.exists(mapping_path):
        return None, None, f"Class mapping configuration not found at: {mapping_path}."

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception as e:
        return None, None, f"Failed to parse class mapping JSON: {e}"

    try:
        from src.cropcare.training.train_model import build_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        num_classes = mapping.get("num_classes", 38)
        model = build_model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        _CACHED_MODEL = model
        _CACHED_METADATA = {
            "mapping": mapping,
            "checkpoint": checkpoint,
            "image_size": checkpoint.get("image_size", 224),
            "model_name": checkpoint.get("model_name", "MobileNetV3-Small-PlantVillage"),
        }
        _CACHED_DEVICE = device

        return _CACHED_MODEL, _CACHED_METADATA, None
    except Exception as e:
        return None, None, f"Failed to load PyTorch model from {model_path}: {e}"


def get_model(
    model_path: str = DEFAULT_MODEL_PATH,
    mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
) -> Tuple[Optional[nn.Module], Optional[Dict[str, Any]]]:
    """Retrieve cached model and metadata mapping."""
    model, meta, _ = _load_model_cached(model_path=model_path, mapping_path=mapping_path)
    return model, meta


def check_image_quality(image: Image.Image) -> List[str]:
    """Perform input image sanity checks (dimensions, blankness, contrast)."""
    warnings: List[str] = []

    width, height = image.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        warnings.append(
            f"Image dimensions ({width}x{height}) are extremely small. Disease symptoms may lack recognizable resolution."
        )

    # Check for blank / solid-color image
    try:
        arr = np.array(image.convert("L"))
        std_val = float(np.std(arr))
        if std_val < BLANK_IMAGE_STD_THRESHOLD:
            warnings.append(
                f"Image exhibits near-zero visual contrast (std={std_val:.1f}). Image may be solid black, white, or blank."
            )
    except Exception:
        pass

    return warnings


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Apply standard ImageNet transformation pipeline for inference."""
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)


def predict_image(
    image: Union[Image.Image, str, Path, bytes, np.ndarray],
    model_path: str = DEFAULT_MODEL_PATH,
    mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    crop_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Screen a leaf photograph and predict plant health status.

    Parameters:
        image: Leaf image represented as PIL Image, file path string/Path, raw bytes, or numpy array.
        model_path: Path to model checkpoint .pth file.
        mapping_path: Path to class_mapping.json.
        confidence_threshold: Threshold below which uncertainty is flagged (default: 0.50).
        crop_filter: Optional crop name filter (e.g. 'Tomato', 'Potato', 'Apple') to condition
                     inference and prevent cross-species false positives.

    Returns:
        Structured prediction dictionary complying with the contract.
    """
    result = _get_empty_prediction_schema()

    # Step 1: Decode image input
    pil_img: Optional[Image.Image] = None
    try:
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            str_path = str(image)
            if not os.path.exists(str_path):
                result["warnings"].append(f"Image file does not exist at: {str_path}")
                return result
            pil_img = Image.open(str_path).convert("RGB")
        elif isinstance(image, bytes):
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            result["warnings"].append(f"Unsupported image input type: {type(image).__name__}")
            return result
    except Exception as e:
        result["warnings"].append(f"Failed to read or decode image: {e}")
        return result

    if pil_img is None:
        result["warnings"].append("Decoded image is None.")
        return result

    # Step 2: Quality Checks
    quality_warnings = check_image_quality(pil_img)
    result["warnings"].extend(quality_warnings)
    if any("near-zero visual contrast" in w for w in quality_warnings):
        result["uncertain"] = True

    # Step 3: Load Model & Mapping
    model, metadata, err = _load_model_cached(model_path=model_path, mapping_path=mapping_path)
    if err or model is None or metadata is None:
        result["warnings"].append(f"Inference unavailable: {err}")
        return result

    result["model_name"] = metadata["model_name"]
    mapping = metadata["mapping"]
    classes = mapping["classes"]
    class_details = mapping.get("class_details", {})
    image_size = metadata.get("image_size", 224)
    device = _CACHED_DEVICE or torch.device("cpu")

    # Step 4: Preprocessing & Forward Pass
    try:
        tensor = preprocess_image(pil_img, image_size=image_size).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Step 5: Probability and Confidence Validation
        prob_dict: Dict[str, float] = {}
        for idx, cls_name in enumerate(classes):
            p = float(probs[idx])
            prob_dict[cls_name] = round(p, 4) if math.isfinite(p) else 0.0

        # Check for crop constraint filter
        active_crop_filter = crop_filter.strip() if (crop_filter and str(crop_filter).strip()) else None
        if active_crop_filter and active_crop_filter.lower() in ("auto-detect", "auto-detect (all crops)", "all", "none"):
            active_crop_filter = None

        if active_crop_filter:
            # Match classes belonging to requested crop
            c_lower = active_crop_filter.lower()
            matching_indices = [
                i for i, c in enumerate(classes)
                if c_lower in c.lower() or c_lower in class_details.get(c, {}).get("crop", "").lower()
            ]

            if matching_indices:
                subset_sum = sum(float(probs[i]) for i in matching_indices)
                top_idx = max(matching_indices, key=lambda i: probs[i])
                top_class = classes[top_idx]
                top_confidence = (float(probs[top_idx]) / subset_sum) if subset_sum > 0 else float(probs[top_idx])

                result["crop_constrained"] = True

                # Top candidates within filtered crop
                sorted_crop_indices = sorted(matching_indices, key=lambda i: probs[i], reverse=True)[:3]
                top_candidates = []
                for i in sorted_crop_indices:
                    cls_k = classes[i]
                    c_prob = (float(probs[i]) / subset_sum) if subset_sum > 0 else float(probs[i])
                    d_name = class_details.get(cls_k, {}).get("disease") or cls_k.split("___")[-1].replace("_", " ")
                    top_candidates.append({
                        "class": cls_k,
                        "crop": class_details.get(cls_k, {}).get("crop") or cls_k.split("___")[0].replace("_", " "),
                        "disease": d_name,
                        "confidence": round(c_prob, 4),
                    })
                result["top_candidates"] = top_candidates
            else:
                # Fallback to unconstrained
                top_idx = int(np.argmax(probs))
                top_class = classes[top_idx]
                top_confidence = float(probs[top_idx])
                sorted_indices = np.argsort(probs)[::-1][:3]
                result["top_candidates"] = [
                    {
                        "class": classes[i],
                        "crop": class_details.get(classes[i], {}).get("crop") or classes[i].split("___")[0].replace("_", " "),
                        "disease": class_details.get(classes[i], {}).get("disease") or classes[i].split("___")[-1].replace("_", " "),
                        "confidence": round(float(probs[i]), 4),
                    }
                    for i in sorted_indices
                ]
        else:
            top_idx = int(np.argmax(probs))
            top_class = classes[top_idx]
            top_confidence = float(probs[top_idx])
            sorted_indices = np.argsort(probs)[::-1][:3]
            result["top_candidates"] = [
                {
                    "class": classes[i],
                    "crop": class_details.get(classes[i], {}).get("crop") or classes[i].split("___")[0].replace("_", " "),
                    "disease": class_details.get(classes[i], {}).get("disease") or classes[i].split("___")[-1].replace("_", " "),
                    "confidence": round(float(probs[i]), 4),
                }
                for i in sorted_indices
            ]

        if not math.isfinite(top_confidence):
            result["warnings"].append("Non-finite probability encountered during inference.")
            return result

        result["predicted_class"] = top_class
        result["confidence"] = round(top_confidence, 4)
        result["probabilities"] = prob_dict

        # Extract structured crop and disease labels
        details = class_details.get(top_class, {})
        result["crop"] = details.get("crop") or top_class.split("___")[0].replace("_", " ")
        result["disease"] = details.get("disease") or top_class.split("___")[-1].replace("_", " ")

        # Step 6: Calibrated Uncertainty Flagging
        if top_confidence < confidence_threshold:
            result["uncertain"] = True
            result["warnings"].append(
                f"Low prediction confidence ({top_confidence*100:.1f}% < {confidence_threshold*100:.0f}%). "
                f"Visual symptoms may be ambiguous, early-stage, or outside the trained distribution. "
                f"Field verification is advised."
            )
        else:
            result["uncertain"] = False

    except Exception as e:
        logger.exception("Inference error in predict_image")
        result["warnings"].append(f"Inference execution failed: {e}")
        result["uncertain"] = True

    return result
