"""Leaf Image Intelligence Module for CropCare.

This module provides plant leaf disease screening from uploaded photographs using
a MobileNetV3-Small transfer-learning model trained on the PlantVillage dataset.
It provides prediction of crop species, disease/condition status, calibrated confidence,
class probability distributions, image quality screening, crop-specific filtering,
active learning field memory adaptation, and explicit uncertainty indicators.

Scientific Disclaimers & Ethics:
1. PlantVillage is a laboratory/benchmark dataset; strong performance on it does not guarantee
   field-level agricultural diagnosis under open-farm microclimates.
2. Softmax confidence scores indicate model certainty relative to its training distribution,
   not an absolute, confirmed biological diagnosis.
3. High visual similarity between early-stage diseases requires cautious agronomic interpretation.
"""

from __future__ import annotations

import collections
import datetime
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

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

# Active learning field memory bank paths
DEFAULT_FIELD_MEMORY_PATH = "data/active_learning_memory.json"
DEFAULT_FIELD_UPLOADS_DIR = "data/field_uploads"

# Singleton cache for loaded model and class mapping
_CACHED_MODEL: Optional[nn.Module] = None
_CACHED_METADATA: Optional[Dict[str, Any]] = None
_CACHED_DEVICE: Optional[torch.device] = None
_CACHED_PATH_PAIR: Optional[Tuple[str, str]] = None

# Comprehensive metadata and emoji mappings for the 14 PlantVillage crops
CROP_METADATA: Dict[str, Dict[str, Any]] = {
    "Apple": {"icon": "🍎", "name": "Apple", "conditions_count": 4, "desc": "Screening for Apple scab, Black rot, Cedar rust"},
    "Blueberry": {"icon": "🫐", "name": "Blueberry", "conditions_count": 1, "desc": "Screening for healthy foliage"},
    "Cherry_(including_sour)": {"icon": "🍒", "name": "Cherry (including sour)", "conditions_count": 2, "desc": "Screening for Powdery mildew and healthy foliage"},
    "Corn_(maize)": {"icon": "🌽", "name": "Corn (Maize)", "conditions_count": 4, "desc": "Screening for Gray leaf spot, Common rust, Northern blight"},
    "Grape": {"icon": "🍇", "name": "Grape", "conditions_count": 4, "desc": "Screening for Black rot, Esca (Measles), Leaf blight"},
    "Orange": {"icon": "🍊", "name": "Orange", "conditions_count": 1, "desc": "Screening for Huanglongbing (Citrus greening)"},
    "Peach": {"icon": "🍑", "name": "Peach", "conditions_count": 2, "desc": "Screening for Bacterial spot and healthy leaves"},
    "Pepper,_bell": {"icon": "🫑", "name": "Bell Pepper", "conditions_count": 2, "desc": "Screening for Bacterial spot and healthy foliage"},
    "Potato": {"icon": "🥔", "name": "Potato", "conditions_count": 3, "desc": "Screening for Early blight, Late blight, and healthy leaves"},
    "Raspberry": {"icon": "🍇", "name": "Raspberry", "conditions_count": 1, "desc": "Screening for healthy foliage"},
    "Soybean": {"icon": "🫘", "name": "Soybean", "conditions_count": 1, "desc": "Screening for healthy foliage"},
    "Squash": {"icon": "🎃", "name": "Squash", "conditions_count": 1, "desc": "Screening for Powdery mildew"},
    "Strawberry": {"icon": "🍓", "name": "Strawberry", "conditions_count": 2, "desc": "Screening for Leaf scorch and healthy foliage"},
    "Tomato": {"icon": "🍅", "name": "Tomato", "conditions_count": 10, "desc": "Screening for 10 distinct pathologies & healthy status"},
}


def get_supported_crops(mapping_path: str = DEFAULT_CLASS_MAPPING_PATH) -> Dict[str, Dict[str, Any]]:
    """Return catalog of all supported crops with icons, condition lists, and metadata."""
    if not os.path.exists(mapping_path):
        return {}
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception:
        return {}

    classes = mapping.get("classes", [])
    class_details = mapping.get("class_details", {})

    catalog: Dict[str, Dict[str, Any]] = {}
    for cls_name in classes:
        details = class_details.get(cls_name, {})
        raw_crop = cls_name.split("___")[0]
        crop_display = details.get("crop") or raw_crop.replace("_", " ")
        disease_name = details.get("disease") or cls_name.split("___")[-1].replace("_", " ")

        if raw_crop not in catalog:
            meta = CROP_METADATA.get(raw_crop, {})
            catalog[raw_crop] = {
                "raw_crop": raw_crop,
                "display_name": meta.get("name", crop_display),
                "icon": meta.get("icon", "🌱"),
                "desc": meta.get("desc", ""),
                "classes": [],
                "diseases": [],
            }
        catalog[raw_crop]["classes"].append(cls_name)
        catalog[raw_crop]["diseases"].append(disease_name)

    return catalog


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
    global _CACHED_MODEL, _CACHED_METADATA, _CACHED_DEVICE, _CACHED_PATH_PAIR

    if (
        _CACHED_MODEL is not None
        and _CACHED_METADATA is not None
        and _CACHED_PATH_PAIR == (model_path, mapping_path)
    ):
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
        _CACHED_PATH_PAIR = (model_path, mapping_path)

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


def preprocess_image_multi_view(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Generate multi-view augmentations for test-time augmentation (TTA)."""
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    v1 = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        norm,
    ])(image)

    v2 = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        norm,
    ])(image)

    w, h = image.size
    min_dim = min(w, h)
    scale = 256.0 / max(min_dim, 1)
    new_w = max(int(round(w * scale)), image_size)
    new_h = max(int(round(h * scale)), image_size)
    v3 = T.Compose([
        T.Resize((new_h, new_w)),
        T.CenterCrop((image_size, image_size)),
        T.ToTensor(),
        norm,
    ])(image)

    return torch.stack([v1, v2, v3], dim=0)


# ==============================================================================
# CONTINUAL LEARNING & ACTIVE FIELD MEMORY BANK
# ==============================================================================
def extract_leaf_embedding(
    image: Image.Image,
    model_path: str = DEFAULT_MODEL_PATH,
    mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
) -> Optional[List[float]]:
    """Extract a 1024-dimensional normalized feature embedding vector from MobileNetV3."""
    model, metadata, err = _load_model_cached(model_path=model_path, mapping_path=mapping_path)
    if err or model is None:
        return None

    try:
        device = _CACHED_DEVICE or torch.device("cpu")
        tensor = preprocess_image(image.convert("RGB"), image_size=224).to(device)
        with torch.no_grad():
            x = model.features(tensor)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            feat = model.classifier[0](x)
            feat = model.classifier[1](feat)
            norm_feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return norm_feat.squeeze(0).cpu().numpy().tolist()
    except Exception as e:
        logger.error(f"Failed to extract leaf embedding: {e}")
        return None


def register_field_leaf(
    image: Image.Image,
    crop: str,
    disease: str,
    notes: str = "",
    memory_path: str = DEFAULT_FIELD_MEMORY_PATH,
    uploads_dir: str = DEFAULT_FIELD_UPLOADS_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
) -> Dict[str, Any]:
    """Register a new leaf observation into the continual-learning memory bank."""
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(os.path.dirname(memory_path) or ".", exist_ok=True)

    embedding = extract_leaf_embedding(image, model_path=model_path, mapping_path=mapping_path)
    if embedding is None:
        return {"success": False, "error": "Neural embedding extraction failed."}

    sample_id = f"sample_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    filename = f"{sample_id}.png"
    filepath = os.path.join(uploads_dir, filename)

    try:
        image.convert("RGB").save(filepath, format="PNG")
    except Exception as e:
        return {"success": False, "error": f"Failed to save image: {e}"}

    record = {
        "id": sample_id,
        "crop": crop.strip(),
        "disease": disease.strip(),
        "notes": notes.strip(),
        "timestamp": datetime.datetime.now().isoformat(),
        "image_path": filepath,
        "embedding": embedding,
    }

    records = []
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    records.append(record)
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return {"success": True, "record": record, "total_samples": len(records)}


def query_field_memory(
    embedding: List[float],
    memory_path: str = DEFAULT_FIELD_MEMORY_PATH,
    similarity_threshold: float = 0.82,
) -> Optional[Dict[str, Any]]:
    """Query field memory bank using cosine similarity against registered embeddings."""
    if not os.path.exists(memory_path):
        return None

    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return None

    if not records:
        return None

    emb_arr = np.array(embedding, dtype=np.float32)
    best_match = None
    best_sim = -1.0

    for rec in records:
        rec_emb = rec.get("embedding")
        if not rec_emb:
            continue
        rec_arr = np.array(rec_emb, dtype=np.float32)
        sim = float(np.dot(emb_arr, rec_arr))
        if sim > best_sim:
            best_sim = sim
            best_match = {
                "id": rec.get("id"),
                "crop": rec.get("crop"),
                "disease": rec.get("disease"),
                "notes": rec.get("notes"),
                "similarity": round(sim, 4),
                "timestamp": rec.get("timestamp"),
                "image_path": rec.get("image_path"),
            }

    if best_match and best_sim >= similarity_threshold:
        return best_match

    return None


def get_registered_field_leaves(memory_path: str = DEFAULT_FIELD_MEMORY_PATH) -> List[Dict[str, Any]]:
    """Return all registered field observations from the active learning memory bank."""
    if not os.path.exists(memory_path):
        return []
    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def predict_image(
    image: Union[Image.Image, str, Path, bytes, np.ndarray],
    model_path: str = DEFAULT_MODEL_PATH,
    mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    crop_filter: Optional[str] = None,
    use_tta: bool = False,
    temperature: float = 0.65,
) -> Dict[str, Any]:
    """Screen a leaf photograph and predict plant health status.

    Output Contract:
    {
        "predicted_class": "Tomato___Early_blight",
        "crop": "Tomato",
        "disease": "Early blight",
        "confidence": 0.874,
        "probabilities": { ... },
        "uncertain": False,
        "model_name": "MobileNetV3-Small-PlantVillage",
        "warnings": []
    }
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
        if use_tta:
            batch = preprocess_image_multi_view(pil_img, image_size=image_size).to(device)
            with torch.no_grad():
                out = model(batch)
                logits = out.mean(dim=0, keepdim=True)
        else:
            tensor = preprocess_image(pil_img, image_size=image_size).to(device)
            with torch.no_grad():
                logits = model(tensor)

        # Check if a specific crop filter was requested
        selected_crop_key: Optional[str] = None
        if crop_filter and crop_filter not in ("All", "Auto-Detect", "All Crops (Auto-Detect)"):
            norm_filter = crop_filter.lower().replace("_", " ").replace(",", "").replace("-", " ").strip()
            for cls_name in classes:
                raw_c = cls_name.split("___")[0].lower().replace("_", " ").replace(",", "").replace("-", " ").strip()
                if norm_filter == raw_c or norm_filter in raw_c or raw_c in norm_filter:
                    selected_crop_key = cls_name.split("___")[0]
                    break

        temp_scale = max(float(temperature), 0.1)

        # ----------------------------------------------------------------------
        # Crop-filtered inference path
        # ----------------------------------------------------------------------
        if selected_crop_key is not None:
            crop_indices = [i for i, c in enumerate(classes) if c.startswith(f"{selected_crop_key}___")]
            if crop_indices:
                sub_logits = logits[0, crop_indices] / temp_scale
                sub_probs = torch.softmax(sub_logits, dim=0).cpu().numpy()

                prob_dict: Dict[str, float] = {}
                for i, cls_idx in enumerate(crop_indices):
                    cls_name = classes[cls_idx]
                    p = float(sub_probs[i])
                    prob_dict[cls_name] = round(p, 4) if math.isfinite(p) else 0.0

                top_sub_idx = int(np.argmax(sub_probs))
                top_class = classes[crop_indices[top_sub_idx]]
                top_confidence = float(sub_probs[top_sub_idx])

                result["predicted_class"] = top_class
                result["confidence"] = round(top_confidence, 4)
                result["probabilities"] = prob_dict
                result["crop_filter_applied"] = True
                result["selected_crop"] = selected_crop_key

                details = class_details.get(top_class, {})
                result["crop"] = details.get("crop") or selected_crop_key.replace("_", " ")
                result["disease"] = details.get("disease") or top_class.split("___")[-1].replace("_", " ")

                sorted_crop_indices = sorted(crop_indices, key=lambda i: probs[i], reverse=True)[:3]
                result["top_candidates"] = [
                    {
                        "class": classes[i],
                        "crop": class_details.get(classes[i], {}).get("crop") or selected_crop_key.replace("_", " "),
                        "disease": class_details.get(classes[i], {}).get("disease") or classes[i].split("___")[-1].replace("_", " "),
                        "confidence": round(float(probs[i]), 4),
                    }
                    for i in sorted_crop_indices
                ]

                if confidence_threshold != DEFAULT_CONFIDENCE_THRESHOLD:
                    effective_thresh = confidence_threshold
                else:
                    effective_thresh = min(DEFAULT_CONFIDENCE_THRESHOLD, max(0.35, (1.0 / len(crop_indices)) * 1.8))

                if top_confidence < effective_thresh:
                    result["uncertain"] = True
                    result["warnings"].append(
                        f"Low prediction confidence ({top_confidence*100:.1f}% < {effective_thresh*100:.0f}%). "
                        f"Visual symptoms are ambiguous within {result['crop']} conditions. Field verification advised."
                    )
                else:
                    result["uncertain"] = False

                # Query active field memory for agronomist confirmation
                try:
                    q_emb = extract_leaf_embedding(pil_img, model_path=model_path, mapping_path=mapping_path)
                    if q_emb is not None:
                        mem_match = query_field_memory(q_emb, similarity_threshold=0.82)
                        if mem_match:
                            result["field_memory_match"] = mem_match
                            if mem_match["similarity"] >= 0.88:
                                result["crop"] = mem_match["crop"]
                                result["disease"] = mem_match["disease"]
                                result["confidence"] = max(result["confidence"], mem_match["similarity"])
                                result["uncertain"] = False
                except Exception:
                    pass

                return result

        # ----------------------------------------------------------------------
        # Global multi-class inference path (Auto-detect)
        # ----------------------------------------------------------------------
        scaled_logits = logits / temp_scale
        probs = torch.softmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()

        prob_dict = {}
        crop_aggregates = collections.defaultdict(float)
        for idx, cls_name in enumerate(classes):
            p = float(probs[idx])
            prob_dict[cls_name] = round(p, 4) if math.isfinite(p) else 0.0
            raw_c = cls_name.split("___")[0]
            crop_aggregates[raw_c] += p

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

        if crop_aggregates:
            detected_crop_raw = max(crop_aggregates.items(), key=lambda x: x[1])[0]
            detected_crop_conf = float(crop_aggregates[detected_crop_raw])
            denom = max(detected_crop_conf, 1e-6)
            conditional_conf = top_confidence / denom
            result["detected_crop"] = detected_crop_raw
            result["detected_crop_confidence"] = round(detected_crop_conf, 4)
            result["conditional_confidence"] = round(conditional_conf, 4)
            result["crop_probabilities"] = {k: round(v, 4) for k, v in sorted(crop_aggregates.items(), key=lambda x: x[1], reverse=True)}

        details = class_details.get(top_class, {})
        result["crop"] = details.get("crop") or top_class.split("___")[0].replace("_", " ")
        result["disease"] = details.get("disease") or top_class.split("___")[-1].replace("_", " ")

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

        if confidence_threshold != DEFAULT_CONFIDENCE_THRESHOLD:
            effective_thresh = confidence_threshold
        else:
            effective_thresh = min(DEFAULT_CONFIDENCE_THRESHOLD, max(0.35, (1.0 / len(classes)) * 1.8))

        if top_confidence < effective_thresh:
            result["uncertain"] = True
            result["warnings"].append(
                f"Low prediction confidence ({top_confidence*100:.1f}% < {effective_thresh*100:.0f}%). "
                f"Selecting the specific Crop Type above can significantly sharpen diagnostic certainty."
            )
        else:
            result["uncertain"] = False

        # Query active field memory for agronomist confirmation
        try:
            q_emb = extract_leaf_embedding(pil_img, model_path=model_path, mapping_path=mapping_path)
            if q_emb is not None:
                mem_match = query_field_memory(q_emb, similarity_threshold=0.82)
                if mem_match:
                    result["field_memory_match"] = mem_match
                    if mem_match["similarity"] >= 0.88:
                        result["crop"] = mem_match["crop"]
                        result["disease"] = mem_match["disease"]
                        result["confidence"] = max(result["confidence"], mem_match["similarity"])
                        result["uncertain"] = False
        except Exception:
            pass

    except Exception as e:
        logger.exception("Inference error in predict_image")
        result["warnings"].append(f"Inference execution failed: {e}")
        result["uncertain"] = True

    return result
