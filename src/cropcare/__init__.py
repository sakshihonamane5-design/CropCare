"""CropCare package initialization."""

from src.cropcare.image_engine import (
    CROP_METADATA,
    check_image_quality,
    get_supported_crops,
    predict_image,
    preprocess_image,
)

try:
    from src.cropcare.climate_engine import get_environment
except (ImportError, AttributeError):
    get_environment = None

__all__ = [
    "predict_image",
    "get_supported_crops",
    "check_image_quality",
    "preprocess_image",
    "CROP_METADATA",
    "get_environment",
]

