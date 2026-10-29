"""CropCare package initialization."""

try:
    from src.cropcare.climate_engine import get_environment
except (ImportError, AttributeError):
    get_environment = None

from src.cropcare.image_engine import predict_image

__all__ = ["predict_image"]
