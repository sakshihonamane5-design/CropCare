"""CropCare package initialization."""

from src.cropcare.climate_engine import get_environment
from src.cropcare.image_engine import predict_image

__all__ = ["get_environment", "predict_image"]

