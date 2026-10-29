"""CropCare package initialization."""

from src.cropcare.climate_engine import get_environment
from src.cropcare.image_engine import predict_image
from src.cropcare.explainability import generate_gradcam
from src.cropcare.alignment_engine import evaluate_environmental_alignment
from src.cropcare.advisory_engine import generate_actionable_advisory
from src.cropcare.report_engine import generate_field_report_html
from src.cropcare.vernacular import translate_diagnosis, generate_audio_script

__all__ = [
    "get_environment",
    "predict_image",
    "generate_gradcam",
    "evaluate_environmental_alignment",
    "generate_actionable_advisory",
    "generate_field_report_html",
    "translate_diagnosis",
    "generate_audio_script",
]


