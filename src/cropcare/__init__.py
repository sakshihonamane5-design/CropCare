"""CropCare package initialization."""

from src.cropcare.agronomy_engine import (
    calculate_foliar_damage_index,
    classify_disease_severity,
    get_nurturing_recommendations,
)
from src.cropcare.evaluation_engine import (
    get_benchmark_summary,
    get_crop_wise_benchmarks,
    get_stress_test_ablation,
    get_top_confusion_pairs,
)
from src.cropcare.multilingual import (
    SUPPORTED_LANGUAGES,
    generate_audio_speech_html,
    t,
    translate_advisory_text,
)
from src.cropcare.image_engine import (
    CROP_METADATA,
    check_image_quality,
    extract_leaf_embedding,
    get_registered_field_leaves,
    get_supported_crops,
    predict_image,
    preprocess_image,
    query_field_memory,
    register_field_leaf,
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
    "extract_leaf_embedding",
    "register_field_leaf",
    "query_field_memory",
    "get_registered_field_leaves",
    "CROP_METADATA",
    "get_environment",
    "calculate_foliar_damage_index",
    "classify_disease_severity",
    "get_nurturing_recommendations",
    "SUPPORTED_LANGUAGES",
    "t",
    "generate_audio_speech_html",
    "translate_advisory_text",
    "get_benchmark_summary",
    "get_crop_wise_benchmarks",
    "get_top_confusion_pairs",
    "get_stress_test_ablation",
]
