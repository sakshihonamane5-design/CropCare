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
    get_model,
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

try:
    from src.cropcare.economic_engine import estimate_economic_impact
except (ImportError, AttributeError):
    estimate_economic_impact = None

try:
    from src.cropcare.advisory_engine import (
        get_weather_smart_ipm,
        generate_recovery_roadmap,
        build_whatsapp_share_url,
    )
except (ImportError, AttributeError):
    get_weather_smart_ipm = None
    generate_recovery_roadmap = None
    build_whatsapp_share_url = None

try:
    from src.cropcare.location_engine import (
        search_location_online,
        search_location_offline,
        get_states,
        get_districts_for_state,
        TOP_PRESET_LOCATIONS,
        AGRI_DISTRICTS,
    )
except (ImportError, AttributeError):
    search_location_online = None
    search_location_offline = None
    get_states = None
    get_districts_for_state = None
    TOP_PRESET_LOCATIONS = {}
    AGRI_DISTRICTS = {}

try:
    from src.cropcare.alignment_engine import compute_climate_alignment
except (ImportError, AttributeError):
    compute_climate_alignment = None

try:
    from src.cropcare.explainability import generate_gradcam
except (ImportError, AttributeError):
    generate_gradcam = None

try:
    from src.cropcare.report_engine import generate_field_report_html
except (ImportError, AttributeError):
    generate_field_report_html = None

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
    "estimate_economic_impact",
    "get_weather_smart_ipm",
    "generate_recovery_roadmap",
    "build_whatsapp_share_url",
    "search_location_online",
    "search_location_offline",
    "get_states",
    "get_districts_for_state",
    "TOP_PRESET_LOCATIONS",
    "AGRI_DISTRICTS",
    "compute_climate_alignment",
    "generate_gradcam",
    "generate_field_report_html",
]
