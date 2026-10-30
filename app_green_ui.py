"""CropCare — Next-Generation Agricultural Intelligence Dashboard.

A premium, guided 4-screen SaaS application providing transfer-learning
leaf screening and hyper-local environmental intelligence.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from src.cropcare.climate_engine import get_environment
from src.cropcare.agronomy_engine import (
    calculate_foliar_damage_index,
    classify_disease_severity,
    get_nurturing_recommendations,
)
from src.cropcare.image_engine import (
    get_registered_field_leaves,
    get_supported_crops,
    predict_image,
    register_field_leaf,
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
    translate_crop_name,
    translate_disease_name,
    translate_diagnosis,
)

# New engines from image-engine branch
try:
    from src.cropcare.economic_engine import estimate_economic_impact
    HAS_ECONOMIC = True
except ImportError:
    HAS_ECONOMIC = False

try:
    from src.cropcare.advisory_engine import (
        get_weather_smart_ipm,
        generate_recovery_roadmap,
        build_whatsapp_share_url,
    )
    HAS_ADVISORY = True
except Exception:
    HAS_ADVISORY = False
    get_weather_smart_ipm = None
    generate_recovery_roadmap = None

    def build_whatsapp_share_url(crop: str, disease: str, confidence_pct: float, location_name: str, urgency: str) -> str:
        import urllib.parse
        msg = f"🌱 *CropCare Alert*\n• Crop: {crop}\n• Diagnosis: {disease} ({confidence_pct:.1f}%)\n• Location: {location_name}\n• Urgency: {urgency}"
        return f"https://api.whatsapp.com/send?text={urllib.parse.quote(msg)}"

try:
    from src.cropcare.location_engine import (
        search_location_online,
        search_location_offline,
        get_states,
        get_districts_for_state,
        TOP_PRESET_LOCATIONS,
        AGRI_DISTRICTS,
    )
    HAS_LOCATION = True
except ImportError:
    HAS_LOCATION = False
    TOP_PRESET_LOCATIONS = {}

try:
    from src.cropcare.alignment_engine import compute_climate_alignment
    HAS_ALIGNMENT = True
except ImportError:
    HAS_ALIGNMENT = False

try:
    from src.cropcare.explainability import generate_gradcam
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

try:
    from src.cropcare.report_engine import generate_field_report_html
    HAS_REPORT = True
except ImportError:
    HAS_REPORT = False

try:
    from src.cropcare.ui_components import (
        render_alignment_gauge,
        render_spray_banner,
    )
    HAS_UI_COMPONENTS = True
except ImportError:
    HAS_UI_COMPONENTS = False


# Configure page layout and title
st.set_page_config(
    page_title="CropCare — Agronomic Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# PREMIUM DESIGN SYSTEM & CUSTOM CSS
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --forest-dark: #123D2A;
        --forest-mid: #1B543A;
        --forest-light: #2A7352;
        --lime-accent: #B7E66B;
        --lime-light: #EBF8D3;
        --ivory-bg: #F6F7EF;
        --surface-white: #FFFFFF;
        --text-dark: #11231B;
        --text-muted: #526E60;
        --border-subtle: rgba(18, 61, 42, 0.08);
        --shadow-card: 0 10px 30px -5px rgba(18, 61, 42, 0.07), 0 4px 12px -2px rgba(18, 61, 42, 0.04);
        --shadow-hero: 0 20px 40px -10px rgba(18, 61, 42, 0.25);
    }

    *, *::before, *::after {
        box-sizing: border-box !important;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--text-dark);
        background-color: var(--ivory-bg);
        overflow-x: hidden !important;
    }

    /* Main container max-width and fluid padding (Prevents Edge Cutting) */
    .block-container {
        max-width: 1240px !important;
        width: 100% !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2.8rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        box-sizing: border-box !important;
        overflow-x: hidden !important;
    }

    /* Top Navigation Brand Bar */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.4rem;
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.4rem;
        width: 100%;
        box-sizing: border-box;
    }
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--forest-dark);
        letter-spacing: -0.02em;
    }
    .brand-tag {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: var(--lime-light);
        color: var(--forest-dark);
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        border: 1px solid rgba(183, 230, 107, 0.5);
    }

    /* Step Progress Indicator (Fluid Responsive) */
    .step-progress-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.8rem;
        box-shadow: var(--shadow-card);
        width: 100%;
        box-sizing: border-box;
        overflow-x: auto;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-muted);
        text-decoration: none;
        white-space: nowrap;
    }
    .step-item.active {
        color: var(--forest-dark);
        font-weight: 700;
    }
    .step-pill {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        background: #E8ECE6;
        color: #6E8578;
        font-weight: 700;
        flex-shrink: 0;
    }
    .step-item.active .step-pill {
        background: var(--forest-dark);
        color: var(--lime-accent);
        box-shadow: 0 0 0 3px rgba(183, 230, 107, 0.4);
    }
    .step-item.completed .step-pill {
        background: var(--lime-accent);
        color: var(--forest-dark);
    }
    .step-divider {
        flex: 1;
        min-width: 12px;
        height: 2px;
        background: #E8ECE6;
        margin: 0 0.6rem;
    }
    .step-divider.active {
        background: var(--forest-dark);
    }

    /* Hero Card (Screen 1) */
    .hero-container {
        background: linear-gradient(135deg, #123D2A 0%, #1A543A 55%, #103222 100%);
        color: #FFFFFF;
        border-radius: 24px;
        padding: 2.2rem 2rem;
        box-shadow: var(--shadow-hero);
        position: relative;
        overflow: hidden;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(183, 230, 107, 0.2);
        width: 100%;
        box-sizing: border-box;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(183, 230, 107, 0.15);
        color: var(--lime-accent);
        border: 1px solid rgba(183, 230, 107, 0.35);
        padding: 0.35rem 0.9rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .hero-title span {
        color: var(--lime-accent);
    }
    .hero-sub {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #D3E4DB;
        max-width: 620px;
        margin-bottom: 2rem;
    }

    /* 3D Botanical Visual Graphic & Farmer Hero Card */
    .botanical-3d-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.2);
    }
    .botanical-badge {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--lime-accent);
        margin-top: 0.6rem;
    }

    /* Farmer Hero Card */
    .farmer-hero-card {
        background: linear-gradient(145deg, #184C34 0%, #0F2E20 100%);
        border: 1px solid rgba(183, 230, 107, 0.25);
        border-radius: 24px;
        padding: 1.25rem;
        box-shadow: var(--shadow-hero);
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
        position: relative;
        overflow: hidden;
        width: 100%;
        box-sizing: border-box;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .farmer-hero-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 24px 50px -8px rgba(18, 61, 42, 0.45);
    }
    .farmer-img-frame {
        position: relative;
        width: 100%;
        border-radius: 18px;
        overflow: hidden;
        aspect-ratio: 4/3;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .farmer-hero-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        transition: transform 0.4s ease;
    }
    .farmer-hero-card:hover .farmer-hero-img {
        transform: scale(1.02);
    }
    .farmer-overlay-badge {
        position: absolute;
        bottom: 12px;
        left: 12px;
        background: rgba(18, 61, 42, 0.88);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(183, 230, 107, 0.4);
        border-radius: 20px;
        padding: 0.35rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--lime-accent);
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .farmer-card-caption {
        padding: 0.2rem 0.4rem 0.2rem;
    }
    .farmer-card-caption .caption-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.08rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.25rem;
    }
    .farmer-card-caption .caption-sub {
        font-size: 0.84rem;
        color: #D3E4DB;
        line-height: 1.45;
    }

    /* Standard Surface Card */
    .surface-card {
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.5rem;
    }
    .surface-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--forest-dark);
        margin-bottom: 0.5rem;
    }
    .surface-card p.card-desc {
        font-size: 0.92rem;
        color: var(--text-muted);
        margin-bottom: 1.2rem;
    }

    /* Feature Cards on Welcome Screen */
    .feature-card {
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 28px -5px rgba(18, 61, 42, 0.1);
    }
    .feature-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: var(--lime-light);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--forest-dark);
        margin-bottom: 0.4rem;
    }
    .feature-body {
        font-size: 0.88rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* Diagnostic Results Card */
    .diagnostic-panel {
        background: #F9FCF7;
        border: 1.5px solid #D6E8D0;
        border-radius: 16px;
        padding: 1.4rem;
        margin-top: 1rem;
    }

    /* Scientific Disclaimer Box */
    .disclaimer-banner {
        background: #FDFEFB;
        border-left: 4px solid var(--forest-mid);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.85rem 1.2rem;
        margin: 1.2rem 0;
        font-size: 0.84rem;
        line-height: 1.5;
        color: #244233;
    }

    /* Crop selection badge & directory styling */
    .crop-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #EBF8D3;
        border: 1px solid rgba(183, 230, 107, 0.6);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-weight: 700;
        font-size: 0.82rem;
        color: #123D2A;
    }
    .crop-meta-box {
        background: #FFFFFF;
        border: 1px solid rgba(18, 61, 42, 0.12);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .pathology-tag {
        display: inline-block;
        background: #F1F5EE;
        color: #1B543A;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        margin: 0.15rem 0.25rem 0.15rem 0;
        border: 1px solid #DFE7DB;
    }
    .prob-bar-container {
        width: 100%;
        background-color: #E8ECE6;
        border-radius: 8px;
        overflow: hidden;
        height: 12px;
        margin-top: 4px;
        margin-bottom: 10px;
    }
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #1B543A 0%, #2A7352 70%, #B7E66B 100%);
        border-radius: 8px;
    }

    /* Streamlit Button Restyling */
    div.stButton > button[kind="primary"] {
        background-color: var(--forest-dark) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.8rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 12px rgba(18, 61, 42, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--forest-mid) !important;
        box-shadow: 0 6px 16px rgba(18, 61, 42, 0.28) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: var(--surface-white) !important;
        color: var(--forest-dark) !important;
        font-weight: 600 !important;
        border: 1.5px solid var(--forest-dark) !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.6rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: var(--ivory-bg) !important;
        border-color: var(--forest-mid) !important;
    }

    /* Floating CropCare AI Assistant Launcher */
    .st-key-chat_launcher_btn {
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        z-index: 999990 !important;
    }
    .st-key-chat_launcher_btn > button {
        border-radius: 50px !important;
        padding: 0.75rem 1.4rem !important;
        background: var(--forest-dark) !important;
        color: var(--lime-accent) !important;
        border: 2px solid var(--lime-accent) !important;
        box-shadow: 0 10px 25px rgba(18, 61, 42, 0.4) !important;
        font-weight: 700 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    .st-key-chat_launcher_btn > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 30px rgba(18, 61, 42, 0.5) !important;
    }

    /* Floating Chat Dialog Panel */
    .st-key-cropcare_chat_panel {
        position: fixed !important;
        bottom: 85px !important;
        right: 24px !important;
        width: 390px !important;
        max-width: calc(100vw - 48px) !important;
        max-height: calc(100vh - 110px) !important;
        background: var(--surface-white) !important;
        border: 1.5px solid rgba(18, 61, 42, 0.18) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 50px rgba(18, 61, 42, 0.3) !important;
        z-index: 999995 !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 1.2rem !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE INITIALIZATION (Zero-Loss Navigation)
# ==============================================================================
if "screen" not in st.session_state:
    st.session_state.screen = 1  # 1: Welcome, 2: Leaf, 3: Climate, 4: Summary

if "app_language" not in st.session_state:
    st.session_state.app_language = "en"

if "leaf_image" not in st.session_state:
    st.session_state.leaf_image = None

if "leaf_image_id" not in st.session_state:
    st.session_state.leaf_image_id = None

if "leaf_result" not in st.session_state:
    st.session_state.leaf_result = None

if "selected_crop_type" not in st.session_state:
    st.session_state.selected_crop_type = "All Crops (Auto-Detect)"

if "use_tta_boost" not in st.session_state:
    st.session_state.use_tta_boost = True

if "climate_lat" not in st.session_state:
    st.session_state.climate_lat = 19.0760

if "climate_lon" not in st.session_state:
    st.session_state.climate_lon = 72.8777

if "climate_days" not in st.session_state:
    st.session_state.climate_days = 30

if "climate_result" not in st.session_state:
    st.session_state.climate_result = None

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


@st.cache_data
def get_image_base64(image_path: str) -> str:
    """Read a local image file and return its base64 data URI string."""
    try:
        p = Path(image_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / image_path
        if p.exists():
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            suffix = p.suffix.lower().replace(".", "")
            mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
            return f"data:image/{mime};base64,{encoded}"
    except Exception:
        pass
    return ""


def format_metric(value: Optional[float], unit: str = "", precision: int = 1) -> str:
    """Format numeric values safely without converting None into 0.0."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f} {unit}".strip()


# Available crop choices with friendly labels and icons
CROP_OPTIONS = [
    ("All Crops (Auto-Detect)", "🌱 All Crops (Auto-Detect — 14 Species, 38 Pathologies)"),
    ("Tomato", "🍅 Tomato (10 Pathologies & Healthy)"),
    ("Potato", "🥔 Potato (Early/Late Blight & Healthy)"),
    ("Corn_(maize)", "🌽 Corn / Maize (Rust, Blight, Leaf Spot)"),
    ("Apple", "🍎 Apple (Scab, Black Rot, Rust & Healthy)"),
    ("Grape", "🍇 Grape (Black Rot, Esca, Blight & Healthy)"),
    ("Pepper,_bell", "🫑 Bell Pepper (Bacterial Spot & Healthy)"),
    ("Strawberry", "🍓 Strawberry (Leaf Scorch & Healthy)"),
    ("Peach", "🍑 Peach (Bacterial Spot & Healthy)"),
    ("Cherry_(including_sour)", "🍒 Cherry (Powdery Mildew & Healthy)"),
    ("Orange", "🍊 Orange (Citrus Greening / Huanglongbing)"),
    ("Squash", "🎃 Squash (Powdery Mildew)"),
    ("Soybean", "🫘 Soybean (Healthy Foliage)"),
    ("Blueberry", "🫐 Blueberry (Healthy Foliage)"),
    ("Raspberry", "🍇 Raspberry (Healthy Foliage)"),
]

# ==============================================================================
# TOP BRANDING, LANGUAGE SWITCHER & PROGRESS BAR
# ==============================================================================
top_brand_col, top_lang_col = st.columns([3.2, 1.2])

lang_keys = list(SUPPORTED_LANGUAGES.keys())
lang_labels = [
    SUPPORTED_LANGUAGES[k].get(
        "label",
        f"{SUPPORTED_LANGUAGES[k].get('flag', '')} {SUPPORTED_LANGUAGES[k].get('native', k)}"
    )
    for k in lang_keys
]
curr_lang_idx = lang_keys.index(st.session_state.app_language) if st.session_state.app_language in lang_keys else 0

with top_brand_col:
    st.markdown(
        f"""
        <div class="brand-bar">
            <div class="brand-logo">🌱 CropCare™</div>
            <div class="brand-tag">{t('app_tagline', st.session_state.app_language)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_lang_col:
    chosen_label = st.selectbox(
        "🌐 Language",
        options=lang_labels,
        index=curr_lang_idx,
        key="global_lang_selector",
        help="Choose preferred interface & audio guidance language (English, हिन्दी, मराठी, తెలుగు, தமிழ், Español)",
    )
    chosen_code = lang_keys[lang_labels.index(chosen_label)]
    if chosen_code != st.session_state.app_language:
        st.session_state.app_language = chosen_code
        st.rerun()

lang = st.session_state.app_language

# Interactive Top Navigation Buttons
t_col1, t_col2, t_col3, t_col4 = st.columns(4)
nav_items = [
    (t_col1, 1, f"🏠 {t('step_1', lang)}"),
    (t_col2, 2, f"🍃 {t('step_2', lang)}"),
    (t_col3, 3, f"🌦️ {t('step_3', lang)}"),
    (t_col4, 4, f"📋 {t('step_4', lang)}"),
]
for col, s_num, s_title in nav_items:
    with col:
        b_type = "primary" if st.session_state.screen == s_num else "secondary"
        if st.button(s_title, key=f"tnav_{s_num}", type=b_type, use_container_width=True):
            st.session_state.screen = s_num
            st.rerun()

# Render Persistent Guided Step Indicator
curr = st.session_state.screen
step1_cls = "active" if curr == 1 else ("completed" if curr > 1 else "")
step2_cls = "active" if curr == 2 else ("completed" if curr > 2 else "")
step3_cls = "active" if curr == 3 else ("completed" if curr > 3 else "")
step4_cls = "active" if curr == 4 else ""

div1_cls = "active" if curr > 1 else ""
div2_cls = "active" if curr > 2 else ""
div3_cls = "active" if curr > 3 else ""

s1_txt = t("step_1", st.session_state.app_language)
s2_txt = t("step_2", st.session_state.app_language)
s3_txt = t("step_3", st.session_state.app_language)
s4_txt = t("step_4", st.session_state.app_language)

st.markdown(
    f"""
    <div class="step-progress-wrapper">
        <div class="step-item {step1_cls}">
            <div class="step-pill">1</div>
            <span>{s1_txt}</span>
        </div>
        <div class="step-divider {div1_cls}"></div>
        <div class="step-item {step2_cls}">
            <div class="step-pill">2</div>
            <span>{s2_txt}</span>
        </div>
        <div class="step-divider {div2_cls}"></div>
        <div class="step-item {step3_cls}">
            <div class="step-pill">3</div>
            <span>{s3_txt}</span>
        </div>
        <div class="step-divider {div3_cls}"></div>
        <div class="step-item {step4_cls}">
            <div class="step-pill">4</div>
            <span>{s4_txt}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SCREEN 1: WELCOME & START
# ==============================================================================
if st.session_state.screen == 1:
    hero_col_left, hero_col_right = st.columns([1.25, 0.95], gap="large")

    with hero_col_left:
        st.markdown(
            f"""
            <div class="hero-container">
                <div class="hero-badge">⚡ {t('precision_tech', st.session_state.app_language)}</div>
                <div class="hero-title">{t('welcome_title', st.session_state.app_language)}</div>
                <div class="hero-sub">
                    {t('welcome_sub', st.session_state.app_language)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1.3])
        with c1:
            if st.button(f"{t('start_screening', lang)} →", type="primary", use_container_width=True):
                st.session_state.screen = 2
                st.rerun()
        with c2:
            if st.button(f"🧠 {t('continual_studio', lang)} →", type="secondary", use_container_width=True):
                st.session_state.screen = 2
                st.rerun()

        st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
        if st.button(t('judges_eval_btn', lang), type="secondary", use_container_width=True):
            st.session_state.screen = 2
            st.rerun()

    with hero_col_right:
        # Realistic Indian farmer inspecting crops in a green agricultural field
        farmer_img_uri = get_image_base64("assets/farmer_hero.jpg")
        if farmer_img_uri:
            st.markdown(
                f"""
                <div class="farmer-hero-card">
                    <div class="farmer-img-frame">
                        <img src="{farmer_img_uri}" alt="Indian farmer inspecting crop leaves in a vibrant green agricultural field" class="farmer-hero-img" />
                        <div class="farmer-overlay-badge">
                            <span>🌱 {t('farmer_badge', lang)}</span>
                        </div>
                    </div>
                    <div class="farmer-card-caption">
                        <div class="caption-title">{t('caption_title', lang)}</div>
                        <div class="caption-sub">{t('caption_sub', lang)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Fallback if image asset fails to load
            st.image("assets/farmer_hero.jpg", caption="Agronomic Field Assessment", use_container_width=True)

    # Three Core Pillars
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3, gap="medium")

    with f_col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🍃</div>
                <div class="feature-title">{t('feat1_title', lang)}</div>
                <div class="feature-body">
                    {t('feat1_desc', lang)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">{t('feat2_title', lang)}</div>
                <div class="feature-body">
                    {t('feat2_desc', lang)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col3:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <div class="feature-title">{t('feat3_title', lang)}</div>
                <div class="feature-body">
                    {t('feat3_desc', lang)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# SCREEN 2: LEAF SCREENING & CONTINUAL LEARNING STUDIO
# ==============================================================================
elif st.session_state.screen == 2:
    st.markdown(
        f"""
        <div class="surface-card">
            <h3>{t('screen2_title', lang)}</h3>
            <p class="card-desc">
                {t('screen2_desc', lang)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_screening, tab_dispatch, tab_studio, tab_library, tab_gradcam, tab_evaluation = st.tabs([
        t('tab_screening', lang),
        t('tab_dispatch', lang),
        t('tab_studio', lang),
        t('tab_library', lang),
        t('tab_gradcam', lang),
        t('tab_evaluation', lang),
    ])

    # --------------------------------------------------------------------------
    # TAB 1: DIAGNOSTIC LEAF SCREENING
    # --------------------------------------------------------------------------
    with tab_screening:
        st.info(
            "💡 **Looking to train/adapt the model on new crops or rare diseases?** "
            "Switch to the **'🧠 2. Continual Learning Studio (Upload & Teach Model)'** tab directly above to register reference photos and expand the memory bank in 1 click!",
            icon="🧠",
        )
        crops_catalog = get_supported_crops()

        st.markdown("#### 🎯 Target Crop Selection")
        crop_labels = [opt[1] for opt in CROP_OPTIONS]
        crop_keys = [opt[0] for opt in CROP_OPTIONS]

        curr_idx = 0
        for idx, (k, _) in enumerate(CROP_OPTIONS):
            if k == st.session_state.selected_crop_type:
                curr_idx = idx
                break

        selected_label = st.selectbox(
            t("target_crop", lang),
            options=crop_labels,
            index=curr_idx,
            help="Targeting your crop eliminates 28+ irrelevant pathologies from the neural softmax, boosting precision and confidence.",
        )

        chosen_crop_key = crop_keys[crop_labels.index(selected_label)]
        if chosen_crop_key != st.session_state.selected_crop_type:
            st.session_state.selected_crop_type = chosen_crop_key
            if st.session_state.leaf_image is not None:
                filter_arg = None if chosen_crop_key == "All Crops (Auto-Detect)" else chosen_crop_key
                st.session_state.leaf_result = predict_image(
                    st.session_state.leaf_image,
                    crop_filter=filter_arg,
                    use_tta=st.session_state.use_tta_boost,
                )

        # Active Crop Metadata Card
        if chosen_crop_key != "All Crops (Auto-Detect)" and chosen_crop_key in crops_catalog:
            c_info = crops_catalog[chosen_crop_key]
            st.markdown(
                f"""
                <div class="crop-meta-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="font-weight: 700; color: #123D2A; font-size: 1.1rem;">
                            {c_info['icon']} Targeting: <strong>{c_info['display_name']}</strong>
                        </div>
                        <div class="crop-badge-pill">⚡ {len(c_info['diseases'])} Monitored Conditions</div>
                    </div>
                    <div style="font-size: 0.85rem; color: #526E60; margin-bottom: 0.6rem;">
                        {c_info.get('desc', 'Inference will focus strictly on pathologies affecting this crop.')}
                    </div>
                    <div>
                        {' '.join([f'<span class="pathology-tag">{d}</span>' for d in c_info['diseases']])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Crops & Diseases Library Expander
        with st.expander("📚 Browse Monitored Crops & Disease Library (14 Species, 38 Pathologies)", expanded=False):
            st.caption("CropCare is trained to differentiate among the following conditions across 14 major agricultural crops:")
            cat_cols = st.columns(2)
            c_idx = 0
            for k_raw, meta in sorted(crops_catalog.items(), key=lambda x: x[1]["display_name"]):
                with cat_cols[c_idx % 2]:
                    st.markdown(
                        f"""
                        <div style="background: #FDFEFB; border: 1px solid #DFE7DB; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;">
                            <div style="font-weight: 700; color: #123D2A; font-size: 0.98rem;">
                                {meta['icon']} {meta['display_name']} <span style="font-size: 0.78rem; font-weight: 600; color: #526E60;">({len(meta['diseases'])} conditions)</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #2A7352; margin-top: 0.35rem; line-height: 1.4;">
                                {' • '.join(meta['diseases'])}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                c_idx += 1

        # Quick Sample Selector
        st.markdown("##### ⚡ Quick-Test with Curated Botanical Samples:")
        sm_c1, sm_c2, sm_c3, sm_c4, sm_c5, sm_c6 = st.columns(6)
        sample_items = [
            (sm_c1, "🍅 Tomato Blight", "data/sample_leaves/sample_tomato_early_blight.png", "Tomato"),
            (sm_c2, "🥔 Potato Blight", "data/sample_leaves/sample_potato_late_blight.png", "Potato"),
            (sm_c3, "🌽 Corn Rust", "data/sample_leaves/sample_corn_common_rust.png", "Corn_(maize)"),
            (sm_c4, "🍎 Apple Scab", "data/sample_leaves/sample_apple_scab.png", "Apple"),
            (sm_c5, "🌿 Tomato Healthy", "data/sample_leaves/sample_tomato_healthy.png", "Tomato"),
            (sm_c6, "🌾 Rice (Continual Demo)", "data/sample_leaves/sample_rice_brown_spot.png", "All Crops (Auto-Detect)"),
        ]

        for col, btn_label, file_path, def_crop in sample_items:
            with col:
                if st.button(btn_label, use_container_width=True):
                    if os.path.exists(file_path):
                        loaded_img = Image.open(file_path)
                        st.session_state.leaf_image = loaded_img
                        st.session_state.selected_crop_type = def_crop
                        filter_arg = None if def_crop == "All Crops (Auto-Detect)" else def_crop
                        st.session_state.leaf_result = predict_image(
                            loaded_img,
                            crop_filter=filter_arg,
                            use_tta=st.session_state.use_tta_boost,
                        )
                        st.rerun()

        # Main Uploader & Preview
        leaf_col_left, leaf_col_right = st.columns([1, 1], gap="large")

        with leaf_col_left:
            st.markdown(f"#### 📷 {t('upload_photo', lang)}")

            input_mode = st.radio(
                "Acquisition Method:",
                options=["📁 Upload Image", "📸 Click Photo (Camera)"],
                horizontal=True,
                help="Choose between selecting an image file or snapping a live photo with your camera.",
            )

            active_file = None
            new_source_id = None

            if input_mode == "📁 Upload Image":
                active_file = st.file_uploader(
                    t('upload_file_label', lang),
                    type=["jpg", "jpeg", "png"],
                    key="screening_uploader",
                    help="High-contrast, focused imagery centered on a single leaf produces the most reliable screening.",
                )
                if active_file is not None:
                    new_source_id = f"upload_{active_file.name}_{active_file.size}"
            else:
                active_file = st.camera_input(
                    t('camera_input_label', lang),
                    key="leaf_camera_input",
                    help="Center the plant leaf with uniform lighting and click Take Photo.",
                )
                if active_file is not None:
                    new_source_id = f"camera_{active_file.name}_{active_file.size}"


            if active_file is not None and new_source_id != st.session_state.leaf_image_id:
                try:
                    img = Image.open(active_file).convert("RGB")
                    st.session_state.leaf_image = img
                    st.session_state.leaf_image_id = new_source_id
                    st.session_state.leaf_result = None
                except Exception as e:
                    st.error(f"Image decoding failed: {e}")

            if st.session_state.leaf_image is not None:
                st.image(st.session_state.leaf_image, caption="Current Leaf Under Analysis", use_container_width=True)

            # Real-Time Accuracy Options
            st.session_state.use_tta_boost = st.checkbox(
                "⚡ Multi-Angle Test-Time Augmentation (TTA)",
                value=st.session_state.use_tta_boost,
                help="Evaluates original, horizontal flip, and aspect-ratio preserving crops to maximize real-world prediction accuracy.",
            )

            # Run Screening Action
            if st.session_state.leaf_image is not None:
                run_btn_label = f"⚡ {t('btn_submit_screening', lang)}"
                if st.button(run_btn_label, type="primary", use_container_width=True):
                    with st.spinner("Executing neural inference via MobileNetV3..."):
                        try:
                            filter_arg = None if st.session_state.selected_crop_type == "All Crops (Auto-Detect)" else st.session_state.selected_crop_type
                            res = predict_image(
                                st.session_state.leaf_image,
                                crop_filter=filter_arg,
                                use_tta=st.session_state.use_tta_boost,
                            )
                            st.session_state.leaf_result = res
                        except Exception as inf_err:
                            st.error(f"Inference execution error: {inf_err}")
            else:
                st.info("Upload a photograph or click one of the quick test sample buttons above.")

        # Diagnostic Output
        with leaf_col_right:
            st.markdown("#### 🔬 Neural Diagnostic Output")

            res = st.session_state.leaf_result
            if res is None:
                weights_found = os.path.exists("models/plantvillage_mobilenetv3.pth")
                mapping_found = os.path.exists("models/class_mapping.json")

                if weights_found and mapping_found:
                    st.markdown(
                        """
                        <div style="background: #FFFFFF; border: 2px dashed #CCD7D0; border-radius: 16px; padding: 2.5rem 1.5rem; text-align: center; color: #526E60;">
                            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📷</div>
                            <div style="font-weight: 700; color: #123D2A; font-size: 1.1rem; margin-bottom: 0.3rem;">Ready for Inference</div>
                            <div style="font-size: 0.88rem;">Select a Crop Type or sample leaf, then click "Run Screening" to begin.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("ℹ️ **Inference Engine in Demo Mode**: Model weights missing.", icon="ℹ️")
            else:
                warnings = res.get("warnings", [])
                for w in warnings:
                    if "Low prediction confidence" not in w:
                        st.warning(f"⚠️ {w}")

                pred_class = res.get("predicted_class")
                if pred_class is None:
                    st.error("Inference could not classify this image.")
                    for w in warnings:
                        st.caption(f"• {w}")
                else:
                    crop = res.get("crop", "Unknown")
                    disease = res.get("disease", "Unknown")
                    display_crop = translate_crop_name(crop, lang)
                    display_disease = translate_disease_name(disease, lang)
                    conf = res.get("confidence", 0.0)
                    uncertain = res.get("uncertain", False)
                    crop_filtered = res.get("crop_filter_applied", False)

                    if crop_filtered:
                        mode_badge = f"""<span class="crop-badge-pill" style="margin-bottom: 0.6rem;">🎯 {t('screening_mode', lang)}: {display_crop}</span>"""
                    else:
                        det_crop = res.get("detected_crop", crop)
                        det_crop_trans = translate_crop_name(det_crop, lang)
                        det_conf = res.get("detected_crop_confidence", 0.0) * 100
                        mode_badge = f"""<span class="crop-badge-pill" style="background: #E8ECE6; color: #1B543A; margin-bottom: 0.6rem;">🔍 {det_crop_trans} ({det_conf:.1f}%)</span>"""

                    st.markdown(
                        f"""
                        <div class="diagnostic-panel">
                            {mode_badge}
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; margin-top: 0.4rem;">
                                <div>
                                    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">{t('crop_identified', lang)}</div>
                                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #123D2A;">{display_crop}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">{t('confidence_score', lang)}</div>
                                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #123D2A;">{conf*100:.1f}%</div>
                                </div>
                            </div>
                            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">{t('visual_condition', lang)}</div>
                            <div style="font-size: 1.25rem; font-weight: 600; color: #1B543A; margin-bottom: 0.8rem;">{display_disease}</div>
                            <div style="font-size: 0.75rem; color: #526E60;">Model: <code>{res.get('model_name', 'MobileNetV3-Small-PlantVillage')}</code> | TTA: <strong>{'Enabled' if st.session_state.use_tta_boost else 'Disabled'}</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Show active learning match badge if available
                    mem_match = res.get("field_memory_match")
                    if mem_match:
                        st.markdown(
                            f"""
                            <div style="background: #EBF8D3; border: 1.5px solid #A3D95B; border-radius: 12px; padding: 0.8rem 1rem; margin-top: 0.8rem;">
                                <div style="font-weight: 700; color: #123D2A; font-size: 0.9rem;">
                                    🧠 Agronomist Field Memory Match: <strong>{mem_match.get('crop')} — {mem_match.get('disease')}</strong>
                                </div>
                                <div style="font-size: 0.8rem; color: #1B543A; margin-top: 0.2rem;">
                                    Matched active memory bank with <strong>{mem_match.get('similarity')*100:.1f}% cosine similarity</strong>.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if uncertain:
                        st.warning(
                            f"⚠️ **Low Confidence Result ({conf*100:.1f}%):** "
                            "Visual indicators are ambiguous or outside laboratory training baseline. "
                            "Try selecting your exact Crop Type above or testing with an uncluttered background.",
                            icon="⚠️",
                        )
                    else:
                        st.success(f"Screening complete: High-confidence visual match ({conf*100:.1f}%).")

                    # Offer 1-click crop focus in Auto-Detect mode
                    if not crop_filtered and res.get("detected_crop"):
                        det_crop_raw = res.get("detected_crop")
                        if st.button(f"🎯 Sharpen: Isolate Analysis Specifically to {det_crop_raw}", use_container_width=True):
                            st.session_state.selected_crop_type = det_crop_raw
                            st.session_state.leaf_result = predict_image(
                                st.session_state.leaf_image,
                                crop_filter=det_crop_raw,
                                use_tta=st.session_state.use_tta_boost,
                            )
                            st.rerun()

                    # Probability Breakdown
                    probs = res.get("probabilities", {})
                    if probs:
                        with st.expander("📊 View Condition Probability Breakdown", expanded=True):
                            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                            for cls_k, p_val in sorted_probs[:6]:
                                c_display = cls_k.split("___")[-1].replace("_", " ")
                                pct = p_val * 100
                                st.markdown(
                                    f"""
                                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: #123D2A;">
                                        <span>{c_display}</span>
                                        <span>{pct:.1f}%</span>
                                    </div>
                                    <div class="prob-bar-container">
                                        <div class="prob-bar-fill" style="width: {pct:.1f}%;"></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # ==================================================================
                    # DISEASE PROGRESSION STAGE & SEVERITY ASSESSMENT
                    # ==================================================================
                    sev = classify_disease_severity(
                        st.session_state.leaf_image,
                        disease_name=disease,
                        confidence=conf,
                    )
                    st.markdown("---")
                    st.markdown(f"#### {t('severity_title', lang)}")

                    sev_colors = sev["color_theme"]
                    st.markdown(
                        f"""
                        <div style="background: #FFFFFF; border: 1.5px solid {sev_colors['border']}; border-radius: 14px; padding: 1.1rem 1.3rem; margin-top: 0.6rem; margin-bottom: 1.2rem; box-shadow: 0 4px 14px rgba(0,0,0,0.04);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                                <span style="background: {sev_colors['badge_bg']}; color: {sev_colors['badge_text']}; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; padding: 0.25rem 0.75rem; border-radius: 20px; border: 1px solid {sev_colors['border']};">
                                    {sev['severity_level']}
                                </span>
                                <span style="font-size: 0.84rem; font-weight: 700; color: #526E60;">
                                    {t('foliar_damage', lang)}: <strong>{sev['damage_percentage']:.1f}% Canopy Affected</strong>
                                </span>
                            </div>
                            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 800; color: #123D2A; margin-bottom: 0.4rem;">
                                {sev['stage_name']}
                            </div>
                            <div style="width: 100%; background: #E8ECE6; border-radius: 8px; height: 10px; margin-bottom: 0.8rem; overflow: hidden;">
                                <div style="width: {max(6.0, sev['damage_percentage'])}%; height: 100%; background: {sev_colors['progress_gradient']}; border-radius: 8px;"></div>
                            </div>
                            <div style="font-size: 0.85rem; color: #2A4736; line-height: 1.5; margin-bottom: 0.6rem;">
                                <strong>Physiological State:</strong> {sev['pathological_state']}
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 600; color: #526E60; border-top: 1px solid #E8ECE6; padding-top: 0.5rem;">
                                <span>⏱️ <strong>{t('urgency_label', lang)}:</strong> {sev['urgency']}</span>
                                <span>📈 <strong>{t('prognosis_label', lang)}:</strong> {sev['prognosis'][:50]}...</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ==================================================================
                    # KISAN EMERGENCY DISPATCH QUICK-ACTIONS (WhatsApp & SMS)
                    # ==================================================================
                    location_name = st.session_state.get("location_name", "Agricultural Field")
                    wa_url_quick = build_whatsapp_share_url(
                        crop=crop, disease=disease,
                        confidence_pct=conf * 100,
                        location_name=location_name,
                        urgency=sev["urgency"],
                    )
                    import urllib.parse
                    sms_body_quick = urllib.parse.quote(
                        f"CropCare Kisan Alert: {display_crop} showing symptoms of {display_disease} ({conf*100:.0f}% confidence, {sev['severity_level']}). Location: {location_name}. Please advise on treatment."
                    )
                    st.markdown(
                        f"""
                        <div style="background:#F0FDF4;border:1.5px solid #86EFAC;border-radius:12px;padding:0.95rem 1.15rem;margin-bottom:1.2rem;">
                            <div style="font-weight:700;color:#166534;font-size:0.95rem;margin-bottom:0.35rem;">
                                🚨 {t('dispatch_title', lang)} (1-Click Action)
                            </div>
                            <div style="font-size:0.83rem;color:#14532D;margin-bottom:0.75rem;">
                                {t('dispatch_desc', lang)}
                            </div>
                            <div style="display:flex;gap:0.75rem;flex-wrap:wrap;">
                                <a href="{wa_url_quick}" target="_blank"
                                   style="display:inline-flex;align-items:center;gap:0.4rem;background:#25D366;color:#FFFFFF;font-weight:700;font-size:0.85rem;padding:0.55rem 0.95rem;border-radius:8px;text-decoration:none;box-shadow:0 2px 6px rgba(37,211,102,0.25);">
                                    📲 {t('whatsapp_btn', lang)}
                                </a>
                                <a href="sms:?body={sms_body_quick}"
                                   style="display:inline-flex;align-items:center;gap:0.4rem;background:#0284C7;color:#FFFFFF;font-weight:700;font-size:0.85rem;padding:0.55rem 0.95rem;border-radius:8px;text-decoration:none;box-shadow:0 2px 6px rgba(2,132,199,0.25);">
                                    {t('sms_btn', lang)}
                                </a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ==================================================================
                    # AGRONOMIC NURTURING & CULTURAL REMEDIATION PROTOCOL
                    # ==================================================================
                    nurture = get_nurturing_recommendations(
                        crop=crop,
                        disease=disease,
                        severity_stage=sev["stage_id"],
                    )
                    st.markdown(f"#### {t('nurturing_title', lang)}")
                    st.caption(f"Science-backed adjustments to irrigation, soil nutrition, and canopy care to eliminate {display_disease}:")

                    n_tab1, n_tab2, n_tab3, n_tab4 = st.tabs([
                        t('tab_irrigation', lang),
                        t('tab_nutrition', lang),
                        t('tab_sanitation', lang),
                        t('tab_bio_defense', lang),
                    ])

                    with n_tab1:
                        st.markdown(
                            f"""
                            <div style="background: #F8FAF7; border-left: 4px solid #0284C7; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.4rem;">
                                <div style="font-weight: 700; color: #0369A1; font-size: 0.95rem; margin-bottom: 0.3rem;">
                                    💧 {nurture['irrigation']['action']}
                                </div>
                                <div style="font-size: 0.85rem; color: #334155; line-height: 1.55; margin-bottom: 0.5rem;">
                                    {nurture['irrigation']['details']}
                                </div>
                                <div style="font-size: 0.8rem; font-weight: 600; color: #0284C7;">
                                    📅 <strong>Frequency Mandate:</strong> {nurture['irrigation']['frequency']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with n_tab2:
                        st.markdown(
                            f"""
                            <div style="background: #FDFEF9; border-left: 4px solid #16A34A; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.4rem;">
                                <div style="font-weight: 700; color: #15803D; font-size: 0.95rem; margin-bottom: 0.3rem;">
                                    🧪 {nurture['nutrition']['action']}
                                </div>
                                <div style="font-size: 0.85rem; color: #334155; line-height: 1.55; margin-bottom: 0.5rem;">
                                    {nurture['nutrition']['details']}
                                </div>
                                <div style="font-size: 0.8rem; font-weight: 600; color: #15803D;">
                                    💡 <strong>Nutrient Guidance:</strong> {nurture['nutrition']['fertilizer_tip']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with n_tab3:
                        st.markdown(
                            f"""
                            <div style="background: #FDFCF7; border-left: 4px solid #D97706; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.4rem;">
                                <div style="font-weight: 700; color: #B45309; font-size: 0.95rem; margin-bottom: 0.3rem;">
                                    ✂️ {nurture['sanitation']['action']}
                                </div>
                                <div style="font-size: 0.85rem; color: #334155; line-height: 1.55; margin-bottom: 0.5rem;">
                                    {nurture['sanitation']['details']}
                                </div>
                                <div style="font-size: 0.8rem; font-weight: 600; color: #B45309;">
                                    🧼 <strong>Implement Hygiene:</strong> {nurture['sanitation'].get('tools', 'Sanitize shears between plants.')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with n_tab4:
                        st.markdown(
                            f"""
                            <div style="background: #FAF7FC; border-left: 4px solid #7C3AED; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.4rem;">
                                <div style="font-weight: 700; color: #6D28D9; font-size: 0.95rem; margin-bottom: 0.3rem;">
                                    🛡️ {nurture['bio_defense']['action']}
                                </div>
                                <div style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                                    {nurture['bio_defense']['details']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ==================================================================
                    # VOICE AGRONOMIST AUDIO ADVISORY (Hands-free Multilingual Player)
                    # ==================================================================
                    audio_text = translate_advisory_text(
                        crop=crop,
                        disease=disease,
                        stage_name=sev["stage_name"],
                        damage_pct=sev["damage_percentage"],
                        irrigation_action=nurture["irrigation"]["action"],
                        nutrition_action=nurture["nutrition"]["action"],
                        sanitation_action=nurture["sanitation"]["action"],
                        bio_action=nurture["bio_defense"]["action"],
                        lang=st.session_state.app_language,
                    )
                    audio_title = f"CropCare Advisory — {crop} / {disease}"
                    audio_html = generate_audio_speech_html(
                        title=audio_title,
                        text_to_speak=audio_text,
                        lang=st.session_state.app_language,
                    )
                    st.markdown("---")
                    st.markdown(f"#### 🎙️ {t('audio_listen_btn', st.session_state.app_language)}")
                    st.caption("Native browser audio synthesizer providing real-time hands-free advisory in the farmer's chosen language:")
                    components.html(audio_html, height=130)

                    st.markdown(
                        """
                        <div class="disclaimer-banner">
                            <strong>Scientific Note:</strong> Softmax confidence indicates model certainty relative to its 
                            PlantVillage benchmark training distribution, not a biological diagnosis. Open-field microclimates 
                            and abiotic stresses can produce similar visual symptoms.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ==================================================================
                    # ECONOMIC IMPACT ESTIMATOR
                    # ==================================================================
                    if HAS_ECONOMIC:
                        st.markdown("---")
                        st.markdown("#### 💰 Economic Impact Estimator")
                        eco_col1, eco_col2 = st.columns([2, 1])
                        with eco_col2:
                            farm_acres = st.number_input(
                                "Farm Size (Acres)", min_value=0.1, max_value=500.0,
                                value=st.session_state.get("farm_acres", 1.0), step=0.5,
                                key="farm_acres_input",
                            )
                            st.session_state["farm_acres"] = farm_acres
                        with eco_col1:
                            eco = estimate_economic_impact(
                                crop_name=crop,
                                disease_name=disease,
                                confidence=conf,
                                farm_acres=farm_acres,
                            )
                        st.markdown(
                            f"""
                            <div style="background:#FFFFFF;border:1.5px solid #D6E0D9;border-radius:14px;padding:1.1rem 1.3rem;margin-top:0.4rem;">
                              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                                <span style="font-size:0.85rem;font-weight:700;color:#526E60;text-transform:uppercase;">Urgency Tier</span>
                                <span style="font-weight:700;font-size:0.92rem;color:#123D2A;">{eco['urgency']}</span>
                              </div>
                              <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;text-align:center;">
                                <div style="background:#F4F7F2;border-radius:10px;padding:0.7rem;">
                                  <div style="font-size:0.72rem;font-weight:700;color:#526E60;text-transform:uppercase;">Yield Loss Range</div>
                                  <div style="font-size:1.2rem;font-weight:800;color:#B45309;">{eco['loss_pct_range']}</div>
                                </div>
                                <div style="background:#F4F7F2;border-radius:10px;padding:0.7rem;">
                                  <div style="font-size:0.72rem;font-weight:700;color:#526E60;text-transform:uppercase;">Revenue At Risk</div>
                                  <div style="font-size:1.1rem;font-weight:800;color:#DC2626;">₹{eco['loss_value_max_inr']:,}</div>
                                </div>
                                <div style="background:#F4F7F2;border-radius:10px;padding:0.7rem;">
                                  <div style="font-size:0.72rem;font-weight:700;color:#526E60;text-transform:uppercase;">Treatment Cost</div>
                                  <div style="font-size:1.1rem;font-weight:800;color:#0369A1;">₹{eco['treatment_cost_inr']:,}</div>
                                </div>
                                <div style="background:#EBF8D3;border-radius:10px;padding:0.7rem;">
                                  <div style="font-size:0.72rem;font-weight:700;color:#526E60;text-transform:uppercase;">ROI if Treated</div>
                                  <div style="font-size:1.2rem;font-weight:800;color:#15803D;">{eco['roi_multiplier']}</div>
                                </div>
                              </div>
                              <div style="font-size:0.82rem;color:#334155;margin-top:0.7rem;line-height:1.5;">{eco['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ==================================================================
                    # WEATHER-SMART IPM ADVISORY & RECOVERY ROADMAP
                    # ==================================================================
                    if HAS_ADVISORY:
                        st.markdown("---")
                        st.markdown("#### 🌿 Weather-Smart IPM Advisory")
                        ipm = get_weather_smart_ipm(
                            disease=disease,
                            climate_result=st.session_state.get("climate_result"),
                        )
                        spray_w = ipm.get("spray_window", {})
                        spray_badge = spray_w.get("badge", "🟡 Check Local Conditions")
                        spray_advice = spray_w.get("advice", "")
                        st.markdown(
                            f"""
                            <div style="background:#FFFFFF;border-left:5px solid #1B543A;border-radius:12px;padding:1rem 1.3rem;margin-bottom:1rem;">
                              <div style="font-size:0.75rem;font-weight:700;color:#526E60;text-transform:uppercase;">Spray Application Window</div>
                              <div style="font-size:1.1rem;font-weight:700;color:#123D2A;margin:0.3rem 0;">{spray_badge}</div>
                              <div style="font-size:0.88rem;color:#334D3F;">{spray_advice}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        ipm_c1, ipm_c2, ipm_c3 = st.columns(3)
                        with ipm_c1:
                            st.markdown("**🌾 Cultural Practices**")
                            for item in ipm.get("cultural", []):
                                st.markdown(f"• {item}")
                        with ipm_c2:
                            st.markdown("**🧫 Biological Controls**")
                            for item in ipm.get("biological", []):
                                st.markdown(f"• {item}")
                        with ipm_c3:
                            st.markdown("**⚗️ Chemical Interventions**")
                            for item in ipm.get("chemical", []):
                                st.markdown(f"• {item}")

                        # 7-Day Recovery Roadmap
                        st.markdown("#### 📅 7-Day Recovery Roadmap")
                        roadmap = generate_recovery_roadmap(
                            crop=crop, disease=disease,
                            spray_status=spray_w.get("status", "OPTIMAL"),
                        )
                        for step in roadmap:
                            st.markdown(
                                f"""
                                <div style="display:flex;gap:1rem;align-items:flex-start;padding:0.6rem 0;border-bottom:1px solid #E8ECE6;">
                                  <div style="min-width:70px;font-weight:700;color:#123D2A;font-size:0.82rem;">{step['day']}</div>
                                  <div>
                                    <div style="font-weight:700;color:#1B543A;font-size:0.88rem;">{step['stage']}</div>
                                    <div style="font-size:0.84rem;color:#334155;line-height:1.5;">{step['action']}</div>
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Kisan Emergency Dispatch: WhatsApp & SMS Alerts
                        location_name = st.session_state.get("location_name", "Agricultural Field")
                        wa_url = build_whatsapp_share_url(
                            crop=crop, disease=disease,
                            confidence_pct=conf * 100,
                            location_name=location_name,
                            urgency=eco["urgency"] if HAS_ECONOMIC else "Review Required",
                        )
                        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                        disp_c1, disp_c2 = st.columns(2)
                        with disp_c1:
                            st.markdown(
                                f"""
                                <a href="{wa_url}" target="_blank"
                                   style="display:flex;align-items:center;justify-content:center;gap:0.5rem;background:#25D366;color:#FFFFFF;font-weight:700;font-size:0.88rem;padding:0.65rem 1rem;border-radius:10px;text-decoration:none;box-shadow:0 3px 10px rgba(37,211,102,0.25);">
                                    📲 {t('whatsapp_btn', lang)}
                                </a>
                                """,
                                unsafe_allow_html=True,
                            )
                        with disp_c2:
                            import urllib.parse
                            sms_content = urllib.parse.quote(
                                f"CropCare Farmer Alert: {display_crop} showing symptoms of {display_disease} ({conf*100:.0f}% confidence). Farm Location: {location_name}. Please advise on treatment."
                            )
                            st.markdown(
                                f"""
                                <a href="sms:?body={sms_content}"
                                   style="display:flex;align-items:center;justify-content:center;gap:0.5rem;background:#0284C7;color:#FFFFFF;font-weight:700;font-size:0.88rem;padding:0.65rem 1rem;border-radius:10px;text-decoration:none;box-shadow:0 3px 10px rgba(2,132,199,0.25);">
                                    📱 Dispatch Kisan SMS Alert to Extension Officer
                                </a>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Multimodal Gemini Vision Second Opinion
                        st.markdown("---")
                        st.markdown("#### 🤖 Gemini Multimodal Vision Second Opinion")
                        st.caption("Request independent multimodal visual verification from Google Gemini to verify symptoms and cross-check CNN predictions.")
                        if st.button("🔍 Request Independent Gemini AI Vision Verification", type="secondary", use_container_width=True, key="btn_gemini_vision"):
                            from src.cropcare.chat_engine import verify_leaf_with_gemini_vision
                            with st.spinner("Consulting Gemini Multimodal Vision..."):
                                gemini_vis = verify_leaf_with_gemini_vision(
                                    image=st.session_state.leaf_image,
                                    predicted_crop=crop,
                                    predicted_disease=disease,
                                )
                                st.session_state["gemini_vision_res"] = gemini_vis

                        if st.session_state.get("gemini_vision_res"):
                            vis_info = st.session_state["gemini_vision_res"]
                            st.markdown(
                                f"""
                                <div style="background:#F0F9FF;border:1.5px solid #BAE6FD;border-radius:12px;padding:1.1rem;margin-top:0.6rem;">
                                    <div style="font-weight:700;color:#0369A1;margin-bottom:0.4rem;font-size:0.95rem;">🤖 Clinical Multimodal AI Verification (Gemini 2.5 Flash)</div>
                                    <div style="font-size:0.9rem;color:#0C4A6E;line-height:1.6;">{vis_info.get('analysis', '')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # --------------------------------------------------------------------------
    # TAB 2: KISAN EMERGENCY DISPATCH CENTER (WHATSAPP & SMS)
    # --------------------------------------------------------------------------
    with tab_dispatch:
        st.markdown(
            f"""
            <div class="crop-meta-box" style="border-left: 4px solid #0284C7; margin-bottom: 1.2rem;">
                <h4 style="color: #0369A1; margin-bottom: 0.3rem;">📲 {t('dispatch_title', lang)}</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 0.5rem;">
                    {t('dispatch_desc', lang)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        loc_str = st.session_state.get("location_name", "Agricultural Field")
        leaf_r = st.session_state.get("leaf_result")

        if leaf_r and leaf_r.get("predicted_class"):
            d_crop = translate_crop_name(leaf_r.get("crop", "Unknown"), lang)
            d_disease = translate_disease_name(leaf_r.get("disease", "Unknown"), lang)
            c_conf = leaf_r.get("confidence", 0.0)

            st.markdown(
                f"""
                <div style="background:#FFFFFF;border:1.5px solid #CBD5E1;border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                    <div style="font-size:0.75rem;font-weight:700;color:#526E60;text-transform:uppercase;">Active Foliar Screening Diagnostic</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.3rem;">
                        <div style="font-size:1.3rem;font-weight:700;color:#123D2A;">{d_crop} — {d_disease}</div>
                        <div style="background:#EBF7EE;color:#1B543A;font-weight:700;padding:0.25rem 0.75rem;border-radius:15px;font-size:0.85rem;">Confidence: {c_conf*100:.1f}%</div>
                    </div>
                    <div style="font-size:0.85rem;color:#526E60;margin-top:0.3rem;">Location: <strong>{loc_str}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c_btn1, c_btn2 = st.columns(2)
            raw_c = leaf_r.get("crop", "Crop")
            raw_d = leaf_r.get("disease", "Disease")
            wa_dispatch_url = build_whatsapp_share_url(
                crop=raw_c, disease=raw_d,
                confidence_pct=c_conf * 100,
                location_name=loc_str,
                urgency="Review Required",
            )
            import urllib.parse
            default_sms_text = f"CropCare Kisan Alert: {d_crop} showing symptoms of {d_disease} ({c_conf*100:.0f}% confidence). Farm Location: {loc_str}. Please review recovery advisory."
            sms_dispatch_body = urllib.parse.quote(default_sms_text)

            with c_btn1:
                st.markdown(
                    f"""
                    <a href="{wa_dispatch_url}" target="_blank"
                       style="display:flex;align-items:center;justify-content:center;gap:0.5rem;background:#25D366;color:#FFFFFF;font-weight:700;font-size:0.95rem;padding:0.8rem 1rem;border-radius:10px;text-decoration:none;box-shadow:0 3px 10px rgba(37,211,102,0.25);margin-bottom:0.8rem;">
                        📲 {t('whatsapp_btn', lang)}
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            with c_btn2:
                st.markdown(
                    f"""
                    <a href="sms:?body={sms_dispatch_body}"
                       style="display:flex;align-items:center;justify-content:center;gap:0.5rem;background:#0284C7;color:#FFFFFF;font-weight:700;font-size:0.95rem;padding:0.8rem 1rem;border-radius:10px;text-decoration:none;box-shadow:0 3px 10px rgba(2,132,199,0.25);margin-bottom:0.8rem;">
                        {t('sms_btn', lang)}
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

            # Custom Interactive SMS & WhatsApp Composer
            with st.expander("📝 Custom Message Composer & Recipient Selector", expanded=True):
                st.markdown("##### Dispatch Settings")
                r_col1, r_col2 = st.columns([1, 1])
                with r_col1:
                    recipient_type = st.selectbox(
                        "Select Recipient Directory:",
                        [
                            "District Extension Officer / KVK",
                            "Kisan Call Center (Toll-Free 1800-180-1551)",
                            "Local Village Krishi Mitra / Panchayat",
                            "Custom Mobile Number",
                        ],
                        key="tab_dispatch_recipient_type",
                    )
                with r_col2:
                    if recipient_type == "Custom Mobile Number":
                        phone_input = st.text_input("Enter Mobile Number (with country code):", value="+91 ", key="custom_phone_input_tab")
                    elif "1800-180-1551" in recipient_type:
                        phone_input = "18001801551"
                        st.text_input("Helpline Number:", value=phone_input, disabled=True, key="kcc_phone_display_tab")
                    else:
                        phone_input = ""
                        st.info("Uses default extension directory or device SMS app.")

                custom_msg = st.text_area(
                    "Edit Alert Message Before Sending:",
                    value=default_sms_text,
                    height=90,
                    key="tab_dispatch_msg_textarea",
                )

                encoded_custom_msg = urllib.parse.quote(custom_msg)
                sms_link_custom = f"sms:{phone_input}?body={encoded_custom_msg}" if phone_input else f"sms:?body={encoded_custom_msg}"
                wa_phone_param = f"&phone={phone_input.replace('+', '').replace(' ', '')}" if phone_input and phone_input != "18001801551" else ""
                wa_link_custom = f"https://api.whatsapp.com/send?text={encoded_custom_msg}{wa_phone_param}"

                act_col1, act_col2, act_col3 = st.columns(3)
                with act_col1:
                    st.markdown(
                        f"""
                        <a href="{sms_link_custom}"
                           style="display:flex;align-items:center;justify-content:center;gap:0.4rem;background:#0284C7;color:#FFFFFF;font-weight:700;font-size:0.85rem;padding:0.6rem 0.8rem;border-radius:8px;text-decoration:none;">
                            📱 Send SMS Now
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )
                with act_col2:
                    st.markdown(
                        f"""
                        <a href="{wa_link_custom}" target="_blank"
                           style="display:flex;align-items:center;justify-content:center;gap:0.4rem;background:#25D366;color:#FFFFFF;font-weight:700;font-size:0.85rem;padding:0.6rem 0.8rem;border-radius:8px;text-decoration:none;">
                            📲 Send WhatsApp Now
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )
                with act_col3:
                    st.markdown(
                        """
                        <a href="tel:18001801551"
                           style="display:flex;align-items:center;justify-content:center;gap:0.4rem;background:#E11D48;color:#FFFFFF;font-weight:700;font-size:0.85rem;padding:0.6rem 0.8rem;border-radius:8px;text-decoration:none;">
                            📞 Call KCC 1800-180-1551
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )

        else:
            st.info("🍃 No active leaf screening found in this session. Select a sample below to simulate a real-time foliar alert, or upload a leaf photo in Tab 1.")
            st.markdown("##### ⚡ 1-Click Quick-Test Dispatch with Field Samples:")
            ts_c1, ts_c2, ts_c3 = st.columns(3)
            with ts_c1:
                if st.button("🍅 Test Tomato Blight Alert", key="tab_disp_btn_tom", use_container_width=True):
                    sample_path = "assets/samples/tomato_early_blight.jpg"
                    if os.path.exists(sample_path):
                        from PIL import Image
                        img = Image.open(sample_path)
                        st.session_state.leaf_image = img
                        st.session_state.selected_crop_type = "Tomato"
                        st.session_state.leaf_result = predict_image(img, crop_filter="Tomato")
                        st.rerun()
            with ts_c2:
                if st.button("🥔 Test Potato Blight Alert", key="tab_disp_btn_pot", use_container_width=True):
                    sample_path = "assets/samples/potato_early_blight.jpg"
                    if os.path.exists(sample_path):
                        from PIL import Image
                        img = Image.open(sample_path)
                        st.session_state.leaf_image = img
                        st.session_state.selected_crop_type = "Potato"
                        st.session_state.leaf_result = predict_image(img, crop_filter="Potato")
                        st.rerun()
            with ts_c3:
                if st.button("🌿 Test Healthy Crop Alert", key="tab_disp_btn_hlth", use_container_width=True):
                    sample_path = "assets/samples/tomato_healthy.jpg"
                    if os.path.exists(sample_path):
                        from PIL import Image
                        img = Image.open(sample_path)
                        st.session_state.leaf_image = img
                        st.session_state.selected_crop_type = "Tomato"
                        st.session_state.leaf_result = predict_image(img, crop_filter="Tomato")
                        st.rerun()

            st.markdown("---")
            st.markdown("##### 📞 National Kisan Emergency Hotlines")
            kcc_c1, kcc_c2 = st.columns(2)
            with kcc_c1:
                st.markdown(
                    """
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:0.9rem;">
                        <div style="font-weight:700;color:#0F172A;font-size:0.9rem;">📞 Kisan Call Center (KCC)</div>
                        <div style="font-size:0.8rem;color:#64748B;margin:0.2rem 0 0.5rem 0;">Toll-Free, 22 Official Languages, 6:00 AM - 10:00 PM</div>
                        <a href="tel:18001801551" style="display:inline-block;background:#166534;color:#FFFFFF;padding:0.4rem 0.8rem;border-radius:6px;font-weight:700;font-size:0.82rem;text-decoration:none;">Dial 1800-180-1551</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kcc_c2:
                st.markdown(
                    """
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:0.9rem;">
                        <div style="font-weight:700;color:#0F172A;font-size:0.9rem;">📱 Krishi Vigyan Kendra (KVK) Portal</div>
                        <div style="font-size:0.8rem;color:#64748B;margin:0.2rem 0 0.5rem 0;">Indian Council of Agricultural Research (ICAR)</div>
                        <a href="https://kvk.icar.gov.in" target="_blank" style="display:inline-block;background:#0284C7;color:#FFFFFF;padding:0.4rem 0.8rem;border-radius:6px;font-weight:700;font-size:0.82rem;text-decoration:none;">Open KVK Portal ↗</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------------------------
    # TAB 3: CONTINUAL LEARNING & FIELD ADAPTATION STUDIO
    # --------------------------------------------------------------------------
    with tab_studio:
        st.markdown(
            """
            <div class="crop-meta-box" style="border-left: 4px solid var(--forest-mid);">
                <h4 style="color: var(--forest-dark); margin-bottom: 0.3rem;">🧠 Continual Learning & Agronomist Memory Bank</h4>
                <p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                    Encountered an unmonitored crop (e.g. <strong>Rice, Wheat, Cotton</strong>) or a new visual pathology?
                    Upload reference photographs here. CropCare extracts 1024-dimensional normalized neural feature embeddings
                    and registers them into the dynamic memory bank for instant few-shot field recognition.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Hackathon Demonstration & Presets
        st.markdown("##### ⚡ Quick Hackathon Demonstration & Presets:")
        d_col1, d_col2 = st.columns([1.5, 1])
        with d_col1:
            if st.button("🚀 1-Click Demo: Teach Model Rice (Brown Spot)", key="btn_demo_rice", type="secondary"):
                sample_rice = "data/sample_leaves/sample_rice_brown_spot.png"
                if os.path.exists(sample_rice):
                    r_img = Image.open(sample_rice)
                    reg_out = register_field_leaf(
                        r_img,
                        crop="Rice",
                        disease="Brown Spot",
                        notes="Oryza sativa lowland paddy specimen with fungal spindle lesions.",
                    )
                    if reg_out.get("success"):
                        st.success(
                            f"✅ Registered sample #{reg_out.get('total_samples', 1)}! "
                            f"Trained neural memory bank for **Rice — Brown Spot**! "
                            f"Now test the '🌾 Rice (Continual Demo)' in Diagnostic Leaf Screening to see immediate detection."
                        )
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {reg_out.get('error')}")
                else:
                    st.error("Demo image not found at data/sample_leaves/sample_rice_brown_spot.png")

        with d_col2:
            if st.button("🗑️ Reset Field Memory Bank", key="btn_reset_mem"):
                mem_path = "data/field_memory/memory_bank.json"
                if os.path.exists(mem_path):
                    try:
                        os.remove(mem_path)
                        st.info("Field memory bank reset successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error resetting memory bank: {e}")
                else:
                    st.info("Memory bank is already empty.")

        st_col_left, st_col_right = st.columns([1, 1], gap="large")

        with st_col_left:
            st.markdown("##### 📥 Register New Reference Sample")
            train_file = st.file_uploader(
                "Upload reference leaf photo (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                key="studio_uploader",
                help="Clear leaf photo with distinct disease symptoms.",
            )

            train_crop = st.text_input(
                "Crop Species Name:",
                placeholder="e.g. Rice, Cotton, Wheat, Sugarcane",
                key="train_crop_input",
            )
            train_disease = st.text_input(
                "Pathology / Condition Label:",
                placeholder="e.g. Blast, Leaf Curl, Rust, Bacterial Spot",
                key="train_disease_input",
            )
            train_notes = st.text_area(
                "Field Notes / Agronomic Observations:",
                placeholder="e.g. Spindle lesions observed during high humidity conditions.",
                key="train_notes_input",
            )

            if st.button("💾 Register Sample & Adapt Neural Core ⚡", type="primary", use_container_width=True):
                if train_file is None:
                    st.error("Please upload a leaf photograph before registering.")
                elif not train_crop.strip() or not train_disease.strip():
                    st.error("Please provide both Crop Species Name and Pathology/Condition Label.")
                else:
                    try:
                        t_img = Image.open(train_file)
                        reg_out = register_field_leaf(
                            t_img,
                            crop=train_crop.strip(),
                            disease=train_disease.strip(),
                            notes=train_notes.strip(),
                        )
                        if reg_out.get("success"):
                            st.success(
                                f"✅ Successfully registered sample #{reg_out.get('total_samples', 1)}! "
                                f"Neural feature memory bank updated for **{train_crop} — {train_disease}**."
                            )
                            st.rerun()
                        else:
                            st.error(f"Registration failed: {reg_out.get('error')}")
                    except Exception as reg_err:
                        st.error(f"Registration error: {reg_err}")

        with st_col_right:
            st.markdown("##### 🗄️ Active Memory Bank Registry")
            reg_leaves = get_registered_field_leaves()

            st.metric("Total Field Registered Samples", len(reg_leaves))

            if reg_leaves:
                for r_item in reversed(reg_leaves[-5:]):
                    st.markdown(
                        f"""
                        <div style="background: #FDFEFB; border: 1px solid #DFE7DB; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.7rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 700; color: #123D2A; font-size: 0.95rem;">
                                    🌾 {r_item.get('crop')} — <span style="color: #1B543A;">{r_item.get('disease')}</span>
                                </div>
                                <span class="crop-badge-pill" style="font-size: 0.72rem;">1024-dim Vector</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #526E60; margin-top: 0.3rem;">
                                {r_item.get('notes') or 'Verified field specimen.'}
                            </div>
                            <div style="font-size: 0.72rem; color: #8FA69B; margin-top: 0.2rem;">
                                Logged: {r_item.get('timestamp')[:19].replace('T', ' ')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No custom field samples registered yet. "
                    "Upload and register a sample on the left to teach the platform any new crop or disease!"
                )

    # --------------------------------------------------------------------------
    # TAB 3: MONITORED CROPS LIBRARY
    # --------------------------------------------------------------------------
    with tab_library:
        st.markdown("#### 📚 Complete Catalog of 14 Monitored Crops & 38 Pathologies")
        st.caption("CropCare is pre-trained to differentiate among the following conditions across 14 major agricultural crops:")
        crops_catalog = get_supported_crops()
        cat_cols = st.columns(2)
        c_idx = 0
        for k_raw, meta in sorted(crops_catalog.items(), key=lambda x: x[1]["display_name"]):
            with cat_cols[c_idx % 2]:
                st.markdown(
                    f"""
                    <div style="background: #FDFEFB; border: 1px solid #DFE7DB; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;">
                        <div style="font-weight: 700; color: #123D2A; font-size: 0.98rem;">
                            {meta['icon']} {meta['display_name']} <span style="font-size: 0.78rem; font-weight: 600; color: #526E60;">({len(meta['diseases'])} conditions)</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #2A7352; margin-top: 0.35rem; line-height: 1.4;">
                            {' • '.join(meta['diseases'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            c_idx += 1

    # --------------------------------------------------------------------------
    # TAB 4: GRAD-CAM EXPLAINABILITY
    # --------------------------------------------------------------------------
    with tab_gradcam:
        st.markdown(
            """
            <div class="crop-meta-box" style="border-left: 4px solid #7C3AED;">
                <h4 style="color: #123D2A; margin-bottom: 0.3rem;">🔬 Grad-CAM Explainability — Neural Attention Heatmap</h4>
                <p style="font-size: 0.88rem; color: #526E60; margin-bottom: 0.3rem;">
                    Gradient-weighted Class Activation Mapping shows <strong>which leaf regions</strong> the AI used to make its diagnosis.
                    Red/warm zones = high-attention lesion areas. Blue/cool zones = low influence.
                    This transparency layer lets agronomists verify predictions are based on authentic pathology, not image artifacts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.get("leaf_image") is not None and HAS_GRADCAM:
            with st.spinner("⚡ Computing Grad-CAM neural attention heatmap..."):
                cam_result = generate_gradcam(
                    image=st.session_state.leaf_image,
                    alpha=0.45,
                )
            if cam_result.get("overlay_image"):
                gcam_c1, gcam_c2 = st.columns(2)
                with gcam_c1:
                    st.image(st.session_state.leaf_image, caption="📷 Original Leaf Photograph", use_container_width=True)
                with gcam_c2:
                    st.image(cam_result["overlay_image"], caption="🔥 Grad-CAM Heatmap Overlay", use_container_width=True)
                # Save for report
                st.session_state["cam_overlay"] = cam_result["overlay_image"]
                st.markdown(
                    f"""
                    <div style="background:#F4F7F2;border-radius:10px;padding:0.9rem 1.2rem;margin-top:0.7rem;">
                        <div style="font-weight:700;color:#123D2A;margin-bottom:0.3rem;">🧠 Neural Attention Analysis</div>
                        <div style="font-size:0.88rem;color:#334D3F;">{cam_result.get('focus_description', 'Heatmap generated.')}</div>
                        <div style="margin-top:0.5rem;font-size:0.8rem;color:#526E60;">
                            Peak Activation Intensity: <strong>{cam_result.get('peak_intensity', 0):.2f}</strong>
                            {'&nbsp;&nbsp;|&nbsp;&nbsp;<em>Note: Optical edge saliency fallback used — PyTorch model gradient hooks unavailable.</em>' if not cam_result.get('success') else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif not HAS_GRADCAM:
            st.warning("Grad-CAM module not available. Ensure `src/cropcare/explainability.py` is present.")
        else:
            st.info("Upload a leaf image in the **Diagnostic Leaf Screening** tab first, then return here to see the Grad-CAM explainability heatmap.")

    # --------------------------------------------------------------------------
    # TAB 5: MODEL EVALUATION & ERROR ANALYSIS (FOR JUDGES)
    # --------------------------------------------------------------------------
    with tab_evaluation:
        st.markdown(
            """
            <div class="crop-meta-box" style="border-left: 4px solid var(--forest-dark); margin-bottom: 1.2rem;">
                <h4 style="color: var(--forest-dark); margin-bottom: 0.3rem;">📊 Empirical Model Evaluation & Error Analysis</h4>
                <p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0;">
                    Rigorous evaluation across 10,861 held-out test images (PlantVillage Benchmark).
                    Reporting overall metrics, class-wise errors, biological confusion pairs, and stress perturbations
                    for <strong>lighting variations</strong>, <strong>sensor blur/quality</strong>, and <strong>unseen conditions</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_summary = get_benchmark_summary()
        e_kpi1, e_kpi2, e_kpi3, e_kpi4, e_kpi5 = st.columns(5)
        with e_kpi1:
            st.metric("Overall Accuracy", f"{b_summary['overall_accuracy']}%", "Top-1 Precision")
        with e_kpi2:
            st.metric("Macro F1-Score", f"{b_summary['macro_f1']}%", "38 Classes")
        with e_kpi3:
            st.metric("Top-3 Accuracy", f"{b_summary['top3_accuracy']}%", "Consensus")
        with e_kpi4:
            st.metric("CPU Latency", f"{b_summary['avg_cpu_latency_ms']} ms", "Edge Optimized")
        with e_kpi5:
            st.metric("Footprint", f"{b_summary['model_footprint_mb']} MB", "MobileNetV3")

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        eval_sub1, eval_sub2, eval_sub3, eval_sub4 = st.tabs([
            "🎯 1. Crop Performance Breakdown",
            "⚠️ 2. Class Confusion & Errors",
            "🔬 3. Lighting & Blur Stress Tests",
            "🧠 4. Unseen Conditions & OOD Defense",
        ])

        with eval_sub1:
            st.markdown("##### 🌾 Class & Crop-Wise Empirical Performance (10,861 Test Images)")
            crop_bench = get_crop_wise_benchmarks()

            selected_eval_crop = st.selectbox(
                "Filter Performance by Crop Species:",
                options=["All Crops Overview"] + list(crop_bench.keys()),
                key="eval_crop_filter",
            )

            if selected_eval_crop == "All Crops Overview":
                eval_df_data = []
                for crp, dt in crop_bench.items():
                    eval_df_data.append({
                        "Crop Species": crp.replace("_", " "),
                        "Monitored Classes": dt["num_classes"],
                        "Test Samples": dt["test_samples"],
                        "Accuracy (%)": f"{dt['accuracy']:.2f}%",
                        "Precision (%)": f"{dt['precision']:.2f}%",
                        "Recall (%)": f"{dt['recall']:.2f}%",
                        "F1-Score (%)": f"{dt['f1_score']:.2f}%",
                        "Primary Error / Confusion Mode": dt["top_confusion"],
                    })
                st.dataframe(pd.DataFrame(eval_df_data), use_container_width=True, hide_index=True)
            else:
                cd = crop_bench[selected_eval_crop]
                st.markdown(
                    f"""
                    <div style="background: #F8FAF7; border: 1.5px solid #CCD7D0; border-radius: 14px; padding: 1.2rem; margin-top: 0.6rem;">
                        <div style="font-size: 1.2rem; font-weight: 700; color: #123D2A; margin-bottom: 0.4rem;">
                            🌾 {selected_eval_crop.replace('_', ' ')} Performance Profile
                        </div>
                        <div style="display: flex; gap: 2rem; margin-bottom: 0.8rem; flex-wrap: wrap;">
                            <div><strong>Test Samples:</strong> {cd['test_samples']}</div>
                            <div><strong>Accuracy:</strong> {cd['accuracy']}%</div>
                            <div><strong>Precision:</strong> {cd['precision']}%</div>
                            <div><strong>Recall:</strong> {cd['recall']}%</div>
                            <div><strong>F1-Score:</strong> {cd['f1_score']}%</div>
                        </div>
                        <div style="font-size: 0.88rem; color: #1B543A; background: #EBF8D3; padding: 0.6rem 0.9rem; border-radius: 8px;">
                            ⚠️ <strong>Top Confusion Boundary:</strong> {cd['top_confusion']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with eval_sub2:
            st.markdown("##### 🔍 Critical Class-Wise Errors & Biological Confusion Analysis")
            st.caption("Addressing hackathon requirement: detailed report on class confusion and morphological root causes:")

            conf_pairs = get_top_confusion_pairs()
            for pair in conf_pairs:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid rgba(18, 61, 42, 0.12); border-left: 4px solid #D97706; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.85rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; flex-wrap: wrap; gap: 0.5rem;">
                            <div style="font-weight: 700; color: #123D2A; font-size: 0.96rem;">
                                🎯 <strong>{pair['true_class'].replace('___', ' — ').replace('_', ' ')}</strong> 
                                <span style="color: #9A3412;">⮂ Confused with ⮂</span> 
                                <strong>{pair['predicted_class'].replace('___', ' — ').replace('_', ' ')}</strong>
                            </div>
                            <span style="background: #FEF3C7; color: #92400E; font-size: 0.8rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 12px;">
                                {pair['error_pct']:.2f}% Error Rate
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #374151; margin-bottom: 0.4rem; line-height: 1.45;">
                            <strong>Biological Root Cause:</strong> {pair['biological_root_cause']}
                        </div>
                        <div style="font-size: 0.82rem; color: #15803D; line-height: 1.45;">
                            🛡️ <strong>CropCare Engineering Mitigation:</strong> {pair['mitigation_strategy']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div style="background: #EBF8D3; border: 1.5px solid #A3D95B; border-radius: 12px; padding: 1rem 1.2rem; margin-top: 1rem;">
                    <div style="font-weight: 700; color: #123D2A; font-size: 0.95rem; margin-bottom: 0.3rem;">
                        ⚡ How CropCare Subspace Isolation Eliminates 100% of Inter-Crop Confusions
                    </div>
                    <div style="font-size: 0.85rem; color: #1B543A; line-height: 1.5;">
                        Standard computer vision architectures compute a global 38-way softmax, which frequently confuses cross-species Solanaceae diseases (e.g. Potato Early Blight mistaken for Tomato Early Blight).
                        By allowing the farmer or agronomist to select their active crop, CropCare masks out irrelevant species logits, reducing inter-crop error to <strong>0.0%</strong>!
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with eval_sub3:
            st.markdown("##### 🔬 Environmental Perturbations: Lighting & Image Quality Degradation")
            st.caption("Empirical ablation testing model vulnerability to illumination shifts, shadows, and low-res camera optics:")

            ablation_data = get_stress_test_ablation()

            # 1. Lighting Variations
            light_exp = ablation_data["lighting_variations"]
            st.markdown(f"###### ☀️ {light_exp['title']}")
            st.caption(light_exp['description'])

            l_df = []
            for row in light_exp["experiments"]:
                l_df.append({
                    "Lighting Condition Tested": row["condition"],
                    "Baseline Model Accuracy": f"{row['baseline_accuracy']}%",
                    "With CropCare 3-View TTA": f"{row['with_cropcare_tta']}%",
                    "Net Accuracy Recovery": row["gain"],
                    "Field Resilience Outcome": row["status"],
                })
            st.dataframe(pd.DataFrame(l_df), use_container_width=True, hide_index=True)
            st.info(f"💡 **Key Finding:** {light_exp['key_takeaway']}")

            # 2. Image Blur & Sensor Quality
            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            blur_exp = ablation_data["image_quality_blur"]
            st.markdown(f"###### 📷 {blur_exp['title']}")
            st.caption(blur_exp['description'])

            b_df = []
            for row in blur_exp["experiments"]:
                b_df.append({
                    "Image Degradation Condition": row["condition"],
                    "Baseline Model Accuracy": f"{row['baseline_accuracy']}%",
                    "With Subspace Isolation": f"{row['with_subspace_isolation']}%",
                    "Net Accuracy Recovery": row["gain"],
                    "Field Resilience Outcome": row["status"],
                })
            st.dataframe(pd.DataFrame(b_df), use_container_width=True, hide_index=True)
            st.info(f"💡 **Key Finding:** {blur_exp['key_takeaway']}")

        with eval_sub4:
            st.markdown("##### 🧠 Unseen Conditions & Out-of-Distribution (OOD) Defense")
            st.caption("Mitigating dangerous overconfidence when presented with unmonitored crops or novel field anomalies:")

            ood_exp = ablation_data["unseen_conditions_ood"]
            for ood_case in ood_exp["experiments"]:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1.5px solid #CCD7D0; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;">
                        <div style="font-weight: 700; color: #123D2A; font-size: 0.95rem; margin-bottom: 0.4rem;">
                            🍂 Condition: {ood_case['condition']}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 0.5rem; font-size: 0.84rem;">
                            <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 0.6rem 0.8rem; color: #991B1B;">
                                <strong>Standard Deep Learning Failure Mode:</strong><br>{ood_case['standard_cnn_behavior']}
                            </div>
                            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.6rem 0.8rem; color: #166534;">
                                <strong>CropCare Safeguard & Triage:</strong><br>{ood_case['cropcare_behavior']}
                            </div>
                        </div>
                        <div style="font-size: 0.82rem; color: #1E40AF; background: #EFF6FF; padding: 0.5rem 0.8rem; border-radius: 8px;">
                            🧠 <strong>Adaptive Solution:</strong> {ood_case['solution_mechanism']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.success(f"🏆 **Engineering Synthesis:** {ood_exp['key_takeaway']}")

        # Download Full Evaluation Report for Judges
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        try:
            with open("reports/model_metrics.json", "r", encoding="utf-8") as rf:
                report_json_str = rf.read()
            st.download_button(
                label="📥 Download Complete Model Benchmark & Metrics Report (JSON)",
                data=report_json_str,
                file_name="cropcare_model_evaluation_metrics.json",
                mime="application/json",
                help="Download the complete benchmark suite, class-wise errors, and perturbation ablation tables.",
            )
        except Exception:
            pass

    # Guided Screen 2 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button(t('btn_back_welcome', lang), type="secondary"):
            st.session_state.screen = 1
            st.rerun()
    with n_col2:
        if st.button(t('btn_goto_climate', lang), type="primary"):
            st.session_state.screen = 3
            st.rerun()


# ==============================================================================
# SCREEN 3: CLIMATE INSIGHTS
# ==============================================================================
elif st.session_state.screen == 3:
    st.markdown(
        f"""
        <div class="surface-card">
            <h3>{t('screen3_title', lang)}</h3>
            <p class="card-desc">
                {t('screen3_desc', lang)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("climate_query_form"):
        # Farmer-friendly location picker (if available)
        if HAS_LOCATION and TOP_PRESET_LOCATIONS:
            preset_options = ["— Select a farming region preset —"] + list(TOP_PRESET_LOCATIONS.keys())
            chosen_preset = st.selectbox(t('preset_loc_label', lang), preset_options, key="preset_loc")
            if chosen_preset != "— Select a farming region preset —":
                preset_lat, preset_lon = TOP_PRESET_LOCATIONS[chosen_preset]
                st.session_state.climate_lat = preset_lat
                st.session_state.climate_lon = preset_lon

        c_input1, c_input2, c_input3 = st.columns([1, 1, 1])
        with c_input1:
            lat = st.number_input(
                t('lat_label', lang),
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.climate_lat),
                step=0.0001,
                format="%.4f",
            )
        with c_input2:
            lon = st.number_input(
                t('lon_label', lang),
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.climate_lon),
                step=0.0001,
                format="%.4f",
            )
        with c_input3:
            window = st.selectbox(
                t('history_window_label', lang),
                options=[7, 14, 30],
                index=[7, 14, 30].index(st.session_state.climate_days)
                if st.session_state.climate_days in [7, 14, 30]
                else 2,
            )

        submit_weather = st.form_submit_button(t('btn_fetch_climate', lang), use_container_width=True)

    if submit_weather:
        st.session_state.climate_lat = lat
        st.session_state.climate_lon = lon
        st.session_state.climate_days = window

        with st.spinner(f"Querying Open-Meteo for {window}-day climate baseline..."):
            try:
                env_data = get_environment(location=(lat, lon), days=window)
                st.session_state.climate_result = env_data
            except Exception as e:
                st.error(f"Climate API execution error: {e}")

    env = st.session_state.climate_result

    if env is not None:
        loc_str = env.get("location", f"({st.session_state.climate_lat}, {st.session_state.climate_lon})")
        source = env.get("source", "Open-Meteo")
        st.markdown(f"**Location:** `{loc_str}` | **Provider:** `{source}`")

        # Warnings
        w_list = env.get("warnings", [])
        if w_list:
            with st.expander("⚠️ Climate Engine Advisories", expanded=False):
                for w in w_list:
                    st.caption(f"• {w}")

        # Metrics Row
        recent = env.get("recent") or {}
        st.markdown(f"#### {t('recent_summary', lang)} ({st.session_state.climate_days} Days)")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Mean Temp", format_metric(recent.get("mean_temperature_c"), "°C"))
        with m2:
            st.metric("Max Temp", format_metric(recent.get("max_temperature_c"), "°C"))
        with m3:
            st.metric("Rainfall", format_metric(recent.get("rainfall_mm"), "mm"))
        with m4:
            st.metric("Humidity", format_metric(recent.get("mean_humidity_pct"), "%"))
        with m5:
            st.metric("Ref ET0", format_metric(recent.get("et0_mm"), "mm"))

        # Anomalies
        st.markdown(f"#### {t('anomalies_title', lang)}")
        st.caption(
            "Anomalies indicate how recent weather compares to historical averages for this "
            "time of year. They describe environmental conditions, not disease evidence."
        )

        anom = env.get("anomalies") or {}
        a1, a2, a3 = st.columns(3)
        with a1:
            t_anom = anom.get("temperature_c")
            st.metric(
                "Temperature Anomaly",
                format_metric(t_anom, "°C"),
                delta=f"{t_anom:+.1f} °C" if t_anom is not None else None,
            )
        with a2:
            r_mm = anom.get("rainfall_mm")
            st.metric(
                "Rainfall Anomaly (mm)",
                format_metric(r_mm, "mm"),
                delta=f"{r_mm:+.1f} mm" if r_mm is not None else None,
            )
        with a3:
            r_pct = anom.get("rainfall_pct")
            st.metric(
                "Rainfall Anomaly (%)",
                format_metric(r_pct, "%"),
                delta=f"{r_pct:+.1f} %" if r_pct is not None else None,
            )

        # Charts
        daily = env.get("daily_series") or []
        if daily:
            st.markdown("#### Daily Trends")
            df = pd.DataFrame(daily)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").set_index("date")

                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    t_cols = [c for c in ["temperature_mean_c", "temperature_max_c"] if c in df.columns]
                    if t_cols and df[t_cols].notna().any().any():
                        st.write("**Daily Temperatures (°C)**")
                        st.line_chart(df[t_cols].rename(columns={
                            "temperature_mean_c": "Mean Temp (°C)",
                            "temperature_max_c": "Max Temp (°C)",
                        }))
                with ch_col2:
                    if "precipitation_mm" in df.columns and df["precipitation_mm"].notna().any():
                        st.write("**Precipitation (mm)**")
                        st.bar_chart(df[["precipitation_mm"]].rename(columns={
                            "precipitation_mm": "Rainfall (mm)",
                        }))
    else:
        st.info("Submit coordinates above to retrieve climate intelligence and anomaly calculations.")

    # ==================================================================
    # BIOMETEOROLOGICAL ALIGNMENT METER (Requires both leaf + climate data)
    # ==================================================================
    if HAS_ALIGNMENT and st.session_state.get("climate_result") and st.session_state.get("leaf_result"):
        leaf_r = st.session_state.leaf_result
        disease_al = leaf_r.get("disease", "")
        align_res = compute_climate_alignment(
            disease=disease_al,
            climate_result=st.session_state.climate_result,
        )
        if align_res:
            st.markdown("---")
            st.markdown(f"#### {t('alignment_title', lang)}")
            al_c1, al_c2 = st.columns([1, 2])
            with al_c1:
                st.markdown(
                    f"""
                    <div style="background:#FFFFFF;border:1px solid #D6E0D9;border-radius:16px;padding:1.5rem;text-align:center;">
                      <div style="font-size:0.8rem;font-weight:700;color:#526E60;text-transform:uppercase;">Alignment Index</div>
                      <div style="font-family:'Space Grotesk',sans-serif;font-size:2.8rem;font-weight:800;color:{align_res['badge_color']};">{align_res['score']}<span style="font-size:1.2rem;color:#526E60;">/100</span></div>
                      <div style="display:inline-block;background:{align_res['badge_color']}1A;color:{align_res['badge_color']};font-weight:700;font-size:0.85rem;padding:0.35rem 0.85rem;border-radius:20px;border:1px solid {align_res['badge_color']}33;margin-top:0.5rem;">{align_res['badge']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with al_c2:
                st.markdown(
                    f"""
                    <div style="background:#F4F7F2;border-radius:12px;padding:1rem 1.2rem;">
                      <div style="font-weight:700;color:#123D2A;margin-bottom:0.4rem;">{align_res.get('category','')} — {align_res.get('pathogen','')}</div>
                      <div style="font-size:0.88rem;color:#334D3F;line-height:1.55;">{align_res['explanation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if align_res.get("matched_factors"):
                    with st.expander("📊 Epidemiological Factor Analysis", expanded=False):
                        for f in align_res["matched_factors"]:
                            st.markdown(f"• {f}")
                if align_res.get("abiotic_stress_flag") and align_res.get("abiotic_mimic"):
                    st.warning(f"⚠️ Abiotic Stress Mimic Detected: {align_res['abiotic_mimic']}")
            # Store for report
            st.session_state["alignment_result"] = align_res

    # Guided Screen 3 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button(t('btn_back_leaf', lang), type="secondary"):
            st.session_state.screen = 2
            st.rerun()
    with n_col2:
        if st.button(t('btn_goto_summary', lang), type="primary"):
            st.session_state.screen = 4
            st.rerun()


# ==============================================================================
# SCREEN 4: EXECUTIVE SUMMARY & AGRONOMIC BRIEFING
# ==============================================================================
elif st.session_state.screen == 4:
    st.markdown(
        f"""
        <div class="surface-card">
            <h3>{t('screen4_title', lang)}</h3>
            <p class="card-desc">
                {t('screen4_desc', lang)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sum_col1, sum_col2 = st.columns([1, 1], gap="large")

    # Column 1: Leaf Screening Audit
    with sum_col1:
        st.markdown(f"#### {t('screening_record', lang)}")
        leaf_res = st.session_state.leaf_result

        if leaf_res and leaf_res.get("predicted_class"):
            raw_crop = leaf_res.get("crop", "Unknown")
            raw_disease = leaf_res.get("disease", "Unknown")
            crop = translate_crop_name(raw_crop, lang)
            disease = translate_disease_name(raw_disease, lang)
            conf = leaf_res.get("confidence", 0.0)
            uncert = leaf_res.get("uncertain", False)
            crop_filtered = leaf_res.get("crop_filter_applied", False)

            mode_label = "Targeted Crop Focus" if crop_filtered else "Multi-Crop Auto-Detect"

            st.markdown(
                f"""
                <div class="diagnostic-panel">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('screening_mode', lang)}</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #1B543A; margin-bottom: 0.6rem;">{mode_label}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('crop_identified', lang)}</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #123D2A; margin-bottom: 0.6rem;">{crop}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('visual_condition', lang)}</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #1B543A; margin-bottom: 0.6rem;">{disease}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('confidence_score', lang)}</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #123D2A;">{conf*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            mem_match = leaf_res.get("field_memory_match")
            if mem_match:
                st.info(f"🧠 Verified against Active Field Memory: {mem_match.get('crop')} — {mem_match.get('disease')}")

            sev_s4 = classify_disease_severity(
                st.session_state.leaf_image,
                disease_name=raw_disease,
                confidence=conf,
            )
            st.markdown(
                f"""
                <div style="background: #FDFEFB; border: 1px solid #DFE7DB; border-radius: 10px; padding: 0.75rem 1rem; margin-top: 0.6rem; margin-bottom: 0.6rem;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #123D2A;">
                        Pathological Stage: <span style="color: #1B543A;">{sev_s4['stage_name']}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #526E60; margin-top: 0.2rem;">
                        Severity: <strong>{sev_s4['severity_level']}</strong> | Foliar Coverage: <strong>{sev_s4['damage_percentage']:.1f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if uncert:
                st.warning("Flagged: Ambiguous / Low Confidence Screening. Field verification advised.")
            else:
                st.success("Flagged: High Optical Confidence Diagnosis")

            # Multilingual Voice Advisory Player
            nurture_s4 = get_nurturing_recommendations(
                crop=raw_crop,
                disease=raw_disease,
                severity_stage=sev_s4.get("stage_id", 1),
            )
            audio_text_s4 = translate_advisory_text(
                crop=crop,
                disease=disease,
                stage_name=sev_s4["stage_name"],
                damage_pct=sev_s4["damage_percentage"],
                irrigation_action=nurture_s4["irrigation"]["action"],
                nutrition_action=nurture_s4["nutrition"]["action"],
                sanitation_action=nurture_s4["sanitation"]["action"],
                bio_action=nurture_s4["bio_defense"]["action"],
                lang=lang,
            )
            audio_title_s4 = f"CropCare Advisory — {crop} / {disease}"
            audio_html_s4 = generate_audio_speech_html(
                title=audio_title_s4,
                text_to_speak=audio_text_s4,
                lang=lang,
            )
            st.markdown(f"##### 🎙️ {t('audio_listen_btn', lang)}")
            components.html(audio_html_s4, height=130)
        else:
            st.info("No leaf screening was completed in this session.")

    # Column 2: Climate Context Audit
    with sum_col2:
        st.markdown(f"#### {t('climate_record', lang)}")
        clim_res = st.session_state.climate_result

        if clim_res and clim_res.get("recent"):
            recent = clim_res.get("recent") or {}
            anom = clim_res.get("anomalies") or {}
            loc = clim_res.get("location", "Selected Coordinates")

            st.markdown(
                f"""
                <div class="diagnostic-panel">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Location Audited</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #123D2A; margin-bottom: 0.6rem;">{loc}</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Window Observed</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #1B543A; margin-bottom: 0.6rem;">{st.session_state.climate_days} Days via Open-Meteo</div>
                    <div style="display: flex; gap: 1.5rem;">
                        <div>
                            <div style="font-size: 0.75rem; color: #526E60;">Mean Temp</div>
                            <div style="font-weight: 700; color: #123D2A;">{format_metric(recent.get('mean_temperature_c'), '°C')}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #526E60;">Rainfall</div>
                            <div style="font-weight: 700; color: #123D2A;">{format_metric(recent.get('rainfall_mm'), 'mm')}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #526E60;">Temp Anomaly</div>
                            <div style="font-weight: 700; color: #123D2A;">{format_metric(anom.get('temperature_c'), '°C')}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No climate data was queried in this session.")

    # ==================================================================
    # FIELD HEALTH CARD DOWNLOAD (Full Agronomic Report)
    # ==================================================================
    if HAS_REPORT and st.session_state.get("leaf_result"):
        st.markdown("---")
        st.markdown(f"#### {t('download_report_btn', lang)}")
        with st.spinner("Generating agronomic report..."):
            report_html = generate_field_report_html(
                leaf_result=st.session_state.get("leaf_result"),
                climate_result=st.session_state.get("climate_result"),
                alignment_result=st.session_state.get("alignment_result"),
                advisory_result=None,
                leaf_image=st.session_state.get("leaf_image"),
                cam_image=st.session_state.get("cam_overlay"),
                farmer_name="Lead Farmer",
                farm_location=st.session_state.get("location_name", "Agricultural Field"),
            )
        st.download_button(
            label=t('download_report_btn', lang),
            data=report_html.encode("utf-8"),
            file_name="CropCare_Field_Report.html",
            mime="text/html",
            use_container_width=True,
        )

        # Screen 4 Kisan Emergency Dispatch
        s4_loc = st.session_state.get("location_name", "Agricultural Field")
        s4_crop_raw = leaf_res.get("crop", "Crop")
        s4_disease_raw = leaf_res.get("disease", "Condition")
        wa_url_s4 = build_whatsapp_share_url(
            crop=s4_crop_raw, disease=s4_disease_raw,
            confidence_pct=conf * 100,
            location_name=s4_loc,
            urgency=sev_s4["urgency"],
        )
        import urllib.parse
        sms_text_s4 = urllib.parse.quote(
            f"CropCare Kisan Executive Report: {crop} diagnosed with {disease} ({conf*100:.0f}% confidence, {sev_s4['severity_level']}). Location: {s4_loc}. Please advise."
        )
        s4_disp1, s4_disp2 = st.columns(2)
        with s4_disp1:
            st.markdown(
                f"""
                <a href="{wa_url_s4}" target="_blank"
                   style="display:flex;align-items:center;justify-content:center;gap:0.5rem;background:#25D366;color:#FFFFFF;font-weight:700;font-size:0.88rem;padding:0.65rem 1rem;border-radius:10px;text-decoration:none;box-shadow:0 3px 10px rgba(37,211,102,0.25);margin-top:0.4rem;">
                    📲 {t('whatsapp_btn', lang)}
                </a>
                """,
                unsafe_allow_html=True,
            )
        with s4_disp2:
            st.markdown(
                f"""
                <a href="sms:?body={sms_text_s4}"
                   style="display:flex;align-items:center;justify-content:center;gap:0.5rem;background:#0284C7;color:#FFFFFF;font-weight:700;font-size:0.88rem;padding:0.65rem 1rem;border-radius:10px;text-decoration:none;box-shadow:0 2px 10px rgba(2,132,199,0.25);margin-top:0.4rem;">
                    {t('sms_btn', lang)}
                </a>
                """,
                unsafe_allow_html=True,
            )

    # Global Agronomic Synthesis Advisory
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>Critical Agronomic Safeguards & Methodology Notice:</strong><br>
            • <strong>Independent Modalities:</strong> Leaf image screening and climate anomaly tracking operate as independent observational streams. Climate anomalies reflect macro-environmental differences and do not prove biotic plant disease.<br>
            • <strong>Screening, Not Diagnosis:</strong> CropCare provides initial decision-support screening based on benchmark laboratory datasets. Never apply chemical interventions (fungicides, pesticides) without on-site agronomic verification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Guided Screen 4 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button(t('btn_back_climate', lang), type="secondary"):
            st.session_state.screen = 3
            st.rerun()
    with n_col2:
        if st.button(t('btn_restart', lang), type="primary"):
            st.session_state.screen = 1
            st.session_state.leaf_image = None
            st.session_state.leaf_image_id = None
            st.session_state.leaf_result = None
            st.session_state.climate_result = None
            st.session_state.selected_crop_type = "All Crops (Auto-Detect)"
            st.rerun()


# ==============================================================================
# FLOATING CROPCARE AI ASSISTANT (Persistent across all 4 workflow screens)
# ==============================================================================
def render_cropcare_chatbot() -> None:
    """Render the floating Gemini AI Assistant launcher and interactive panel."""
    chat_lang = st.session_state.app_language
    # Floating Launcher Button in Bottom-Right
    launcher_label = f"💬 {t('chat_btn_open', chat_lang)}" if not st.session_state.chat_open else f"✕ {t('chat_btn_close', chat_lang)}"
    if st.button(
        launcher_label,
        key="chat_launcher_btn",
        help="Ask questions about crop diseases, screening, or climate insights",
    ):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

    # Floating Dialog Panel when Open
    if st.session_state.chat_open:
        with st.container(key="cropcare_chat_panel"):
            head_col1, head_col2, head_col3 = st.columns([3, 0.8, 0.8])
            with head_col1:
                st.markdown(
                    f"<div style='font-family: \"Space Grotesk\", sans-serif; font-size: 1.15rem; font-weight: 700; color: #123D2A;'>{t('chat_panel_title', chat_lang)}</div>",
                    unsafe_allow_html=True,
                )
            with head_col2:
                if st.button("🗑️", key="chat_clear_btn", help="Clear conversation history"):
                    st.session_state.chat_messages = []
                    st.rerun()
            with head_col3:
                if st.button("✕", key="chat_close_btn", help="Close Assistant"):
                    st.session_state.chat_open = False
                    st.rerun()

            st.caption("Agronomic Guidance • Powered by Google Gemini")

            # Inline Gemini API Key Setup (if not already set in environment or secrets)
            from src.cropcare.chat_engine import get_gemini_api_key
            curr_gemini_key = get_gemini_api_key()
            if not curr_gemini_key:
                with st.expander("🔑 Configure Gemini API Key", expanded=False):
                    entered_key = st.text_input("Enter Google Gemini API Key:", type="password", key="user_gemini_api_key_input")
                    if entered_key and st.button("Save Key", key="save_gemini_key_btn"):
                        os.environ["GEMINI_API_KEY"] = entered_key.strip()
                        st.success("API Key configured for this session!")
                        st.rerun()

            # Suggested Prompts (shown when conversation is new)
            if len(st.session_state.chat_messages) <= 1:
                st.markdown(
                    "<div style='font-size: 0.8rem; font-weight: 600; color: #526E60; margin-bottom: 0.2rem;'>Suggested Questions:</div>",
                    unsafe_allow_html=True,
                )
                chip_col1, chip_col2 = st.columns(2)
                from src.cropcare.chat_engine import SUGGESTED_PROMPTS

                with chip_col1:
                    if st.button(SUGGESTED_PROMPTS[0], key="sug_chip_0", use_container_width=True):
                        _submit_chat_message(SUGGESTED_PROMPTS[0])
                        st.rerun()
                with chip_col2:
                    if st.button(SUGGESTED_PROMPTS[1], key="sug_chip_1", use_container_width=True):
                        _submit_chat_message(SUGGESTED_PROMPTS[1])
                        st.rerun()

            # Message History Box
            chat_container = st.container(height=260)
            with chat_container:
                if not st.session_state.chat_messages:
                    st.markdown(
                        "<div style='color: #526E60; font-size: 0.86rem; text-align: center; margin-top: 2rem; line-height: 1.5;'>"
                        "👋 <strong>Welcome!</strong><br>Ask about crop diseases, symptoms, your leaf screening results, or local climate stress factors."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    for turn in st.session_state.chat_messages:
                        role = turn.get("role", "user")
                        with st.chat_message(role, avatar="🧑‍🌾" if role == "user" else "🌱"):
                            st.markdown(turn.get("content", ""))

            # Input Prompt
            prompt_text = st.chat_input(t('chat_placeholder', chat_lang), key="chat_user_prompt")
            if prompt_text:
                _submit_chat_message(prompt_text)
                st.rerun()


def _submit_chat_message(user_text: str) -> None:
    """Send user text to Gemini with contextual session state and append reply."""
    st.session_state.chat_messages.append({"role": "user", "content": user_text})
    from src.cropcare.chat_engine import build_agronomic_context, generate_chat_response

    ctx = build_agronomic_context(
        screen=st.session_state.screen,
        leaf_result=st.session_state.leaf_result,
        climate_result=st.session_state.climate_result,
        climate_days=st.session_state.climate_days,
    )
    with st.spinner("Consulting CropCare Assistant..."):
        reply = generate_chat_response(
            messages=st.session_state.chat_messages,
            context_str=ctx,
        )
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


render_cropcare_chatbot()
