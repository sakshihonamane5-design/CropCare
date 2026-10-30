"""CropCare — Next-Generation Agricultural Intelligence Dashboard.

A premium, guided 4-screen SaaS application providing transfer-learning
leaf screening, Grad-CAM explainability, hyper-local environmental intelligence,
biometeorological concurrence analysis, weather-smart IPM advisories,
economic loss estimation, Gemini Multimodal Vision verification, and a 24/7 persistent Gemini AI Agronomist.
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

from src.cropcare.explainability import generate_gradcam
from src.cropcare.alignment_engine import evaluate_environmental_alignment
from src.cropcare.advisory_engine import (
    generate_actionable_advisory,
    generate_recovery_roadmap,
    build_whatsapp_share_url,
)
from src.cropcare.report_engine import generate_field_report_html
from src.cropcare.vernacular import translate_diagnosis, generate_audio_script, t
from src.cropcare.economic_engine import estimate_economic_impact
from src.cropcare.location_engine import (
    TOP_PRESET_LOCATIONS,
    search_location_online,
    get_states,
    get_districts_for_state,
)
from src.cropcare.chat_engine import (
    build_agronomic_context,
    generate_chat_response,
    verify_leaf_with_gemini_vision,
    SUGGESTED_PROMPTS,
)

# Configure page layout and title (Must be first Streamlit command)
st.set_page_config(
    page_title="CropCare — Precision Agronomic Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
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

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--text-dark);
        background-color: var(--ivory-bg);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    /* Top Navigation Brand Bar */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.5rem;
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        box-shadow: var(--shadow-card);
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
        letter-spacing: 0.06em;
        background: var(--lime-light);
        color: var(--forest-dark);
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        border: 1px solid rgba(183, 230, 107, 0.5);
    }

    /* Step Progress Indicator */
    .step-progress-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 0.75rem 1.25rem;
        margin-top: 1.2rem;
        margin-bottom: 1.8rem;
        box-shadow: var(--shadow-card);
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-muted);
        text-decoration: none;
    }
    .step-item.active {
        color: var(--forest-dark);
        font-weight: 700;
    }
    .step-pill {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #E5EBE7;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .step-item.active .step-pill {
        background: var(--forest-dark);
        color: var(--lime-accent);
    }
    .step-item.completed .step-pill {
        background: var(--lime-accent);
        color: var(--forest-dark);
    }
    .step-divider {
        flex: 1;
        height: 2px;
        background: #E5EBE7;
        margin: 0 1rem;
    }
    .step-divider.active {
        background: var(--lime-accent);
    }

    /* Hero Component */
    .hero-container {
        background: linear-gradient(135deg, #123D2A 0%, #1B543A 60%, #0F2E20 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-hero);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -40%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(183, 230, 107, 0.15) 0%, rgba(183, 230, 107, 0) 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(183, 230, 107, 0.15);
        border: 1px solid rgba(183, 230, 107, 0.3);
        color: var(--lime-accent);
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
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
    }
    .hero-title span {
        color: var(--lime-accent);
    }
    .hero-sub {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #D3E4DB;
        max-width: 620px;
        margin-bottom: 2rem;
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

    /* 3D Botanical Visual Graphic */
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

    /* Stats bar on hero */
    .stat-row {
        display: flex;
        gap: 2rem;
        margin-top: 1.8rem;
        flex-wrap: wrap;
    }
    .stat-item {
        text-align: center;
    }
    .stat-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--lime-accent);
        line-height: 1;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #A0C4AB;
        margin-top: 0.25rem;
    }

    /* Surface Card */
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
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--forest-dark);
        margin-bottom: 0.5rem;
    }
    .feature-body {
        font-size: 0.88rem;
        color: var(--text-muted);
        line-height: 1.55;
    }

    /* Diagnostic Results Panel */
    .diagnostic-panel {
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.5rem;
    }

    /* Gauge Card */
    .gauge-card {
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: var(--shadow-card);
    }

    /* Economic Loss Card */
    .economic-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FBF8 100%);
        border: 1.5px solid rgba(27, 84, 58, 0.2);
        border-radius: 18px;
        padding: 1.6rem;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.5rem;
    }
    .loss-stat-pill {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .roi-badge {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }

    /* 7-Day Roadmap Timeline */
    .roadmap-card {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(18, 61, 42, 0.04);
        border-left: 5px solid var(--forest-mid);
    }

    /* Location Confirmation Card */
    .location-confirmed-card {
        background: #EBF8D3;
        border: 1px solid rgba(18, 61, 42, 0.25);
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        background: #F4F6F0;
        border: 1px dashed #A0B2A6;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.82rem;
        color: #405B4D;
        line-height: 1.5;
        margin-top: 2rem;
    }

    /* Tips box */
    .tips-box {
        background: linear-gradient(135deg, #EBF8D3 0%, #F0FBE0 100%);
        border: 1px solid rgba(183, 230, 107, 0.4);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: var(--forest-dark);
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }

    /* STREAMLIT SIDEBAR: Dedicated CropCare AI Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #092015 0%, #123D2A 100%) !important;
        border-right: 1px solid rgba(183, 230, 107, 0.25) !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2ECE5;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .sidebar-ai-header {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(183, 230, 107, 0.35);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #B7E66B;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px #B7E66B;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }
    .ai-context-chip {
        background: rgba(183, 230, 107, 0.12);
        border: 1px solid rgba(183, 230, 107, 0.35);
        color: #DDF3A8 !important;
        border-radius: 10px;
        padding: 0.45rem 0.75rem;
        font-size: 0.78rem;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
defaults = {
    "screen": 1,
    "leaf_image": None,
    "leaf_image_id": None,
    "leaf_result": None,
    "cam_result": None,
    "selected_crop_filter": "🔍 Auto-Detect (All 14 Crops)",
    "gemini_vision_res": None,
    "climate_lat": 19.9975,  # Default to Nashik, Maharashtra
    "climate_lon": 73.7898,
    "climate_loc_name": "Nashik, Maharashtra",
    "climate_days": 30,
    "climate_result": None,
    "lang": "en",
    "chat_messages": [],
    "farmer_name": "",
    "farm_name": "",
    "farm_acres": 2.0,
    "loc_search_query": "",
    "loc_search_results": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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


def display_image(image, caption: Optional[str] = None) -> None:
    """Safely render an image across Streamlit versions."""
    try:
        st.image(image, caption=caption, use_column_width=True)
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


# ==============================================================================
# TOP BRANDING & LANGUAGE SWITCHER
# ==============================================================================
b_col1, b_col2, b_col3 = st.columns([3.2, 1.3, 0.9])
with b_col1:
    st.markdown(
        f"""
        <div class="brand-bar">
            <div class="brand-logo">🌱 CropCare™</div>
            <div class="brand-tag">{t('brand_tag', st.session_state.lang)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with b_col2:
    lang_map = {
        "English": "en",
        "हिन्दी (Hindi)": "hi",
        "मराठी (Marathi)": "mr",
    }
    rev_lang_map = {"en": 0, "hi": 1, "mr": 2}
    current_selection_index = rev_lang_map.get(st.session_state.lang, 0)
    selected_language_name = st.selectbox(
        "Language / भाषा",
        options=list(lang_map.keys()),
        index=current_selection_index,
        key="lang_selector_widget",
        label_visibility="collapsed",
    )
    new_lang_code = lang_map[selected_language_name]
    if new_lang_code != st.session_state.lang:
        st.session_state.lang = new_lang_code
        st.rerun()

with b_col3:
    st.markdown(
        """
        <div style="background: rgba(18, 61, 42, 0.07); border: 1px solid rgba(18, 61, 42, 0.15);
                    border-radius: 12px; padding: 0.45rem 0.8rem; text-align: center;">
            <span class="live-dot" style="margin-right: 5px;"></span>
            <span style="font-size: 0.82rem; font-weight: 700; color: #123D2A;">AI Advisor</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

curr_lang = st.session_state.lang

# Step Indicator
curr = st.session_state.screen
step1_cls = "active" if curr == 1 else ("completed" if curr > 1 else "")
step2_cls = "active" if curr == 2 else ("completed" if curr > 2 else "")
step3_cls = "active" if curr == 3 else ("completed" if curr > 3 else "")
step4_cls = "active" if curr == 4 else ""
div1_cls = "active" if curr > 1 else ""
div2_cls = "active" if curr > 2 else ""
div3_cls = "active" if curr > 3 else ""

st.markdown(
    f"""
    <div class="step-progress-wrapper">
        <div class="step-item {step1_cls}">
            <div class="step-pill">{'✓' if curr > 1 else '1'}</div>
            <span>{t('step1', curr_lang)}</span>
        </div>
        <div class="step-divider {div1_cls}"></div>
        <div class="step-item {step2_cls}">
            <div class="step-pill">{'✓' if curr > 2 else '2'}</div>
            <span>{t('step2', curr_lang)}</span>
        </div>
        <div class="step-divider {div2_cls}"></div>
        <div class="step-item {step3_cls}">
            <div class="step-pill">{'✓' if curr > 3 else '3'}</div>
            <span>{t('step3', curr_lang)}</span>
        </div>
        <div class="step-divider {div3_cls}"></div>
        <div class="step-item {step4_cls}">
            <div class="step-pill">4</div>
            <span>{t('step4', curr_lang)}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col = st.container()


# ==============================================================================
# SCREEN 1: WELCOME & START
# ==============================================================================
def render_screen1():
    with main_col:
        hero_col_left, hero_col_right = st.columns([1.25, 0.95], gap="large")

        with hero_col_left:
            st.markdown(
                f"""
                <div class="hero-container">
                    <div class="hero-badge">{t('hero_badge', curr_lang)}</div>
                    <div class="hero-title">{t('hero_title', curr_lang)}</div>
                    <div class="hero-sub">{t('hero_sub', curr_lang)}</div>
                    <div class="stat-row">
                        <div class="stat-item">
                            <div class="stat-num">38</div>
                            <div class="stat-label">Disease Classes</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-num">14</div>
                            <div class="stat-label">Crop Species</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-num">15ms</div>
                            <div class="stat-label">CPU Inference</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-num">3</div>
                            <div class="stat-label">Languages (EN/HI/MR)</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("👨‍🌾 Optional: Enter Farmer & Field Profile (Personalises report)", expanded=False):
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    fname = st.text_input("Farmer Name / किसान का नाम", value=st.session_state.farmer_name, placeholder="e.g. Ramesh Patil", key="farmer_name_inp")
                with p_c2:
                    farmn = st.text_input("Farm Name & Area / खेत का नाम", value=st.session_state.farm_name, placeholder="e.g. Patil Farms, Nashik", key="farm_name_inp")
                acres = st.number_input("Landholding (Acres) — for Economic Impact calculation", min_value=0.25, max_value=500.0, value=float(st.session_state.farm_acres), step=0.5, key="acres_inp")
                if st.button("Save Farm Profile", key="save_farmer_details"):
                    st.session_state.farmer_name = fname
                    st.session_state.farm_name = farmn
                    st.session_state.farm_acres = acres
                    st.success("Details saved successfully! ✅")

            if st.button(t("btn_begin", curr_lang), type="primary", use_container_width=True, key="begin_btn"):
                st.session_state.screen = 2
                st.rerun()

        with hero_col_right:
            farmer_img_uri = get_image_base64("assets/farmer_hero.jpg")
            if farmer_img_uri:
                st.markdown(
                    f"""
                    <div class="farmer-hero-card">
                        <div class="farmer-img-frame">
                            <img src="{farmer_img_uri}" alt="Farmer inspecting crop foliage" class="farmer-hero-img" />
                            <div class="farmer-overlay-badge">
                                <span>🌱 Dual-Evidence Intelligence Active</span>
                            </div>
                        </div>
                        <div class="farmer-card-caption">
                            <div class="caption-title">Real-Time Agronomic Assessment</div>
                            <div class="caption-sub">Field-tested insights connecting leaf diagnostics with hyper-local weather baselines.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="botanical-3d-card" style="background: linear-gradient(145deg, #184C34 0%, #0F2E20 100%);">
                        <div style="font-size: 4.5rem; margin-bottom: 0.5rem;">🌿</div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">
                            MobileNetV3 Neural Core
                        </div>
                        <div style="font-size: 0.88rem; color: #D3E4DB; margin-top: 0.4rem;">
                            14 Crops • 38 Pathology Classes<br>Hyper-Local Reanalysis Weather Telemetry
                        </div>
                        <div class="botanical-badge">38-Class Transfer Learning Benchmark</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        f_col1, f_col2, f_col3 = st.columns(3, gap="medium")
        features_row1 = [
            ("🍃", t('feat1_title', curr_lang), t('feat1_desc', curr_lang)),
            ("🌦️", t('feat2_title', curr_lang), t('feat2_desc', curr_lang)),
            ("🛡️", t('feat3_title', curr_lang), t('feat3_desc', curr_lang)),
        ]
        for col, (icon, title, body) in zip([f_col1, f_col2, f_col3], features_row1):
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-icon">{icon}</div>
                        <div class="feature-title">{title}</div>
                        <div class="feature-body">{body}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        f_col4, f_col5, f_col6 = st.columns(3, gap="medium")
        features_row2 = [
            ("🔥", "Grad-CAM Explainability", "Visual AI heatmaps prove to judges and farmers exactly which leaf lesion guided the model's prediction."),
            ("💰", "Yield & Economic Loss Estimator", "Calculates real financial loss in ₹/acre and demonstrates intervention ROI for farm profitability."),
            ("🤖", "Gemini Agronomic Assistant", "Dedicated persistent AI sidebar answering farmer questions with live scan context in 3 languages."),
        ]
        for col, (icon, title, body) in zip([f_col4, f_col5, f_col6], features_row2):
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-icon">{icon}</div>
                        <div class="feature-title">{title}</div>
                        <div class="feature-body">{body}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==============================================================================
# SCREEN 2: LEAF SCREENING, EXPLAINABLE AI & GEMINI VISION VERIFICATION
# ==============================================================================
def render_screen2():
    with main_col:
        st.markdown(
            f"""
            <div class="surface-card">
                <h3>{t('screen2_title', curr_lang)}</h3>
                <p class="card-desc">{t('screen2_desc', curr_lang)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        leaf_col_left, leaf_col_right = st.columns([1, 1], gap="large")

        with leaf_col_left:
            # 1-Click Instant Test Samples
            st.markdown("##### ⚡ Instant Benchmark Samples (Click to Test):")
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("🍅 Tomato: Early Blight", use_container_width=True, key="smpl_t_eb"):
                    p = Path(__file__).resolve().parent / "assets/samples/tomato_early_blight.jpg"
                    if p.exists():
                        st.session_state.leaf_image = Image.open(p).convert("RGB")
                        st.session_state.leaf_image_id = "sample_tomato_early_blight"
                        st.session_state.selected_crop_filter = "Tomato"
                        st.session_state.leaf_result = None
                        st.session_state.cam_result = None
                        st.session_state.gemini_vision_res = None
                        st.rerun()
                if st.button("🍅 Tomato: Late Blight", use_container_width=True, key="smpl_t_lb"):
                    p = Path(__file__).resolve().parent / "assets/samples/tomato_late_blight.jpg"
                    if p.exists():
                        st.session_state.leaf_image = Image.open(p).convert("RGB")
                        st.session_state.leaf_image_id = "sample_tomato_late_blight"
                        st.session_state.selected_crop_filter = "Tomato"
                        st.session_state.leaf_result = None
                        st.session_state.cam_result = None
                        st.session_state.gemini_vision_res = None
                        st.rerun()
            with sc2:
                if st.button("🥔 Potato: Early Blight", use_container_width=True, key="smpl_p_eb"):
                    p = Path(__file__).resolve().parent / "assets/samples/potato_early_blight.jpg"
                    if p.exists():
                        st.session_state.leaf_image = Image.open(p).convert("RGB")
                        st.session_state.leaf_image_id = "sample_potato_early_blight"
                        st.session_state.selected_crop_filter = "Potato"
                        st.session_state.leaf_result = None
                        st.session_state.cam_result = None
                        st.session_state.gemini_vision_res = None
                        st.rerun()
                if st.button("🍃 Tomato: Healthy Leaf", use_container_width=True, key="smpl_t_hl"):
                    p = Path(__file__).resolve().parent / "assets/samples/tomato_healthy.jpg"
                    if p.exists():
                        st.session_state.leaf_image = Image.open(p).convert("RGB")
                        st.session_state.leaf_image_id = "sample_tomato_healthy"
                        st.session_state.selected_crop_filter = "Tomato"
                        st.session_state.leaf_result = None
                        st.session_state.cam_result = None
                        st.session_state.gemini_vision_res = None
                        st.rerun()

            st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

            # Target Crop Species Filter
            crop_options = [
                "🔍 Auto-Detect (All 14 Crops)",
                "Tomato",
                "Potato",
                "Apple",
                "Grape",
                "Corn_(maize)",
                "Pepper,_bell",
                "Strawberry",
                "Peach",
                "Orange",
                "Soybean",
                "Squash",
                "Cherry",
            ]
            c_idx = 0
            if st.session_state.selected_crop_filter in crop_options:
                c_idx = crop_options.index(st.session_state.selected_crop_filter)

            selected_crop_choice = st.selectbox(
                "🌾 Target Crop Species (Eliminates false cross-crop detections):",
                options=crop_options,
                index=c_idx,
                key="crop_filter_select",
                help="Conditioning inference on your known crop species prevents cross-species false positives.",
            )
            st.session_state.selected_crop_filter = selected_crop_choice

            st.markdown("#### 📷 Leaf Image Acquisition")
            input_mode = st.radio(
                "Acquisition Method:",
                options=["📁 Upload Image", "📸 Take Photo with Camera"],
                horizontal=True,
                key="acquisition_mode",
            )

            active_file = None
            new_source_id = None

            if input_mode == "📁 Upload Image":
                active_file = st.file_uploader(
                    t("upload_label", curr_lang),
                    type=["jpg", "jpeg", "png"],
                    key="leaf_uploader",
                    help="Upload a clear, focused leaf photograph with visible symptom spots.",
                )
                if active_file is not None:
                    new_source_id = f"upload_{active_file.name}_{active_file.size}"
            else:
                active_file = st.camera_input(
                    "Capture live leaf photograph",
                    key="leaf_camera",
                    help="Center the plant leaf in the camera preview and click 'Take Photo'.",
                )
                if active_file is not None:
                    new_source_id = f"camera_{active_file.name}_{active_file.size}"

            if active_file is not None and new_source_id != st.session_state.leaf_image_id:
                try:
                    img = Image.open(active_file).convert("RGB")
                    st.session_state.leaf_image = img
                    st.session_state.leaf_image_id = new_source_id
                    st.session_state.leaf_result = None
                    st.session_state.cam_result = None
                    st.session_state.gemini_vision_res = None
                except Exception as e:
                    st.error(f"Image decoding failed: {e}")

            if st.session_state.leaf_image is not None:
                display_image(st.session_state.leaf_image, caption="Current Leaf Under Assessment")

                b1, b2 = st.columns([1.5, 1])
                with b1:
                    run_screening = st.button(t("btn_run_screening", curr_lang), type="primary", use_container_width=True, key="run_screen_btn")
                with b2:
                    if st.button("🔄 Clear / Retake", type="secondary", use_container_width=True, key="clear_leaf_btn"):
                        st.session_state.leaf_image = None
                        st.session_state.leaf_image_id = None
                        st.session_state.leaf_result = None
                        st.session_state.cam_result = None
                        st.session_state.gemini_vision_res = None
                        st.rerun()

                if run_screening:
                    with st.spinner("Executing neural inference & Grad-CAM explainability..."):
                        try:
                            from src.cropcare.image_engine import predict_image

                            crop_filt = None if "Auto-Detect" in selected_crop_choice else selected_crop_choice
                            res = predict_image(st.session_state.leaf_image, crop_filter=crop_filt)
                            st.session_state.leaf_result = res

                            # Run Grad-CAM heatmap
                            cam_res = generate_gradcam(st.session_state.leaf_image)
                            st.session_state.cam_result = cam_res
                            st.session_state.gemini_vision_res = None
                        except Exception as inf_err:
                            st.error(f"Inference execution error: {inf_err}")
            else:
                st.info("Please select a sample leaf above, upload an image, or take a live camera shot.")

        with leaf_col_right:
            st.markdown("#### 🔬 Neural Diagnostic Output")

            res = st.session_state.leaf_result
            if res is None:
                st.markdown(
                    """
                    <div style="background: #FFFFFF; border: 2px dashed #CCD7D0; border-radius: 16px; padding: 2.5rem 1.5rem; text-align: center; color: #526E60;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🍃</div>
                        <div style="font-weight: 700; color: #123D2A; font-size: 1.1rem; margin-bottom: 0.3rem;">Ready for Inference</div>
                        <div style="font-size: 0.88rem;">MobileNetV3 transfer model is initialized.<br>Click "Run Leaf Screening" to classify and generate Grad-CAM heatmap.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                crop = res.get("crop", "Unknown")
                disease = res.get("disease", "Unknown")
                conf = res.get("confidence", 0.0)
                uncert = res.get("uncertain", False)
                latency = res.get("latency_ms", 15.0)

                trans_crop, trans_disease = translate_diagnosis(crop, disease, lang=curr_lang)

                tab_diag, tab_cam, tab_gemini = st.tabs([
                    "📋 Primary Diagnosis",
                    "🔥 Grad-CAM Heatmap",
                    "🤖 AI Second Opinion (Gemini Vision)",
                ])

                with tab_diag:
                    st.markdown(
                        f"""
                        <div class="diagnostic-panel">
                            <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('target_crop', curr_lang)}</div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #123D2A; margin-bottom: 0.8rem;">{trans_crop}</div>
                            <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('screened_condition', curr_lang)}</div>
                            <div style="font-size: 1.2rem; font-weight: 600; color: #1B543A; margin-bottom: 0.8rem;">{trans_disease}</div>
                            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.4rem;">
                                <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('confidence', curr_lang)}</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #123D2A;">{conf*100:.1f}%</div>
                            </div>
                            <div style="background: #E5EBE7; border-radius: 8px; height: 8px; overflow: hidden;">
                                <div style="background: var(--forest-dark); height: 100%; width: {min(100.0, conf*100):.1f}%;"></div>
                            </div>
                            <div style="margin-top: 0.8rem; font-size: 0.78rem; color: #526E60;">
                                ⚡ Inference latency: <strong>{latency:.1f} ms</strong> on CPU {'• Crop-Constrained' if res.get('crop_constrained') else ''}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Top alternative candidates
                    if res.get("top_candidates") and len(res["top_candidates"]) > 1:
                        st.markdown("##### 📊 Top Diagnostic Candidates:")
                        for cand in res["top_candidates"]:
                            c_pct = cand["confidence"] * 100
                            st.write(f"• **{cand['crop']} — {cand['disease']}**: {c_pct:.1f}%")
                            st.progress(min(1.0, max(0.0, cand["confidence"])))

                    if uncert:
                        st.warning("⚠️ **Low Confidence Flag:** Model uncertainty is elevated. Visual symptoms may be ambiguous; physical field inspection or Gemini Vision verification is recommended.")
                    else:
                        st.success("✅ **Calibrated Confidence:** Optical symptoms closely match trained botanical benchmarks.")

                with tab_cam:
                    cam_data = st.session_state.cam_result
                    if cam_data and cam_data.get("overlay_image"):
                        st.markdown(
                            """
                            <div class="tips-box">
                                <strong>Explainable AI (XAI):</strong> The heatmap highlights the exact pixel clusters that triggered the disease classification. Red/Yellow zones show high neural attention.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        c_col1, c_col2 = st.columns(2)
                        with c_col1:
                            display_image(st.session_state.leaf_image, caption="Original Input")
                        with c_col2:
                            display_image(cam_data["overlay_image"], caption="Grad-CAM Attention Heatmap")
                    else:
                        st.info("Grad-CAM visualization will generate upon running screening.")

                with tab_gemini:
                    st.markdown(
                        """
                        <div class="tips-box">
                            <strong>Dual-AI Agronomic Architecture:</strong> While MobileNetV3 runs locally in 15ms, Google Gemini Multimodal Vision verifies leaf morphology, lesion borders, and chlorotic margins to eliminate laboratory background artifacts.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button("🔬 Request Gemini Multimodal Second Opinion", type="primary", key="btn_run_gemini_vision", use_container_width=True):
                        with st.spinner("Gemini Multimodal Vision analyzing leaf lesions and morphology..."):
                            v_res = verify_leaf_with_gemini_vision(
                                st.session_state.leaf_image,
                                predicted_crop=res.get("crop"),
                                predicted_disease=res.get("disease"),
                            )
                            st.session_state.gemini_vision_res = v_res
                            st.rerun()

                    if st.session_state.gemini_vision_res:
                        gv = st.session_state.gemini_vision_res
                        st.markdown(
                            f"""
                            <div class="diagnostic-panel" style="border-left: 5px solid #1B543A; margin-top: 1rem;">
                                <div style="font-size: 0.8rem; font-weight: 700; color: #123D2A; text-transform: uppercase; margin-bottom: 0.6rem;">
                                    🤖 Gemini Multimodal Clinical Evaluation
                                </div>
                                <div style="font-size: 0.92rem; color: #11231B; line-height: 1.6;">
                                    {gv.get("analysis", "")}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Vernacular Audio Script
                audio_script = generate_audio_script(crop, disease, conf, "Aligned", "Review guidance", lang=curr_lang)
                with st.expander("🔊 Spoken Vernacular Narration (Audio Script)", expanded=False):
                    st.info(audio_script)

        # Navigation
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        nav1, nav2 = st.columns([1, 1])
        with nav1:
            if st.button(t("btn_back_welcome", curr_lang), type="secondary", key="s2_back"):
                st.session_state.screen = 1
                st.rerun()
        with nav2:
            if st.button(t("btn_goto_climate", curr_lang), type="primary", key="s2_next"):
                st.session_state.screen = 3
                st.rerun()


# ==============================================================================
# SCREEN 3: CLIMATE INSIGHTS & FARMER-FRIENDLY LOCATION ENGINE
# ==============================================================================
def render_screen3():
    with main_col:
        st.markdown(
            f"""
            <div class="surface-card">
                <h3>{t('screen3_title', curr_lang)}</h3>
                <p class="card-desc">{t('screen3_desc', curr_lang)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📍 Select Farm Location (Farmer-Friendly)")
        st.markdown(
            """
            <div class="tips-box">
                🌾 <strong>No GPS Coordinates Needed:</strong>
                Choose your agricultural district, search any village or town name, or pick your State & District.
                CropCare automatically determines the exact coordinates and pulls Open-Meteo weather telemetry.
            </div>
            """,
            unsafe_allow_html=True,
        )

        loc_tab1, loc_tab2, loc_tab3, loc_tab4 = st.tabs([
            "🌾 Top Agricultural Belts (1-Click)",
            "🔍 Search Any Village / Town / City",
            "🗺️ State & District Selector",
            "⚙️ Advanced GPS Coordinates",
        ])

        lat = st.session_state.climate_lat
        lon = st.session_state.climate_lon
        loc_name = st.session_state.climate_loc_name

        # TAB 1: 1-Click Top Agricultural Belts
        with loc_tab1:
            preset_labels = list(TOP_PRESET_LOCATIONS.keys())
            chosen_preset = st.selectbox(
                "Select Major Agricultural Belt:",
                options=preset_labels,
                index=0,
                key="preset_loc_picker",
                help="Pre-configured hubs representing major fruit, grain, and cash crop clusters.",
            )
            p_lat, p_lon = TOP_PRESET_LOCATIONS[chosen_preset]
            if st.button("Apply Selected Agri-Belt", key="apply_preset_btn"):
                st.session_state.climate_lat = p_lat
                st.session_state.climate_lon = p_lon
                st.session_state.climate_loc_name = chosen_preset.split(" (")[0]
                st.rerun()

        # TAB 2: Real-Time Village / City Search
        with loc_tab2:
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                q_text = st.text_input(
                    "Enter Village, Taluka, Town, or District Name:",
                    placeholder="e.g. Baramati, Solapur, Bhatinda, Kopargaon, Muzaffarnagar...",
                    key="village_search_input",
                )
            with search_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                do_search = st.button("Search 🔍", key="exec_loc_search", use_container_width=True)

            if do_search and q_text:
                with st.spinner("Searching global and Indian geocoding databases..."):
                    found = search_location_online(q_text)
                    st.session_state.loc_search_results = found

            if st.session_state.loc_search_results:
                st.markdown("**Search Results (Click to Set):**")
                for idx, item in enumerate(st.session_state.loc_search_results):
                    c_txt, c_btn = st.columns([3.5, 1])
                    with c_txt:
                        st.markdown(f"**{item['display']}** `({item['latitude']:.4f}°N, {item['longitude']:.4f}°E)`")
                    with c_btn:
                        if st.button("Select Location", key=f"sel_loc_{idx}", use_container_width=True):
                            st.session_state.climate_lat = item["latitude"]
                            st.session_state.climate_lon = item["longitude"]
                            st.session_state.climate_loc_name = f"{item['name']}, {item.get('state', '')}"
                            st.success(f"Selected: {item['name']} ✅")
                            st.rerun()

        # TAB 3: State & District Selector
        with loc_tab3:
            s_col1, s_col2 = st.columns(2)
            states = get_states()
            with s_col1:
                selected_state = st.selectbox("Select State:", options=states, index=0, key="state_selector")
            with s_col2:
                dist_dict = get_districts_for_state(selected_state)
                dist_names = list(dist_dict.keys())
                selected_district = st.selectbox("Select District / Tehsil:", options=dist_names, key="district_selector")

            if selected_district in dist_dict:
                d_lat, d_lon, d_crops = dist_dict[selected_district]
                st.caption(f"🌿 Primary Regional Crops in {selected_district}: **{d_crops}**")
                if st.button("Set State & District Location", key="apply_state_dist_btn"):
                    st.session_state.climate_lat = d_lat
                    st.session_state.climate_lon = d_lon
                    st.session_state.climate_loc_name = f"{selected_district}, {selected_state}"
                    st.rerun()

        # TAB 4: Advanced Lat / Lon (For Researchers)
        with loc_tab4:
            st.caption("For agricultural scientists, research plots, and drone pilots:")
            man_c1, man_c2 = st.columns(2)
            with man_c1:
                manual_lat = st.number_input(
                    "Latitude (°N)",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(st.session_state.climate_lat),
                    step=0.0001,
                    format="%.4f",
                    key="man_lat_inp",
                )
            with man_c2:
                manual_lon = st.number_input(
                    "Longitude (°E)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=float(st.session_state.climate_lon),
                    step=0.0001,
                    format="%.4f",
                    key="man_lon_inp",
                )
            if st.button("Apply Exact GPS Coordinates", key="apply_man_gps_btn"):
                st.session_state.climate_lat = manual_lat
                st.session_state.climate_lon = manual_lon
                st.session_state.climate_loc_name = f"Custom ({manual_lat:.2f}°N, {manual_lon:.2f}°E)"
                st.rerun()

        # Location Active Confirmation Card
        lat = st.session_state.climate_lat
        lon = st.session_state.climate_lon
        loc_name = st.session_state.climate_loc_name

        st.markdown(
            f"""
            <div class="location-confirmed-card">
                <div style="font-weight: 700; color: #123D2A; font-size: 1.05rem;">
                    📍 Active Farm Location: {loc_name}
                </div>
                <div style="color: #2A7352; font-size: 0.85rem; margin-top: 0.25rem;">
                    Coordinates: <code>{lat:.4f}°N, {lon:.4f}°E</code> &nbsp;•&nbsp; Calibrated with Open-Meteo Reanalysis Weather Grid
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Observation Window
        st.markdown("#### 🗓️ Weather Observation Window")
        window_options = {
            "Last 7 Days (Short-Term Fluctuation)": 7,
            "Last 14 Days (Incubation Period)": 14,
            "Last 30 Days (Recommended Agronomic Baseline)": 30,
        }
        w_labels = list(window_options.keys())
        w_vals = list(window_options.values())
        default_w_idx = w_vals.index(st.session_state.climate_days) if st.session_state.climate_days in w_vals else 2
        selected_window_label = st.radio(
            "Observation Window Duration:",
            options=w_labels,
            index=default_w_idx,
            horizontal=True,
            key="window_radio",
        )
        window = window_options[selected_window_label]

        if st.button(t("btn_fetch_climate", curr_lang), type="primary", use_container_width=True, key="fetch_climate_btn"):
            st.session_state.climate_days = window
            with st.spinner(f"Querying Open-Meteo ERA5 Reanalysis for {loc_name}..."):
                try:
                    from src.cropcare.climate_engine import get_environment

                    env_res = get_environment(
                        location=(lat, lon),
                        days=window,
                        baseline_years=[2023, 2024, 2025],
                    )
                    st.session_state.climate_result = env_res
                    st.success(f"✅ Retrieved {window}-day climate telemetry for {loc_name}!")
                except Exception as clim_err:
                    st.error(f"Failed to retrieve climate telemetry: {clim_err}")

        # Display Climate Analysis
        env_data = st.session_state.climate_result
        if env_data and env_data.get("recent"):
            recent = env_data.get("recent") or {}
            anomalies = env_data.get("anomalies") or {}

            st.markdown("#### 📊 Environmental Summary Telemetry")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(
                    label="Mean Temperature",
                    value=format_metric(recent.get("mean_temperature_c"), "°C"),
                    delta=format_metric(anomalies.get("temperature_c"), "°C vs baseline")
                    if anomalies.get("temperature_c") is not None
                    else None,
                )
            with m_col2:
                st.metric(
                    label="Max Temperature",
                    value=format_metric(recent.get("max_temperature_c"), "°C"),
                )
            with m_col3:
                st.metric(
                    label="Total Rainfall",
                    value=format_metric(recent.get("rainfall_mm"), "mm"),
                    delta=format_metric(anomalies.get("rainfall_mm"), "mm vs baseline")
                    if anomalies.get("rainfall_mm") is not None
                    else None,
                    delta_color="normal",
                )
            with m_col4:
                st.metric(
                    label="Relative Humidity",
                    value=format_metric(recent.get("mean_humidity_pct"), "%"),
                )

            # Daily Time Series Chart
            daily_series = env_data.get("daily_series") or []
            if daily_series:
                with st.expander("📈 View Daily Weather Trends (Temperature & Rainfall)", expanded=True):
                    df = pd.DataFrame(daily_series)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date")

                        if "temperature_mean_c" in df.columns and df["temperature_mean_c"].notna().any():
                            st.write("**Daily Mean Temperature (°C)**")
                            st.line_chart(df[["temperature_mean_c"]].rename(columns={"temperature_mean_c": "Mean Temp (°C)"}))

                        if "precipitation_mm" in df.columns and df["precipitation_mm"].notna().any():
                            st.write("**Precipitation (mm)**")
                            st.bar_chart(df[["precipitation_mm"]].rename(columns={"precipitation_mm": "Rainfall (mm)"}))

        # Navigation
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        nav1, nav2 = st.columns([1, 1])
        with nav1:
            if st.button(t("btn_back_leaf", curr_lang), type="secondary", key="s3_back"):
                st.session_state.screen = 2
                st.rerun()
        with nav2:
            if st.button(t("btn_goto_summary", curr_lang), type="primary", key="s3_next"):
                st.session_state.screen = 4
                st.rerun()


# ==============================================================================
# SCREEN 4: EXECUTIVE SUMMARY & AGRONOMIC DECISION SUPPORT
# ==============================================================================
def render_screen4():
    with main_col:
        st.markdown(
            f"""
            <div class="surface-card">
                <h3>{t('screen4_title', curr_lang)}</h3>
                <p class="card-desc">{t('screen4_desc', curr_lang)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sum_col1, sum_col2 = st.columns([1, 1], gap="large")

        # Column 1: Optical Screening Record
        with sum_col1:
            st.markdown("#### 🍃 Optical Screening Record")
            leaf_res = st.session_state.leaf_result

            if leaf_res and leaf_res.get("predicted_class"):
                crop = leaf_res.get("crop", "Unknown")
                disease = leaf_res.get("disease", "Unknown")
                conf = leaf_res.get("confidence", 0.0)
                uncert = leaf_res.get("uncertain", False)

                trans_crop, trans_disease = translate_diagnosis(crop, disease, lang=curr_lang)

                st.markdown(
                    f"""
                    <div class="diagnostic-panel">
                        <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('target_crop', curr_lang)}</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #123D2A; margin-bottom: 0.6rem;">{trans_crop}</div>
                        <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('screened_condition', curr_lang)}</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #1B543A; margin-bottom: 0.6rem;">{trans_disease}</div>
                        <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('confidence', curr_lang)}</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #123D2A;">{conf*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if uncert:
                    st.warning("Flagged: Ambiguous / Low Confidence Screening")
                else:
                    st.success("Flagged: Reliable Optical Confidence")
            else:
                st.info("No leaf screening was completed in this session.")

        # Column 2: Climate Context Record
        with sum_col2:
            st.markdown("#### 🌦️ Environmental Context Record")
            clim_res = st.session_state.climate_result

            if clim_res and clim_res.get("recent"):
                recent = clim_res.get("recent") or {}
                anom = clim_res.get("anomalies") or {}
                loc = st.session_state.climate_loc_name or clim_res.get("location", "Selected Coordinates")

                st.markdown(
                    f"""
                    <div class="diagnostic-panel">
                        <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Location Audited</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #123D2A; margin-bottom: 0.6rem;">{loc}</div>
                        <div style="font-size: 0.78rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Window Observed</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #1B543A; margin-bottom: 0.6rem;">{st.session_state.climate_days} Days via Open-Meteo</div>
                        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
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

        # ------------------------------------------------------------------
        # FEATURE 1: ECONOMIC IMPACT & YIELD RISK ESTIMATOR
        # ------------------------------------------------------------------
        leaf_res = st.session_state.leaf_result
        clim_res = st.session_state.climate_result
        acres = float(st.session_state.farm_acres)

        if leaf_res and leaf_res.get("predicted_class"):
            crop = leaf_res.get("crop", "Tomato")
            disease = leaf_res.get("disease", "Early_blight")
            conf = leaf_res.get("confidence", 0.85)

            econ = estimate_economic_impact(crop, disease, conf, acres)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### 💰 Projected Economic Impact & Yield Risk")

            st.markdown(
                f"""
                <div class="economic-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;">
                        <div>
                            <span style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; color: #123D2A;">
                                Financial Risk Analysis ({acres:.1f} Acres {crop})
                            </span>
                        </div>
                        <div>
                            <span class="loss-stat-pill">{econ['urgency']}</span>
                        </div>
                    </div>
                    <div style="font-size: 0.9rem; color: #334D3F; margin-bottom: 1.2rem;">
                        {econ['description']}
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        <div style="background: #FFFFFF; padding: 1rem; border-radius: 12px; border: 1px solid #E2ECE5;">
                            <div style="font-size: 0.75rem; color: #526E60; text-transform: uppercase; font-weight: 700;">Potential Yield Loss</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: {'#DC2626' if not econ['is_healthy'] else '#15803D'}; font-family: 'Space Grotesk';">
                                {econ['loss_pct_range']}
                            </div>
                        </div>
                        <div style="background: #FFFFFF; padding: 1rem; border-radius: 12px; border: 1px solid #E2ECE5;">
                            <div style="font-size: 0.75rem; color: #526E60; text-transform: uppercase; font-weight: 700;">Revenue at Risk</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #B91C1C; font-family: 'Space Grotesk';">
                                ₹{econ['loss_value_min_inr']:,} – ₹{econ['loss_value_max_inr']:,}
                            </div>
                        </div>
                        <div style="background: #FFFFFF; padding: 1rem; border-radius: 12px; border: 1px solid #E2ECE5;">
                            <div style="font-size: 0.75rem; color: #526E60; text-transform: uppercase; font-weight: 700;">Est. Treatment Cost</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #1E3A8A; font-family: 'Space Grotesk';">
                                ₹{econ['treatment_cost_inr']:,}
                            </div>
                        </div>
                        <div style="background: #FFFFFF; padding: 1rem; border-radius: 12px; border: 1px solid #E2ECE5;">
                            <div style="font-size: 0.75rem; color: #526E60; text-transform: uppercase; font-weight: 700;">Net Harvest Value Saved</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #15803D; font-family: 'Space Grotesk';">
                                {econ['net_savings_inr']} <span class="roi-badge">{econ['roi_multiplier']} ROI</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # BIOMETEOROLOGICAL CONCURRENCE INDEX
        # ------------------------------------------------------------------
        align_res = evaluate_environmental_alignment(leaf_res, clim_res)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"#### {t('alignment_title', curr_lang)}")

        m_col1, m_col2 = st.columns([1.2, 2], gap="large")
        with m_col1:
            st.markdown(
                f"""
                <div class="gauge-card">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Alignment Index</div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 800; color: {align_res['badge_color']};">{align_res['score']}<span style="font-size: 1.2rem; color: #526E60;">/100</span></div>
                    <div style="display: inline-block; background: {align_res['badge_color']}1A; color: {align_res['badge_color']}; font-weight: 700; font-size: 0.85rem; padding: 0.35rem 0.85rem; border-radius: 20px; border: 1px solid {align_res['badge_color']}33; margin-top: 0.5rem;">
                        {align_res['badge']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown("**Biological Concurrence Synthesis:**")
            st.write(align_res["explanation"])
            if align_res.get("matched_factors"):
                with st.expander("🌡️ Matched Pathogen Environmental Vectors", expanded=True):
                    for f in align_res["matched_factors"]:
                        st.write(f"• {f}")
            if align_res.get("abiotic_stress_flag"):
                st.warning(f"⚠️ **Abiotic Divergence Warning:** {align_res.get('abiotic_stress_summary')}")

        # ------------------------------------------------------------------
        # WEATHER-SMART IPM & SPRAY WINDOW
        # ------------------------------------------------------------------
        adv_res = generate_actionable_advisory(leaf_res, clim_res)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"#### {t('ipm_title', curr_lang)}")

        sp_box = adv_res.get("spray_window", {})
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border-left: 5px solid #1B543A; border-radius: 12px; padding: 1.2rem 1.5rem; box-shadow: var(--shadow-card); margin-bottom: 1.2rem;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Spray Application Guidance</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; color: #123D2A; margin: 0.3rem 0;">{sp_box.get('badge', 'Advisory Standby')}</div>
                <div style="font-size: 0.92rem; color: #334D3F;">{sp_box.get('advice', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_cult, tab_bio, tab_chem = st.tabs(["🌾 Cultural Sanitation", "🌿 Biological Shields", "🧪 Targeted Interventions"])
        with tab_cult:
            for item in adv_res.get("cultural", []):
                st.write(f"• {item}")
        with tab_bio:
            for item in adv_res.get("biological", []):
                st.write(f"• {item}")
        with tab_chem:
            for item in adv_res.get("chemical", []):
                st.write(f"• {item}")
            st.caption("⚠️ Chemical treatments are regulatory guidelines. Adhere strictly to local extension labels.")

        # ------------------------------------------------------------------
        # FEATURE 2: 7-DAY ACTIONABLE RECOVERY ROADMAP
        # ------------------------------------------------------------------
        if leaf_res and leaf_res.get("predicted_class"):
            crop = leaf_res.get("crop", "Crop")
            disease = leaf_res.get("disease", "Condition")
            roadmap = generate_recovery_roadmap(crop, disease, sp_box.get("status", "OPTIMAL"))

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### 📅 7-Day Actionable Recovery Roadmap")

            for step in roadmap:
                st.markdown(
                    f"""
                    <div class="roadmap-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                            <span style="font-weight: 700; color: #123D2A; font-size: 1.05rem;">{step['day']}: {step['stage']}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #334D3F; line-height: 1.5;">
                            {step['action']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ------------------------------------------------------------------
        # FEATURE 3: KISAN HELPLINE & 1-CLICK WHATSAPP SHARE
        # ------------------------------------------------------------------
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📲 Field Sharing & Agricultural Helpline")

        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if leaf_res and leaf_res.get("predicted_class"):
                crop = leaf_res.get("crop", "Crop")
                disease = leaf_res.get("disease", "Condition")
                conf = leaf_res.get("confidence", 0.0) * 100.0
                loc = st.session_state.climate_loc_name
                wa_url = build_whatsapp_share_url(crop, disease, conf, loc, "Immediate Action")

                st.markdown(
                    f"""
                    <a href="{wa_url}" target="_blank" style="text-decoration: none;">
                        <div style="background: #25D366; color: #FFFFFF; font-weight: 700; text-align: center;
                                    padding: 0.85rem 1.2rem; border-radius: 12px; font-size: 0.95rem; box-shadow: 0 4px 12px rgba(37,211,102,0.3);">
                            💬 Share Report on WhatsApp (KVK / Agronomist)
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Run leaf screening to enable one-click WhatsApp sharing.")

        with act_col2:
            st.markdown(
                """
                <a href="tel:18001801551" style="text-decoration: none;">
                    <div style="background: #123D2A; color: #B7E66B; font-weight: 700; text-align: center;
                                padding: 0.85rem 1.2rem; border-radius: 12px; font-size: 0.95rem; box-shadow: 0 4px 12px rgba(18,61,42,0.25);">
                        📞 Call Kisan Call Centre (Toll-Free 1800-180-1551)
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # DOWNLOADABLE FIELD DIAGNOSTIC HEALTH CARD
        # ------------------------------------------------------------------
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        cam_img = st.session_state.cam_result.get("overlay_image") if st.session_state.cam_result else None
        farmer_name = st.session_state.farmer_name or "Field Agronomist"
        farm_location = st.session_state.climate_loc_name or (clim_res.get("location", "Agricultural Plot") if clim_res else "Agricultural Plot")

        report_html = generate_field_report_html(
            leaf_result=leaf_res,
            climate_result=clim_res,
            alignment_result=align_res,
            advisory_result=adv_res,
            leaf_image=st.session_state.leaf_image,
            cam_image=cam_img,
            farmer_name=farmer_name,
            farm_location=farm_location,
        )

        rc_col1, rc_col2 = st.columns([1.5, 1])
        with rc_col1:
            st.download_button(
                label=t("btn_download_card", curr_lang),
                data=report_html,
                file_name="CropCare_Field_Health_Card.html",
                mime="text/html",
                type="primary",
                use_container_width=True,
            )
        with rc_col2:
            st.caption("📄 Print-ready HTML Field Health Card. Open in browser and press Ctrl+P to save as official PDF report.")

        # Disclaimer
        st.markdown(
            """
            <div class="disclaimer-banner">
                <strong>Critical Agronomic Safeguards:</strong><br>
                • <strong>Independent Modalities:</strong> Leaf image screening and climate anomaly tracking operate as independent observational streams. Climate anomalies reflect macro-environmental differences and do not prove biotic plant disease.<br>
                • <strong>Screening, Not Diagnosis:</strong> CropCare provides initial decision-support screening. Never apply chemical interventions without on-site agronomic verification.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        nav1, nav2 = st.columns([1, 1])
        with nav1:
            if st.button(t("btn_back_leaf", curr_lang), type="secondary", key="s4_back"):
                st.session_state.screen = 3
                st.rerun()
        with nav2:
            if st.button(t("btn_restart", curr_lang), type="primary", key="s4_restart"):
                st.session_state.screen = 1
                st.session_state.leaf_image = None
                st.session_state.leaf_image_id = None
                st.session_state.leaf_result = None
                st.session_state.cam_result = None
                st.session_state.gemini_vision_res = None
                st.session_state.climate_result = None
                st.rerun()


# ==============================================================================
# RENDER ACTIVE SCREEN
# ==============================================================================
if st.session_state.screen == 1:
    render_screen1()
elif st.session_state.screen == 2:
    render_screen2()
elif st.session_state.screen == 3:
    render_screen3()
elif st.session_state.screen == 4:
    render_screen4()


# ==============================================================================
# PERSISTENT 24/7 AI ASSISTANT IN DEDICATED SIDEBAR
# ==============================================================================
def _submit_chat_message(user_text: str) -> None:
    """Send user text to Gemini with contextual session state and append reply."""
    st.session_state.chat_messages.append({"role": "user", "content": user_text})

    ctx = build_agronomic_context(
        screen=st.session_state.screen,
        leaf_result=st.session_state.leaf_result,
        climate_result=st.session_state.climate_result,
        climate_days=st.session_state.climate_days,
    )
    with st.spinner("Consulting CropCare AI Advisor..."):
        reply = generate_chat_response(
            messages=st.session_state.chat_messages,
            context_str=ctx,
        )
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


def render_sidebar_chatbot():
    with st.sidebar:
        # Header
        st.markdown(
            """
            <div class="sidebar-ai-header">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
                            🌱 CropCare AI Advisor
                        </div>
                        <div style="font-size: 0.75rem; color: #A0C4AB; margin-top: 0.15rem;">
                            Live Agronomic Intelligence · Gemini
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.4rem;">
                        <span class="live-dot"></span>
                        <span style="font-size: 0.75rem; color: #B7E66B; font-weight: 700;">Online</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Context Summary Chip
        leaf_ok = st.session_state.leaf_result is not None
        clim_ok = st.session_state.climate_result is not None

        c_text = "<strong>Current Farm Telemetry:</strong><br>"
        if leaf_ok:
            lr = st.session_state.leaf_result
            c_text += f"🍃 {lr.get('crop')}: {lr.get('disease')} ({lr.get('confidence',0)*100:.1f}%)<br>"
        else:
            c_text += "🍃 Leaf: Pending screening (Screen 2)<br>"

        if clim_ok:
            c_text += f"🌦️ Weather: {st.session_state.climate_loc_name} loaded"
        else:
            c_text += f"🌦️ Weather: {st.session_state.climate_loc_name} (Pending fetch)"

        st.markdown(f'<div class="ai-context-chip">{c_text}</div>', unsafe_allow_html=True)

        # Suggested Questions
        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #A0C4AB; margin-bottom: 0.35rem;'>💬 Quick Agronomic Queries:</div>", unsafe_allow_html=True)
        smart_prompts = [
            "🌱 What is the best organic treatment?",
            "🌦️ Can I spray today with current humidity?",
            "💰 What is my projected economic loss risk?",
            "📅 Give me a 7-day action roadmap",
        ]
        for idx, prompt in enumerate(smart_prompts):
            if st.button(prompt, key=f"sidebar_prompt_{idx}", use_container_width=True):
                _submit_chat_message(prompt)
                st.rerun()

        # Chat Message Container
        chat_box = st.container(height=380)
        with chat_box:
            if not st.session_state.chat_messages:
                st.markdown(
                    """
                    <div style='color: #A0C4AB; font-size: 0.85rem; text-align: center; margin-top: 2rem; line-height: 1.6;'>
                    👋 <strong>Namaste, Farmer!</strong><br>
                    I'm your 24/7 AI Agronomic Advisor.<br>
                    Ask me any question in English, Hindi, or Marathi about your crop, symptoms, weather, or treatments.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                for turn in st.session_state.chat_messages:
                    role = turn.get("role", "user")
                    with st.chat_message(role, avatar="🧑‍🌾" if role == "user" else "🌱"):
                        st.markdown(turn.get("content", ""))

        # Clear Chat Button
        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat History", key="clear_sidebar_chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

        # Chat Input at bottom of sidebar
        chat_user_in = st.chat_input("Ask CropCare AI about your crops...", key="sidebar_chat_input")
        if chat_user_in:
            _submit_chat_message(chat_user_in)
            st.rerun()


# Render sidebar chatbot
render_sidebar_chatbot()
