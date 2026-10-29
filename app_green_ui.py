"""CropCare — Next-Generation Agricultural Intelligence Dashboard.

A premium, guided 4-screen SaaS application providing transfer-learning
leaf screening and hyper-local environmental intelligence.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.cropcare.climate_engine import get_environment
from src.cropcare.image_engine import get_supported_crops, predict_image

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

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--text-dark);
        background-color: var(--ivory-bg);
    }

    /* Main container max-width and padding */
    .block-container {
        max-width: 1140px;
        padding-top: 2rem;
        padding-bottom: 3rem;
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
        margin-bottom: 1.8rem;
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

    /* Step Progress Indicator */
    .step-progress-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 2rem;
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
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        background: #E8ECE6;
        color: #6E8578;
        font-weight: 700;
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
        height: 2px;
        background: #E8ECE6;
        margin: 0 0.8rem;
    }
    .step-divider.active {
        background: var(--forest-dark);
    }

    /* Hero Card (Screen 1) */
    .hero-container {
        background: linear-gradient(135deg, #123D2A 0%, #1A543A 55%, #103222 100%);
        color: #FFFFFF;
        border-radius: 24px;
        padding: 3rem 2.8rem;
        box-shadow: var(--shadow-hero);
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
        border: 1px solid rgba(183, 230, 107, 0.2);
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE INITIALIZATION (Zero-Loss Navigation)
# ==============================================================================
if "screen" not in st.session_state:
    st.session_state.screen = 1  # 1: Welcome, 2: Leaf, 3: Climate, 4: Summary

if "leaf_image" not in st.session_state:
    st.session_state.leaf_image = None

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
# TOP BRANDING & PROGRESS BAR
# ==============================================================================
st.markdown(
    """
    <div class="brand-bar">
        <div class="brand-logo">🌱 CropCare™</div>
        <div class="brand-tag">Agronomic Intelligence Platform</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Persistent Guided Step Indicator
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
            <div class="step-pill">1</div>
            <span>Welcome</span>
        </div>
        <div class="step-divider {div1_cls}"></div>
        <div class="step-item {step2_cls}">
            <div class="step-pill">2</div>
            <span>Leaf Screening</span>
        </div>
        <div class="step-divider {div2_cls}"></div>
        <div class="step-item {step3_cls}">
            <div class="step-pill">3</div>
            <span>Climate Insights</span>
        </div>
        <div class="step-divider {div3_cls}"></div>
        <div class="step-item {step4_cls}">
            <div class="step-pill">4</div>
            <span>Executive Summary</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SCREEN 1: WELCOME & START
# ==============================================================================
if st.session_state.screen == 1:
    hero_col_left, hero_col_right = st.columns([1.3, 0.9], gap="large")

    with hero_col_left:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-badge">⚡ Precision Agricultural Technology</div>
                <div class="hero-title">Read the leaf.<br><span>Read the weather.</span></div>
                <div class="hero-sub">
                    CropCare bridges transfer-learning computer vision with hyper-local climate intelligence.
                    Screen crop leaves for preliminary visual symptoms while auditing environmental stress 
                    factors through historical weather baselines.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1.5])
        with c1:
            if st.button("Start Screening →", type="primary", use_container_width=True):
                st.session_state.screen = 2
                st.rerun()

    with hero_col_right:
        st.markdown(
            """
            <div class="botanical-3d-card" style="background: linear-gradient(145deg, #184C34 0%, #0F2E20 100%);">
                <svg width="220" height="220" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 15px 25px rgba(0,0,0,0.35));">
                    <defs>
                        <radialGradient id="leafGlow" cx="50%" cy="30%" r="70%">
                            <stop offset="0%" stop-color="#D9F99D" stop-opacity="0.9"/>
                            <stop offset="50%" stop-color="#65A30D" stop-opacity="0.8"/>
                            <stop offset="100%" stop-color="#14532D" stop-opacity="1"/>
                        </radialGradient>
                        <linearGradient id="stemGrad" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#B7E66B"/>
                            <stop offset="100%" stop-color="#1B543A"/>
                        </linearGradient>
                        <filter id="shadow3D" x="-20%" y="-20%" width="140%" height="140%">
                            <feDropShadow dx="3" dy="10" stdDeviation="8" flood-color="#000000" flood-opacity="0.4"/>
                        </filter>
                    </defs>
                    <path d="M100 20 C140 50 170 110 100 180 C30 110 60 50 100 20 Z" fill="url(#leafGlow)" filter="url(#shadow3D)"/>
                    <path d="M100 20 Q105 100 100 180" stroke="url(#stemGrad)" stroke-width="4" stroke-linecap="round"/>
                    <path d="M100 60 Q130 50 145 70" stroke="#B7E66B" stroke-width="2" stroke-linecap="round" stroke-opacity="0.7"/>
                    <path d="M100 90 Q70 80 55 100" stroke="#B7E66B" stroke-width="2" stroke-linecap="round" stroke-opacity="0.7"/>
                    <path d="M100 120 Q130 110 140 130" stroke="#B7E66B" stroke-width="2" stroke-linecap="round" stroke-opacity="0.7"/>
                    <path d="M100 145 Q75 140 65 155" stroke="#B7E66B" stroke-width="2" stroke-linecap="round" stroke-opacity="0.7"/>
                </svg>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; color: #FFFFFF; margin-top: 1rem;">
                    MobileNetV3 Neural Core
                </div>
                <div class="botanical-badge">38-Class Transfer Learning Transfer Benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Three Core Pillars
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3, gap="medium")

    with f_col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🍃</div>
                <div class="feature-title">Targeted Crop Screening</div>
                <div class="feature-body">
                    Select your crop species (Tomato, Potato, Corn, Apple, etc.) to isolate known pathologies,
                    eliminating cross-species confusion and delivering maximum real-time confidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌦️</div>
                <div class="feature-title">Historical Weather Baselines</div>
                <div class="feature-body">
                    Automated Open-Meteo multi-year comparisons highlighting temperature and precipitation
                    deviations over 7, 14, or 30 days.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <div class="feature-title">Agronomic Safeguards</div>
                <div class="feature-body">
                    Strict boundary between statistical model certainty and confirmed biology. Always 
                    advocates field verification before chemical intervention.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# SCREEN 2: LEAF SCREENING
# ==============================================================================
elif st.session_state.screen == 2:
    st.markdown(
        """
        <div class="surface-card">
            <h3>🍃 Step 2: Optical Leaf Health Screening</h3>
            <p class="card-desc">
                Select your crop type or use Auto-Detect. Upload an uncompressed leaf photograph or select
                from curated test samples. The neural model will isolate conditions and calculate calibrated confidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # CROP SELECTOR BAR & TARGET ISOLATION
    # --------------------------------------------------------------------------
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
        "Choose Crop Type to isolate diagnosis (or keep Auto-Detect):",
        options=crop_labels,
        index=curr_idx,
        help="Targeting your crop eliminates 28+ irrelevant pathologies from the neural softmax, boosting precision and confidence.",
    )

    chosen_crop_key = crop_keys[crop_labels.index(selected_label)]
    if chosen_crop_key != st.session_state.selected_crop_type:
        st.session_state.selected_crop_type = chosen_crop_key
        # Re-run inference automatically if an image is already uploaded
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

    # --------------------------------------------------------------------------
    # CROPS & DISEASES ENCYCLOPEDIA EXPANDER
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # QUICK SAMPLE SELECTOR
    # --------------------------------------------------------------------------
    st.markdown("##### ⚡ Quick-Test with Curated Botanical Samples:")
    sm_c1, sm_c2, sm_c3, sm_c4, sm_c5 = st.columns(5)
    sample_items = [
        (sm_c1, "🍅 Tomato Early Blight", "data/sample_leaves/sample_tomato_early_blight.png", "Tomato"),
        (sm_c2, "🥔 Potato Late Blight", "data/sample_leaves/sample_potato_late_blight.png", "Potato"),
        (sm_c3, "🌽 Corn Rust", "data/sample_leaves/sample_corn_common_rust.png", "Corn_(maize)"),
        (sm_c4, "🍎 Apple Scab", "data/sample_leaves/sample_apple_scab.png", "Apple"),
        (sm_c5, "🌿 Tomato Healthy", "data/sample_leaves/sample_tomato_healthy.png", "Tomato"),
    ]

    for col, btn_label, file_path, def_crop in sample_items:
        with col:
            if st.button(btn_label, use_container_width=True):
                if os.path.exists(file_path):
                    loaded_img = Image.open(file_path)
                    st.session_state.leaf_image = loaded_img
                    st.session_state.selected_crop_type = def_crop
                    st.session_state.leaf_result = predict_image(
                        loaded_img,
                        crop_filter=def_crop,
                        use_tta=st.session_state.use_tta_boost,
                    )
                    st.rerun()

    # --------------------------------------------------------------------------
    # MAIN UPLOADER & PREVIEW
    # --------------------------------------------------------------------------
    leaf_col_left, leaf_col_right = st.columns([1, 1], gap="large")

    with leaf_col_left:
        st.markdown("#### 📤 Upload Leaf Photograph")
        uploaded_file = st.file_uploader(
            "Upload image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-contrast, focused imagery centered on a single leaf produces the most reliable screening.",
        )

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.session_state.leaf_image = img
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
            run_btn_label = f"Run {chosen_crop_key.replace('_', ' ')} Screening ⚡" if chosen_crop_key != "All Crops (Auto-Detect)" else "Run Multi-Crop Screening ⚡"
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

    # --------------------------------------------------------------------------
    # DIAGNOSTIC OUTPUT
    # --------------------------------------------------------------------------
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
                st.info(
                    "ℹ️ **Inference Engine in Demo Mode**\n\n"
                    "Model weights or mapping files are missing from the `models/` directory.",
                    icon="ℹ️",
                )
        else:
            # Show image-quality warnings if any
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
                conf = res.get("confidence", 0.0)
                uncertain = res.get("uncertain", False)
                crop_filtered = res.get("crop_filter_applied", False)

                # Visual badge indicating whether diagnosis was isolated or auto-detected
                if crop_filtered:
                    mode_badge = f"""<span class="crop-badge-pill" style="margin-bottom: 0.6rem;">🎯 Targeted Crop Mode: {crop}</span>"""
                else:
                    det_crop = res.get("detected_crop", crop)
                    det_conf = res.get("detected_crop_confidence", 0.0) * 100
                    mode_badge = f"""<span class="crop-badge-pill" style="background: #E8ECE6; color: #1B543A; margin-bottom: 0.6rem;">🔍 Auto-Detected: {det_crop} ({det_conf:.1f}% crop certainty)</span>"""

                st.markdown(
                    f"""
                    <div class="diagnostic-panel">
                        {mode_badge}
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; margin-top: 0.4rem;">
                            <div>
                                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">Crop Species</div>
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #123D2A;">{crop}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">Confidence</div>
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #123D2A;">{conf*100:.1f}%</div>
                            </div>
                        </div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">Screened Condition</div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: #1B543A; margin-bottom: 0.8rem;">{disease}</div>
                        <div style="font-size: 0.75rem; color: #526E60;">Model: <code>{res.get('model_name', 'MobileNetV3-Small-PlantVillage')}</code> | TTA: <strong>{'Enabled' if st.session_state.use_tta_boost else 'Disabled'}</strong></div>
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

                # If user used Auto-Detect and had lower confidence, offer 1-click crop focus
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

                # Probability Distribution Bars
                probs = res.get("probabilities", {})
                if probs:
                    with st.expander("📊 View Condition Probability Breakdown", expanded=True):
                        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        # Show top conditions
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

    # Guided Screen 2 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button("← Back to Welcome", type="secondary"):
            st.session_state.screen = 1
            st.rerun()
    with n_col2:
        if st.button("Continue to Climate Insights →", type="primary"):
            st.session_state.screen = 3
            st.rerun()


# ==============================================================================
# SCREEN 3: CLIMATE INSIGHTS
# ==============================================================================
elif st.session_state.screen == 3:
    st.markdown(
        """
        <div class="surface-card">
            <h3>🌦️ Step 3: Hyper-Local Environmental Analytics</h3>
            <p class="card-desc">
                Retrieve historical climate observations and multi-year anomaly baselines via Open-Meteo.
                Environmental stress (heat, moisture deficit, humidity) often predisposes crops to pathogen vulnerability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("climate_query_form"):
        c_input1, c_input2, c_input3 = st.columns([1, 1, 1])
        with c_input1:
            lat = st.number_input(
                "Latitude (°N)",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.climate_lat),
                step=0.0001,
                format="%.4f",
            )
        with c_input2:
            lon = st.number_input(
                "Longitude (°E)",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.climate_lon),
                step=0.0001,
                format="%.4f",
            )
        with c_input3:
            window = st.selectbox(
                "Historical Weather Window",
                options=[7, 14, 30],
                index=[7, 14, 30].index(st.session_state.climate_days)
                if st.session_state.climate_days in [7, 14, 30]
                else 2,
            )

        submit_weather = st.form_submit_button("Fetch Climate Insights ⚡", use_container_width=True)

    if submit_weather:
        st.session_state.climate_lat = lat
        st.session_state.climate_lon = lon
        st.session_state.climate_days = window

        with st.spinner(f"Querying Open-Meteo for {window}-day climate baseline..."):
            try:
                from src.cropcare.climate_engine import get_environment

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
        st.markdown(f"#### Recent Summary ({st.session_state.climate_days} Days)")

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
        st.markdown("#### Climate Anomalies vs Historical Baseline")
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

    # Guided Screen 3 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button("← Back to Leaf Screening", type="secondary"):
            st.session_state.screen = 2
            st.rerun()
    with n_col2:
        if st.button("Proceed to Executive Summary →", type="primary"):
            st.session_state.screen = 4
            st.rerun()


# ==============================================================================
# SCREEN 4: EXECUTIVE SUMMARY & AGRONOMIC BRIEFING
# ==============================================================================
elif st.session_state.screen == 4:
    st.markdown(
        """
        <div class="surface-card">
            <h3>📋 Step 4: Executive Agronomic Summary</h3>
            <p class="card-desc">
                Synthesized briefing consolidating optical leaf screening and hyper-local environmental data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sum_col1, sum_col2 = st.columns([1, 1], gap="large")

    # Column 1: Leaf Screening Audit
    with sum_col1:
        st.markdown("#### 🍃 Optical Screening Record")
        leaf_res = st.session_state.leaf_result

        if leaf_res and leaf_res.get("predicted_class"):
            crop = leaf_res.get("crop", "Unknown")
            disease = leaf_res.get("disease", "Unknown")
            conf = leaf_res.get("confidence", 0.0)
            uncert = leaf_res.get("uncertain", False)
            crop_filtered = leaf_res.get("crop_filter_applied", False)

            mode_label = "Targeted Crop Focus" if crop_filtered else "Multi-Crop Auto-Detect"

            st.markdown(
                f"""
                <div class="diagnostic-panel">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Screening Mode</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #1B543A; margin-bottom: 0.6rem;">{mode_label}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Crop Identified</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #123D2A; margin-bottom: 0.6rem;">{crop}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Visual Condition</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #1B543A; margin-bottom: 0.6rem;">{disease}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #526E60; text-transform: uppercase;">Confidence Score</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #123D2A;">{conf*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if uncert:
                st.warning("Flagged: Ambiguous / Low Confidence Screening. Field verification advised.")
            else:
                st.success("Flagged: High Optical Confidence Diagnosis")
        else:
            st.info("No leaf screening was completed in this session.")

    # Column 2: Climate Context Audit
    with sum_col2:
        st.markdown("#### 🌦️ Environmental Context Record")
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
        if st.button("← Back to Climate Insights", type="secondary"):
            st.session_state.screen = 3
            st.rerun()
    with n_col2:
        if st.button("🔄 Start New Screening Session", type="primary"):
            st.session_state.screen = 1
            st.session_state.leaf_image = None
            st.session_state.leaf_result = None
            st.session_state.climate_result = None
            st.session_state.selected_crop_type = "All Crops (Auto-Detect)"
            st.rerun()
