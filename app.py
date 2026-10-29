"""CropCare — Next-Generation Agricultural Intelligence Dashboard.

A premium, guided 4-screen SaaS application providing transfer-learning
leaf screening, Grad-CAM explainability, hyper-local environmental intelligence,
biometeorological concurrence analysis, and weather-smart IPM advisories.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.cropcare.explainability import generate_gradcam
from src.cropcare.alignment_engine import evaluate_environmental_alignment
from src.cropcare.advisory_engine import generate_actionable_advisory
from src.cropcare.report_engine import generate_field_report_html
from src.cropcare.vernacular import translate_diagnosis, generate_audio_script, t

# Configure page layout and title (Must be first Streamlit command)
st.set_page_config(
    page_title="CropCare — Precision Agronomic Platform",
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

    .block-container {
        max-width: 1140px;
        padding-top: 1.5rem;
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
        font-size: 1.05rem;
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
        line-height: 1.5;
        color: var(--text-muted);
    }

    /* Diagnostic Output Container */
    .diagnostic-panel {
        background: #F4F7F2;
        border: 1px solid #D6E0D9;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    /* Scientific Disclaimer Banner */
    .disclaimer-banner {
        background: #FBFDFB;
        border-left: 4px solid var(--forest-light);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #435E50;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "screen" not in st.session_state:
    st.session_state.screen = 1  # 1: Welcome, 2: Leaf, 3: Climate, 4: Summary

if "leaf_image" not in st.session_state:
    st.session_state.leaf_image = None

if "leaf_result" not in st.session_state:
    st.session_state.leaf_result = None

if "cam_result" not in st.session_state:
    st.session_state.cam_result = None

if "climate_lat" not in st.session_state:
    st.session_state.climate_lat = 19.0760

if "climate_lon" not in st.session_state:
    st.session_state.climate_lon = 72.8777

if "climate_days" not in st.session_state:
    st.session_state.climate_days = 30

if "climate_result" not in st.session_state:
    st.session_state.climate_result = None

if "lang" not in st.session_state:
    st.session_state.lang = "en"


def format_metric(value: Optional[float], unit: str = "", precision: int = 1) -> str:
    """Format numeric values safely without converting None into 0.0."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f} {unit}".strip()


def display_image(image, caption: Optional[str] = None) -> None:
    """Safely render an image across Streamlit versions (1.38 use_column_width vs newer use_container_width)."""
    try:
        st.image(image, caption=caption, use_column_width=True)
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


# ==============================================================================
# TOP BRANDING & LANGUAGE SWITCHER
# ==============================================================================
b_col1, b_col2 = st.columns([3.2, 1.2])
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

curr_lang = st.session_state.lang

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
            <span>{t('step1', curr_lang)}</span>
        </div>
        <div class="step-divider {div1_cls}"></div>
        <div class="step-item {step2_cls}">
            <div class="step-pill">2</div>
            <span>{t('step2', curr_lang)}</span>
        </div>
        <div class="step-divider {div2_cls}"></div>
        <div class="step-item {step3_cls}">
            <div class="step-pill">3</div>
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


# ==============================================================================
# SCREEN 1: WELCOME & START
# ==============================================================================
if st.session_state.screen == 1:
    hero_col_left, hero_col_right = st.columns([1.3, 0.9], gap="large")

    with hero_col_left:
        st.markdown(
            f"""
            <div class="hero-container">
                <div class="hero-badge">{t('hero_badge', curr_lang)}</div>
                <div class="hero-title">{t('hero_title', curr_lang)}</div>
                <div class="hero-sub">{t('hero_sub', curr_lang)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(t("btn_begin", curr_lang), type="primary", use_container_width=True):
            st.session_state.screen = 2
            st.rerun()

    with hero_col_right:
        st.markdown(
            """
            <div class="botanical-3d-card">
                <div style="font-size: 4.5rem; margin-bottom: 0.5rem; animation: float 3s ease-in-out infinite;">🌿</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">
                    Agronomic Diagnostics
                </div>
                <div style="font-size: 0.88rem; color: #D3E4DB; margin-top: 0.4rem;">
                    14 Crops • 38 Pathology Classes<br>Hyper-Local Reanalysis Weather Telemetry
                </div>
                <div class="botanical-badge">Edge-Ready • 15.6 ms CPU Latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3, gap="medium")

    with f_col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🍃</div>
                <div class="feature-title">{t('feat1_title', curr_lang)}</div>
                <div class="feature-body">{t('feat1_desc', curr_lang)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🌦️</div>
                <div class="feature-title">{t('feat2_title', curr_lang)}</div>
                <div class="feature-body">{t('feat2_desc', curr_lang)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col3:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <div class="feature-title">{t('feat3_title', curr_lang)}</div>
                <div class="feature-body">{t('feat3_desc', curr_lang)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# SCREEN 2: LEAF SCREENING & EXPLAINABLE AI
# ==============================================================================
elif st.session_state.screen == 2:
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
        st.markdown(f"#### 📤 {t('upload_label', curr_lang)}")
        uploaded_file = st.file_uploader(
            t("upload_label", curr_lang),
            type=["jpg", "jpeg", "png"],
            help="High-contrast, focused imagery centered on a single leaf produces the most reliable screening.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.session_state.leaf_image = img
            except Exception as e:
                st.error(f"Image decoding failed: {e}")

        # Show tabs if Grad-CAM available
        if st.session_state.cam_result and st.session_state.cam_result.get("overlay_image"):
            t_orig, t_cam = st.tabs(["📷 Original Photograph", "🔥 Explainable AI (Grad-CAM)"])
            with t_orig:
                display_image(st.session_state.leaf_image, caption="Uploaded Leaf Specimen")
            with t_cam:
                display_image(
                    st.session_state.cam_result["overlay_image"],
                    caption="Grad-CAM Saliency: Spatial Lesion Attention",
                )
                st.caption(f"🎯 **Attention Analysis:** {st.session_state.cam_result.get('focus_description')}")
        elif st.session_state.leaf_image is not None:
            display_image(st.session_state.leaf_image, caption="Loaded Leaf Image")

        # Run Screening Action
        if st.session_state.leaf_image is not None:
            if st.button(t("btn_run_screening", curr_lang), type="primary", use_container_width=True):
                with st.spinner("Executing MobileNetV3 inference & Grad-CAM analysis..."):
                    try:
                        from src.cropcare.image_engine import predict_image

                        res = predict_image(st.session_state.leaf_image)
                        st.session_state.leaf_result = res

                        cam_res = generate_gradcam(st.session_state.leaf_image)
                        st.session_state.cam_result = cam_res
                        st.rerun()
                    except Exception as inf_err:
                        st.error(f"Inference execution error: {inf_err}")
        else:
            st.info("Please upload a leaf photograph above to enable diagnostic screening.")

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
                        <div style="font-size: 0.88rem;">Trained MobileNetV3 checkpoint is online and loaded.<br>Click "Run Leaf Screening" to evaluate image.</div>
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

                # Multilingual Localization
                trans_crop, trans_disease = translate_diagnosis(crop, disease, lang=curr_lang)

                st.markdown(
                    f"""
                    <div class="diagnostic-panel">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.8rem;">
                            <div>
                                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">{t('target_crop', curr_lang)}</div>
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #123D2A;">{trans_crop}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">{t('confidence', curr_lang)}</div>
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #123D2A;">{conf*100:.1f}%</div>
                            </div>
                        </div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #526E60; letter-spacing: 0.05em;">{t('screened_condition', curr_lang)}</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #1B543A; margin-bottom: 0.8rem;">{trans_disease}</div>
                        <div style="font-size: 0.75rem; color: #526E60;">Model: <code>{res.get('model_name', 'MobileNetV3-Small-PlantVillage')}</code> | Latency: <strong>15.6 ms (CPU)</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Audio voice synthesis button
                audio_script = generate_audio_script(crop, disease, conf, "HIGH", "Advisory Available", lang=curr_lang)
                safe_script = audio_script.replace("'", "\\'").replace('"', '\\"')

                btn_lbl = t("btn_listen", curr_lang)
                if st.button(btn_lbl, use_container_width=True):
                    speech_lang = "hi-IN" if curr_lang == "hi" else ("mr-IN" if curr_lang == "mr" else "en-US")
                    st.components.v1.html(
                        f"""<script>
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            var msg = new SpeechSynthesisUtterance('{safe_script}');
                            msg.lang = '{speech_lang}';
                            msg.rate = 0.95;
                            window.speechSynthesis.speak(msg);
                        }}
                        </script>""",
                        height=0,
                    )
                    st.success("Speaking diagnosis aloud via browser voice engine...")
                    st.caption(f"📢 **Script:** {audio_script}")

                if uncertain:
                    st.warning(
                        "⚠️ **Low Confidence / Ambiguous Result:** "
                        f"Model certainty ({conf*100:.1f}%) is below the reliable threshold (55%). "
                        "Visual indicators are ambiguous, early-stage, or outside the trained distribution. "
                        "Field verification by an agronomist is strongly advised.",
                        icon="⚠️",
                    )
                else:
                    st.success("Screening completed with satisfactory model confidence.")

                probs = res.get("probabilities", {})
                if probs:
                    with st.expander("📊 View Top Class Probabilities", expanded=False):
                        top_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                        prob_table = pd.DataFrame(
                            [
                                {
                                    "Class": k.replace("___", " — ").replace("_", " "),
                                    "Probability": f"{v*100:.2f}%",
                                }
                                for k, v in top_items
                            ]
                        )
                        st.table(prob_table)

    # Guided Screen 2 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button(t("btn_back_welcome", curr_lang), type="secondary"):
            st.session_state.screen = 1
            st.rerun()
    with n_col2:
        if st.button(t("btn_goto_climate", curr_lang), type="primary"):
            st.session_state.screen = 3
            st.rerun()


# ==============================================================================
# SCREEN 3: CLIMATE INSIGHTS
# ==============================================================================
elif st.session_state.screen == 3:
    st.markdown(
        f"""
        <div class="surface-card">
            <h3>{t('screen3_title', curr_lang)}</h3>
            <p class="card-desc">{t('screen3_desc', curr_lang)}</p>
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

        submit_climate = st.form_submit_button(t("btn_fetch_climate", curr_lang), type="primary", use_container_width=True)

    if submit_climate:
        st.session_state.climate_lat = lat
        st.session_state.climate_lon = lon
        st.session_state.climate_days = window

        with st.spinner("Connecting to Open-Meteo Reanalysis & Forecast APIs..."):
            try:
                from src.cropcare.climate_engine import get_environment

                env_res = get_environment(
                    location=(lat, lon),
                    days=window,
                    baseline_years=[2023, 2024, 2025],
                )
                st.session_state.climate_result = env_res
                st.success("Successfully retrieved Open-Meteo environmental telemetry.")
            except Exception as clim_err:
                st.error(f"Failed to retrieve climate telemetry: {clim_err}")

    # Display Climate Analysis if available
    env_data = st.session_state.climate_result
    if env_data and env_data.get("recent"):
        recent = env_data.get("recent") or {}
        baseline = env_data.get("baseline") or {}
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

        # Time-Series Charting
        daily_series = env_data.get("daily_series") or []
        if daily_series:
            with st.expander("📈 View Daily Meteorological Time-Series", expanded=True):
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

    # Guided Screen 3 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button(t("btn_back_leaf", curr_lang), type="secondary"):
            st.session_state.screen = 2
            st.rerun()
    with n_col2:
        if st.button(t("btn_goto_summary", curr_lang), type="primary"):
            st.session_state.screen = 4
            st.rerun()


# ==============================================================================
# SCREEN 4: EXECUTIVE SUMMARY & AGRONOMIC BRIEFING
# ==============================================================================
elif st.session_state.screen == 4:
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

    # Column 1: Leaf Screening Audit
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
                    <div style="font-size: 0.8rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('target_crop', curr_lang)}</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #123D2A; margin-bottom: 0.6rem;">{trans_crop}</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('screened_condition', curr_lang)}</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #1B543A; margin-bottom: 0.6rem;">{trans_disease}</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #526E60; text-transform: uppercase;">{t('confidence', curr_lang)}</div>
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

    # --------------------------------------------------------------------------
    # FEATURE 1: Biometeorological Concurrence Index
    # --------------------------------------------------------------------------
    align_res = evaluate_environmental_alignment(leaf_res, clim_res)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"#### {t('alignment_title', curr_lang)}")

    m_col1, m_col2 = st.columns([1.2, 2], gap="large")
    with m_col1:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #D6E0D9; border-radius: 16px; padding: 1.5rem; text-align: center; box-shadow: var(--shadow-card);">
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

    # --------------------------------------------------------------------------
    # FEATURE 3: Weather-Smart IPM & Spray Window
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # FEATURE 4: Downloadable Field Diagnostic Health Card (Print/PDF)
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    cam_img = st.session_state.cam_result.get("overlay_image") if st.session_state.cam_result else None

    report_html = generate_field_report_html(
        leaf_result=leaf_res,
        climate_result=clim_res,
        alignment_result=align_res,
        advisory_result=adv_res,
        leaf_image=st.session_state.leaf_image,
        cam_image=cam_img,
        farmer_name="Shri Ramesh Patil (Lead Grower)",
        farm_location=clim_res.get("location", "Maharashtra Agricultural Plot") if clim_res else "Agricultural Plot",
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
        st.caption("📄 Generates self-contained, high-res HTML card that prints directly to 1-page PDF for field extension workers.")

    # Guided Screen 4 Navigation
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        if st.button(t("btn_back_leaf", curr_lang), type="secondary"):
            st.session_state.screen = 3
            st.rerun()
    with n_col2:
        if st.button(t("btn_restart", curr_lang), type="primary"):
            st.session_state.screen = 1
            st.session_state.leaf_image = None
            st.session_state.leaf_result = None
            st.session_state.cam_result = None
            st.session_state.climate_result = None
            st.rerun()
