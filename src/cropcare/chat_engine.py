"""CropCare AI Assistant — Gemini Integration Module.

Provides agronomic chatbot intelligence powered by the Google Gemini API,
integrating real-time session context (optical leaf predictions and climate anomalies)
with strict agricultural safety protocols, plus Gemini Multimodal Vision verification.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """You are the CropCare AI Assistant, an expert precision agronomy guide assisting farmers, growers, and agricultural students.

Core Agronomic Safeguards & Methodology:
1. Preliminary Screening: Predictions from the MobileNetV3 model (trained on the PlantVillage benchmark dataset) are preliminary optical screening assessments, NOT confirmed biological or pathological diagnoses.
2. Calibrated Uncertainty: If the model flagged uncertainty (confidence below 55%) or image quality warnings (e.g. low contrast, small size), explicitly emphasize that visual patterns are ambiguous and recommend capturing clear, well-lit, centered leaf photographs or obtaining on-site agronomic inspection.
3. Abiotic vs Biotic Distinction: Climate observations and multi-year anomalies describe environmental conditions (heat stress, drought, excess moisture) that can predispose crops to disease, but weather deviations DO NOT prove a biological pathogen is present.
4. Chemical Safety & Responsible Guidance: Never prescribe unverified chemical dosages, off-label pesticide mixtures, or dangerous chemical recommendations. Recommend sustainable cultural practices, field sanitation, and Integrated Pest Management (IPM), and urge consultation with local agricultural extension services or certified agronomists before applying chemical interventions.
5. Factual Grounding: Never fabricate missing sensor readings or invent predictions. If screening or climate data is missing from the session context, clearly inform the user.
6. Tone & Clarity: Deliver clear, concise, actionable, and encouraging advice accessible to non-technical agriculturalists.
"""

SUGGESTED_PROMPTS = [
    "Explain my leaf screening result",
    "What do the climate anomalies mean?",
    "How can I take a better leaf photo?",
    "What are good cultural practices for late blight?",
]


def get_gemini_api_key() -> Optional[str]:
    """Retrieve the Gemini API key from Streamlit secrets or environment variables."""
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
            if key and str(key).strip():
                return str(key).strip()
    except Exception:
        pass

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    return None


def get_configured_model() -> str:
    """Retrieve the configured Gemini model name with fallback to DEFAULT_GEMINI_MODEL."""
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
            m = str(st.secrets["GEMINI_MODEL"]).strip()
            if m:
                return m
    except Exception:
        pass

    env_model = os.environ.get("GEMINI_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()

    return DEFAULT_GEMINI_MODEL


def build_agronomic_context(
    screen: int = 1,
    leaf_result: Optional[Dict[str, Any]] = None,
    climate_result: Optional[Dict[str, Any]] = None,
    climate_days: int = 30,
) -> str:
    """Construct structured session context for the Gemini prompt from session state."""
    screen_names = {
        1: "Welcome / Start",
        2: "Leaf Screening",
        3: "Climate Insights",
        4: "Executive Summary",
    }
    active_screen_name = screen_names.get(screen, f"Step {screen}")

    lines: List[str] = [
        f"Active Application Step: {active_screen_name}",
        "",
        "--- Leaf Vision Screening Assessment ---",
    ]

    if leaf_result and leaf_result.get("predicted_class"):
        crop = leaf_result.get("crop", "Unknown")
        disease = leaf_result.get("disease", "Unknown")
        conf = leaf_result.get("confidence", 0.0)
        uncertain = leaf_result.get("uncertain", False)
        model_name = leaf_result.get("model_name", "MobileNetV3-Small")
        warnings = leaf_result.get("warnings", [])

        lines.append(f"Model Name: {model_name}")
        lines.append(f"Identified Crop: {crop}")
        lines.append(f"Detected Visual Condition: {disease}")
        lines.append(f"Statistical Confidence: {conf * 100:.1f}%")
        lines.append(f"Uncertainty Flag: {'TRUE (Low confidence / Ambiguous visual)' if uncertain else 'FALSE'}")

        if warnings:
            lines.append("Model Advisories: " + " | ".join(warnings))

        probs = leaf_result.get("probabilities", {})
        if probs:
            top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_str = ", ".join([f"{k.replace('___', ' - ')} ({v*100:.1f}%)" for k, v in top3])
            lines.append(f"Top Candidate Classes: {top3_str}")
    else:
        lines.append("Optical Screening: No leaf image has been screened yet in this session.")

    lines.append("")
    lines.append("--- Hyper-Local Climate Context ---")

    if climate_result and climate_result.get("recent"):
        loc = climate_result.get("location", "Queried Coordinates")
        provider = climate_result.get("source", "Open-Meteo")
        recent = climate_result.get("recent") or {}
        anomalies = climate_result.get("anomalies") or {}
        c_warnings = climate_result.get("warnings", [])

        lines.append(f"Location: {loc} (Provider: {provider})")
        lines.append(f"Observation Window: {climate_days} Days")

        def _fmt(val: Optional[float], unit: str) -> str:
            return f"{val:.1f} {unit}" if val is not None else "Data Unavailable"

        lines.append(f"Mean Temperature: {_fmt(recent.get('mean_temperature_c'), '°C')}")
        lines.append(f"Max Temperature: {_fmt(recent.get('max_temperature_c'), '°C')}")
        lines.append(f"Total Rainfall: {_fmt(recent.get('rainfall_mm'), 'mm')}")
        lines.append(f"Mean Humidity: {_fmt(recent.get('mean_humidity_pct'), '%')}")
        lines.append(f"Evapotranspiration ET0: {_fmt(recent.get('et0_mm'), 'mm')}")

        t_anom = anomalies.get("temperature_c")
        r_anom_mm = anomalies.get("rainfall_mm")
        r_anom_pct = anomalies.get("rainfall_pct")
        lines.append(
            f"Temperature Anomaly vs Baseline: {_fmt(t_anom, '°C') if t_anom is not None else 'N/A'}"
        )
        lines.append(
            f"Rainfall Anomaly vs Baseline: {_fmt(r_anom_mm, 'mm')} ({_fmt(r_anom_pct, '%') if r_anom_pct is not None else 'N/A'})"
        )
        if c_warnings:
            lines.append("Climate Advisories: " + " | ".join(c_warnings))
    else:
        lines.append("Climate Analytics: No weather data has been retrieved yet in this session.")

    return "\n".join(lines)


def generate_chat_response(
    messages: List[Dict[str, str]],
    context_str: str,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> str:
    """Generate an agronomic response using the official google-genai SDK."""
    if not messages:
        return "How can I assist you with your crops today? You can ask about leaf symptoms, climate anomalies, or screening instructions."

    resolved_key = api_key or get_gemini_api_key()
    if not resolved_key and client is None:
        return (
            "⚠️ **Gemini API Key Required**\n\n"
            "The CropCare AI Assistant is ready, but requires a Google Gemini API key to communicate:\n\n"
            "1. **Local Development:** Add `GEMINI_API_KEY = \"your_api_key\"` into `.streamlit/secrets.toml` "
            "or set the environment variable `export GEMINI_API_KEY=\"your_key\"`.\n"
            "2. **Streamlit Community Cloud:** Add `GEMINI_API_KEY` under your App Settings > Secrets.\n\n"
            "*Note: All leaf screening and climate analytics continue to operate normally without an API key.*"
        )

    resolved_model = model_name or get_configured_model()

    if client is None:
        try:
            from google import genai

            client = genai.Client(api_key=resolved_key)
        except Exception as e:
            logger.exception("Failed to initialize Google GenAI Client")
            return f"⚠️ Unable to initialize Google Gemini client: {e}"

    recent_turns = messages[-10:]
    formatted_contents: List[Any] = []
    for turn in recent_turns:
        role = "user" if turn.get("role") == "user" else "model"
        content_text = str(turn.get("content", ""))
        formatted_contents.append({"role": role, "parts": [{"text": content_text}]})

    full_system_instruction = f"{SYSTEM_INSTRUCTION}\n\n### Current CropCare Application Session Context:\n{context_str}"

    try:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=full_system_instruction,
            temperature=0.4,
        )
        response = client.models.generate_content(
            model=resolved_model,
            contents=formatted_contents,
            config=config,
        )

        if response and response.text:
            return response.text.strip()
        return "I processed your request, but received an empty response. Please try rephrasing your question."

    except Exception as api_err:
        err_msg = str(api_err)
        logger.warning(f"Gemini API request failed: {err_msg}")

        if "quota" in err_msg.lower() or "429" in err_msg:
            return "⚠️ **API Quota Exceeded:** The Gemini API rate limit or quota has been reached. Please check your quota or try again in a few moments."
        if "key" in err_msg.lower() or "auth" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
            return "⚠️ **Authentication Error:** The configured Gemini API key appears invalid or lacks permissions. Please verify your API key."
        if "timeout" in err_msg.lower() or "connection" in err_msg.lower():
            return "⚠️ **Network Timeout:** Could not establish a connection to the Gemini API. Please check your internet connection."

        return f"⚠️ **Assistant Error:** The AI assistant encountered an issue while generating a response. ({type(api_err).__name__})"


def verify_leaf_with_gemini_vision(
    image: Image.Image,
    predicted_crop: Optional[str] = None,
    predicted_disease: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Provide multimodal AI second opinion on a leaf photograph using Gemini Vision.

    Evaluates visual symptoms, verifies crop and pathology independently of the
    benchmark CNN, and identifies false positives caused by laboratory background shifts.
    """
    resolved_key = api_key or get_gemini_api_key()
    if not resolved_key and client is None:
        return {
            "status": "API_KEY_REQUIRED",
            "verified": False,
            "analysis": (
                "⚠️ **Gemini Vision Verification Unavailable:** "
                "Set `GEMINI_API_KEY` in `.streamlit/secrets.toml` or environment variables to enable "
                "real-time multimodal AI leaf verification."
            ),
        }

    resolved_model = model_name or get_configured_model()

    if client is None:
        try:
            from google import genai

            client = genai.Client(api_key=resolved_key)
        except Exception as e:
            return {
                "status": "CLIENT_ERROR",
                "verified": False,
                "analysis": f"Failed to initialize Gemini Vision: {e}",
            }

    prompt = (
        "You are an expert plant pathologist and precision agronomist. "
        "Analyze this plant leaf photograph thoroughly.\n\n"
        f"Preliminary Edge Model Output:\n"
        f"- Candidate Crop: {predicted_crop or 'Unknown'}\n"
        f"- Candidate Condition: {predicted_disease or 'Unknown'}\n\n"
        "Provide your clinical evaluation:\n"
        "1. Crop Species: Identify the exact or most probable crop species.\n"
        "2. Leaf Condition: Identify the disease (or if the leaf is healthy).\n"
        "3. Morphological Symptoms: Describe what you see (e.g. concentric target rings, water-soaked margins, chlorosis, pustules, necrotic spots, or uniform green lamina).\n"
        "4. Model Consensus: State whether you AGREE, PARTIALLY AGREE, or DISAGREE with the preliminary Edge Model output.\n"
        "5. Immediate Action: Give 1-2 practical field steps for the grower.\n\n"
        "Format your answer with clean Markdown headers and bullet points."
    )

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=[image, prompt],
        )
        if response and response.text:
            return {
                "status": "SUCCESS",
                "verified": True,
                "analysis": response.text.strip(),
            }
        return {
            "status": "EMPTY_RESPONSE",
            "verified": False,
            "analysis": "Gemini Vision returned an empty response.",
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "verified": False,
            "analysis": f"Vision analysis encountered an error: {e}",
        }
