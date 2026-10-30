"""CropCare ΓÇö Vernacular & Voice Accessibility Module.

Provides multilingual localization (English, Hindi, Marathi) and audio narration
scripts for Indian agricultural extension workers and smallholder farmers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# Multilingual Crop & Disease Lexicon
VERNACULAR_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Crops
    "Apple": {"hi": "αñ╕αÑçαñ¼ (Apple)", "mr": "αñ╕αñ½αñ░αñÜαñéαñª (Apple)"},
    "Blueberry": {"hi": "αñ¼αÑìαñ▓αÑéαñ¼αÑçαñ░αÑÇ (Blueberry)", "mr": "αñ¼αÑìαñ▓αÑéαñ¼αÑçαñ░αÑÇ (Blueberry)"},
    "Cherry": {"hi": "αñÜαÑçαñ░αÑÇ (Cherry)", "mr": "αñÜαÑçαñ░αÑÇ (Cherry)"},
    "Corn": {"hi": "αñ«αñòαÑìαñòαñ╛ (Corn / Maize)", "mr": "αñ«αñòαñ╛ (Corn)"},
    "Grape": {"hi": "αñàαñéαñùαÑéαñ░ (Grape)", "mr": "αñªαÑìαñ░αñ╛αñòαÑìαñ╖αÑç (Grape)"},
    "Orange": {"hi": "αñ╕αñéαññαñ░αñ╛ (Orange / Citrus)", "mr": "αñ╕αñéαññαÑìαñ░αÑÇ (Orange)"},
    "Peach": {"hi": "αñåαñíαñ╝αÑé (Peach)", "mr": "αñ¬αÑÇαñÜ (Peach)"},
    "Pepper_bell": {"hi": "αñ╢αñ┐αñ«αñ▓αñ╛ αñ«αñ┐αñ░αÑìαñÜ (Bell Pepper)", "mr": "αñóαÑïαñ¼αñ│αÑÇ αñ«αñ┐αñ░αñÜαÑÇ (Capsicum)"},
    "Potato": {"hi": "αñåαñ▓αÑé (Potato)", "mr": "αñ¼αñƒαñ╛αñƒαñ╛ (Potato)"},
    "Raspberry": {"hi": "αñ░αñ╛αñ╕αÑìαñ¬αñ¼αÑçαñ░αÑÇ (Raspberry)", "mr": "αñ░αñ╛αñ╕αÑìαñ¬αñ¼αÑçαñ░αÑÇ (Raspberry)"},
    "Soybean": {"hi": "αñ╕αÑïαñ»αñ╛αñ¼αÑÇαñ¿ (Soybean)", "mr": "αñ╕αÑïαñ»αñ╛αñ¼αÑÇαñ¿ (Soybean)"},
    "Squash": {"hi": "αñòαñªαÑìαñªαÑé / αññαÑïαñ░αÑÇ (Squash)", "mr": "αñ¡αÑïαñ¬αñ│αñ╛ (Squash)"},
    "Strawberry": {"hi": "αñ╕αÑìαñƒαÑìαñ░αÑëαñ¼αÑçαñ░αÑÇ (Strawberry)", "mr": "αñ╕αÑìαñƒαÑìαñ░αÑëαñ¼αÑçαñ░αÑÇ (Strawberry)"},
    "Tomato": {"hi": "αñƒαñ«αñ╛αñƒαñ░ (Tomato)", "mr": "αñƒαÑïαñ«αÑàαñƒαÑï (Tomato)"},

    # Conditions & Diseases
    "scab": {"hi": "αñ╕αÑìαñòαÑêαñ¼ αñ¬αñ¬αñíαñ╝αÑÇ αñ░αÑïαñù (Scab)", "mr": "αñ╕αÑìαñòαÑàαñ¼ αñ░αÑïαñù (Scab)"},
    "black_rot": {"hi": "αñòαñ╛αñ▓αñ╛ αñ╕αñíαñ╝αñ¿ αñ░αÑïαñù (Black Rot)", "mr": "αñòαñ╛αñ│αñ╛ αñòαÑüαñ£αñ╡αñ╛ (Black Rot)"},
    "rust": {"hi": "αñùαÑçαñ░αÑüαñå / αñ░αññαÑüαñå αñ½αñéαñùαñ╕ (Rust)", "mr": "αññαñ╛αñéαñ¼αÑçαñ░αñ╛ αñ░αÑïαñù (Rust)"},
    "early_blight": {"hi": "αñàαñùαÑçαññαÑÇ αñ¥αÑüαñ▓αñ╕αñ╛ (Early Blight)", "mr": "αñàαñùαÑçαññαÑÇ αñòαñ░αñ¬αñ╛ (Early Blight)"},
    "late_blight": {"hi": "αñ¬αñ¢αÑçαññαÑÇ αñ¥αÑüαñ▓αñ╕αñ╛ (Late Blight)", "mr": "αñ¬αñ¢αÑçαññαÑÇ αñòαñ░αñ¬αñ╛ (Late Blight)"},
    "leaf_spot": {"hi": "αñ¬αññαÑìαññαñ┐αñ»αÑïαñé αñòαÑç αñºαñ¼αÑìαñ¼αÑç (Leaf Spot)", "mr": "αñ¬αñ╛αñ¿αñ╛αñ╡αñ░αÑÇαñ▓ αñáαñ┐αñ¬αñòαÑç (Leaf Spot)"},
    "powdery_mildew": {"hi": "αñ╕αñ½αÑçαñª αñÜαÑéαñ░αÑìαñúαÑÇ αñ½αñ½αÑéαñéαñª (Powdery Mildew)", "mr": "αñ¡αÑüαñ░αÑÇ αñ░αÑïαñù (Powdery Mildew)"},
    "bacterial_spot": {"hi": "αñ£αÑÇαñ╡αñ╛αñúαÑü αñºαñ¼αÑìαñ¼αñ╛ αñ░αÑïαñù (Bacterial Spot)", "mr": "αñ£αñ┐αñ╡αñ╛αñúαÑé αñáαñ┐αñ¬αñòαÑç (Bacterial Spot)"},
    "yellow_leaf_curl_virus": {"hi": "αñ¬αññαÑìαññαÑÇ αñ«αÑüαñíαñ╝αñ¿ αñ╡αñ┐αñ╖αñ╛αñúαÑü (Leaf Curl Virus)", "mr": "αñ¬αñ░αÑìαñúαñùαÑüαñÜαÑìαñ¢ αñ╡αñ┐αñ╖αñ╛αñúαÑé (Leaf Curl)"},
    "mosaic_virus": {"hi": "αñ«αÑïαñ£αñ╝αÑçαñò αñ╡αñ╛αñ»αñ░αñ╕ (Mosaic Virus)", "mr": "αñ«αÑïαñ¥αÑàαñò αñ╡αñ┐αñ╖αñ╛αñúαÑé (Mosaic Virus)"},
    "spider_mites": {"hi": "αñ▓αñ╛αñ▓ αñ«αñòαñíαñ╝αÑÇ / αñ«αñ╛αñçαñƒαÑìαñ╕ αñòαÑÇαñƒ (Spider Mites)", "mr": "αñ▓αñ╛αñ▓ αñòαÑïαñ│αÑÇ αñòαÑÇαñí (Spider Mites)"},
    "healthy": {"hi": "αñ╕αÑìαñ╡αñ╕αÑìαñÑ αñöαñ░ αñ¿αñ┐αñ░αÑïαñùαÑÇ αñ½αñ╕αñ▓ (Healthy)", "mr": "αñ¿αñ┐αñ░αÑïαñùαÑÇ αñåαñúαñ┐ αñëαññαÑìαññαñ« αñ¬αÑÇαñò (Healthy)"},
}

UI_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "brand_tag": "Precision Agronomic Intelligence",
        "step1": "Welcome",
        "step2": "Leaf Screening",
        "step3": "Climate Insights",
        "step4": "Executive Summary",
        "hero_badge": "ΓÜí Precision Agricultural Technology",
        "hero_title": "Read the leaf.<br><span>Read the weather.</span>",
        "hero_sub": "A dual-evidence agronomic screening platform combining MobileNetV3 deep learning leaf screening with Open-Meteo microclimate anomaly intelligence.",
        "btn_begin": "Begin Health Screening ΓåÆ",
        "feat1_title": "Leaf Vision Screening",
        "feat1_desc": "Upload leaf photographs. Detect visual symptoms across 14 crop species with calibrated uncertainty handling.",
        "feat2_title": "Historical Weather Baselines",
        "feat2_desc": "Automated Open-Meteo multi-year comparisons highlighting temperature and precipitation deviations.",
        "feat3_title": "Agronomic Safeguards",
        "feat3_desc": "Strict boundary between statistical model certainty and confirmed biology. Always advocates field verification.",
        "screen2_title": "≡ƒìâ Step 2: Optical Leaf Health Screening",
        "screen2_desc": "Upload an uncompressed leaf photograph in bright, uniform lighting. The MobileNetV3-Small neural network evaluates visual symptom patterns and generates a Grad-CAM explainability heatmap.",
        "upload_label": "Select leaf image (JPG, JPEG, PNG)",
        "btn_run_screening": "Run Leaf Screening & Explainable AI ΓÜí",
        "target_crop": "Target Crop",
        "screened_condition": "Screened Condition",
        "confidence": "Confidence",
        "btn_listen": "≡ƒöè Listen to Diagnosis (Audio)",
        "btn_back_welcome": "ΓåÉ Back to Welcome",
        "btn_goto_climate": "Continue to Climate Insights ΓåÆ",
        "screen3_title": "≡ƒîª∩╕Å Step 3: Hyper-Local Environmental Analytics",
        "screen3_desc": "Retrieve historical climate observations and multi-year anomaly baselines via Open-Meteo. Environmental stress often predisposes crops to pathogen vulnerability.",
        "btn_fetch_climate": "Fetch Microclimate Intelligence ≡ƒôí",
        "btn_back_leaf": "ΓåÉ Back to Leaf Screening",
        "btn_goto_summary": "Proceed to Executive Summary ΓåÆ",
        "screen4_title": "≡ƒôï Step 4: Executive Agronomic Summary",
        "screen4_desc": "Synthesized briefing consolidating optical leaf screening and hyper-local environmental data.",
        "alignment_title": "≡ƒö¼ Biometeorological DiseaseΓÇôClimate Alignment",
        "ipm_title": "≡ƒÜ£ Weather-Smart IPM & Application Window",
        "btn_download_card": "≡ƒôÑ Download Official Field Health Card (Print / PDF Ready)",
        "btn_restart": "≡ƒöä Start New Screening Session",
    },
    "hi": {
        "brand_tag": "αñ╕αñƒαÑÇαñò αñòαÑâαñ╖αñ┐ αñ░αÑïαñù αñÅαñ╡αñé αñ«αÑîαñ╕αñ« αñ╡αñ┐αñ╢αÑìαñ▓αÑçαñ╖αñú αñ«αñéαñÜ",
        "step1": "αñ¬αñ░αñ┐αñÜαñ»",
        "step2": "αñ¬αññαÑìαññαñ╛ αñ£αñ╛αñéαñÜ",
        "step3": "αñ«αÑîαñ╕αñ« αñ╡αñ┐αñ╢αÑìαñ▓αÑçαñ╖αñú",
        "step4": "αñòαñ╛αñ░αÑìαñ»αñòαñ╛αñ░αÑÇ αñ╕αñ╛αñ░αñ╛αñéαñ╢",
        "hero_badge": "ΓÜí αñåαñºαÑüαñ¿αñ┐αñò αñòαÑâαñ╖αñ┐ αñ¿αñ┐αñªαñ╛αñ¿ αññαñòαñ¿αÑÇαñò",
        "hero_title": "αñ¬αññαÑìαññαÑç αñòαÑï αñ╕αñ«αñ¥αÑçαñéαÑñ<br><span>αñ«αÑîαñ╕αñ« αñòαÑï αñ¬αñ░αñûαÑçαñéαÑñ</span>",
        "hero_sub": "αñ¬αññαÑìαññαÑç αñòαÑÇ αññαñ╕αÑìαñ╡αÑÇαñ░ αñ╕αÑç αñÅαñåαñê αñ░αÑïαñù αñ¬αñ╣αñÜαñ╛αñ¿ αñöαñ░ αñôαñ¬αñ¿-αñ«αÑçαñƒαñ┐αñ»αÑï αñ«αÑîαñ╕αñ« αñ╡αñ┐αñ╕αñéαñùαññαñ┐αñ»αÑïαñé αñòαÑï αñ«αñ┐αñ▓αñ╛αñòαñ░ αñ╕αñƒαÑÇαñò αñ½αñ╕αñ▓ αñ╕αÑìαñ╡αñ╛αñ╕αÑìαñÑαÑìαñ» αñ¬αñ░αñ╛αñ«αñ░αÑìαñ╢ αñªαÑçαñ¿αÑç αñ╡αñ╛αñ▓αñ╛ αñåαñºαÑüαñ¿αñ┐αñò αñòαÑâαñ╖αñ┐ αñ«αñéαñÜαÑñ",
        "btn_begin": "αñ½αñ╕αñ▓ αñ£αñ╛αñéαñÜ αñ╢αÑüαñ░αÑé αñòαñ░αÑçαñé ΓåÆ",
        "feat1_title": "αñ¬αññαÑìαññαñ╛ αñ╡αñ┐αñ£αñ╝αñ¿ αñ£αñ╛αñéαñÜ",
        "feat1_desc": "αñ¬αññαÑìαññαÑç αñòαñ╛ αñ½αÑïαñƒαÑï αñàαñ¬αñ▓αÑïαñí αñòαñ░αÑçαñéαÑñ αñ«αÑïαñ¼αñ╛αñçαñ▓αñ¿αÑçαñƒ-αñ╡αÑÇ3 αñ«αÑëαñíαñ▓ αñªαÑìαñ╡αñ╛αñ░αñ╛ 14 αñ½αñ╕αñ▓αÑïαñé αñ«αÑçαñé 38 αñ¬αÑìαñ░αñòαñ╛αñ░ αñòαÑç αñ░αÑïαñùαÑïαñé αñòαÑÇ αñ╕αñƒαÑÇαñò αñ¬αñ╣αñÜαñ╛αñ¿ αñòαñ░αÑçαñéαÑñ",
        "feat2_title": "αñ«αÑîαñ╕αñ« αñöαñ░ αñ£αñ▓αñ╡αñ╛αñ»αÑü αñçαññαñ┐αñ╣αñ╛αñ╕",
        "feat2_desc": "αñôαñ¬αñ¿-αñ«αÑçαñƒαñ┐αñ»αÑï αñëαñ¬αñùαÑìαñ░αñ╣ αñíαÑçαñƒαñ╛ αñ╕αÑç αñ¬αñ┐αñ¢αñ▓αÑç 3 αñ╡αñ░αÑìαñ╖αÑïαñé αñòαÑç αññαñ╛αñ¬αñ«αñ╛αñ¿ αñöαñ░ αñ╡αñ░αÑìαñ╖αñ╛ αñòαÑç αñ╕αñ╛αñÑ αñ╡αñ░αÑìαññαñ«αñ╛αñ¿ αñ«αÑîαñ╕αñ« αñòαÑÇ αññαÑüαñ▓αñ¿αñ╛αÑñ",
        "feat3_title": "αñ╡αÑêαñ£αÑìαñ₧αñ╛αñ¿αñ┐αñò αñòαÑâαñ╖αñ┐ αñ╕αÑüαñ░αñòαÑìαñ╖αñ╛",
        "feat3_desc": "αñÅαñåαñê αñàαñ¿αÑüαñ«αñ╛αñ¿ αñöαñ░ αñ╡αñ╛αñ╕αÑìαññαñ╡αñ┐αñò αñ£αÑêαñ╡αñ┐αñò αñ╕αÑìαñÑαñ┐αññαñ┐ αñ«αÑçαñé αñ╕αÑìαñ¬αñ╖αÑìαñƒ αñàαñéαññαñ░αÑñ αñ¼αñ┐αñ¿αñ╛ αñ¬αÑüαñ╖αÑìαñƒαñ┐ αñòαÑç αñàαñ¿αñ╛αñ╡αñ╢αÑìαñ»αñò αñ░αñ╕αñ╛αñ»αñ¿αÑïαñé αñ╕αÑç αñ¼αñÜαñ╛αñ╡αÑñ",
        "screen2_title": "≡ƒìâ αñÜαñ░αñú αÑ¿: αñ¬αññαÑìαññαÑç αñòαÑç αñ╕αÑìαñ╡αñ╛αñ╕αÑìαñÑαÑìαñ» αñòαÑÇ αñÅαñåαñê αñ£αñ╛αñéαñÜ",
        "screen2_desc": "αñ¬αññαÑìαññαÑç αñòαñ╛ αñ╕αÑìαñ¬αñ╖αÑìαñƒ αñ½αÑïαñƒαÑï αñàαñ¬αñ▓αÑïαñí αñòαñ░αÑçαñéαÑñ αñ¿αÑìαñ»αÑéαñ░αñ▓ αñ¿αÑçαñƒαñ╡αñ░αÑìαñò αñ░αÑïαñù αñòαÑç αñ▓αñòαÑìαñ╖αñú αñ¬αñ╣αñÜαñ╛αñ¿αÑçαñùαñ╛ αñöαñ░ αñùαÑìαñ░αÑêαñí-αñòαÑêαñ« (Grad-CAM) αñ╣αÑÇαñƒαñ«αÑêαñ¬ αñ¼αñ¿αñ╛αñòαñ░ αñ¼αññαñ╛αñÅαñùαñ╛ αñòαñ┐ αñ«αÑëαñíαñ▓ αñ¿αÑç αñòαñ┐αñ╕ αñºαñ¼αÑìαñ¼αÑç αñ¬αñ░ αñºαÑìαñ»αñ╛αñ¿ αñªαñ┐αñ»αñ╛αÑñ",
        "upload_label": "αñ¬αññαÑìαññαÑç αñòαñ╛ αñ½αÑïαñƒαÑï αñÜαÑüαñ¿αÑçαñé (JPG, JPEG, PNG)",
        "btn_run_screening": "αñ£αñ╛αñéαñÜ αñöαñ░ AI αñ╣αÑÇαñƒαñ«αÑêαñ¬ αñ╢αÑüαñ░αÑé αñòαñ░αÑçαñé ΓÜí",
        "target_crop": "αñ¬αñ╣αñÜαñ╛αñ¿αÑÇ αñùαñê αñ½αñ╕αñ▓",
        "screened_condition": "αñ░αÑïαñù αñ»αñ╛ αñ╕αÑìαñÑαñ┐αññαñ┐",
        "confidence": "αñ╕αñƒαÑÇαñòαññαñ╛ / αñ¿αñ┐αñ╢αÑìαñÜαñ┐αññαññαñ╛",
        "btn_listen": "≡ƒöè αñ░αÑïαñùαñ¿αñ┐αñªαñ╛αñ¿ αñæαñíαñ┐αñ»αÑï αñ╕αÑüαñ¿αÑçαñé (Audio)",
        "btn_back_welcome": "ΓåÉ αñ¬αñ░αñ┐αñÜαñ» αñ¬αñ░ αñ╡αñ╛αñ¬αñ╕ αñ£αñ╛αñÅαñé",
        "btn_goto_climate": "αñ«αÑîαñ╕αñ« αñ╡αñ┐αñ╢αÑìαñ▓αÑçαñ╖αñú αñ¬αñ░ αñåαñùαÑç αñ¼αñóαñ╝αÑçαñé ΓåÆ",
        "screen3_title": "≡ƒîª∩╕Å αñÜαñ░αñú αÑ⌐: αñ╕αÑìαñÑαñ╛αñ¿αÑÇαñ» αñ«αÑîαñ╕αñ« αñ╡ αñ£αñ▓αñ╡αñ╛αñ»αÑü αñ╡αñ┐αñ╢αÑìαñ▓αÑçαñ╖αñú",
        "screen3_desc": "αñôαñ¬αñ¿-αñ«αÑçαñƒαñ┐αñ»αÑï αñ╕αÑç αññαñ╛αñ¬αñ«αñ╛αñ¿ αñöαñ░ αñ╡αñ░αÑìαñ╖αñ╛ αñòαÑç αñÉαññαñ┐αñ╣αñ╛αñ╕αñ┐αñò αñ╡αñ┐αñÜαñ▓αñ¿αÑïαñé αñòαÑÇ αñ£αñ╛αñéαñÜ αñòαñ░αÑçαñéαÑñ αñ«αÑîαñ╕αñ« αñòαñ╛ αññαñ¿αñ╛αñ╡ (αñùαñ░αÑìαñ«αÑÇ αñ»αñ╛ αñ╕αÑéαñûαÑç αñòαÑÇ αñ╕αÑìαñÑαñ┐αññαñ┐) αñàαñòαÑìαñ╕αñ░ αñ¬αÑîαñºαÑïαñé αñ«αÑçαñé αñ½αñéαñùαñ╕ αñöαñ░ αñ░αÑïαñù αñ¼αñóαñ╝αñ╛αññαñ╛ αñ╣αÑêαÑñ",
        "btn_fetch_climate": "αñ«αÑîαñ╕αñ« αñíαÑçαñƒαñ╛ αñ¬αÑìαñ░αñ╛αñ¬αÑìαññ αñòαñ░αÑçαñé ≡ƒôí",
        "btn_back_leaf": "ΓåÉ αñ¬αññαÑìαññαñ╛ αñ£αñ╛αñéαñÜ αñ¬αñ░ αñ╡αñ╛αñ¬αñ╕ αñ£αñ╛αñÅαñé",
        "btn_goto_summary": "αñòαñ╛αñ░αÑìαñ»αñòαñ╛αñ░αÑÇ αñ░αñ┐αñ¬αÑïαñ░αÑìαñƒ αñ¬αñ░ αñåαñùαÑç αñ¼αñóαñ╝αÑçαñé ΓåÆ",
        "screen4_title": "≡ƒôï αñÜαñ░αñú αÑ¬: αñ╕αñéαñ¬αÑéαñ░αÑìαñú αñòαñ╛αñ░αÑìαñ»αñòαñ╛αñ░αÑÇ αñòαÑâαñ╖αñ┐ αñ╕αñ╛αñ░αñ╛αñéαñ╢",
        "screen4_desc": "αñ¬αññαÑìαññαÑç αñòαÑç αñªαÑâαñ╢αÑìαñ» αñ▓αñòαÑìαñ╖αñúαÑïαñé αñöαñ░ αñ╕αÑìαñÑαñ╛αñ¿αÑÇαñ» αñ«αÑîαñ╕αñ« αñíαÑçαñƒαñ╛ αñòαñ╛ αñ╕αñéαñ»αÑüαñòαÑìαññ αñ╡αñ┐αñ╢αÑìαñ▓αÑçαñ╖αñú αñÅαñ╡αñé αñòαÑâαñ╖αñ┐ αñ╡αÑêαñ£αÑìαñ₧αñ╛αñ¿αñ┐αñòαÑïαñé αñªαÑìαñ╡αñ╛αñ░αñ╛ αñàαñ¿αÑüαñ«αÑïαñªαñ┐αññ αñ¬αñ░αñ╛αñ«αñ░αÑìαñ╢αÑñ",
        "alignment_title": "≡ƒö¼ αñ¼αñ╛αñ»αÑïαñ«αÑçαñƒαñ┐αñ»αÑïαñ░αÑïαñ▓αÑëαñ£αñ┐αñòαñ▓ αñ░αÑïαñù-αñ£αñ▓αñ╡αñ╛αñ»αÑü αññαñ╛αñ▓αñ«αÑçαñ▓ (Alignment Index)",
        "ipm_title": "≡ƒÜ£ αñ«αÑîαñ╕αñ«-αñàαñ¿αÑüαñòαÑéαñ▓ αñ½αñ╕αñ▓ αñ╕αÑüαñ░αñòαÑìαñ╖αñ╛ αñ╡ αñ╕αÑìαñ¬αÑìαñ░αÑç αñ╕αñ«αñ» (IPM Advisory)",
        "btn_download_card": "≡ƒôÑ αñòαñ┐αñ╕αñ╛αñ¿ αñ½αÑÇαñ▓αÑìαñí αñ╣αÑçαñ▓αÑìαñÑ αñòαñ╛αñ░αÑìαñí αñíαñ╛αñëαñ¿αñ▓αÑïαñí αñòαñ░αÑçαñé (Print / PDF)",
        "btn_restart": "≡ƒöä αñ¿αñê αñ£αñ╛αñéαñÜ αñ╢αÑüαñ░αÑé αñòαñ░αÑçαñé",
    },
    "mr": {
        "brand_tag": "αñàαñÜαÑéαñò αñòαÑâαñ╖αÑÇ αñ░αÑïαñù αñåαñúαñ┐ αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñ╡αñ┐αñ╢αÑìαñ▓αÑçαñ╖αñú αñ«αñéαñÜ",
        "step1": "αñôαñ│αñû",
        "step2": "αñ¬αñ╛αñ¿ αññαñ¬αñ╛αñ╕αñúαÑÇ",
        "step3": "αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñ«αñ╛αñ╣αñ┐αññαÑÇ",
        "step4": "αñ╕αñéαñ¬αÑéαñ░αÑìαñú αñòαÑâαñ╖αÑÇ αñ╕αñ▓αÑìαñ▓αñ╛",
        "hero_badge": "ΓÜí αñåαñºαÑüαñ¿αñ┐αñò αñòαÑâαñ╖αÑÇ αñ¿αñ┐αñªαñ╛αñ¿ αññαñéαññαÑìαñ░αñ£αÑìαñ₧αñ╛αñ¿",
        "hero_title": "αñ¬αñ╛αñ¿αñ╛αñ▓αñ╛ αñôαñ│αñûαñ╛.<br><span>αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñ╕αñ«αñ£αÑéαñ¿ αñÿαÑìαñ»αñ╛.</span>",
        "hero_sub": "αñ¬αñ╛αñ¿αñ╛αñÜαÑç αñÅαñåαñ» αñ½αÑïαñƒαÑï αñ╕αÑìαñòαÑàαñ¿αñ┐αñéαñù αñåαñúαñ┐ αñôαñ¬αñ¿-αñ«αÑçαñƒαñ┐αñô αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñíαÑçαñƒαñ╛ αñÅαñòαññαÑìαñ░ αñòαñ░αÑéαñ¿ αñ¬αñ┐αñòαñ╛αñÜαÑìαñ»αñ╛ αñåαñ░αÑïαñùαÑìαñ»αñ╛αñÜαÑÇ αñàαñÜαÑéαñò αñ«αñ╛αñ╣αñ┐αññαÑÇ αñªαÑçαñúαñ╛αñ░αÑç αñ╡αÑìαñ»αñ╛αñ╕αñ¬αÑÇαñá.",
        "btn_begin": "αñ¬αñ┐αñòαñ╛αñÜαÑÇ αññαñ¬αñ╛αñ╕αñúαÑÇ αñ╕αÑüαñ░αÑé αñòαñ░αñ╛ ΓåÆ",
        "feat1_title": "αñ¬αñ╛αñ¿ αñªαÑâαñ╖αÑìαñƒαÑÇ αññαñ¬αñ╛αñ╕αñúαÑÇ",
        "feat1_desc": "αñ¬αñ╛αñ¿αñ╛αñÜαñ╛ αñ½αÑïαñƒαÑï αñàαñ¬αñ▓αÑïαñí αñòαñ░αñ╛. 14 αñ¬αñ┐αñòαñ╛αñéαñ«αñºαÑÇαñ▓ αñ░αÑïαñùαñ╛αñéαñÜαÑç αñàαñÜαÑéαñò αñ¿αñ┐αñªαñ╛αñ¿ αñåαñúαñ┐ αñàαñ¿αñ┐αñ╢αÑìαñÜαñ┐αññαññαñ╛ αñ╡αÑìαñ»αñ╡αñ╕αÑìαñÑαñ╛αñ¬αñ¿.",
        "feat2_title": "αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñåαñúαñ┐ αñ¬αñ░αÑìαñ£αñ¿αÑìαñ» αñçαññαñ┐αñ╣αñ╛αñ╕",
        "feat2_desc": "αñôαñ¬αñ¿-αñ«αÑçαñƒαñ┐αñôαñ╡αñ░αÑéαñ¿ αñ«αñ╛αñùαÑÇαñ▓ 3 αñ╡αñ░αÑìαñ╖αñ╛αñéαññαÑÇαñ▓ αññαñ╛αñ¬αñ«αñ╛αñ¿ αñåαñúαñ┐ αñ¬αñ╛αñ╡αñ╕αñ╛αñÜαÑìαñ»αñ╛ αñÉαññαñ┐αñ╣αñ╛αñ╕αñ┐αñò αñ¿αÑïαñéαñªαÑÇαñéαñÜαÑÇ αññαñ¬αñ╛αñ╕αñúαÑÇ.",
        "feat3_title": "αñ╡αÑêαñ£αÑìαñ₧αñ╛αñ¿αñ┐αñò αñòαÑâαñ╖αÑÇ αñ╕αÑüαñ░αñòαÑìαñ╖αñ╛",
        "feat3_desc": "αñÅαñåαñ» αñàαñéαñªαñ╛αñ£ αñåαñúαñ┐ αñ¬αÑìαñ░αññαÑìαñ»αñòαÑìαñ╖ αñ╢αÑçαññαñ╛αññαÑÇαñ▓ αñ╡αñ╕αÑìαññαÑüαñ╕αÑìαñÑαñ┐αññαÑÇαñ«αñºαÑÇαñ▓ αñàαñÜαÑéαñò αñ½αñ░αñò. αñÑαÑçαñƒ αñ░αñ╛αñ╕αñ╛αñ»αñ¿αñ┐αñò αñ½αñ╡αñ╛αñ░αñúαÑÇ αñƒαñ╛αñ│αñ╛.",
        "screen2_title": "≡ƒìâ αñ¬αñ╛αñ»αñ░αÑÇ αÑ¿: αñ¬αñ╛αñ¿αñ╛αñéαñÜαÑìαñ»αñ╛ αñåαñ░αÑïαñùαÑìαñ»αñ╛αñÜαÑÇ αñÅαñåαñ» αññαñ¬αñ╛αñ╕αñúαÑÇ",
        "screen2_desc": "αñ¬αñ╛αñ¿αñ╛αñÜαñ╛ αñ╕αÑìαñ¬αñ╖αÑìαñƒ αñ½αÑïαñƒαÑï αñàαñ¬αñ▓αÑïαñí αñòαñ░αñ╛. αñ¿αÑìαñ»αÑéαñ░αñ▓ αñ¿αÑçαñƒαñ╡αñ░αÑìαñò αñ░αÑïαñùαñ╛αñÜαÑÇ αñ▓αñòαÑìαñ╖αñúαÑç αñôαñ│αñûαÑéαñ¿ αñ╣αÑÇαñƒαñ«αÑàαñ¬ αññαñ»αñ╛αñ░ αñòαñ░αÑçαñ▓.",
        "upload_label": "αñ¬αñ╛αñ¿αñ╛αñÜαñ╛ αñ½αÑïαñƒαÑï αñ¿αñ┐αñ╡αñíαñ╛ (JPG, JPEG, PNG)",
        "btn_run_screening": "αññαñ¬αñ╛αñ╕αñúαÑÇ αñåαñúαñ┐ AI αñ╣αÑÇαñƒαñ«αÑàαñ¬ αñ╕αÑüαñ░αÑé αñòαñ░αñ╛ ΓÜí",
        "target_crop": "αñôαñ│αñûαñ▓αÑçαñ▓αÑç αñ¬αÑÇαñò",
        "screened_condition": "αñ░αÑïαñùαñ╛αñÜαÑÇ αñ╕αÑìαñÑαñ┐αññαÑÇ",
        "confidence": "αñûαñ╛αññαÑìαñ░αÑÇ / αñàαñÜαÑéαñòαññαñ╛",
        "btn_listen": "≡ƒöè αñ░αÑïαñùαñ¿αñ┐αñªαñ╛αñ¿ αñÉαñòαñ╛ (Audio)",
        "btn_back_welcome": "ΓåÉ αñ«αñ╛αñùαÑç αñ£αñ╛",
        "btn_goto_climate": "αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñ«αñ╛αñ╣αñ┐αññαÑÇαñòαñíαÑç αñ¬αÑüαñóαÑç αñ£αñ╛ ΓåÆ",
        "screen3_title": "≡ƒîª∩╕Å αñ¬αñ╛αñ»αñ░αÑÇ αÑ⌐: αñ╕αÑìαñÑαñ╛αñ¿αñ┐αñò αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñåαñúαñ┐ αñ¬αñ░αÑìαñ£αñ¿αÑìαñ» αñ╡αñ┐αñ╢αÑìαñ▓αÑçαñ╖αñú",
        "screen3_desc": "αñôαñ¬αñ¿-αñ«αÑçαñƒαñ┐αñôαñ╡αñ░αÑéαñ¿ αññαñ╛αñ¬αñ«αñ╛αñ¿ αñåαñúαñ┐ αñ¬αñ╛αñ╡αñ╕αñ╛αñÜαÑìαñ»αñ╛ αñÉαññαñ┐αñ╣αñ╛αñ╕αñ┐αñò αñ¿αÑïαñéαñªαÑÇαñéαñÜαÑÇ αññαñ¬αñ╛αñ╕αñúαÑÇ αñòαñ░αñ╛.",
        "btn_fetch_climate": "αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñíαÑçαñƒαñ╛ αñ«αñ┐αñ│αñ╡αñ╛ ≡ƒôí",
        "btn_back_leaf": "ΓåÉ αñ¬αñ╛αñ¿ αññαñ¬αñ╛αñ╕αñúαÑÇαñòαñíαÑç αñ¬αñ░αññ αñ£αñ╛",
        "btn_goto_summary": "αñàαñéαññαñ┐αñ« αñàαñ╣αñ╡αñ╛αñ▓αñ╛αñòαñíαÑç αñ¬αÑüαñóαÑç αñ£αñ╛ ΓåÆ",
        "screen4_title": "≡ƒôï αñ¬αñ╛αñ»αñ░αÑÇ αÑ¬: αñ╕αñéαñ¬αÑéαñ░αÑìαñú αñòαÑâαñ╖αÑÇ αñ╕αñ▓αÑìαñ▓αñ╛ αñ╡ αñàαñ╣αñ╡αñ╛αñ▓",
        "screen4_desc": "αñ¬αñ╛αñ¿αñ╛αñÜαÑç αñ▓αñòαÑìαñ╖αñú αñåαñúαñ┐ αñ╕αÑìαñÑαñ╛αñ¿αñ┐αñò αñ╣αñ╡αñ╛αñ«αñ╛αñ¿ αñíαÑçαñƒαñ╛αñÜαñ╛ αñÅαñòαññαÑìαñ░αñ┐αññ αñàαñ¡αÑìαñ»αñ╛αñ╕ αñòαñ░αÑéαñ¿ αññαñ»αñ╛αñ░ αñòαÑçαñ▓αÑçαñ▓αñ╛ αñàαñ╣αñ╡αñ╛αñ▓.",
        "alignment_title": "≡ƒö¼ αñ░αÑïαñù αñåαñúαñ┐ αñ╣αñ╡αñ╛αñ«αñ╛αñ¿αñ╛αñÜαñ╛ αñ£αÑêαñ╡αñ┐αñò αññαñ╛αñ│αñ«αÑçαñ│ (Alignment Index)",
        "ipm_title": "≡ƒÜ£ αñ╣αñ╡αñ╛αñ«αñ╛αñ¿αñ╛αñ¿αÑüαñ╕αñ╛αñ░ αñ½αñ╡αñ╛αñ░αñúαÑÇ αñ╡ αñ¬αÑÇαñò αñ╕αñéαñ░αñòαÑìαñ╖αñú (IPM Advisory)",
        "btn_download_card": "≡ƒôÑ αñ╢αÑçαññαñòαñ░αÑÇ αñ╣αÑçαñ▓αÑìαñÑ αñòαñ╛αñ░αÑìαñí αñíαñ╛αñëαñ¿αñ▓αÑïαñí αñòαñ░αñ╛ (Print / PDF)",
        "btn_restart": "≡ƒöä αñ¿αñ╡αÑÇαñ¿ αññαñ¬αñ╛αñ╕αñúαÑÇ αñ╕αÑüαñ░αÑé αñòαñ░αñ╛",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Retrieve localized UI string for a given key and language code."""
    lang_dict = UI_STRINGS.get(lang, UI_STRINGS["en"])
    return lang_dict.get(key, UI_STRINGS["en"].get(key, key))


def translate_diagnosis(crop: str, disease: str, lang: str = "en") -> Tuple[str, str]:
    """Translate crop and disease labels into selected language."""
    if lang == "en":
        return crop, disease

    trans_crop = crop
    crop_match = VERNACULAR_TRANSLATIONS.get(crop)
    if crop_match and lang in crop_match:
        trans_crop = crop_match[lang]

    trans_disease = disease
    dis_norm = disease.lower().replace(" ", "_")
    for k, v in VERNACULAR_TRANSLATIONS.items():
        if k in dis_norm and lang in v:
            trans_disease = v[lang]
            break

    return trans_crop, trans_disease


def generate_audio_script(
    crop: str,
    disease: str,
    confidence: float,
    alignment_status: str,
    spray_badge: str,
    lang: str = "hi",
) -> str:
    """Generate spoken diagnosis script for text-to-speech narration."""
    conf_pct = int(confidence * 100)
    is_healthy = "healthy" in disease.lower()

    if lang == "hi":
        c_hi, d_hi = translate_diagnosis(crop, disease, "hi")
        if is_healthy:
            return (
                f"αñòαñ┐αñ╕αñ╛αñ¿ αñ«αñ┐αññαÑìαñ░, αñòαÑìαñ░αÑëαñ¬-αñòαÑçαñ»αñ░ αñ£αñ╛αñéαñÜ αñòαÑç αñàαñ¿αÑüαñ╕αñ╛αñ░ αñåαñ¬αñòαñ╛ {c_hi} αñ¼αñ┐αñ▓αÑìαñòαÑüαñ▓ αñ╕αÑìαñ╡αñ╕αÑìαñÑ αñöαñ░ αñ¿αñ┐αñ░αÑïαñùαÑÇ αñ╣αÑêαÑñ "
                f"αñ«αÑëαñíαñ▓ αñòαÑÇ αñ¿αñ┐αñ╢αÑìαñÜαñ┐αññαññαñ╛ {conf_pct} αñ¬αÑìαñ░αññαñ┐αñ╢αññ αñ╣αÑêαÑñ αñòαñ┐αñ╕αÑÇ αñ¡αÑÇ αñ░αñ╛αñ╕αñ╛αñ»αñ¿αñ┐αñò αñ¢αñ┐αñíαñ╝αñòαñ╛αñ╡ αñòαÑÇ αñåαñ╡αñ╢αÑìαñ»αñòαññαñ╛ αñ¿αñ╣αÑÇαñé αñ╣αÑêαÑñ"
            )
        else:
            return (
                f"αñòαñ┐αñ╕αñ╛αñ¿ αñ«αñ┐αññαÑìαñ░, αñòαÑìαñ░αÑëαñ¬-αñòαÑçαñ»αñ░ αñ£αñ╛αñéαñÜ αñòαÑç αñàαñ¿αÑüαñ╕αñ╛αñ░ αñåαñ¬αñòαÑç {c_hi} αñòαÑç αñ¬αÑîαñºαÑç αñ«αÑçαñé {d_hi} αñòαÑç αñ▓αñòαÑìαñ╖αñú αñ¬αñ╛αñÅ αñùαñÅ αñ╣αÑêαñéαÑñ "
                f"αñ«αÑëαñíαñ▓ αñòαÑÇ αñ¿αñ┐αñ╢αÑìαñÜαñ┐αññαññαñ╛ {conf_pct} αñ¬αÑìαñ░αññαñ┐αñ╢αññ αñ╣αÑêαÑñ {spray_badge}αÑñ αñòαÑâαñ¬αñ»αñ╛ αñ╕αÑìαñÑαñ╛αñ¿αÑÇαñ» αñòαÑâαñ╖αñ┐ αñàαñºαñ┐αñòαñ╛αñ░αÑÇ αñ╕αÑç αñ¬αÑüαñ╖αÑìαñƒαñ┐ αñòαñ░αñòαÑç αñ╣αÑÇ αñ¢αñ┐αñíαñ╝αñòαñ╛αñ╡ αñòαñ░αÑçαñéαÑñ"
            )
    elif lang == "mr":
        c_mr, d_mr = translate_diagnosis(crop, disease, "mr")
        if is_healthy:
            return (
                f"αñ╢αÑçαññαñòαñ░αÑÇ αñ¼αñéαñºαÑéαñéαñ¿αÑï, αñòαÑìαñ░αÑëαñ¬-αñòαÑçαñàαñ░ αññαñ¬αñ╛αñ╕αñúαÑÇαñ¿αÑüαñ╕αñ╛αñ░ αñåαñ¬αñ▓αÑç {c_mr} αñ¬αÑÇαñò αñ¬αÑéαñ░αÑìαñúαñ¬αñúαÑç αñ¿αñ┐αñ░αÑïαñùαÑÇ αñåαñ╣αÑç. "
                f"αñûαñ╛αññαÑìαñ░αÑÇ {conf_pct} αñƒαñòαÑìαñòαÑç αñåαñ╣αÑç. αñ░αñ╛αñ╕αñ╛αñ»αñ¿αñ┐αñò αñ½αñ╡αñ╛αñ░αñúαÑÇαñÜαÑÇ αñùαñ░αñ£ αñ¿αñ╛αñ╣αÑÇ."
            )
        else:
            return (
                f"αñ╢αÑçαññαñòαñ░αÑÇ αñ¼αñéαñºαÑéαñéαñ¿αÑï, αñåαñ¬αñ▓αÑìαñ»αñ╛ {c_mr} αñ¬αñ┐αñòαñ╛αñ╡αñ░ {d_mr} αñÜαÑÇ αñ▓αñòαÑìαñ╖αñúαÑç αñåαñóαñ│αñ▓αÑÇ αñåαñ╣αÑçαññ. "
                f"αññαñ¬αñ╛αñ╕αñúαÑÇαñÜαÑÇ αñûαñ╛αññαÑìαñ░αÑÇ {conf_pct} αñƒαñòαÑìαñòαÑç αñåαñ╣αÑç. {spray_badge}αÑñ αñòαÑâαñ¬αñ»αñ╛ αññαñ£αÑìαñ£αÑìαñ₧αñ╛αñéαñÜαÑìαñ»αñ╛ αñ╕αñ▓αÑìαñ▓αÑìαñ»αñ╛αñ¿αÑçαñÜ αñ»αÑïαñùαÑìαñ» αñëαñ¬αñ╛αñ»αñ»αÑïαñ£αñ¿αñ╛ αñòαñ░αñ╛."
            )
    else:
        if is_healthy:
            return (
                f"CropCare Screening: Your {crop} plant appears healthy with {conf_pct} percent model certainty. "
                "No chemical fungicide intervention is necessary."
            )
        else:
            return (
                f"CropCare Screening: Visual symptoms on your {crop} leaf indicate {disease} with {conf_pct} percent certainty. "
                f"Advisory status: {spray_badge}. Please physically verify before spraying."
            )
