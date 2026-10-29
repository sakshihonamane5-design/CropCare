"""CropCare — Vernacular & Voice Accessibility Module.

Provides multilingual localization (English, Hindi, Marathi) and audio narration
scripts for Indian agricultural extension workers and smallholder farmers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Multilingual Crop & Disease Lexicon
VERNACULAR_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Crops
    "Apple": {"hi": "सेब (Apple)", "mr": "सफरचंद (Apple)"},
    "Blueberry": {"hi": "ब्लूबेरी (Blueberry)", "mr": "ब्लूबेरी (Blueberry)"},
    "Cherry": {"hi": "चेरी (Cherry)", "mr": "चेरी (Cherry)"},
    "Corn": {"hi": "मक्का (Corn / Maize)", "mr": "मका (Corn)"},
    "Grape": {"hi": "अंगूर (Grape)", "mr": "द्राक्षे (Grape)"},
    "Orange": {"hi": "संतरा (Orange / Citrus)", "mr": "संत्री (Orange)"},
    "Peach": {"hi": "आड़ू (Peach)", "mr": "पीच (Peach)"},
    "Pepper_bell": {"hi": "शिमला मिर्च (Bell Pepper)", "mr": "ढोबळी मिरची (Capsicum)"},
    "Potato": {"hi": "आलू (Potato)", "mr": "बटाटा (Potato)"},
    "Raspberry": {"hi": "रास्पबेरी (Raspberry)", "mr": "रास्पबेरी (Raspberry)"},
    "Soybean": {"hi": "सोयाबीन (Soybean)", "mr": "सोयाबीन (Soybean)"},
    "Squash": {"hi": "कद्दू / तोरी (Squash)", "mr": "भोपळा (Squash)"},
    "Strawberry": {"hi": "स्ट्रॉबेरी (Strawberry)", "mr": "स्ट्रॉबेरी (Strawberry)"},
    "Tomato": {"hi": "टमाटर (Tomato)", "mr": "टोमॅटो (Tomato)"},

    # Conditions & Diseases
    "scab": {"hi": "स्कैब पपड़ी रोग (Scab)", "mr": "स्कॅब रोग (Scab)"},
    "black_rot": {"hi": "काला सड़न रोग (Black Rot)", "mr": "काळा कुजवा (Black Rot)"},
    "rust": {"hi": "गेरुआ / रतुआ फंगस (Rust)", "mr": "तांबेरा रोग (Rust)"},
    "early_blight": {"hi": "अगेती झुलसा (Early Blight)", "mr": "अगेती करपा (Early Blight)"},
    "late_blight": {"hi": "पछेती झुलसा (Late Blight)", "mr": "पछेती करपा (Late Blight)"},
    "leaf_spot": {"hi": "पत्तियों के धब्बे (Leaf Spot)", "mr": "पानावरील ठिपके (Leaf Spot)"},
    "powdery_mildew": {"hi": "सफेद चूर्णी फफूंद (Powdery Mildew)", "mr": "भुरी रोग (Powdery Mildew)"},
    "bacterial_spot": {"hi": "जीवाणु धब्बा रोग (Bacterial Spot)", "mr": "जिवाणू ठिपके (Bacterial Spot)"},
    "yellow_leaf_curl_virus": {"hi": "पत्ती मुड़न विषाणु (Leaf Curl Virus)", "mr": "पर्णगुच्छ विषाणू (Leaf Curl)"},
    "mosaic_virus": {"hi": "मोज़ेक वायरस (Mosaic Virus)", "mr": "मोझॅक विषाणू (Mosaic Virus)"},
    "spider_mites": {"hi": "लाल मकड़ी / माइट्स कीट (Spider Mites)", "mr": "लाल कोळी कीड (Spider Mites)"},
    "healthy": {"hi": "स्वस्थ और निरोगी फसल (Healthy)", "mr": "निरोगी आणि उत्तम पीक (Healthy)"},
}

UI_STRINGS = {
    "en": {
        "title": "Optical & Environmental Intelligence",
        "step1": "1. Mission Overview",
        "step2": "2. Leaf Screening",
        "step3": "3. Climate Context",
        "step4": "4. Executive Briefing",
        "upload_btn": "Upload Leaf Photograph",
        "run_screening": "Run Leaf Screening ⚡",
        "crop_label": "Target Crop",
        "condition_label": "Screened Condition",
        "confidence_label": "Model Certainty",
        "listen_btn": "🔊 Listen to Diagnosis (Audio)",
    },
    "hi": {
        "title": "पत्ता स्वास्थ्य एवं जलवायु विश्लेषण",
        "step1": "१. परिचय",
        "step2": "२. पत्ता जांच",
        "step3": "३. मौसम व जलवायु",
        "step4": "४. कृषि परामर्श रिपोर्ट",
        "upload_btn": "पत्ते की फोटो अपलोड करें",
        "run_screening": "जांच शुरू करें ⚡",
        "crop_label": "पहचानी गई फसल",
        "condition_label": "रोग या स्थिति",
        "confidence_label": "मॉडल सटीकता",
        "listen_btn": "🔊 निदान ऑडियो सुनें (Audio)",
    },
    "mr": {
        "title": "पीक आरोग्य आणि हवामान विश्लेषण",
        "step1": "१. ओळख",
        "step2": "२. पान तपासणी",
        "step3": "३. हवामान माहिती",
        "step4": "४. संपूर्ण कृषी सल्ला",
        "upload_btn": "पानाचा फोटो अपलोड करा",
        "run_screening": "तपासणी सुरू करा ⚡",
        "crop_label": "ओळखलेले पीक",
        "condition_label": "रोगाची स्थिती",
        "confidence_label": "खात्री",
        "listen_btn": "🔊 रोगनिदान ऐका (Audio)",
    },
}


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
                f"किसान मित्र, क्रॉप-केयर जांच के अनुसार आपका {c_hi} बिल्कुल स्वस्थ और निरोगी है। "
                f"मॉडल की निश्चितता {conf_pct} प्रतिशत है। किसी भी रासायनिक छिड़काव की आवश्यकता नहीं है।"
            )
        else:
            return (
                f"किसान मित्र, क्रॉप-केयर जांच के अनुसार आपके {c_hi} के पौधे में {d_hi} के लक्षण पाए गए हैं। "
                f"मॉडल की निश्चितता {conf_pct} प्रतिशत है। {spray_badge}। कृपया स्थानीय कृषि अधिकारी से पुष्टि करके ही छिड़काव करें।"
            )
    elif lang == "mr":
        c_mr, d_mr = translate_diagnosis(crop, disease, "mr")
        if is_healthy:
            return (
                f"शेतकरी बंधूंनो, क्रॉप-केअर तपासणीनुसार आपले {c_mr} पीक पूर्णपणे निरोगी आहे. "
                f"खात्री {conf_pct} टक्के आहे. रासायनिक फवारणीची गरज नाही."
            )
        else:
            return (
                f"शेतकरी बंधूंनो, आपल्या {c_mr} पिकावर {d_mr} ची लक्षणे आढळली आहेत. "
                f"तपासणीची खात्री {conf_pct} टक्के आहे. {spray_badge}। कृपया तज्ज्ञांच्या सल्ल्यानेच योग्य उपाययोजना करा."
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
