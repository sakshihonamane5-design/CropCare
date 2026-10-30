"""CropCare — Vernacular & Voice Accessibility Module.

Provides multilingual localization (English, Hindi, Marathi) and audio narration
scripts for Indian agricultural extension workers and smallholder farmers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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
    "spider_mites": {"hi": "लाल मकड़ी कीट (Spider Mites)", "mr": "लाल कोळी कीड (Spider Mites)"},
    "healthy": {"hi": "स्वस्थ और निरोगी फसल (Healthy)", "mr": "निरोगी आणि उत्तम पीक (Healthy)"},
}

UI_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "brand_tag": "Precision Agronomic Intelligence",
        "step1": "Welcome",
        "step2": "Leaf Screening",
        "step3": "Climate Insights",
        "step4": "Executive Summary",
        "hero_badge": "⚡ Precision Agricultural Technology",
        "hero_title": "Read the leaf.<br><span>Read the weather.</span>",
        "hero_sub": "A dual-evidence agronomic screening platform combining MobileNetV3 deep learning leaf screening with Open-Meteo microclimate anomaly intelligence.",
        "btn_begin": "Begin Health Screening →",
        "feat1_title": "Leaf Vision Screening",
        "feat1_desc": "Upload leaf photographs. Detect visual symptoms across 14 crop species with calibrated uncertainty handling.",
        "feat2_title": "Historical Weather Baselines",
        "feat2_desc": "Automated Open-Meteo multi-year comparisons highlighting temperature and precipitation deviations.",
        "feat3_title": "Agronomic Safeguards",
        "feat3_desc": "Clear distinction between model certainty and confirmed diagnosis. Always advises field verification.",
        "alignment_title": "🔬 Biometeorological Disease—Climate Alignment",
    },
    "hi": {
        "brand_tag": "सटीक कृषि बुद्धिमत्ता",
        "step1": "स्वागत",
        "step2": "पत्ती जांच",
        "step3": "मौसम विश्लेषण",
        "step4": "सम्पूर्ण रिपोर्ट",
        "hero_badge": "⚡ सटीक कृषि प्रौद्योगिकी",
        "hero_title": "पत्ती को पहचानें।<br><span>मौसम को समझें।</span>",
        "hero_sub": "कंप्यूटर विज़न और ओपन-मेटियो मौसम विसंगतियों को जोड़ने वाला आधुनिक कृषि मंच।",
        "btn_begin": "फसल स्वास्थ्य जांच शुरू करें →",
        "feat1_title": "पत्ती रोग पहचान",
        "feat1_desc": "१४ फसलों में पत्ती के फोटो से रोगों की पहचान और निश्चितता स्कोर।",
        "feat2_title": "मौसम इतिहास विश्लेषण",
        "feat2_desc": "ओपन-मेटियो द्वारा कई वर्षों के तापमान और वर्षा के आंकड़ों से विसंगति की गणना।",
        "feat3_title": "कृषि सुरक्षा और सावधानियां",
        "feat3_desc": "मॉडल अनुमान और वास्तविक रोग में अंतर। रासायनिक छिड़काव से पहले सत्यापन आवश्यक।",
        "alignment_title": "🔬 रोग और मौसम का जैविक तालमेल",
    },
    "mr": {
        "brand_tag": "अचूक कृषी बुद्धिमत्ता",
        "step1": "स्वागत",
        "step2": "पाने तपासणी",
        "step3": "हवामान विश्लेषण",
        "step4": "अंतिम अहवाल",
        "hero_badge": "⚡ अचूक कृषी तंत्रज्ञान",
        "hero_title": "पाने ओळखा.<br><span>हवामान समजून घ्या.</span>",
        "hero_sub": "कॉम्प्युटर व्हिजन आणि हवामान विसंगतींचा अभ्यास करणारे प्रगत कृषी व्यासपीठ.",
        "btn_begin": "आरोग्य तपासणी सुरू करा →",
        "feat1_title": "पाने रोग तपासणी",
        "feat1_desc": "१४ पिकांच्या पानांचे फोटो तपासून रोगांचे अचूक निदान आणि खात्री स्तर.",
        "feat2_title": "ऐतिहासिक हवामान नोंदी",
        "feat2_desc": "ओपन-मेटिओ द्वारे अनेक वर्षांच्या हवामानाशी तुलना करून तापमानातील बदल समजून घ्या.",
        "feat3_title": "कृषी सुरक्षा आणि खबरदारी",
        "feat3_desc": "मॉडेलचा अंदाज आणि प्रत्यक्ष रोग यातील फरक. फवारणीपूर्वी कृषी तज्ज्ञांचा सल्ला घ्या.",
        "alignment_title": "🔬 रोग आणि हवामानाचा जैविक ताळमेळ",
    },
}


def translate_diagnosis(crop: str, disease: str, lang: str = "hi") -> Tuple[str, str]:
    """Translate crop and disease names to vernacular language (Hindi or Marathi)."""
    if lang not in ("hi", "mr"):
        return crop, disease

    trans = VERNACULAR_TRANSLATIONS.get(crop, {})
    crop_v = trans.get(lang, crop)

    dis_norm = disease.lower().replace(" ", "_")
    dis_trans = VERNACULAR_TRANSLATIONS.get(dis_norm, {})
    dis_v = dis_trans.get(lang, disease)

    return crop_v, dis_v


def get_ui_text(key: str, lang: str = "en") -> str:
    """Retrieve UI string in selected language with fallback to English."""
    lang_dict = UI_STRINGS.get(lang, UI_STRINGS["en"])
    return lang_dict.get(key, UI_STRINGS["en"].get(key, key))


def generate_audio_script(
    crop: str,
    disease: str,
    confidence_pct: float,
    alignment_status: Optional[str] = None,
    spray_window: Optional[str] = None,
    lang: str = "hi",
    **kwargs: Any,
) -> str:
    """Generate audio narrative text for Text-to-Speech (TTS) read-aloud."""
    crop_v, dis_v = translate_diagnosis(crop, disease, lang)
    conf_val = confidence_pct * 100 if 0.0 <= confidence_pct <= 1.0 else confidence_pct

    if lang == "hi":
        return (
            f"क्रॉपकेयर फसल स्वास्थ्य रिपोर्ट। आपकी {crop_v} की फसल में {dis_v} के लक्षण पाए गए हैं। "
            f"इसकी निश्चितता {conf_val:.0f} प्रतिशत है। "
            f"कृपया नजदीकी कृषि अधिकारी से संपर्क करें और उचित उपचार करें।"
        )
    elif lang == "mr":
        return (
            f"क्रॉपकेअर पीक आरोग्य अहवाल. आपल्या {crop_v} पिकावर {dis_v} चे लक्षणे आढळली आहेत. "
            f"याची खात्री {conf_val:.0f} टक्के आहे. "
            f"कृपया कृषी अधिकाऱ्यांचा सल्ला घेऊन योग्य उपाययोजना करा."
        )
    else:
        return (
            f"CropCare Plant Health Advisory. Foliar screening detected {disease} on your {crop} crop "
            f"with {conf_val:.0f}% confidence. "
            f"Consult your local agricultural extension officer for confirmed diagnosis."
        )
