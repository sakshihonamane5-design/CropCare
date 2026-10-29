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
    "spider_mites": {"hi": "लाल मकड़ी / माइट्स कीट (Spider Mites)", "mr": "लाल कोळी कीड (Spider Mites)"},
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
        "feat3_desc": "Strict boundary between statistical model certainty and confirmed biology. Always advocates field verification.",
        "screen2_title": "🍃 Step 2: Optical Leaf Health Screening",
        "screen2_desc": "Upload an uncompressed leaf photograph in bright, uniform lighting. The MobileNetV3-Small neural network evaluates visual symptom patterns and generates a Grad-CAM explainability heatmap.",
        "upload_label": "Select leaf image (JPG, JPEG, PNG)",
        "btn_run_screening": "Run Leaf Screening & Explainable AI ⚡",
        "target_crop": "Target Crop",
        "screened_condition": "Screened Condition",
        "confidence": "Confidence",
        "btn_listen": "🔊 Listen to Diagnosis (Audio)",
        "btn_back_welcome": "← Back to Welcome",
        "btn_goto_climate": "Continue to Climate Insights →",
        "screen3_title": "🌦️ Step 3: Hyper-Local Environmental Analytics",
        "screen3_desc": "Retrieve historical climate observations and multi-year anomaly baselines via Open-Meteo. Environmental stress often predisposes crops to pathogen vulnerability.",
        "btn_fetch_climate": "Fetch Microclimate Intelligence 📡",
        "btn_back_leaf": "← Back to Leaf Screening",
        "btn_goto_summary": "Proceed to Executive Summary →",
        "screen4_title": "📋 Step 4: Executive Agronomic Summary",
        "screen4_desc": "Synthesized briefing consolidating optical leaf screening and hyper-local environmental data.",
        "alignment_title": "🔬 Biometeorological Disease–Climate Alignment",
        "ipm_title": "🚜 Weather-Smart IPM & Application Window",
        "btn_download_card": "📥 Download Official Field Health Card (Print / PDF Ready)",
        "btn_restart": "🔄 Start New Screening Session",
    },
    "hi": {
        "brand_tag": "सटीक कृषि रोग एवं मौसम विश्लेषण मंच",
        "step1": "परिचय",
        "step2": "पत्ता जांच",
        "step3": "मौसम विश्लेषण",
        "step4": "कार्यकारी सारांश",
        "hero_badge": "⚡ आधुनिक कृषि निदान तकनीक",
        "hero_title": "पत्ते को समझें।<br><span>मौसम को परखें।</span>",
        "hero_sub": "पत्ते की तस्वीर से एआई रोग पहचान और ओपन-मेटियो मौसम विसंगतियों को मिलाकर सटीक फसल स्वास्थ्य परामर्श देने वाला आधुनिक कृषि मंच।",
        "btn_begin": "फसल जांच शुरू करें →",
        "feat1_title": "पत्ता विज़न जांच",
        "feat1_desc": "पत्ते का फोटो अपलोड करें। मोबाइलनेट-वी3 मॉडल द्वारा 14 फसलों में 38 प्रकार के रोगों की सटीक पहचान करें।",
        "feat2_title": "मौसम और जलवायु इतिहास",
        "feat2_desc": "ओपन-मेटियो उपग्रह डेटा से पिछले 3 वर्षों के तापमान और वर्षा के साथ वर्तमान मौसम की तुलना।",
        "feat3_title": "वैज्ञानिक कृषि सुरक्षा",
        "feat3_desc": "एआई अनुमान और वास्तविक जैविक स्थिति में स्पष्ट अंतर। बिना पुष्टि के अनावश्यक रसायनों से बचाव।",
        "screen2_title": "🍃 चरण २: पत्ते के स्वास्थ्य की एआई जांच",
        "screen2_desc": "पत्ते का स्पष्ट फोटो अपलोड करें। न्यूरल नेटवर्क रोग के लक्षण पहचानेगा और ग्रैड-कैम (Grad-CAM) हीटमैप बनाकर बताएगा कि मॉडल ने किस धब्बे पर ध्यान दिया।",
        "upload_label": "पत्ते का फोटो चुनें (JPG, JPEG, PNG)",
        "btn_run_screening": "जांच और AI हीटमैप शुरू करें ⚡",
        "target_crop": "पहचानी गई फसल",
        "screened_condition": "रोग या स्थिति",
        "confidence": "सटीकता / निश्चितता",
        "btn_listen": "🔊 रोगनिदान ऑडियो सुनें (Audio)",
        "btn_back_welcome": "← परिचय पर वापस जाएं",
        "btn_goto_climate": "मौसम विश्लेषण पर आगे बढ़ें →",
        "screen3_title": "🌦️ चरण ३: स्थानीय मौसम व जलवायु विश्लेषण",
        "screen3_desc": "ओपन-मेटियो से तापमान और वर्षा के ऐतिहासिक विचलनों की जांच करें। मौसम का तनाव (गर्मी या सूखे की स्थिति) अक्सर पौधों में फंगस और रोग बढ़ाता है।",
        "btn_fetch_climate": "मौसम डेटा प्राप्त करें 📡",
        "btn_back_leaf": "← पत्ता जांच पर वापस जाएं",
        "btn_goto_summary": "कार्यकारी रिपोर्ट पर आगे बढ़ें →",
        "screen4_title": "📋 चरण ४: संपूर्ण कार्यकारी कृषि सारांश",
        "screen4_desc": "पत्ते के दृश्य लक्षणों और स्थानीय मौसम डेटा का संयुक्त विश्लेषण एवं कृषि वैज्ञानिकों द्वारा अनुमोदित परामर्श।",
        "alignment_title": "🔬 बायोमेटियोरोलॉजिकल रोग-जलवायु तालमेल (Alignment Index)",
        "ipm_title": "🚜 मौसम-अनुकूल फसल सुरक्षा व स्प्रे समय (IPM Advisory)",
        "btn_download_card": "📥 किसान फील्ड हेल्थ कार्ड डाउनलोड करें (Print / PDF)",
        "btn_restart": "🔄 नई जांच शुरू करें",
    },
    "mr": {
        "brand_tag": "अचूक कृषी रोग आणि हवामान विश्लेषण मंच",
        "step1": "ओळख",
        "step2": "पान तपासणी",
        "step3": "हवामान माहिती",
        "step4": "संपूर्ण कृषी सल्ला",
        "hero_badge": "⚡ आधुनिक कृषी निदान तंत्रज्ञान",
        "hero_title": "पानाला ओळखा.<br><span>हवामान समजून घ्या.</span>",
        "hero_sub": "पानाचे एआय फोटो स्कॅनिंग आणि ओपन-मेटिओ हवामान डेटा एकत्र करून पिकाच्या आरोग्याची अचूक माहिती देणारे व्यासपीठ.",
        "btn_begin": "पिकाची तपासणी सुरू करा →",
        "feat1_title": "पान दृष्टी तपासणी",
        "feat1_desc": "पानाचा फोटो अपलोड करा. 14 पिकांमधील रोगांचे अचूक निदान आणि अनिश्चितता व्यवस्थापन.",
        "feat2_title": "हवामान आणि पर्जन्य इतिहास",
        "feat2_desc": "ओपन-मेटिओवरून मागील 3 वर्षांतील तापमान आणि पावसाच्या ऐतिहासिक नोंदींची तपासणी.",
        "feat3_title": "वैज्ञानिक कृषी सुरक्षा",
        "feat3_desc": "एआय अंदाज आणि प्रत्यक्ष शेतातील वस्तुस्थितीमधील अचूक फरक. थेट रासायनिक फवारणी टाळा.",
        "screen2_title": "🍃 पायरी २: पानांच्या आरोग्याची एआय तपासणी",
        "screen2_desc": "पानाचा स्पष्ट फोटो अपलोड करा. न्यूरल नेटवर्क रोगाची लक्षणे ओळखून हीटमॅप तयार करेल.",
        "upload_label": "पानाचा फोटो निवडा (JPG, JPEG, PNG)",
        "btn_run_screening": "तपासणी आणि AI हीटमॅप सुरू करा ⚡",
        "target_crop": "ओळखलेले पीक",
        "screened_condition": "रोगाची स्थिती",
        "confidence": "खात्री / अचूकता",
        "btn_listen": "🔊 रोगनिदान ऐका (Audio)",
        "btn_back_welcome": "← मागे जा",
        "btn_goto_climate": "हवामान माहितीकडे पुढे जा →",
        "screen3_title": "🌦️ पायरी ३: स्थानिक हवामान आणि पर्जन्य विश्लेषण",
        "screen3_desc": "ओपन-मेटिओवरून तापमान आणि पावसाच्या ऐतिहासिक नोंदींची तपासणी करा.",
        "btn_fetch_climate": "हवामान डेटा मिळवा 📡",
        "btn_back_leaf": "← पान तपासणीकडे परत जा",
        "btn_goto_summary": "अंतिम अहवालाकडे पुढे जा →",
        "screen4_title": "📋 पायरी ४: संपूर्ण कृषी सल्ला व अहवाल",
        "screen4_desc": "पानाचे लक्षण आणि स्थानिक हवामान डेटाचा एकत्रित अभ्यास करून तयार केलेला अहवाल.",
        "alignment_title": "🔬 रोग आणि हवामानाचा जैविक ताळमेळ (Alignment Index)",
        "ipm_title": "🚜 हवामानानुसार फवारणी व पीक संरक्षण (IPM Advisory)",
        "btn_download_card": "📥 शेतकरी हेल्थ कार्ड डाउनलोड करा (Print / PDF)",
        "btn_restart": "🔄 नवीन तपासणी सुरू करा",
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
