"""CropCare Multilingual & Accessibility Voice Engine.

Provides multi-language translations (English, Hindi, Marathi, Telugu, Tamil, Spanish)
and browser-native Text-to-Speech (TTS) audio narration for rural and field agriculturalists.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional

# Supported application languages
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "flag": "🇬🇧", "voice_code": "en-US"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳", "voice_code": "hi-IN"},
    "mr": {"name": "Marathi", "native": "मराठी", "flag": "🇮🇳", "voice_code": "mr-IN"},
    "te": {"name": "Telugu", "native": "తెలుగు", "flag": "🇮🇳", "voice_code": "te-IN"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳", "voice_code": "ta-IN"},
    "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸", "voice_code": "es-ES"},
}

# Core application UI vocabulary
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "app_title": {
        "en": "CropCare — Agronomic Intelligence",
        "hi": "क्रॉपकेयर — कृषि बुद्धिमत्ता प्रणाली",
        "mr": "क्रॉपकेअर — कृषी बुद्धिमत्ता प्रणाली",
        "te": "క్రాప్‌కేర్ — వ్యవసాయ మేధస్సు",
        "ta": "கிராப்கேர் — வேளாண் நுண்ணறிவு",
        "es": "CropCare — Inteligencia Agronómica",
    },
    "hero_title": {
        "en": "Read the leaf. Read the weather.",
        "hi": "पत्ती को पहचानें। मौसम को समझें।",
        "mr": "पाने ओळखा. हवामान समजून घ्या.",
        "te": "ఆకును చదవండి. వాతావరణాన్ని అర్థం చేసుకోండి.",
        "ta": "இலையைப் படியுங்கள். வானிலையை அறியுங்கள்.",
        "es": "Lea la hoja. Conozca el clima.",
    },
    "hero_sub": {
        "en": "CropCare bridges transfer-learning computer vision with hyper-local climate intelligence and voice advisories for farmers.",
        "hi": "क्रॉपकेयर कंप्यूटर विज़न, स्थानीय मौसम विश्लेषण और किसानों के लिए आवाज सलाह को एक साथ लाता है।",
        "mr": "क्रॉपकेअर कॉम्प्युटर व्हिजन, स्थानिक हवामान विश्लेषण आणि शेतकऱ्यांसाठी ध्वनी सल्ला एकत्र आणते.",
        "te": "క్రాప్‌కేర్ కంప్యూటర్ విజన్ మరియు వాతావరణ విశ్లేషణను రైతులకు అందిస్తుంది.",
        "ta": "கிராப்கேர் கணினி பார்வை மற்றும் உள்ளூர் வானிலை நுண்ணறிவை விவசாயிகளுக்கு வழங்குகிறது.",
        "es": "CropCare une la visión por computadora con inteligencia climática local y asesoría por voz para agricultores.",
    },
    "start_screening": {
        "en": "Start Leaf Screening →",
        "hi": "पत्ती जांच शुरू करें →",
        "mr": "पाने तपासणी सुरू करा →",
        "te": "ఆకు పరీక్ష ప్రారంభించండి →",
        "ta": "இலை பரிசோதனையைத் தொடங்குங்கள் →",
        "es": "Iniciar Análisis de Hoja →",
    },
    "studio_btn": {
        "en": "🧠 Continual Learning Studio Tab →",
        "hi": "🧠 सतत शिक्षण स्टूडियो (नया फसल जोड़ें) →",
        "mr": "🧠 निरंतर शिक्षण स्टुडिओ (नवीन पीक जोडा) →",
        "te": "🧠 కొత్త పంటను నేర్పండి →",
        "ta": "🧠 புதிய பயிரைக் கற்றுக்கொடுங்கள் →",
        "es": "🧠 Estudio de Aprendizaje Continuo →",
    },
    "target_crop": {
        "en": "Choose Crop Type to isolate diagnosis (or keep Auto-Detect):",
        "hi": "सटीक निदान के लिए अपनी फसल चुनें (या स्वतः पहचान रखें):",
        "mr": "अचूक निदानासाठी तुमचे पीक निवडा (किंवा स्वयंचलित ठेवा):",
        "te": "ఖచ్చితమైన రోగ నిర్ధారణ కోసం పంటను ఎంచుకోండి:",
        "ta": "துல்லியமான நோயறிதலுக்கு உங்கள் பயிரைத் தேர்ந்தெடுக்கவும்:",
        "es": "Elija el tipo de cultivo para aislar el diagnóstico:",
    },
    "listen_audio": {
        "en": "🔊 Listen to Agronomic Advisory (Voice Reader)",
        "hi": "🔊 कृषि सलाह ऑडियो सुनें (आवाज में)",
        "mr": "🔊 कृषी सल्ला ऐका (आवाजात)",
        "te": "🔊 వ్యవసాయ సలహాను వినండి",
        "ta": "🔊 வேளாண் ஆலோசனையைக் கேளுங்கள்",
        "es": "🔊 Escuchar asesoría agronómica por voz",
    },
    "stop_audio": {
        "en": "⏹️ Stop Audio",
        "hi": "⏹️ ऑडियो बंद करें",
        "mr": "⏹️ आवाज थांबवा",
        "te": "⏹️ ఆడియో ఆపు",
        "ta": "⏹️ ஆடியோவை நிறுத்து",
        "es": "⏹️ Detener audio",
    },
    "nurturing_title": {
        "en": "🌾 Farmer Nurturing Protocol: Actionable Crop Care Changes",
        "hi": "🌾 किसान पोषण प्रोटोकॉल: फसल सुधार के लिए जरूरी बदलाव",
        "mr": "🌾 शेतकरी संवर्धन प्रोटोकॉल: पीक रक्षणासाठी आवश्यक बदल",
        "te": "🌾 రైతు సంరక్షణ ప్రణాళిక: పంట మార్పులు",
        "ta": "🌾 விவசாயி பராமரிப்பு நெறிமுறை: பயிர் மாற்றங்கள்",
        "es": "🌾 Protocolo de Cuidado: Cambios Agronómicos Recomendados",
    },
    "tab_irrigation": {
        "en": "💧 1. Irrigation Regimen",
        "hi": "💧 १. सिंचाई प्रबंधन",
        "mr": "💧 १. पाणी व्यवस्थापन",
        "te": "💧 1. నీటిపారుదల",
        "ta": "💧 1. நீர்ப்பாசனம்",
        "es": "💧 1. Régimen de Riego",
    },
    "tab_nutrition": {
        "en": "🧪 2. Soil & Nutrients (NPK)",
        "hi": "🧪 २. खाद एवं पोषण (NPK)",
        "mr": "🧪 २. खते व पोषण (NPK)",
        "te": "🧪 2. ఎరువులు (NPK)",
        "ta": "🧪 2. உரம் (NPK)",
        "es": "🧪 2. Suelo y Nutrientes (NPK)",
    },
    "tab_sanitation": {
        "en": "✂️ 3. Canopy & Sanitation",
        "hi": "✂️ ३. छंटाई एवं स्वच्छता",
        "mr": "✂️ ३. छाटणी व स्वच्छता",
        "te": "✂️ 3. కత్తిరింపు & పరిశుభ్రత",
        "ta": "✂️ 3. கவாத்து & சுகாதாரம்",
        "es": "✂️ 3. Poda y Saneamiento",
    },
    "tab_bio_defense": {
        "en": "🛡️ 4. Bio-Defense Shield",
        "hi": "🛡️ ४. जैविक सुरक्षा कवच",
        "mr": "🛡️ ४. जैविक संरक्षण",
        "te": "🛡️ 4. జీవ రక్షణ",
        "ta": "🛡️ 4. உயிரி பாதுகாப்பு",
        "es": "🛡️ 4. Escudo Bio-Defensa",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Retrieve translated string with automatic English fallback."""
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang) or entry.get("en") or key


def generate_audio_speech_html(text: str, lang: str = "en", element_id: str = "cropcare_tts") -> str:
    """Generate safe, zero-dependency HTML5 SpeechSynthesis component.
    
    Operates offline natively in all mobile and desktop browsers with zero cloud TTS latency.
    """
    clean_text = html.escape(text.replace("\n", " ").replace('"', "'").strip())
    voice_code = SUPPORTED_LANGUAGES.get(lang, {}).get("voice_code", "en-US")
    btn_label = t("listen_audio", lang)
    stop_label = t("stop_audio", lang)

    return f"""
    <div style="margin: 0.8rem 0; padding: 0.8rem 1rem; background: #F1F8EB; border: 1.5px solid #B7E66B; border-radius: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span style="font-size: 1.4rem;">🎙️</span>
            <div>
                <div style="font-weight: 700; color: #123D2A; font-size: 0.9rem;">Voice Agronomist Audio Reader</div>
                <div style="font-size: 0.76rem; color: #526E60;">Listen to diagnosis and nurturing advisory in {SUPPORTED_LANGUAGES.get(lang, {}).get('native', 'your language')}</div>
            </div>
        </div>
        <div style="display: flex; gap: 0.4rem;">
            <button id="{element_id}_play_btn" onclick="playCropCareAudio_{element_id}()" 
                style="background: #123D2A; color: #FFFFFF; font-weight: 700; font-size: 0.82rem; border: none; border-radius: 8px; padding: 0.45rem 1rem; cursor: pointer; display: flex; align-items: center; gap: 0.35rem; transition: background 0.2s;">
                {btn_label}
            </button>
            <button id="{element_id}_stop_btn" onclick="stopCropCareAudio_{element_id}()" 
                style="background: #FFFFFF; color: #991B1B; border: 1.5px solid #FCA5A5; font-weight: 600; font-size: 0.82rem; border-radius: 8px; padding: 0.45rem 0.8rem; cursor: pointer;">
                {stop_label}
            </button>
        </div>
    </div>

    <script>
    function playCropCareAudio_{element_id}() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const textToSpeak = "{clean_text}";
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.lang = "{voice_code}";
            utterance.rate = 0.95;
            utterance.pitch = 1.0;

            const voices = window.speechSynthesis.getVoices();
            const matchingVoice = voices.find(v => v.lang === "{voice_code}" || v.lang.startsWith("{lang}"));
            if (matchingVoice) {{
                utterance.voice = matchingVoice;
            }}

            const playBtn = document.getElementById('{element_id}_play_btn');
            playBtn.style.background = '#2A7352';
            playBtn.innerText = '🔊 Speaking...';

            utterance.onend = function() {{
                playBtn.style.background = '#123D2A';
                playBtn.innerText = '{btn_label}';
            }};

            utterance.onerror = function() {{
                playBtn.style.background = '#123D2A';
                playBtn.innerText = '{btn_label}';
            }};

            window.speechSynthesis.speak(utterance);
        }} else {{
            alert('Speech synthesis is not supported on this browser.');
        }}
    }}

    function stopCropCareAudio_{element_id}() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const playBtn = document.getElementById('{element_id}_play_btn');
            if (playBtn) {{
                playBtn.style.background = '#123D2A';
                playBtn.innerText = '{btn_label}';
            }}
        }}
    }}
    </script>
    """


def translate_advisory_text(
    crop: str,
    disease: str,
    stage_name: str,
    damage_pct: float,
    irrigation_action: str,
    nutrition_action: str,
    sanitation_action: str,
    bio_action: str,
    lang: str = "en",
) -> str:
    """Generate audio transcript text tailored to the selected language."""
    if lang == "hi":
        return (
            f"क्रॉपकेयर फसल स्वास्थ्य रिपोर्ट। फसल: {crop}। स्थिति: {disease}। "
            f"गंभीरता: {stage_name}, लगभग {damage_pct:.0f} प्रतिशत पत्ती प्रभावित। "
            f"किसान के लिए जरूरी सलाह: "
            f"पहला, सिंचाई: {irrigation_action}। "
            f"दूसरा, खाद: {nutrition_action}। "
            f"तीसरा, स्वच्छता: {sanitation_action}। "
            f"चौथा, जैविक सुरक्षा: {bio_action}।"
        )
    elif lang == "mr":
        return (
            f"क्रॉपकेअर पीक आरोग्य अहवाल. पीक: {crop}. स्थिती: {disease}. "
            f"गंभीरता: {stage_name}, अंदाजे {damage_pct:.0f} टक्के पाने बाधित. "
            f"शेतकऱ्यांसाठी कृषी सल्ला: "
            f"एक, पाणी नियोजन: {irrigation_action}. "
            f"दोन, खत व्यवस्थापन: {nutrition_action}. "
            f"तीन, स्वच्छता: {sanitation_action}. "
            f"चार, जैविक संरक्षण: {bio_action}."
        )
    elif lang == "es":
        return (
            f"Informe de salud CropCare. Cultivo: {crop}. Condición: {disease}. "
            f"Gravedad: {stage_name}, con {damage_pct:.0f} por ciento de follaje afectado. "
            f"Recomendaciones para el agricultor: "
            f"Riego: {irrigation_action}. "
            f"Nutrición: {nutrition_action}. "
            f"Saneamiento: {sanitation_action}. "
            f"Bio-defensa: {bio_action}."
        )
    else:
        return (
            f"CropCare Plant Health Report. Crop: {crop}. Condition: {disease}. "
            f"Severity: {stage_name}, approximately {damage_pct:.0f} percent foliar damage. "
            f"Farmer Action Plan: "
            f"First, Irrigation: {irrigation_action}. "
            f"Second, Nutrition: {nutrition_action}. "
            f"Third, Sanitation: {sanitation_action}. "
            f"Fourth, Bio-Defense: {bio_action}."
        )
