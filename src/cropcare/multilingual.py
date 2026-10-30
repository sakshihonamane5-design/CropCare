"""CropCare Multilingual & Accessibility Voice Engine.

Provides comprehensive multi-language translations (English, Hindi, Marathi, Telugu, Tamil, Spanish),
vernacular crop & disease localization, and browser-native Text-to-Speech (TTS) audio narration.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional, Tuple

SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "flag": "🇬🇧", "voice_code": "en-US", "label": "🇬🇧 English"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳", "voice_code": "hi-IN", "label": "🇮🇳 हिन्दी (Hindi)"},
    "mr": {"name": "Marathi", "native": "मराठी", "flag": "🇮🇳", "voice_code": "mr-IN", "label": "🇮🇳 मराठी (Marathi)"},
    "te": {"name": "Telugu", "native": "తెలుగు", "flag": "🇮🇳", "voice_code": "te-IN", "label": "🇮🇳 తెలుగు (Telugu)"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳", "voice_code": "ta-IN", "label": "🇮🇳 தமிழ் (Tamil)"},
    "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸", "voice_code": "es-ES", "label": "🇪🇸 Español (Spanish)"},
}

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "app_title": {
        "en": "CropCare — Agronomic Intelligence",
        "hi": "क्रॉपकेयर — कृषि बुद्धिमत्ता प्रणाली",
        "mr": "क्रॉपकेअर — कृषी बुद्धिमत्ता प्रणाली",
        "te": "క్రాప్‌కేర్ — వ్యవసాయ మేధస్సు",
        "ta": "கிராப்கேர் — வேளாண் நுண்ணறிவு",
        "es": "CropCare — Inteligencia Agronómica",
    },
    "app_tagline": {
        "en": "Precision Agronomic Intelligence & Climate Early-Warning",
        "hi": "सटीक कृषि बुद्धिमत्ता एवं मौसम पूर्व-चेतावनी",
        "mr": "अचूक कृषी बुद्धिमत्ता आणि हवामान पूर्वसूचना",
        "te": "ఖచ్చితమైన వ్యవసాయ మేధస్సు & వాతావరణ ముందస్తు హెచ్చరిక",
        "ta": "துல்லியமான வேளாண் நுண்ணறிவு & வானிலை முன்னெச்சரிக்கை",
        "es": "Inteligencia Agronómica de Precisión y Alerta Climática",
    },
    "step_1": {
        "en": "1. Welcome",
        "hi": "१. स्वागत",
        "mr": "१. स्वागत",
        "te": "1. స్వాగతం",
        "ta": "1. வரவேற்பு",
        "es": "1. Bienvenida",
    },
    "step_2": {
        "en": "2. Leaf Screening",
        "hi": "२. पत्ती जांच",
        "mr": "२. पाने तपासणी",
        "te": "2. ఆకు పరీక్ష",
        "ta": "2. இலை பரிசோதனை",
        "es": "2. Análisis Foliar",
    },
    "step_3": {
        "en": "3. Climate Insights",
        "hi": "३. मौसम विश्लेषण",
        "mr": "३. हवामान विश्लेषण",
        "te": "3. వాతావరణం",
        "ta": "3. வானிலை தகவல்",
        "es": "3. Análisis Climático",
    },
    "step_4": {
        "en": "4. Executive Summary",
        "hi": "४. सम्पूर्ण रिपोर्ट",
        "mr": "४. अंतिम अहवाल",
        "te": "4. నివేదిక",
        "ta": "4. இறுதி அறிக்கை",
        "es": "4. Resumen Ejecutivo",
    },
    "precision_tech": {
        "en": "Precision Agricultural Technology",
        "hi": "सटीक कृषि प्रौद्योगिकी",
        "mr": "अचूक कृषी तंत्रज्ञान",
        "te": "ఖచ్చితమైన వ్యవసాయ సాంకేతికత",
        "ta": "துல்லியமான விவசாய தொழில்நுட்பம்",
        "es": "Tecnología Agrícola de Precisión",
    },
    "hero_title": {
        "en": "Read the leaf. Read the weather.",
        "hi": "पत्ती को पहचानें। मौसम को समझें।",
        "mr": "पाने ओळखा. हवामान समजून घ्या.",
        "te": "ఆకును చదవండి. వాతావరణాన్ని అర్థం చేసుకోండి.",
        "ta": "இலையைப் படியுங்கள். வானிலையை அறியுங்கள்.",
        "es": "Lea la hoja. Conozca el clima.",
    },
    "welcome_title": {
        "en": "Read the leaf.<br><span>Read the weather.</span>",
        "hi": "पत्ती को पहचानें।<br><span>मौसम को समझें।</span>",
        "mr": "पाने ओळखा.<br><span>हवामान समजून घ्या.</span>",
        "te": "ఆకును చదవండి.<br><span>వాతావరణాన్ని అర్థం చేసుకోండి.</span>",
        "ta": "இலையைப் படியுங்கள்.<br><span>வானிலையை அறியுங்கள்.</span>",
        "es": "Lea la hoja.<br><span>Conozca el clima.</span>",
    },
    "hero_sub": {
        "en": "CropCare bridges transfer-learning computer vision with hyper-local climate intelligence and voice advisories for farmers.",
        "hi": "क्रॉपकेयर कंप्यूटर विज़न, स्थानीय मौसम विश्लेषण और किसानों के लिए आवाज सलाह को एक साथ लाता है।",
        "mr": "क्रॉपकेअर कॉम्प्युटर व्हिजन, स्थानिक हवामान विश्लेषण आणि शेतकऱ्यांसाठी ध्वनी सल्ला एकत्र आणते.",
        "te": "క్రాప్‌కేర్ కంప్యూటర్ విజన్ మరియు వాతావరణ విశ్లేషణను రైతులకు అందిస్తుంది.",
        "ta": "கிராப்கேர் கணினி பார்வை மற்றும் உள்ளூர் வானிலை நுண்ணறிவை விவசாயிகளுக்கு வழங்குகிறது.",
        "es": "CropCare une la visión por computadora con inteligencia climática local y asesoría por voz para agricultores.",
    },
    "welcome_sub": {
        "en": "A dual-evidence agronomic screening platform combining MobileNetV3 deep learning leaf screening with Open-Meteo microclimate anomaly intelligence.",
        "hi": "मोबाइलनेटवी३ डीप लर्निंग पत्ती जांच और ओपन-मेटियो सूक्ष्म-जलवायु विसंगति विश्लेषण को जोड़ने वाला एकीकृत कृषि मंच।",
        "mr": "मोबाईलनेटव्ही३ डीप लर्निंग पाने तपासणी आणि ओपन-मेटिओ हवामान विश्लेषण एकत्र आणणारे आधुनिक कृषी व्यासपीठ.",
        "te": "డీప్ లెర్నింగ్ ఆకు పరీక్ష మరియు వాతావరణ విశ్లేషణను కలిపే సమగ్ర వ్యవసాయ వేదిక.",
        "ta": "ஆழ்ந்த கற்றல் இலை பரிசோதனை மற்றும் நுண் வானிலை நுண்ணறிவை இணைக்கும் வேளாண் தளம்.",
        "es": "Plataforma agronómica de doble evidencia que combina visión por computadora MobileNetV3 con inteligencia climática de Open-Meteo.",
    },
    "start_screening": {
        "en": "Start Leaf Screening",
        "hi": "पत्ती जांच शुरू करें",
        "mr": "पाने तपासणी सुरू करा",
        "te": "ఆకు పరీక్ష ప్రారంభించండి",
        "ta": "இலை பரிசோதனையைத் தொடங்குங்கள்",
        "es": "Iniciar Análisis de Hoja",
    },
    "continual_studio": {
        "en": "Continual Learning Studio (Teach Model)",
        "hi": "सतत शिक्षण स्टूडियो (मॉडल को सिखाएं)",
        "mr": "निरंतर शिक्षण स्टुडिओ (मॉडेलला शिकवा)",
        "te": "కొత్త పంట నేర్పండి",
        "ta": "புதிய பயிர் பயிற்சி",
        "es": "Estudio de Aprendizaje Continuo",
    },
    "judges_eval_btn": {
        "en": "📊 Model Benchmark Metrics & Class Error Analysis (For Judges) →",
        "hi": "📊 मॉडल बेंचमार्क और त्रुटि विश्लेषण (निर्णायकों के लिए) →",
        "mr": "📊 मॉडेल अचूकता मेट्रिक्स आणि त्रुटी विश्लेषण (परीक्षकांसाठी) →",
        "te": "📊 మోడల్ బెంచ్‌మార్క్ & లోపాల విశ్లేషణ →",
        "ta": "📊 மாதிரி மதிப்பீடு & பிழை பகுப்பாய்வு →",
        "es": "📊 Métricas del Modelo y Análisis de Errores (Jueces) →",
    },
    "farmer_badge": {
        "en": "Field Intelligence Active",
        "hi": "खेत बुद्धिमत्ता सक्रिय",
        "mr": "शेतकरी बुद्धिमत्ता सक्रिय",
        "te": "ఫీల్డ్ ఇంటెలిజెన్స్ యాక్టివ్",
        "ta": "கள நுண்ணறிவு செயலில்",
        "es": "Inteligencia de Campo Activa",
    },
    "caption_title": {
        "en": "Real-Time Agronomic Assessment",
        "hi": "सटीक वास्तविक समय फसल मूल्यांकन",
        "mr": "थेट वेळेत पीक आरोग्य मूल्यांकन",
        "te": "రియల్-టైమ్ పంట అంచనా",
        "ta": "நிகழ்நேர பயிர் மதிப்பீடு",
        "es": "Evaluación Agronómica en Tiempo Real",
    },
    "caption_sub": {
        "en": "Field-tested intelligence connecting visual foliar symptoms with historical weather dynamics.",
        "hi": "पत्तियों के दृश्य लक्षणों को ऐतिहासिक मौसम चक्र से जोड़ने वाली कृषि प्रणाली।",
        "mr": "पानावरील लक्षणांना ऐतिहासिक हवामानाशी जोडणारी प्रगत कृषी प्रणाली.",
        "te": "ఆకు లక్షణాలను వాతావరణంతో అనుసంధానించే సాంకేతికత.",
        "ta": "இலை அறிகுறிகளை வரலாற்று வானிலையுடன் இணைக்கும் வேளாண் நுண்ணறிவு.",
        "es": "Inteligencia probada en campo que conecta síntomas foliares con la dinámica climática histórica.",
    },
    "feat1_title": {
        "en": "Targeted Crop Screening",
        "hi": "लक्षित फसल जांच",
        "mr": "लक्षित पीक तपासणी",
        "te": "లక్ష్య పంట పరీక్ష",
        "ta": "இலக்கு பயிர் பரிசோதனை",
        "es": "Análisis de Cultivos Dirigido",
    },
    "feat1_desc": {
        "en": "Select your crop species (Tomato, Potato, Corn, Apple, etc.) to isolate known pathologies, eliminating cross-species confusion and delivering maximum real-time confidence.",
        "hi": "अपनी फसल (टमाटर, आलू, मक्का, सेब आदि) चुनें ताकि केवल उसी फसल के रोगों का विश्लेषण हो और उच्चतम सटीकता प्राप्त हो सके।",
        "mr": "आपले पीक (टोमॅटो, बटाटा, मका, सफरचंद इ.) निवडा जेणेकरून इतर पिकांचा संभ्रम न होता सर्वोच्च अचूकता मिळेल.",
        "te": "మీ పంటను ఎంచుకోండి, ఇతర పంటల గందరగోళాన్ని నివారించి అత్యున్నత ఖచ్చితత్వాన్ని పొందండి.",
        "ta": "உங்கள் பயிரைத் தேர்ந்தெடுத்து துல்லியமான நோயறிதலைப் பெறுங்கள்.",
        "es": "Seleccione su cultivo (Tomate, Papa, Maíz, etc.) para aislar patologías conocidas y maximizar la certeza diagnóstica.",
    },
    "feat2_title": {
        "en": "Continual Learning Studio",
        "hi": "सतत शिक्षण स्टूडियो",
        "mr": "निरंतर शिक्षण स्टुडिओ",
        "te": "నిరంతర అభ్యాస స్టూడియో",
        "ta": "தொடர் கற்றல் அரங்கம்",
        "es": "Estudio de Aprendizaje Continuo",
    },
    "feat2_desc": {
        "en": "Encountered an unmonitored crop (e.g. Rice, Wheat, Cotton)? Upload reference photos in the Continual Learning tab to train the dynamic 1024-dim neural feature bank in real time!",
        "hi": "नई फसल (जैसे धान, गेहूं, कपास)? फोटो अपलोड करें और 1024-आयामी न्यूरल मेमोरी में तुरंत नई फसलों व रोगों को सिखाएं!",
        "mr": "नवीन पीक (उदा. भात, गहू, कापूस)? फोटो अपलोड करून 1024-डायमेंशनल मेमरी बँकेत तत्काळ नवीन पिकांचे प्रशिक्षण द्या!",
        "te": "కొత్త పంట లేదా కొత్త తెగులు కనిపిస్తే, రిఫరెన్స్ ఫోటోను అప్‌లోడ్ చేసి సిస్టమ్‌కు నేర్పండి.",
        "ta": "புதிய பயிர் அல்லது நோயைக் கண்டால், புகைப்படத்தை பதிவேற்றி மாதிரிக்கு உடனடியாக கற்றுக்கொடுங்கள்.",
        "es": "¿Cultivo no monitoreado (Arroz, Trigo, Algodón)? ¡Suba fotos de referencia para enseñar al banco de memoria neuronal en tiempo real!",
    },
    "feat3_title": {
        "en": "Agronomic Safeguards",
        "hi": "कृषि सुरक्षा और सावधानियां",
        "mr": "कृषी सुरक्षा आणि खबरदारी",
        "te": "వ్యవసాయ భద్రతా ప్రమాణాలు",
        "ta": "வேளாண் பாதுகாப்பு நெறிமுறைகள்",
        "es": "Salvaguardas Agronómicas",
    },
    "feat3_desc": {
        "en": "Strict boundary between statistical model certainty and confirmed biology. Always advocates field verification before chemical intervention.",
        "hi": "सांख्यिकीय मॉडल और वास्तविक जैविक स्थिति के बीच अंतर समझें। रासायनिक छिड़काव से पहले हमेशा खेत सत्यापन जरूरी है।",
        "mr": "सांख्यिकीय अंदाज आणि प्रत्यक्ष परिस्थिती यात फरक असतो. रासायनिक फवारणीपूर्वी प्रत्यक्ष तपासणी आवश्यक आहे.",
        "te": "రసాయన పిచికారీ చేయడానికి ముందు క్షేత్రస్థాయి పరిశీలన తప్పనిసరి.",
        "ta": "இரசாயன தெளிப்புக்கு முன் எப்போதும் கள சரிபார்ப்பை மேற்கொள்ளுங்கள்.",
        "es": "Límite estricto entre certeza estadística y biología confirmada. Siempre promueve verificación en campo antes de químicos.",
    },
    "screen2_title": {
        "en": "🍃 Step 2: Optical Leaf Health Screening & Continual Learning",
        "hi": "🍃 चरण २: पत्ती स्वास्थ्य जांच एवं सतत शिक्षण",
        "mr": "🍃 पायरी २: पानांची आरोग्य तपासणी आणि निरंतर शिक्षण",
        "te": "🍃 దశ 2: ఆకు ఆరోగ్య పరీక్ష & నిరంతర అభ్యాసం",
        "ta": "🍃 படி 2: இலை ஆரோக்கிய பரிசோதனை & தொடர் கற்றல்",
        "es": "🍃 Paso 2: Análisis Óptico de Salud Foliar y Aprendizaje Continuo",
    },
    "screen2_desc": {
        "en": "Screen crop leaves across 14 supported species or register new reference observations into the active learning memory bank to teach the system new crops and pathologies in real time.",
        "hi": "१४ फसलों की पत्तियों की जांच करें या नए फसल/रोगों को सिखाने के लिए सक्रिय शिक्षण मेमोरी में संदर्भ फोटो दर्ज करें।",
        "mr": "१४ पिकांच्या पानांची तपासणी करा किंवा नवीन पिकांचे फोटो मेमरी बँकेत जोडून मॉडेलला त्वरित प्रशिक्षित करा.",
        "te": "14 రకాల పంట ఆకులను పరీక్షించండి లేదా కొత్త పంటలను నేర్పడానికి ఫోటోను నమోదు చేయండి.",
        "ta": "14 பயிர் வகைகளை பரிசோதிக்கவும் அல்லது புதிய பயிர்களை பயிற்றுவிக்க புகைப்படத்தை பதிவு செய்யவும்.",
        "es": "Analice hojas de cultivos en 14 especies o registre nuevas observaciones en el banco de memoria para enseñar nuevos cultivos en tiempo real.",
    },
    "tab_screening": {
        "en": "🍃 1. Diagnostic Leaf Screening",
        "hi": "🍃 १. पत्ती रोग निदान जांच",
        "mr": "🍃 १. पाने रोग निदान तपासणी",
        "te": "🍃 1. ఆకు రోగ నిర్ధారణ",
        "ta": "🍃 1. இலை நோய் கண்டறிதல்",
        "es": "🍃 1. Diagnóstico Foliar",
    },
    "tab_studio": {
        "en": "🧠 2. Continual Learning Studio (Upload & Teach Model)",
        "hi": "🧠 २. सतत शिक्षण स्टूडियो (मॉडल को सिखाएं)",
        "mr": "🧠 २. निरंतर शिक्षण स्टुडिओ (मॉडेलला शिकवा)",
        "te": "🧠 2. నిరంతర అభ్యాస స్టూడియో",
        "ta": "🧠 2. தொடர் கற்றல் அரங்கம்",
        "es": "🧠 2. Estudio de Aprendizaje Continuo",
    },
    "tab_library": {
        "en": "📚 3. Monitored Crops Library (14 Species, 38 Pathologies)",
        "hi": "📚 ३. समर्थित फसल पुस्तकालय (१४ फसलें, ३८ रोग)",
        "mr": "📚 ३. समर्थित पिकांची माहिती (१४ पिके, ३८ रोग)",
        "te": "📚 3. పంటల సమాచారం (14 రకాలు, 38 తెగుళ్ళు)",
        "ta": "📚 3. ஆதரிக்கப்படும் பயிர்கள் (14 பயிர்கள், 38 நோய்கள்)",
        "es": "📚 3. Biblioteca de Cultivos (14 Especies, 38 Patologías)",
    },
    "tab_gradcam": {
        "en": "🔬 4. Grad-CAM Explainability (AI Vision)",
        "hi": "🔬 ४. ग्रैड-कैम दृश्य स्पष्टीकरण (एआई विज़न)",
        "mr": "🔬 ४. ग्रॅड-कॅम दृष्टी स्पष्टीकरण (AI व्हिजन)",
        "te": "🔬 4. గ్రాడ్-క్యామ్ విశ్లేషణ (AI)",
        "ta": "🔬 4. கிராட்-கேம் விளக்கம் (AI பார்வை)",
        "es": "🔬 4. Explicabilidad Grad-CAM (Visión IA)",
    },
    "tab_evaluation": {
        "en": "📊 5. Model Evaluation & Error Analysis (For Judges)",
        "hi": "📊 ५. मॉडल मूल्यांकन एवं त्रुटि विश्लेषण (निर्णायकों के लिए)",
        "mr": "📊 ५. मॉडेल अचूकता व त्रुटी विश्लेषण (परीक्षकांसाठी)",
        "te": "📊 5. మోడల్ మూల్యాంకనం & విశ్లేషణ",
        "ta": "📊 5. மாதிரி மதிப்பீடு & பிழை பகுப்பாய்வு",
        "es": "📊 5. Evaluación del Modelo y Análisis de Errores (Jueces)",
    },
    "target_crop": {
        "en": "Choose Crop Type to isolate diagnosis (or keep Auto-Detect):",
        "hi": "सटीक निदान के लिए अपनी फसल चुनें (या स्वतः पहचान रखें):",
        "mr": "अचूक निदानासाठी तुमचे पीक निवडा (किंवा स्वयंचलित ठेवा):",
        "te": "ఖచ్చితమైన రోగ నిర్ధారణ కోసం పంటను ఎంచుకోండి:",
        "ta": "துல்லியமான நோயறிதலுக்கு உங்கள் பயிரைத் தேர்ந்தெடுக்கவும்:",
        "es": "Elija el tipo de cultivo para aislar el diagnóstico:",
    },
    "upload_photo": {
        "en": "Leaf Image Acquisition",
        "hi": "पत्ती का फोटो चुनें / खींचे",
        "mr": "पानाचा फोटो निवडा / काढा",
        "te": "ఆకు ఫోటోను ఎంచుకోండి",
        "ta": "இலை புகைப்படத்தைப் பதிவேற்றவும்",
        "es": "Adquisición de Imagen Foliar",
    },
    "upload_file_label": {
        "en": "Choose Leaf Image (JPEG, PNG)",
        "hi": "पत्ती की फोटो चुनें (JPEG, PNG)",
        "mr": "पानाचा फोटो निवडा (JPEG, PNG)",
        "te": "ఆకు ఫోటోను ఎంచుకోండి (JPEG, PNG)",
        "ta": "இலை புகைப்படத்தைத் தேர்ந்தெடுக்கவும் (JPEG, PNG)",
        "es": "Seleccionar Imagen Foliar (JPEG, PNG)",
    },
    "camera_input_label": {
        "en": "Take Photo of Leaf with Device Camera",
        "hi": "मोबाइल कैमरे से पत्ती का फोटो खींचें",
        "mr": "कॅमेऱ्याने पानाचा थेट फोटो काढा",
        "te": "కెమెరాతో ఆకు ఫోటో తీయండి",
        "ta": "கேமரா மூலம் இலை புகைப்படம் எடுக்கவும்",
        "es": "Tomar Foto de la Hoja con la Cámara",
    },
    "use_camera_toggle": {
        "en": "📷 Use Device Camera",
        "hi": "📷 कैमरे का उपयोग करें",
        "mr": "📷 कॅमेरा वापरा",
        "te": "📷 కెమెరా ఉపయోగించండి",
        "ta": "📷 கேமராவைப் பயன்படுத்தவும்",
        "es": "📷 Usar Cámara del Dispositivo",
    },
    "sample_leaves_title": {
        "en": "Or Try Real Field Samples:",
        "hi": "या नमूना पत्तियों से तुरंत जांचें:",
        "mr": "किंवा नमुना पानांवरून थेट तपासा:",
        "te": "లేదా నమూనా ఆకులతో పరీక్షించండి:",
        "ta": "அல்லது மாதிரி இலைகளை முயற்சிக்கவும்:",
        "es": "O Pruebe Muestras Reales de Campo:",
    },
    "btn_submit_screening": {
        "en": "⚡ Screen Leaf with AI Model",
        "hi": "⚡ एआई मॉडल से पत्ती की जांच करें",
        "mr": "⚡ AI मॉडेलने पानाची तपासणी करा",
        "te": "⚡ AI మోడల్‌తో ఆకును పరీక్షించండి",
        "ta": "⚡ AI மாதிரி மூலம் இலையை ஆய்வு செய்யவும்",
        "es": "⚡ Analizar Hoja con Modelo IA",
    },
    "diag_result_title": {
        "en": "Diagnostic Screening Result",
        "hi": "रोग निदान परिणाम",
        "mr": "रोग निदान तपासणी निकाल",
        "te": "రోగ నిర్ధారణ ఫలితం",
        "ta": "நோயறிதல் முடிவு",
        "es": "Resultado del Diagnóstico",
    },
    "screening_mode": {
        "en": "Screening Mode",
        "hi": "जांच का तरीका",
        "mr": "तपासणी पद्धत",
        "te": "పరీక్ష విధానం",
        "ta": "ஆய்வு முறை",
        "es": "Modo de Análisis",
    },
    "crop_identified": {
        "en": "Crop Identified",
        "hi": "पहचानी गई फसल",
        "mr": "ओळखलेले पीक",
        "te": "గుర్తించిన పంట",
        "ta": "கண்டறியப்பட்ட பயிர்",
        "es": "Cultivo Identificado",
    },
    "visual_condition": {
        "en": "Visual Condition / Pathology",
        "hi": "लक्षण / रोग की स्थिति",
        "mr": "लक्षणे / रोगाची स्थिती",
        "te": "లక్షణాలు / తెగులు",
        "ta": "அறிகுறிகள் / நோய் நிலை",
        "es": "Condición Visual / Patología",
    },
    "confidence_score": {
        "en": "Confidence Score",
        "hi": "मॉडल निश्चितता (कॉन्फिडेंस)",
        "mr": "मॉडेल खात्री (कॉन्फिडन्स)",
        "te": "ఖచ్చితత్వ స్కోరు",
        "ta": "நம்பகத்தன்மை மதிப்பீடு",
        "es": "Nivel de Certeza del Modelo",
    },
    "top_probs_title": {
        "en": "Top Condition Probabilities",
        "hi": "संभावित रोगों का प्रतिशत",
        "mr": "संभाव्य रोगांची टक्केवारी",
        "te": "అగ్ర సంభావ్యతలు",
        "ta": "அதிக சாத்தியக்கூறுகள்",
        "es": "Probabilidades de Condiciones Principales",
    },
    "severity_title": {
        "en": "🔬 Pathological Severity & Foliar Progression Stage",
        "hi": "🔬 रोग की गंभीरता एवं फैलाव का स्तर",
        "mr": "🔬 रोगाची तीव्रता आणि प्रसाराचा टप्पा",
        "te": "🔬 తెగులు తీవ్రత మరియు దశ",
        "ta": "🔬 நோயின் தீவிரம் மற்றும் நிலை",
        "es": "🔬 Severidad Patológica y Etapa de Progresión Foliar",
    },
    "foliar_damage": {
        "en": "Foliar Damage",
        "hi": "पत्ती का प्रभावित हिस्सा",
        "mr": "बाधित पानाचे प्रमाण",
        "te": "ఆకు నష్టం శాతం",
        "ta": "இலை சேத சதவீதம்",
        "es": "Daño Foliar",
    },
    "urgency_label": {
        "en": "Urgency",
        "hi": "तत्कालता",
        "mr": "तातडी",
        "te": "అత్యవసరం",
        "ta": "அவசரம்",
        "es": "Urgencia",
    },
    "prognosis_label": {
        "en": "Prognosis",
        "hi": "रोग का पूर्वानुमान",
        "mr": "पुढील अंदाज",
        "te": "రోగ అంచనా",
        "ta": "முன்னறிவிப்பு",
        "es": "Pronóstico",
    },
    "economic_title": {
        "en": "💰 Economic Impact & Harvest Risk Estimator",
        "hi": "💰 आर्थिक नुकसान एवं उपज जोखिम आकलन",
        "mr": "💰 आर्थिक नुकसान व उत्पन्न जोखीम अंदाज",
        "te": "💰 ఆర్థిక నష్ట అంచనా",
        "ta": "💰 பொருளாதார இழப்பு மதிப்பீடு",
        "es": "💰 Estimador de Impacto Económico y Riesgo de Cosecha",
    },
    "farm_size_label": {
        "en": "Farm Size (Acres)",
        "hi": "खेत का आकार (एकड़ में)",
        "mr": "शेताचे क्षेत्रफळ (एकर)",
        "te": "పొలం విస్తీర్ణం (ఎకరాలు)",
        "ta": "நிலப்பரப்பு (ஏக்கர்)",
        "es": "Tamaño de la Finca (Acres)",
    },
    "urgency_tier": {
        "en": "Urgency Tier",
        "hi": "प्राथमिकता स्तर",
        "mr": "प्राधान्य पातळी",
        "te": "ప్రాధాన్యత స్థాయి",
        "ta": "முன்னுரிமை நிலை",
        "es": "Nivel de Urgencia",
    },
    "yield_loss_range": {
        "en": "Yield Loss Range",
        "hi": "उपज में संभावित कमी",
        "mr": "उत्पादनातील संभाव्य घट",
        "te": "దిగుబడి నష్టం శ్రేణి",
        "ta": "விளைச்சல் இழப்பு வரம்பு",
        "es": "Rango de Pérdida de Rendimiento",
    },
    "revenue_at_risk": {
        "en": "Revenue At Risk",
        "hi": "संभावित वित्तीय जोखिम",
        "mr": "धोक्यात असलेले उत्पन्न",
        "te": "ఆర్థిక నష్ట ప్రమాదం",
        "ta": "ஆபத்தில் உள்ள வருவாய்",
        "es": "Ingresos en Riesgo",
    },
    "treatment_cost": {
        "en": "Treatment Cost",
        "hi": "उपचार खर्च",
        "mr": "उपाययोजना खर्च",
        "te": "చికిత్స ఖర్చు",
        "ta": "சிகிச்சை செலவு",
        "es": "Costo de Tratamiento",
    },
    "roi_treated": {
        "en": "ROI if Treated",
        "hi": "उपचार का लाभ (ROI)",
        "mr": "उपचाराचा परतावा (ROI)",
        "te": "చికిత్స ప్రయోజనం (ROI)",
        "ta": "சிகிச்சை வருமானம் (ROI)",
        "es": "ROI si se Aplica Tratamiento",
    },
    "ipm_title": {
        "en": "🌿 Weather-Smart IPM Advisory & Spray Guidance",
        "hi": "🌿 मौसम-अनुकूल कीट प्रबंधन (IPM) व छिड़काव सलाह",
        "mr": "🌿 हवामानानुसार कीड व्यवस्थापन (IPM) व फवारणी सल्ला",
        "te": "🌿 వాతావరణ అనుకూల పిచికారీ సలహా (IPM)",
        "ta": "🌿 வானிலைக்கேற்ற பூச்சி மேலாண்மை (IPM)",
        "es": "🌿 Asesoría MIP Inteligente según el Clima",
    },
    "cultural_tactics": {
        "en": "🌾 Cultural Practices",
        "hi": "🌾 कृषि पद्धतियां व स्वच्छता",
        "mr": "🌾 कृषी पद्धती व स्वच्छता",
        "te": "🌾 సాంప్రదాయ పద్ధతులు",
        "ta": "🌾 வேளாண் நடைமுறைகள்",
        "es": "🌾 Prácticas Culturales",
    },
    "biological_tactics": {
        "en": "🧫 Biological Controls",
        "hi": "🧫 जैविक नियंत्रण",
        "mr": "🧫 जैविक नियंत्रण उपाय",
        "te": "🧫 జీవ నియంత్రణలు",
        "ta": "🧫 உயிரியல் கட்டுப்பாடுகள்",
        "es": "🧫 Controles Biológicos",
    },
    "chemical_tactics": {
        "en": "⚗️ Chemical Interventions (Targeted)",
        "hi": "⚗️ रासायनिक उपाय (सत्यापन पश्चात)",
        "mr": "⚗️ रासायनिक उपाय (तपासणीनंतर)",
        "te": "⚗️ రసాయన చర్యలు",
        "ta": "⚗️ இரசாயன தலையீடுகள்",
        "es": "⚗️ Intervenciones Químicas",
    },
    "roadmap_title": {
        "en": "📅 7-Day Agronomic Recovery Roadmap",
        "hi": "📅 ७-दिवसीय फसल सुधार कार्ययोजना",
        "mr": "📅 ७-दिवसांची पीक पुनर्प्राप्ती कृती आराखडा",
        "te": "📅 7 రోజుల పునరుద్ధరణ ప్రణాళిక",
        "ta": "📅 7 நாள் மீட்புத் திட்டம்",
        "es": "📅 Plan de Acción de Recuperación de 7 Días",
    },
    "whatsapp_btn": {
        "en": "📲 Share Alert via WhatsApp",
        "hi": "📲 व्हाट्सएप पर कृषि अधिकारी को भेजें",
        "mr": "📲 व्हॉट्सॲपवर कृषी अधिकाऱ्याला पाठवा",
        "te": "📲 వాట్సాప్‌లో పంపండి",
        "ta": "📲 வாட்ஸ்அப்பில் பகிரவும்",
        "es": "📲 Compartir Alerta por WhatsApp",
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
    "audio_listen_btn": {
        "en": "Listen to Voice Agronomist Advisory",
        "hi": "कृषि सलाहकार की आवाज में सलाह सुनें",
        "mr": "कृषी सल्लागाराच्या आवाजात सल्ला ऐका",
        "te": "వాయిస్ సలహా వినండి",
        "ta": "குரல் ஆலோசனையைக் கேளுங்கள்",
        "es": "Escuchar Asesoría del Agrónomo por Voz",
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
    "btn_back_welcome": {
        "en": "← Back to Welcome",
        "hi": "← मुख्य पृष्ठ पर वापस",
        "mr": "← मुख्य पानावर परत जा",
        "te": "← వెనుకకు",
        "ta": "← மீண்டும் முகப்புக்கு",
        "es": "← Volver a Bienvenida",
    },
    "btn_goto_climate": {
        "en": "Continue to Climate Insights →",
        "hi": "मौसम विश्लेषण की ओर आगे बढ़ें →",
        "mr": "हवामान विश्लेषणाकडे पुढे जा →",
        "te": "వాతావరణ సమాచారానికి వెళ్ళండి →",
        "ta": "வானிலை தகவலுக்குச் செல்லவும் →",
        "es": "Continuar a Análisis Climático →",
    },
    "screen3_title": {
        "en": "🌦️ Step 3: Hyper-Local Environmental Analytics",
        "hi": "🌦️ चरण ३: स्थानीय मौसम एवं पर्यावरण विश्लेषण",
        "mr": "🌦️ पायरी ३: स्थानिक हवामान आणि पर्यावरण विश्लेषण",
        "te": "🌦️ దశ 3: వాతావరణ విశ్లేషణ",
        "ta": "🌦️ படி 3: உள்ளூர் வானிலை பகுப்பாய்வு",
        "es": "🌦️ Paso 3: Análisis Ambiental Hiperlocal",
    },
    "screen3_desc": {
        "en": "Retrieve historical climate observations and multi-year anomaly baselines via Open-Meteo. Environmental stress (heat, moisture deficit, humidity) often predisposes crops to pathogen vulnerability.",
        "hi": "ओपन-मेटियो द्वारा मौसम की ऐतिहासिक स्थिति और विसंगतियां देखें। अत्यधिक गर्मी या नमी फसल को रोगों के प्रति संवेदनशील बनाती है।",
        "mr": "ओपन-मेटिओ द्वारे ऐतिहासिक हवामान व विसंगती तपासा. उष्णता आणि पाण्याचा ताण पिकांवर रोग वाढवू शकतो.",
        "te": "చారిత్రక వాతావరణ డేటా మరియు మార్పులను పరిశీలించండి.",
        "ta": "வரலாற்று வானிலை மற்றும் முரண்பாடுகளை ஆராயுங்கள்.",
        "es": "Obtenga observaciones climáticas históricas y anomalías multianuales con Open-Meteo para evaluar estrés ambiental.",
    },
    "preset_loc_label": {
        "en": "📍 Quick Region Preset (or enter coordinates below)",
        "hi": "📍 कृषि क्षेत्र चुनें (या नीचे अक्षांश-देशांतर दर्ज करें)",
        "mr": "📍 कृषी परिसर निवडा (किंवा खाली अक्षांश-रेखांश भरा)",
        "te": "📍 వ్యవసాయ ప్రాంతం ఎంచుకోండి",
        "ta": "📍 வேளாண் மண்டலத்தைத் தேர்ந்தெடுக்கவும்",
        "es": "📍 Región Agrícola Predefinida (o ingrese coordenadas)",
    },
    "lat_label": {
        "en": "Latitude (°N)",
        "hi": "अक्षांश (°N)",
        "mr": "अक्षांश (°N)",
        "te": "అక్షాంశం (°N)",
        "ta": "அட்சரேகை (°N)",
        "es": "Latitud (°N)",
    },
    "lon_label": {
        "en": "Longitude (°E)",
        "hi": "देशांतर (°E)",
        "mr": "रेखांश (°E)",
        "te": "రేఖాంశం (°E)",
        "ta": "தீர்க்கரேகை (°E)",
        "es": "Longitud (°E)",
    },
    "history_window_label": {
        "en": "Historical Weather Window (Days)",
        "hi": "मौसम डेटा की अवधि (दिन)",
        "mr": "हवामान डेटाचा कालावधी (दिवस)",
        "te": "వాతావరణ కాల వ్యవధి (రోజులు)",
        "ta": "வானிலை கால அளவு (நாட்கள்)",
        "es": "Ventana Histórica de Clima (Días)",
    },
    "btn_fetch_climate": {
        "en": "Fetch Climate Insights ⚡",
        "hi": "मौसम डेटा प्राप्त करें ⚡",
        "mr": "हवामान माहिती मिळवा ⚡",
        "te": "వాతావరణ సమాచారం పొందండి ⚡",
        "ta": "வானிலை தகவலைப் பெறுக ⚡",
        "es": "Obtener Datos Climáticos ⚡",
    },
    "recent_summary": {
        "en": "Recent Summary",
        "hi": "हालिया मौसम सारांश",
        "mr": "नुकताच हवामान सारांश",
        "te": "ఇటీవలి వాతావరణ సారాంశం",
        "ta": "சமீபத்திய வானிலை சுருக்கம்",
        "es": "Resumen Reciente",
    },
    "anomalies_title": {
        "en": "Climate Anomalies vs Historical Baseline",
        "hi": "ऐतिहासिक औसत की तुलना में मौसम विसंगतियां",
        "mr": "ऐतिहासिक सरासरीच्या तुलनेत हवामान विसंगती",
        "te": "వాతావరణ వ్యత్యాసాలు",
        "ta": "வானிலை முரண்பாடுகள்",
        "es": "Anomalías Climáticas vs Línea Base Histórica",
    },
    "alignment_title": {
        "en": "🧬 Biometeorological Disease Alignment Index",
        "hi": "🧬 रोग और मौसम का जैविक तालमेल (अलाइनमेंट इंडेक्स)",
        "mr": "🧬 रोग आणि हवामानाचा जैविक ताळमेळ (अलाइनमेंट इंडेक्स)",
        "te": "🧬 తెగులు-వాతావరణ అనుకూలత",
        "ta": "🧬 நோய் மற்றும் வானிலை பொருத்தம்",
        "es": "🧬 Índice de Alineación Biometeorológica de la Enfermedad",
    },
    "btn_back_leaf": {
        "en": "← Back to Leaf Screening",
        "hi": "← पत्ती जांच पर वापस",
        "mr": "← पाने तपासणीकडे परत",
        "te": "← ఆకు పరీక్షకు తిరిగి వెళ్ళండి",
        "ta": "← இலை பரிசோதனைக்குத் திரும்பு",
        "es": "← Volver a Análisis Foliar",
    },
    "btn_goto_summary": {
        "en": "Proceed to Executive Summary →",
        "hi": "सम्पूर्ण रिपोर्ट की ओर आगे बढ़ें →",
        "mr": "अंतिम अहवालाकडे पुढे जा →",
        "te": "సారాంశ నివేదికకు వెళ్ళండి →",
        "ta": "இறுதி அறிக்கைக்குச் செல்லவும் →",
        "es": "Proceder al Resumen Ejecutivo →",
    },
    "screen4_title": {
        "en": "📋 Step 4: Executive Agronomic Summary",
        "hi": "📋 चरण ४: सम्पूर्ण कृषि सारांश एवं रिपोर्ट",
        "mr": "📋 पायरी ४: संपूर्ण कृषी सल्ला व अंतिम अहवाल",
        "te": "📋 దశ 4: సమగ్ర వ్యవసాయ నివేదిక",
        "ta": "📋 படி 4: இறுதி வேளாண் சுருக்கம்",
        "es": "📋 Paso 4: Resumen Agronómico Ejecutivo",
    },
    "screen4_desc": {
        "en": "Synthesized briefing consolidating optical leaf screening and hyper-local environmental data.",
        "hi": "पत्ती के लक्षण और स्थानीय मौसम के तालमेल से तैयार की गई समग्र रिपोर्ट।",
        "mr": "पानावरील लक्षणे आणि स्थानिक हवामानाचा एकत्रित अभ्यास करून तयार केलेला संपूर्ण अहवाल.",
        "te": "ఆకు లక్షణాలు మరియు వాతావరణ సమాచారంతో కూడిన సమగ్ర నివేదిక.",
        "ta": "இலை பரிசோதனை மற்றும் உள்ளூர் வானிலை ஒருங்கிணைந்த அறிக்கை.",
        "es": "Informe consolidado que sintetiza el análisis foliar y los datos ambientales hiperlocales.",
    },
    "screening_record": {
        "en": "🍃 Leaf Screening Record",
        "hi": "🍃 पत्ती जांच का विवरण",
        "mr": "🍃 पाने तपासणीची नोंद",
        "te": "🍃 ఆకు పరీక్ష రికార్డు",
        "ta": "🍃 இலை பரிசோதனை பதிவு",
        "es": "🍃 Registro de Análisis Foliar",
    },
    "climate_record": {
        "en": "🌦️ Environmental Context Record",
        "hi": "🌦️ पर्यावरण एवं मौसम का विवरण",
        "mr": "🌦️ हवामान व पर्यावरण नोंद",
        "te": "🌦️ వాతావరణ రికార్డు",
        "ta": "🌦️ சுற்றுச்சூழல் பதிவு",
        "es": "🌦️ Registro de Contexto Ambiental",
    },
    "download_report_btn": {
        "en": "📥 Download Field Health Card (HTML/Print as PDF)",
        "hi": "📥 किसान फील्ड हेल्थ कार्ड डाउनलोड करें (प्रिंट / PDF)",
        "mr": "📥 शेतकरी फील्ड हेल्थ कार्ड डाउनलोड करा (प्रिंट / PDF)",
        "te": "📥 ఫీల్డ్ హెల్త్ కార్డు డౌన్‌లోడ్ చేయండి (PDF)",
        "ta": "📥 கள சுகாதார அட்டையைப் பதிவிறக்கவும் (PDF)",
        "es": "📥 Descargar Tarjeta de Salud de Campo (PDF / Impresión)",
    },
    "btn_back_climate": {
        "en": "← Back to Climate Insights",
        "hi": "← मौसम विश्लेषण पर वापस",
        "mr": "← हवामान विश्लेषणाकडे परत",
        "te": "← వాతావరణ సమాచారానికి తిరిగి వెళ్ళండి",
        "ta": "← வானிலை தகவலுக்குத் திரும்பு",
        "es": "← Volver a Análisis Climático",
    },
    "btn_restart": {
        "en": "🔄 Start New Screening Session",
        "hi": "🔄 नई जांच सत्र शुरू करें",
        "mr": "🔄 नवीन तपासणी सत्र सुरू करा",
        "te": "🔄 కొత్త పరీక్షను ప్రారంభించండి",
        "ta": "🔄 புதிய பரிசோதனையைத் தொடங்கவும்",
        "es": "🔄 Iniciar Nueva Sesión de Análisis",
    },
    "chat_btn_open": {
        "en": "💬 CropCare AI",
        "hi": "💬 क्रॉपकेयर एआई",
        "mr": "💬 क्रॉपकेअर AI",
        "te": "💬 క్రాప్‌కేర్ AI",
        "ta": "💬 கிராப்கேர் AI",
        "es": "💬 CropCare IA",
    },
    "chat_btn_close": {
        "en": "✕ Close Assistant",
        "hi": "✕ सहायक बंद करें",
        "mr": "✕ मदतनीस बंद करा",
        "te": "✕ మూసివేయి",
        "ta": "✕ உதவியாளரை மூடு",
        "es": "✕ Cerrar Asistente",
    },
    "chat_panel_title": {
        "en": "🤖 CropCare Agronomic AI Assistant",
        "hi": "🤖 क्रॉपकेयर कृषि विशेषज्ञ एआई सहायक",
        "mr": "🤖 क्रॉपकेअर कृषी तज्ज्ञ AI मदतनीस",
        "te": "🤖 క్రాప్‌కేర్ వ్యవసాయ AI సహాయకుడు",
        "ta": "🤖 கிராப்கேர் வேளாண் AI உதவியாளர்",
        "es": "🤖 Asistente Agronómico IA CropCare",
    },
    "chat_placeholder": {
        "en": "Ask about symptoms, sprays, weather, or crop health...",
        "hi": "लक्षण, छिड़काव, मौसम या फसल स्वास्थ्य के बारे में पूछें...",
        "mr": "लक्षणे, फवारणी, हवामान किंवा पीक आरोग्याबद्दल विचारा...",
        "te": "లక్షణాలు, పిచికారీ, వాతావరణం గురించి అడగండి...",
        "ta": "அறிகுறிகள், தெளிப்பு, வானிலை பற்றி கேட்கவும்...",
        "es": "Pregunte sobre síntomas, rociado, clima o salud del cultivo...",
    },
    "chat_clear_btn": {
        "en": "🗑️ Clear Chat",
        "hi": "🗑️ बातचीत साफ करें",
        "mr": "🗑️ संवाद पुसून टाका",
        "te": "🗑️ చాట్ తొలగించు",
        "ta": "🗑️ உரையாடலை அழிக்கவும்",
        "es": "🗑️ Limpiar Chat",
    },
}

# Crop & Disease Lexicon for Multilingual Translation
CROP_LEXICON: Dict[str, Dict[str, str]] = {
    "Tomato": {"hi": "टमाटर (Tomato)", "mr": "टोमॅटो (Tomato)", "te": "టమోటా (Tomato)", "ta": "தக்காளி (Tomato)", "es": "Tomate"},
    "Potato": {"hi": "आलू (Potato)", "mr": "बटाटा (Potato)", "te": "బంగాళాదుంప (Potato)", "ta": "உருளைக்கிழங்கு (Potato)", "es": "Papa"},
    "Corn": {"hi": "मक्का (Corn / Maize)", "mr": "मका (Corn)", "te": "మొక్కజొన్న (Corn)", "ta": "மக்காச்சோளம் (Corn)", "es": "Maíz"},
    "Corn_(maize)": {"hi": "मक्का (Corn / Maize)", "mr": "मका (Corn)", "te": "మొక్కజొన్న (Corn)", "ta": "மக்காச்சோளம் (Corn)", "es": "Maíz"},
    "Apple": {"hi": "सेब (Apple)", "mr": "सफरचंद (Apple)", "te": "ఆపిల్ (Apple)", "ta": "ஆப்பிள் (Apple)", "es": "Manzana"},
    "Grape": {"hi": "अंगूर (Grape)", "mr": "द्राक्षे (Grape)", "te": "ద్రాక్ష (Grape)", "ta": "திராட்சை (Grape)", "es": "Uva"},
    "Orange": {"hi": "संतरा (Orange)", "mr": "संत्री (Orange)", "te": "నారింజ (Orange)", "ta": "ஆரஞ்சு (Orange)", "es": "Naranja"},
    "Peach": {"hi": "आड़ू (Peach)", "mr": "पीच (Peach)", "te": "పీచ్ (Peach)", "ta": "பீச் (Peach)", "es": "Melocotón"},
    "Pepper,_bell": {"hi": "शिमला मिर्च (Bell Pepper)", "mr": "ढोबळी मिरची (Capsicum)", "te": "బెల్ పెప్పర్ (Bell Pepper)", "ta": "குடைமிளகாய் (Bell Pepper)", "es": "Pimiento"},
    "Strawberry": {"hi": "स्ट्रॉबेरी (Strawberry)", "mr": "स्ट्रॉबेरी (Strawberry)", "te": "స్ట్రాబెర్రీ (Strawberry)", "ta": "ஸ்ட்ராபெரி (Strawberry)", "es": "Fresa"},
    "Cherry_(including_sour)": {"hi": "चेरी (Cherry)", "mr": "चेरी (Cherry)", "te": "చెర్రీ (Cherry)", "ta": "செர்ரி (Cherry)", "es": "Cereza"},
    "Squash": {"hi": "कद्दू / तोरी (Squash)", "mr": "भोपळा (Squash)", "te": "గుమ్మడికాయ (Squash)", "ta": "பூசணிக்காய் (Squash)", "es": "Calabaza"},
    "Soybean": {"hi": "सोयाबीन (Soybean)", "mr": "सोयाबीन (Soybean)", "te": "సోయాబీన్ (Soybean)", "ta": "சோயாபீன் (Soybean)", "es": "Soja"},
    "Blueberry": {"hi": "ब्लूबेरी (Blueberry)", "mr": "ब्लूबेरी (Blueberry)", "te": "బ్లూబెర్రీ (Blueberry)", "ta": "புளூபெர்ரி (Blueberry)", "es": "Arándano"},
    "Raspberry": {"hi": "रास्पबेरी (Raspberry)", "mr": "रास्पबेरी (Raspberry)", "te": "రాస్ప్బెర్రీ (Raspberry)", "ta": "ராஸ்பெர்ரி (Raspberry)", "es": "Frambuesa"},
}

DISEASE_LEXICON: Dict[str, Dict[str, str]] = {
    "early_blight": {"hi": "अगेती झुलसा (Early Blight)", "mr": "अगेती करपा (Early Blight)", "te": "ప్రారంభ ఎండు తెగులు (Early Blight)", "ta": "ஆரம்பகால கருகல் (Early Blight)", "es": "Tizón temprano"},
    "late_blight": {"hi": "पछेती झुलसा (Late Blight)", "mr": "पछेती करपा (Late Blight)", "te": "తీవ్ర ఎండు తెగులు (Late Blight)", "ta": "தாமதமான கருகல் (Late Blight)", "es": "Tizón tardío"},
    "bacterial_spot": {"hi": "जीवाणु धब्बा रोग (Bacterial Spot)", "mr": "जिवाणू ठिपके (Bacterial Spot)", "te": "బ్యాక్టీరియా మచ్చ తెగులు (Bacterial Spot)", "ta": "பாக்டீரியா புள்ளி (Bacterial Spot)", "es": "Mancha bacteriana"},
    "black_rot": {"hi": "काला सड़न रोग (Black Rot)", "mr": "काळा कुजवा (Black Rot)", "te": "నల్ల కుళ్ళు (Black Rot)", "ta": "கருப்பு அழுகல் (Black Rot)", "es": "Podredumbre negra"},
    "powdery_mildew": {"hi": "सफेद चूर्णी फफूंद (Powdery Mildew)", "mr": "भुरी रोग (Powdery Mildew)", "te": "బూడిద తెగులు (Powdery Mildew)", "ta": "சாம்பல் நோய் (Powdery Mildew)", "es": "Oídio"},
    "leaf_scorch": {"hi": "पत्ती झुलसा (Leaf Scorch)", "mr": "करपा (Leaf Scorch)", "te": "ఆకు మాడటం (Leaf Scorch)", "ta": "இலை கருகல் (Leaf Scorch)", "es": "Quemadura foliar"},
    "leaf_spot": {"hi": "पत्ती धब्बा रोग (Leaf Spot)", "mr": "पानावरील ठिपके (Leaf Spot)", "te": "ఆకు మచ్చ తెగులు (Leaf Spot)", "ta": "இலைப்புள்ளி (Leaf Spot)", "es": "Mancha foliar"},
    "rust": {"hi": "गेरुआ / रतुआ फंगस (Rust)", "mr": "तांबेरा रोग (Rust)", "te": "తుప్పు తెగులు (Rust)", "ta": "துரு நோய் (Rust)", "es": "Roya"},
    "scab": {"hi": "स्कैब पपड़ी रोग (Scab)", "mr": "स्कॅब रोग (Scab)", "te": "గజ్జి తెగులు (Scab)", "ta": "சொறி நோய் (Scab)", "es": "Sarna"},
    "yellow_leaf_curl_virus": {"hi": "पत्ती मुड़न विषाणु (Leaf Curl Virus)", "mr": "पर्णगुच्छ विषाणू (Leaf Curl)", "te": "ఆకు ముడుత వైరస్ (Leaf Curl Virus)", "ta": "இலை சுருள் வைரஸ் (Leaf Curl Virus)", "es": "Virus del enrollamiento amarillo"},
    "mosaic_virus": {"hi": "मोज़ेक वायरस (Mosaic Virus)", "mr": "मोझॅक विषाणू (Mosaic Virus)", "te": "మొజాయిక్ వైరస్ (Mosaic Virus)", "ta": "மொசைக் வைரஸ் (Mosaic Virus)", "es": "Virus del mosaico"},
    "spider_mites": {"hi": "लाल मकड़ी कीट (Spider Mites)", "mr": "लाल कोळी कीड (Spider Mites)", "te": "ఎర్ర నల్లి (Spider Mites)", "ta": "சிவப்பு சிலந்தி பூச்சி (Spider Mites)", "es": "Ácaros / Araña roja"},
    "healthy": {"hi": "स्वस्थ और निरोगी फसल (Healthy)", "mr": "निरोगी आणि उत्तम पीक (Healthy)", "te": "ఆరోగ్యకరమైన పంట (Healthy)", "ta": "ஆரோக்கியமான பயிர் (Healthy)", "es": "Cultivo Saludable"},
}


def t(key: str, lang: str = "en") -> str:
    """Retrieve translated string with automatic English fallback."""
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang) or entry.get("en") or key


def translate_crop_name(crop: str, lang: str = "en") -> str:
    """Translate crop name into selected language."""
    if lang == "en" or not crop:
        return crop
    clean_crop = crop.strip()
    match = CROP_LEXICON.get(clean_crop)
    if match and lang in match:
        return match[lang]
    c_lower = clean_crop.lower().replace("_", " ")
    for k, v in CROP_LEXICON.items():
        k_clean = k.lower().replace("_", " ")
        if (k_clean in c_lower or c_lower in k_clean) and lang in v:
            return v[lang]
    return crop


def translate_disease_name(disease: str, lang: str = "en") -> str:
    """Translate disease / pathology condition into selected language."""
    if lang == "en" or not disease:
        return disease
    d_norm = disease.lower().replace(" ", "_").replace("-", "_")
    for k, v in DISEASE_LEXICON.items():
        if k in d_norm and lang in v:
            return v[lang]
    return disease


def translate_diagnosis(crop: str, disease: str, lang: str = "en") -> Tuple[str, str]:
    """Translate crop and disease pair into selected language."""
    return translate_crop_name(crop, lang), translate_disease_name(disease, lang)


def generate_audio_speech_html(title: str = "CropCare Advisory", text_to_speak: str = "", lang: str = "en", element_id: str = "cropcare_tts") -> str:
    """Generate safe, zero-dependency HTML5 SpeechSynthesis component."""
    clean_text = html.escape(text_to_speak.replace("\n", " ").replace('"', "'").strip())
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
    c_trans = translate_crop_name(crop, lang)
    d_trans = translate_disease_name(disease, lang)

    if lang == "hi":
        return (
            f"क्रॉपकेयर फसल स्वास्थ्य रिपोर्ट। फसल: {c_trans}। स्थिति: {d_trans}। "
            f"गंभीरता: {stage_name}, लगभग {damage_pct:.0f} प्रतिशत पत्ती प्रभावित। "
            f"किसान के लिए जरूरी सलाह: "
            f"पहला, सिंचाई: {irrigation_action}। "
            f"दूसरा, खाद: {nutrition_action}। "
            f"तीसरा, स्वच्छता: {sanitation_action}। "
            f"चौथा, जैविक सुरक्षा: {bio_action}।"
        )
    elif lang == "mr":
        return (
            f"क्रॉपकेअर पीक आरोग्य अहवाल. पीक: {c_trans}. स्थिती: {d_trans}. "
            f"गंभीरता: {stage_name}, अंदाजे {damage_pct:.0f} टक्के पाने बाधित. "
            f"शेतकऱ्यांसाठी कृषी सल्ला: "
            f"एक, पाणी नियोजन: {irrigation_action}. "
            f"दोन, खत व्यवस्थापन: {nutrition_action}. "
            f"तीन, स्वच्छता: {sanitation_action}. "
            f"चार, जैविक संरक्षण: {bio_action}."
        )
    elif lang == "te":
        return (
            f"క్రాప్‌కేర్ పంట ఆరోగ్య నివేదిక. పంట: {c_trans}. పరిస్థితి: {d_trans}. "
            f"తీవ్రత: {stage_name}, సుమారు {damage_pct:.0f} శాతం ఆకు దెబ్బతిన్నది. "
            f"రైతుకు ముఖ్యమైన సలహా: "
            f"నీటిపారుదల: {irrigation_action}. "
            f"ఎరువులు: {nutrition_action}. "
            f"పరిశుభ్రత: {sanitation_action}. "
            f"జీవ రక్షణ: {bio_action}."
        )
    elif lang == "ta":
        return (
            f"கிராப்கேர் பயிர் ஆரோக்கிய அறிக்கை. பயிர்: {c_trans}. நிலை: {d_trans}. "
            f"தீவிரம்: {stage_name}, சுமார் {damage_pct:.0f} சதவீதம் இலை பாதிக்கப்பட்டுள்ளது. "
            f"விவசாயிக்கான முக்கிய ஆலோசனை: "
            f"நீர்ப்பாசனம்: {irrigation_action}. "
            f"உரம்: {nutrition_action}. "
            f"சுகாதாரம்: {sanitation_action}. "
            f"உயிரி பாதுகாப்பு: {bio_action}."
        )
    elif lang == "es":
        return (
            f"Informe de salud CropCare. Cultivo: {c_trans}. Condición: {d_trans}. "
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
