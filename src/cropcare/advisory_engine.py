"""CropCare — Weather-Smart Agronomic Advisory & IPM Engine.

This module provides actionable Integrated Pest Management (IPM) recommendations
tailored to the predicted crop disease and cross-referenced against current
meteorological conditions.

Core Philosophy:
1. Prioritizes cultural, preventive, and organic/biological options before chemical interventions.
2. Dynamically evaluates weather conditions (rain, heat, wind) to advise WHEN and HOW to spray,
   preventing chemical runoff, wastage, and foliar phytotoxicity.
3. Emphasizes that chemical recommendations are generic guidelines subject to local
   agricultural extension and manufacturer label approval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# IPM Advisory Database for agricultural disease families
IPM_ADVISORY_DB: Dict[str, Dict[str, Any]] = {
    "Late_blight": {
        "disease_name": "Late Blight (Phytophthora infestans)",
        "cultural": [
            "Destroy and bury infected plant cull piles immediately to eliminate primary inoculum.",
            "Avoid overhead sprinkler irrigation; convert to drip or furrow irrigation to minimize leaf wetness.",
            "Maintain wider inter-row spacing to facilitate air circulation and canopy drying.",
        ],
        "biological": [
            "Apply Trichoderma viride or Bacillus subtilis foliar spray as preventive biological shields.",
            "Utilize copper-based soap formulations early in the season before extensive lesions establish.",
        ],
        "chemical": [
            "Foliar protective spray with Mancozeb (75% WP @ 2.5 g/L) or Chlorothalonil (2 g/L).",
            "Under active outbreak conditions, apply systemic Metalaxyl + Mancozeb (2 g/L) with 10–14 day rotation.",
        ],
        "favorable_weather": "Cool, overcast days with high humidity (>85%) and recurrent drizzle.",
    },
    "Early_blight": {
        "disease_name": "Early Blight (Alternaria solani)",
        "cultural": [
            "Prune and burn the lowest 3–4 sets of leaves touching the soil where Alternaria spores overwinter.",
            "Apply thick organic mulch (paddy straw / plastic film) to prevent rainwater splashing spores from soil.",
            "Implement a minimum 2-to-3 year crop rotation with non-solanaceous crops (avoid tomato-potato cycles).",
        ],
        "biological": [
            "Foliar bio-control with Pseudomonas fluorescens (10 g/L) applied every 10 days.",
            "Neem oil (Azadirachtin 0.03% @ 5 mL/L) as a broad-spectrum fungistatic agent.",
        ],
        "chemical": [
            "Chlorothalonil or Mancozeb protective spray upon seeing first concentric ring spots on lower leaves.",
            "For progressing infection, Difenoconazole (25% EC @ 0.5 mL/L) or Azoxystrobin (1 mL/L).",
        ],
        "favorable_weather": "Warm days (24–29°C) alternating with dew-heavy nights or occasional rain showers.",
    },
    "Bacterial_spot": {
        "disease_name": "Bacterial Spot (Xanthomonas spp.)",
        "cultural": [
            "Strictly avoid working in or handling wet crop canopies to prevent mechanical bacteria transfer.",
            "Use certified pathogen-free seeds treated with hot water (50°C for 25 minutes).",
            "Eliminate nightshade weeds and volunteer host plants around field boundaries.",
        ],
        "biological": [
            "Foliar spray of Bacillus amyloliquefaciens or Copper Hydroxide formulations.",
        ],
        "chemical": [
            "Copper Oxychloride (50% WP @ 2.5 g/L) combined with Streptomycin Sulphate (0.1 g/L).",
            "Rotate with Mancozeb to prevent copper-resistant Xanthomonas bacterial strains.",
        ],
        "favorable_weather": "Warm, humid conditions with driving wind-blown rains that create microscopic leaf wounds.",
    },
    "Powdery_mildew": {
        "disease_name": "Powdery Mildew (Erysiphales)",
        "cultural": [
            "Prune dense internal foliage to maximize solar penetration (UV light naturally suppresses mildew conidia).",
            "Avoid excessive nitrogen fertilizer which stimulates lush, susceptible vegetative growth.",
        ],
        "biological": [
            "Wettable Sulfur (80% WP @ 3 g/L) or potassium bicarbonate (5 g/L) foliar spray.",
            "Ampelomyces quisqualis hyper-parasitic bio-fungicide.",
        ],
        "chemical": [
            "Hexaconazole (5% EC @ 1 mL/L) or Myclobutanil (10% WP @ 1 g/L) at first sign of white talcum-like patches.",
        ],
        "favorable_weather": "Warm dry days (20–27°C) with humid nights; inhibited by continuous heavy downpours.",
    },
    "Rust": {
        "disease_name": "Common / Cedar / Leaf Rust (Puccinia spp.)",
        "cultural": [
            "Plant rust-resistant certified hybrid cultivars.",
            "Deep plow crop residue after harvest to bury overwintering teliospores.",
        ],
        "biological": [
            "Bio-spray of Trichoderma harzianum or botanical extracts.",
        ],
        "chemical": [
            "Propiconazole (25% EC @ 1 mL/L) or Tebuconazole (1 mL/L) applied at emergence of pustules.",
        ],
        "favorable_weather": "Prolonged leaf wetness (6–8 hours) combined with moderate temperatures (16–24°C).",
    },
    "Yellow_Leaf_Curl_Virus": {
        "disease_name": "Yellow Leaf Curl Virus (Begomovirus)",
        "cultural": [
            "Install yellow sticky traps (15–20 traps/acre) across field perimeter to monitor and trap whitefly vectors.",
            "Use 40–50 mesh insect-proof netting in seedling nurseries to prevent early vector inoculation.",
            "Rogue out and destroy infected stunting plants within the first 30 days after transplanting.",
        ],
        "biological": [
            "Spray Neem oil (10,000 ppm @ 2 mL/L) or Verticillium lecanii to parasitize whitefly nymphs.",
        ],
        "chemical": [
            "Target vector (Whitefly): Foliar application of Imidacloprid (17.8% SL @ 0.3 mL/L) or Thiamethoxam (25% WG @ 0.3 g/L).",
            "Note: Chemical fungicides do NOT cure viral infections; vector suppression is the sole remedy.",
        ],
        "favorable_weather": "Hot, dry, drought-stressed weather that accelerates whitefly generation turnover.",
    },
    "Spider_mites": {
        "disease_name": "Two-spotted Spider Mite (Tetranychus urticae)",
        "cultural": [
            "Wash foliage with overhead water jets during early infestations to disrupt webbing and dislodge mites.",
            "Keep farm access roads watered to suppress dust, which harms natural mite predators.",
        ],
        "biological": [
            "Release predatory mites (Phytoseiulus persimilis or Neoseiulus californicus).",
            "Spray botanical Pongamia or Neem oil extract @ 3% concentration.",
        ],
        "chemical": [
            "Specific acaricide: Spiromesifen (22.9% SC @ 1 mL/L) or Propargite (57% EC @ 2 mL/L).",
        ],
        "favorable_weather": "Drought stress, high ambient heat (>30°C), and dusty dry spells.",
    },
    "Healthy": {
        "disease_name": "Healthy Plant Tissue",
        "cultural": [
            "Maintain balanced N-P-K fertilization according to local soil health card recommendations.",
            "Monitor soil moisture tension with tensiometers to prevent cyclic water stress.",
            "Continue periodic bi-weekly scouting for early pest or disease foci.",
        ],
        "biological": [
            "Apply prophylactic bio-stimulants (seaweed extract or humic acid) to boost plant immune defense.",
        ],
        "chemical": [
            "No chemical intervention required. Protect beneficial pollinators and soil microbiome.",
        ],
        "favorable_weather": "Balanced seasonal conditions with consistent moisture and mild temperatures.",
    },
}


def _match_advisory(disease_str: str) -> Dict[str, Any]:
    """Retrieve IPM profile matching disease label."""
    norm = disease_str.replace(" ", "_").lower()
    if "healthy" in norm:
        return IPM_ADVISORY_DB["Healthy"]
    for key, data in IPM_ADVISORY_DB.items():
        if key.lower() in norm:
            return data
    # Fallback to Early Blight / general leaf spot recommendations
    if "blight" in norm or "spot" in norm:
        return IPM_ADVISORY_DB["Early_blight"]
    if "mildew" in norm:
        return IPM_ADVISORY_DB["Powdery_mildew"]
    if "virus" in norm:
        return IPM_ADVISORY_DB["Yellow_Leaf_Curl_Virus"]
    if "mite" in norm:
        return IPM_ADVISORY_DB["Spider_mites"]
    return IPM_ADVISORY_DB["Early_blight"]


def generate_actionable_advisory(
    leaf_result: Optional[Any] = None,
    climate_result: Optional[Dict[str, Any]] = None,
    *,
    disease: Optional[str] = None,
    crop: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate integrated agronomic advisory with dynamic weather-smart spray rules.

    Args:
        leaf_result: Dictionary returned by predict_image(), or disease string.
        climate_result: Dictionary returned by get_environment().
        disease: Optional disease name string.
        crop: Optional crop name string.

    Returns:
        Structured IPM plan with cultural, biological, chemical, and spray timing advisories.
    """
    if isinstance(leaf_result, str):
        disease = leaf_result
        leaf_result = None

    if leaf_result is None and disease:
        leaf_result = {
            "predicted_class": f"{crop or 'Crop'}___{disease}",
            "crop": crop or "Crop",
            "disease": disease,
        }

    if climate_result is None and "climate" in kwargs:
        climate_result = kwargs["climate"]

    if not leaf_result or not isinstance(leaf_result, dict) or not leaf_result.get("predicted_class"):
        return {
            "status": "AWAITING_INPUT",
            "title": "Awaiting Leaf Screening",
            "cultural": ["Upload a leaf photograph to generate targeted agronomic action steps."],
            "biological": [],
            "chemical": [],
            "spray_window": {
                "status": "HOLD",
                "badge": "⏸️ No Diagnosis",
                "advice": "Screen leaf health first before planning application windows.",
            },
        }

    target_disease = disease or leaf_result.get("disease", "Unknown")
    is_healthy = "healthy" in str(target_disease).lower()
    profile = _match_advisory(str(target_disease))

    # Weather-Smart Spray Window Analysis
    spray_badge = "🟢 Favorable Application Window"
    spray_status = "OPTIMAL"
    spray_notes: List[str] = []

    if is_healthy:
        spray_badge = "⚪ No Spray Needed"
        spray_status = "NONE"
        spray_notes.append("Plant shows healthy vigor. Chemical fungicides are unnecessary and harmful to beneficial fauna.")
    elif climate_result and climate_result.get("recent"):
        recent = climate_result.get("recent") or {}
        anomalies = climate_result.get("anomalies") or {}

        t_max = recent.get("max_temperature_c") or 25.0
        t_mean = recent.get("mean_temperature_c") or 22.0
        rain = recent.get("rainfall_mm") or 0.0
        rain_anom = anomalies.get("rainfall_mm") or 0.0

        # Check for rain / wash-off risk
        if rain > 20.0 or rain_anom > 15.0:
            spray_status = "HOLD_RAIN"
            spray_badge = "🔴 Rain Washoff Alert (Do Not Spray)"
            spray_notes.append(
                "Recent excessive rainfall indicates saturated foliage. Do NOT apply foliar contact sprays today; "
                "active ingredients will wash off before absorption, polluting soil runoff without controlling fungus."
            )
        elif t_max > 32.0 or (anomalies.get("temperature_c") or 0.0) > 3.0:
            spray_status = "TIME_RESTRICTION"
            spray_badge = "🟡 Dawn Spray Only (Heat Evaporation Risk)"
            spray_notes.append(
                f"High maximum temperatures ({t_max:.1f}°C) observed. Spray strictly during cool early morning hours "
                "(6:00 AM – 8:30 AM). Midday application will cause droplet evaporation and chemical leaf burn (phytotoxicity)."
            )
        else:
            spray_notes.append(
                "Recent ambient temperature and moisture levels fall within optimal agrochemical absorption thresholds. "
                "Ensure spray equipment uses fine mist nozzles for uniform leaf undersurface coverage."
            )
    else:
        spray_badge = "🟡 Weather Unverified (Check Local Sky)"
        spray_status = "CAUTION"
        spray_notes.append("Local microclimate unverified. Ensure no rain is forecasted for 6 hours post-application.")

    return {
        "status": "READY",
        "disease_target": profile.get("disease_name", disease),
        "cultural": profile.get("cultural", []),
        "biological": profile.get("biological", []),
        "chemical": profile.get("chemical", []),
        "spray_window": {
            "status": spray_status,
            "badge": spray_badge,
            "advice": " ".join(spray_notes),
        },
    }


def generate_recovery_roadmap(crop: str, disease: str, spray_status: str = "OPTIMAL") -> List[Dict[str, str]]:
    """Generate an actionable 7-day chronologically structured recovery plan for the farmer."""
    is_healthy = "healthy" in disease.lower()

    if is_healthy:
        return [
            {"day": "Day 1–2", "stage": "Nutritional Maintenance", "action": "Maintain scheduled micro-irrigation and balanced N-P-K fertigation."},
            {"day": "Day 3–4", "stage": "Canopy Inspection", "action": "Inspect lower canopy leaves for early signs of aphids, mites, or micro-lesions."},
            {"day": "Day 5–7", "stage": "Preventive Bio-Shield", "action": "Apply preventive neem oil extract (3 mL/L) to maintain leaf resistance."},
        ]

    spray_timing = "Proceed with morning spray (6:30–8:30 AM) using fine mist nozzles."
    if "HOLD" in spray_status:
        spray_timing = "Rain expected: HOLD spray until foliage dries and rain clears for at least 6 consecutive hours."
    elif "TIME" in spray_status:
        spray_timing = "High temperature alert: Spray strictly between 6:00 AM and 8:00 AM to prevent phytotoxic burn."

    return [
        {
            "day": "Day 1",
            "stage": "🚨 Immediate Field Sanitation",
            "action": f"Carefully prune infected lower leaves showing {disease.replace('_', ' ')} lesions. Do NOT compost; dispose outside the farm boundary to stop sporulation.",
        },
        {
            "day": "Day 2–3",
            "stage": "🎯 Weather-Smart Targeted Treatment",
            "action": f"{spray_timing} Apply recommended biological or contact protective fungicide to both upper and lower leaf surfaces.",
        },
        {
            "day": "Day 4–5",
            "stage": "🧪 Nutrient Balancing & Stress Recovery",
            "action": "Avoid high-nitrogen fertilizer (excess N softens leaf cuticle). Apply soluble Potassium Silicate or Zinc sulphate to strengthen leaf cell walls.",
        },
        {
            "day": "Day 7",
            "stage": "🔬 Efficacy Check & Re-screening",
            "action": "Take fresh leaf photos and re-scan using CropCare. Verify that lesion spread has halted and new apical growth is disease-free.",
        },
    ]


def build_whatsapp_share_url(crop: str, disease: str, confidence_pct: float, location_name: str, urgency: str) -> str:
    """Build a pre-filled WhatsApp Web / Mobile sharing URL for sending field alerts to extension officers."""
    import urllib.parse

    msg = (
        f"🌱 *CropCare Agronomic Alert*\n\n"
        f"• *Crop:* {crop}\n"
        f"• *Diagnosis:* {disease.replace('_', ' ')} ({confidence_pct:.1f}% confidence)\n"
        f"• *Location:* {location_name}\n"
        f"• *Urgency:* {urgency}\n\n"
        f"Generated via CropCare AI Agronomic Platform. Please review recommended IPM treatment."
    )
    return f"https://api.whatsapp.com/send?text={urllib.parse.quote(msg)}"


# Alias for backwards compatibility
get_weather_smart_ipm = generate_actionable_advisory


