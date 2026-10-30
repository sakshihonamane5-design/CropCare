"""CropCare — Agronomic Biometeorological Alignment Engine.

This module bridges the gap between optical leaf-health classification and
micro-environmental weather observations. It evaluates epidemiological concurrence:
whether recent hyper-local temperature, precipitation anomalies, and humidity
favor the biological proliferation of the identified disease pathogen, or whether
abiotic environmental stress (e.g. drought, heat scorch) may explain or exacerbate
the observed symptoms.

Scientific Principles:
1. Plant diseases have well-documented biometeorological envelopes (temperature & humidity thresholds).
2. Concurrence does NOT prove biological causation; it provides probabilistic decision support.
3. Abiotic symptoms (drought chlorosis, heat burn) frequently mimic or compound biotic blights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Epidemiological knowledge base for PlantVillage classes
DISEASE_PROFILES: Dict[str, Dict[str, Any]] = {
    # FUNGAL BLIGHTS & SPOTS (High humidity, moderate/warm temperatures, wet leaves)
    "Early_blight": {
        "category": "Fungal",
        "pathogen": "Alternaria solani",
        "temp_opt": (24, 29),
        "temp_range": (18, 32),
        "min_humidity": 75,
        "prefers_rain": True,
        "abiotic_mimic": "Severe potassium deficiency or heat-induced chlorosis",
    },
    "Late_blight": {
        "category": "Oomycete / Fungal",
        "pathogen": "Phytophthora infestans",
        "temp_opt": (15, 22),
        "temp_range": (10, 25),
        "min_humidity": 85,
        "prefers_rain": True,
        "abiotic_mimic": "Frost injury or waterlogging root necrosis",
    },
    "Powdery_mildew": {
        "category": "Fungal",
        "pathogen": "Erysiphales spp.",
        "temp_opt": (20, 27),
        "temp_range": (15, 30),
        "min_humidity": 60,
        "prefers_rain": False,  # Powdery mildew thrives in high RH but dry leaf surface (inhibited by free water)
        "abiotic_mimic": "Sunscald or pesticide residue deposits",
    },
    "Leaf_spot": {
        "category": "Fungal",
        "pathogen": "Cercospora / Septoria spp.",
        "temp_opt": (20, 28),
        "temp_range": (16, 32),
        "min_humidity": 75,
        "prefers_rain": True,
        "abiotic_mimic": "Ozone injury or chemical drift damage",
    },
    "Black_rot": {
        "category": "Fungal / Bacterial",
        "pathogen": "Guignardia bidwellii / Xanthomonas",
        "temp_opt": (22, 28),
        "temp_range": (18, 32),
        "min_humidity": 80,
        "prefers_rain": True,
        "abiotic_mimic": "Magnesium deficiency or hail bruising",
    },
    "Rust": {
        "category": "Fungal",
        "pathogen": "Puccinia spp.",
        "temp_opt": (18, 24),
        "temp_range": (12, 28),
        "min_humidity": 85,
        "prefers_rain": True,
        "abiotic_mimic": "Iron chlorosis or severe mite feeding",
    },
    "Scab": {
        "category": "Fungal",
        "pathogen": "Venturia inaequalis",
        "temp_opt": (17, 24),
        "temp_range": (10, 26),
        "min_humidity": 80,
        "prefers_rain": True,
        "abiotic_mimic": "Mechanical leaf friction or russeting",
    },

    # BACTERIAL INFECTIONS (Warm, driving rain, splashing, high temperature anomalies)
    "Bacterial_spot": {
        "category": "Bacterial",
        "pathogen": "Xanthomonas campestris",
        "temp_opt": (24, 30),
        "temp_range": (20, 35),
        "min_humidity": 80,
        "prefers_rain": True,
        "abiotic_mimic": "High nitrogen leaf burn or wind desiccation",
    },

    # VIRAL DISEASES (Whitefly / aphid vector driven; thrive in warm, dry spells)
    "Yellow_Leaf_Curl_Virus": {
        "category": "Viral (Vector: Bemisia tabaci)",
        "pathogen": "Begomovirus",
        "temp_opt": (27, 34),
        "temp_range": (22, 38),
        "min_humidity": 40,
        "prefers_rain": False,  # Hot dry weather explodes whitefly vector populations
        "abiotic_mimic": "Severe heat curl or hormonal herbicide damage",
    },
    "Mosaic_virus": {
        "category": "Viral (Vector: Aphids / Mechanical)",
        "pathogen": "Tobamovirus / Potyvirus",
        "temp_opt": (22, 30),
        "temp_range": (18, 35),
        "min_humidity": 50,
        "prefers_rain": None,
        "abiotic_mimic": "Zinc deficiency or genetic variegation",
    },

    # MITES & PESTS (Explode in hot, dry, dusty drought anomalies)
    "Spider_mites": {
        "category": "Arachnid / Pest",
        "pathogen": "Tetranychus urticae",
        "temp_opt": (28, 36),
        "temp_range": (24, 40),
        "min_humidity": 30,
        "prefers_rain": False,  # Extreme heat + drought deficit causes rapid egg development
        "abiotic_mimic": "Atmospheric drought stippling",
    },
}


def _match_disease_profile(disease_str: str) -> Optional[Dict[str, Any]]:
    """Match a disease string against known epidemiological profiles."""
    norm = disease_str.replace(" ", "_").lower()
    for key, profile in DISEASE_PROFILES.items():
        if key.lower() in norm:
            return profile
    # Fallback to general fungal if spots or blight mentioned
    if "blight" in norm or "spot" in norm:
        return DISEASE_PROFILES["Leaf_spot"]
    if "mildew" in norm:
        return DISEASE_PROFILES["Powdery_mildew"]
    if "rot" in norm:
        return DISEASE_PROFILES["Black_rot"]
    return None


def evaluate_environmental_alignment(
    leaf_result: Optional[Dict[str, Any]],
    climate_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate epidemiological concurrence between visual classification and climate data.

    Args:
        leaf_result: Dictionary returned by predict_image().
        climate_result: Dictionary returned by get_environment().

    Returns:
        Structured concurrence report with score, badge, explanation, and risk factors.
    """
    default_res = {
        "status": "INSUFFICIENT_DATA",
        "score": 50,
        "badge": "⚪ Data Incomplete",
        "badge_color": "#526E60",
        "category": "Undetermined",
        "explanation": "Provide both leaf screening and location weather to compute environmental concurrence.",
        "matched_factors": [],
        "abiotic_stress_flag": False,
        "abiotic_stress_summary": None,
    }

    if not leaf_result or not leaf_result.get("predicted_class"):
        return default_res

    disease = leaf_result.get("disease", "Unknown")
    is_healthy = "healthy" in disease.lower()

    if not climate_result or not climate_result.get("recent"):
        default_res["explanation"] = f"Visual screening predicted {disease}, but local weather observations are unavailable."
        return default_res

    recent = climate_result.get("recent") or {}
    anomalies = climate_result.get("anomalies") or {}

    t_mean = recent.get("mean_temperature_c")
    t_max = recent.get("max_temperature_c")
    rain = recent.get("rainfall_mm")
    rh = recent.get("mean_humidity_pct")
    temp_anom = anomalies.get("temperature_c")
    rain_anom = anomalies.get("rainfall_mm")

    # If climate data is missing actual metrics
    if t_mean is None and rain is None:
        default_res["explanation"] = "Local climate station returned null values; cannot compute alignment."
        return default_res

    # Special handling for Healthy plants
    if is_healthy:
        factors = []
        score = 80
        if t_mean is not None and 18 <= t_mean <= 28:
            factors.append(f"Moderate mean temperature ({t_mean:.1f}°C) within normal vegetative envelope.")
        if rain is not None and rain > 15:
            factors.append(f"Adequate moisture availability ({rain:.1f} mm).")
        if temp_anom is not None and abs(temp_anom) > 3.0:
            score -= 20
            factors.append(f"Noticeable temperature anomaly ({temp_anom:+.1f}°C) may induce physiological stress.")

        return {
            "status": "OPTIMAL_GROWTH",
            "score": score,
            "badge": "🟢 Healthy Growth Envelope",
            "badge_color": "#1B543A",
            "category": "Vegetative Balance",
            "explanation": "Leaf visual structure exhibits no diagnostic lesions, and local weather supports normal vegetative development.",
            "matched_factors": factors,
            "abiotic_stress_flag": False,
            "abiotic_stress_summary": None,
        }

    # Match disease profile
    profile = _match_disease_profile(disease)
    if not profile:
        return {
            "status": "MODERATE_ALIGNMENT",
            "score": 60,
            "badge": "🟡 Contextual Plausibility",
            "badge_color": "#B45309",
            "category": "General Observation",
            "explanation": f"Visual symptoms indicate {disease}. General ambient conditions fall within typical seasonal cultivation bounds.",
            "matched_factors": [f"Mean local temperature: {t_mean if t_mean is not None else 'N/A'}°C"],
            "abiotic_stress_flag": False,
            "abiotic_stress_summary": None,
        }

    # Calculate biological concurrence score
    score = 50
    matched_factors: List[str] = []
    abiotic_flag = False
    abiotic_desc = None

    # 1. Temperature Analysis
    if t_mean is not None:
        t_opt_low, t_opt_high = profile["temp_opt"]
        t_rng_low, t_rng_high = profile["temp_range"]
        if t_opt_low <= t_mean <= t_opt_high:
            score += 25
            matched_factors.append(f"Observed temperature ({t_mean:.1f}°C) matches optimal pathogen virulence range ({t_opt_low}–{t_opt_high}°C).")
        elif t_rng_low <= t_mean <= t_rng_high:
            score += 10
            matched_factors.append(f"Observed temperature ({t_mean:.1f}°C) lies within pathogen permissive envelope.")
        else:
            score -= 15
            matched_factors.append(f"Ambient temperature ({t_mean:.1f}°C) departs from standard pathogen incubation range.")

    # 2. Moisture & Humidity Analysis
    if rh is not None:
        min_rh = profile.get("min_humidity", 70)
        if rh >= min_rh:
            score += 20
            matched_factors.append(f"Relative humidity ({rh:.1f}%) exceeds incubation threshold ({min_rh}%).")
        else:
            score -= 10
            matched_factors.append(f"Mean humidity ({rh:.1f}%) is relatively dry for high-pressure sporulation.")

    # 3. Rainfall Anomaly Analysis
    if rain_anom is not None and rain is not None:
        prefers_rain = profile.get("prefers_rain")
        if prefers_rain is True:
            if rain_anom > 5.0 or rain > 35.0:
                score += 20
                matched_factors.append(f"Elevated precipitation ({rain:.1f} mm, anom: {rain_anom:+.1f} mm) facilitates leaf wetness and spore splash.")
            elif rain_anom < -20.0 and rain < 10.0:
                score -= 25
                abiotic_flag = True
                abiotic_desc = (
                    f"Severe rainfall deficit ({rain_anom:+.1f} mm) observed. "
                    f"Visual symptoms may be exacerbated by water stress/drought scorching rather than heavy fungal pressure."
                )
                matched_factors.append(f"Precipitation deficit ({rain_anom:+.1f} mm) suggests dry microclimate.")
        elif prefers_rain is False:
            # Pests like spider mites or powdery mildew prefer dry/hot spells
            if rain_anom < 0 or (temp_anom is not None and temp_anom > 1.5):
                score += 20
                matched_factors.append(f"Warm/dry anomaly ({temp_anom:+.1f}°C) accelerates pest multiplication and foliar desiccation.")

    # Clamp score
    score = max(10, min(95, score))

    if score >= 75:
        status = "HIGH_ALIGNMENT"
        badge = "🟢 Strong Environmental Concurrence"
        badge_color = "#15803D"
        explanation = (
            f"Hyper-local weather conditions in the observation window strongly align with the known biological "
            f"incubators of {profile.get('pathogen', disease)}. Ambient thermal and moisture profiles provide high "
            f"epidemiological plausibility for the visually screened symptoms."
        )
    elif score <= 40 or abiotic_flag:
        status = "DIVERGENT_STRESS"
        badge = "🟠 Environmental Divergence / Abiotic Stress"
        badge_color = "#D97706"
        explanation = (
            f"While optical screening identifies symptoms resembling {disease}, recent environmental telemetry reveals "
            f"significant divergence from classic biological conditions. {abiotic_desc or 'Abiotic factors like drought or heat stress likely contribute to foliage chlorosis.'}"
        )
    else:
        status = "MODERATE_ALIGNMENT"
        badge = "🟡 Moderate Climate Plausibility"
        badge_color = "#B45309"
        explanation = (
            f"Recent weather provides moderate plausibility for {disease}. Conditions fall within acceptable "
            f"pathogen parameters, but lack extreme incubation spikes. On-field scouting is recommended."
        )

    return {
        "status": status,
        "score": score,
        "badge": badge,
        "badge_color": badge_color,
        "category": profile.get("category", "Agricultural Pathology"),
        "pathogen": profile.get("pathogen", "Biological Pathogen"),
        "explanation": explanation,
        "matched_factors": matched_factors,
        "abiotic_stress_flag": abiotic_flag,
        "abiotic_stress_summary": abiotic_desc,
        "abiotic_mimic": profile.get("abiotic_mimic"),
    }
