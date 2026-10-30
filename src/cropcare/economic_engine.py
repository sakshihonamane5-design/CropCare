"""CropCare — Economic Impact & Yield Risk Estimation Engine.

Translates visual disease diagnoses and environmental stress metrics into
tangible economic metrics for farmers and agricultural loan officers:
- Estimated yield loss percentage range
- Financial value at risk per acre (in INR ₹ and USD $)
- Intervention cost vs net harvest preservation (ROI)
- Urgency tier and action priority
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Agronomic benchmarks per crop: (avg_yield_tons_acre, farmgate_price_inr_per_ton)
CROP_ECONOMIC_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "Tomato": {"yield_tons_acre": 18.0, "price_per_ton_inr": 22000.0, "baseline_rev_acre": 396000.0},
    "Potato": {"yield_tons_acre": 12.0, "price_per_ton_inr": 16000.0, "baseline_rev_acre": 192000.0},
    "Corn_(maize)": {"yield_tons_acre": 3.8, "price_per_ton_inr": 21000.0, "baseline_rev_acre": 79800.0},
    "Grape": {"yield_tons_acre": 10.0, "price_per_ton_inr": 45000.0, "baseline_rev_acre": 450000.0},
    "Apple": {"yield_tons_acre": 8.5, "price_per_ton_inr": 65000.0, "baseline_rev_acre": 552500.0},
    "Pepper__bell": {"yield_tons_acre": 14.0, "price_per_ton_inr": 28000.0, "baseline_rev_acre": 392000.0},
    "Strawberry": {"yield_tons_acre": 7.0, "price_per_ton_inr": 90000.0, "baseline_rev_acre": 630000.0},
    "Peach": {"yield_tons_acre": 8.0, "price_per_ton_inr": 50000.0, "baseline_rev_acre": 400000.0},
    "Orange": {"yield_tons_acre": 11.0, "price_per_ton_inr": 35000.0, "baseline_rev_acre": 385000.0},
    "Soybean": {"yield_tons_acre": 1.2, "price_per_ton_inr": 42000.0, "baseline_rev_acre": 50400.0},
    "Cherry": {"yield_tons_acre": 5.0, "price_per_ton_inr": 120000.0, "baseline_rev_acre": 600000.0},
    "Squash": {"yield_tons_acre": 12.0, "price_per_ton_inr": 18000.0, "baseline_rev_acre": 216000.0},
}


# Estimated yield loss ranges if untreated by disease type
DISEASE_LOSS_PROFILES: Dict[str, Dict[str, Any]] = {
    "Late_blight": {
        "loss_min_pct": 50,
        "loss_max_pct": 95,
        "urgency": "🚨 EMERGENCY (Within 48h)",
        "urgency_badge": "Critical",
        "description": "Late blight moves rapidly via sporangia under humid weather; can wipe out entire plots within 7-10 days.",
    },
    "Early_blight": {
        "loss_min_pct": 20,
        "loss_max_pct": 45,
        "urgency": "⚠️ HIGH PRIORITY (3-5 Days)",
        "urgency_badge": "High",
        "description": "Progressive defoliation causes sunscald on fruits and premature vine death.",
    },
    "Bacterial_spot": {
        "loss_min_pct": 25,
        "loss_max_pct": 55,
        "urgency": "⚠️ HIGH PRIORITY (3-5 Days)",
        "urgency_badge": "High",
        "description": "Causes unmarketable fruit spotting and severe canopy defoliation.",
    },
    "Black_rot": {
        "loss_min_pct": 40,
        "loss_max_pct": 80,
        "urgency": "🚨 CRITICAL (48-72h)",
        "urgency_badge": "Critical",
        "description": "Mummifies grape berries and attacks young shoot growth.",
    },
    "Powdery_mildew": {
        "loss_min_pct": 15,
        "loss_max_pct": 35,
        "urgency": "🟡 MODERATE (5-7 Days)",
        "urgency_badge": "Moderate",
        "description": "White mycelium blocks photosynthetic active radiation, stunting fruit sizing.",
    },
    "Leaf_scorch": {
        "loss_min_pct": 15,
        "loss_max_pct": 30,
        "urgency": "🟡 MODERATE (5-7 Days)",
        "urgency_badge": "Moderate",
        "description": "Reduces effective leaf area and crown vigor for subsequent flower flushes.",
    },
    "healthy": {
        "loss_min_pct": 0,
        "loss_max_pct": 0,
        "urgency": "✅ ZERO RISK — MAINTAIN GOOD PRACTICES",
        "urgency_badge": "Negligible",
        "description": "Foliage exhibits vigorous chlorophyll concentration and no visible lesion necroses.",
    },
}


def estimate_economic_impact(
    crop_name: str,
    disease_name: str,
    confidence: float = 0.85,
    farm_acres: float = 1.0,
) -> Dict[str, Any]:
    """Calculate projected yield loss, revenue at risk, and intervention ROI.

    Parameters:
        crop_name: Normalized crop name (e.g. 'Tomato', 'Potato')
        disease_name: Normalized disease key (e.g. 'Early_blight', 'healthy')
        confidence: Model prediction certainty (0.0 to 1.0)
        farm_acres: Farm landholding size in acres

    Returns:
        Structured dictionary with economic and loss indicators.
    """
    bench = CROP_ECONOMIC_BENCHMARKS.get(
        crop_name,
        {"yield_tons_acre": 10.0, "price_per_ton_inr": 25000.0, "baseline_rev_acre": 250000.0},
    )

    # Match disease profile
    profile = None
    for key, val in DISEASE_LOSS_PROFILES.items():
        if key.lower() in disease_name.lower():
            profile = val
            break
    if profile is None:
        if "healthy" in disease_name.lower():
            profile = DISEASE_LOSS_PROFILES["healthy"]
        else:
            profile = {
                "loss_min_pct": 15,
                "loss_max_pct": 40,
                "urgency": "⚠️ ATTENTION NEEDED (5-7 Days)",
                "urgency_badge": "Moderate",
                "description": "Visual symptoms indicate potential fungal or physiological stress.",
            }

    loss_min = profile["loss_min_pct"]
    loss_max = profile["loss_max_pct"]
    is_healthy = loss_max == 0

    baseline_rev = bench["baseline_rev_acre"] * farm_acres
    loss_rev_min = round((baseline_rev * (loss_min / 100.0)), -2)
    loss_rev_max = round((baseline_rev * (loss_max / 100.0)), -2)

    # Treatment costs per acre (IPM bio-fungicide, spray equipment labor)
    treatment_cost_per_acre = 1500.0 if not is_healthy else 0.0
    total_treatment_cost = treatment_cost_per_acre * farm_acres

    # Projected net savings
    net_savings_min = max(0.0, loss_rev_min - total_treatment_cost)
    net_savings_max = max(0.0, loss_rev_max - total_treatment_cost)

    roi_multiplier = (
        round(loss_rev_max / total_treatment_cost, 1)
        if total_treatment_cost > 0 and loss_rev_max > 0
        else 0.0
    )

    return {
        "crop": crop_name,
        "disease": disease_name,
        "is_healthy": is_healthy,
        "urgency": profile["urgency"],
        "urgency_badge": profile["urgency_badge"],
        "loss_pct_range": f"{loss_min}% – {loss_max}%" if not is_healthy else "0%",
        "loss_min_pct": loss_min,
        "loss_max_pct": loss_max,
        "baseline_harvest_value_inr": round(baseline_rev),
        "loss_value_min_inr": round(loss_rev_min),
        "loss_value_max_inr": round(loss_rev_max),
        "treatment_cost_inr": round(total_treatment_cost),
        "net_savings_inr": f"₹{int(net_savings_min):,} – ₹{int(net_savings_max):,}" if not is_healthy else "₹0",
        "roi_multiplier": f"{roi_multiplier}x" if roi_multiplier > 0 else "N/A",
        "description": profile["description"],
    }

