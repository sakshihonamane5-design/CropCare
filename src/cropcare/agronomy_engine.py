"""CropCare Agronomy & Disease Pathology Intelligence Engine.

Provides automated visual foliar severity quantification, professional phytopathological stage
classification, and science-backed crop nurturing adjustments (irrigation, nutrition, sanitation, bio-defense).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image


def calculate_foliar_damage_index(image: Image.Image) -> float:
    """Calculate Pathological Foliar Damage Percentage (0-100%) from raw leaf imagery.
    
    Uses color-space segmentation to isolate healthy chlorophyll pixels against
    necrotic, chlorotic, and blighted foliar lesions while ignoring neutral backgrounds.
    """
    try:
        rgb_img = image.convert("RGB")
        arr = np.array(rgb_img, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        brightness = (r + g + b) / 3.0

        # Background exclusion mask (isolating plant tissue from neutral light backgrounds)
        leaf_mask = (brightness < 220) & ((g > 40) | (r > 40) | (b > 40))
        total_leaf_pixels = int(np.sum(leaf_mask))

        if total_leaf_pixels < 100:
            return 0.0

        # Healthy chlorophyll green foliage: green channel clearly exceeds red and blue
        healthy_chlorophyll = leaf_mask & (g > r + 12) & (g > b + 12) & (g > 60)

        # Lesion, chlorosis, necrotic or fungal tissue
        lesion_pixels = leaf_mask & (~healthy_chlorophyll)

        damage_ratio = float(np.sum(lesion_pixels)) / float(total_leaf_pixels)
        damage_pct = round(max(0.0, min(100.0, damage_ratio * 100.0)), 1)
        return damage_pct
    except Exception:
        return 0.0


def classify_disease_severity(
    image: Optional[Image.Image],
    disease_name: str,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """Classify disease severity into professional agronomic stages with physiological metrics.
    
    Stages:
    - Stage 0: Asymptomatic (Pristine Foliar Vigor)
    - Stage I: Incipient / Sub-Clinical Inception (Mild Severity, <15% coverage)
    - Stage II: Established Epiphytotic / Active Folio-Necrosis (Moderate Severity, 15-35% coverage)
    - Stage III: Advanced Acute Senescence / Systemic Necrosis (High Severity, >35% coverage)
    """
    is_healthy = "healthy" in disease_name.lower() or "normal" in disease_name.lower()

    if is_healthy:
        return {
            "stage_id": 0,
            "stage_code": "STAGE_0",
            "stage_name": "Asymptomatic Foliar State",
            "severity_level": "Optimal / Pristine Vigor",
            "damage_percentage": 0.0,
            "color_theme": {
                "badge_bg": "#EBF8D3",
                "badge_text": "#14532D",
                "border": "#A3D95B",
                "progress_gradient": "linear-gradient(90deg, #15803D 0%, #22C55E 100%)",
            },
            "pathological_state": (
                "Normal physiological homeostasis. Active chlorophyll photosynthetic output with no "
                "macroscopic foliar lesions or pathogen colonization."
            ),
            "prognosis": "Favorable. Standard maintenance protocols are recommended to sustain vegetative vigor.",
            "urgency": "Routine Maintenance",
        }

    # For diseased samples, analyze visual foliar damage
    damage_pct = calculate_foliar_damage_index(image) if image is not None else 18.0

    if damage_pct < 15.0:
        return {
            "stage_id": 1,
            "stage_code": "STAGE_1",
            "stage_name": "Stage I: Incipient / Sub-Clinical Inception",
            "severity_level": "Mild (Low Pathological Severity)",
            "damage_percentage": damage_pct,
            "color_theme": {
                "badge_bg": "#FEF3C7",
                "badge_text": "#92400E",
                "border": "#FCD34D",
                "progress_gradient": "linear-gradient(90deg, #F59E0B 0%, #FBBF24 100%)",
            },
            "pathological_state": (
                "Punctate, localized micro-lesions confined to outer epidermal laminae. Pathogen is in early "
                "vegetative colonization before widespread intercellular mycelial expansion."
            ),
            "prognosis": (
                "High recovery probability (>90%) with prompt cultural intervention, canopy aeration, and moisture regulation."
            ),
            "urgency": "Preventative Action Advised (48–72h)",
        }
    elif damage_pct < 35.0:
        return {
            "stage_id": 2,
            "stage_code": "STAGE_2",
            "stage_name": "Stage II: Established Moderate Epiphytotic",
            "severity_level": "Moderate Severity (Active Folio-Necrosis)",
            "damage_percentage": damage_pct,
            "color_theme": {
                "badge_bg": "#FFEDD5",
                "badge_text": "#C2410C",
                "border": "#FDBA74",
                "progress_gradient": "linear-gradient(90deg, #EA580C 0%, #F97316 100%)",
            },
            "pathological_state": (
                "Confluent necrotic spots with expanding chlorotic halos. Vascular transport is moderately "
                "impaired; pathogen is entering secondary sporulation stage."
            ),
            "prognosis": (
                "Guarded. Spread can be arrested with targeted bio-protection and sanitation, but affected foliage will require selective debridement."
            ),
            "urgency": "Active Remediation Required (24–48h)",
        }
    else:
        return {
            "stage_id": 3,
            "stage_code": "STAGE_3",
            "stage_name": "Stage III: Advanced Acute Senescence",
            "severity_level": "Critical / Severe Necrotic Blight",
            "damage_percentage": damage_pct,
            "color_theme": {
                "badge_bg": "#FEE2E2",
                "badge_text": "#991B1B",
                "border": "#FCA5A5",
                "progress_gradient": "linear-gradient(90deg, #DC2626 0%, #EF4444 100%)",
            },
            "pathological_state": (
                "Extensive parenchymal tissue destruction and cellular collapse. High density of active "
                "fruiting bodies/spores with imminent total leaf senescence and cross-canopy dispersal."
            ),
            "prognosis": (
                "Critical. Severe risk of widespread field contagion and yield loss. Immediate physical sanitation and containment mandatory."
            ),
            "urgency": "Immediate Emergency Intervention (<24h)",
        }


def get_nurturing_recommendations(crop: str, disease: str, severity_stage: int = 1) -> Dict[str, Any]:
    """Generate science-backed, professional agronomic nurturing changes for the farmer.
    
    Divided into 4 strategic operational pillars:
    1. Irrigation & Moisture Dynamics
    2. Soil & Nutritional Balancing (NPK & Micronutrients)
    3. Canopy Architecture & Cultural Sanitation
    4. Bio-Protective Defense & Preventative Regimen
    """
    dis_lower = disease.lower()
    crop_lower = crop.lower()
    is_healthy = "healthy" in dis_lower or "normal" in dis_lower

    if is_healthy:
        return {
            "headline": f"Optimizing Sustained Foliar Health & Resilience for {crop}",
            "irrigation": {
                "action": "Maintain Root-Zone Consistency",
                "details": (
                    "Apply water strictly at the soil base using drip lines or furrow irrigation. Avoid splashing water onto leaves, "
                    "especially during late afternoon hours when dew extension fosters latent fungal spores."
                ),
                "frequency": "Water deeply 2–3 times per week, maintaining 60–75% field moisture capacity.",
            },
            "nutrition": {
                "action": "Balanced Vegetative & Reproductive N-P-K",
                "details": (
                    "Maintain balanced N:P:K ratios (e.g. 10-10-10 or 12-8-16 with micronutrients). Supplement with Calcium (Ca) "
                    "and Boron (B) to reinforce cell wall pectin integrity and enhance natural pest immunity."
                ),
                "fertilizer_tip": "Avoid excessive sudden Nitrogen surges which cause rapid soft foliar growth vulnerable to pathogens.",
            },
            "sanitation": {
                "action": "Preventative Canopy Maintenance",
                "details": (
                    "Prune old senescing leaves touching the soil surface. Maintain row spacing to ensure wind circulation and fast drying "
                    "of natural morning dew."
                ),
                "tools": "Clean pruning implements with 70% isopropyl alcohol periodically.",
            },
            "bio_defense": {
                "action": "Prophylactic Plant Shielding",
                "details": (
                    "Apply prophylactic beneficial microbes (such as Bacillus amyloliquefaciens or Trichoderma harzianum) to root zone and "
                    "foliage to occupy microbial niches before pathogenic fungi can establish."
                ),
            },
        }

    # Specialized Pathogen-Specific Protocols
    if "blight" in dis_lower:  # Early Blight, Late Blight
        return {
            "headline": f"Agronomic Remediation Protocol: Foliar Blight Management in {crop}",
            "irrigation": {
                "action": "Strict Leaf-Dryness Mandate (Sub-Surface Irrigation Only)",
                "details": (
                    "Immediately terminate all overhead sprinkler watering. Blight zoospores and conidia require continuous free water on the "
                    "leaf surface for >4 hours to germinate. Transition 100% to drip irrigation or soaker hoses directly at the root zone."
                ),
                "frequency": "Irrigate only in the early morning so that any surface humidity evaporates rapidly under sunlight.",
            },
            "nutrition": {
                "action": "Suppress Nitrogen; Augment Potassium (K) & Calcium (Ca)",
                "details": (
                    "Temporarily suspend high-Nitrogen synthetic fertilizers. Excessive vegetative Nitrogen softens cell walls and fuels "
                    "pathogen haustorial penetration. Apply foliar Potassium Silicate (K₂SiO₃) and Calcium Nitrate to induce systemic acquired "
                    "resistance (SAR) and thicken the leaf cuticle."
                ),
                "fertilizer_tip": "Target NPK ratio of 1:2:3 during active blight outbreaks.",
            },
            "sanitation": {
                "action": "Lower Canopy Debridement & Bio-Sanitation",
                "details": (
                    "Prune and incinerate/bag all affected leaves up to 30 cm (12 inches) above the ground. Apply a 5 cm organic straw or "
                    "bark mulch barrier over bare soil to prevent rainwater splash from rebounding soil-borne Alternaria/Phytophthora spores."
                ),
                "tools": "Sterilize shears with a 10% sodium hypochlorite (bleach) or 70% alcohol dip between individual plants.",
            },
            "bio_defense": {
                "action": "Preventative Bio-Fungicidal Barrier",
                "details": (
                    "For Stage I/II: Apply liquid copper octanoate (copper soap) or Bacillus subtilis bioprotectant on both upper and lower leaf "
                    "surfaces every 7–10 days. Ensure full underside foliar coverage where humidity microclimates thrive."
                ),
            },
        }

    elif "rust" in dis_lower:  # Common Rust, Cedar Apple Rust
        return {
            "headline": f"Agronomic Remediation Protocol: Puccinia Rust Containment in {crop}",
            "irrigation": {
                "action": "Reduce Microclimate Canopy Humidity",
                "details": (
                    "Rust urediniospores proliferate rapidly in still, humid air (>80% RH) between 16°C and 25°C. Water at ground level and "
                    "avoid late evening watering that leaves the canopy damp overnight."
                ),
                "frequency": "Adjust irrigation to promote deep soil soaking followed by surface dry periods.",
            },
            "nutrition": {
                "action": "Silicon & Potassium Fortification",
                "details": (
                    "Foliar applications of soluble silicon (Si) help deposit a hard silica-cellulose matrix directly beneath the leaf cuticle, "
                    "physically blocking the entry of fungal germ tubes. Maintain optimal Phosphorus (P) for root stress recovery."
                ),
                "fertilizer_tip": "Avoid heavy ammonium-based fertilizers; balance with nitrate forms.",
            },
            "sanitation": {
                "action": "Foliar Density Thinning & Airflow Channeling",
                "details": (
                    "Prune dense internal foliage to open sunlight channels through the crop canopy. High UV exposure naturally attenuates "
                    "rust spore viability."
                ),
                "tools": "Remove and discard early sporulating leaves with visible pustules into sealed bags—do not leave in compost piles.",
            },
            "bio_defense": {
                "action": "Sulfur or Bio-Antagonist Applications",
                "details": (
                    "Apply wettable micronized sulfur or neem oil extract at the first sign of pustules. For organic systems, Trichoderma harzianum "
                    "has documented hyper-parasitic activity against rust fungal mycelium."
                ),
            },
        }

    elif "scab" in dis_lower or "black rot" in dis_lower:
        return {
            "headline": f"Agronomic Remediation Protocol: Scab / Black Rot Remediation in {crop}",
            "irrigation": {
                "action": "Soil Infiltration Control & Drainage",
                "details": (
                    "Improve soil aeration and avoid standing water pools near the crop root flare. Use targeted micro-drip emitters placed "
                    "outside the drip line to prevent fungal ascospore dispersal."
                ),
                "frequency": "Irrigate during sunrise hours to allow canopy wind-drying within 60 minutes.",
            },
            "nutrition": {
                "action": "Trace Element Balancing (Zinc, Manganese, Boron)",
                "details": (
                    "Deficiencies in Zinc (Zn) and Manganese (Mn) weaken leaf resistance against ascomycete fungi. Apply chelated micronutrient "
                    "foliar sprays in conjunction with modest Potash (K₂O)."
                ),
                "fertilizer_tip": "Maintain neutral soil pH (6.2–6.8) to maximize trace element bio-availability.",
            },
            "sanitation": {
                "action": "Complete Leaf Litter Removal & Floor Sanitation",
                "details": (
                    "Scab and black rot pathogens overwinter in fallen leaf litter and mummified fruit/cankers. Rake, shred, and compost with "
                    "high heat or safely burn all fallen botanical debris. Prune cankered twigs 10 cm behind visible lesions."
                ),
                "tools": "Apply pruning sealant or wound paste on cuts greater than 1 cm in diameter.",
            },
            "bio_defense": {
                "action": "Early Season Protective Fungicidal Cover",
                "details": (
                    "Apply potassium bicarbonate (Kaligreen/MilStop) or copper hydroxide protective barriers before forecasted rain events to "
                    "neutralize germinating ascospores on the foliage."
                ),
            },
        }

    elif "mildew" in dis_lower:  # Powdery Mildew
        return {
            "headline": f"Agronomic Remediation Protocol: Erysiphales Mildew Control in {crop}",
            "irrigation": {
                "action": "Canopy Moisture Counter-Strategy",
                "details": (
                    "Unlike downy mildew, powdery mildew spores actually fail to germinate in free standing water, but thrive in shaded high "
                    "humidity air. Water roots deeply while maintaining maximum sunlight penetration through the crown."
                ),
                "frequency": "Consistent deep watering to prevent drought stress, which predisposes plants to powdery mildew.",
            },
            "nutrition": {
                "action": "Alkaline Foliar Barrier & Potassium Bicarbonate",
                "details": (
                    "Apply potassium bicarbonate (KHCO₃) spray (3g/L) with a mild horticultural soap surfactant. This raises the foliar surface "
                    "pH to >8.0, which collapses powdery mildew mycelial cell walls and inhibits spore reproduction."
                ),
                "fertilizer_tip": "Keep Nitrogen moderate; lush rapid growth is highly susceptible.",
            },
            "sanitation": {
                "action": "Sunlight Exposure & Spacing Optimization",
                "details": (
                    "Thin overlapping vegetative shoots. Ensure minimum 50 cm inter-plant clearance. Direct sunlight destroys mildew conidia "
                    "within 2–4 hours of exposure."
                ),
                "tools": "Remove heavily coated leaves immediately to reduce the airborne spore load in the greenhouse or field.",
            },
            "bio_defense": {
                "action": "Ampelomyces quisqualis or Neem Oil Complex",
                "details": (
                    "Ampelomyces quisqualis is a natural fungal hyperparasite of powdery mildew. Alternatively, cold-pressed 0.5% neem oil "
                    "emulsion smothers active powdery mycelium on contact."
                ),
            },
        }

    elif "spot" in dis_lower or "scorch" in dis_lower:  # Bacterial Spot, Leaf Spot, Scorch
        return {
            "headline": f"Agronomic Remediation Protocol: Bacterial & Fungal Foliar Spot Control in {crop}",
            "irrigation": {
                "action": "Zero-Splash Moisture Protocol",
                "details": (
                    "Bacterial pathogens (Xanthomonas, Pseudomonas) spread primarily via mechanical rain and irrigation splashing. Avoid working "
                    "in or walking through the field when foliage is wet from dew or rainfall."
                ),
                "frequency": "Use buried drip lines (5–10 cm depth) or mulched surface drip to eliminate soil-to-leaf splashing entirely.",
            },
            "nutrition": {
                "action": "Nutrient Immuno-Priming (Copper & Boron)",
                "details": (
                    "Apply balanced micronutrients with trace Copper (Cu) and Boron (B) to stabilize membrane transport and prevent tissue "
                    "chlorosis. Avoid chemical burns from over-fertilizing with salty synthetic salts."
                ),
                "fertilizer_tip": "Top-dress with composted worm castings to introduce beneficial root rhizosphere bacteria.",
            },
            "sanitation": {
                "action": "Foliar Pruning & 3-Year Crop Rotation",
                "details": (
                    "Do not plant susceptible crops in the same soil bed consecutively. Rotate solanaceous and cucurbit crops with grains or "
                    "legumes on a 3-year cycle to exhaust residual soil inoculum."
                ),
                "tools": "Regularly sanitize stakes, trellises, and harvest crates with disinfectant.",
            },
            "bio_defense": {
                "action": "Copper-Bactericide Barrier & Biological Bacteriophages",
                "details": (
                    "Apply fixed copper (copper oxychloride / basic copper sulfate) tank-mixed with mancozeb for synergistic broad-spectrum "
                    "bacterial and fungal foliar suppression."
                ),
            },
        }

    # Universal Agronomic Fallback Protocol (covers custom crops like Rice, Wheat, Cotton, etc.)
    return {
        "headline": f"Agronomic Remediation Protocol: Integrated Pathology Management in {crop}",
        "irrigation": {
            "action": "Root-Zone Drip Delivery & Moisture Moderation",
            "details": (
                "Avoid overhead foliar wetting. Shift irrigation strictly to root-zone drip or furrow watering in early morning. Prolonged leaf "
                "wetness (>3 hours) is the universal biological prerequisite for foliar pathogenic invasion."
            ),
            "frequency": "Water deeply when top 3 cm of soil is dry; ensure field drainage channels are clear of standing water.",
        },
        "nutrition": {
            "action": "Calibrate N-P-K & Apply Tissue-Strengthening Minerals",
            "details": (
                "Reduce synthetic Nitrogen by 30% to stop rapid soft vegetative growth. Supplement with Potassium (K) to boost physiological "
                "drought and disease resistance, along with Calcium (Ca) for structural cell integrity."
            ),
            "fertilizer_tip": "Apply seaweed extract or humic acid foliar spray to relieve abiotic and pathogen-induced stress.",
        },
        "sanitation": {
            "action": "Infected Tissue Pruning & Field Hygiene",
            "details": (
                "Prune lower infected foliage and remove any fallen botanical debris from the soil. Space crop rows to promote laminar wind "
                "aeration throughout the field canopy."
            ),
            "tools": "Sanitize all pruning tools and implements with 70% alcohol between rows.",
        },
        "bio_defense": {
            "action": "Broad-Spectrum Biological Antagonists",
            "details": (
                "Spray broad-spectrum protective bioprotectants (Bacillus subtilis or copper octanoate) on both adaxial (upper) and abaxial (lower) "
                "leaf surfaces every 7 to 10 days until symptom progression halts."
            ),
        },
    }

