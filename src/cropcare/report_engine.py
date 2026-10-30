"""CropCare ΓÇö Field Diagnostic Report Engine.

Generates self-contained, print-ready agronomic diagnostic reports in HTML/PDF format
and Markdown summaries for field extension workers and farmers.
"""

from __future__ import annotations

import base64
import datetime
import io
from typing import Any, Dict, Optional

from PIL import Image


def _pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def generate_field_report_html(
    leaf_result: Optional[Dict[str, Any]],
    climate_result: Optional[Dict[str, Any]],
    alignment_result: Optional[Dict[str, Any]],
    advisory_result: Optional[Dict[str, Any]],
    leaf_image: Optional[Image.Image] = None,
    cam_image: Optional[Image.Image] = None,
    farmer_name: str = "Lead Farmer",
    farm_location: str = "Agricultural Plot",
) -> str:
    """Generate a self-contained, styled HTML report with print-to-PDF formatting."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    leaf_img_b64 = _pil_to_base64(leaf_image) if leaf_image else ""
    cam_img_b64 = _pil_to_base64(cam_image) if cam_image else ""

    crop = leaf_result.get("crop", "Unknown") if leaf_result else "N/A"
    disease = leaf_result.get("disease", "Unknown") if leaf_result else "N/A"
    conf = (leaf_result.get("confidence", 0.0) * 100) if leaf_result else 0.0
    model = leaf_result.get("model_name", "MobileNetV3-Small-PlantVillage") if leaf_result else "N/A"

    recent = climate_result.get("recent", {}) if climate_result else {}
    anom = climate_result.get("anomalies", {}) if climate_result else {}
    loc = climate_result.get("location", farm_location) if climate_result else farm_location

    t_mean = f"{recent.get('mean_temperature_c'):.1f} ┬░C" if recent.get("mean_temperature_c") is not None else "N/A"
    rain = f"{recent.get('rainfall_mm'):.1f} mm" if recent.get("rainfall_mm") is not None else "N/A"
    t_anom = f"{anom.get('temperature_c'):+.1f} ┬░C" if anom.get("temperature_c") is not None else "N/A"
    p_anom = f"{anom.get('rainfall_mm'):+.1f} mm" if anom.get("rainfall_mm") is not None else "N/A"

    align_badge = alignment_result.get("badge", "N/A") if alignment_result else "N/A"
    align_score = alignment_result.get("score", 0) if alignment_result else 0
    align_expl = alignment_result.get("explanation", "") if alignment_result else ""

    spray_badge = advisory_result.get("spray_window", {}).get("badge", "N/A") if advisory_result else "N/A"
    spray_advice = advisory_result.get("spray_window", {}).get("advice", "") if advisory_result else ""

    cult_items = "".join(f"<li>{x}</li>" for x in (advisory_result.get("cultural", []) if advisory_result else []))
    bio_items = "".join(f"<li>{x}</li>" for x in (advisory_result.get("biological", []) if advisory_result else []))
    chem_items = "".join(f"<li>{x}</li>" for x in (advisory_result.get("chemical", []) if advisory_result else []))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CropCare Agronomic Field Health Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
  body {{
    font-family: 'Plus Jakarta Sans', Arial, sans-serif;
    color: #11231B;
    background: #F9FAF7;
    margin: 0;
    padding: 24px;
  }}
  .card {{
    max-width: 900px;
    margin: 0 auto;
    background: #FFFFFF;
    border: 1px solid #D6E0D9;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #E8ECE6;
    padding-bottom: 20px;
    margin-bottom: 24px;
  }}
  .logo {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #123D2A;
  }}
  .meta {{
    text-align: right;
    font-size: 12px;
    color: #526E60;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }}
  .panel {{
    background: #F4F7F2;
    border: 1px solid #DFE6DC;
    border-radius: 12px;
    padding: 16px;
  }}
  .panel-title {{
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 700;
    color: #526E60;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }}
  .panel-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #123D2A;
  }}
  .img-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }}
  .img-box {{
    background: #EEF2EB;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
  }}
  .img-box img {{
    max-width: 100%;
    height: 220px;
    object-fit: cover;
    border-radius: 8px;
  }}
  .img-caption {{
    font-size: 12px;
    font-weight: 600;
    color: #1B543A;
    margin-top: 8px;
  }}
  .badge {{
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    background: #EBF8D3;
    color: #123D2A;
    margin-bottom: 8px;
  }}
  .disclaimer {{
    font-size: 11px;
    color: #6E8578;
    background: #FAFBF9;
    border-left: 3px solid #1B543A;
    padding: 10px 14px;
    margin-top: 24px;
  }}
  @media print {{
    body {{ background: #FFFFFF; padding: 0; }}
    .card {{ border: none; box-shadow: none; padding: 0; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div>
      <div class="logo">≡ƒî▒ CropCareΓäó Intelligence</div>
      <div style="font-size: 13px; color: #526E60;">Climate-Aware Agronomic Screening & Advisory</div>
    </div>
    <div class="meta">
      <strong>Generated:</strong> {timestamp}<br>
      <strong>Location:</strong> {loc}<br>
      <strong>Farmer / Plot:</strong> {farmer_name}
    </div>
  </div>

  <div class="img-grid">
    <div class="img-box">
      {"<img src='" + leaf_img_b64 + "' alt='Leaf'>" if leaf_img_b64 else "<div style='height: 220px; display:flex; align-items:center; justify-content:center;'>No Image</div>"}
      <div class="img-caption">Original Leaf Photograph</div>
    </div>
    <div class="img-box">
      {"<img src='" + cam_img_b64 + "' alt='Grad-CAM Heatmap'>" if cam_img_b64 else "<div style='height: 220px; display:flex; align-items:center; justify-content:center;'>Grad-CAM Unavailable</div>"}
      <div class="img-caption">Explainable AI (Grad-CAM Saliency)</div>
    </div>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="panel-title">Diagnostic Screening</div>
      <div class="panel-val">{crop}</div>
      <div style="font-size: 14px; font-weight: 600; color: #1B543A; margin-top: 4px;">{disease}</div>
      <div style="font-size: 12px; color: #526E60; margin-top: 8px;">Confidence: <strong>{conf:.1f}%</strong> | Model: {model}</div>
    </div>

    <div class="panel">
      <div class="panel-title">Hyper-Local Climate Profile</div>
      <div style="font-size: 14px; font-weight: 600; color: #123D2A;">Mean Temp: {t_mean} (Anom: {t_anom})</div>
      <div style="font-size: 14px; font-weight: 600; color: #123D2A; margin-top: 4px;">Rainfall: {rain} (Anom: {p_anom})</div>
      <div style="font-size: 12px; color: #526E60; margin-top: 8px;">Source: Open-Meteo Reanalysis Telemetry</div>
    </div>
  </div>

  <div class="panel" style="margin-bottom: 20px;">
    <div class="panel-title">Biometeorological Concurrence Analysis</div>
    <div class="badge">{align_badge} (Alignment Index: {align_score}/100)</div>
    <div style="font-size: 13px; line-height: 1.5; color: #123D2A;">{align_expl}</div>
  </div>

  <div class="panel">
    <div class="panel-title">Actionable Integrated Pest Management (IPM) & Spray Window</div>
    <div style="font-weight: 700; color: #1B543A; margin-bottom: 4px;">{spray_badge}</div>
    <div style="font-size: 13px; color: #334D3F; margin-bottom: 12px;">{spray_advice}</div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 12px;">
      <div>
        <strong>≡ƒî╛ Cultural & Sanitation Tactics:</strong>
        <ul style="padding-left: 18px; margin: 4px 0;">{cult_items}</ul>
      </div>
      <div>
        <strong>≡ƒº¬ Targeted Interventions (Subject to Verification):</strong>
        <ul style="padding-left: 18px; margin: 4px 0;">{bio_items}{chem_items}</ul>
      </div>
    </div>
  </div>

  <div class="disclaimer">
    <strong>Scientific & Diagnostic Safeguards:</strong> This screening report is produced by statistical ML models trained on controlled benchmark datasets (PlantVillage) and reanalysis weather grids. Visual leaf lesions can have abiotic origins (drought, nutritional deficit). Do not execute chemical treatments without physical verification by an agricultural extension specialist.
  </div>

  <div class="no-print" style="margin-top: 24px; text-align: center;">
    <button onclick="window.print()" style="background: #123D2A; color: #FFFFFF; font-weight: 700; border: none; padding: 12px 28px; border-radius: 8px; cursor: pointer; font-size: 14px;">
      ≡ƒû¿∩╕Å Print or Save as PDF
    </button>
  </div>
</div>
</body>
</html>"""
    return html
