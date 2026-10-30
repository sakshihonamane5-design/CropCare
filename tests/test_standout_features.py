"""Unit tests for CropCare Standout Features (Offline).

Tests:
1. Alignment Engine (Disease-Climate Biometeorology concurrence).
2. Grad-CAM Explainability (Heatmap generation and contrast fallback).
3. Advisory Engine (Weather-Smart IPM and spray restrictions).
4. Report Engine (HTML print-to-PDF card generation).
5. Vernacular Module (Hindi/Marathi translation and audio scripts).
"""

from __future__ import annotations

import unittest
from PIL import Image
import numpy as np

from src.cropcare.alignment_engine import evaluate_environmental_alignment
from src.cropcare.explainability import generate_gradcam, _apply_colormap
from src.cropcare.advisory_engine import generate_actionable_advisory
from src.cropcare.report_engine import generate_field_report_html
from src.cropcare.vernacular import translate_diagnosis, generate_audio_script


class TestStandoutFeaturesOffline(unittest.TestCase):
    def setUp(self):
        # Create synthetic test leaf image (100x100 RGB)
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add synthetic dark spot
        arr[40:60, 40:60] = [30, 80, 20]
        self.test_img = Image.fromarray(arr)

        self.mock_leaf_result = {
            "predicted_class": "Tomato___Early_blight",
            "crop": "Tomato",
            "disease": "Early blight",
            "confidence": 0.94,
            "uncertain": False,
            "model_name": "MobileNetV3-Small-Test",
        }

        self.mock_climate_result = {
            "location": "Pune (18.5204, 73.8567)",
            "period_days": 30,
            "recent": {
                "mean_temperature_c": 26.5,
                "max_temperature_c": 32.0,
                "rainfall_mm": 18.0,
                "mean_humidity_pct": 78.0,
                "et0_mm": 110.0,
            },
            "anomalies": {
                "temperature_c": 2.5,
                "rainfall_mm": -15.0,
                "rainfall_pct": -45.0,
            },
        }

    def test_alignment_engine_high_concurrence(self):
        """Verify alignment engine calculates positive concurrence for matching pathogen envelope."""
        res = evaluate_environmental_alignment(self.mock_leaf_result, self.mock_climate_result)
        self.assertIn("score", res)
        self.assertIn("badge", res)
        self.assertIn("explanation", res)
        self.assertGreaterEqual(res["score"], 50)
        self.assertFalse(res["abiotic_stress_flag"])

    def test_alignment_engine_drought_stress_divergence(self):
        """Verify severe drought with rain deficit flags abiotic divergence."""
        drought_climate = {
            "recent": {
                "mean_temperature_c": 36.0,
                "max_temperature_c": 42.0,
                "rainfall_mm": 0.0,
                "mean_humidity_pct": 30.0,
            },
            "anomalies": {
                "temperature_c": 5.0,
                "rainfall_mm": -80.0,
                "rainfall_pct": -95.0,
            },
        }
        res = evaluate_environmental_alignment(self.mock_leaf_result, drought_climate)
        self.assertEqual(res["status"], "DIVERGENT_STRESS")
        self.assertTrue(res["abiotic_stress_flag"])
        self.assertIn("drought", res["explanation"].lower())

    def test_gradcam_overlay_generation(self):
        """Verify Grad-CAM produces a valid blended PIL image."""
        res = generate_gradcam(self.test_img)
        self.assertIn("overlay_image", res)
        self.assertIn("heatmap_image", res)
        self.assertIsInstance(res["overlay_image"], Image.Image)
        self.assertEqual(res["overlay_image"].size, self.test_img.size)
        self.assertGreater(res["peak_intensity"], 0.0)

    def test_advisory_engine_rain_hold(self):
        """Verify advisory engine halts chemical spraying under heavy rain anomaly."""
        rainy_climate = {
            "recent": {"rainfall_mm": 65.0, "mean_temperature_c": 22.0, "max_temperature_c": 26.0},
            "anomalies": {"rainfall_mm": 40.0},
        }
        adv = generate_actionable_advisory(self.mock_leaf_result, rainy_climate)
        self.assertEqual(adv["spray_window"]["status"], "HOLD_RAIN")
        self.assertIn("Do Not Spray", adv["spray_window"]["badge"])
        self.assertGreater(len(adv["cultural"]), 0)
        self.assertGreater(len(adv["biological"]), 0)

    def test_report_engine_html_generation(self):
        """Verify HTML field report is generated with embedded styling and base64 images."""
        align = evaluate_environmental_alignment(self.mock_leaf_result, self.mock_climate_result)
        adv = generate_actionable_advisory(self.mock_leaf_result, self.mock_climate_result)
        cam = generate_gradcam(self.test_img)

        html = generate_field_report_html(
            leaf_result=self.mock_leaf_result,
            climate_result=self.mock_climate_result,
            alignment_result=align,
            advisory_result=adv,
            leaf_image=self.test_img,
            cam_image=cam["overlay_image"],
            farmer_name="Shri Ramesh Patil",
            farm_location="Pune, Maharashtra",
        )
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Shri Ramesh Patil", html)
        self.assertIn("Early blight", html)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn("window.print()", html)

    def test_vernacular_translation_and_audio(self):
        """Verify Hindi and Marathi translations and audio script creation."""
        c_hi, d_hi = translate_diagnosis("Tomato", "early_blight", lang="hi")
        self.assertIn("αñƒαñ«αñ╛αñƒαñ░", c_hi)
        self.assertIn("αñ¥αÑüαñ▓αñ╕αñ╛", d_hi)

        c_mr, d_mr = translate_diagnosis("Potato", "late_blight", lang="mr")
        self.assertIn("αñ¼αñƒαñ╛αñƒαñ╛", c_mr)
        self.assertIn("αñòαñ░αñ¬αñ╛", d_mr)

        script_hi = generate_audio_script("Tomato", "Early blight", 0.94, "HIGH_ALIGNMENT", "Spray Window Open", lang="hi")
        self.assertIn("αñƒαñ«αñ╛αñƒαñ░", script_hi)
        self.assertIn("αñ¬αÑìαñ░αññαñ┐αñ╢αññ", script_hi)


if __name__ == "__main__":
    unittest.main()
