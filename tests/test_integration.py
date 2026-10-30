"""Comprehensive End-to-End Integration Tests for CropCare.

Tests complete cross-module pipeline:
1. Optical leaf screening (MobileNetV3)
2. Explainable AI Heatmap (Grad-CAM)
3. Climate & Environmental Intelligence (Open-Meteo mock)
4. Biometeorological Concurrence Analysis
5. Weather-Smart IPM Advisory
6. One-Click Field Health Card Generation
"""

from __future__ import annotations

import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from src.cropcare.image_engine import predict_image
from src.cropcare.explainability import generate_gradcam
from src.cropcare.climate_engine import get_environment
from src.cropcare.alignment_engine import evaluate_environmental_alignment
from src.cropcare.advisory_engine import generate_actionable_advisory
from src.cropcare.report_engine import generate_field_report_html
from src.cropcare.vernacular import translate_diagnosis, generate_audio_script


class TestEndToEndCropCarePipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic valid green leaf with brown lesion
        img_arr = np.full((224, 224, 3), 100, dtype=np.uint8)
        img_arr[:, :, 1] = 160  # Greenish
        img_arr[80:140, 80:140] = [120, 50, 20]  # Necrotic spot
        self.leaf_img = Image.fromarray(img_arr)

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_full_pipeline_synergy(self, mock_get):
        """Verify seamless execution from uploaded photo to PDF field report."""
        recent_daily = {
            "time": ["2026-10-01", "2026-10-02", "2026-10-03"],
            "temperature_2m_mean": [27.0, 28.0, 29.0],
            "temperature_2m_max": [33.0, 34.0, 35.0],
            "precipitation_sum": [10.0, 15.0, 15.0],
            "relative_humidity_2m_mean": [76.0, 78.0, 80.0],
            "et0_fao_evapotranspiration": [3.5, 4.0, 4.5],
        }
        baseline_daily = {
            "time": ["2023-10-01", "2023-10-02", "2023-10-03"],
            "temperature_2m_mean": [24.0, 25.0, 26.0],
            "temperature_2m_max": [29.0, 30.0, 31.0],
            "precipitation_sum": [5.0, 10.0, 5.0],
            "relative_humidity_2m_mean": [60.0, 62.0, 64.0],
            "et0_fao_evapotranspiration": [3.0, 3.2, 3.4],
        }

        def side_effect(url, params, timeout):
            if "forecast" in url:
                return {"daily": recent_daily}, None
            if "archive" in url:
                return {"daily": baseline_daily}, None
            return None, "Unexpected endpoint"

        mock_get.side_effect = side_effect

        # 2. Run Optical Inference
        leaf_res = predict_image(self.leaf_img)
        self.assertIsNotNone(leaf_res)
        self.assertIn("confidence", leaf_res)
        self.assertIn("crop", leaf_res)

        # 3. Run Grad-CAM Explainability
        cam_res = generate_gradcam(self.leaf_img)
        self.assertTrue(cam_res["success"])
        self.assertIn("overlay_image", cam_res)

        # 4. Run Climate Engine
        climate_res = get_environment(location=(18.5204, 73.8567), days=3, baseline_years=[2023])
        self.assertEqual(climate_res["anomalies"]["temperature_c"], 3.0)
        self.assertEqual(climate_res["anomalies"]["rainfall_mm"], 20.0)

        # 5. Evaluate Biometeorological Concurrence
        align_res = evaluate_environmental_alignment(leaf_res, climate_res)
        self.assertIn(align_res["status"], ["HIGH_ALIGNMENT", "MODERATE_ALIGNMENT", "DIVERGENT_STRESS", "OPTIMAL_GROWTH"])
        self.assertGreaterEqual(align_res["score"], 0)

        # 6. Generate Weather-Smart IPM Advisory
        adv_res = generate_actionable_advisory(leaf_res, climate_res)
        self.assertIn("spray_window", adv_res)
        self.assertGreater(len(adv_res["cultural"]), 0)

        # 7. Generate Downloadable Field Health Card
        html_report = generate_field_report_html(
            leaf_result=leaf_res,
            climate_result=climate_res,
            alignment_result=align_res,
            advisory_result=adv_res,
            leaf_image=self.leaf_img,
            cam_image=cam_res["overlay_image"],
            farmer_name="Shri Santosh Jadhav",
            farm_location="Pune, Maharashtra",
        )
        self.assertIn("CropCare™ Intelligence", html_report)
        self.assertIn("Shri Santosh Jadhav", html_report)
        self.assertIn("window.print()", html_report)

        # 8. Vernacular Audio Verification
        audio_hi = generate_audio_script(leaf_res["crop"] or "Crop", leaf_res["disease"] or "Condition", leaf_res["confidence"], "HIGH", "Ready", lang="hi")
        self.assertIsInstance(audio_hi, str)
        self.assertGreater(len(audio_hi), 10)


if __name__ == "__main__":
    unittest.main()
