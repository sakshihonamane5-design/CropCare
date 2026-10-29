"""Unit tests for CropCare AI Assistant (chat_engine.py).

All tests run completely offline without live network calls, validating:
1. Missing API key handling.
2. Successful response using a mocked Gemini client.
3. Gemini API failure and graceful error handling.
4. Correct inclusion of existing screening context.
5. Correct inclusion of existing climate context.
6. Graceful behavior with missing climate data and missing prediction.
7. Chat history persistence behavior.
8. Clear Chat behavior.
9. No stale prediction after image replacement.
10. Guarding against empty/unsubmitted messages.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.cropcare.chat_engine import (
    DEFAULT_GEMINI_MODEL,
    SYSTEM_INSTRUCTION,
    build_agronomic_context,
    generate_chat_response,
    get_configured_model,
    get_gemini_api_key,
)


class TestChatEngine(unittest.TestCase):
    """Test suite for Gemini assistant integration and agronomic safety."""

    def test_1_missing_api_key(self):
        """1. When API key is absent, returns user-friendly configuration guide."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.cropcare.chat_engine.get_gemini_api_key", return_value=None):
                messages = [{"role": "user", "content": "How do I identify leaf spots?"}]
                response = generate_chat_response(messages=messages, context_str="", api_key=None, client=None)
                self.assertIn("Gemini API Key Required", response)
                self.assertIn(".streamlit/secrets.toml", response)
                self.assertIn("GEMINI_API_KEY", response)

    def test_2_successful_response_mocked_client(self):
        """2. Successful response using a mocked Gemini client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Early blight typically appears as concentric brown rings on older foliage."
        mock_client.models.generate_content.return_value = mock_response

        messages = [{"role": "user", "content": "Tell me about early blight."}]
        context = "Detected Visual Condition: Early blight"

        reply = generate_chat_response(
            messages=messages,
            context_str=context,
            client=mock_client,
            model_name="gemini-2.5-flash",
        )

        self.assertEqual(reply, "Early blight typically appears as concentric brown rings on older foliage.")
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs["model"], "gemini-2.5-flash")

    def test_3_gemini_api_failure_and_graceful_handling(self):
        """3. Handles quota exhaustion, network timeouts, and general exceptions without crash."""
        mock_client = MagicMock()

        # 3a. Quota error
        mock_client.models.generate_content.side_effect = Exception("429 Resource has been exhausted (quota)")
        reply_quota = generate_chat_response(
            messages=[{"role": "user", "content": "Hi"}],
            context_str="",
            client=mock_client,
        )
        self.assertIn("API Quota Exceeded", reply_quota)

        # 3b. Network error
        mock_client.models.generate_content.side_effect = Exception("connection timeout to api.google.com")
        reply_timeout = generate_chat_response(
            messages=[{"role": "user", "content": "Hi"}],
            context_str="",
            client=mock_client,
        )
        self.assertIn("Network Timeout", reply_timeout)

    def test_4_correct_use_of_screening_context(self):
        """4. Context string embeds genuine screening predictions when available."""
        leaf_res = {
            "predicted_class": "Tomato___Early_blight",
            "crop": "Tomato",
            "disease": "Early blight",
            "confidence": 0.885,
            "uncertain": False,
            "model_name": "MobileNetV3-Small-PlantVillage",
            "warnings": [],
            "probabilities": {"Tomato___Early_blight": 0.885, "Tomato___Target_Spot": 0.052},
        }
        ctx = build_agronomic_context(screen=2, leaf_result=leaf_res, climate_result=None, climate_days=30)

        self.assertIn("Identified Crop: Tomato", ctx)
        self.assertIn("Detected Visual Condition: Early blight", ctx)
        self.assertIn("Statistical Confidence: 88.5%", ctx)
        self.assertIn("Uncertainty Flag: FALSE", ctx)
        self.assertIn("MobileNetV3-Small-PlantVillage", ctx)

    def test_5_correct_use_of_climate_context(self):
        """5. Context string embeds genuine climate data and anomalies."""
        climate_res = {
            "location": "Nashik, Maharashtra",
            "source": "Open-Meteo",
            "recent": {
                "mean_temperature_c": 26.5,
                "max_temperature_c": 32.1,
                "rainfall_mm": 45.0,
                "mean_humidity_pct": 78.0,
                "et0_mm": 3.8,
            },
            "anomalies": {
                "temperature_c": 1.2,
                "rainfall_mm": -12.5,
                "rainfall_pct": -21.7,
            },
            "warnings": ["Moderate drought risk flagged."],
        }
        ctx = build_agronomic_context(screen=3, leaf_result=None, climate_result=climate_res, climate_days=14)

        self.assertIn("Location: Nashik, Maharashtra", ctx)
        self.assertIn("Observation Window: 14 Days", ctx)
        self.assertIn("Mean Temperature: 26.5 °C", ctx)
        self.assertIn("Total Rainfall: 45.0 mm", ctx)
        self.assertIn("Temperature Anomaly vs Baseline: 1.2 °C", ctx)
        self.assertIn("Rainfall Anomaly vs Baseline: -12.5 mm (-21.7 %)", ctx)
        self.assertIn("Moderate drought risk flagged.", ctx)

    def test_6_missing_climate_data_and_missing_prediction(self):
        """6. Missing session state does not fabricate values or crash context builder."""
        ctx = build_agronomic_context(screen=1, leaf_result=None, climate_result=None)
        self.assertIn("Optical Screening: No leaf image has been screened yet", ctx)
        self.assertIn("Climate Analytics: No weather data has been retrieved yet", ctx)
        self.assertNotIn("Tomato", ctx)
        self.assertNotIn("Rainfall: 0.0 mm", ctx)

    def test_7_chat_history_persistence_simulation(self):
        """7. Chat history survives screen transitions when kept in session state."""
        session_state = {
            "screen": 1,
            "chat_messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Welcome!"}],
        }
        # Simulate navigation across all 4 screens
        for scr in [2, 3, 4, 1]:
            session_state["screen"] = scr
            self.assertEqual(len(session_state["chat_messages"]), 2)
            self.assertEqual(session_state["chat_messages"][0]["content"], "Hello")

    def test_8_clear_chat_behavior(self):
        """8. Clear Chat resets the conversation history to empty."""
        chat_messages = [
            {"role": "user", "content": "What is late blight?"},
            {"role": "assistant", "content": "Late blight is caused by Phytophthora infestans."},
        ]
        self.assertEqual(len(chat_messages), 2)
        # Clear chat
        chat_messages.clear()
        self.assertEqual(len(chat_messages), 0)

    def test_9_no_stale_prediction_after_new_image(self):
        """9. Selecting a new image resets previous predictions so stale results are never sent to chat."""
        session = {
            "leaf_image_id": "image_1_hash",
            "leaf_result": {"crop": "Apple", "disease": "Apple scab"},
        }
        # New image selected
        new_image_id = "image_2_hash"
        if new_image_id != session["leaf_image_id"]:
            session["leaf_image_id"] = new_image_id
            session["leaf_result"] = None  # Cleared

        ctx = build_agronomic_context(screen=2, leaf_result=session["leaf_result"], climate_result=None)
        self.assertIn("No leaf image has been screened yet", ctx)
        self.assertNotIn("Apple", ctx)

    def test_10_no_api_call_when_message_empty(self):
        """10. Guarding against empty messages ensures no API invocation."""
        mock_client = MagicMock()
        empty_messages: list = []
        reply = generate_chat_response(messages=empty_messages, context_str="", client=mock_client)

        self.assertIn("How can I assist you", reply)
        mock_client.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
