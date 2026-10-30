"""Unit tests for the Multilingual & Audio Voice Engine."""

import unittest
from src.cropcare.multilingual import (
    SUPPORTED_LANGUAGES,
    generate_audio_speech_html,
    t,
    translate_advisory_text,
)


class TestMultilingual(unittest.TestCase):
    """Test language retrieval, translations, and HTML5 speech synthesis markup."""

    def test_supported_languages(self):
        self.assertIn("en", SUPPORTED_LANGUAGES)
        self.assertIn("hi", SUPPORTED_LANGUAGES)
        self.assertIn("mr", SUPPORTED_LANGUAGES)
        self.assertEqual(SUPPORTED_LANGUAGES["hi"]["name"], "Hindi")

    def test_translation_retrieval(self):
        self.assertIn("Read the leaf", t("hero_title", "en"))
        self.assertIn("पत्ती को पहचानें", t("hero_title", "hi"))
        self.assertIn("पाने ओळखा", t("hero_title", "mr"))
        # Fallback test
        self.assertEqual(t("hero_title", "unknown_lang"), t("hero_title", "en"))

    def test_audio_speech_html_generation(self):
        html_code = generate_audio_speech_html("Tomato Early Blight detected.", lang="hi")
        self.assertIn("speechSynthesis", html_code)
        self.assertIn("hi-IN", html_code)
        self.assertIn("Voice Agronomist", html_code)

    def test_translate_advisory_text(self):
        advisory_en = translate_advisory_text(
            crop="Tomato",
            disease="Early Blight",
            stage_name="Stage II",
            damage_pct=24.0,
            irrigation_action="Drip only",
            nutrition_action="Reduce Nitrogen",
            sanitation_action="Prune lower leaves",
            bio_action="Copper spray",
            lang="en",
        )
        self.assertIn("Tomato", advisory_en)
        self.assertIn("Drip only", advisory_en)

        advisory_hi = translate_advisory_text(
            crop="टमाटर",
            disease="अगेती झुलसा",
            stage_name="चरण २",
            damage_pct=24.0,
            irrigation_action="ड्रिप सिंचाई करें",
            nutrition_action="नाइट्रोजन घटाएं",
            sanitation_action="निचली पत्तियां हटाएं",
            bio_action="कॉपर छिड़कें",
            lang="hi",
        )
        self.assertIn("क्रॉपकेयर फसल स्वास्थ्य रिपोर्ट", advisory_hi)
        self.assertIn("ड्रिप सिंचाई करें", advisory_hi)


if __name__ == "__main__":
    unittest.main()
