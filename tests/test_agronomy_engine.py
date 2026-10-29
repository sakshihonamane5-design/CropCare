"""Unit tests for the Agronomy Intelligence Engine."""

import unittest
from PIL import Image
from src.cropcare.agronomy_engine import (
    calculate_foliar_damage_index,
    classify_disease_severity,
    get_nurturing_recommendations,
)


class TestAgronomyEngine(unittest.TestCase):
    """Test foliar damage calculation, severity classification, and nurturing plans."""

    def setUp(self):
        # Create a synthetic healthy leaf (green)
        self.healthy_leaf = Image.new("RGB", (100, 100), (45, 160, 40))
        # Create a synthetic diseased leaf (mostly brown/necrotic)
        self.diseased_leaf = Image.new("RGB", (100, 100), (90, 60, 30))

    def test_damage_index_healthy(self):
        dmg = calculate_foliar_damage_index(self.healthy_leaf)
        self.assertLess(dmg, 10.0)

    def test_damage_index_diseased(self):
        dmg = calculate_foliar_damage_index(self.diseased_leaf)
        self.assertGreater(dmg, 50.0)

    def test_healthy_classification(self):
        res = classify_disease_severity(self.healthy_leaf, "Tomato healthy")
        self.assertEqual(res["stage_id"], 0)
        self.assertEqual(res["severity_level"], "Optimal / Pristine Vigor")
        self.assertIn("Asymptomatic", res["stage_name"])

    def test_diseased_classification_stages(self):
        res_mild = classify_disease_severity(self.healthy_leaf, "Tomato early blight")
        self.assertEqual(res_mild["stage_id"], 1)

        res_severe = classify_disease_severity(self.diseased_leaf, "Tomato early blight")
        self.assertEqual(res_severe["stage_id"], 3)
        self.assertIn("Critical", res_severe["severity_level"])

    def test_nurturing_recommendations_structure(self):
        recs = get_nurturing_recommendations("Tomato", "Early blight", severity_stage=2)
        self.assertIn("headline", recs)
        self.assertIn("irrigation", recs)
        self.assertIn("nutrition", recs)
        self.assertIn("sanitation", recs)
        self.assertIn("bio_defense", recs)

        # Check specific agronomic content
        self.assertIn("Nitrogen", recs["nutrition"]["action"] + recs["nutrition"]["details"])
        self.assertIn("drip", recs["irrigation"]["details"].lower())

    def test_healthy_nurturing_recommendations(self):
        recs = get_nurturing_recommendations("Apple", "Apple healthy", severity_stage=0)
        self.assertIn("Sustained Foliar Health", recs["headline"])
        self.assertIn("drip", recs["irrigation"]["details"].lower())


if __name__ == "__main__":
    unittest.main()
