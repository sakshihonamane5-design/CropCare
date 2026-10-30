"""Unit tests for Economic Engine and Recovery Roadmap."""

import unittest
from src.cropcare.economic_engine import (
    estimate_economic_impact,
    CROP_ECONOMIC_BENCHMARKS,
    DISEASE_LOSS_PROFILES,
)
from src.cropcare.advisory_engine import (
    generate_recovery_roadmap,
    build_whatsapp_share_url,
)


class TestEconomicEngine(unittest.TestCase):
    """Test suite for economic loss estimation and recovery roadmaps."""

    def test_economic_loss_calculation(self):
        """Test economic loss calculations for high-risk disease (Tomato Late Blight)."""
        res = estimate_economic_impact("Tomato", "Late_blight", confidence=0.92, farm_acres=2.0)
        self.assertEqual(res["crop"], "Tomato")
        self.assertFalse(res["is_healthy"])
        self.assertEqual(res["urgency_badge"], "Critical")
        self.assertGreater(res["loss_value_max_inr"], res["loss_value_min_inr"])
        self.assertGreater(res["treatment_cost_inr"], 0)
        self.assertIn("x", res["roi_multiplier"])

    def test_economic_loss_healthy(self):
        """Test economic loss calculation for healthy crop."""
        res = estimate_economic_impact("Potato", "healthy", confidence=0.98, farm_acres=1.0)
        self.assertTrue(res["is_healthy"])
        self.assertEqual(res["loss_pct_range"], "0%")
        self.assertEqual(res["urgency_badge"], "Negligible")
        self.assertEqual(res["loss_value_max_inr"], 0)

    def test_recovery_roadmap_generation(self):
        """Test generation of 7-day chronological roadmap."""
        roadmap = generate_recovery_roadmap("Tomato", "Early_blight", spray_status="OPTIMAL")
        self.assertEqual(len(roadmap), 4)
        self.assertEqual(roadmap[0]["day"], "Day 1")
        self.assertIn("Sanitation", roadmap[0]["stage"])

    def test_whatsapp_share_url(self):
        """Test WhatsApp share URL builder."""
        url = build_whatsapp_share_url("Tomato", "Early_blight", 94.2, "Nashik", "High")
        self.assertIn("https://api.whatsapp.com/send?text=", url)
        self.assertIn("Early", url)
        self.assertIn("Nashik", url)


if __name__ == "__main__":
    unittest.main()

