"""Unit tests for Location Engine (location_engine.py)."""

import unittest
from unittest.mock import MagicMock, patch
from src.cropcare.location_engine import (
    AGRI_DISTRICTS,
    TOP_PRESET_LOCATIONS,
    search_location_offline,
    search_location_online,
    get_states,
    get_districts_for_state,
)


class TestLocationEngine(unittest.TestCase):
    """Test suite for agricultural location lookup and geocoding."""

    def test_premapped_states_and_districts(self):
        """Verify pre-mapped states and districts exist and have valid coordinates."""
        states = get_states()
        self.assertIn("Maharashtra", states)
        self.assertIn("Punjab", states)

        dists = get_districts_for_state("Maharashtra")
        self.assertIn("Nashik", dists)
        lat, lon, crops = dists["Nashik"]
        self.assertAlmostEqual(lat, 19.9975, places=2)
        self.assertAlmostEqual(lon, 73.7898, places=2)

    def test_top_preset_locations(self):
        """Verify top preset locations are non-empty and well-formed."""
        self.assertGreaterEqual(len(TOP_PRESET_LOCATIONS), 8)
        for name, coords in TOP_PRESET_LOCATIONS.items():
            self.assertEqual(len(coords), 2)
            self.assertTrue(-90 <= coords[0] <= 90)
            self.assertTrue(-180 <= coords[1] <= 180)

    def test_search_location_offline(self):
        """Test fuzzy offline search by district or state name."""
        results = search_location_offline("Nashik")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "Nashik")
        self.assertEqual(results[0]["state"], "Maharashtra")

        results_crops = search_location_offline("Orange")
        self.assertGreater(len(results_crops), 0)

    def test_search_location_online_with_mock(self):
        """Test online geocoding with mock response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"results": [{"name": "Baramati", "admin1": "Maharashtra", "country": "India", "latitude": 18.15, "longitude": 74.58}]}'
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_response
        mock_context.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_context):
            res = search_location_online("Baramati")
            self.assertGreater(len(res), 0)
            self.assertEqual(res[0]["name"], "Baramati")
            self.assertEqual(res[0]["latitude"], 18.15)


if __name__ == "__main__":
    unittest.main()

