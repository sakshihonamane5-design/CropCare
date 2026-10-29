"""Unit tests for CropCare Climate & Environmental Intelligence Module.

All tests are designed to execute entirely offline with mock data, verifying
parsing, aggregation, anomaly calculations, error handling, contract stability,
and CSV exports.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.cropcare.climate_engine import (
    DEFAULT_PERIOD_DAYS,
    calculate_anomalies,
    compute_period_stats,
    export_climate_csv,
    fetch_historical_baseline,
    get_environment,
    parse_daily_series,
    resolve_location,
)


class TestLocationResolution(unittest.TestCase):
    """Tests for location inputs and resolution logic."""

    def test_valid_tuple_coords(self):
        lat, lon, label, warnings = resolve_location((18.5204, 73.8567))
        self.assertEqual(lat, 18.5204)
        self.assertEqual(lon, 73.8567)
        self.assertIn("18.5204", label)
        self.assertEqual(warnings, [])

    def test_valid_dict_coords(self):
        lat, lon, label, warnings = resolve_location({"lat": -1.286389, "lon": 36.817223, "name": "Nairobi Farm"})
        self.assertEqual(lat, -1.28639)
        self.assertEqual(lon, 36.81722)
        self.assertEqual(label, "Nairobi Farm")
        self.assertEqual(warnings, [])

    def test_valid_dict_coords_alternative_keys(self):
        lat, lon, label, warnings = resolve_location({"latitude": 36.7783, "longitude": -119.4179})
        self.assertEqual(lat, 36.7783)
        self.assertEqual(lon, -119.4179)
        self.assertEqual(warnings, [])

    def test_valid_string_coords(self):
        lat, lon, label, warnings = resolve_location("23.0225, 72.5714")
        self.assertEqual(lat, 23.0225)
        self.assertEqual(lon, 72.5714)
        self.assertEqual(warnings, [])

    def test_invalid_latitude_range(self):
        lat, lon, label, warnings = resolve_location((95.0, 73.0))
        self.assertIsNone(lat)
        self.assertIsNone(lon)
        self.assertTrue(any("Invalid latitude" in w for w in warnings))

    def test_invalid_longitude_range(self):
        lat, lon, label, warnings = resolve_location((18.0, 195.0))
        self.assertIsNone(lat)
        self.assertIsNone(lon)
        self.assertTrue(any("Invalid longitude" in w for w in warnings))

    def test_invalid_string_format(self):
        lat, lon, label, warnings = resolve_location("not_a_coord, nor_a_number")
        # Should attempt geocoding or fail gracefully
        self.assertIsNone(lat)
        self.assertTrue(len(warnings) > 0)

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_geocoding_success(self, mock_get):
        mock_get.return_value = (
            {
                "results": [
                    {
                        "name": "Pune",
                        "latitude": 18.5196,
                        "longitude": 73.8553,
                        "country": "India",
                        "admin1": "Maharashtra",
                    }
                ]
            },
            None,
        )
        lat, lon, label, warnings = resolve_location("Pune")
        self.assertEqual(lat, 18.5196)
        self.assertEqual(lon, 73.8553)
        self.assertIn("Pune", label)
        self.assertIn("India", label)
        self.assertEqual(warnings, [])

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_geocoding_not_found(self, mock_get):
        mock_get.return_value = ({"results": []}, None)
        lat, lon, label, warnings = resolve_location("NonExistentCityXYZ999")
        self.assertIsNone(lat)
        self.assertIsNone(lon)
        self.assertTrue(any("could not be resolved" in w for w in warnings))

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_geocoding_network_failure(self, mock_get):
        mock_get.return_value = (None, "Network timeout")
        lat, lon, label, warnings = resolve_location("Nashik")
        self.assertIsNone(lat)
        self.assertIsNone(lon)
        self.assertTrue(any("Geocoding request failed" in w for w in warnings))


class TestDailyParsingAndAggregation(unittest.TestCase):
    """Tests for parsing raw Open-Meteo payloads and statistical aggregation."""

    def test_valid_daily_parsing(self):
        raw = {
            "time": ["2026-10-01", "2026-10-02"],
            "temperature_2m_mean": [24.5, 25.1],
            "temperature_2m_max": [30.2, 31.0],
            "precipitation_sum": [5.2, 0.0],
            "relative_humidity_2m_mean": [72.0, 68.5],
            "et0_fao_evapotranspiration": [3.8, 4.1],
        }
        series, warnings = parse_daily_series(raw)
        self.assertEqual(len(series), 2)
        self.assertEqual(series[0]["date"], "2026-10-01")
        self.assertEqual(series[0]["temperature_mean_c"], 24.5)
        self.assertEqual(series[0]["precipitation_mm"], 5.2)
        self.assertEqual(series[1]["precipitation_mm"], 0.0)
        self.assertEqual(warnings, [])

    def test_parsing_with_none_and_nan(self):
        raw = {
            "time": ["2026-10-01", "2026-10-02", "2026-10-03"],
            "temperature_2m_mean": [25.0, None, float("nan")],
            "precipitation_sum": [None, 10.5, 5.0],
        }
        series, _ = parse_daily_series(raw)
        self.assertEqual(len(series), 3)
        self.assertEqual(series[0]["temperature_mean_c"], 25.0)
        self.assertIsNone(series[1]["temperature_mean_c"])
        self.assertIsNone(series[2]["temperature_mean_c"])
        self.assertIsNone(series[0]["precipitation_mm"])
        self.assertEqual(series[1]["precipitation_mm"], 10.5)

    def test_period_stats_computation(self):
        daily = [
            {
                "date": "2026-10-01",
                "temperature_mean_c": 20.0,
                "temperature_max_c": 25.0,
                "precipitation_mm": 10.0,
                "relative_humidity_pct": 60.0,
                "et0_mm": 3.0,
            },
            {
                "date": "2026-10-02",
                "temperature_mean_c": 30.0,
                "temperature_max_c": 35.0,
                "precipitation_mm": 20.0,
                "relative_humidity_pct": 80.0,
                "et0_mm": 5.0,
            },
        ]
        stats, warnings = compute_period_stats(daily)
        self.assertEqual(stats["mean_temperature_c"], 25.0)  # (20 + 30) / 2
        self.assertEqual(stats["max_temperature_c"], 35.0)   # max(25, 35)
        self.assertEqual(stats["rainfall_mm"], 30.0)         # 10 + 20
        self.assertEqual(stats["mean_humidity_pct"], 70.0)   # (60 + 80) / 2
        self.assertEqual(stats["et0_mm"], 8.0)               # 3 + 5
        self.assertEqual(warnings, [])

    def test_period_stats_partial_missing_observations(self):
        daily = [
            {"date": "2026-10-01", "temperature_mean_c": 22.0, "precipitation_mm": None},
            {"date": "2026-10-02", "temperature_mean_c": None, "precipitation_mm": 15.0},
            {"date": "2026-10-03", "temperature_mean_c": 26.0, "precipitation_mm": 5.0},
        ]
        stats, warnings = compute_period_stats(daily)
        # Mean temp calculated over 2 valid days: (22 + 26) / 2 = 24.0
        self.assertEqual(stats["mean_temperature_c"], 24.0)
        # Rainfall sum over 2 valid days: 15 + 5 = 20.0
        self.assertEqual(stats["rainfall_mm"], 20.0)
        # Explicit warnings generated
        self.assertTrue(any("Mean temperature: 1 missing day(s)" in w for w in warnings))
        self.assertTrue(any("Precipitation: 1 missing day(s)" in w for w in warnings))

    def test_period_stats_completely_missing(self):
        daily = [
            {"date": "2026-10-01", "temperature_mean_c": None, "precipitation_mm": None},
            {"date": "2026-10-02", "temperature_mean_c": None, "precipitation_mm": None},
        ]
        stats, warnings = compute_period_stats(daily)
        self.assertIsNone(stats["mean_temperature_c"])
        self.assertIsNone(stats["rainfall_mm"])
        self.assertTrue(any("Mean temperature data is entirely missing" in w for w in warnings))


class TestAnomalyCalculations(unittest.TestCase):
    """Tests for manually verifiable anomaly and deviation calculations."""

    def test_manually_verifiable_synthetic_example_1_positive_anomalies(self):
        """Synthetic fixture: Warmer and wetter recent period.
        Recent: Mean Temp = 28.0 C, Rainfall = 120.0 mm
        Baseline: Mean Temp = 25.5 C, Rainfall = 100.0 mm
        Expected Temp Anomaly: 28.0 - 25.5 = +2.5 C
        Expected Rainfall Deviation: 120.0 - 100.0 = +20.0 mm
        Expected Rainfall Pct: (20.0 / 100.0) * 100 = +20.0%
        """
        recent = {"mean_temperature_c": 28.0, "rainfall_mm": 120.0}
        baseline = {"mean_temperature_c": 25.5, "rainfall_mm": 100.0}
        anomalies, warnings = calculate_anomalies(recent, baseline)
        self.assertEqual(anomalies["temperature_c"], 2.5)
        self.assertEqual(anomalies["rainfall_mm"], 20.0)
        self.assertEqual(anomalies["rainfall_pct"], 20.0)
        self.assertEqual(warnings, [])

    def test_manually_verifiable_synthetic_example_2_negative_anomalies(self):
        """Synthetic fixture: Cooler and drier recent period (drought stress).
        Recent: Mean Temp = 21.0 C, Rainfall = 35.0 mm
        Baseline: Mean Temp = 23.5 C, Rainfall = 70.0 mm
        Expected Temp Anomaly: 21.0 - 23.5 = -2.5 C
        Expected Rainfall Deviation: 35.0 - 70.0 = -35.0 mm
        Expected Rainfall Pct: (-35.0 / 70.0) * 100 = -50.0%
        """
        recent = {"mean_temperature_c": 21.0, "rainfall_mm": 35.0}
        baseline = {"mean_temperature_c": 23.5, "rainfall_mm": 70.0}
        anomalies, warnings = calculate_anomalies(recent, baseline)
        self.assertEqual(anomalies["temperature_c"], -2.5)
        self.assertEqual(anomalies["rainfall_mm"], -35.0)
        self.assertEqual(anomalies["rainfall_pct"], -50.0)
        self.assertEqual(warnings, [])

    def test_manually_verifiable_synthetic_example_3_zero_baseline_rainfall(self):
        """Synthetic fixture: Arid region where historical baseline rainfall is 0.0 mm.
        Recent: Mean Temp = 32.0 C, Rainfall = 15.0 mm
        Baseline: Mean Temp = 31.0 C, Rainfall = 0.0 mm
        Expected Temp Anomaly: 32.0 - 31.0 = +1.0 C
        Expected Rainfall Deviation: 15.0 - 0.0 = +15.0 mm
        Expected Rainfall Pct: None (division by zero undefined; must not crash or return infinity)
        """
        recent = {"mean_temperature_c": 32.0, "rainfall_mm": 15.0}
        baseline = {"mean_temperature_c": 31.0, "rainfall_mm": 0.0}
        anomalies, warnings = calculate_anomalies(recent, baseline)
        self.assertEqual(anomalies["temperature_c"], 1.0)
        self.assertEqual(anomalies["rainfall_mm"], 15.0)
        self.assertIsNone(anomalies["rainfall_pct"])
        self.assertTrue(any("Baseline rainfall is zero" in w for w in warnings))

    def test_missing_recent_or_baseline_returns_none(self):
        recent = {"mean_temperature_c": 25.0, "rainfall_mm": None}
        baseline = {"mean_temperature_c": None, "rainfall_mm": 50.0}
        anomalies, warnings = calculate_anomalies(recent, baseline)
        self.assertIsNone(anomalies["temperature_c"])
        self.assertIsNone(anomalies["rainfall_mm"])
        self.assertIsNone(anomalies["rainfall_pct"])
        self.assertEqual(len(warnings), 2)


class TestHistoricalBaselineRetrieval(unittest.TestCase):
    """Tests for baseline window mapping and multi-year aggregation."""

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_baseline_multi_year_averaging(self, mock_get):
        # Mock 2 historical baseline years: 2022 and 2023
        def side_effect(url, params, timeout):
            year = int(params["start_date"].split("-")[0])
            if year == 2022:
                return {
                    "daily": {
                        "time": ["2022-10-01", "2022-10-02"],
                        "temperature_2m_mean": [24.0, 26.0],  # Mean: 25.0
                        "temperature_2m_max": [28.0, 30.0],
                        "precipitation_sum": [30.0, 20.0],    # Sum: 50.0
                        "relative_humidity_2m_mean": [60.0, 70.0],
                        "et0_fao_evapotranspiration": [3.0, 3.0],
                    }
                }, None
            elif year == 2023:
                return {
                    "daily": {
                        "time": ["2023-10-01", "2023-10-02"],
                        "temperature_2m_mean": [26.0, 28.0],  # Mean: 27.0
                        "temperature_2m_max": [30.0, 32.0],
                        "precipitation_sum": [40.0, 30.0],    # Sum: 70.0
                        "relative_humidity_2m_mean": [65.0, 75.0],
                        "et0_fao_evapotranspiration": [4.0, 4.0],
                    }
                }, None
            return None, "Not found"

        mock_get.side_effect = side_effect

        baseline, warnings = fetch_historical_baseline(
            lat=18.52,
            lon=73.85,
            recent_end_date="2026-10-02",
            period_days=2,
            baseline_years=[2022, 2023],
        )

        # Baseline mean temp = (25.0 + 27.0) / 2 = 26.0 C
        self.assertEqual(baseline["mean_temperature_c"], 26.0)
        # Baseline rainfall = (50.0 + 70.0) / 2 = 60.0 mm
        self.assertEqual(baseline["rainfall_mm"], 60.0)
        self.assertEqual(baseline["years"], [2022, 2023])

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_baseline_insufficient_data(self, mock_get):
        # When all baseline years fail
        mock_get.return_value = (None, "HTTP 500 Server Error")
        baseline, warnings = fetch_historical_baseline(
            lat=18.52,
            lon=73.85,
            recent_end_date="2026-10-02",
            period_days=2,
            baseline_years=[2022, 2023],
        )
        self.assertIsNone(baseline["mean_temperature_c"])
        self.assertIsNone(baseline["rainfall_mm"])
        self.assertEqual(baseline["years"], [])
        self.assertTrue(any("Insufficient historical baseline data" in w for w in warnings))


class TestGetEnvironmentPublicInterface(unittest.TestCase):
    """Tests for the primary public interface get_environment()."""

    def setUp(self):
        self.synthetic_recent_daily = {
            "time": ["2026-10-01", "2026-10-02", "2026-10-03"],
            "temperature_2m_mean": [26.0, 28.0, 30.0],
            "temperature_2m_max": [31.0, 33.0, 35.0],
            "precipitation_sum": [10.0, 20.0, 10.0],
            "relative_humidity_2m_mean": [65.0, 70.0, 75.0],
            "et0_fao_evapotranspiration": [3.5, 4.0, 4.5],
        }
        self.synthetic_baseline_daily = {
            "time": ["2023-10-01", "2023-10-02", "2023-10-03"],
            "temperature_2m_mean": [24.0, 25.0, 26.0],
            "temperature_2m_max": [29.0, 30.0, 31.0],
            "precipitation_sum": [5.0, 10.0, 5.0],
            "relative_humidity_2m_mean": [60.0, 62.0, 64.0],
            "et0_fao_evapotranspiration": [3.0, 3.2, 3.4],
        }

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_full_pipeline_success(self, mock_get):
        def side_effect(url, params, timeout):
            if "forecast" in url:
                return {"daily": self.synthetic_recent_daily}, None
            if "archive" in url:
                return {"daily": self.synthetic_baseline_daily}, None
            return None, "Unexpected endpoint"

        mock_get.side_effect = side_effect

        result = get_environment((18.5204, 73.8567), days=3, baseline_years=[2023])

        # Validate schema adherence
        self.assertIn("location", result)
        self.assertIn("period_days", result)
        self.assertEqual(result["period_days"], 3)
        self.assertEqual(result["source"], "Open-Meteo")

        # Verify computed recent values
        self.assertEqual(result["recent"]["mean_temperature_c"], 28.0)
        self.assertEqual(result["recent"]["max_temperature_c"], 35.0)
        self.assertEqual(result["recent"]["rainfall_mm"], 40.0)

        # Verify computed baseline values
        self.assertEqual(result["baseline"]["mean_temperature_c"], 25.0)
        self.assertEqual(result["baseline"]["rainfall_mm"], 20.0)
        self.assertEqual(result["baseline"]["years"], [2023])

        # Verify anomalies:
        # temp = 28.0 - 25.0 = 3.0 C
        # rain = 40.0 - 20.0 = 20.0 mm
        # rain_pct = (20.0 / 20.0) * 100 = 100.0%
        self.assertEqual(result["anomalies"]["temperature_c"], 3.0)
        self.assertEqual(result["anomalies"]["rainfall_mm"], 20.0)
        self.assertEqual(result["anomalies"]["rainfall_pct"], 100.0)

        # Series length
        self.assertEqual(len(result["daily_series"]), 3)

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_api_timeout_handling(self, mock_get):
        mock_get.return_value = (None, "Network timeout while connecting")
        result = get_environment((18.52, 73.85), days=30)

        # Must not crash, must return valid schema with None values
        self.assertIsNone(result["recent"]["mean_temperature_c"])
        self.assertIsNone(result["baseline"]["mean_temperature_c"])
        self.assertIsNone(result["anomalies"]["temperature_c"])
        self.assertEqual(result["daily_series"], [])
        self.assertTrue(any("Network timeout" in w for w in result["warnings"]))

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_malformed_json_response(self, mock_get):
        mock_get.return_value = (None, "Malformed JSON response")
        result = get_environment((18.52, 73.85), days=30)
        self.assertIsNone(result["recent"]["rainfall_mm"])
        self.assertTrue(any("Malformed JSON" in w for w in result["warnings"]))

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_missing_daily_key_in_payload(self, mock_get):
        mock_get.return_value = ({"error": "rate limited"}, None)
        result = get_environment((18.52, 73.85), days=30)
        self.assertIsNone(result["recent"]["mean_temperature_c"])
        self.assertTrue(any("Missing daily" in w for w in result["warnings"]))

    def test_invalid_coordinates_handling(self):
        result = get_environment((120.0, 200.0), days=30)
        self.assertIsNone(result["recent"]["mean_temperature_c"])
        self.assertTrue(any("Invalid latitude" in w for w in result["warnings"]))

    @patch("src.cropcare.climate_engine.execute_http_get")
    def test_insufficient_baseline_data(self, mock_get):
        # Recent succeeds, but baseline fails
        def side_effect(url, params, timeout):
            if "forecast" in url:
                return {"daily": self.synthetic_recent_daily}, None
            return None, "Baseline unavailable"

        mock_get.side_effect = side_effect
        result = get_environment((18.52, 73.85), days=3, baseline_years=[2023])

        # Recent is populated
        self.assertEqual(result["recent"]["mean_temperature_c"], 28.0)
        # Baseline is None
        self.assertIsNone(result["baseline"]["mean_temperature_c"])
        self.assertEqual(result["baseline"]["years"], [])
        # Anomalies are None
        self.assertIsNone(result["anomalies"]["temperature_c"])
        self.assertIsNone(result["anomalies"]["rainfall_mm"])
        # Warning explains why
        self.assertTrue(any("Insufficient historical baseline data" in w for w in result["warnings"]))


class TestCsvExport(unittest.TestCase):
    """Tests for CSV generation and exported format."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_export_climate_csv_structure(self):
        sample_data = {
            "location": "Pune, Maharashtra, India",
            "period_days": 2,
            "source": "Open-Meteo",
            "recent": {
                "mean_temperature_c": 25.5,
                "max_temperature_c": 31.0,
                "rainfall_mm": 12.4,
                "mean_humidity_pct": 68.0,
                "et0_mm": 7.2,
            },
            "baseline": {
                "mean_temperature_c": 24.0,
                "rainfall_mm": 10.0,
                "mean_humidity_pct": 65.0,
                "et0_mm": 6.8,
                "years": [2021, 2022, 2023],
                "start_date": "2021-10-01",
                "end_date": "2023-10-02",
            },
            "anomalies": {
                "temperature_c": 1.5,
                "rainfall_mm": 2.4,
                "rainfall_pct": 24.0,
            },
            "daily_series": [
                {
                    "date": "2026-10-01",
                    "temperature_mean_c": 25.0,
                    "temperature_max_c": 30.5,
                    "precipitation_mm": 10.4,
                    "relative_humidity_pct": 70.0,
                    "et0_mm": 3.5,
                },
                {
                    "date": "2026-10-02",
                    "temperature_mean_c": 26.0,
                    "temperature_max_c": 31.5,
                    "precipitation_mm": 2.0,
                    "relative_humidity_pct": 66.0,
                    "et0_mm": 3.7,
                },
            ],
            "warnings": [],
        }

        csv_path = export_climate_csv(sample_data, output_dir=self.test_dir)
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 2)
        # Verify columns
        self.assertEqual(rows[0]["date"], "2026-10-01")
        self.assertEqual(rows[0]["location"], "Pune, Maharashtra, India")
        self.assertEqual(rows[0]["temperature_mean_c"], "25.0")
        self.assertEqual(rows[0]["precipitation_mm"], "10.4")
        self.assertEqual(rows[0]["baseline_mean_temperature_c"], "24.0")
        self.assertEqual(rows[0]["temperature_anomaly_c"], "1.5")
        self.assertEqual(rows[0]["rainfall_deviation_mm"], "2.4")


if __name__ == "__main__":
    unittest.main()
