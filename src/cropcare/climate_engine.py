"""Climate & Environmental Intelligence Engine for CropCare.

This module provides environmental data retrieval and climate anomaly analysis
for user-specified agricultural locations. It retrieves historical and recent
weather reanalysis observations via public Open-Meteo APIs, computes localized
recent-period environmental statistics, compares them against multi-year historical
baselines, and returns a structured output schema ready for UI integration.

Scientific and Ethical Principles:
1. Historical and reanalysis data are modelled reanalysis products, not live field sensor readings.
2. Weather anomalies represent localized departures from recent multi-year baselines and do not
   constitute definitive proof of global climate change.
3. Baseline comparisons are statistical references, not validated crop-specific agronomic thresholds.
4. No crop disease risk scores or disease causation claims are produced by this engine.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse

try:
    import requests
except ImportError:
    requests = None  # Fallback to urllib if requests is missing

logger = logging.getLogger(__name__)

# API Endpoints
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Defaults
DEFAULT_PERIOD_DAYS = 30
DEFAULT_BASELINE_YEARS_COUNT = 3
DEFAULT_CONNECT_TIMEOUT_SEC = 5.0
DEFAULT_READ_TIMEOUT_SEC = 10.0


def _build_empty_schema(location_label: str = "Unknown", period_days: int = DEFAULT_PERIOD_DAYS) -> Dict[str, Any]:
    """Construct an empty, fully initialized output dictionary complying with the contract.

    Args:
        location_label: Human-readable location description or coordinates.
        period_days: Number of days analysed in the recent window.

    Returns:
        Structured dictionary with all keys set to None / empty collections.
    """
    return {
        "location": location_label,
        "period_days": int(period_days),
        "source": "Open-Meteo",
        "recent": {
            "mean_temperature_c": None,
            "max_temperature_c": None,
            "rainfall_mm": None,
            "mean_humidity_pct": None,
            "et0_mm": None,
        },
        "baseline": {
            "mean_temperature_c": None,
            "rainfall_mm": None,
            "mean_humidity_pct": None,
            "et0_mm": None,
            "years": [],
            "start_date": None,
            "end_date": None,
        },
        "anomalies": {
            "temperature_c": None,
            "rainfall_mm": None,
            "rainfall_pct": None,
        },
        "daily_series": [],
        "warnings": [],
    }


def execute_http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT_SEC, DEFAULT_READ_TIMEOUT_SEC),
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Execute an HTTP GET request with sensible timeouts and graceful error handling.

    Args:
        url: Target API URL.
        params: Query parameters dictionary.
        timeout: Tuple of (connect_timeout, read_timeout) in seconds.

    Returns:
        Tuple of (parsed_json_dict, error_message). On success, error_message is None.
    """
    params = params or {}

    if requests is not None:
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code != 200:
                return None, f"HTTP Error {response.status_code} from {url}"
            try:
                return response.json(), None
            except Exception as e:
                return None, f"Malformed JSON response from {url}: {e}"
        except requests.exceptions.Timeout:
            return None, f"Network timeout while connecting to {url}"
        except requests.exceptions.ConnectionError:
            return None, f"Network connection error while reaching {url}"
        except requests.exceptions.RequestException as e:
            return None, f"HTTP request failed: {e}"
        except Exception as e:
            return None, f"Unexpected request failure: {e}"

    # Standard library urllib fallback
    try:
        import urllib.request
        query_str = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_str}" if query_str else url
        total_timeout = timeout[0] + timeout[1]
        req = urllib.request.Request(full_url, headers={"User-Agent": "CropCare-ClimateEngine/1.0"})
        with urllib.request.urlopen(req, timeout=total_timeout) as res:
            if res.status != 200:
                return None, f"HTTP Error {res.status} from {url}"
            payload = res.read().decode("utf-8")
            try:
                return json.loads(payload), None
            except Exception as e:
                return None, f"Malformed JSON response from {url}: {e}"
    except urllib.error.HTTPError as e:
        return None, f"HTTP Error {e.code} from {url}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Network connection error: {e.reason}"
    except TimeoutError:
        return None, f"Network timeout while connecting to {url}"
    except Exception as e:
        return None, f"Unexpected request failure via urllib: {e}"


def resolve_location(
    location: Any,
    timeout: Tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT_SEC, DEFAULT_READ_TIMEOUT_SEC),
) -> Tuple[Optional[float], Optional[float], str, List[str]]:
    """Resolve user-specified location to latitude, longitude, and descriptive label.

    Supports:
        - Tuple/List of (lat, lon)
        - Dict with 'latitude'/'lat' and 'longitude'/'lon'
        - Coordinate string e.g. "18.5204, 73.8567"
        - Place name string resolved via Open-Meteo Geocoding API e.g. "Pune"

    Args:
        location: Raw location input.
        timeout: Network timeout tuple for geocoding requests.

    Returns:
        Tuple of (latitude, longitude, resolved_name, warnings_list).
    """
    warnings: List[str] = []

    def _validate_coords(lat: float, lon: float, label: str) -> Tuple[Optional[float], Optional[float], str, List[str]]:
        if not (-90.0 <= lat <= 90.0):
            return None, None, label, [f"Invalid latitude {lat}. Latitude must be between -90 and 90 degrees."]
        if not (-180.0 <= lon <= 180.0):
            return None, None, label, [f"Invalid longitude {lon}. Longitude must be between -180 and 180 degrees."]
        return round(lat, 5), round(lon, 5), label, []

    # 1. Coordinate tuple or list
    if isinstance(location, (list, tuple)):
        if len(location) != 2:
            return None, None, str(location), ["Coordinate list/tuple must contain exactly 2 numbers (lat, lon)."]
        try:
            lat = float(location[0])
            lon = float(location[1])
            label = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            return _validate_coords(lat, lon, label)
        except (ValueError, TypeError):
            return None, None, str(location), ["Coordinates could not be converted to float."]

    # 2. Dictionary input
    if isinstance(location, dict):
        lat_val = location.get("latitude") if "latitude" in location else location.get("lat")
        lon_val = location.get("longitude") if "longitude" in location else location.get("lon")
        if lat_val is None or lon_val is None:
            return None, None, str(location), ["Location dictionary must specify 'latitude'/'lat' and 'longitude'/'lon'."]
        try:
            lat = float(lat_val)
            lon = float(lon_val)
            label = location.get("name") or location.get("label") or f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            return _validate_coords(lat, lon, label)
        except (ValueError, TypeError):
            return None, None, str(location), ["Coordinates in dictionary could not be converted to float."]

    # 3. String input
    if isinstance(location, str):
        trimmed = location.strip()
        if not trimmed:
            return None, None, "Empty location", ["Location string cannot be empty."]

        # Check if string is "lat, lon"
        if "," in trimmed:
            parts = [p.strip() for p in trimmed.split(",")]
            if len(parts) == 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    label = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
                    return _validate_coords(lat, lon, label)
                except ValueError:
                    # Not pure numbers, fall through to geocoding
                    pass

        # Geocoding via Open-Meteo Geocoding API
        params = {
            "name": trimmed,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        data, err = execute_http_get(OPEN_METEO_GEOCODING_URL, params=params, timeout=timeout)
        if err:
            return None, None, trimmed, [f"Geocoding request failed: {err}"]
        if not data or not data.get("results"):
            return None, None, trimmed, [f"Location '{trimmed}' could not be resolved by geocoding service."]

        top_match = data["results"][0]
        try:
            lat = float(top_match["latitude"])
            lon = float(top_match["longitude"])
            name = top_match.get("name", trimmed)
            admin1 = top_match.get("admin1", "")
            country = top_match.get("country", "")
            parts = [p for p in [name, admin1, country] if p]
            label = ", ".join(parts) if parts else trimmed
            return _validate_coords(lat, lon, label)
        except (KeyError, ValueError, TypeError) as e:
            return None, None, trimmed, [f"Invalid geocoding response structure: {e}"]

    return None, None, str(location), [f"Unsupported location type: {type(location).__name__}."]


def parse_daily_series(raw_daily: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse raw daily arrays from Open-Meteo into standardized observation records.

    Standardized record structure:
        - date: ISO-8601 string (YYYY-MM-DD)
        - temperature_mean_c: float | None
        - temperature_max_c: float | None
        - precipitation_mm: float | None
        - relative_humidity_pct: float | None
        - et0_mm: float | None

    Args:
        raw_daily: Dictionary containing daily arrays from Open-Meteo API response.

    Returns:
        Tuple of (daily_series_list, warnings_list).
    """
    warnings: List[str] = []
    if not raw_daily or not isinstance(raw_daily, dict):
        return [], ["Raw daily weather payload is empty or invalid."]

    dates = raw_daily.get("time") or []
    if not dates:
        return [], ["No daily timestamps found in weather response."]

    n_days = len(dates)
    mean_temps = raw_daily.get("temperature_2m_mean") or [None] * n_days
    max_temps = raw_daily.get("temperature_2m_max") or [None] * n_days
    precips = raw_daily.get("precipitation_sum") or [None] * n_days
    humidities = raw_daily.get("relative_humidity_2m_mean") or [None] * n_days
    et0s = raw_daily.get("et0_fao_evapotranspiration") or [None] * n_days

    series: List[Dict[str, Any]] = []

    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) else round(f, 2)
        except (ValueError, TypeError):
            return None

    for i in range(n_days):
        date_str = str(dates[i])
        record = {
            "date": date_str,
            "temperature_mean_c": _safe_float(mean_temps[i]) if i < len(mean_temps) else None,
            "temperature_max_c": _safe_float(max_temps[i]) if i < len(max_temps) else None,
            "precipitation_mm": _safe_float(precips[i]) if i < len(precips) else None,
            "relative_humidity_pct": _safe_float(humidities[i]) if i < len(humidities) else None,
            "et0_mm": _safe_float(et0s[i]) if i < len(et0s) else None,
        }
        series.append(record)

    return series, warnings


def compute_period_stats(daily_series: List[Dict[str, Any]]) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Compute aggregate statistics for a given series of daily observations.

    Aggregates:
        - mean_temperature_c: Mean of valid daily mean temperatures.
        - max_temperature_c: Maximum of valid daily maximum temperatures.
        - rainfall_mm: Sum of valid daily precipitation observations.
        - mean_humidity_pct: Mean of valid daily relative humidity.
        - et0_mm: Sum of valid daily reference evapotranspiration observations.

    Missing observation handling:
        Missing values (None) are not silently converted to zero.
        - If all observations for a variable are missing, the aggregate is None.
        - If partial observations are missing, averages/sums are computed from available
          data, and an explicit informational warning is added.

    Args:
        daily_series: List of daily observation dictionaries.

    Returns:
        Tuple of (stats_dict, warnings_list).
    """
    warnings: List[str] = []
    if not daily_series:
        return {
            "mean_temperature_c": None,
            "max_temperature_c": None,
            "rainfall_mm": None,
            "mean_humidity_pct": None,
            "et0_mm": None,
        }, ["Daily series is empty; cannot compute summary statistics."]

    total_days = len(daily_series)

    # 1. Mean Temperature
    valid_t_mean = [d["temperature_mean_c"] for d in daily_series if d.get("temperature_mean_c") is not None]
    if valid_t_mean:
        mean_t = round(sum(valid_t_mean) / len(valid_t_mean), 2)
        if len(valid_t_mean) < total_days:
            warnings.append(
                f"Mean temperature: {total_days - len(valid_t_mean)} missing day(s); calculated over {len(valid_t_mean)} available days."
            )
    else:
        mean_t = None
        warnings.append("Mean temperature data is entirely missing for the analysed period.")

    # 2. Max Temperature
    valid_t_max = [d["temperature_max_c"] for d in daily_series if d.get("temperature_max_c") is not None]
    if valid_t_max:
        max_t = round(max(valid_t_max), 2)
    else:
        max_t = None
        warnings.append("Maximum temperature data is entirely missing for the analysed period.")

    # 3. Rainfall (Precipitation Sum)
    valid_precip = [d["precipitation_mm"] for d in daily_series if d.get("precipitation_mm") is not None]
    if valid_precip:
        rainfall_sum = round(sum(valid_precip), 2)
        if len(valid_precip) < total_days:
            warnings.append(
                f"Precipitation: {total_days - len(valid_precip)} missing day(s); cumulative rainfall may be an underestimate."
            )
    else:
        rainfall_sum = None
        warnings.append("Precipitation data is entirely missing for the analysed period.")

    # 4. Relative Humidity
    valid_rh = [d["relative_humidity_pct"] for d in daily_series if d.get("relative_humidity_pct") is not None]
    if valid_rh:
        mean_rh = round(sum(valid_rh) / len(valid_rh), 2)
        if len(valid_rh) < total_days:
            warnings.append(
                f"Relative humidity: {total_days - len(valid_rh)} missing day(s); calculated over {len(valid_rh)} available days."
            )
    else:
        mean_rh = None

    # 5. Reference Evapotranspiration (ET0)
    valid_et0 = [d["et0_mm"] for d in daily_series if d.get("et0_mm") is not None]
    if valid_et0:
        et0_sum = round(sum(valid_et0), 2)
        if len(valid_et0) < total_days:
            warnings.append(
                f"Reference ET0: {total_days - len(valid_et0)} missing day(s); cumulative ET0 may be an underestimate."
            )
    else:
        et0_sum = None

    stats = {
        "mean_temperature_c": mean_t,
        "max_temperature_c": max_t,
        "rainfall_mm": rainfall_sum,
        "mean_humidity_pct": mean_rh,
        "et0_mm": et0_sum,
    }
    return stats, warnings


def calculate_anomalies(
    recent_stats: Dict[str, Optional[float]],
    baseline_stats: Dict[str, Optional[float]],
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Calculate environmental anomalies between recent conditions and historical baseline.

    Formulas:
        temperature_anomaly_c = recent_mean_temperature_c - baseline_mean_temperature_c
        rainfall_deviation_mm = recent_total_precipitation_mm - baseline_total_precipitation_mm
        rainfall_pct = (rainfall_deviation_mm / baseline_rainfall_mm) * 100  (only if baseline > 0)

    Args:
        recent_stats: Dictionary containing recent summary metrics.
        baseline_stats: Dictionary containing baseline summary metrics.

    Returns:
        Tuple of (anomalies_dict, warnings_list).
    """
    warnings: List[str] = []
    anomalies: Dict[str, Optional[float]] = {
        "temperature_c": None,
        "rainfall_mm": None,
        "rainfall_pct": None,
    }

    # Temperature Anomaly
    r_temp = recent_stats.get("mean_temperature_c")
    b_temp = baseline_stats.get("mean_temperature_c")
    if r_temp is not None and b_temp is not None:
        anomalies["temperature_c"] = round(r_temp - b_temp, 2)
    else:
        if r_temp is None:
            warnings.append("Recent mean temperature unavailable; temperature anomaly could not be computed.")
        if b_temp is None:
            warnings.append("Baseline mean temperature unavailable; temperature anomaly could not be computed.")

    # Rainfall Deviation
    r_rain = recent_stats.get("rainfall_mm")
    b_rain = baseline_stats.get("rainfall_mm")
    if r_rain is not None and b_rain is not None:
        deviation = round(r_rain - b_rain, 2)
        anomalies["rainfall_mm"] = deviation
        if b_rain > 0:
            anomalies["rainfall_pct"] = round((deviation / b_rain) * 100.0, 2)
        else:
            anomalies["rainfall_pct"] = None
            warnings.append("Baseline rainfall is zero; percentage rainfall deviation is undefined.")
    else:
        if r_rain is None:
            warnings.append("Recent rainfall unavailable; rainfall deviation could not be computed.")
        if b_rain is None:
            warnings.append("Baseline rainfall unavailable; rainfall deviation could not be computed.")

    return anomalies, warnings


def _map_calendar_window(end_date_str: str, period_days: int, target_year: int) -> Tuple[str, str]:
    """Map a recent period window of length `period_days` to the equivalent calendar window in `target_year`.

    Safely handles leap year adjustments (Feb 29 mapped to Feb 28 in non-leap years) and
    ensures the target window has the exact same day span.

    Args:
        end_date_str: ISO string of recent end date (YYYY-MM-DD).
        period_days: Number of days in the period.
        target_year: The historical year to map to.

    Returns:
        Tuple of (start_date_str, end_date_str) in ISO format for target_year.
    """
    recent_end = datetime.date.fromisoformat(end_date_str)
    month = recent_end.month
    day = recent_end.day

    try:
        target_end = datetime.date(target_year, month, day)
    except ValueError:
        # Handles Feb 29 mapped to non-leap year
        target_end = datetime.date(target_year, 2, 28)

    target_start = target_end - datetime.timedelta(days=period_days - 1)
    return target_start.isoformat(), target_end.isoformat()


def fetch_historical_baseline(
    lat: float,
    lon: float,
    recent_end_date: str,
    period_days: int,
    baseline_years: Optional[List[int]] = None,
    timeout: Tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT_SEC, DEFAULT_READ_TIMEOUT_SEC),
) -> Tuple[Dict[str, Any], List[str]]:
    """Retrieve historical reanalysis baseline data for equivalent calendar dates across previous years.

    For each specified baseline year, queries Open-Meteo Archive API for the corresponding
    calendar window. Computes baseline averages across all successfully retrieved years.

    Args:
        lat: Latitude of location.
        lon: Longitude of location.
        recent_end_date: ISO date string for recent observation period end.
        period_days: Number of days in the window.
        baseline_years: Explicit list of historical years to compare against, or None for defaults.
        timeout: HTTP timeout tuple.

    Returns:
        Tuple of (baseline_dict, warnings_list).
    """
    warnings: List[str] = []
    end_year = int(recent_end_date.split("-")[0])

    if baseline_years is None:
        baseline_years = [end_year - 3, end_year - 2, end_year - 1]

    if not baseline_years:
        return {
            "mean_temperature_c": None,
            "rainfall_mm": None,
            "mean_humidity_pct": None,
            "et0_mm": None,
            "years": [],
            "start_date": None,
            "end_date": None,
        }, ["No baseline years specified; baseline cannot be calculated."]

    successful_years: List[int] = []
    yearly_t_means: List[float] = []
    yearly_precips: List[float] = []
    yearly_humidities: List[float] = []
    yearly_et0s: List[float] = []

    earliest_start: Optional[str] = None
    latest_end: Optional[str] = None

    for year in baseline_years:
        b_start, b_end = _map_calendar_window(recent_end_date, period_days, year)
        if earliest_start is None or b_start < earliest_start:
            earliest_start = b_start
        if latest_end is None or b_end > latest_end:
            latest_end = b_end

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": b_start,
            "end_date": b_end,
            "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean,et0_fao_evapotranspiration",
            "timezone": "UTC",
        }

        data, err = execute_http_get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
        if err or not data or "daily" not in data:
            warnings.append(f"Historical baseline for year {year} ({b_start} to {b_end}) unavailable: {err or 'Missing daily data'}")
            continue

        series, _ = parse_daily_series(data.get("daily"))
        if not series:
            warnings.append(f"Historical baseline series for year {year} was empty.")
            continue

        stats, _ = compute_period_stats(series)
        successful_years.append(year)

        if stats.get("mean_temperature_c") is not None:
            yearly_t_means.append(stats["mean_temperature_c"])
        if stats.get("rainfall_mm") is not None:
            yearly_precips.append(stats["rainfall_mm"])
        if stats.get("mean_humidity_pct") is not None:
            yearly_humidities.append(stats["mean_humidity_pct"])
        if stats.get("et0_mm") is not None:
            yearly_et0s.append(stats["et0_mm"])

    if not successful_years:
        warnings.append("Insufficient historical baseline data; no baseline years could be retrieved.")
        return {
            "mean_temperature_c": None,
            "rainfall_mm": None,
            "mean_humidity_pct": None,
            "et0_mm": None,
            "years": [],
            "start_date": None,
            "end_date": None,
        }, warnings

    baseline_stats = {
        "mean_temperature_c": round(sum(yearly_t_means) / len(yearly_t_means), 2) if yearly_t_means else None,
        "rainfall_mm": round(sum(yearly_precips) / len(yearly_precips), 2) if yearly_precips else None,
        "mean_humidity_pct": round(sum(yearly_humidities) / len(yearly_humidities), 2) if yearly_humidities else None,
        "et0_mm": round(sum(yearly_et0s) / len(yearly_et0s), 2) if yearly_et0s else None,
        "years": sorted(successful_years),
        "start_date": earliest_start,
        "end_date": latest_end,
    }

    if len(successful_years) < len(baseline_years):
        warnings.append(
            f"Baseline calculated with {len(successful_years)} of {len(baseline_years)} requested years: {successful_years}."
        )

    return baseline_stats, warnings


def export_climate_csv(
    data: Dict[str, Any],
    output_dir: str = "data/climate",
    filename: Optional[str] = None,
) -> str:
    """Export climate observations and baseline anomalies to a CSV file.

    Columns included:
        - date: Date of daily observation (YYYY-MM-DD)
        - location: Resolved location string or coordinates
        - source: Data provider ("Open-Meteo")
        - temperature_mean_c: Mean daily 2m temperature (°C)
        - temperature_max_c: Max daily 2m temperature (°C)
        - precipitation_mm: Daily precipitation total (mm)
        - relative_humidity_pct: Mean daily relative humidity (%)
        - et0_mm: Reference evapotranspiration ET0 (mm)
        - baseline_mean_temperature_c: Historical baseline mean temperature for the period (°C)
        - baseline_rainfall_mm: Historical baseline total rainfall for the period (mm)
        - temperature_anomaly_c: Period temperature anomaly (°C)
        - rainfall_deviation_mm: Period rainfall deviation (mm)

    Args:
        data: Environment dictionary returned by get_environment().
        output_dir: Directory where the CSV should be saved.
        filename: Optional custom filename. If None, generated automatically.

    Returns:
        String path of the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    location_str = str(data.get("location", "unknown"))
    safe_loc = re.sub(r"[^\w\-_]", "_", location_str).strip("_")
    safe_loc = re.sub(r"_+", "_", safe_loc)[:30] or "unknown_loc"

    daily_series = data.get("daily_series") or []
    if daily_series:
        start_date = daily_series[0].get("date", "start")
        end_date = daily_series[-1].get("date", "end")
    else:
        start_date = "nodata"
        end_date = "nodata"

    if not filename:
        filename = f"climate_{safe_loc}_{start_date}_{end_date}.csv"

    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "date",
        "location",
        "source",
        "temperature_mean_c",
        "temperature_max_c",
        "precipitation_mm",
        "relative_humidity_pct",
        "et0_mm",
        "baseline_mean_temperature_c",
        "baseline_rainfall_mm",
        "temperature_anomaly_c",
        "rainfall_deviation_mm",
    ]

    baseline_info = data.get("baseline") or {}
    anomaly_info = data.get("anomalies") or {}

    b_temp = baseline_info.get("mean_temperature_c")
    b_rain = baseline_info.get("rainfall_mm")
    a_temp = anomaly_info.get("temperature_c")
    a_rain = anomaly_info.get("rainfall_mm")
    source_name = data.get("source", "Open-Meteo")

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for obs in daily_series:
            row = {
                "date": obs.get("date"),
                "location": location_str,
                "source": source_name,
                "temperature_mean_c": obs.get("temperature_mean_c"),
                "temperature_max_c": obs.get("temperature_max_c"),
                "precipitation_mm": obs.get("precipitation_mm"),
                "relative_humidity_pct": obs.get("relative_humidity_pct"),
                "et0_mm": obs.get("et0_mm"),
                "baseline_mean_temperature_c": b_temp,
                "baseline_rainfall_mm": b_rain,
                "temperature_anomaly_c": a_temp,
                "rainfall_deviation_mm": a_rain,
            }
            writer.writerow(row)

    return filepath


def get_environment(
    location: Any,
    days: int = DEFAULT_PERIOD_DAYS,
    baseline_years: Optional[List[int]] = None,
    export_csv: bool = False,
    csv_dir: Optional[str] = "data/climate",
    timeout: Tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT_SEC, DEFAULT_READ_TIMEOUT_SEC),
) -> Dict[str, Any]:
    """Retrieve environmental data and historical climate anomalies for a given location.

    This is the primary public interface of the Climate & Environmental Intelligence Module.
    It guarantees a stable output contract and will never raise uncaught exceptions,
    returning structured warnings and None metrics if external dependencies or network fail.

    Output Contract:
    {
        "location": "Resolved location description or coordinates",
        "period_days": 30,
        "source": "Open-Meteo",
        "recent": {
            "mean_temperature_c": None,
            "max_temperature_c": None,
            "rainfall_mm": None,
            "mean_humidity_pct": None,
            "et0_mm": None
        },
        "baseline": {
            "mean_temperature_c": None,
            "rainfall_mm": None,
            "mean_humidity_pct": None,
            "et0_mm": None,
            "years": [],
            "start_date": None,
            "end_date": None
        },
        "anomalies": {
            "temperature_c": None,
            "rainfall_mm": None,
            "rainfall_pct": None
        },
        "daily_series": [],
        "warnings": []
    }

    Args:
        location: Coordinates tuple (lat, lon), dict, string "lat, lon", or place name string.
        days: Analysis duration in days (default: 30).
        baseline_years: List of historical baseline years (default: previous 3 years).
        export_csv: If True, writes results to a CSV inside `csv_dir`.
        csv_dir: Output folder for CSV export (default: "data/climate").
        timeout: Tuple of (connect_timeout, read_timeout) in seconds.

    Returns:
        Structured result dictionary complying with the required contract.
    """
    days = max(1, int(days))
    initial_label = str(location) if location is not None else "Unknown"
    result = _build_empty_schema(location_label=initial_label, period_days=days)

    try:
        # Step 1: Location Resolution
        lat, lon, resolved_label, loc_warnings = resolve_location(location, timeout=timeout)
        result["location"] = resolved_label
        result["warnings"].extend(loc_warnings)

        if lat is None or lon is None:
            return result

        # Step 2: Retrieve Recent Weather
        # Use Forecast API with past_days (up to 92 days) or fallback to Archive API if days > 90
        recent_data: Optional[Dict[str, Any]] = None
        recent_err: Optional[str] = None

        if days <= 90:
            params = {
                "latitude": lat,
                "longitude": lon,
                "past_days": days,
                "forecast_days": 0,
                "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean,et0_fao_evapotranspiration",
                "timezone": "UTC",
            }
            recent_data, recent_err = execute_http_get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout)
        else:
            today = datetime.date.today()
            end_d = today - datetime.timedelta(days=1)
            start_d = end_d - datetime.timedelta(days=days - 1)
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean,et0_fao_evapotranspiration",
                "timezone": "UTC",
            }
            recent_data, recent_err = execute_http_get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)

        if recent_err or not recent_data or "daily" not in recent_data:
            result["warnings"].append(
                f"Failed to retrieve recent weather observations: {recent_err or 'Missing daily block in response.'}"
            )
            return result

        # Step 3: Parse Daily Observations
        daily_series, parse_warnings = parse_daily_series(recent_data["daily"])
        result["daily_series"] = daily_series
        result["warnings"].extend(parse_warnings)

        if not daily_series:
            result["warnings"].append("Recent weather returned zero daily observations.")
            return result

        # Step 4: Calculate Recent Period Summary Statistics
        recent_stats, stats_warnings = compute_period_stats(daily_series)
        result["recent"] = recent_stats
        result["warnings"].extend(stats_warnings)

        recent_start_date = daily_series[0]["date"]
        recent_end_date = daily_series[-1]["date"]

        # Step 5: Retrieve Historical Baseline & Calculate Baseline Statistics
        baseline_stats, baseline_warnings = fetch_historical_baseline(
            lat=lat,
            lon=lon,
            recent_end_date=recent_end_date,
            period_days=len(daily_series),
            baseline_years=baseline_years,
            timeout=timeout,
        )
        result["baseline"] = baseline_stats
        result["warnings"].extend(baseline_warnings)

        # Step 6: Compute Climate Anomalies
        anomalies, anomaly_warnings = calculate_anomalies(result["recent"], result["baseline"])
        result["anomalies"] = anomalies
        result["warnings"].extend(anomaly_warnings)

        # Step 7: Optional CSV Export
        if export_csv:
            target_csv_dir = csv_dir or "data/climate"
            try:
                saved_path = export_climate_csv(result, output_dir=target_csv_dir)
                result["csv_path"] = saved_path
            except Exception as e:
                result["warnings"].append(f"CSV export failed: {e}")

    except Exception as exc:
        logger.exception("Unexpected error occurred in get_environment")
        result["warnings"].append(f"Unexpected processing error: {exc}")

    return result
