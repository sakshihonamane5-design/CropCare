# CropCare — Climate & Environmental Intelligence Module
**Module:** Member 2: Climate & Environmental Intelligence Engineer  
**Date:** October 2026  
**Status:** Complete & Tested  

---

## 1. Executive Summary & Purpose

The **Climate & Environmental Intelligence Module** (`src/cropcare/climate_engine.py`) provides localized weather intelligence and historical climate anomaly analysis for agricultural fields in **CropCare**. 

When farmers or agronomists screen crops, visual disease symptoms on leaf surfaces must be contextualized within recent environmental conditions (e.g., rainfall surges, prolonged drought stress, high humidity, or unseasonable temperature anomalies). This module autonomously retrieves weather observations for any user-specified global location, constructs an equivalent calendar baseline over historical years, computes transparent environmental anomalies, and returns a strictly typed, crash-resilient dictionary ready for frontend rendering (Member 3) and report synthesis.

---

## 2. Stable Public Interface & Contract Specification

The module exposes a stable, primary public entrypoint designed for integration by Member 3:

```python
from src.cropcare.climate_engine import get_environment

result = get_environment(location="Pune", days=30)
```

### 2.1 Function Signature
```python
def get_environment(
    location: Any,
    days: int = 30,
    baseline_years: Optional[List[int]] = None,
    export_csv: bool = False,
    csv_dir: Optional[str] = "data/climate",
    timeout: Tuple[float, float] = (5.0, 10.0),
) -> Dict[str, Any]
```

### 2.2 Supported Location Inputs
1. **Coordinate Tuple or List**: `(18.5204, 73.8567)` or `[-1.2863, 36.8172]`
2. **Coordinate Dictionary**: `{"latitude": 18.5204, "longitude": 73.8567}` or `{"lat": 18.5204, "lon": 73.8567}`
3. **Coordinate String**: `"18.5204, 73.8567"`
4. **Place Name String**: `"Pune"`, `"Nairobi, Kenya"`, `"Salinas, California"` (automatically resolved via Open-Meteo Geocoding API)

### 2.3 Required Output Contract (JSON / Dictionary Schema)
The function guarantees returning a valid dictionary adhering to this exact schema under all circumstances (even during total network failure):

```json
{
  "location": "Resolved location description or coordinates",
  "period_days": 30,
  "source": "Open-Meteo",
  "recent": {
    "mean_temperature_c": 27.86,
    "max_temperature_c": 34.2,
    "rainfall_mm": 9.0,
    "mean_humidity_pct": 52.47,
    "et0_mm": 151.7
  },
  "baseline": {
    "mean_temperature_c": 24.75,
    "rainfall_mm": 82.43,
    "mean_humidity_pct": 73.7,
    "et0_mm": 121.24,
    "years": [2023, 2024, 2025],
    "start_date": "2023-09-29",
    "end_date": "2025-10-28"
  },
  "anomalies": {
    "temperature_c": 3.11,
    "rainfall_mm": -73.43,
    "rainfall_pct": -89.08
  },
  "daily_series": [
    {
      "date": "2026-09-29",
      "temperature_mean_c": 25.7,
      "temperature_max_c": 31.2,
      "precipitation_mm": 2.7,
      "relative_humidity_pct": 76.0,
      "et0_mm": 4.57
    }
  ],
  "warnings": []
}
```

---

## 3. API Endpoints & Environmental Variables

Data is retrieved from the public, non-commercial endpoints provided by **Open-Meteo**:

| Purpose | Endpoint URL | Key Query Parameters |
| :--- | :--- | :--- |
| **Recent Observations** | `https://api.open-meteo.com/v1/forecast` | `latitude`, `longitude`, `past_days=30`, `forecast_days=0`, `daily=...`, `timezone=UTC` |
| **Historical Baseline** | `https://archive-api.open-meteo.com/v1/archive` | `latitude`, `longitude`, `start_date`, `end_date`, `daily=...`, `timezone=UTC` |
| **Place Name Geocoding**| `https://geocoding-api.open-meteo.com/v1/search`| `name`, `count=1`, `language=en`, `format=json` |

### 3.1 Variables Collected & Standard Units
- **Mean Daily Temperature (°C)**: `temperature_2m_mean`
- **Maximum Daily Temperature (°C)**: `temperature_2m_max`
- **Daily Precipitation Sum (mm)**: `precipitation_sum`
- **Mean Relative Humidity (%)**: `relative_humidity_2m_mean`
- **Reference Evapotranspiration ET₀ (mm)**: `et0_fao_evapotranspiration` (FAO-56 Penman-Monteith equation)

---

## 4. Calculation Methodology & Baseline Construction

### 4.1 Recent-Period Statistics
For a requested analysis window of $N$ days (default: 30 days):
- **$\text{Recent Mean Temperature } (T_{\text{recent}})$**: Arithmetic mean of valid daily mean temperatures.
- **$\text{Recent Max Temperature } (T_{\text{max}})$**: Maximum daily peak temperature during the window.
- **$\text{Recent Total Rainfall } (P_{\text{recent}})$**: Sum of all daily precipitation observations.
- **$\text{Recent Mean Humidity } (H_{\text{recent}})$**: Arithmetic mean of daily relative humidity.
- **$\text{Recent Total } \text{ET}_0$**: Cumulative reference evapotranspiration.

### 4.2 Historical Baseline Construction
To enable meaningful comparisons, the baseline is computed using **identical calendar windows across preceding historical years**:
1. For recent period ending on date $D_{\text{end}}$ (month $M$, day $D$):
2. For each historical year $Y \in \text{baseline\_years}$ (default: previous 3 years, e.g. $[2023, 2024, 2025]$):
   - The equivalent target end date is established: $\text{Date}(Y, M, D)$ (with Feb 29 mapped to Feb 28 in non-leap years).
   - The historical window begins $N-1$ days earlier, ensuring exactly $N$ days are queried.
3. For each baseline year $Y$, total rainfall $P_Y$, mean temperature $T_Y$, mean humidity $H_Y$, and cumulative $\text{ET}_{0, Y}$ are calculated.
4. **Baseline Values**:
   $$\text{Baseline Mean Temperature } (T_{\text{baseline}}) = \frac{1}{K}\sum_{Y=1}^{K} T_Y$$
   $$\text{Baseline Rainfall } (P_{\text{baseline}}) = \frac{1}{K}\sum_{Y=1}^{K} P_Y$$
   $$\text{Baseline Humidity } (H_{\text{baseline}}) = \frac{1}{K}\sum_{Y=1}^{K} H_Y$$
   $$\text{Baseline } \text{ET}_0 = \frac{1}{K}\sum_{Y=1}^{K} \text{ET}_{0, Y}$$

### 4.3 Anomaly Formulas
- **Temperature Anomaly (°C)**:
  $$\Delta T = T_{\text{recent}} - T_{\text{baseline}}$$
- **Rainfall Deviation (mm)**:
  $$\Delta P = P_{\text{recent}} - P_{\text{baseline}}$$
- **Percentage Rainfall Deviation (%)**:
  $$\% \Delta P = \begin{cases} \left(\frac{\Delta P}{P_{\text{baseline}}}\right) \times 100 & \text{if } P_{\text{baseline}} > 0 \\ \text{None} & \text{if } P_{\text{baseline}} = 0 \end{cases}$$

### 4.4 Handling Missing Data
- **No Silent Zeroes**: Missing values (`None` / `null`) are never replaced with zero.
- **Mean Variables**: Averages are computed using available valid observations. If days are missing, an explicit warning is added (e.g. `"Mean temperature: 2 missing day(s); calculated over 28 available days."`).
- **Cumulative Totals (Rainfall / ET₀)**: Sums are calculated from valid days, and a warning is raised notifying the user that the total may be an underestimate (e.g. `"Precipitation: 2 missing day(s); cumulative rainfall may be an underestimate."`).
- **Entirely Missing Series**: If all observations for a variable are missing, the aggregate returns `None` and an explanatory warning is registered.

---

## 5. Scientific Principles, Limitations & Assumptions

1. **Reanalysis vs. Live Sensor Readings**: Data retrieved from Open-Meteo's weather models (ERA5 / ECMWF / GFS) are reanalysis and forecast products. They provide regional estimates rather than direct on-farm microclimate sensor measurements.
2. **Local Anomaly vs. Climate Change**: Historical baseline departures reflect short-term localized weather deviations over a 3-year baseline. They do not constitute evidence or proof of long-term global climate change.
3. **Statistical Baseline vs. Agronomic Threshold**: Baseline statistics represent historical norms for that calendar period, not physiologically validated crop stress thresholds for specific cultivars.
4. **No Disease Causality Claims**: High humidity or unseasonable rainfall may represent environmental conditions favorable to pathogens, but this engine does **not** assert disease causality or compute arbitrary disease risk scores.

---

## 6. Manually Verifiable Numerical Examples (Synthetic Fixtures)

The following three examples demonstrate the exact calculation logic using controlled synthetic test fixtures (labelled as synthetic fixtures, not real observations):

### Fixture 1: Warmer and Wetter Period (Positive Anomalies)
* **Synthetic Test Scenario**: Monsoon extension with elevated temperatures.
* **Period**: 3 days.
* **Recent Observations**:
  * Day 1: 26.0°C, 10.0 mm
  * Day 2: 28.0°C, 30.0 mm
  * Day 3: 30.0°C, 80.0 mm
  * $\text{Recent Mean Temperature} = \frac{26.0 + 28.0 + 30.0}{3} = \mathbf{28.0^\circ\text{C}}$
  * $\text{Recent Total Rainfall} = 10.0 + 30.0 + 80.0 = \mathbf{120.0\text{ mm}}$
* **Baseline Historical Years**:
  * Year 1: Mean Temp = 25.0°C, Total Rainfall = 90.0 mm
  * Year 2: Mean Temp = 26.0°C, Total Rainfall = 110.0 mm
  * $\text{Baseline Mean Temperature} = \frac{25.0 + 26.0}{2} = \mathbf{25.5^\circ\text{C}}$
  * $\text{Baseline Rainfall} = \frac{90.0 + 110.0}{2} = \mathbf{100.0\text{ mm}}$
* **Calculated Anomalies**:
  * $\text{Temperature Anomaly} = 28.0 - 25.5 = \mathbf{+2.5^\circ\text{C}}$
  * $\text{Rainfall Deviation} = 120.0 - 100.0 = \mathbf{+20.0\text{ mm}}$
  * $\text{Rainfall Percentage} = \left(\frac{+20.0}{100.0}\right) \times 100 = \mathbf{+20.0\%}$

---

### Fixture 2: Cooler and Drier Period (Drought Stress)
* **Synthetic Test Scenario**: Unseasonable dry spell and cold front.
* **Period**: 3 days.
* **Recent Observations**:
  * Day 1: 20.0°C, 10.0 mm
  * Day 2: 21.0°C, 15.0 mm
  * Day 3: 22.0°C, 10.0 mm
  * $\text{Recent Mean Temperature} = \frac{20.0 + 21.0 + 22.0}{3} = \mathbf{21.0^\circ\text{C}}$
  * $\text{Recent Total Rainfall} = 10.0 + 15.0 + 10.0 = \mathbf{35.0\text{ mm}}$
* **Baseline Historical Years**:
  * Year 1: Mean Temp = 23.0°C, Total Rainfall = 60.0 mm
  * Year 2: Mean Temp = 24.0°C, Total Rainfall = 80.0 mm
  * $\text{Baseline Mean Temperature} = \frac{23.0 + 24.0}{2} = \mathbf{23.5^\circ\text{C}}$
  * $\text{Baseline Rainfall} = \frac{60.0 + 80.0}{2} = \mathbf{70.0\text{ mm}}$
* **Calculated Anomalies**:
  * $\text{Temperature Anomaly} = 21.0 - 23.5 = \mathbf{-2.5^\circ\text{C}}$
  * $\text{Rainfall Deviation} = 35.0 - 70.0 = \mathbf{-35.0\text{ mm}}$
  * $\text{Rainfall Percentage} = \left(\frac{-35.0}{70.0}\right) \times 100 = \mathbf{-50.0\%}$

---

### Fixture 3: Arid Baseline (Zero Baseline Rainfall & Division Guard)
* **Synthetic Test Scenario**: Arid pre-monsoon season where historical baseline had zero precipitation.
* **Period**: 3 days.
* **Recent Observations**:
  * Day 1: 31.0°C, 5.0 mm
  * Day 2: 32.0°C, 10.0 mm
  * Day 3: 33.0°C, 0.0 mm
  * $\text{Recent Mean Temperature} = \frac{31.0 + 32.0 + 33.0}{3} = \mathbf{32.0^\circ\text{C}}$
  * $\text{Recent Total Rainfall} = 5.0 + 10.0 + 0.0 = \mathbf{15.0\text{ mm}}$
* **Baseline Historical Years**:
  * Year 1: Mean Temp = 31.0°C, Total Rainfall = 0.0 mm
  * Year 2: Mean Temp = 31.0°C, Total Rainfall = 0.0 mm
  * $\text{Baseline Mean Temperature} = \mathbf{31.0^\circ\text{C}}$
  * $\text{Baseline Rainfall} = \mathbf{0.0\text{ mm}}$
* **Calculated Anomalies**:
  * $\text{Temperature Anomaly} = 32.0 - 31.0 = \mathbf{+1.0^\circ\text{C}}$
  * $\text{Rainfall Deviation} = 15.0 - 0.0 = \mathbf{+15.0\text{ mm}}$
  * $\text{Rainfall Percentage} = \mathbf{None}$ *(Division by zero protected; warning: `"Baseline rainfall is zero; percentage rainfall deviation is undefined."`)*

---

## 7. CSV Export Functionality & Data Dictionary

Results can be exported to standard CSV files inside `data/climate/`:

```python
from src.cropcare.climate_engine import export_climate_csv

filepath = export_climate_csv(result, output_dir="data/climate")
```

Or directly via `get_environment(..., export_csv=True)`.

### 7.1 CSV Columns & Definitions
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `date` | String | Observation date in ISO format (`YYYY-MM-DD`). |
| `location` | String | Resolved location name or formatted latitude/longitude. |
| `source` | String | Data source attribution (`Open-Meteo`). |
| `temperature_mean_c` | Float | Mean daily 2m temperature (°C). |
| `temperature_max_c` | Float | Maximum peak daily 2m temperature (°C). |
| `precipitation_mm` | Float | Daily precipitation sum (mm). |
| `relative_humidity_pct` | Float | Mean daily relative humidity (%). |
| `et0_mm` | Float | Daily reference evapotranspiration ET₀ (mm). |
| `baseline_mean_temperature_c` | Float | Historical multi-year baseline mean temperature for the period (°C). |
| `baseline_rainfall_mm` | Float | Historical multi-year baseline total rainfall for the period (mm). |
| `temperature_anomaly_c` | Float | Period temperature anomaly ($\Delta T = T_{\text{recent}} - T_{\text{baseline}}$). |
| `rainfall_deviation_mm` | Float | Period rainfall deviation ($\Delta P = P_{\text{recent}} - P_{\text{baseline}}$). |

---

## 8. Unit Test Suite & Execution Results

Unit tests are implemented in `tests/test_climate_engine.py`. Tests execute **100% offline** with mocked network fixtures, guaranteeing fast, deterministic execution without dependence on external APIs or internet connectivity.

### 8.1 Test Classes & Coverage
1. **`TestLocationResolution`**:
   - `test_valid_tuple_coords`: Validates tuple parsing and bounds.
   - `test_valid_dict_coords`: Validates dictionary extraction with standard keys.
   - `test_valid_dict_coords_alternative_keys`: Validates fallback to `'latitude'` / `'longitude'`.
   - `test_valid_string_coords`: Validates comma-separated coordinate strings.
   - `test_invalid_latitude_range`: Validates bounds check for latitude > 90.
   - `test_invalid_longitude_range`: Validates bounds check for longitude > 180.
   - `test_invalid_string_format`: Validates graceful non-crash on corrupt strings.
   - `test_geocoding_success`: Mocks Open-Meteo Geocoding API response parsing.
   - `test_geocoding_not_found`: Validates warning when no matching place is found.
   - `test_geocoding_network_failure`: Validates network failure during geocoding.
2. **`TestDailyParsingAndAggregation`**:
   - `test_valid_daily_parsing`: Validates array transposition into records.
   - `test_parsing_with_none_and_nan`: Ensures `NaN` and `None` are preserved safely.
   - `test_period_stats_computation`: Validates standard means, maxes, and sums.
   - `test_period_stats_partial_missing_observations`: Validates warnings and averages over available days.
   - `test_period_stats_completely_missing`: Validates return of `None` when observations are absent.
3. **`TestAnomalyCalculations`**:
   - `test_manually_verifiable_synthetic_example_1_positive_anomalies`: Verifies Fixture 1.
   - `test_manually_verifiable_synthetic_example_2_negative_anomalies`: Verifies Fixture 2.
   - `test_manually_verifiable_synthetic_example_3_zero_baseline_rainfall`: Verifies Fixture 3.
   - `test_missing_recent_or_baseline_returns_none`: Verifies graceful `None` on missing baseline.
4. **`TestHistoricalBaselineRetrieval`**:
   - `test_baseline_multi_year_averaging`: Validates multi-year historical averaging.
   - `test_baseline_insufficient_data`: Validates graceful warning on complete baseline failure.
5. **`TestGetEnvironmentPublicInterface`**:
   - `test_full_pipeline_success`: End-to-end integration test with mocked payloads.
   - `test_api_timeout_handling`: Validates resilience under network timeouts.
   - `test_malformed_json_response`: Validates handling of invalid JSON syntax.
   - `test_missing_daily_key_in_payload`: Validates handling of missing daily key.
   - `test_invalid_coordinates_handling`: Validates out-of-bounds coordinates handling.
   - `test_insufficient_baseline_data`: Validates recent data returned even if baseline fails.
6. **`TestCsvExport`**:
   - `test_export_climate_csv_structure`: Validates CSV creation, header names, and row values.

### 8.2 Actual Test Execution Output
```text
C:\Users\aishw\CropAnalysisMiniHack\CropCare> python -m unittest tests/test_climate_engine.py
............................
----------------------------------------------------------------------
Ran 28 tests in 1.228s

OK
```

All **28 test cases** passed with 0 failures and 0 errors.

---

## 9. Integration Instructions for Team

### 9.1 How to Run Tests
From the root of the repository:
```bash
python -m unittest tests/test_climate_engine.py
```
Or with discovery:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 9.2 How Member 3 (UI / Integration) Invokes the Module
```python
from src.cropcare.climate_engine import get_environment

# Example A: By City Name
climate_data = get_environment("Nashik, India", days=30)

# Example B: By Coordinates
climate_data = get_environment(location=(18.5204, 73.8567), days=30)

# Accessing recent summary
mean_temp = climate_data["recent"]["mean_temperature_c"]
total_rain = climate_data["recent"]["rainfall_mm"]

# Accessing anomalies
temp_diff = climate_data["anomalies"]["temperature_c"]
rain_diff = climate_data["anomalies"]["rainfall_mm"]
rain_pct = climate_data["anomalies"]["rainfall_pct"]

# Accessing daily time-series for UI charting
chart_series = climate_data["daily_series"]
for day in chart_series:
    print(day["date"], day["temperature_mean_c"], day["precipitation_mm"])
```

### 9.3 How to Export a CSV
```python
from src.cropcare.climate_engine import get_environment, export_climate_csv

# Option 1: Automatic export during retrieval
data = get_environment("Pune", days=30, export_csv=True)
print("Saved to:", data.get("csv_path"))

# Option 2: Explicit export from an existing result dict
filepath = export_climate_csv(data, output_dir="data/climate", filename="pune_oct2026.csv")
```
