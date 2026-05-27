from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOK_DIR = PROJECT_DIR / "Notebooks"
MODEL_DIR = NOTEBOOK_DIR / "models"
REPORT_DIR = PROJECT_DIR / "Results"

RAW_DATA_PATH = RAW_DIR / "rain_weather_hourly.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "rainfall_features.csv"
METRICS_PATH = REPORT_DIR / "metrics.json"
REGRESSION_MODEL_PATH = MODEL_DIR / "rainfall_osel_regressor.joblib"
PREDICTION_PLOT_PATH = REPORT_DIR / "rainfall_predictions.png"

FORECAST_HORIZON_HOURS = 6
RAIN_ALERT_THRESHOLD_MM = 1.0
RANDOM_STATE = 42
OPEN_METEO_HOURLY_VARIABLES = [
    "rain",
    "precipitation",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
]

LOCATIONS = [
    {"city": "Ha Noi", "latitude": 21.0278, "longitude": 105.8342},
]

BASE_COLUMNS = [
    "rain",
    "precipitation",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
]

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "rain",
    "precipitation",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "hour",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "rain_lag_1h",
    "rain_lag_3h",
    "rain_roll_sum_6h",
    "rain_roll_sum_12h",
    "precipitation_lag_1h",
    "precipitation_roll_sum_6h",
    "temp_lag_1h",
    "humidity_lag_1h",
    "dew_point_lag_1h",
    "cloud_cover_lag_1h",
]

CATEGORICAL_COLUMNS = ["city"]
REGRESSION_TARGET = "target_rain_next_6h"
CLASSIFICATION_TARGET = "target_rain_alert_6h"
