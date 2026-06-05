from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .config import (
    BASE_COLUMNS,
    FEATURE_COLUMNS,
    LOCATIONS,
    OPEN_METEO_HOURLY_VARIABLES,
)
from .preprocess import add_lag_features, add_time_features


WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LOCAL_TIMEZONE = "Asia/Bangkok"
LAG_LOOKBACK_HOURS = 12


def _fetch_hourly_json(url: str, params: dict[str, Any]) -> pd.DataFrame:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly:
        raise ValueError("Open-Meteo did not return hourly weather data.")
    return pd.DataFrame(hourly)


def _build_api_params(location: dict[str, Any], start_date: str, end_date: str) -> dict[str, Any]:
    return {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": LOCAL_TIMEZONE,
        "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
    }


def _weather_endpoint(target_timestamp: pd.Timestamp) -> str:
    today = pd.Timestamp.now(tz=LOCAL_TIMEZONE).date()
    return WEATHER_FORECAST_URL if target_timestamp.date() >= today else WEATHER_ARCHIVE_URL


def fetch_weather_window(target_timestamp: datetime | pd.Timestamp) -> pd.DataFrame:
    target = pd.Timestamp(target_timestamp).floor("h")
    start = target - timedelta(hours=LAG_LOOKBACK_HOURS)
    location = LOCATIONS[0]

    url = _weather_endpoint(target)
    weather_df = _fetch_hourly_json(
        url,
        _build_api_params(
            location=location,
            start_date=start.date().isoformat(),
            end_date=target.date().isoformat(),
        ),
    )
    weather_df = weather_df.rename(columns={"time": "timestamp"})
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["city"] = location["city"]
    weather_df["latitude"] = location["latitude"]
    weather_df["longitude"] = location["longitude"]
    weather_df = weather_df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    weather_df = weather_df[weather_df["timestamp"] <= target].reset_index(drop=True)
    return weather_df


def build_prediction_features(target_timestamp: datetime | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.Timestamp(target_timestamp).floor("h")
    weather_df = fetch_weather_window(target)

    for column in BASE_COLUMNS:
        weather_df[column] = pd.to_numeric(weather_df[column], errors="coerce")
        weather_df[column] = weather_df.groupby("city")[column].transform(
            lambda series: series.interpolate(limit_direction="both")
        )
        weather_df[column] = weather_df[column].fillna(weather_df[column].median())

    features_df = add_time_features(weather_df.copy())
    features_df = add_lag_features(features_df)
    features_df = features_df.drop(columns=["city", "precipitation"], errors="ignore")

    exact_rows = features_df[features_df["timestamp"] == target]
    if exact_rows.empty:
        available_start = weather_df["timestamp"].min()
        available_end = weather_df["timestamp"].max()
        raise ValueError(
            "No hourly weather data found for "
            f"{target:%Y-%m-%d %H:%M}. Available range: "
            f"{available_start:%Y-%m-%d %H:%M} to {available_end:%Y-%m-%d %H:%M}."
        )

    input_frame = exact_rows.iloc[[0]][FEATURE_COLUMNS].copy()
    if input_frame.isna().any(axis=None):
        missing = input_frame.columns[input_frame.isna().any()].tolist()
        raise ValueError(f"Not enough weather history to build lag features: {', '.join(missing)}.")

    return input_frame, weather_df
