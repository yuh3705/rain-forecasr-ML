from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BASE_COLUMNS,
    FORECAST_HORIZON_HOURS,
    PROCESSED_DATA_PATH,
    PROCESSED_DIR,
    RAIN_ALERT_THRESHOLD_MM,
    RAW_DATA_PATH,
)
from .utils import ensure_dirs


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("city", group_keys=False)
    df["rain_lag_1h"] = grouped["rain"].shift(1)
    df["rain_lag_3h"] = grouped["rain"].shift(3)
    df["rain_roll_sum_6h"] = grouped["rain"].transform(lambda series: series.shift(1).rolling(6).sum())
    df["rain_roll_sum_12h"] = grouped["rain"].transform(lambda series: series.shift(1).rolling(12).sum())
    df["precipitation_lag_1h"] = grouped["precipitation"].shift(1)
    df["precipitation_roll_sum_6h"] = grouped["precipitation"].transform(
        lambda series: series.shift(1).rolling(6).sum()
    )
    df["temp_lag_1h"] = grouped["temperature_2m"].shift(1)
    df["humidity_lag_1h"] = grouped["relative_humidity_2m"].shift(1)
    df["dew_point_lag_1h"] = grouped["dew_point_2m"].shift(1)
    df["cloud_cover_lag_1h"] = grouped["cloud_cover"].shift(1)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("city", group_keys=False)
    target = grouped["rain"].transform(
        lambda series: series.shift(-1)
        .rolling(FORECAST_HORIZON_HOURS, min_periods=FORECAST_HORIZON_HOURS)
        .sum()
        .shift(-(FORECAST_HORIZON_HOURS - 1))
    )
    df["target_rain_next_6h"] = target
    df["target_rain_alert_6h"] = (target >= RAIN_ALERT_THRESHOLD_MM).astype(int)
    return df


def preprocess_dataset() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_DATA_PATH}. Run crawl_data first.")

    ensure_dirs(PROCESSED_DIR)
    df = pd.read_csv(RAW_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    for column in BASE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df.groupby("city")[column].transform(lambda series: series.interpolate(limit_direction="both"))
        df[column] = df[column].fillna(df[column].median())

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_targets(df)
    df = df.dropna().reset_index(drop=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset: {PROCESSED_DATA_PATH} ({len(df):,} rows)")
    return df


def main() -> None:
    preprocess_dataset()


if __name__ == "__main__":
    main()
