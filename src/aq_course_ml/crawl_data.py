from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import requests

from .config import LOCATIONS, OPEN_METEO_HOURLY_VARIABLES, RAW_DATA_PATH, RAW_DIR
from .utils import ensure_dirs


WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl hourly rainfall and weather data from Open-Meteo.")
    parser.add_argument("--start-date", default="2025-05-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2025-10-31", help="End date in YYYY-MM-DD format.")
    return parser.parse_args()


def fetch_hourly_json(url: str, params: dict[str, Any]) -> pd.DataFrame:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly:
        raise ValueError(f"No hourly data returned from {url}")
    return pd.DataFrame(hourly)


def fetch_location(location: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    common_params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Bangkok",
    }
    weather_df = fetch_hourly_json(
        WEATHER_ARCHIVE_URL,
        {
            **common_params,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
        },
    )
    weather_df["city"] = location["city"]
    weather_df["latitude"] = location["latitude"]
    weather_df["longitude"] = location["longitude"]
    return weather_df


def crawl_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    ensure_dirs(RAW_DIR)
    frames = [fetch_location(location, start_date, end_date) for location in LOCATIONS]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset: {RAW_DATA_PATH} ({len(df):,} rows)")
    return df


def main() -> None:
    args = parse_args()
    crawl_dataset(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
