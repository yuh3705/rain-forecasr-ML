from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.aq_course_ml.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    METRICS_PATH,
    PREDICTION_PLOT_PATH,
    PROCESSED_DATA_PATH,
    RAIN_ALERT_THRESHOLD_MM,
    REGRESSION_MODEL_PATH,
)


st.set_page_config(page_title="Rainfall Forecast OSEL", layout="wide")


@st.cache_resource
def load_model():
    if not REGRESSION_MODEL_PATH.exists():
        return None
    return joblib.load(REGRESSION_MODEL_PATH)


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])


def default_feature_values(df: pd.DataFrame) -> dict[str, float | str]:
    if df.empty:
        defaults: dict[str, float | str] = {column: 0.0 for column in FEATURE_COLUMNS}
        for column in CATEGORICAL_COLUMNS:
            defaults[column] = "Ha Noi"
        return defaults
    latest = df.sort_values("timestamp").iloc[-1]
    return {column: latest[column] for column in FEATURE_COLUMNS + CATEGORICAL_COLUMNS}


def build_input_frame(values: dict[str, float | str]) -> pd.DataFrame:
    return pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS + CATEGORICAL_COLUMNS}])


def parse_pasted_row(raw_text: str) -> dict[str, float | str]:
    values = [item.strip() for item in raw_text.split(",")]
    expected = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    if len(values) != len(expected):
        raise ValueError(f"Expected {len(expected)} values, received {len(values)}.")

    parsed: dict[str, float | str] = {}
    for column, value in zip(expected, values):
        parsed[column] = value if column in CATEGORICAL_COLUMNS else float(value)
    return parsed


model = load_model()
df = load_processed_data()
defaults = default_feature_values(df)

st.title("Rainfall Forecast with OSEL")

if model is None:
    st.warning("Model file not found. Run `python run_pipeline.py` before using the app.")

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Mode", ["Manual", "Paste CSV row"], horizontal=True)

input_values: dict[str, float | str] | None = None

if input_mode == "Manual":
    st.subheader("Weather Features")
    cols = st.columns(4)
    manual_values: dict[str, float | str] = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        with cols[index % 4]:
            manual_values[column] = st.number_input(
                column,
                value=float(defaults[column]),
                format="%.4f",
                key=f"manual_{column}",
            )
    for column in CATEGORICAL_COLUMNS:
        manual_values[column] = st.text_input(column, value=str(defaults[column]), key=f"manual_{column}")
    if st.button("Predict"):
        input_values = manual_values
else:
    st.subheader("Paste Feature Row")
    expected_order = ", ".join(FEATURE_COLUMNS + CATEGORICAL_COLUMNS)
    st.code(expected_order, language="text")
    raw_text = st.text_area("CSV values", height=120)
    if st.button("Predict from row"):
        try:
            input_values = parse_pasted_row(raw_text)
        except ValueError as exc:
            st.error(str(exc))

if input_values is not None and model is not None:
    input_frame = build_input_frame(input_values)
    predicted_rain = float(model.predict(input_frame)[0])
    will_rain = predicted_rain >= RAIN_ALERT_THRESHOLD_MM

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Predicted rain in next 6h", f"{predicted_rain:.2f} mm")
    with right:
        if will_rain:
            st.error("Rain alert")
        else:
            st.success("No significant rain alert")

if METRICS_PATH.exists() or PREDICTION_PLOT_PATH.exists():
    st.markdown("---")
    metric_col, plot_col = st.columns([1, 2])
    with metric_col:
        if METRICS_PATH.exists():
            st.subheader("Metrics")
            st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    with plot_col:
        if PREDICTION_PLOT_PATH.exists():
            st.subheader("Prediction Plot")
            st.image(str(PREDICTION_PLOT_PATH))
