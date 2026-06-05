from __future__ import annotations

from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from src.aq_course_ml.config import (
    RAIN_ALERT_THRESHOLD_MM,
    REGRESSION_MODEL_PATH,
)
from src.aq_course_ml.predict_by_time import build_prediction_features


st.set_page_config(page_title="Rainfall Forecast OSEL", layout="wide")


@st.cache_resource
def load_model():
    if not REGRESSION_MODEL_PATH.exists():
        return None
    return joblib.load(REGRESSION_MODEL_PATH)


@st.cache_data(ttl=3600)
def build_features_for_datetime(target_datetime: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_prediction_features(target_datetime)


def show_prediction(input_frame: pd.DataFrame, context_frame: pd.DataFrame | None = None) -> None:
    if model is None:
        return

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

    if context_frame is not None:
        with st.expander("Weather data used for this prediction"):
            st.dataframe(context_frame.tail(13), use_container_width=True)


model = load_model()

st.title("Rainfall Forecast with OSEL")

if model is None:
    st.warning("Model file not found. Run `python run_pipeline.py` before using the app.")

input_frame: pd.DataFrame | None = None
context_frame: pd.DataFrame | None = None

st.subheader("Forecast Time")
date_col, hour_col = st.columns([1, 1])
default_timestamp = pd.Timestamp.now(tz="Asia/Bangkok").floor("h").to_pydatetime()
with date_col:
    selected_date = st.date_input("Date", value=default_timestamp.date())
with hour_col:
    selected_hour = st.selectbox(
        "Hour",
        list(range(24)),
        index=default_timestamp.hour,
        format_func=lambda hour: f"{hour:02d}:00",
    )

target_datetime = datetime.combine(selected_date, datetime.min.time()).replace(hour=int(selected_hour))
if st.button("Predict rainfall", type="primary"):
    try:
        input_frame, context_frame = build_features_for_datetime(target_datetime)
    except Exception as exc:
        st.error(f"Could not build prediction input: {exc}")

if input_frame is not None and model is not None:
    show_prediction(input_frame, context_frame)
