from __future__ import annotations

from datetime import datetime
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.aq_course_ml.config import (
    RAIN_ALERT_THRESHOLD_MM,
    REGRESSION_MODEL_PATH,
)
from src.aq_course_ml.predict_by_time import build_prediction_features


st.set_page_config(page_title="Rainfall Forecast OSEL", layout="wide")


class TimeSeriesStackingRegressor:
    """Compatibility class for OSEL models saved from Notebooks/OSEL.ipynb."""

    def predict(self, X):
        base_predictions = np.column_stack([model.predict(X) for _, model in self.base_models_])
        return self.final_estimator_.predict(self._combine_features(base_predictions, X))

    def _combine_features(self, base_predictions, X):
        if self.passthrough:
            passthrough_features = self.passthrough_preprocessor_.transform(X)
            if hasattr(passthrough_features, "toarray"):
                passthrough_features = passthrough_features.toarray()
            return np.hstack([base_predictions, passthrough_features])
        return base_predictions


def register_model_compatibility() -> None:
    setattr(sys.modules["__main__"], "TimeSeriesStackingRegressor", TimeSeriesStackingRegressor)

    try:
        import sklearn.compose._column_transformer as column_transformer
    except ImportError:
        return

    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        column_transformer._RemainderColsList = _RemainderColsList


@st.cache_resource
def load_model():
    if not REGRESSION_MODEL_PATH.exists():
        return None
    register_model_compatibility()
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
