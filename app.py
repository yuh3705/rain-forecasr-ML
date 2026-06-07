from __future__ import annotations

from datetime import datetime
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.aq_course_ml.config import (
    MODEL_DIR,
    RAIN_ALERT_THRESHOLD_MM,
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
def load_model(model_path: str):
    path = MODEL_DIR / model_path
    if not path.exists():
        return None
    register_model_compatibility()
    return joblib.load(path)


def available_model_names() -> list[str]:
    return sorted(path.name for path in MODEL_DIR.glob("*.joblib"))


@st.cache_data(ttl=3600)
def build_features_for_datetime(target_datetime: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_prediction_features(target_datetime)


def show_prediction(
    selected_model,
    input_frame: pd.DataFrame,
    context_frame: pd.DataFrame | None = None,
) -> None:
    if selected_model is None:
        return

    try:
        predicted_rain = float(selected_model.predict(input_frame)[0])
    except Exception as exc:
        st.error(f"Selected model is not compatible with the current prediction features: {exc}")
        return
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


st.title("Rainfall Forecast")

model_names = available_model_names()
if not model_names:
    st.warning("No model files found in `Notebooks/models`. Run the notebooks or training pipeline first.")
    st.stop()

default_model_name = "osel_regressor.joblib"
default_model_index = model_names.index(default_model_name) if default_model_name in model_names else 0
selected_model_name = st.selectbox(
    "Model",
    model_names,
    index=default_model_index,
)
model = load_model(selected_model_name)

if model is None:
    st.warning(f"Model file not found: `Notebooks/models/{selected_model_name}`.")
    st.stop()

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
    show_prediction(model, input_frame, context_frame)
