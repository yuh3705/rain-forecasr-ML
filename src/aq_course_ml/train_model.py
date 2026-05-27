from __future__ import annotations

import math

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_COLUMNS,
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    PREDICTION_PLOT_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAIN_ALERT_THRESHOLD_MM,
    REGRESSION_MODEL_PATH,
    REGRESSION_TARGET,
    REPORT_DIR,
)
from .osel import OSELRegressor
from .utils import ensure_dirs, write_json


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURE_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        sparse_threshold=0.0,
    )


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(df) * train_ratio)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def train_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Pipeline, dict[str, float], pd.Series]:
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                OSELRegressor(
                    n_iter=8,
                    cv_splits=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    non_negative=True,
                ),
            ),
        ]
    )
    x_train = train_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y_train = train_df[REGRESSION_TARGET]
    x_test = test_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y_test = test_df[REGRESSION_TARGET]

    model.fit(x_train, y_train)
    predictions = pd.Series(model.predict(x_test), index=test_df.index)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    return model, metrics, predictions


def evaluate_alert_from_regression(test_df: pd.DataFrame, rain_predictions: pd.Series) -> dict[str, object]:
    y_test = test_df[CLASSIFICATION_TARGET]
    predictions = (rain_predictions >= RAIN_ALERT_THRESHOLD_MM).astype(int)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics


def save_prediction_plot(test_df: pd.DataFrame, predictions: pd.Series) -> None:
    preview = test_df[["timestamp", REGRESSION_TARGET]].copy()
    preview["predicted_rain_next_6h"] = predictions.values
    preview = preview.sort_values("timestamp").head(300)

    plt.figure(figsize=(12, 5))
    plt.plot(preview["timestamp"], preview[REGRESSION_TARGET], label="Actual rain next 6h")
    plt.plot(preview["timestamp"], preview["predicted_rain_next_6h"], label="Predicted rain next 6h")
    plt.xlabel("Time")
    plt.ylabel("Rain (mm)")
    plt.title("Next-6-Hour Rainfall Forecast with OSEL on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=150)
    plt.close()


def train_models() -> dict[str, object]:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}. Run preprocess first.")

    ensure_dirs(MODEL_DIR, REPORT_DIR)
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])
    train_df, test_df = time_split(df)

    regressor, regression_metrics, regression_predictions = train_regressor(train_df, test_df)
    alert_metrics = evaluate_alert_from_regression(test_df, regression_predictions)

    joblib.dump(regressor, REGRESSION_MODEL_PATH)
    save_prediction_plot(test_df, regression_predictions)

    metrics = {
        "rows": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "regression_rain_next_6h_osel": regression_metrics,
        "rain_alert_from_regression_6h": alert_metrics,
        "osel_best_params": regressor.named_steps["model"].best_params_,
        "osel_best_cv_neg_mae": regressor.named_steps["model"].best_score_,
    }
    write_json(METRICS_PATH, metrics)
    print(f"Saved OSEL rainfall regression model: {REGRESSION_MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    return metrics


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
