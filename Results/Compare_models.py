from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.aq_course_ml.config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, PROCESSED_DATA_PATH, RANDOM_STATE, REGRESSION_TARGET
from src.aq_course_ml.train_model import build_preprocessor


RESULT_DIR = Path(__file__).resolve().parent
COMPARISON_CSV_PATH = RESULT_DIR / "model_comparison.csv"
COMPARISON_PLOT_PATH = RESULT_DIR / "comparison_mae_rmse_r2.png"
TRAIN_START = "2020-01-01"
TRAIN_END = "2025-12-31 23:59:59"
TEST_START = "2026-01-01"


MODEL_SPECS = {
    "Ridge": {
        "estimator": Ridge(),
        "param_grid": {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
        "train_limit": None,
    },
    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "param_grid": {
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_leaf": [1, 3, 6],
            "model__min_samples_split": [2, 8, 16],
        },
        "train_limit": None,
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [120, 180],
            "model__max_depth": [10, 14, None],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "train_limit": None,
    },
    "Extra Trees": {
        "estimator": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [120, 180],
            "model__max_depth": [10, 14, None],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "train_limit": None,
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [120, 180],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_depth": [2, 3],
        },
        "train_limit": None,
    },
    "AdaBoost": {
        "estimator": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        ),
        "param_grid": {
            "model__n_estimators": [80, 120],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__estimator__max_depth": [4, 6, 8],
            "model__estimator__min_samples_leaf": [2, 4],
        },
        "train_limit": None,
    },
    "KNN": {
        "estimator": KNeighborsRegressor(),
        "param_grid": {"model__n_neighbors": [6, 12, 24], "model__weights": ["uniform", "distance"]},
        "train_limit": 10000,
    },
    "SVR": {
        "estimator": SVR(kernel="rbf"),
        "param_grid": {"model__C": [1.0, 10.0], "model__epsilon": [0.05, 0.1], "model__gamma": ["scale", "auto"]},
        "train_limit": 5000,
    },
}


def limit_train_rows(train_df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(train_df) <= limit:
        return train_df
    step = max(1, len(train_df) // limit)
    return train_df.iloc[::step].tail(limit).copy()


def evaluate_model(name: str, spec: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    search_df = limit_train_rows(train_df, spec["train_limit"])
    base_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", spec["estimator"]),
        ]
    )
    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=spec["param_grid"],
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=4),
        n_jobs=-1,
        refit=True,
    )
    x_search = search_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y_search = search_df[REGRESSION_TARGET]
    search.fit(x_search, y_search)

    model = search.best_estimator_
    x_train = train_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y_train = train_df[REGRESSION_TARGET]
    x_test = test_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y_test = test_df[REGRESSION_TARGET]
    model.fit(x_train, y_train)
    predictions = pd.Series(model.predict(x_test), index=test_df.index).clip(lower=0)
    return {
        "model": name,
        "grid_search_rows": int(len(search_df)),
        "best_params": search.best_params_,
        "best_cv_neg_mae": float(search.best_score_),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }


def save_plot(results: pd.DataFrame) -> None:
    axes = results.set_index("model")[["mae", "rmse", "r2"]].plot(kind="bar", figsize=(11, 5), rot=25)
    axes.set_title("2026 Test Comparison After Time-Series Grid Search")
    axes.set_xlabel("Model")
    axes.set_ylabel("Score")
    axes.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=150)
    plt.close()


def main() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}. Run preprocess first.")

    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])
    train_df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)].copy()
    test_df = df[df["timestamp"] >= TEST_START].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Crawl data from 2020-01-01 through the current date first.")

    results = pd.DataFrame([evaluate_model(name, spec, train_df, test_df) for name, spec in MODEL_SPECS.items()])
    results.to_csv(COMPARISON_CSV_PATH, index=False)
    save_plot(results)
    print(results[["model", "grid_search_rows", "mae", "rmse", "r2"]].to_string(index=False))
    print(f"Saved comparison CSV: {COMPARISON_CSV_PATH}")
    print(f"Saved comparison plot: {COMPARISON_PLOT_PATH}")
    return results


if __name__ == "__main__":
    main()
