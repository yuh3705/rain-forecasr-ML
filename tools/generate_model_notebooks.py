from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_DIR / "Notebooks"


MODEL_SPECS = {
    "Ridge": {
        "title": "Ridge Regression",
        "imports": "from sklearn.linear_model import Ridge",
        "estimator": "Ridge(alpha=1.0)",
        "param_grid": "{'model__alpha': [0.1, 1.0, 10.0, 50.0]}",
        "train_limit": "None",
    },
    "Decision_Tree": {
        "title": "Decision Tree Regressor",
        "imports": "from sklearn.tree import DecisionTreeRegressor",
        "estimator": "DecisionTreeRegressor(random_state=RANDOM_STATE)",
        "param_grid": (
            "{'model__max_depth': [6, 10, 14, None], "
            "'model__min_samples_leaf': [1, 3, 6], "
            "'model__min_samples_split': [2, 8, 16]}"
        ),
        "train_limit": "None",
    },
    "Random_Forest": {
        "title": "Random Forest Regressor",
        "imports": "from sklearn.ensemble import RandomForestRegressor",
        "estimator": "RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)",
        "param_grid": (
            "{'model__n_estimators': [120, 180], "
            "'model__max_depth': [10, 14, None], "
            "'model__min_samples_leaf': [1, 2, 4]}"
        ),
        "train_limit": "None",
    },
    "Extra_Trees": {
        "title": "Extra Trees Regressor",
        "imports": "from sklearn.ensemble import ExtraTreesRegressor",
        "estimator": "ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)",
        "param_grid": (
            "{'model__n_estimators': [120, 180], "
            "'model__max_depth': [10, 14, None], "
            "'model__min_samples_leaf': [1, 2, 4]}"
        ),
        "train_limit": "None",
    },
    "Gradient_Boosting": {
        "title": "Gradient Boosting Regressor",
        "imports": "from sklearn.ensemble import GradientBoostingRegressor",
        "estimator": "GradientBoostingRegressor(random_state=RANDOM_STATE)",
        "param_grid": (
            "{'model__n_estimators': [120, 180], "
            "'model__learning_rate': [0.03, 0.05, 0.08], "
            "'model__max_depth': [2, 3]}"
        ),
        "train_limit": "None",
    },
    "AdaBoost": {
        "title": "AdaBoost Regressor",
        "imports": "from sklearn.ensemble import AdaBoostRegressor\nfrom sklearn.tree import DecisionTreeRegressor",
        "estimator": (
            "AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=RANDOM_STATE), "
            "random_state=RANDOM_STATE)"
        ),
        "param_grid": (
            "{'model__n_estimators': [80, 120], "
            "'model__learning_rate': [0.03, 0.05, 0.1], "
            "'model__estimator__max_depth': [4, 6, 8], "
            "'model__estimator__min_samples_leaf': [2, 4]}"
        ),
        "train_limit": "None",
    },
    "KNN": {
        "title": "KNN Regressor",
        "imports": "from sklearn.neighbors import KNeighborsRegressor",
        "estimator": "KNeighborsRegressor()",
        "param_grid": "{'model__n_neighbors': [6, 12, 24], 'model__weights': ['uniform', 'distance']}",
        "train_limit": "10000",
    },
    "SVR": {
        "title": "Support Vector Regressor",
        "imports": "from sklearn.svm import SVR",
        "estimator": "SVR(kernel='rbf')",
        "param_grid": "{'model__C': [1.0, 10.0], 'model__epsilon': [0.05, 0.1], 'model__gamma': ['scale', 'auto']}",
        "train_limit": "5000",
    },
    "XGBoost": {
        "title": "XGBoost Regressor",
        "imports": "from xgboost import XGBRegressor",
        "estimator": (
            "XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, "
            "n_jobs=-1, tree_method='hist')"
        ),
        "param_grid": (
            "{'model__n_estimators': [200, 400], "
            "'model__max_depth': [3, 5], "
            "'model__learning_rate': [0.03, 0.05], "
            "'model__subsample': [0.8, 1.0], "
            "'model__colsample_bytree': [0.8, 1.0]}"
        ),
        "train_limit": "None",
    },
    "LightGBM": {
        "title": "LightGBM Regressor",
        "imports": "from lightgbm import LGBMRegressor",
        "estimator": (
            "LGBMRegressor(objective='regression', random_state=RANDOM_STATE, "
            "n_jobs=-1, verbosity=-1)"
        ),
        "param_grid": (
            "{'model__n_estimators': [200, 400], "
            "'model__num_leaves': [31, 63], "
            "'model__learning_rate': [0.03, 0.05], "
            "'model__subsample': [0.8, 1.0], "
            "'model__colsample_bytree': [0.8, 1.0]}"
        ),
        "train_limit": "None",
    },
}


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def notebook_for_model(file_stem: str, spec: dict[str, str]) -> dict[str, object]:
    setup = f"""from pathlib import Path
import json
import math
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
{spec["imports"]}

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'Notebooks' else Path.cwd()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.aq_course_ml.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAIN_ALERT_THRESHOLD_MM,
    REGRESSION_TARGET,
    REPORT_DIR,
)
from src.aq_course_ml.train_model import build_preprocessor

TRAIN_START = '2020-01-01'
TRAIN_END = '2025-12-31 23:59:59'
TEST_START = '2026-01-01'
MODEL_NAME = '{spec["title"]}'
MODEL_FILE = MODEL_DIR / '{file_stem.lower()}_regressor.joblib'
METRICS_FILE = REPORT_DIR / '{file_stem.lower()}_metrics.json'
PLOT_FILE = REPORT_DIR / '{file_stem.lower()}_predictions.png'
PARAM_GRID = {spec["param_grid"]}
TRAIN_LIMIT = {spec["train_limit"]}
"""
    load_split = """df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['timestamp'])
train_df = df[(df['timestamp'] >= TRAIN_START) & (df['timestamp'] <= TRAIN_END)].copy()
test_df = df[df['timestamp'] >= TEST_START].copy()

print('Rows:', {'total': len(df), 'train': len(train_df), 'test': len(test_df)})
print('Train:', train_df['timestamp'].min(), '->', train_df['timestamp'].max())
print('Test:', test_df['timestamp'].min(), '->', test_df['timestamp'].max())

if train_df.empty or test_df.empty:
    raise ValueError('Train/test split is empty. Crawl data from 2020-01-01 through the current date first.')

x_train = train_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
y_train = train_df[REGRESSION_TARGET]
x_test = test_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
y_test = test_df[REGRESSION_TARGET]

if TRAIN_LIMIT is not None and len(train_df) > TRAIN_LIMIT:
    step = max(1, len(train_df) // TRAIN_LIMIT)
    search_df = train_df.iloc[::step].tail(TRAIN_LIMIT).copy()
else:
    search_df = train_df

x_train_search = search_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
y_train_search = search_df[REGRESSION_TARGET]
print('Rows used for grid search:', len(search_df))
"""
    grid_search = f"""base_pipeline = Pipeline([
    ('preprocess', build_preprocessor()),
    ('model', {spec["estimator"]}),
])

tscv = TimeSeriesSplit(n_splits=4)
search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=PARAM_GRID,
    scoring='neg_mean_absolute_error',
    cv=tscv,
    n_jobs=-1,
    refit=True,
)
search.fit(x_train_search, y_train_search)

best_params = search.best_params_
best_cv_neg_mae = float(search.best_score_)
best_params, best_cv_neg_mae
"""
    final_train = """model = search.best_estimator_
model.fit(x_train, y_train)
predictions = pd.Series(model.predict(x_test), index=test_df.index).clip(lower=0)

regression_metrics = {
    'mae': float(mean_absolute_error(y_test, predictions)),
    'rmse': float(math.sqrt(mean_squared_error(y_test, predictions))),
    'r2': float(r2_score(y_test, predictions)),
}
regression_metrics
"""
    alert = """alert_predictions = (predictions >= RAIN_ALERT_THRESHOLD_MM).astype(int)
alert_metrics = {
    'accuracy': float(accuracy_score(test_df['target_rain_alert_6h'], alert_predictions)),
    'f1': float(f1_score(test_df['target_rain_alert_6h'], alert_predictions, zero_division=0)),
    'confusion_matrix': confusion_matrix(test_df['target_rain_alert_6h'], alert_predictions, labels=[0, 1]).tolist(),
    'classification_report': classification_report(
        test_df['target_rain_alert_6h'],
        alert_predictions,
        labels=[0, 1],
        zero_division=0,
        output_dict=True,
    ),
}
alert_metrics
"""
    save = """MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

preview = test_df[['timestamp', REGRESSION_TARGET]].copy()
preview['prediction'] = predictions.values
preview = preview.sort_values('timestamp').head(500)

plt.figure(figsize=(12, 5))
plt.plot(preview['timestamp'], preview[REGRESSION_TARGET], label='Actual rain next 6h')
plt.plot(preview['timestamp'], preview['prediction'], label='Predicted rain next 6h')
plt.xlabel('Time')
plt.ylabel('Rain (mm)')
plt.title(f'{MODEL_NAME}: 2026 Test Forecast')
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()

metrics = {
    'model': MODEL_NAME,
    'train_period': ['2020-01-01', '2025-12-31'],
    'test_period': ['2026-01-01', str(test_df['timestamp'].max())],
    'rows': {'total': int(len(df)), 'train': int(len(train_df)), 'test': int(len(test_df))},
    'grid_search_rows': int(len(search_df)),
    'best_params': best_params,
    'best_cv_neg_mae': best_cv_neg_mae,
    'regression': regression_metrics,
    'rain_alert': alert_metrics,
}
METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')

print('Saved model:', MODEL_FILE)
print('Saved metrics:', METRICS_FILE)
print('Saved plot:', PLOT_FILE)
"""
    return {
        "cells": [
            markdown_cell(
                f"# {spec['title']}\n\n"
                "Train tu 2020-01-01 den het 2025, tune bang GridSearchCV + TimeSeriesSplit, "
                "va test tu 2026-01-01 den ngay hien tai co trong dataset."
            ),
            code_cell(setup),
            markdown_cell("## Load Data And Date Split\n\nSplit theo thoi gian, khong shuffle."),
            code_cell(load_split),
            markdown_cell("## Grid Search With TimeSeriesSplit"),
            code_cell(grid_search),
            markdown_cell("## Refit Best Model And Test On 2026"),
            code_cell(final_train),
            markdown_cell("## Rain Alert Test"),
            code_cell(alert),
            markdown_cell("## Save Model, Metrics, And Plot"),
            code_cell(save),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for file_stem, spec in MODEL_SPECS.items():
        path = NOTEBOOK_DIR / f"{file_stem}.ipynb"
        path.write_text(json.dumps(notebook_for_model(file_stem, spec), indent=1), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
