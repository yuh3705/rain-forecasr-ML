from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit, cross_val_score


class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking regressor that builds meta-features with time-series folds only."""

    def __init__(
        self,
        cv_splits: int = 3,
        random_state: int | None = None,
        n_jobs: int = -1,
        passthrough: bool = True,
        rf_n_estimators: int = 120,
        rf_max_depth: int | None = 12,
        rf_min_samples_leaf: int = 2,
        extra_n_estimators: int = 120,
        extra_max_depth: int | None = 12,
        extra_min_samples_leaf: int = 2,
        gbr_n_estimators: int = 120,
        gbr_learning_rate: float = 0.05,
        gbr_max_depth: int = 3,
        ridge_alpha: float = 1.0,
        final_alpha: float = 1.0,
    ) -> None:
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.extra_n_estimators = extra_n_estimators
        self.extra_max_depth = extra_max_depth
        self.extra_min_samples_leaf = extra_min_samples_leaf
        self.gbr_n_estimators = gbr_n_estimators
        self.gbr_learning_rate = gbr_learning_rate
        self.gbr_max_depth = gbr_max_depth
        self.ridge_alpha = ridge_alpha
        self.final_alpha = final_alpha

    def fit(self, X, y):
        X = self._as_array(X)
        y = np.asarray(y, dtype=float)
        base_models = self._build_base_models()
        oof_predictions = np.full((X.shape[0], len(base_models)), np.nan)
        splitter = TimeSeriesSplit(n_splits=self.cv_splits)

        for train_index, valid_index in splitter.split(X):
            for model_index, (_, estimator) in enumerate(base_models):
                fold_model = clone(estimator)
                fold_model.fit(X[train_index], y[train_index])
                oof_predictions[valid_index, model_index] = fold_model.predict(X[valid_index])

        meta_rows = ~np.isnan(oof_predictions).any(axis=1)
        if not np.any(meta_rows):
            raise ValueError("Not enough rows to build time-series stacking meta-features.")

        meta_X = self._combine_features(oof_predictions[meta_rows], X[meta_rows])
        self.final_estimator_ = Ridge(alpha=self.final_alpha)
        self.final_estimator_.fit(meta_X, y[meta_rows])

        self.base_models_ = []
        for name, estimator in base_models:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.base_models_.append((name, fitted))
        return self

    def predict(self, X):
        X = self._as_array(X)
        base_predictions = np.column_stack([model.predict(X) for _, model in self.base_models_])
        return self.final_estimator_.predict(self._combine_features(base_predictions, X))

    def _build_base_models(self):
        return [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=self.rf_n_estimators,
                    max_depth=self.rf_max_depth,
                    min_samples_leaf=self.rf_min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
            ),
            (
                "extra",
                ExtraTreesRegressor(
                    n_estimators=self.extra_n_estimators,
                    max_depth=self.extra_max_depth,
                    min_samples_leaf=self.extra_min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=self.gbr_n_estimators,
                    learning_rate=self.gbr_learning_rate,
                    max_depth=self.gbr_max_depth,
                    random_state=self.random_state,
                ),
            ),
            ("ridge", Ridge(alpha=self.ridge_alpha)),
        ]

    def _combine_features(self, base_predictions: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return np.hstack([base_predictions, X])
        return base_predictions

    @staticmethod
    def _as_array(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=float)


class OSELRegressor(BaseEstimator, RegressorMixin):
    """Optimized Stacking Ensemble Learning for time-series regression."""

    def __init__(
        self,
        n_iter: int = 8,
        cv_splits: int = 3,
        random_state: int | None = None,
        n_jobs: int = -1,
        non_negative: bool = True,
    ) -> None:
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.non_negative = non_negative

    def fit(self, X, y):
        X = TimeSeriesStackingRegressor._as_array(X)
        y = np.asarray(y, dtype=float)
        candidates = list(
            ParameterSampler(
                self._param_distributions(),
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        )
        splitter = TimeSeriesSplit(n_splits=self.cv_splits)
        best_score = -np.inf
        best_params: dict[str, float | int | None] = {}

        for params in candidates:
            candidate = TimeSeriesStackingRegressor(
                cv_splits=self.cv_splits,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **params,
            )
            scores = cross_val_score(
                candidate,
                X,
                y,
                scoring="neg_mean_absolute_error",
                cv=splitter,
                n_jobs=1,
            )
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_params = params

        self.best_estimator_ = TimeSeriesStackingRegressor(
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **best_params,
        )
        self.best_estimator_.fit(X, y)
        self.best_params_ = best_params
        self.best_score_ = best_score
        return self

    def predict(self, X):
        predictions = self.best_estimator_.predict(X)
        if self.non_negative:
            predictions = np.clip(predictions, 0.0, None)
        return predictions

    @staticmethod
    def _param_distributions() -> dict[str, list[float | int | None]]:
        return {
            "rf_n_estimators": [80, 120, 180],
            "rf_max_depth": [8, 12, 16, None],
            "rf_min_samples_leaf": [1, 2, 4],
            "extra_n_estimators": [80, 120, 180],
            "extra_max_depth": [8, 12, 16, None],
            "extra_min_samples_leaf": [1, 2, 4],
            "gbr_n_estimators": [80, 120, 180],
            "gbr_learning_rate": [0.03, 0.05, 0.08],
            "gbr_max_depth": [2, 3, 4],
            "ridge_alpha": [0.1, 1.0, 10.0],
            "final_alpha": [0.1, 1.0, 10.0],
        }
