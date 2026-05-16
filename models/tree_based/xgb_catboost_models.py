"""XGBoost and CatBoost demand forecasting models."""
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
from loguru import logger
from models.base_model import BaseModel


# ─── XGBoost ──────────────────────────────────────────────────────────────────

XGB_DEFAULT_PARAMS = {
    "objective":        "reg:absoluteerror",
    "eval_metric":      ["rmse", "mae"],
    "tree_method":      "hist",          # fast histogram algorithm
    "booster":          "dart",          # dart = dropout trees, reduces overfitting
    "rate_drop":        0.1,
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "max_depth":        7,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}


class XGBDemandModel(BaseModel):
    def __init__(self, params: dict | None = None):
        super().__init__("XGBoost", {**XGB_DEFAULT_PARAMS, **(params or {})})

    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "XGBDemandModel":
        self.feature_names_ = list(X.columns)

        # early_stopping_rounds goes in the constructor for XGBoost >= 2.0
        params = {**self.params}
        if eval_set:
            params["early_stopping_rounds"] = 50

        self.model = xgb.XGBRegressor(**params)

        fit_kwargs = {"verbose": False}
        if eval_set:
            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        try:
            best_iter = self.model.best_iteration
        except AttributeError:
            best_iter = self.params.get("n_estimators", "N/A")
        logger.info(f"XGBoost trained — best iteration: {best_iter}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model.predict(X[self.feature_names_]), 0)

    def get_feature_importance(self) -> pd.Series:
        scores = self.model.get_booster().get_fscore()
        return pd.Series(scores).sort_values(ascending=False)


# ─── CatBoost ─────────────────────────────────────────────────────────────────

CB_DEFAULT_PARAMS = {
    "loss_function":    "MAE",
    "eval_metric":      "RMSE",
    "iterations":       1000,
    "learning_rate":    0.05,
    "depth":            8,
    "l2_leaf_reg":      3,
    "bootstrap_type":   "Bernoulli",
    "subsample":        0.8,
    "random_seed":      42,
    "verbose":          False,
    "thread_count":     -1,
}


class CatBoostDemandModel(BaseModel):
    """
    CatBoost handles categorical features natively — no manual encoding needed.
    We pass categorical columns directly as cat_features.
    """
    def __init__(self, params: dict | None = None):
        super().__init__("CatBoost", {**CB_DEFAULT_PARAMS, **(params or {})})
        self.cat_features_: list = []

    def _build_model(self) -> cb.CatBoostRegressor:
        return cb.CatBoostRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "CatBoostDemandModel":
        self.feature_names_ = list(X.columns)
        # Auto-detect categorical columns
        self.cat_features_ = [c for c in X.columns
                               if X[c].dtype == "object" or X[c].dtype.name == "category"]
        self.model = self._build_model()

        pool = cb.Pool(X, y, cat_features=self.cat_features_)
        eval_pool = None
        if eval_set:
            X_val, y_val = eval_set
            eval_pool = cb.Pool(X_val, y_val, cat_features=self.cat_features_)

        self.model.fit(pool, eval_set=eval_pool,
                       early_stopping_rounds=50 if eval_pool else None)
        self.is_fitted = True
        logger.info(f"CatBoost trained — {self.model.tree_count_} trees, "
                    f"{len(self.cat_features_)} categorical features")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model.predict(X[self.feature_names_]), 0)

    def get_feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.get_feature_importance(),
            index=self.feature_names_
        ).sort_values(ascending=False)
