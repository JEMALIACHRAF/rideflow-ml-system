"""LightGBM demand forecasting model with early stopping and custom loss."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger
from models.base_model import BaseModel


DEFAULT_PARAMS = {
    "objective":        "regression_l1",  # MAE loss — robust to demand spikes
    "metric":           ["rmse", "mae"],
    "boosting_type":    "gbdt",
    "num_leaves":       127,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 20,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_estimators":     1000,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     42,
}


class LGBMDemandModel(BaseModel):
    def __init__(self, params: dict | None = None):
        super().__init__("LightGBM", {**DEFAULT_PARAMS, **(params or {})})

    def _build_model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "LGBMDemandModel":
        self.feature_names_ = list(X.columns)
        self.model = self._build_model()

        fit_kwargs = {}
        if eval_set:
            X_val, y_val = eval_set
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "callbacks": [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            }

        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        logger.info(f"LightGBM trained — best iteration: {self.model.best_iteration_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names_])
        return np.maximum(preds, 0)  # demand cannot be negative

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)
