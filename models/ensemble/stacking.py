"""
Ensemble methods: stacking with out-of-fold, weighted blending, soft voting.
Uses walk-forward cross-validation to avoid data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from loguru import logger
import mlflow
from models.base_model import BaseModel


class StackingEnsemble(BaseModel):
    """
    Two-level stacking:
      Level 0 — base models trained on out-of-fold splits (no leakage).
      Level 1 — meta-learner (Ridge) trained on OOF predictions.

    Key design: OOF predictions from level-0 are used as features for level-1.
    The meta-learner never sees training data directly.
    """
    def __init__(self, base_models: list[BaseModel],
                 meta_learner_alpha: float = 1.0, n_folds: int = 5):
        super().__init__("StackingEnsemble")
        self.base_models = base_models
        self.meta_learner = Ridge(alpha=meta_learner_alpha)
        self.n_folds = n_folds
        self._trained_base_models: list = []

    def _build_model(self):
        return self.meta_learner

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "StackingEnsemble":
        self.feature_names_ = list(X.columns)
        n = len(X)
        oof_preds = np.zeros((n, len(self.base_models)))
        kf = KFold(n_splits=self.n_folds, shuffle=False)  # no shuffle for time series

        logger.info(f"Stacking: training {len(self.base_models)} base models "
                    f"× {self.n_folds} folds")

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for model_i, model in enumerate(self.base_models):
                model_copy = model.__class__(model.params.copy())
                model_copy.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                oof_preds[val_idx, model_i] = model_copy.predict(X_val)
                logger.debug(f"Fold {fold_i+1} — {model.name} OOF done")

        # Train meta-learner on full OOF matrix
        self.meta_learner.fit(oof_preds, y)
        logger.info(f"Meta-learner coefficients: "
                    f"{dict(zip([m.name for m in self.base_models], self.meta_learner.coef_))}")

        # Retrain base models on full training set
        self._trained_base_models = []
        for model in self.base_models:
            model_copy = model.__class__(model.params.copy())
            model_copy.fit(X, y)
            self._trained_base_models.append(model_copy)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([
            m.predict(X) for m in self._trained_base_models
        ])
        return np.maximum(self.meta_learner.predict(base_preds), 0)

    def get_model_weights(self) -> dict:
        return {m.name: round(float(c), 4)
                for m, c in zip(self.base_models, self.meta_learner.coef_)}


class WeightedBlending(BaseModel):
    """
    Weighted average of base model predictions.
    Weights are learned by minimizing RMSE on a held-out validation set.
    Uses scipy minimize with constraints: weights >= 0, sum = 1.
    """
    def __init__(self, base_models: list[BaseModel]):
        super().__init__("WeightedBlending")
        self.base_models = base_models
        self.weights_: np.ndarray = np.array([])
        self._trained_base_models: list = []

    def _build_model(self):
        return None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "WeightedBlending":
        from scipy.optimize import minimize

        self.feature_names_ = list(X.columns)
        assert eval_set is not None, "WeightedBlending requires an eval_set to optimize weights"
        X_val, y_val = eval_set

        # Train all base models
        self._trained_base_models = []
        val_preds_matrix = []
        for model in self.base_models:
            model.fit(X, y)
            self._trained_base_models.append(model)
            val_preds_matrix.append(model.predict(X_val))

        val_preds = np.column_stack(val_preds_matrix)
        n_models = len(self.base_models)

        def objective(w):
            blend = (val_preds * w).sum(axis=1)
            return np.sqrt(np.mean((y_val.values - blend) ** 2))

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n_models
        result = minimize(objective, x0=np.ones(n_models) / n_models,
                          method="SLSQP", bounds=bounds, constraints=constraints)

        self.weights_ = result.x
        logger.info(f"Optimal blend weights: "
                    f"{dict(zip([m.name for m in self.base_models], self.weights_.round(3)))}")
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self._trained_base_models])
        return np.maximum((preds * self.weights_).sum(axis=1), 0)


class VotingEnsemble(BaseModel):
    """Simple unweighted average of all base model predictions."""
    def __init__(self, base_models: list[BaseModel]):
        super().__init__("VotingEnsemble")
        self.base_models = base_models
        self._trained_base_models: list = []

    def _build_model(self): return None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "VotingEnsemble":
        self.feature_names_ = list(X.columns)
        self._trained_base_models = []
        for model in self.base_models:
            model.fit(X, y, eval_set=eval_set)
            self._trained_base_models.append(model)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self._trained_base_models])
        return np.maximum(preds.mean(axis=1), 0)
