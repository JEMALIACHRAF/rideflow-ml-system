"""
Abstract base class for all RideFlow models.
Enforces consistent interface: fit / predict / save / load / log_to_mlflow.
"""
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class BaseModel(ABC):
    """
    All models inherit from this class.
    Provides: fit, predict, evaluate, save, load, mlflow logging.
    """
    def __init__(self, name: str, params: dict | None = None):
        self.name = name
        self.params = params or {}
        self.model: Any = None
        self.feature_names_: list = []
        self.is_fitted: bool = False

    @abstractmethod
    def _build_model(self) -> Any:
        """Instantiate the underlying model with self.params."""
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "BaseModel":
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions (never negative for demand)."""
        ...

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compute standard regression metrics."""
        preds = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        mape = float(np.mean(np.abs((y - preds) / (np.abs(y) + 1.0))))
        mae  = np.mean(np.abs(y - preds))
        r2   = 1 - np.sum((y - preds) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-9)
        metrics = {"rmse": round(rmse, 4), "mape": round(mape, 4),
                   "mae": round(mae, 4), "r2": round(r2, 4)}
        logger.info(f"{self.name} → RMSE={rmse:.3f}  MAPE={mape:.3%}  R²={r2:.3f}")
        return metrics

    def save(self, path: str | Path) -> Path:
        """Pickle the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.name} → {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load a pickled model."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded model from {path}")
        return obj

    def log_to_mlflow(self, metrics: dict, params: dict | None = None,
                      artifact_path: Optional[Path] = None) -> None:
        """Log params, metrics, and model artifact to MLflow."""
        mlflow.log_params(params or self.params)
        mlflow.log_metrics(metrics)
        if artifact_path:
            mlflow.log_artifact(str(artifact_path))
        logger.debug(f"Logged {self.name} to MLflow")
