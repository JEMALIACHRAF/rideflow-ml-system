"""
Model output calibration.
Demand models output raw counts — calibration corrects systematic bias
(e.g. underestimating peak hours, overestimating night demand).

Isotonic regression: flexible monotonic correction.
Platt scaling: sigmoid-based, less prone to overfitting on small val sets.
Temperature scaling: single-param calibration (fast, interpretable).
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
from scipy.optimize import minimize_scalar
from loguru import logger
import matplotlib.pyplot as plt


class IsotonicCalibrator:
    """
    Isotonic regression calibration.
    Fits a monotonic mapping: raw_pred → calibrated_pred.
    Best when: model has systematic non-linear bias over prediction range.
    """
    def __init__(self, out_of_bounds: str = "clip"):
        self.iso = IsotonicRegression(out_of_bounds=out_of_bounds)
        self.is_fitted = False

    def fit(self, raw_preds: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(raw_preds, y_true)
        self.is_fitted = True
        # Compute calibration error
        calibrated = self.transform(raw_preds)
        before = np.mean(np.abs(raw_preds - y_true))
        after  = np.mean(np.abs(calibrated - y_true))
        logger.info(f"Isotonic calibration: MAE {before:.3f} → {after:.3f} "
                    f"(Δ={after - before:+.3f})")
        return self

    def transform(self, raw_preds: np.ndarray) -> np.ndarray:
        return self.iso.predict(raw_preds)

    def fit_transform(self, raw_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(raw_preds, y_true).transform(raw_preds)


class PlattCalibrator:
    """
    Platt scaling: fits a linear model A*f(x) + B on top of raw predictions.
    Less flexible than isotonic but generalises better with small datasets.
    """
    def __init__(self):
        self.A: float = 1.0
        self.B: float = 0.0
        self.is_fitted = False

    def fit(self, raw_preds: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        # Normalise targets to [0,1] for fitting, then rescale back
        self._y_min = y_true.min()
        self._y_max = y_true.max()
        y_norm = (y_true - self._y_min) / (self._y_max - self._y_min + 1e-8)

        def neg_log_likelihood(params):
            A, B = params
            probs = expit(A * raw_preds + B)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(y_norm * np.log(probs) + (1 - y_norm) * np.log(1 - probs))

        from scipy.optimize import minimize
        result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="L-BFGS-B")
        self.A, self.B = result.x
        self.is_fitted = True
        logger.info(f"Platt calibration: A={self.A:.4f}, B={self.B:.4f}")
        return self

    def transform(self, raw_preds: np.ndarray) -> np.ndarray:
        probs = expit(self.A * raw_preds + self.B)
        return probs * (self._y_max - self._y_min) + self._y_min

    def fit_transform(self, raw_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(raw_preds, y_true).transform(raw_preds)


class TemperatureScaling:
    """
    Single-parameter calibration: divide logits by temperature T.
    T > 1 → softer (more uncertain) predictions.
    T < 1 → sharper predictions.
    Fast to fit, interpretable, good for overconfident models.
    """
    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted = False

    def fit(self, raw_preds: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        def objective(T):
            calibrated = raw_preds / T
            return np.mean((calibrated - y_true) ** 2)

        result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        self.is_fitted = True
        logger.info(f"Temperature scaling: T={self.temperature:.4f}")
        return self

    def transform(self, raw_preds: np.ndarray) -> np.ndarray:
        return raw_preds / self.temperature

    def fit_transform(self, raw_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(raw_preds, y_true).transform(raw_preds)


def calibration_curve_data(raw_preds: np.ndarray, y_true: np.ndarray,
                           n_bins: int = 20) -> pd.DataFrame:
    """
    Reliability diagram data: compare predicted vs actual demand in quantile bins.
    Used to visualise over/underestimation by demand level.
    """
    df = pd.DataFrame({"pred": raw_preds, "actual": y_true})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin").agg(
        mean_pred=("pred", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count"),
    ).reset_index()
    grouped["bias"] = grouped["mean_pred"] - grouped["mean_actual"]
    return grouped


def compare_calibrators(raw_preds: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Fit all three calibrators and compare MAE on the same data."""
    results = []
    for name, cls in [("Isotonic", IsotonicCalibrator),
                      ("Platt", PlattCalibrator),
                      ("Temperature", TemperatureScaling)]:
        cal = cls()
        cal_preds = cal.fit_transform(raw_preds.copy(), y_true.copy())
        mae = np.mean(np.abs(cal_preds - y_true))
        results.append({"calibrator": name, "mae": round(float(mae), 4)})

    raw_mae = np.mean(np.abs(raw_preds - y_true))
    results.insert(0, {"calibrator": "Uncalibrated", "mae": round(float(raw_mae), 4)})
    return pd.DataFrame(results).sort_values("mae")
