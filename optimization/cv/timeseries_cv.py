"""
Time-series cross-validation strategies.
Walk-forward CV with mandatory gap to prevent data leakage.
Purged K-Fold for cases with overlapping label horizons.
"""
import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from loguru import logger


class WalkForwardCV:
    """
    Walk-forward (expanding window) cross-validation.
    Each fold: train on [0..t], validate on [t+gap..t+gap+val_size].
    Gap prevents leakage when features use future information (lags, rolling).

    Example with n=1000, n_splits=5, val_size=100, gap=24:
        Fold 1: train [0..400], val [424..524]
        Fold 2: train [0..500], val [524..624]
        ...
    """
    def __init__(self, n_splits: int = 5, gap_hours: int = 24,
                 val_size: int | None = None, min_train_size: int | None = None):
        self.n_splits      = n_splits
        self.gap_hours     = gap_hours
        self.val_size      = val_size
        self.min_train_size = min_train_size

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        val_size = self.val_size or max(24, n // (self.n_splits + 1))
        min_train = self.min_train_size or val_size * 2

        # Compute fold start points
        total_val = val_size * self.n_splits
        available = n - min_train - self.gap_hours - total_val
        if available < 0:
            raise ValueError(
                f"Not enough data for {self.n_splits} folds. "
                f"Need at least {min_train + self.gap_hours + total_val} rows, got {n}."
            )

        for fold in range(self.n_splits):
            train_end = min_train + (available * fold // max(self.n_splits - 1, 1))
            val_start = train_end + self.gap_hours
            val_end   = min(val_start + val_size, n)

            train_idx = np.arange(0, train_end)
            val_idx   = np.arange(val_start, val_end)

            logger.debug(f"Fold {fold+1}: train [0..{train_end}] "
                         f"val [{val_start}..{val_end}] (gap={self.gap_hours}h)")
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class SlidingWindowCV:
    """
    Fixed-size sliding window CV.
    Unlike expanding window, each fold trains on the same number of samples.
    Better for detecting concept drift (recent data may behave differently).
    """
    def __init__(self, n_splits: int = 5, train_size: int = 500,
                 val_size: int = 100, gap_hours: int = 24):
        self.n_splits   = n_splits
        self.train_size = train_size
        self.val_size   = val_size
        self.gap_hours  = gap_hours

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        step = (n - self.train_size - self.gap_hours - self.val_size) // self.n_splits

        for fold in range(self.n_splits):
            offset    = fold * step
            train_start = offset
            train_end   = offset + self.train_size
            val_start   = train_end + self.gap_hours
            val_end     = val_start + self.val_size

            if val_end > n:
                break

            train_idx = np.arange(train_start, train_end)
            val_idx   = np.arange(val_start, val_end)
            logger.debug(f"Sliding fold {fold+1}: train [{train_start}..{train_end}] "
                         f"val [{val_start}..{val_end}]")
            yield train_idx, val_idx


class PurgedKFold:
    """
    Purged K-Fold for financial / ride-hailing time-series.
    Removes samples from the training set that overlap temporally with validation.
    Useful when target horizon > 1 (e.g. predicting 3h ahead creates overlap).

    Args:
        n_splits:    Number of folds.
        purge_gap:   Number of rows to purge from train boundary near val.
        embargo_pct: Fraction of val fold to embargo at the end (prevents future leakage).
    """
    def __init__(self, n_splits: int = 5, purge_gap: int = 24, embargo_pct: float = 0.01):
        self.n_splits    = n_splits
        self.purge_gap   = purge_gap
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_size = n // self.n_splits
        embargo   = int(fold_size * self.embargo_pct)

        for fold in range(self.n_splits):
            val_start = fold * fold_size
            val_end   = val_start + fold_size

            # Purge: remove train samples that are within purge_gap of val boundary
            purge_start = max(0, val_start - self.purge_gap)
            purge_end   = min(n, val_end + embargo)

            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(purge_end, n),
            ])
            val_idx = np.arange(val_start, val_end)

            logger.debug(f"Purged fold {fold+1}: train={len(train_idx)}, "
                         f"val={len(val_idx)}, purged={self.purge_gap + embargo} rows")
            yield train_idx, val_idx


def cross_validate_model(model_cls, model_params: dict,
                         X: pd.DataFrame, y: pd.Series,
                         cv, metrics_fn) -> pd.DataFrame:
    """
    Generic CV runner. Returns a DataFrame of per-fold metrics.

    Args:
        model_cls:    Model class (must implement fit/predict).
        model_params: Params dict passed to model_cls constructor.
        X, y:         Features and target.
        cv:           CV splitter with .split(X) method.
        metrics_fn:   Function(y_true, y_pred) → dict of metrics.
    """
    results = []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = model_cls(model_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        preds = model.predict(X_val)

        fold_metrics = metrics_fn(y_val.values, preds)
        fold_metrics["fold"] = fold + 1
        fold_metrics["train_size"] = len(tr_idx)
        fold_metrics["val_size"]   = len(val_idx)
        results.append(fold_metrics)
        logger.info(f"Fold {fold+1}: {fold_metrics}")

    df = pd.DataFrame(results)
    logger.success(f"\nCV Summary:\n{df.describe().round(4)}")
    return df
