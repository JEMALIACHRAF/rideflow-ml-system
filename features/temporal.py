"""
Temporal feature engineering.
Creates lag features, rolling statistics, and Fourier terms for seasonality.
"""
import numpy as np
import pandas as pd
from loguru import logger


def add_lag_features(df: pd.DataFrame, target: str, lags: list[int],
                     group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Add lag features with optional group-by (e.g. per zone).

    Args:
        df: DataFrame sorted by timestamp.
        target: Column to lag.
        lags: List of lag periods (in rows, i.e. hours if hourly data).
        group_cols: Columns to group by before computing lags.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"{target}_lag_{lag}h"
        if group_cols:
            df[col_name] = df.groupby(group_cols)[target].shift(lag)
        else:
            df[col_name] = df[target].shift(lag)
    logger.debug(f"Added {len(lags)} lag features")
    return df


def add_rolling_features(df: pd.DataFrame, target: str, windows: list[int],
                         group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max features.

    Uses min_periods=1 to avoid NaN at the start of each group.
    """
    df = df.copy()
    for w in windows:
        prefix = f"{target}_roll{w}h"
        if group_cols:
            grp = df.groupby(group_cols)[target]
            df[f"{prefix}_mean"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df[f"{prefix}_std"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
            df[f"{prefix}_max"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
            df[f"{prefix}_min"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())
        else:
            s = df[target].shift(1)
            df[f"{prefix}_mean"] = s.rolling(w, min_periods=1).mean()
            df[f"{prefix}_std"]  = s.rolling(w, min_periods=1).std().fillna(0)
            df[f"{prefix}_max"]  = s.rolling(w, min_periods=1).max()
            df[f"{prefix}_min"]  = s.rolling(w, min_periods=1).min()
    logger.debug(f"Added rolling features for windows {windows}")
    return df


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar features: hour, day of week, month, is_weekend, is_rush_hour, etc."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    df["hour"]          = ts.dt.hour
    df["day_of_week"]   = ts.dt.dayofweek
    df["day_of_month"]  = ts.dt.day
    df["month"]         = ts.dt.month
    df["week_of_year"]  = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"]    = (ts.dt.dayofweek >= 5).astype(int)
    df["is_friday_night"] = ((ts.dt.dayofweek == 4) & (ts.dt.hour >= 20)).astype(int)
    df["is_rush_am"]    = ((ts.dt.dayofweek < 5) & ts.dt.hour.between(7, 9)).astype(int)
    df["is_rush_pm"]    = ((ts.dt.dayofweek < 5) & ts.dt.hour.between(17, 19)).astype(int)
    df["is_night"]      = ts.dt.hour.between(0, 5).astype(int)
    df["quarter"]       = ts.dt.quarter

    # French public holidays (simplified)
    public_holidays = {
        (1, 1), (5, 1), (5, 8), (7, 14), (8, 15),
        (11, 1), (11, 11), (12, 25),
    }
    df["is_holiday"] = df.apply(
        lambda r: int((r["month"], r["day_of_month"]) in public_holidays), axis=1
    )
    logger.debug("Added calendar features")
    return df


def add_fourier_features(df: pd.DataFrame, timestamp_col: str = "timestamp",
                         periods: dict | None = None) -> pd.DataFrame:
    """
    Fourier terms to capture seasonality without overfitting.

    Args:
        periods: dict mapping period_name → (period_in_hours, n_harmonics)
                 e.g. {"daily": (24, 3), "weekly": (168, 2)}
    """
    if periods is None:
        periods = {"daily": (24, 3), "weekly": (168, 2), "yearly": (8760, 2)}

    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    t = (ts - ts.min()).dt.total_seconds() / 3600  # hours since start

    for name, (period, n_harmonics) in periods.items():
        for k in range(1, n_harmonics + 1):
            df[f"fourier_{name}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"fourier_{name}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    logger.debug(f"Added Fourier features: {list(periods.keys())}")
    return df


def add_demand_velocity(df: pd.DataFrame, target: str,
                        group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Velocity features: rate of change over last 1h, 3h, 6h.
    Captures acceleration/deceleration of demand.
    """
    df = df.copy()
    for h in [1, 3, 6]:
        col = f"{target}_velocity_{h}h"
        if group_cols:
            df[col] = df.groupby(group_cols)[target].transform(
                lambda x: x.diff(h) / h
            )
        else:
            df[col] = df[target].diff(h) / h
    return df


def build_temporal_features(df: pd.DataFrame, target: str = "demand",
                             lags: list[int] | None = None,
                             rolling_windows: list[int] | None = None) -> pd.DataFrame:
    """Full temporal feature pipeline."""
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
    if rolling_windows is None:
        rolling_windows = [3, 6, 12, 24]

    df = add_calendar_features(df)
    df = add_fourier_features(df)
    df = add_lag_features(df, target, lags, group_cols=["zone"])
    df = add_rolling_features(df, target, rolling_windows, group_cols=["zone"])
    df = add_demand_velocity(df, target, group_cols=["zone"])
    return df
