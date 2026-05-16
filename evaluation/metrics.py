"""
Evaluation module.
Standard ML metrics + demand-specific metrics + revenue backtesting.
"""
import numpy as np
import pandas as pd
from loguru import logger


# ─── Metrics ──────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """MAPE with epsilon guard for zero actuals (common in low-demand zones at night)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.9) -> float:
    """
    Pinball (quantile) loss. Useful when over-prediction costs less than under-prediction.
    At quantile=0.9: penalises under-prediction 9× more than over-prediction.
    """
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1) * err)))

def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  alpha: float = 0.1) -> float:
    """
    Winkler score for prediction intervals.
    Penalises wide intervals and coverage failures simultaneously.
    """
    width   = upper - lower
    penalty = np.where(y_true < lower, 2 / alpha * (lower - y_true),
               np.where(y_true > upper, 2 / alpha * (y_true - upper), 0))
    return float(np.mean(width + penalty))

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        prefix: str = "") -> dict:
    """Return all standard metrics in a flat dict."""
    p = f"{prefix}_" if prefix else ""
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    r2_val  = 1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-9)
    return {
        f"{p}rmse":         round(rmse(y_true, y_pred), 4),
        f"{p}mape":         round(mape(y_true, y_pred), 4),
        f"{p}mae":          round(mae_val, 4),
        f"{p}r2":           round(float(r2_val), 4),
        f"{p}pinball_90":   round(pinball_loss(y_true, y_pred, 0.9), 4),
        f"{p}max_error":    round(float(np.max(np.abs(y_true - y_pred))), 4),
    }


# ─── Zone-level evaluation ────────────────────────────────────────────────────

def evaluate_by_zone(df: pd.DataFrame, y_pred_col: str = "pred",
                     target_col: str = "demand", zone_col: str = "zone") -> pd.DataFrame:
    """Per-zone RMSE and MAPE — identifies which zones are hardest to predict."""
    results = []
    for zone, grp in df.groupby(zone_col):
        m = compute_all_metrics(grp[target_col].values, grp[y_pred_col].values, prefix="")
        m["zone"] = zone
        m["n_samples"] = len(grp)
        results.append(m)
    return pd.DataFrame(results).set_index("zone").sort_values("mape", ascending=False)


def evaluate_by_hour(df: pd.DataFrame, y_pred_col: str = "pred",
                     target_col: str = "demand") -> pd.DataFrame:
    """Per-hour MAPE — identifies peak hours where model underperforms."""
    results = []
    for hour, grp in df.groupby("hour"):
        m = compute_all_metrics(grp[target_col].values, grp[y_pred_col].values)
        m["hour"] = hour
        results.append(m)
    return pd.DataFrame(results).set_index("hour")


# ─── Revenue backtesting ──────────────────────────────────────────────────────

BASE_PRICE_EUR = 8.0          # base fare per trip (EUR)
SURGE_SCHEDULE = [             # (demand_threshold_fraction, multiplier)
    (1.5, 1.2),
    (2.0, 1.5),
    (3.0, 1.8),
    (4.0, 2.5),
]


def demand_to_surge(demand: float, zone_baseline: float) -> float:
    """Map demand/baseline ratio to surge multiplier."""
    ratio = demand / (zone_baseline + 1e-3)
    mult = 1.0
    for threshold, m in SURGE_SCHEDULE:
        if ratio >= threshold:
            mult = m
    return mult


def simulate_revenue(df: pd.DataFrame, pred_col: str = "pred",
                     target_col: str = "demand",
                     zone_col: str = "zone") -> dict:
    """
    Compare revenue under perfect-information pricing vs model-based pricing.
    Assumes: price = BASE_PRICE × surge_multiplier.
    Demand captured = min(actual, served) where served ∝ price sent to drivers.
    """
    zone_baselines = df.groupby(zone_col)[target_col].median().to_dict()

    rows = []
    for _, row in df.iterrows():
        baseline = zone_baselines.get(str(row[zone_col]), 10)
        actual   = row[target_col]
        pred     = max(row[pred_col], 0)

        surge_actual = demand_to_surge(actual, baseline)
        surge_pred   = demand_to_surge(pred,   baseline)

        # Perfect pricing revenue
        revenue_perfect = actual * BASE_PRICE_EUR * surge_actual
        # Model-based pricing revenue
        revenue_model   = actual * BASE_PRICE_EUR * surge_pred  # actual riders take the price

        rows.append({
            "revenue_perfect": revenue_perfect,
            "revenue_model":   revenue_model,
            "surge_actual":    surge_actual,
            "surge_pred":      surge_pred,
        })

    result_df = pd.DataFrame(rows)
    total_perfect = result_df["revenue_perfect"].sum()
    total_model   = result_df["revenue_model"].sum()
    efficiency    = total_model / total_perfect

    summary = {
        "total_revenue_perfect_eur": round(total_perfect, 2),
        "total_revenue_model_eur":   round(total_model, 2),
        "revenue_efficiency":        round(efficiency, 4),
        "revenue_loss_eur":          round(total_perfect - total_model, 2),
        "mean_surge_actual":         round(result_df["surge_actual"].mean(), 3),
        "mean_surge_pred":           round(result_df["surge_pred"].mean(), 3),
    }
    logger.info(f"Backtest: revenue efficiency = {efficiency:.1%}, "
                f"loss = {summary['revenue_loss_eur']:.0f} EUR")
    return summary


# ─── PSI: Population Stability Index ─────────────────────────────────────────

def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI measures distribution shift between training and production data.
    PSI < 0.1   → stable
    PSI 0.1–0.2 → slight drift, monitor
    PSI > 0.2   → significant drift, consider retraining
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6

    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=bins)[0] / len(actual)

    # Replace zeros to avoid log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi_value = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    level = "stable" if psi_value < 0.1 else "monitor" if psi_value < 0.2 else "RETRAIN"
    logger.info(f"PSI = {psi_value:.4f} → {level}")
    return psi_value
