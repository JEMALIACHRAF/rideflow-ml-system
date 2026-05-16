"""
Dynamic pricing engine.
Computes surge multiplier based on demand/supply ratio and elasticity model.
Includes A/B testing framework for comparing pricing strategies.
"""
import numpy as np
import pandas as pd
from loguru import logger


SURGE_SCHEDULE = [
    (1.5, 1.2),
    (2.0, 1.5),
    (3.0, 1.8),
    (4.0, 2.5),
]


class SurgeModel:
    """Rule-based surge multiplier with learned zone baselines."""
    def __init__(self):
        self.zone_baselines_: dict = {}

    def fit(self, df: pd.DataFrame, target: str = "demand") -> "SurgeModel":
        self.zone_baselines_ = df.groupby("zone")[target].median().to_dict()
        logger.info(f"Surge model fitted — {len(self.zone_baselines_)} zone baselines")
        return self

    def predict_surge(self, zone: str, predicted_demand: float) -> float:
        baseline = self.zone_baselines_.get(str(zone), 10.0)
        ratio = predicted_demand / (baseline + 1e-3)
        mult = 1.0
        for threshold, m in SURGE_SCHEDULE:
            if ratio >= threshold:
                mult = m
        return mult

    def compute_price(self, zone: str, predicted_demand: float,
                      base_price: float = 8.0) -> dict:
        surge = self.predict_surge(zone, predicted_demand)
        return {
            "zone": zone,
            "base_price": base_price,
            "surge_multiplier": round(surge, 2),
            "final_price": round(base_price * surge, 2),
            "demand_ratio": round(
                predicted_demand / (self.zone_baselines_.get(str(zone), 10) + 1e-3), 3
            ),
        }


class ABTestFramework:
    """
    Simple A/B testing framework for pricing strategies.
    Randomly assigns requests to strategy A or B based on zone hash.
    Tracks revenue and conversion metrics per strategy.
    """
    def __init__(self, strategy_a: SurgeModel, strategy_b: SurgeModel,
                 split_ratio: float = 0.5):
        self.strategy_a  = strategy_a
        self.strategy_b  = strategy_b
        self.split_ratio = split_ratio
        self.results_a: list = []
        self.results_b: list = []

    def route(self, zone: str, predicted_demand: float,
              base_price: float = 8.0) -> dict:
        """Route request to A or B based on zone hash for sticky assignment."""
        use_a = (hash(zone) % 100) < (self.split_ratio * 100)
        strategy = self.strategy_a if use_a else self.strategy_b
        result = strategy.compute_price(zone, predicted_demand, base_price)
        result["variant"] = "A" if use_a else "B"
        (self.results_a if use_a else self.results_b).append(result)
        return result

    def summary(self) -> pd.DataFrame:
        """Compare mean price and surge across A and B variants."""
        rows = []
        for variant, results in [("A", self.results_a), ("B", self.results_b)]:
            if not results:
                continue
            df = pd.DataFrame(results)
            rows.append({
                "variant": variant,
                "n_requests": len(df),
                "mean_price": round(df["final_price"].mean(), 2),
                "mean_surge": round(df["surge_multiplier"].mean(), 2),
                "pct_surged": round((df["surge_multiplier"] > 1).mean(), 3),
            })
        return pd.DataFrame(rows)
