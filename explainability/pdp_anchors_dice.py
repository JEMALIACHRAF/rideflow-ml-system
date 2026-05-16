"""
Advanced explainability:
- PDP (Partial Dependence Plots): marginal effect of one feature
- ICE (Individual Conditional Expectation): per-sample effect curves
- Anchors: IF-THEN rule explanations (high-precision local rules)
- DiCE: counterfactual explanations ("what would need to change?")
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from typing import Any, Callable
from sklearn.inspection import partial_dependence


# ─── PDP / ICE ────────────────────────────────────────────────────────────────

class PDPICEExplainer:
    """
    Partial Dependence Plots and Individual Conditional Expectation curves.
    PDP = average of ICE lines = marginal effect across population.
    ICE = per-sample effect = heterogeneity of the feature's impact.
    """
    def __init__(self, model: Any, X_train: pd.DataFrame,
                 output_dir: str = "reports/pdp"):
        self.model      = model
        self.X_train    = X_train
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_pdp(self, feature: str, n_points: int = 50,
                 save: bool = True) -> dict:
        """
        1D PDP for a single feature.
        Returns grid values and average predictions.
        """
        feat_idx = list(self.X_train.columns).index(feature)
        grid_vals = np.linspace(
            self.X_train[feature].quantile(0.05),
            self.X_train[feature].quantile(0.95),
            n_points,
        )
        avg_preds = []
        for val in grid_vals:
            X_mod = self.X_train.copy()
            X_mod[feature] = val
            avg_preds.append(self.model.predict(X_mod).mean())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grid_vals, avg_preds, lw=2.5, color="#2563eb")
        ax.fill_between(grid_vals, avg_preds, alpha=0.1, color="#2563eb")
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Average predicted demand", fontsize=12)
        ax.set_title(f"PDP — Effect of '{feature}' on Demand", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            path = self.output_dir / f"pdp_{feature.replace('/', '_')}.png"
            plt.savefig(path, dpi=150)
            logger.info(f"Saved PDP → {path}")
        plt.close()
        return {"feature": feature, "grid": grid_vals.tolist(), "avg_pred": avg_preds}

    def plot_ice(self, feature: str, n_samples: int = 200,
                 n_points: int = 50, save: bool = True) -> None:
        """
        ICE curves: one line per sample, showing individual feature effects.
        Centred ICE (c-ICE) subtracts the prediction at feature minimum
        to highlight heterogeneity rather than level.
        """
        sample = self.X_train.sample(n=min(n_samples, len(self.X_train)), random_state=42)
        grid_vals = np.linspace(
            self.X_train[feature].quantile(0.05),
            self.X_train[feature].quantile(0.95),
            n_points,
        )
        ice_curves = []
        for _, row in sample.iterrows():
            X_mod = pd.DataFrame([row] * n_points, columns=self.X_train.columns)
            X_mod[feature] = grid_vals
            ice_curves.append(self.model.predict(X_mod))

        ice_arr = np.array(ice_curves)
        # Centre: subtract first value so all lines start at 0
        ice_centred = ice_arr - ice_arr[:, 0:1]
        avg_centred = ice_centred.mean(axis=0)

        fig, ax = plt.subplots(figsize=(9, 6))
        for curve in ice_centred:
            ax.plot(grid_vals, curve, alpha=0.08, color="#94a3b8", lw=0.8)
        ax.plot(grid_vals, avg_centred, lw=2.5, color="#dc2626", label="PDP (centred)")
        ax.axhline(0, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Demand change (centred)", fontsize=12)
        ax.set_title(f"c-ICE — '{feature}' individual effects (n={n_samples})", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            path = self.output_dir / f"ice_{feature.replace('/', '_')}.png"
            plt.savefig(path, dpi=150)
            logger.info(f"Saved ICE → {path}")
        plt.close()

    def plot_2d_pdp(self, feature1: str, feature2: str,
                    n_points: int = 20, save: bool = True) -> None:
        """
        2D PDP interaction plot between two features.
        Reveals synergistic/antagonistic interactions.
        Example: hour × weather → rush-hour rain spike.
        """
        grid1 = np.linspace(self.X_train[feature1].quantile(0.1),
                             self.X_train[feature1].quantile(0.9), n_points)
        grid2 = np.linspace(self.X_train[feature2].quantile(0.1),
                             self.X_train[feature2].quantile(0.9), n_points)

        Z = np.zeros((n_points, n_points))
        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                X_mod = self.X_train.copy()
                X_mod[feature1] = v1
                X_mod[feature2] = v2
                Z[i, j] = self.model.predict(X_mod).mean()

        fig, ax = plt.subplots(figsize=(9, 7))
        cp = ax.contourf(grid2, grid1, Z, levels=20, cmap="RdYlGn")
        plt.colorbar(cp, ax=ax, label="Predicted demand")
        ax.set_xlabel(feature2, fontsize=12)
        ax.set_ylabel(feature1, fontsize=12)
        ax.set_title(f"2D PDP — {feature1} × {feature2}", fontsize=13)
        plt.tight_layout()
        if save:
            path = self.output_dir / f"pdp2d_{feature1}_{feature2}.png"
            plt.savefig(path, dpi=150)
        plt.close()


# ─── Anchors ──────────────────────────────────────────────────────────────────

class AnchorExplainer:
    """
    Anchors: IF-THEN rule explanations.
    An anchor is a set of conditions that suffice to produce the prediction
    with high precision regardless of other feature values.

    Example output:
        IF hour IN [17, 18, 19]
        AND is_weekend = 0
        AND weather = 'rain'
        THEN demand > 18  (precision = 0.94, coverage = 0.12)
    """
    def __init__(self, X_train: pd.DataFrame,
                 predict_fn: Callable,
                 categorical_names: dict | None = None):
        from alibi.explainers import AnchorTabular
        self.X_train = X_train
        self.predict_fn = predict_fn
        self.feature_names = list(X_train.columns)
        cat_names = categorical_names or {}

        self.explainer_ = AnchorTabular(
            predictor=predict_fn,
            feature_names=self.feature_names,
            categorical_names=cat_names,
            seed=42,
        )
        self.explainer_.fit(X_train.values, disc_perc=(25, 50, 75))
        logger.info("Anchor explainer ready")

    def explain(self, X_row: np.ndarray | pd.Series,
                threshold: float = 0.90) -> dict:
        """
        Find the anchor (IF-THEN rule) for a single prediction.

        Args:
            threshold: minimum precision of the anchor (default 0.90).
        Returns dict with anchor conditions, precision, coverage.
        """
        if isinstance(X_row, pd.Series):
            X_row = X_row.values
        X_row = X_row.astype(np.float32).reshape(1, -1)

        exp = self.explainer_.explain(X_row, threshold=threshold)
        prediction = float(self.predict_fn(X_row)[0])

        result = {
            "prediction":  round(prediction, 2),
            "anchor":      list(exp.anchor),
            "precision":   round(float(exp.precision), 4),
            "coverage":    round(float(exp.coverage), 4),
            "rule_str":    "IF " + " AND ".join(exp.anchor) if exp.anchor else "No anchor found",
        }
        logger.info(f"Anchor: {result['rule_str']} "
                    f"(precision={result['precision']:.0%}, coverage={result['coverage']:.1%})")
        return result


# ─── DiCE Counterfactuals ─────────────────────────────────────────────────────

class CounterfactualExplainer:
    """
    DiCE (Diverse Counterfactual Explanations).
    Answers: "What minimal changes would alter this prediction by X?"

    Example: "If temperature were +5°C and hour were 18 instead of 3,
              demand would increase from 2 to 18 rides."

    Useful for:
    - Driver incentives: "Go to zone 11 instead of zone 20 for 3× more rides"
    - Pricing: "If weather improves, price should drop by 15%"
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 predict_fn: Callable,
                 continuous_features: list[str] | None = None,
                 output_dir: str = "reports/counterfactuals"):
        import dice_ml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(X_train.columns)
        self.continuous_features = continuous_features or list(X_train.columns)

        data_interface = dice_ml.Data(
            dataframe=pd.concat([X_train, y_train.rename("demand")], axis=1),
            continuous_features=self.continuous_features,
            outcome_name="demand",
        )
        model_interface = dice_ml.Model(model=predict_fn, backend="sklearn")
        self.dice_ = dice_ml.Dice(data_interface, model_interface, method="random")
        logger.info("DiCE counterfactual explainer ready")

    def generate(self, X_row: pd.DataFrame,
                 desired_range: tuple = (15, 30),
                 n_cfs: int = 3,
                 features_to_vary: list[str] | None = None) -> pd.DataFrame:
        """
        Generate diverse counterfactuals for X_row.

        Args:
            desired_range: target prediction range (e.g., 15–30 rides).
            n_cfs: number of counterfactuals to generate.
            features_to_vary: which features can be changed (default: all continuous).

        Returns DataFrame comparing original vs each counterfactual.
        """
        cf_exp = self.dice_.generate_counterfactuals(
            X_row,
            total_CFs=n_cfs,
            desired_range=list(desired_range),
            features_to_vary=features_to_vary or "all",
        )
        cf_df = cf_exp.cf_examples_list[0].final_cfs_df
        original_pred = float(cf_exp.cf_examples_list[0].test_pred)

        logger.info(f"DiCE: {n_cfs} counterfactuals generated "
                    f"(original pred = {original_pred:.1f})")

        # Highlight which features changed
        original_vals = X_row.iloc[0]
        diff_rows = []
        for _, cf_row in cf_df.iterrows():
            changes = {
                col: f"{original_vals[col]:.2f} → {cf_row[col]:.2f}"
                for col in self.feature_names
                if col in cf_row and abs(float(cf_row[col]) - float(original_vals[col])) > 1e-3
            }
            diff_rows.append({
                "cf_prediction": round(float(cf_row.get("demand", 0)), 1),
                "n_changes": len(changes),
                "changes": changes,
            })

        result = pd.DataFrame(diff_rows)
        result.to_csv(self.output_dir / "counterfactuals.csv", index=False)
        return result
