"""
SHAP explainability — global, local, interaction effects.
Covers: feature importance, waterfall plots, force plots, dependence plots.
Works with any tree-based model (TreeExplainer) or black-box (KernelExplainer).
"""
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from typing import Any


class SHAPExplainer:
    """
    Unified SHAP wrapper for tree-based and arbitrary models.
    Provides global importance, local explanations, and interaction effects.
    """
    def __init__(self, model: Any, model_type: str = "tree",
                 feature_names: list[str] | None = None,
                 output_dir: str = "reports/shap"):
        self.model        = model
        self.model_type   = model_type
        self.feature_names = feature_names
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.explainer_   = None
        self.shap_values_ : np.ndarray | None = None
        self.expected_value_: float = 0.0

    def fit(self, X: pd.DataFrame, sample_size: int = 2000) -> "SHAPExplainer":
        """Compute SHAP values on a sample of X."""
        sample = X.sample(min(sample_size, len(X)), random_state=42)

        logger.info(f"Building {self.model_type} SHAP explainer on {len(sample)} samples...")
        if self.model_type == "tree":
            self.explainer_ = shap.TreeExplainer(self.model)
            self.shap_values_ = self.explainer_.shap_values(sample)
        else:
            background = shap.sample(X, min(100, len(X)))
            self.explainer_ = shap.KernelExplainer(self.model.predict, background)
            self.shap_values_ = self.explainer_.shap_values(sample, nsamples=100)

        self.expected_value_ = float(
            self.explainer_.expected_value
            if not isinstance(self.explainer_.expected_value, list)
            else self.explainer_.expected_value[0]
        )
        self.sample_ = sample
        logger.success(f"SHAP computed — expected value: {self.expected_value_:.2f}")
        return self

    # ── Global importance ──────────────────────────────────────────────────────

    def global_importance(self) -> pd.Series:
        """Mean |SHAP| across all samples — global feature importance ranking."""
        importance = pd.Series(
            np.abs(self.shap_values_).mean(axis=0),
            index=self.sample_.columns,
        ).sort_values(ascending=False)
        return importance

    def plot_summary(self, max_display: int = 20, save: bool = True) -> None:
        """Beeswarm summary plot: feature importance + direction of effect."""
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values_, self.sample_,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Summary — Global Feature Importance", fontsize=13)
        plt.tight_layout()
        if save:
            path = self.output_dir / "shap_summary.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved → {path}")
        plt.close()

    def plot_bar_importance(self, max_display: int = 20, save: bool = True) -> None:
        """Bar chart of mean |SHAP| — cleaner than beeswarm for presentations."""
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values_, self.sample_,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Feature Importance (mean |SHAP|)", fontsize=13)
        plt.tight_layout()
        if save:
            path = self.output_dir / "shap_bar_importance.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved → {path}")
        plt.close()

    # ── Local explanation ──────────────────────────────────────────────────────

    def explain_single(self, X_row: pd.Series, save: bool = True,
                       filename: str = "shap_waterfall.png") -> dict:
        """
        Waterfall plot for a single prediction.
        Shows: base value + each feature's contribution → final prediction.
        """
        row_df = X_row.to_frame().T if isinstance(X_row, pd.Series) else X_row
        sv = self.explainer_.shap_values(row_df)[0]
        pred = self.expected_value_ + sv.sum()

        fig, ax = plt.subplots(figsize=(10, 6))
        explanation = shap.Explanation(
            values=sv,
            base_values=self.expected_value_,
            data=row_df.values[0],
            feature_names=list(row_df.columns),
        )
        shap.waterfall_plot(explanation, show=False, max_display=15)
        plt.title(f"SHAP Waterfall — Prediction: {pred:.1f} rides", fontsize=12)
        plt.tight_layout()
        if save:
            path = self.output_dir / filename
            plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        # Return top contributors as dict
        contributions = pd.Series(sv, index=row_df.columns).sort_values(key=abs, ascending=False)
        return {
            "prediction": round(pred, 2),
            "base_value": round(self.expected_value_, 2),
            "top_contributors": contributions.head(10).round(3).to_dict(),
        }

    def plot_force(self, idx: int = 0, save: bool = True) -> None:
        """Force plot for a single sample — horizontal stacked bar view."""
        shap.initjs()
        force = shap.force_plot(
            self.expected_value_,
            self.shap_values_[idx],
            self.sample_.iloc[idx],
            matplotlib=True,
            show=False,
        )
        if save:
            path = self.output_dir / f"shap_force_{idx}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved → {path}")
        plt.close()

    # ── Interaction effects ────────────────────────────────────────────────────

    def plot_dependence(self, feature: str, interaction_feature: str = "auto",
                        save: bool = True) -> None:
        """
        Dependence plot: SHAP value of `feature` vs its raw value.
        Coloured by `interaction_feature` to reveal interaction effects.
        E.g.: SHAP(hour) vs hour, coloured by weather → rush-hour × rain interaction.
        """
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.dependence_plot(
            feature,
            self.shap_values_,
            self.sample_,
            interaction_index=interaction_feature,
            show=False,
            ax=ax,
        )
        ax.set_title(f"SHAP Dependence: {feature}", fontsize=12)
        plt.tight_layout()
        if save:
            fname = f"shap_dependence_{feature.replace('/', '_')}.png"
            path  = self.output_dir / fname
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved → {path}")
        plt.close()

    def plot_top_dependences(self, n: int = 5) -> None:
        """Plot dependence for top-N most important features."""
        top_features = self.global_importance().head(n).index.tolist()
        for feat in top_features:
            self.plot_dependence(feat)

    # ── Cohort analysis ────────────────────────────────────────────────────────

    def shap_by_group(self, group_col: str) -> pd.DataFrame:
        """
        Mean SHAP value per feature per group (e.g. zone cluster, hour bucket).
        Useful for understanding why the model behaves differently across segments.
        """
        df = self.sample_.copy()
        df["__group__"] = df[group_col]
        shap_df = pd.DataFrame(self.shap_values_, columns=self.sample_.columns,
                                index=self.sample_.index)
        shap_df["__group__"] = df["__group__"].values
        result = shap_df.groupby("__group__").mean()
        return result

    def get_html_report(self) -> str:
        """
        Generate standalone HTML force plot for all samples.
        Can be embedded in the main explainability report.
        """
        html = shap.force_plot(
            self.expected_value_,
            self.shap_values_[:200],
            self.sample_.iloc[:200],
        )
        return shap.save_html(str(self.output_dir / "shap_force_all.html"), html)
