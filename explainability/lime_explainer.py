"""
LIME (Local Interpretable Model-agnostic Explanations).
Explains individual predictions by fitting a local linear model
around the point of interest using perturbed samples.

Complements SHAP: LIME is model-agnostic and works as a sanity check.
When SHAP and LIME agree → high confidence in the explanation.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from lime.lime_tabular import LimeTabularExplainer
from typing import Any, Callable


class LIMEExplainer:
    """
    LIME wrapper for tabular regression models.
    Supports any model with a .predict() method.
    """
    def __init__(self, X_train: pd.DataFrame,
                 predict_fn: Callable,
                 categorical_features: list[int] | None = None,
                 output_dir: str = "reports/lime",
                 mode: str = "regression"):
        self.predict_fn  = predict_fn
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(X_train.columns)

        cat_idxs = categorical_features or [
            i for i, c in enumerate(X_train.columns)
            if X_train[c].dtype == "object"
        ]

        self.explainer_ = LimeTabularExplainer(
            training_data=X_train.values.astype(np.float32),
            feature_names=self.feature_names,
            categorical_features=cat_idxs,
            mode=mode,
            random_state=42,
            discretize_continuous=True,
            discretizer="quartile",
        )
        logger.info(f"LIME explainer ready — {len(self.feature_names)} features")

    def explain_instance(self, X_row: pd.Series | np.ndarray,
                         n_features: int = 15,
                         n_samples: int = 5000,
                         save: bool = True,
                         filename: str = "lime_explanation.png") -> dict:
        """
        Explain a single prediction.

        Returns dict with:
            - prediction: model output
            - intercept: local model intercept
            - top_features: list of (feature_condition, weight) pairs
            - local_r2: fit quality of the local linear model
        """
        if isinstance(X_row, pd.Series):
            X_row = X_row.values
        X_row = X_row.astype(np.float32)

        exp = self.explainer_.explain_instance(
            X_row,
            self.predict_fn,
            num_features=n_features,
            num_samples=n_samples,
        )

        prediction = float(self.predict_fn(X_row.reshape(1, -1))[0])
        local_r2   = exp.score
        top        = exp.as_list()

        logger.info(f"LIME — pred={prediction:.2f}, local R²={local_r2:.3f}")

        if save:
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME Explanation — Prediction: {prediction:.1f} rides\n"
                         f"Local R² = {local_r2:.3f}", fontsize=11)
            plt.tight_layout()
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved → {path}")

        return {
            "prediction":   round(prediction, 2),
            "intercept":    round(float(exp.intercept[1]
                                        if hasattr(exp, "intercept") and
                                        isinstance(exp.intercept, dict)
                                        else exp.local_pred[0]), 2),
            "local_r2":     round(float(local_r2), 4),
            "top_features": [(feat, round(float(w), 4)) for feat, w in top],
        }

    def explain_batch(self, X: pd.DataFrame, n: int = 5,
                      n_features: int = 10) -> list[dict]:
        """Explain n randomly sampled rows and return list of explanation dicts."""
        sample = X.sample(n, random_state=42)
        explanations = []
        for i, (idx, row) in enumerate(sample.iterrows()):
            exp = self.explain_instance(
                row.values, n_features=n_features, save=True,
                filename=f"lime_batch_{i}.png"
            )
            exp["row_index"] = idx
            explanations.append(exp)
        return explanations

    def compare_with_shap(self, shap_importance: pd.Series,
                          lime_explanation: dict,
                          top_n: int = 10) -> pd.DataFrame:
        """
        Compare top features from SHAP (global) vs LIME (local) for one prediction.
        Agreement score = Jaccard overlap of top-N feature sets.
        """
        shap_top = set(shap_importance.head(top_n).index)
        lime_top = set(feat.split(" ")[0] for feat, _ in lime_explanation["top_features"][:top_n])

        # Try to match LIME conditions to feature names
        lime_features_matched = set()
        for feat_cond, _ in lime_explanation["top_features"][:top_n]:
            for fname in shap_top:
                if fname in feat_cond:
                    lime_features_matched.add(fname)
                    break

        overlap   = len(shap_top & lime_features_matched)
        union     = len(shap_top | lime_features_matched)
        jaccard   = overlap / union if union > 0 else 0.0

        report = pd.DataFrame({
            "rank":            range(1, top_n + 1),
            "shap_feature":    list(shap_importance.head(top_n).index),
            "lime_condition":  [c for c, _ in lime_explanation["top_features"][:top_n]],
            "lime_weight":     [w for _, w in lime_explanation["top_features"][:top_n]],
        })
        report.attrs["jaccard_overlap"] = round(jaccard, 3)
        logger.info(f"SHAP vs LIME agreement (Jaccard@{top_n}) = {jaccard:.2%}")
        return report
