"""
Feature selection module.
Combines SHAP importance, mutual information, and Boruta algorithm.
Outputs a ranked feature list and a reduced DataFrame.
"""
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from loguru import logger
from typing import Optional


class BorutaSelector:
    """
    Simplified Boruta algorithm.
    Compares real feature importances against shadow (shuffled) features.
    Features that consistently beat the best shadow feature are confirmed.
    """
    def __init__(self, n_estimators: int = 100, n_trials: int = 20,
                 alpha: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.n_trials = n_trials
        self.alpha = alpha
        self.random_state = random_state
        self.confirmed_features_: list = []
        self.tentative_features_: list = []
        self.rejected_features_: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaSelector":
        np.random.seed(self.random_state)
        feature_names = list(X.columns)
        hits = {f: 0 for f in feature_names}

        for trial in range(self.n_trials):
            # Create shadow features (shuffled copies)
            shadow = X.copy()
            shadow.columns = [f"shadow_{c}" for c in X.columns]
            for col in shadow.columns:
                shadow[col] = shadow[col].sample(frac=1, random_state=trial).values

            X_aug = pd.concat([X, shadow], axis=1)
            model = RandomForestRegressor(n_estimators=self.n_estimators,
                                          random_state=self.random_state + trial, n_jobs=-1)
            model.fit(X_aug, y)
            importances = pd.Series(model.feature_importances_, index=X_aug.columns)

            shadow_max = importances[[c for c in X_aug.columns if c.startswith("shadow_")]].max()
            for f in feature_names:
                if importances[f] > shadow_max:
                    hits[f] += 1

            if (trial + 1) % 5 == 0:
                logger.debug(f"Boruta trial {trial + 1}/{self.n_trials}")

        threshold = self.n_trials * (1 - self.alpha)
        for f in feature_names:
            if hits[f] >= threshold:
                self.confirmed_features_.append(f)
            elif hits[f] >= self.n_trials * self.alpha:
                self.tentative_features_.append(f)
            else:
                self.rejected_features_.append(f)

        logger.info(f"Boruta: {len(self.confirmed_features_)} confirmed, "
                    f"{len(self.tentative_features_)} tentative, "
                    f"{len(self.rejected_features_)} rejected")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        selected = self.confirmed_features_ + self.tentative_features_
        return X[[c for c in selected if c in X.columns]]


class FeatureSelector:
    """
    Ensemble feature selection combining SHAP, mutual information, and Boruta.
    Uses a voting mechanism: features selected by 2+ methods are kept.
    """
    def __init__(self, top_n_shap: int = 40, top_n_mi: int = 40,
                 use_boruta: bool = True, random_state: int = 42):
        self.top_n_shap = top_n_shap
        self.top_n_mi = top_n_mi
        self.use_boruta = use_boruta
        self.random_state = random_state
        self.selected_features_: list = []
        self.shap_importance_: Optional[pd.Series] = None
        self.mi_importance_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        logger.info("Running SHAP feature importance...")
        shap_features = self._shap_selection(X, y)

        logger.info("Running mutual information...")
        mi_features = self._mi_selection(X, y)

        votes = {}
        for f in X.columns:
            votes[f] = int(f in shap_features) + int(f in mi_features)

        if self.use_boruta:
            logger.info("Running Boruta...")
            boruta = BorutaSelector(n_trials=15, random_state=self.random_state)
            boruta.fit(X.sample(min(5000, len(X)), random_state=self.random_state), y)
            for f in boruta.confirmed_features_:
                votes[f] = votes.get(f, 0) + 1

        # Keep features with 2+ votes (majority agreement)
        self.selected_features_ = [f for f, v in votes.items() if v >= 2]

        # Always keep at minimum top SHAP features
        if len(self.selected_features_) < 20:
            self.selected_features_ = list(shap_features[:20])

        logger.success(f"Selected {len(self.selected_features_)} features out of {len(X.columns)}")
        return self

    def _shap_selection(self, X: pd.DataFrame, y: pd.Series) -> list:
        sample = X.sample(min(3000, len(X)), random_state=self.random_state)
        y_sample = y.loc[sample.index]
        model = lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        model.fit(sample, y_sample)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        self.shap_importance_ = pd.Series(
            np.abs(shap_values).mean(axis=0), index=X.columns
        ).sort_values(ascending=False)
        return list(self.shap_importance_.head(self.top_n_shap).index)

    def _mi_selection(self, X: pd.DataFrame, y: pd.Series) -> list:
        sample = X.sample(min(5000, len(X)), random_state=self.random_state)
        y_sample = y.loc[sample.index]
        mi = mutual_info_regression(sample.fillna(0), y_sample,
                                    random_state=self.random_state)
        self.mi_importance_ = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        return list(self.mi_importance_.head(self.top_n_mi).index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.selected_features_ if c in X.columns]
        return X[cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_importance_report(self) -> pd.DataFrame:
        """Merge SHAP and MI importance scores into one report."""
        report = pd.DataFrame({
            "shap_importance": self.shap_importance_ or pd.Series(dtype=float),
            "mi_importance":   self.mi_importance_  or pd.Series(dtype=float),
        }).fillna(0)
        report["selected"] = report.index.isin(self.selected_features_)
        return report.sort_values("shap_importance", ascending=False)
