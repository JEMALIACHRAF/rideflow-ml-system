"""
Hyperparameter optimization with Optuna.
Uses TPE sampler + MedianPruner for efficient search.
Optimizes model params AND feature selection threshold jointly.
"""
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from loguru import logger
import mlflow


optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMOptimizer:
    """
    Optuna HPO for LightGBM.
    Prunes unpromising trials early using LightGBM's callback + Optuna pruning integration.
    """
    def __init__(self, n_trials: int = 200, timeout: int = 3600,
                 cv_folds: int = 5, metric: str = "rmse"):
        self.n_trials = n_trials
        self.timeout  = timeout
        self.cv_folds = cv_folds
        self.metric   = metric
        self.best_params_: dict = {}
        self.study_: optuna.Study | None = None

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            "objective":        "regression_l1",
            "metric":           self.metric,
            "verbosity":        -1,
            "boosting_type":    trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "n_estimators":     1000,
        }

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, self.metric)

        # Walk-forward CV (no shuffle — temporal data)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=False)
        scores = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                    pruning_callback,
                ],
            )
            preds  = model.predict(X_val)
            rmse   = np.sqrt(np.mean((y_val.values - preds) ** 2))
            scores.append(rmse)

            # Report intermediate for pruning
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def optimize(self, X: pd.DataFrame, y: pd.Series,
                 experiment_name: str = "lgbm-hpo") -> dict:
        logger.info(f"Starting Optuna HPO: {self.n_trials} trials, timeout={self.timeout}s")

        self.study_ = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42, n_startup_trials=20),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
            study_name=experiment_name,
        )
        self.study_.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        self.best_params_ = self.study_.best_params
        logger.success(f"HPO done — best RMSE: {self.study_.best_value:.4f}")
        logger.info(f"Best params: {self.best_params_}")

        # Log to MLflow
        with mlflow.start_run(run_name=f"optuna_{experiment_name}"):
            mlflow.log_params(self.best_params_)
            mlflow.log_metric("best_cv_rmse", self.study_.best_value)
            mlflow.log_metric("n_trials_completed",
                              len([t for t in self.study_.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]))

        return self.best_params_

    def get_importance_plot_data(self) -> pd.DataFrame:
        """Return param importances from the finished study."""
        importances = optuna.importance.get_param_importances(self.study_)
        return pd.Series(importances).sort_values(ascending=False).to_frame("importance")


class XGBOptimizer:
    """Optuna HPO for XGBoost with same TPE + pruning approach."""
    def __init__(self, n_trials: int = 150, timeout: int = 2400):
        self.n_trials = n_trials
        self.timeout  = timeout
        self.best_params_: dict = {}

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            "objective":        "reg:absoluteerror",
            "tree_method":      "hist",
            "booster":          trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state":     42, "n_jobs": -1, "verbosity": 0,
        }

        from sklearn.model_selection import cross_val_score
        from xgboost import XGBRegressor
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error",
                                 n_jobs=-1)
        return float(-scores.mean())

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> dict:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(),
        )
        study.optimize(lambda t: self._objective(t, X, y),
                       n_trials=self.n_trials, timeout=self.timeout)
        self.best_params_ = study.best_params
        logger.success(f"XGB HPO done — best RMSE: {study.best_value:.4f}")
        return self.best_params_
