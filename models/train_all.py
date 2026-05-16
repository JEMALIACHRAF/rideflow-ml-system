"""
Master training script.
Trains all 5 models, runs HPO on the best one, builds stacking ensemble,
generates explainability report, and logs everything to MLflow.

Usage:
    python models/train_all.py --experiment rideflow-v1 --hpo
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import mlflow
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from config import settings
from producers.gps_producer import generate as gen_data
from features.temporal import build_temporal_features
from features.geospatial import build_geospatial_features
from features.selection import FeatureSelector
from models.tree_based.lgbm_model import LGBMDemandModel
from models.tree_based.xgb_catboost_models import XGBDemandModel, CatBoostDemandModel
from models.ensemble.stacking import StackingEnsemble, WeightedBlending, VotingEnsemble
from optimization.hpo.optuna_lgbm import LGBMOptimizer
from optimization.cv.timeseries_cv import WalkForwardCV
from optimization.calibration.calibrators import compare_calibrators
from evaluation.metrics import compute_all_metrics, evaluate_by_zone, simulate_revenue
from explainability.shap_explainer import SHAPExplainer
from explainability.report_generator import generate_report

app  = typer.Typer()
cons = Console()


def load_or_generate_data() -> pd.DataFrame:
    path = settings.RAW_DIR / "rides_synthetic.parquet"
    if path.exists():
        logger.info(f"Loading existing data: {path}")
        return pd.read_parquet(path)
    logger.info("Generating synthetic data (90 days)...")
    return gen_data(n_days=90, output=settings.RAW_DIR)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building features...")
    df = df.sort_values(["zone", "timestamp"])
    df = build_temporal_features(df)
    df = build_geospatial_features(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.dropna()


def split_data(df: pd.DataFrame):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    logger.info(f"Train: {train['timestamp'].min().date()} → {train['timestamp'].max().date()}")
    logger.info(f"Val:   {val['timestamp'].min().date()} → {val['timestamp'].max().date()}")
    logger.info(f"Test:  {test['timestamp'].min().date()} → {test['timestamp'].max().date()}")
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"timestamp", "demand", "zone", "weather", "event_name",
            "lat", "lon", "zone_cluster", "is_event"}
    return [c for c in df.columns if c not in drop and df[c].dtype != "object"]


@app.command()
def train(
    experiment: str = typer.Option("rideflow-v1", help="MLflow experiment name"),
    hpo:        bool = typer.Option(False, help="Run Optuna HPO on LightGBM"),
    n_hpo_trials: int = typer.Option(50, help="Number of HPO trials"),
):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_or_generate_data()
    df = build_features(df)
    train_df, val_df, test_df = split_data(df)

    feat_cols = get_feature_cols(df)
    X_train, y_train = train_df[feat_cols].fillna(0), train_df["demand"]
    X_val,   y_val   = val_df[feat_cols].fillna(0),   val_df["demand"]
    X_test,  y_test  = test_df[feat_cols].fillna(0),  test_df["demand"]

    # ── Feature selection ─────────────────────────────────────────────────────
    logger.info("Running feature selection...")
    selector = FeatureSelector(use_boruta=False)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel   = selector.transform(X_val)
    X_test_sel  = selector.transform(X_test)
    selected_features = selector.selected_features_
    logger.success(f"Selected {len(selected_features)} features")

    # ── HPO (optional) ────────────────────────────────────────────────────────
    lgbm_params = {}
    if hpo:
        logger.info("Running Optuna HPO for LightGBM...")
        optimizer = LGBMOptimizer(n_trials=n_hpo_trials, timeout=600)
        lgbm_params = optimizer.optimize(X_train_sel, y_train, experiment)

    # ── Train all models ──────────────────────────────────────────────────────
    models_to_train = {
        "LightGBM":  LGBMDemandModel(lgbm_params or None),
        "XGBoost":   XGBDemandModel(),
        "CatBoost":  CatBoostDemandModel(),
    }

    trained_models = {}
    all_metrics    = {}

    with mlflow.start_run(run_name="all_models"):
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            with mlflow.start_run(run_name=name, nested=True):
                model.fit(X_train_sel, y_train, eval_set=(X_val_sel, y_val))
                metrics = model.evaluate(X_test_sel, y_test)
                model.log_to_mlflow(metrics)
                trained_models[name] = model
                all_metrics[name]    = metrics
                model.save(settings.MODELS_DIR / f"{name.lower()}.pkl")

        # ── Stacking ensemble ─────────────────────────────────────────────────
        logger.info("Training stacking ensemble...")
        base_models = list(trained_models.values())
        stacker = StackingEnsemble(base_models, n_folds=3)
        stacker.fit(X_train_sel, y_train)
        stack_metrics = stacker.evaluate(X_test_sel, y_test)
        trained_models["Stacking"] = stacker
        all_metrics["Stacking"]    = stack_metrics
        stacker.save(settings.MODELS_DIR / "stacking.pkl")

        # ── Weighted blending ─────────────────────────────────────────────────
        logger.info("Training weighted blending...")
        blender = WeightedBlending(base_models)
        blender.fit(X_train_sel, y_train, eval_set=(X_val_sel, y_val))
        blend_metrics = blender.evaluate(X_test_sel, y_test)
        trained_models["Blending"] = blender
        all_metrics["Blending"]    = blend_metrics

        # ── Calibration comparison ────────────────────────────────────────────
        best_model_name = min(all_metrics, key=lambda k: all_metrics[k]["mape"])
        best_model = trained_models[best_model_name]
        raw_preds  = best_model.predict(X_val_sel)
        cal_report = compare_calibrators(raw_preds, y_val.values)
        logger.info(f"\nCalibration report:\n{cal_report}")

        # ── Revenue backtest ──────────────────────────────────────────────────
        test_with_preds = test_df.copy()
        test_with_preds["pred"] = best_model.predict(X_test_sel)
        revenue = simulate_revenue(test_with_preds)
        mlflow.log_metrics({f"revenue_{k}": v for k, v in revenue.items()
                             if isinstance(v, (int, float))})

        # ── Print leaderboard ─────────────────────────────────────────────────
        table = Table(title="Model Leaderboard", show_header=True)
        table.add_column("Model"); table.add_column("MAPE"); table.add_column("RMSE"); table.add_column("R²")
        for name, m in sorted(all_metrics.items(), key=lambda x: x[1]["mape"]):
            table.add_row(name, f"{m['mape']:.3%}", f"{m['rmse']:.2f}", f"{m['r2']:.3f}")
        cons.print(table)

        # ── Explainability ────────────────────────────────────────────────────
        logger.info(f"Generating SHAP report for best model: {best_model_name}")
        if hasattr(best_model, "model"):  # tree-based
            shap_exp = SHAPExplainer(
                best_model.model,
                model_type="tree",
                output_dir=str(settings.REPORTS_DIR / "shap"),
            )
            shap_exp.fit(X_test_sel)
            shap_exp.plot_summary(save=True)
            shap_exp.plot_bar_importance(save=True)
            shap_exp.plot_top_dependences(n=5)

            # Waterfall for first test sample
            shap_exp.explain_single(X_test_sel.iloc[0], save=True)

            # Full HTML report
            generate_report(
                model_name=best_model_name,
                shap_importance=shap_exp.global_importance(),
                metrics=all_metrics[best_model_name],
                shap_dir=settings.REPORTS_DIR / "shap",
                lime_dir=settings.REPORTS_DIR / "lime",
                pdp_dir=settings.REPORTS_DIR / "pdp",
                output_path=settings.REPORTS_DIR / "explainability_report.html",
            )
            mlflow.log_artifact(str(settings.REPORTS_DIR / "explainability_report.html"))

        # Save best model as default
        best_model.save(settings.MODELS_DIR / "best_model.pkl")
        mlflow.log_param("best_model", best_model_name)
        logger.success(f"Training complete. Best model: {best_model_name} "
                       f"(MAPE={all_metrics[best_model_name]['mape']:.2%})")


if __name__ == "__main__":
    app()
