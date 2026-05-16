"""
Notebook 03 — Model Comparison & Explainability Showcase
Trains all models, benchmarks them, and walks through every explainability tool.
This is the "show-off" notebook for GitHub — rich outputs, clear narrative.
"""
# %% Imports
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console
from rich.table import Table

from config import settings
from producers.gps_producer import generate as gen_data
from features.temporal import build_temporal_features
from features.geospatial import build_geospatial_features
from models.tree_based.lgbm_model import LGBMDemandModel
from models.tree_based.xgb_catboost_models import XGBDemandModel, CatBoostDemandModel
from models.ensemble.stacking import VotingEnsemble, WeightedBlending
from optimization.cv.timeseries_cv import WalkForwardCV, cross_validate_model
from optimization.calibration.calibrators import compare_calibrators, calibration_curve_data
from evaluation.metrics import compute_all_metrics, evaluate_by_zone, simulate_revenue, psi
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.pdp_anchors_dice import PDPICEExplainer

sns.set_theme(style="whitegrid")
PLOTS = Path("reports/model_comparison"); PLOTS.mkdir(parents=True, exist_ok=True)
cons  = Console()

# %% ─── 1. DATA PREPARATION ───────────────────────────────────────────────────
print("=" * 60)
print("1. Loading and preparing data")
print("=" * 60)

raw_path = settings.RAW_DIR / "rides_synthetic.parquet"
if not raw_path.exists():
    df_raw = gen_data(n_days=90, output=settings.RAW_DIR)
else:
    df_raw = pd.read_parquet(raw_path)

df = df_raw.sort_values(["zone", "timestamp"])
df = build_temporal_features(df, lags=[1, 2, 3, 6, 12, 24, 48, 168],
                              rolling_windows=[3, 6, 12, 24])
df = build_geospatial_features(df)
df = df.dropna()

DROP = {"timestamp","demand","zone","weather","event_name",
        "lat","lon","zone_cluster","is_event"}
feat_cols = [c for c in df.columns if c not in DROP and df[c].dtype != "object"]

df["timestamp"] = pd.to_datetime(df["timestamp"])
train_df = df[df["timestamp"] < "2024-10-01"]
val_df   = df[(df["timestamp"] >= "2024-10-01") & (df["timestamp"] < "2024-12-01")]
test_df  = df[df["timestamp"] >= "2024-12-01"]

X_train, y_train = train_df[feat_cols].fillna(0), train_df["demand"]
X_val,   y_val   = val_df[feat_cols].fillna(0),   val_df["demand"]
X_test,  y_test  = test_df[feat_cols].fillna(0),  test_df["demand"]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Features: {len(feat_cols)}")

# %% ─── 2. TRAIN ALL MODELS ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Training models")
print("=" * 60)

models = {
    "LightGBM":  LGBMDemandModel({"n_estimators": 300, "num_leaves": 63}),
    "XGBoost":   XGBDemandModel({"n_estimators": 300, "booster": "gbtree"}),
    "CatBoost":  CatBoostDemandModel({"iterations": 300}),
}

trained = {}
all_metrics = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    metrics = model.evaluate(X_test, y_test)
    trained[name] = model
    all_metrics[name] = metrics
    print(f"  {name}: MAPE={metrics['mape']:.2%}  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.3f}")

# Voting ensemble
print("  Training VotingEnsemble...")
voter = VotingEnsemble(list(trained.values()))
voter.fit(X_train, y_train, eval_set=(X_val, y_val))
all_metrics["VotingEnsemble"] = voter.evaluate(X_test, y_test)
trained["VotingEnsemble"] = voter

# Weighted blending
print("  Training WeightedBlending...")
blender = WeightedBlending(list(models.values()))
blender.fit(X_train, y_train, eval_set=(X_val, y_val))
all_metrics["WeightedBlending"] = blender.evaluate(X_test, y_test)
trained["WeightedBlending"] = blender

# %% ─── 3. LEADERBOARD ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Model Leaderboard")
print("=" * 60)

table = Table(title="Model Benchmark — Test Set", show_header=True, header_style="bold blue")
for col in ["Model", "MAPE", "RMSE", "MAE", "R²"]:
    table.add_column(col)
for name, m in sorted(all_metrics.items(), key=lambda x: x[1]["mape"]):
    table.add_row(name, f"{m['mape']:.2%}", f"{m['rmse']:.2f}",
                  f"{m['mae']:.2f}", f"{m['r2']:.3f}")
cons.print(table)

# Visualise leaderboard
names  = list(all_metrics.keys())
mapes  = [all_metrics[n]["mape"] * 100 for n in names]
colors = ["#16a34a" if m == min(mapes) else "#2563eb" for m in mapes]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, mapes, color=colors, edgecolor="white", width=0.6)
ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
ax.set_title("Model Comparison — MAPE on Test Set (lower is better)", fontsize=12)
ax.set_ylabel("MAPE (%)")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(PLOTS / "model_leaderboard.png", dpi=150)
plt.close()

# %% ─── 4. WALK-FORWARD CROSS-VALIDATION ─────────────────────────────────────
print("\n" + "=" * 60)
print("4. Walk-Forward Cross-Validation")
print("=" * 60)

cv = WalkForwardCV(n_splits=4, gap_hours=24, min_train_size=1000)
cv_scores = []

for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_train)):
    X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    m = LGBMDemandModel({"n_estimators": 200, "num_leaves": 31})
    m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    metrics = m.evaluate(X_vl, y_vl)
    metrics["fold"] = fold_i + 1
    metrics["train_size"] = len(tr_idx)
    cv_scores.append(metrics)
    print(f"  Fold {fold_i+1}: MAPE={metrics['mape']:.2%}  RMSE={metrics['rmse']:.2f}")

cv_df = pd.DataFrame(cv_scores)
print(f"\n  CV MAPE: {cv_df['mape'].mean():.2%} ± {cv_df['mape'].std():.2%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(cv_df["fold"], cv_df["mape"] * 100, "o-", color="#2563eb", lw=2)
axes[0].fill_between(cv_df["fold"],
                      (cv_df["mape"].mean() - cv_df["mape"].std()) * 100,
                      (cv_df["mape"].mean() + cv_df["mape"].std()) * 100,
                      alpha=0.15, color="#2563eb")
axes[0].set_title("Walk-Forward CV — MAPE per Fold"); axes[0].set_ylabel("MAPE (%)")
axes[0].set_xlabel("Fold")

axes[1].bar(cv_df["fold"], cv_df["train_size"], color="#7c3aed")
axes[1].set_title("Training Size (expanding window)"); axes[1].set_xlabel("Fold")
axes[1].set_ylabel("# training samples")
plt.tight_layout()
plt.savefig(PLOTS / "walkforward_cv.png", dpi=150)
plt.close()

# %% ─── 5. CALIBRATION ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Model Calibration")
print("=" * 60)

best_name  = min(all_metrics, key=lambda k: all_metrics[k]["mape"])
best_model = trained[best_name]
raw_preds  = best_model.predict(X_val)
cal_report = compare_calibrators(raw_preds, y_val.values)
print(cal_report.to_string(index=False))

cal_data = calibration_curve_data(raw_preds, y_val.values, n_bins=15)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(cal_data["mean_pred"], cal_data["mean_actual"],
                s=cal_data["count"]/5, alpha=0.7, color="#2563eb")
max_val = max(cal_data[["mean_pred","mean_actual"]].max())
axes[0].plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect calibration")
axes[0].set_xlabel("Mean predicted demand"); axes[0].set_ylabel("Mean actual demand")
axes[0].set_title("Reliability Diagram"); axes[0].legend()

axes[1].bar(range(len(cal_data)), cal_data["bias"], color=["#dc2626" if b > 0 else "#16a34a"
                                                             for b in cal_data["bias"]])
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("Prediction Bias by Demand Bucket")
axes[1].set_xlabel("Demand quantile bin"); axes[1].set_ylabel("Bias (pred - actual)")
plt.tight_layout()
plt.savefig(PLOTS / "calibration.png", dpi=150)
plt.close()

# %% ─── 6. ZONE-LEVEL EVALUATION ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Zone-Level Evaluation")
print("=" * 60)

test_with_preds = test_df.copy()
test_with_preds["pred"] = best_model.predict(X_test)
zone_metrics = evaluate_by_zone(test_with_preds)
print(zone_metrics[["mape","rmse","r2","n_samples"]].head(10).round(4))

fig, ax = plt.subplots(figsize=(12, 5))
zone_metrics["mape"].sort_values().plot(kind="bar", ax=ax, color="#f59e0b")
ax.axhline(zone_metrics["mape"].mean(), color="#dc2626", lw=1.5,
            linestyle="--", label=f"Mean MAPE: {zone_metrics['mape'].mean():.2%}")
ax.set_title("Per-Zone MAPE — Test Set", fontsize=12)
ax.set_ylabel("MAPE"); ax.set_xlabel("Arrondissement")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "zone_mape.png", dpi=150)
plt.close()

# %% ─── 7. REVENUE BACKTEST ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Revenue Backtesting")
print("=" * 60)

revenue = simulate_revenue(test_with_preds)
for k, v in revenue.items():
    print(f"  {k}: {v}")

# %% ─── 8. SHAP EXPLAINABILITY ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. SHAP Explainability")
print("=" * 60)

lgbm_model = trained["LightGBM"]
shap_exp = SHAPExplainer(
    model=lgbm_model.model,
    model_type="tree",
    output_dir="reports/shap",
)
shap_exp.fit(X_test, sample_size=1000)

importance = shap_exp.global_importance()
print("Top 10 features by mean |SHAP|:")
print(importance.head(10).round(4))

shap_exp.plot_summary(max_display=20, save=True)
shap_exp.plot_bar_importance(max_display=20, save=True)
shap_exp.plot_top_dependences(n=3)

single_exp = shap_exp.explain_single(X_test.iloc[0], save=True)
print(f"\nSample prediction: {single_exp['prediction']} rides")
print("Top contributors:")
for feat, val in list(single_exp["top_contributors"].items())[:5]:
    direction = "↑" if val > 0 else "↓"
    print(f"  {direction} {feat}: {val:+.2f}")

# %% ─── 9. LIME EXPLAINABILITY ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. LIME Explainability")
print("=" * 60)

lime_exp = LIMEExplainer(
    X_train=X_train,
    predict_fn=lgbm_model.predict,
    output_dir="reports/lime",
)
lime_result = lime_exp.explain_instance(
    X_test.iloc[0], n_features=10, n_samples=2000,
    save=True, filename="lime_explanation.png"
)
print(f"LIME prediction: {lime_result['prediction']}  |  Local R²: {lime_result['local_r2']:.3f}")
print("Top LIME features:")
for feat, w in lime_result["top_features"][:5]:
    print(f"  {feat}: {w:+.4f}")

# SHAP vs LIME agreement
agreement = lime_exp.compare_with_shap(importance, lime_result, top_n=10)
print(f"\nSHAP vs LIME Jaccard@10: {agreement.attrs.get('jaccard_overlap', 'N/A')}")

# %% ─── 10. PDP / ICE ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. PDP & ICE Plots")
print("=" * 60)

pdp_exp = PDPICEExplainer(
    model=lgbm_model,
    X_train=X_train,
    output_dir="reports/pdp",
)

for feature in ["hour", "temperature_c", "demand_lag_1h"]:
    if feature in X_train.columns:
        result = pdp_exp.plot_pdp(feature, save=True)
        pdp_exp.plot_ice(feature, n_samples=100, save=True)
        print(f"  PDP+ICE saved for: {feature}")

# 2D interaction
if "hour" in X_train.columns and "temperature_c" in X_train.columns:
    pdp_exp.plot_2d_pdp("hour", "temperature_c", n_points=15, save=True)
    print("  2D PDP saved: hour × temperature_c")

# %% ─── 11. PSI DRIFT SIMULATION ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. PSI Distribution Drift Simulation")
print("=" * 60)

# Simulate how PSI increases as data shifts over time
np.random.seed(42)
shifts   = np.linspace(0, 3, 10)
psi_vals = []
ref_data = np.random.lognormal(2.5, 0.4, 1000)
for shift in shifts:
    cur_data = np.random.lognormal(2.5 + shift * 0.2, 0.4 + shift * 0.05, 1000)
    psi_vals.append(psi(ref_data, cur_data))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(shifts, psi_vals, "o-", color="#2563eb", lw=2)
ax.axhline(0.1, color="#f59e0b", lw=1.5, linestyle="--", label="Warn threshold (0.1)")
ax.axhline(0.2, color="#dc2626", lw=1.5, linestyle="--", label="Retrain threshold (0.2)")
ax.fill_between(shifts, 0, psi_vals, alpha=0.1, color="#2563eb")
ax.set_xlabel("Distribution Shift Magnitude"); ax.set_ylabel("PSI")
ax.set_title("PSI as Function of Distribution Shift", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "psi_simulation.png", dpi=150)
plt.close()

print(f"\nPSI values: {[round(p, 3) for p in psi_vals]}")
print("PSI > 0.2 triggers automatic model retraining in production")

print("\n" + "=" * 60)
print("Notebook 03 complete — all outputs saved to reports/")
print("=" * 60)
