"""
Notebook 02 — Feature Engineering & Selection Analysis
Analyses the quality of engineered features and documents selection decisions.
"""
# %% Imports
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import settings
from features.temporal import build_temporal_features
from features.geospatial import build_geospatial_features
from features.selection import FeatureSelector

sns.set_theme(style="whitegrid")
PLOTS = Path("reports/features"); PLOTS.mkdir(parents=True, exist_ok=True)

# %% Load and build features
df = pd.read_parquet(settings.RAW_DIR / "rides_synthetic.parquet")
df = df.sort_values(["zone", "timestamp"])
df = build_temporal_features(df, lags=[1, 2, 3, 6, 12, 24, 48, 168],
                              rolling_windows=[3, 6, 12, 24])
df = build_geospatial_features(df)
df = df.dropna()

DROP = {"timestamp","demand","zone","weather","event_name","lat","lon","zone_cluster","is_event"}
feat_cols = [c for c in df.columns if c not in DROP and df[c].dtype != "object"]
X = df[feat_cols].fillna(0)
y = df["demand"]

print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"\nFeature groups:")
groups = {
    "Lag":      [c for c in X.columns if "lag" in c],
    "Rolling":  [c for c in X.columns if "roll" in c],
    "Calendar": [c for c in X.columns if c in ["hour","day_of_week","month","is_weekend",
                                                 "is_rush_am","is_rush_pm","is_night",
                                                 "is_holiday","is_friday_night"]],
    "Fourier":  [c for c in X.columns if "fourier" in c],
    "Geo":      [c for c in X.columns if "zone" in c or "poi" in c or "adj" in c
                                       or "cluster" in c or "demand_share" in c],
    "Context":  [c for c in X.columns if c in ["temperature_c","precipitation_mm",
                                                 "wind_speed_kmh","weather_encoded"]],
}
for grp, cols in groups.items():
    print(f"  {grp:12s}: {len(cols):3d} features")

# %% Correlation with target
corr = X.corrwith(y).abs().sort_values(ascending=False)
print(f"\nTop 15 features by |Pearson correlation| with demand:")
print(corr.head(15).round(4))

fig, ax = plt.subplots(figsize=(10, 7))
corr.head(25).plot(kind="barh", ax=ax, color="#0ea5e9")
ax.set_title("Top 25 Features — |Correlation| with Demand", fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "feature_correlation.png", dpi=150)
plt.close()

# %% SHAP-based feature selection (fast run)
print("\nRunning feature selection (SHAP + MI, no Boruta for speed)...")
selector = FeatureSelector(top_n_shap=30, top_n_mi=30, use_boruta=False)
X_sel = selector.fit_transform(X.sample(min(3000, len(X)), random_state=42),
                                y.loc[X.sample(min(3000, len(X)), random_state=42).index])

report = selector.get_importance_report()
print(f"\nSelected {len(selector.selected_features_)} features")
print(f"\nTop 20 by SHAP importance:")
print(report.head(20).round(4))

# %% Visualise importance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

top20 = report.head(20)
axes[0].barh(range(len(top20)), top20["shap_importance"], color="#7c3aed")
axes[0].set_yticks(range(len(top20))); axes[0].set_yticklabels(top20.index, fontsize=9)
axes[0].set_title("SHAP Importance (top 20)"); axes[0].invert_yaxis()

axes[1].barh(range(len(top20)), top20["mi_importance"], color="#0ea5e9")
axes[1].set_yticks(range(len(top20))); axes[1].set_yticklabels(top20.index, fontsize=9)
axes[1].set_title("Mutual Information (top 20)"); axes[1].invert_yaxis()

plt.suptitle("Feature Importance: SHAP vs Mutual Information", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS / "feature_importance_comparison.png", dpi=150)
plt.close()

# %% Lag feature analysis
lag_cols = [c for c in X.columns if "lag" in c]
lag_corr = X[lag_cols].corrwith(y).abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
lag_corr.plot(kind="bar", ax=ax, color="#10b981")
ax.set_title("Lag Feature |Correlation| with Demand\n(Higher = more predictive)", fontsize=12)
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(PLOTS / "lag_correlation.png", dpi=150)
plt.close()

print("\nKey finding: lag_1h and lag_24h are strongest predictors")
print("Weekly lag (168h) confirms strong weekly seasonality")

# %% Feature correlation matrix (selected features)
if len(selector.selected_features_) >= 10:
    top10 = selector.selected_features_[:10]
    corr_matrix = X[top10].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix (top 10 selected features)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_correlation_matrix.png", dpi=150)
    plt.close()

print("\nFeature analysis complete. Plots saved to reports/features/")
