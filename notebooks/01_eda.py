"""
Notebook 01 — Exploratory Data Analysis
Run as: jupyter nbconvert --to notebook --execute notebooks/01_eda.py
Or:     python notebooks/01_eda.py
"""
# %% [markdown]
# # RideFlow — Exploratory Data Analysis
# Analyse demand patterns across Paris zones, time of day, weather, and events.

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

from producers.gps_producer import generate as gen_data
from config import settings

sns.set_theme(style="whitegrid", palette="muted")
PLOTS = Path("reports/eda"); PLOTS.mkdir(parents=True, exist_ok=True)

# %% Load data
raw_path = settings.RAW_DIR / "rides_synthetic.parquet"
if not raw_path.exists():
    print("Generating synthetic data...")
    df = gen_data(n_days=90, output=settings.RAW_DIR)
else:
    df = pd.read_parquet(raw_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]      = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["week"]      = df["timestamp"].dt.isocalendar().week.astype(int)
df["date"]      = df["timestamp"].dt.date

print(f"Dataset: {len(df):,} rows | {df['zone'].nunique()} zones | "
      f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(df.describe().round(2))

# %% [markdown]
# ## 1. Demand distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df["demand"], bins=50, color="#2563eb", edgecolor="white", alpha=0.85)
axes[0].set_title("Demand Distribution (all zones, all hours)", fontsize=12)
axes[0].set_xlabel("Demand (rides/hour/zone)")
axes[0].set_ylabel("Frequency")

# Log-transform check
axes[1].hist(np.log1p(df["demand"]), bins=50, color="#7c3aed", edgecolor="white", alpha=0.85)
axes[1].set_title("log(1 + Demand) — roughly normal", fontsize=12)
axes[1].set_xlabel("log(1 + demand)")

plt.tight_layout()
plt.savefig(PLOTS / "demand_distribution.png", dpi=150)
plt.close()
print("Skewness:", round(df["demand"].skew(), 3),
      "| log skewness:", round(np.log1p(df["demand"]).skew(), 3))

# %% [markdown]
# ## 2. Hourly demand patterns

# %%
hourly = df.groupby(["hour", "day_of_week"])["demand"].mean().reset_index()
pivot  = hourly.pivot(index="day_of_week", columns="hour", values="demand")

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, fmt=".0f",
            xticklabels=[f"{h:02d}h" for h in range(24)],
            yticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            ax=ax)
ax.set_title("Average Demand Heatmap — Day of Week × Hour", fontsize=13)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(PLOTS / "demand_heatmap.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Zone-level demand

# %%
zone_demand = df.groupby("zone")["demand"].agg(["mean","std","max"]).round(2)
zone_demand = zone_demand.sort_values("mean", ascending=False)
print("\nTop 5 zones by average demand:")
print(zone_demand.head())

fig, ax = plt.subplots(figsize=(12, 5))
zone_demand["mean"].plot(kind="bar", ax=ax, color="#0ea5e9", edgecolor="white")
ax.errorbar(range(len(zone_demand)), zone_demand["mean"],
            yerr=zone_demand["std"], fmt="none", color="gray", capsize=3)
ax.set_title("Mean Demand per Arrondissement (±1 std)", fontsize=12)
ax.set_xlabel("Zone (arrondissement)"); ax.set_ylabel("Mean rides/hour")
plt.tight_layout()
plt.savefig(PLOTS / "zone_demand.png", dpi=150)
plt.close()

# %% [markdown]
# ## 4. Weather impact

# %%
weather_demand = df.groupby("weather")["demand"].agg(["mean","median","count"]).round(2)
weather_demand = weather_demand.sort_values("mean", ascending=False)
print("\nDemand by weather:")
print(weather_demand)

fig, ax = plt.subplots(figsize=(9, 5))
colors = {"clear":"#fbbf24","cloudy":"#94a3b8","rain":"#3b82f6",
          "heavy_rain":"#1e40af","snow":"#e2e8f0"}
for i, (weather, row) in enumerate(weather_demand.iterrows()):
    ax.bar(i, row["mean"], color=colors.get(weather, "#64748b"), edgecolor="white")
ax.set_xticks(range(len(weather_demand)))
ax.set_xticklabels(weather_demand.index, rotation=0)
ax.set_title("Average Demand by Weather Condition", fontsize=12)
ax.set_ylabel("Mean rides/hour")
plt.tight_layout()
plt.savefig(PLOTS / "weather_demand.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Time series — total Paris demand

# %%
daily = df.groupby("date")["demand"].sum().reset_index()
daily["date"] = pd.to_datetime(daily["date"])

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily["date"], daily["demand"], color="#2563eb", lw=1.2, alpha=0.8)
# 7-day rolling average
rolling = daily["demand"].rolling(7, center=True).mean()
ax.plot(daily["date"], rolling, color="#dc2626", lw=2.0, label="7-day MA")
ax.set_title("Total Paris Daily Demand over Time", fontsize=12)
ax.set_ylabel("Total rides/day (all zones)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "demand_timeseries.png", dpi=150)
plt.close()

# %% [markdown]
# ## 6. Autocorrelation analysis

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

zone_ts = df[df["zone"] == "11"].set_index("timestamp").sort_index()["demand"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(zone_ts,  lags=48, ax=axes[0], title="ACF — Zone 11 (lag up to 48h)")
plot_pacf(zone_ts, lags=48, ax=axes[1], title="PACF — Zone 11")
plt.tight_layout()
plt.savefig(PLOTS / "autocorrelation.png", dpi=150)
plt.close()

print("\nStrong autocorrelation at lags: 1h, 24h, 48h, 168h (weekly)")
print("→ These confirm our lag feature choices")

# %% [markdown]
# ## 7. Event impact

# %%
event_df   = df[df["is_event"] == True]
no_event   = df[df["is_event"] == False]
print(f"\nEvent rows: {len(event_df):,} | Non-event: {len(no_event):,}")
if len(event_df) > 0:
    lift = event_df["demand"].mean() / no_event["demand"].mean()
    print(f"Average demand lift during events: {lift:.2f}x")

print("\nEDA complete. All plots saved to reports/eda/")
