"""
RideFlow — Streamlit dashboard.
Live demand scoring, surge pricing, SHAP explanation per prediction.

Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RideFlow ML Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e40af, #3b82f6);
    color: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center; margin-bottom: 0.5rem;
}
.metric-card .val { font-size: 2rem; font-weight: 700; }
.metric-card .lbl { font-size: 0.75rem; opacity: 0.85; text-transform: uppercase; }
.surge-low    { color: #16a34a; font-weight: 700; }
.surge-medium { color: #f59e0b; font-weight: 700; }
.surge-high   { color: #dc2626; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
st.sidebar.title("🚗 RideFlow ML")
st.sidebar.markdown("Dynamic demand & pricing dashboard")
st.sidebar.divider()

zone = st.sidebar.selectbox(
    "Paris Zone (Arrondissement)", [str(i) for i in range(1, 21)],
    index=10, format_func=lambda z: f"{z}e arr."
)
hour = st.sidebar.slider("Hour of Day", 0, 23, datetime.now().hour)
day  = st.sidebar.selectbox(
    "Day of Week", list(range(7)),
    format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d]
)
month   = st.sidebar.selectbox("Month", list(range(1, 13)))
weather = st.sidebar.selectbox(
    "Weather", ["clear","cloudy","rain","heavy_rain","snow"],
    format_func=lambda w: {"clear":"☀️ Clear","cloudy":"☁️ Cloudy","rain":"🌧️ Rain",
                            "heavy_rain":"⛈️ Heavy Rain","snow":"❄️ Snow"}[w]
)
temp    = st.sidebar.slider("Temperature (°C)", -5, 35, 15)
precip  = st.sidebar.slider("Precipitation (mm)", 0.0, 30.0, 0.0, 0.5)
is_event = st.sidebar.checkbox("Major Event in Zone")

lag_1h   = st.sidebar.slider("Demand last hour",   0, 60, 12)
lag_24h  = st.sidebar.slider("Demand same time yesterday", 0, 60, 11)
lag_168h = st.sidebar.slider("Demand same time last week",  0, 60, 10)
roll24   = st.sidebar.slider("24h rolling mean",   0, 60, 11)

st.sidebar.divider()
predict_btn = st.sidebar.button("🔮 Predict Demand & Price", type="primary", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚗 RideFlow ML — Live Demand & Pricing")
st.caption(f"Zone {zone}e arr. | {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]} "
           f"{hour:02d}:00 | {weather}")

# ── API call ──────────────────────────────────────────────────────────────────
def call_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict/demand", json=payload, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def call_price(zone: str, demand: float) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/price",
                          json={"zone": zone, "predicted_demand": demand,
                                "zone_baseline": 10.0}, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def call_health() -> dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).json()
    except Exception:
        return {"status": "unreachable"}

# ── Health check ──────────────────────────────────────────────────────────────
health = call_health()
status_col = st.columns([1, 3])[0]
if health.get("status") == "ok":
    st.success(f"✅ API connected | Model: {health.get('model_version','unknown')}")
else:
    st.warning("⚠️ API offline — showing simulated predictions")

# ── Prediction ────────────────────────────────────────────────────────────────
payload = {
    "zone": zone, "hour": hour, "day_of_week": day,
    "month": month, "weather": weather, "temperature_c": float(temp),
    "precipitation_mm": float(precip), "is_event": is_event,
    "demand_lag_1h": float(lag_1h), "demand_lag_24h": float(lag_24h),
    "demand_lag_168h": float(lag_168h), "demand_roll24h_mean": float(roll24),
}

# Auto-predict on load or button
if predict_btn or True:
    result = call_predict(payload)

    if result is None:
        # Fallback mock
        base = lag_1h * 0.6 + lag_24h * 0.3 + (4 if is_event else 0)
        weather_mult = {"clear":1.0,"cloudy":1.05,"rain":1.4,"heavy_rain":1.7,"snow":1.3}
        mock_demand = max(0, base * weather_mult.get(weather, 1.0) * np.random.lognormal(0, 0.1))
        result = {
            "predicted_demand": round(mock_demand, 1),
            "confidence_low":   round(mock_demand * 0.8, 1),
            "confidence_high":  round(mock_demand * 1.2, 1),
            "latency_ms":       0, "cached": False,
        }

    demand  = result["predicted_demand"]
    price_r = call_price(zone, demand)
    surge   = price_r["surge_multiplier"] if price_r else 1.0
    price   = price_r["final_price_eur"]  if price_r else 8.0

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{demand:.0f}</div>
            <div class="lbl">Predicted rides/hour</div></div>""",
            unsafe_allow_html=True)
    with c2:
        surge_cls = "surge-low" if surge <= 1.0 else "surge-medium" if surge < 1.8 else "surge-high"
        st.markdown(f"""<div class="metric-card">
            <div class="val">{surge:.1f}×</div>
            <div class="lbl">Surge Multiplier</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="val">€{price:.2f}</div>
            <div class="lbl">Final Price / Ride</div></div>""",
            unsafe_allow_html=True)
    with c4:
        latency = result.get("latency_ms", 0)
        cached  = "⚡ cached" if result.get("cached") else "🔄 computed"
        st.markdown(f"""<div class="metric-card">
            <div class="val">{latency:.0f}ms</div>
            <div class="lbl">API Latency ({cached})</div></div>""",
            unsafe_allow_html=True)

    st.markdown("")

    # ── Confidence interval bar ────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Prediction Interval")
        low   = result["confidence_low"]
        high  = result["confidence_high"]
        fig, ax = plt.subplots(figsize=(8, 1.8))
        ax.barh(["Demand"], [high - low], left=[low], height=0.4,
                color="#bfdbfe", label=f"80% interval [{low:.0f}–{high:.0f}]")
        ax.axvline(demand, color="#1e40af", lw=2.5, label=f"Prediction: {demand:.0f}")
        ax.set_xlim(max(0, low - 5), high + 5)
        ax.set_xlabel("Rides/hour"); ax.legend(loc="upper right", fontsize=9)
        ax.set_yticks([]); ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Pricing Breakdown")
        st.markdown(f"""
| Component | Value |
|---|---|
| Base fare | €8.00 |
| Demand ratio | {price_r['demand_ratio'] if price_r else 'N/A'} |
| Surge multiplier | **{surge:.2f}×** |
| **Final price** | **€{price:.2f}** |
""")

    # ── Demand curve by hour ───────────────────────────────────────────────────
    st.subheader("Simulated Demand Profile — All Hours Today")
    hour_mults = {
        0:0.3,1:0.2,2:0.15,3:0.1,4:0.1,5:0.2,6:0.5,7:1.2,8:1.8,9:1.4,
        10:1.0,11:1.1,12:1.3,13:1.2,14:1.0,15:1.0,16:1.1,17:1.6,
        18:1.9,19:1.7,20:1.4,21:1.3,22:1.1,23:0.7,
    }
    w_mult = {"clear":1.0,"cloudy":1.05,"rain":1.4,"heavy_rain":1.7,"snow":1.3}
    base   = lag_24h * w_mult.get(weather, 1.0)
    hours  = list(range(24))
    demands_all = [base * hour_mults[h] for h in hours]

    surge_colors = ["#dc2626" if base * hour_mults[h] / 10 >= 2.0 else
                    "#f59e0b" if base * hour_mults[h] / 10 >= 1.5 else
                    "#16a34a" for h in hours]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(hours, demands_all, color=surge_colors, edgecolor="white", width=0.8)
    ax.axvline(hour, color="#1e40af", lw=2, linestyle="--", label=f"Current hour ({hour}:00)")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Predicted demand")
    ax.set_xticks(hours); ax.set_xticklabels([f"{h:02d}h" for h in hours], fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_extra = [Patch(color="#16a34a", label="No surge"),
                    Patch(color="#f59e0b", label="Surge 1.2–1.5×"),
                    Patch(color="#dc2626", label="Surge 1.8–2.5×")]
    ax.legend(handles=legend_extra + ax.get_legend_handles_labels()[0],
              loc="upper left", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Feature input summary ─────────────────────────────────────────────────
    with st.expander("📋 Raw Feature Inputs (sent to API)"):
        st.json(payload)
