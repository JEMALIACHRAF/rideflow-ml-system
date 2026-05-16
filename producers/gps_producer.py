"""
Synthetic data producer — generates realistic ride-hailing data for Paris.
Simulates GPS trips, weather conditions, and city events.

Usage:
    python producers/gps_producer.py --n-days 90 --output data/raw/
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

app = typer.Typer()

# Paris arrondissements approximate centers (lat, lon)
ZONE_CENTERS = {
    "1": (48.8606, 2.3477),  "2": (48.8672, 2.3469),  "3": (48.8632, 2.3590),
    "4": (48.8534, 2.3523),  "5": (48.8462, 2.3510),  "6": (48.8496, 2.3339),
    "7": (48.8566, 2.3122),  "8": (48.8752, 2.3078),  "9": (48.8771, 2.3378),
    "10": (48.8761, 2.3600), "11": (48.8589, 2.3794), "12": (48.8407, 2.3877),
    "13": (48.8315, 2.3561), "14": (48.8285, 2.3266), "15": (48.8418, 2.2956),
    "16": (48.8636, 2.2701), "17": (48.8848, 2.3128), "18": (48.8917, 2.3455),
    "19": (48.8803, 2.3799), "20": (48.8637, 2.3990),
}

# Peak demand patterns
HOUR_MULTIPLIERS = {
    0: 0.3, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.1, 5: 0.2,
    6: 0.5, 7: 1.2, 8: 1.8, 9: 1.4, 10: 1.0, 11: 1.1,
    12: 1.3, 13: 1.2, 14: 1.0, 15: 1.0, 16: 1.1, 17: 1.6,
    18: 1.9, 19: 1.7, 20: 1.4, 21: 1.3, 22: 1.1, 23: 0.7,
}
DAY_MULTIPLIERS = {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.3, 5: 1.8, 6: 1.5}
ZONE_BASE_DEMAND = {z: random.uniform(5, 25) for z in ZONE_CENTERS}

WEATHER_CONDITIONS = ["clear", "cloudy", "rain", "heavy_rain", "snow"]
WEATHER_DEMAND_IMPACT = {"clear": 1.0, "cloudy": 1.05, "rain": 1.4, "heavy_rain": 1.6, "snow": 1.3}

PARIS_EVENTS = [
    {"name": "PSG_match", "zone": "16", "demand_boost": 3.0, "duration_h": 4},
    {"name": "concert_bercy", "zone": "12", "demand_boost": 2.5, "duration_h": 3},
    {"name": "roland_garros", "zone": "16", "demand_boost": 2.0, "duration_h": 8},
    {"name": "fashion_week", "zone": "8", "demand_boost": 1.5, "duration_h": 12},
    {"name": "bastille_day", "zone": "7", "demand_boost": 4.0, "duration_h": 6},
]


def generate_demand(zone: str, dt: datetime, weather: str, active_events: list) -> int:
    """Compute realistic demand with noise."""
    base = ZONE_BASE_DEMAND[zone]
    hour_mult = HOUR_MULTIPLIERS[dt.hour]
    day_mult = DAY_MULTIPLIERS[dt.weekday()]
    weather_mult = WEATHER_DEMAND_IMPACT[weather]

    event_mult = 1.0
    for evt in active_events:
        if evt["zone"] == zone:
            event_mult = max(event_mult, evt["demand_boost"])

    demand = base * hour_mult * day_mult * weather_mult * event_mult
    noise = np.random.lognormal(0, 0.15)
    return max(0, int(demand * noise))


def generate_weather_series(n_hours: int) -> list:
    """Markov chain weather transitions."""
    transition = {
        "clear":      {"clear": 0.7, "cloudy": 0.25, "rain": 0.04, "heavy_rain": 0.01, "snow": 0.0},
        "cloudy":     {"clear": 0.3, "cloudy": 0.45, "rain": 0.2,  "heavy_rain": 0.04, "snow": 0.01},
        "rain":       {"clear": 0.1, "cloudy": 0.3,  "rain": 0.45, "heavy_rain": 0.13, "snow": 0.02},
        "heavy_rain": {"clear": 0.05,"cloudy": 0.2,  "rain": 0.4,  "heavy_rain": 0.3,  "snow": 0.05},
        "snow":       {"clear": 0.1, "cloudy": 0.3,  "rain": 0.2,  "heavy_rain": 0.1,  "snow": 0.3},
    }
    weather = ["clear"]
    for _ in range(n_hours - 1):
        probs = transition[weather[-1]]
        weather.append(random.choices(list(probs.keys()), weights=list(probs.values()))[0])
    return weather


@app.command()
def generate(
    n_days: int = typer.Option(90, help="Number of days to generate"),
    output: Path = typer.Option("data/raw", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Generate synthetic ride-hailing data for Paris."""
    np.random.seed(seed)
    random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=h) for h in range(n_days * 24)]
    weather_series = generate_weather_series(len(timestamps))

    # Generate active events (random days)
    event_schedule = []
    for evt in PARIS_EVENTS:
        for _ in range(n_days // 14):  # ~2 per event per month
            day = random.randint(0, n_days - 1)
            hour = random.randint(14, 20)
            event_start = start + timedelta(days=day, hours=hour)
            event_schedule.append({**evt, "start": event_start,
                                    "end": event_start + timedelta(hours=evt["duration_h"])})

    logger.info(f"Generating {n_days} days of data for {len(ZONE_CENTERS)} zones...")
    records = []
    for i, (dt, weather) in enumerate(zip(timestamps, weather_series)):
        active_events = [e for e in event_schedule if e["start"] <= dt <= e["end"]]
        for zone in ZONE_CENTERS:
            demand = generate_demand(zone, dt, weather, active_events)
            lat, lon = ZONE_CENTERS[zone]
            records.append({
                "timestamp": dt.isoformat(),
                "zone": zone,
                "demand": demand,
                "weather": weather,
                "temperature_c": 12 + 8 * np.sin(2 * np.pi * dt.timetuple().tm_yday / 365)
                                  + np.random.normal(0, 2),
                "precipitation_mm": 0 if weather == "clear" else np.random.exponential(2),
                "wind_speed_kmh": np.random.gamma(2, 5),
                "is_event": len(active_events) > 0,
                "event_name": active_events[0]["name"] if active_events else None,
                "lat": lat + np.random.normal(0, 0.005),
                "lon": lon + np.random.normal(0, 0.005),
            })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out_path = output / "rides_synthetic.parquet"
    df.to_parquet(out_path, index=False)
    logger.success(f"Saved {len(df):,} rows → {out_path}")
    logger.info(f"Demand stats: mean={df['demand'].mean():.1f}, "
                f"max={df['demand'].max()}, std={df['demand'].std():.1f}")
    return df


if __name__ == "__main__":
    app()
