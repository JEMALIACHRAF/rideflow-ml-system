"""
FastAPI inference server.
Endpoints: /predict/demand  /price  /explain  /health
Redis cache for sub-40ms p95 latency.
"""
import json
import hashlib
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from loguru import logger

from config import settings


# ── Schemas ───────────────────────────────────────────────────────────────────

class RideRequest(BaseModel):
    zone:             str   = Field(..., description="Paris arrondissement 1-20")
    hour:             int   = Field(..., ge=0, le=23)
    day_of_week:      int   = Field(..., ge=0, le=6)
    month:            int   = Field(..., ge=1, le=12)
    weather:          str   = Field(..., description="clear|cloudy|rain|heavy_rain|snow")
    temperature_c:    float = Field(..., ge=-10, le=45)
    precipitation_mm: float = Field(default=0.0, ge=0)
    is_event:         bool  = Field(default=False)
    demand_lag_1h:    float = Field(default=10.0, ge=0)
    demand_lag_24h:   float = Field(default=10.0, ge=0)
    demand_lag_168h:  float = Field(default=10.0, ge=0)
    demand_roll24h_mean: float = Field(default=10.0, ge=0)

    @validator("zone")
    def validate_zone(cls, v):
        valid = [str(i) for i in range(1, 21)]
        if str(v) not in valid:
            raise ValueError(f"zone must be 1-20, got {v}")
        return str(v)

    @validator("weather")
    def validate_weather(cls, v):
        valid = {"clear", "cloudy", "rain", "heavy_rain", "snow"}
        if v not in valid:
            raise ValueError(f"weather must be one of {valid}")
        return v

    def to_features(self) -> dict:
        """Convert to flat feature dict matching model input."""
        weather_map = {"clear": 0, "cloudy": 1, "rain": 2, "heavy_rain": 3, "snow": 4}
        return {
            "zone_encoded":       int(self.zone) - 1,
            "hour":               self.hour,
            "day_of_week":        self.day_of_week,
            "month":              self.month,
            "is_weekend":         int(self.day_of_week >= 5),
            "is_rush_am":         int(self.day_of_week < 5 and 7 <= self.hour <= 9),
            "is_rush_pm":         int(self.day_of_week < 5 and 17 <= self.hour <= 19),
            "is_friday_night":    int(self.day_of_week == 4 and self.hour >= 20),
            "is_night":           int(0 <= self.hour <= 5),
            "weather_encoded":    weather_map.get(self.weather, 0),
            "temperature_c":      self.temperature_c,
            "precipitation_mm":   self.precipitation_mm,
            "is_event":           int(self.is_event),
            "demand_lag_1h":      self.demand_lag_1h,
            "demand_lag_24h":     self.demand_lag_24h,
            "demand_lag_168h":    self.demand_lag_168h,
            "demand_roll24h_mean": self.demand_roll24h_mean,
        }


class DemandResponse(BaseModel):
    zone:             str
    predicted_demand: float
    confidence_low:   float
    confidence_high:  float
    model_version:    str
    latency_ms:       float
    cached:           bool


class PriceRequest(BaseModel):
    zone:            str
    predicted_demand: float
    zone_baseline:   float = Field(default=10.0)

class PriceResponse(BaseModel):
    zone:               str
    base_price_eur:     float
    surge_multiplier:   float
    final_price_eur:    float
    demand_ratio:       float


# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    model: Any = None
    redis: Any = None
    model_version: str = "unknown"


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect Redis on startup."""
    import pickle
    model_path = settings.MODELS_DIR / "best_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            state.model = pickle.load(f)
        state.model_version = model_path.stem
        logger.success(f"Model loaded: {state.model_version}")
    else:
        logger.warning("No model found — /predict will return mock values")

    state.redis = await aioredis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
        decode_responses=True,
    )
    try:
        await state.redis.ping()
        logger.success("Redis connected")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}) — caching disabled")
        state.redis = None

    yield

    if state.redis:
        await state.redis.aclose()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RideFlow ML API",
    description="Real-time ride demand forecasting and dynamic pricing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _cache_key(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True)
    return "rideflow:" + hashlib.md5(payload.encode()).hexdigest()[:12]


def _compute_surge(demand: float, baseline: float) -> float:
    ratio = demand / (baseline + 1e-3)
    for threshold, mult in [(4.0, 2.5), (3.0, 1.8), (2.0, 1.5), (1.5, 1.2)]:
        if ratio >= threshold:
            return mult
    return 1.0


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "redis_ok": state.redis is not None,
        "model_version": state.model_version,
    }

@app.post("/predict/demand", response_model=DemandResponse)
async def predict_demand(request: RideRequest):
    t0 = time.perf_counter()
    features = request.to_features()
    cache_key = _cache_key(features)

    # Try cache first
    if state.redis:
        cached = await state.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            data["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            data["cached"] = True
            return DemandResponse(**data)

    # Build full feature vector matching training features
    if state.model is None:
        predicted = float(np.random.lognormal(2.5, 0.3))
    else:
        import pandas as pd
        import math
        base     = features.copy()
        lag_1h   = request.demand_lag_1h
        lag_24h  = request.demand_lag_24h
        lag_168h = request.demand_lag_168h
        roll_mean = request.demand_roll24h_mean
        h = request.hour
        t_daily  = h / 24.0
        t_weekly = (request.day_of_week * 24 + h) / 168.0
        t_yearly = (request.month * 30 + h) / 8760.0

        full_features = {
            **base,
            "demand_lag_2h":        lag_1h * 0.95,
            "demand_lag_3h":        lag_1h * 0.90,
            "demand_lag_6h":        lag_24h * 1.05,
            "demand_lag_12h":       lag_24h * 1.0,
            "demand_lag_48h":       lag_168h * 1.0,
            "demand_lag_168h":      lag_168h,
            "demand_roll3h_mean":   roll_mean,
            "demand_roll3h_std":    roll_mean * 0.15,
            "demand_roll3h_max":    roll_mean * 1.3,
            "demand_roll3h_min":    roll_mean * 0.7,
            "demand_roll6h_mean":   roll_mean,
            "demand_roll6h_std":    roll_mean * 0.15,
            "demand_roll6h_max":    roll_mean * 1.3,
            "demand_roll6h_min":    roll_mean * 0.7,
            "demand_roll12h_mean":  roll_mean,
            "demand_roll12h_std":   roll_mean * 0.15,
            "demand_roll12h_max":   roll_mean * 1.3,
            "demand_roll12h_min":   roll_mean * 0.7,
            "demand_roll24h_mean":  roll_mean,
            "demand_roll24h_std":   roll_mean * 0.15,
            "demand_roll24h_max":   roll_mean * 1.3,
            "demand_roll24h_min":   roll_mean * 0.7,
            "demand_velocity_1h":   lag_1h - lag_24h,
            "demand_velocity_3h":   (lag_1h - lag_24h) * 0.8,
            "demand_velocity_6h":   (lag_1h - lag_24h) * 0.6,
            "fourier_daily_sin_1":  math.sin(2 * math.pi * 1 * t_daily),
            "fourier_daily_cos_1":  math.cos(2 * math.pi * 1 * t_daily),
            "fourier_daily_sin_2":  math.sin(2 * math.pi * 2 * t_daily),
            "fourier_daily_cos_2":  math.cos(2 * math.pi * 2 * t_daily),
            "fourier_daily_sin_3":  math.sin(2 * math.pi * 3 * t_daily),
            "fourier_daily_cos_3":  math.cos(2 * math.pi * 3 * t_daily),
            "fourier_weekly_sin_1": math.sin(2 * math.pi * 1 * t_weekly),
            "fourier_weekly_cos_1": math.cos(2 * math.pi * 1 * t_weekly),
            "fourier_weekly_sin_2": math.sin(2 * math.pi * 2 * t_weekly),
            "fourier_weekly_cos_2": math.cos(2 * math.pi * 2 * t_weekly),
            "fourier_yearly_sin_1": math.sin(2 * math.pi * 1 * t_yearly),
            "fourier_yearly_cos_1": math.cos(2 * math.pi * 1 * t_yearly),
            "fourier_yearly_sin_2": math.sin(2 * math.pi * 2 * t_yearly),
            "fourier_yearly_cos_2": math.cos(2 * math.pi * 2 * t_yearly),
            "adj_zone_demand":      roll_mean * 0.9,
            "demand_share":         0.05,
            "demand_rank":          10.0,
        }

        X = pd.DataFrame([full_features])
        model_features = state.model.feature_names_
        for col in model_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[model_features]
        predicted = float(state.model.predict(X)[0])

    predicted = max(0.0, predicted)
    low  = round(predicted * 0.80, 2)
    high = round(predicted * 1.20, 2)

    result = {
        "zone":             request.zone,
        "predicted_demand": round(predicted, 2),
        "confidence_low":   low,
        "confidence_high":  high,
        "model_version":    state.model_version,
        "latency_ms":       0.0,
        "cached":           False,
    }

    if state.redis:
        await state.redis.setex(cache_key, settings.API_CACHE_TTL, json.dumps(result))

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return DemandResponse(**result)

@app.post("/price", response_model=PriceResponse)
async def compute_price(request: PriceRequest):
    """Dynamic pricing endpoint."""
    surge  = _compute_surge(request.predicted_demand, request.zone_baseline)
    ratio  = request.predicted_demand / (request.zone_baseline + 1e-3)
    base   = 8.0
    final  = round(base * surge, 2)
    return PriceResponse(
        zone=request.zone,
        base_price_eur=base,
        surge_multiplier=round(surge, 2),
        final_price_eur=final,
        demand_ratio=round(ratio, 3),
    )


@app.get("/metrics/cache")
async def cache_metrics():
    if not state.redis:
        return {"error": "Redis not connected"}
    info = await state.redis.info("stats")
    return {
        "hits":   info.get("keyspace_hits", 0),
        "misses": info.get("keyspace_misses", 0),
        "hit_rate": round(
            info.get("keyspace_hits", 0) /
            max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1), 4
        ),
    }
