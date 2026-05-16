"""
Feature store — dual online/offline architecture.

Online store  (Redis):  latest features per zone, TTL=10min, <5ms reads.
Offline store (Parquet): full history, used for training and backfill.

This pattern mirrors production feature stores (Feast, Tecton) but is
fully self-contained — no external dependencies beyond Redis and Parquet.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import redis
from loguru import logger

from config import settings


# ─── Online Feature Store (Redis) ─────────────────────────────────────────────

class OnlineFeatureStore:
    """
    Redis-backed online store.
    Key pattern: features:{zone}:{feature_group}
    Value: JSON-serialised dict of feature name → value.

    Used by the FastAPI serving layer to retrieve real-time features
    without re-computing them on every request.
    """
    def __init__(self, host: str = settings.REDIS_HOST,
                 port: int = settings.REDIS_PORT,
                 ttl: int = settings.ONLINE_FEATURES_TTL):
        self.ttl = ttl
        try:
            self.client = redis.Redis(host=host, port=port, db=0,
                                       decode_responses=True, socket_timeout=2)
            self.client.ping()
            self._available = True
            logger.info(f"Online store connected: {host}:{port}")
        except redis.exceptions.ConnectionError:
            logger.warning("Redis unavailable — online store disabled")
            self._available = False

    def _key(self, zone: str, group: str) -> str:
        return f"features:{zone}:{group}"

    def write(self, zone: str, group: str, features: dict) -> bool:
        """Write feature dict for a zone. Returns True on success."""
        if not self._available:
            return False
        try:
            payload = json.dumps({k: float(v) if isinstance(v, (np.floating, float)) else v
                                   for k, v in features.items()})
            self.client.setex(self._key(zone, group), self.ttl, payload)
            return True
        except Exception as e:
            logger.warning(f"Online store write failed: {e}")
            return False

    def read(self, zone: str, group: str) -> dict | None:
        """Retrieve features for a zone. Returns None if missing or stale."""
        if not self._available:
            return None
        try:
            raw = self.client.get(self._key(zone, group))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def read_many(self, zones: list[str], group: str) -> dict[str, dict]:
        """Batch read features for multiple zones."""
        if not self._available:
            return {}
        pipe = self.client.pipeline()
        for zone in zones:
            pipe.get(self._key(zone, group))
        results = {}
        for zone, raw in zip(zones, pipe.execute()):
            if raw:
                results[zone] = json.loads(raw)
        return results

    def update_zone_features(self, zone: str, df_latest: pd.DataFrame) -> None:
        """
        Compute and cache the latest feature snapshot for a zone.
        Called by the Spark streaming job after each micro-batch.
        """
        if df_latest.empty:
            return
        latest = df_latest.sort_values("timestamp").iloc[-1]
        numeric = {k: round(float(v), 4) for k, v in latest.items()
                   if isinstance(v, (int, float, np.number)) and k != "demand"}
        numeric["updated_at"] = time.time()
        self.write(zone, "demand_features", numeric)

    def health(self) -> dict:
        if not self._available:
            return {"status": "unavailable"}
        info = self.client.info("server")
        return {
            "status":      "ok",
            "redis_version": info.get("redis_version"),
            "used_memory":  info.get("used_memory_human"),
        }


# ─── Offline Feature Store (Parquet) ──────────────────────────────────────────

class OfflineFeatureStore:
    """
    Parquet-based offline store.
    Partitioned by year/month for efficient time-range reads.
    Used for model training, backtesting, and drift monitoring.
    """
    def __init__(self, base_path: Path = settings.FEATURE_STORE_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Offline store at: {self.base_path}")

    def _partition_path(self, year: int, month: int) -> Path:
        return self.base_path / f"year={year}" / f"month={month:02d}"

    def write(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
        """
        Write DataFrame to partitioned Parquet store.
        Overwrites existing partition for the same year/month.
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df["_year"]  = df[timestamp_col].dt.year
        df["_month"] = df[timestamp_col].dt.month

        for (year, month), grp in df.groupby(["_year", "_month"]):
            part_path = self._partition_path(year, month)
            part_path.mkdir(parents=True, exist_ok=True)
            out = part_path / "data.parquet"
            grp.drop(columns=["_year", "_month"]).to_parquet(out, index=False)

        logger.info(f"Offline store: wrote {len(df):,} rows across "
                    f"{df.groupby(['_year', '_month']).ngroups} partitions")

    def read(self, start: str | datetime, end: str | datetime,
             zones: list[str] | None = None) -> pd.DataFrame:
        """
        Read features for a date range. Efficient: only loads relevant partitions.
        """
        start = pd.Timestamp(start)
        end   = pd.Timestamp(end)
        dfs   = []

        # Enumerate required year/month partitions
        period = pd.period_range(start, end, freq="M")
        for p in period:
            part_path = self._partition_path(p.year, p.month)
            data_file = part_path / "data.parquet"
            if data_file.exists():
                df_part = pd.read_parquet(data_file)
                dfs.append(df_part)

        if not dfs:
            logger.warning(f"No data found between {start} and {end}")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        if zones:
            df = df[df["zone"].isin(zones)]

        logger.info(f"Offline store read: {len(df):,} rows "
                    f"[{start.date()} → {end.date()}]")
        return df.sort_values("timestamp")

    def get_latest_snapshot(self, n_hours: int = 24) -> pd.DataFrame:
        """Read the most recent n_hours of data from the store."""
        # Find most recent partition
        partitions = sorted(self.base_path.rglob("data.parquet"))
        if not partitions:
            return pd.DataFrame()
        latest_df = pd.read_parquet(partitions[-1])
        latest_df["timestamp"] = pd.to_datetime(latest_df["timestamp"])
        cutoff = latest_df["timestamp"].max() - pd.Timedelta(hours=n_hours)
        return latest_df[latest_df["timestamp"] >= cutoff]

    def list_partitions(self) -> pd.DataFrame:
        """List all available partitions with row counts."""
        rows = []
        for f in sorted(self.base_path.rglob("data.parquet")):
            df = pd.read_parquet(f)
            rows.append({"partition": str(f.parent.relative_to(self.base_path)),
                         "n_rows": len(df),
                         "size_kb": round(f.stat().st_size / 1024, 1)})
        return pd.DataFrame(rows)


# ─── Unified FeatureStore facade ──────────────────────────────────────────────

class FeatureStore:
    """
    Unified interface combining online and offline stores.
    Serving path: reads from Redis (fast).
    Training path: reads from Parquet (complete history).
    """
    def __init__(self):
        self.online  = OnlineFeatureStore()
        self.offline = OfflineFeatureStore()

    def get_serving_features(self, zone: str) -> dict | None:
        """Get latest features for real-time serving from Redis."""
        return self.online.read(zone, "demand_features")

    def get_training_data(self, start: str, end: str,
                          zones: list[str] | None = None) -> pd.DataFrame:
        """Get historical features for model training from Parquet."""
        return self.offline.read(start, end, zones)

    def push_realtime_features(self, zone: str, features: dict) -> None:
        """Called by Spark streaming to update online store."""
        self.online.write(zone, "demand_features", features)

    def materialize(self, df: pd.DataFrame) -> None:
        """Write a batch of features to both online and offline store."""
        self.offline.write(df)
        # Update online store with latest per zone
        for zone, grp in df.groupby("zone"):
            self.online.update_zone_features(str(zone), grp)
        logger.success(f"Materialized {len(df):,} rows to both stores")
