# conftest.py — shared pytest configuration and fixtures
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import numpy as np
import pandas as pd


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "integration: mark as integration test (requires services)")
    config.addinivalue_line("markers", "gpu: mark as requiring GPU")


@pytest.fixture(scope="session")
def tiny_df():
    """Very small dataset for ultra-fast unit tests."""
    np.random.seed(0)
    n, zones = 200, ["1", "2", "3"]
    timestamps = pd.date_range("2024-01-01", periods=n // len(zones), freq="h")
    rows = []
    for zone in zones:
        for ts in timestamps:
            rows.append({
                "timestamp": ts, "zone": zone,
                "demand": max(0, int(np.random.lognormal(2.3, 0.4))),
                "weather": "clear", "temperature_c": 12.0,
                "precipitation_mm": 0.0, "is_event": False, "event_name": None,
                "lat": 48.85, "lon": 2.35,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def medium_df():
    """Medium dataset (90 days) — used for CV and ensemble tests."""
    np.random.seed(42)
    n, zones = 2000, [str(i) for i in range(1, 6)]
    timestamps = pd.date_range("2024-01-01", periods=n // len(zones), freq="h")
    rows = []
    for zone in zones:
        for ts in timestamps:
            rows.append({
                "timestamp": ts, "zone": zone,
                "demand": max(0, int(np.random.lognormal(2.5, 0.4))),
                "weather": np.random.choice(["clear", "rain", "cloudy"]),
                "temperature_c": np.random.normal(12, 5),
                "precipitation_mm": np.random.exponential(1),
                "is_event": False, "event_name": None,
                "lat": 48.85, "lon": 2.35,
            })
    return pd.DataFrame(rows)
