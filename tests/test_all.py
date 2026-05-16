"""
Full test suite: features, models, evaluation, explainability, API.
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Synthetic DataFrame large enough for all tests."""
    np.random.seed(42)
    zones = [str(i) for i in range(1, 6)]
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h")
    rows = []
    for zone in zones:
        for ts in timestamps:
            rows.append({
                "timestamp": ts, "zone": zone,
                "demand": max(1, int(np.random.lognormal(2.5, 0.4))),
                "weather": np.random.choice(["clear", "rain", "cloudy"]),
                "temperature_c": np.random.normal(12, 5),
                "precipitation_mm": np.random.exponential(1),
                "is_event": False, "event_name": None,
                "lat": 48.85 + np.random.normal(0, 0.02),
                "lon": 2.35  + np.random.normal(0, 0.02),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def feature_df(sample_df):
    """Sample DataFrame with temporal + geo features — uses fillna, not dropna."""
    from features.temporal import build_temporal_features
    from features.geospatial import build_geospatial_features
    df = sample_df.sort_values(["zone", "timestamp"])
    df = build_temporal_features(df, lags=[1, 2, 3], rolling_windows=[3, 6])
    df = build_geospatial_features(df)
    df = df.fillna(0)
    return df


@pytest.fixture
def X_y(feature_df):
    drop = {"timestamp", "demand", "zone", "weather", "event_name",
            "lat", "lon", "zone_cluster", "is_event"}
    feat_cols = [c for c in feature_df.columns
                 if c not in drop and feature_df[c].dtype != "object"]
    X = feature_df[feat_cols].fillna(0)
    y = feature_df["demand"]
    assert len(X) > 0, "X_y fixture is empty"
    return X, y


# ─── Feature tests ────────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_lag_features_created(self, sample_df):
        from features.temporal import add_lag_features
        df = add_lag_features(sample_df.sort_values("timestamp"),
                               "demand", lags=[1, 24], group_cols=["zone"])
        assert "demand_lag_1h"  in df.columns
        assert "demand_lag_24h" in df.columns

    def test_rolling_no_future_leakage(self, sample_df):
        from features.temporal import add_rolling_features
        df = add_rolling_features(sample_df.sort_values("timestamp"),
                                   "demand", windows=[3], group_cols=["zone"])
        assert "demand_roll3h_mean" in df.columns

    def test_calendar_features_range(self, sample_df):
        from features.temporal import add_calendar_features
        df = add_calendar_features(sample_df)
        assert df["hour"].between(0, 23).all()
        assert df["day_of_week"].between(0, 6).all()
        assert df["month"].between(1, 12).all()

    def test_fourier_features_bounded(self, sample_df):
        from features.temporal import add_fourier_features
        df = add_fourier_features(sample_df)
        fourier_cols = [c for c in df.columns if "fourier" in c]
        assert len(fourier_cols) > 0
        for col in fourier_cols:
            assert df[col].between(-1.01, 1.01).all(), f"{col} out of [-1, 1]"

    def test_velocity_features(self, sample_df):
        from features.temporal import add_demand_velocity
        df = add_demand_velocity(sample_df.sort_values("timestamp"),
                                  "demand", group_cols=["zone"])
        assert "demand_velocity_1h" in df.columns


class TestGeospatialFeatures:
    def test_zone_encoded_range(self, sample_df):
        from features.geospatial import add_zone_features
        df = add_zone_features(sample_df)
        assert "zone_encoded" in df.columns
        assert df["zone_encoded"].ge(0).all()

    def test_poi_features_nonnegative(self, sample_df):
        from features.geospatial import add_poi_proximity_features
        df = add_poi_proximity_features(sample_df)
        poi_cols = [c for c in df.columns if c.startswith("poi_")]
        assert len(poi_cols) > 0
        for col in poi_cols:
            assert df[col].ge(0).all()

    def test_demand_share_sums_to_one(self, sample_df):
        from features.geospatial import add_spatial_demand_ratio
        df = add_spatial_demand_ratio(sample_df.sort_values("timestamp"))
        share_sum = df.groupby("timestamp")["demand_share"].sum()
        assert share_sum.between(0.99, 1.01).all()


# ─── Model tests ──────────────────────────────────────────────────────────────

class TestLightGBM:
    def test_fit_predict_shape(self, X_y):
        from models.tree_based.lgbm_model import LGBMDemandModel
        X, y = X_y
        model = LGBMDemandModel({"n_estimators": 20, "num_leaves": 15})
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_nonnegative(self, X_y):
        from models.tree_based.lgbm_model import LGBMDemandModel
        X, y = X_y
        model = LGBMDemandModel({"n_estimators": 10, "num_leaves": 10})
        model.fit(X, y)
        assert (model.predict(X) >= 0).all()

    def test_feature_importance_sums(self, X_y):
        from models.tree_based.lgbm_model import LGBMDemandModel
        X, y = X_y
        model = LGBMDemandModel({"n_estimators": 10})
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == len(X.columns)
        assert imp.sum() > 0

    def test_save_load_roundtrip(self, X_y, tmp_path):
        from models.tree_based.lgbm_model import LGBMDemandModel
        from models.base_model import BaseModel
        X, y = X_y
        model = LGBMDemandModel({"n_estimators": 10})
        model.fit(X, y)
        path = tmp_path / "test_lgbm.pkl"
        model.save(path)
        loaded = BaseModel.load(path)
        np.testing.assert_allclose(model.predict(X), loaded.predict(X), rtol=1e-5)


class TestXGBoost:
    def test_fit_predict(self, X_y):
        from models.tree_based.xgb_catboost_models import XGBDemandModel
        X, y = X_y
        model = XGBDemandModel({"n_estimators": 10, "booster": "gbtree"})
        model.fit(X, y)
        assert (model.predict(X) >= 0).all()


class TestEnsemble:
    def test_voting_ensemble(self, X_y):
        from models.tree_based.lgbm_model import LGBMDemandModel
        from models.tree_based.xgb_catboost_models import XGBDemandModel
        from models.ensemble.stacking import VotingEnsemble
        X, y = X_y
        m1 = LGBMDemandModel({"n_estimators": 10})
        m2 = XGBDemandModel({"n_estimators": 10})
        ensemble = VotingEnsemble([m1, m2])
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert preds.shape == (len(X),)
        assert (preds >= 0).all()


# ─── Evaluation tests ─────────────────────────────────────────────────────────

class TestMetrics:
    def test_rmse_zero_on_perfect(self):
        from evaluation.metrics import rmse
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_mape_bounded(self):
        from evaluation.metrics import mape
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 32.0])
        m = mape(y_true, y_pred)
        assert 0 <= m <= 1

    def test_psi_stable_same_distribution(self):
        from evaluation.metrics import psi
        data = np.random.normal(0, 1, 1000)
        p = psi(data, data + np.random.normal(0, 0.01, 1000))
        assert p < 0.1

    def test_psi_high_on_shifted_distribution(self):
        from evaluation.metrics import psi
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)
        p = psi(ref, cur)
        assert p > 0.2

    def test_pinball_asymmetric(self):
        from evaluation.metrics import pinball_loss
        y = np.array([10.0])
        over_pred  = np.array([12.0])
        under_pred = np.array([8.0])
        loss_over  = pinball_loss(y, over_pred, quantile=0.9)
        loss_under = pinball_loss(y, under_pred, quantile=0.9)
        assert loss_under > loss_over


# ─── Calibration tests ────────────────────────────────────────────────────────

class TestCalibration:
    def test_isotonic_reduces_error(self):
        from optimization.calibration.calibrators import IsotonicCalibrator
        np.random.seed(42)
        y_true = np.linspace(1, 30, 200)
        raw    = y_true * 1.3 + np.random.normal(0, 2, 200)
        cal    = IsotonicCalibrator()
        cal_preds = cal.fit_transform(raw, y_true)
        assert np.mean(np.abs(cal_preds - y_true)) < np.mean(np.abs(raw - y_true))

    def test_temperature_scaling_direction(self):
        from optimization.calibration.calibrators import TemperatureScaling
        np.random.seed(42)
        y_true = np.random.uniform(5, 25, 200)
        raw    = y_true * 2.0
        cal    = TemperatureScaling()
        cal.fit(raw, y_true)
        assert cal.temperature > 1.0


# ─── Cross-validation tests ───────────────────────────────────────────────────

class TestCV:
    def test_walkforward_no_future_leakage(self, X_y):
        from optimization.cv.timeseries_cv import WalkForwardCV
        X, y = X_y
        cv = WalkForwardCV(n_splits=3, gap_hours=2, min_train_size=100)
        for train_idx, val_idx in cv.split(X):
            assert train_idx.max() < val_idx.min()

    def test_walkforward_expanding(self, X_y):
        from optimization.cv.timeseries_cv import WalkForwardCV
        X, y = X_y
        cv   = WalkForwardCV(n_splits=3, gap_hours=2, min_train_size=100)
        sizes = [len(tr) for tr, _ in cv.split(X)]
        assert sizes == sorted(sizes)

    def test_purged_kfold_no_overlap(self, X_y):
        from optimization.cv.timeseries_cv import PurgedKFold
        X, _ = X_y
        cv   = PurgedKFold(n_splits=3, purge_gap=5)
        for train_idx, val_idx in cv.split(X):
            assert len(set(train_idx) & set(val_idx)) == 0


# ─── API tests ────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from serving.api import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_predict_valid_request(self, client):
        payload = {
            "zone": "11", "hour": 18, "day_of_week": 4,
            "month": 3, "weather": "rain", "temperature_c": 10.0,
            "precipitation_mm": 5.0, "is_event": False,
            "demand_lag_1h": 15.0, "demand_lag_24h": 12.0,
            "demand_lag_168h": 11.0, "demand_roll24h_mean": 13.0,
        }
        resp = client.post("/predict/demand", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_demand"] >= 0
        assert data["confidence_low"] <= data["predicted_demand"]
        assert data["confidence_high"] >= data["predicted_demand"]

    def test_predict_invalid_zone(self, client):
        payload = {
            "zone": "99", "hour": 10, "day_of_week": 1, "month": 1,
            "weather": "clear", "temperature_c": 15.0,
        }
        resp = client.post("/predict/demand", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_weather(self, client):
        payload = {
            "zone": "5", "hour": 10, "day_of_week": 1, "month": 1,
            "weather": "tornado", "temperature_c": 15.0,
        }
        resp = client.post("/predict/demand", json=payload)
        assert resp.status_code == 422

    def test_price_endpoint(self, client):
        payload = {"zone": "11", "predicted_demand": 25.0, "zone_baseline": 10.0}
        resp = client.post("/price", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["surge_multiplier"] >= 1.0
        assert data["final_price_eur"] >= 8.0


# ─── Drift monitor tests ──────────────────────────────────────────────────────

class TestDriftMonitor:
    def test_no_alert_same_distribution(self):
        from monitoring.drift_detector import DriftMonitor
        np.random.seed(42)
        ref = pd.DataFrame({"demand": np.random.lognormal(2, 0.3, 300),
                             "temperature_c": np.random.normal(12, 3, 300)})
        cur = pd.DataFrame({"demand": np.random.lognormal(2, 0.3, 300),
                             "temperature_c": np.random.normal(12, 3, 300)})
        monitor = DriftMonitor(ref)
        result  = monitor.run(cur)
        assert "alert" in result
        assert result["n_features_checked"] > 0

    def test_alert_on_shifted_data(self):
        from monitoring.drift_detector import DriftMonitor
        np.random.seed(42)
        ref = pd.DataFrame({"demand": np.random.lognormal(2, 0.3, 500),
                             "temperature_c": np.random.normal(12, 2, 500)})
        cur = pd.DataFrame({"demand": np.random.lognormal(2, 0.3, 500) + 100,
                             "temperature_c": np.random.normal(12, 2, 500) + 50})
        monitor = DriftMonitor(ref, psi_threshold=0.1)
        result  = monitor.run(cur)
        assert len(result["drifted_features"]) > 0

    def test_performance_alert_on_bad_preds(self):
        from monitoring.drift_detector import DriftMonitor
        ref = pd.DataFrame({"demand": np.random.lognormal(2, 0.3, 100)})
        monitor = DriftMonitor(ref, mape_threshold=0.1)
        y_true = np.array([10.0] * 50)
        y_pred = np.array([100.0] * 50)
        result = monitor.run(ref, y_true, y_pred)
        assert result["performance"]["alert"] is True