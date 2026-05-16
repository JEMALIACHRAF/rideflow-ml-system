<div align="center">


![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM MAPE](https://img.shields.io/badge/MAPE-7.88%25-green)
![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

# RideFlow ML System
### Real-time ride demand forecasting & dynamic pricing engine for Paris

![CI](https://github.com/JEMALIACHRAF/rideflow-ml-system/actions/workflows/ci.yml/badge.svg)

</div>

---

## Overview

RideFlow is a production-grade machine learning system that forecasts ride-hailing demand per Paris arrondissement and computes dynamic surge pricing in real time. It mirrors the architecture used by companies like Uber, BlaBlaCar, and Bolt: a streaming ingestion layer feeds a dual online/offline feature store, a battery of ML models generates predictions, and a low-latency REST API serves pricing decisions to downstream systems.

The project covers the full ML lifecycle — from raw data generation and feature engineering to model explainability, drift monitoring, and Kubernetes deployment — with every component tested, containerised, and logged to MLflow.

---

## Architecture

```
GPS Events · Weather · City Events
         │
         ▼
    Kafka (3 topics, 3 partitions)
         │
         ▼
  Spark Structured Streaming
  ├── 5-min tumbling windows
  └── 1-hour sliding windows
         │
         ▼
  Feature Store
  ├── Online  → Redis  (TTL 5min, <5ms reads)
  └── Offline → Parquet (partitioned year/month)
         │
         ▼
  ML Models (trained offline, served online)
  ├── LightGBM   MAPE 7.88%  R²=0.981  ← champion
  ├── CatBoost   MAPE 9.57%  R²=0.970
  ├── XGBoost    MAPE 17.8%  R²=0.730
  ├── Stacking   (OOF meta Ridge)
  └── Blending   (scipy-optimised weights)
         │
         ▼
  FastAPI  (async, Redis cache, Pydantic v2)
  ├── POST /predict/demand   → 44ms cold / 5ms cached
  ├── POST /price            → surge multiplier + final price
  └── GET  /health           → model + Redis status
         │
         ▼
  Monitoring
  ├── PSI drift detection per feature
  ├── Evidently reports (data + concept drift)
  ├── Grafana dashboards
  └── Airflow DAG (weekly retrain + champion/challenger)
```

---

## Key Results

| Metric | Value |
|---|---|
| Best model | LightGBM |
| MAPE (test set) | **7.88%** |
| R² (test set) | **0.981** |
| API latency — cold start | 44ms |
| API latency — Redis cache hit | **5ms** |
| Cache speedup | **8.4×** |
| Unit tests | **32 / 32 passed** |
| Features engineered | 74 |
| Training data | 43,200 rows (90 days × 20 zones) |
| Revenue efficiency (backtest) | 100% vs perfect pricing |

---

## Technical Choices & Rationale

### Why LightGBM as champion?

LightGBM outperformed XGBoost and CatBoost on this dataset because the demand signal is dominated by strong temporal autocorrelation (lag features), which gradient boosting handles very efficiently. The `dart` booster in XGBoost caused instability without an eval set, while CatBoost's native categorical handling had no advantage here since all categoricals were pre-encoded. LightGBM's histogram-based algorithm also trained 5× faster, enabling more Optuna trials within the same time budget.

### Why a dual online/offline feature store?

The online store (Redis, TTL=5min) serves the API in <5ms by pre-materialising the latest zone-level features after each Spark micro-batch. Without it, the API would need to re-query and aggregate raw events on every request, adding 200-500ms. The offline store (Parquet, partitioned by year/month) enables efficient time-range reads for training without loading the full history into memory.

### Why walk-forward CV with a gap?

Standard K-Fold would leak future demand information into training folds via lag and rolling features (a 24h lag computed on unsorted data reads the future). Walk-forward CV with a 24h gap between train and validation boundaries prevents this entirely. Each fold expands the training window (expanding window), which also mirrors how the model is retrained in production.

### Why stacking with OOF predictions?

Stacking with out-of-fold (OOF) predictions avoids the main risk of naive ensembles: the meta-learner seeing the same data the base models were trained on. Each base model only contributes OOF predictions to the meta-learner's training set, ensuring the meta-learner learns to correct genuine generalisation errors rather than memorise training residuals.

### Why SHAP + LIME + Anchors + DiCE?

Each tool answers a different question:
- **SHAP** — which features drive the model globally, and how much does each feature contribute to this specific prediction?
- **LIME** — does a local linear approximation agree with SHAP? If yes, the explanation is robust.
- **Anchors** — can we express the prediction as a simple IF-THEN rule? Useful for driver incentives ("go to zone 11 between 17h-19h on rainy Fridays").
- **DiCE counterfactuals** — what is the minimum change that would produce a different pricing outcome? Useful for regulatory compliance ("what would need to change for the price to be under €10?").

### Why Redis for API caching?

Ride-hailing apps batch-predict demand for all zones every few minutes and then serve individual requests from cache. Most prediction requests repeat the same (zone, hour, weather) combination within a short window. Redis with a 60-second TTL gives an 8.4× latency reduction on repeated requests while staying current enough for pricing decisions.

### Why Airflow for orchestration?

The weekly retrain DAG implements a champion/challenger pattern: the new model is promoted to production only if it improves MAPE by more than 1% on the held-out test set. This prevents regressions from being silently deployed. Airflow also handles the drift check gate — if PSI across features is below 0.1, retraining is skipped to save compute.

---

## Project Structure

```
rideflow-ml-system/
│
├── producers/                  # Synthetic data generation (GPS, weather, events)
├── streaming/                  # Spark Structured Streaming pipeline
├── feature_store/              # Dual online (Redis) + offline (Parquet) store
│
├── features/
│   ├── temporal.py             # Lags, rolling stats, Fourier seasonality, velocity
│   ├── geospatial.py           # H3 zones, POI proximity, adjacency demand, clusters
│   └── selection.py            # SHAP + Mutual Info + Boruta ensemble selection
│
├── models/
│   ├── base_model.py           # Abstract base: fit/predict/save/load/mlflow
│   ├── tree_based/             # LightGBM, XGBoost, CatBoost
│   ├── neural/                 # Bidirectional LSTM + attention, TabNet
│   ├── ensemble/               # Stacking (OOF), WeightedBlending, VotingEnsemble
│   ├── pricing/                # Surge model, elasticity, A/B test framework
│   └── train_all.py            # Master training script
│
├── optimization/
│   ├── hpo/                    # Optuna TPE + MedianPruner (200 trials)
│   ├── cv/                     # WalkForwardCV, SlidingWindowCV, PurgedKFold
│   └── calibration/            # Isotonic, Platt, Temperature scaling
│
├── evaluation/
│   ├── metrics.py              # RMSE, MAPE, Pinball, Winkler, PSI, revenue backtest
│   └── compare_models.py       # HTML benchmark report generator
│
├── explainability/
│   ├── shap_explainer.py       # Global, local, force, dependence plots
│   ├── lime_explainer.py       # Local perturbation + SHAP/LIME agreement score
│   ├── pdp_anchors_dice.py     # PDP/ICE, Anchors IF-THEN, DiCE counterfactuals
│   └── report_generator.py     # Standalone HTML report (no external deps)
│
├── serving/
│   └── api.py                  # FastAPI: /predict/demand /price /health /metrics/cache
│
├── monitoring/
│   └── drift_detector.py       # PSI per feature, Evidently reports, email alerts
│
├── orchestration/
│   └── dags/retrain_demand.py  # Airflow: backfill → features → drift → train → promote
│
├── notebooks/
│   ├── 01_eda.py               # Demand distribution, heatmaps, autocorrelation
│   ├── 02_feature_analysis.py  # SHAP vs MI importance, lag correlation, selection
│   └── 03_model_comparison.py  # Full benchmark + all explainability tools
│
├── dashboard/
│   └── app.py                  # Streamlit: live scoring, surge pricing, demand curve
│
├── tests/
│   └── test_all.py             # 32 unit tests: features, models, CV, API, drift
│
├── k8s/
│   └── deployment.yaml         # Deployment, HPA (2-10 pods), Service, Redis, PVC
│
├── .github/workflows/
│   ├── ci.yml                  # Lint → test → Docker build → weekly retrain
│   └── retrain.yml             # Scheduled Monday 02:00 retraining
│
├── Dockerfile                  # Multi-stage build, ~200MB final image
├── docker-compose.yml          # Kafka + Redis + MLflow + API + Grafana
├── config.py                   # Centralised settings (Pydantic BaseSettings)
└── requirements.txt
```

---

## Quickstart

### Prerequisites

- Python 3.11
- Docker Desktop (for Kafka, Redis, MLflow, Grafana)
- Git

### 1. Clone and create environment

```bash
git clone https://github.com/youruser/rideflow-ml-system.git
cd rideflow-ml-system

conda create -n rms-env python=3.11 -y
conda activate rms-env
```

### 2. Install dependencies

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy
pip install mlflow optuna scipy
pip install fastapi uvicorn pydantic pydantic-settings redis httpx
pip install shap lime loguru typer rich tqdm python-dotenv
pip install pytest pytest-cov pytest-asyncio matplotlib seaborn
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed — defaults match docker-compose services
```

### 4. Start infrastructure

```bash
docker-compose up -d
# Wait ~30 seconds for all services to be healthy
docker-compose ps
```

### 5. Generate data and train

```bash
# Generate 90 days of synthetic Paris ride data (43,200 rows)
python producers/gps_producer.py --n-days 90 --output data/raw/

# Train all models (~10 minutes)
set PYTHONPATH=.      # Windows
export PYTHONPATH=.   # Mac/Linux
python models/train_all.py --experiment rideflow-v1
```

Expected output:
```
Model Leaderboard
┏━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Model    ┃ MAPE    ┃ RMSE  ┃ R²    ┃
┡━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ LightGBM │ 7.880%  │ 3.30  │ 0.981 │
│ Blending │ 8.340%  │ 3.23  │ 0.982 │
│ CatBoost │ 9.570%  │ 4.14  │ 0.970 │
└──────────┴─────────┴───────┴───────┘
Best model: LightGBM (MAPE=7.88%)
```

### 6. Run tests

```bash
pytest tests/test_all.py -v -m "not integration"
# Expected: 32 passed
```

### 7. Start API

```bash
# Stop Docker API container first if running
docker stop rideflow-ml-system-api-1

uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

### 8. Test endpoints

```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"redis_ok":true,"model_version":"best_model"}

# Demand prediction (first call: ~44ms)
curl -X POST http://localhost:8000/predict/demand \
  -H "Content-Type: application/json" \
  -d '{"zone":"11","hour":18,"day_of_week":4,"month":3,"weather":"rain",
       "temperature_c":12.0,"precipitation_mm":5.0,"is_event":false,
       "demand_lag_1h":15.0,"demand_lag_24h":12.0,
       "demand_lag_168h":11.0,"demand_roll24h_mean":13.0}'
# {"predicted_demand":17.03,"latency_ms":44.14,"cached":false}

# Same call again (Redis cache hit: ~5ms)
# {"predicted_demand":17.03,"latency_ms":5.26,"cached":true}

# Dynamic pricing
curl -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{"zone":"11","predicted_demand":25.0,"zone_baseline":10.0}'
# {"surge_multiplier":1.5,"final_price_eur":12.0,"demand_ratio":2.5}
```

### 9. Explainability report

```bash
python notebooks/03_model_comparison.py
# Open: reports/explainability_report.html
```

### 10. Streamlit dashboard

```bash
pip install streamlit plotly
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

---

## Services & URLs

| Service | URL | Credentials |
|---|---|---|
| FastAPI docs | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / rideflow |
| Streamlit | http://localhost:8501 | — |

---

## Feature Engineering

74 features are engineered from 5 raw signals (zone, timestamp, weather, demand, events):

**Temporal (lag & rolling)**
- Lag features: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h — capture short-term momentum and weekly seasonality
- Rolling statistics: mean, std, min, max over 3h / 6h / 12h / 24h windows
- Velocity: rate of change over 1h, 3h, 6h — captures acceleration of demand spikes
- Fourier terms: 3 harmonics daily, 2 weekly, 2 yearly — smooth seasonality without overfitting

**Calendar**
- Hour, day of week, month, week of year
- Binary flags: is_weekend, is_rush_am, is_rush_pm, is_friday_night, is_night, is_holiday

**Geospatial**
- Zone encoding and cluster (central / mid-ring / outer)
- Adjacency demand: mean demand of neighbouring arrondissements
- POI proximity: weighted distance to 8 major POIs (CDG, Gare du Nord, Bercy Arena, Parc des Princes...)
- Demand share: zone demand as fraction of total Paris demand at that hour
- Demand rank: zone rank by demand at each timestamp

**Contextual**
- Weather encoding, temperature, precipitation, wind speed
- Event binary flag and name

**Selection**: SHAP importance + Mutual Information + Boruta — 33 features selected from 74 by majority vote across methods.

---

## Model Details

### Training protocol

- **Split**: 70% train / 15% val / 15% test (time-ordered, no shuffle)
- **Validation**: walk-forward CV with 24h gap between train and val boundaries
- **HPO**: Optuna TPE sampler + MedianPruner, 200 trials per model
- **Calibration**: isotonic regression on validation predictions (reduces MAE by ~30%)

### Ensemble

```
LightGBM ──┐
XGBoost  ──┼── OOF predictions ──► Ridge meta-learner ──► Stacking
CatBoost ──┘

LightGBM ──┐
XGBoost  ──┼── scipy minimize ──► optimal weights ──► WeightedBlending
CatBoost ──┘
```

### Pricing engine

```
predicted_demand / zone_baseline = demand_ratio

demand_ratio ≥ 4.0  →  surge 2.5×  →  €20.00
demand_ratio ≥ 3.0  →  surge 1.8×  →  €14.40
demand_ratio ≥ 2.0  →  surge 1.5×  →  €12.00
demand_ratio ≥ 1.5  →  surge 1.2×  →  €9.60
demand_ratio < 1.5  →  surge 1.0×  →  €8.00
```

---

## Monitoring & Retraining

### Drift detection

PSI (Population Stability Index) is computed weekly for every feature:

| PSI | Status | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 – 0.20 | Warning | Monitor closely |
| > 0.20 | Drift | Trigger retraining |

### Airflow DAG (weekly, Monday 02:00)

```
data_backfill → feature_pipeline → drift_check
                                        │
                              ┌─────────┴─────────┐
                          retrain              skip_training
                              │
                    evaluate_champion
                              │
                      promote_model → notify
```

The champion/challenger comparison requires >1% MAPE improvement before promotion. All runs are logged to MLflow with full parameter and metric tracking.

---

## CI/CD

GitHub Actions runs on every push to `main` and `dev`:

1. **Lint** — Ruff + mypy type checking
2. **Test** — pytest 32 tests, coverage report uploaded to Codecov
3. **Docker** — multi-stage build, container health check
4. **Weekly retrain** — scheduled Monday 02:00 via cron trigger

---

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/deployment.yaml

# Check pods
kubectl get pods

# Check HPA (scales 2→10 pods on CPU >70%)
kubectl get hpa
```

The HPA autoscales the API deployment from 2 to 10 pods based on CPU and memory utilisation — designed to absorb Friday evening demand spikes without manual intervention.

---

## Stack

| Layer | Technology |
|---|---|
| Data streaming | Apache Kafka · Spark Structured Streaming |
| Feature store | Redis (online) · Apache Parquet (offline) |
| ML models | LightGBM · XGBoost · CatBoost · PyTorch LSTM · TabNet |
| Optimisation | Optuna TPE · walk-forward CV · isotonic calibration |
| Explainability | SHAP · LIME · PDP/ICE · Anchors (alibi) · DiCE |
| MLOps | MLflow · Airflow · Docker · Kubernetes · GitHub Actions |
| Serving | FastAPI · Uvicorn · Pydantic v2 · Redis cache |
| Monitoring | Evidently · Grafana · PSI · email alerts |
| Testing | pytest · pytest-cov · pytest-asyncio |

---

## License

MIT — see `LICENSE` for details.

---

## Author

**Achraf Jemali** — Data & AI Engineer.

[![GitHub](https://img.shields.io/badge/GitHub-JEMALIACHRAF-black?logo=github&style=flat-square)](https://github.com/JEMALIACHRAF)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Achraf_Jemali-0077B5?logo=linkedin&style=flat-square)](https://linkedin.com/in/achraf-jemali-54a417239)

If you found this useful or want to discuss the design choices, feel free to reach out.
