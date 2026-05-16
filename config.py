"""Central configuration — all paths, constants, and env vars."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────
    ROOT_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = ROOT_DIR / "saved_models"
    REPORTS_DIR: Path = ROOT_DIR / "reports"

    # ── Kafka ──────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_GPS: str = "rideflow.gps"
    KAFKA_TOPIC_WEATHER: str = "rideflow.weather"
    KAFKA_TOPIC_EVENTS: str = "rideflow.events"

    # ── Redis ──────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL_SECONDS: int = 300

    # ── MLflow ─────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "rideflow-demand"

    # ── Model ──────────────────────────────────────────────
    TARGET_COL: str = "demand"
    HORIZON_HOURS: int = 1
    N_LAGS: list = [1, 2, 3, 6, 12, 24, 48, 168]
    ROLLING_WINDOWS: list = [3, 6, 12, 24]
    ZONES: list = [str(i) for i in range(1, 21)]  # Paris arrondissements 1-20

    # ── Feature store ──────────────────────────────────────
    FEATURE_STORE_PATH: Path = DATA_DIR / "feature_store"
    ONLINE_FEATURES_TTL: int = 600  # seconds

    # ── Training ───────────────────────────────────────────
    TRAIN_START: str = "2024-01-01"
    TRAIN_END: str = "2024-09-30"
    VAL_START: str = "2024-10-01"
    VAL_END: str = "2024-11-30"
    TEST_START: str = "2024-12-01"
    TEST_END: str = "2024-12-31"
    CV_N_SPLITS: int = 5
    CV_GAP_HOURS: int = 24

    # ── Optuna ─────────────────────────────────────────────
    OPTUNA_N_TRIALS: int = 200
    OPTUNA_TIMEOUT_SECONDS: int = 3600

    # ── API ────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CACHE_TTL: int = 60

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
for d in [settings.DATA_DIR, settings.RAW_DIR, settings.PROCESSED_DIR,
          settings.MODELS_DIR, settings.REPORTS_DIR, settings.FEATURE_STORE_PATH]:
    d.mkdir(parents=True, exist_ok=True)
