"""TempNet: ANN-based next-day temperature prediction on synthetic weather data."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_CSV = DATA_DIR / "weather_data.csv"
PROCESSED_CSV = PROCESSED_DIR / "weather_processed.csv"
MODEL_PATH = MODELS_DIR / "temperature_ann.keras"
INPUT_SCALER_PATH = MODELS_DIR / "input_scaler.pkl"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.pkl"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"

SEED = 42

# Seven current-day (day t) features used as ANN inputs
FEATURES = [
    "max_temperature",
    "min_temperature",
    "humidity",
    "wind_speed",
    "rainfall",
    "cloud_cover",
    "solar_radiation",
]

# Two next-day (day t+1) targets predicted by the ANN
TARGETS = ["next_max_temperature", "next_min_temperature"]
