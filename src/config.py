from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

DATASET_PATH = DATA_DIR / "crop_dataset.csv"
RISK_REF_PATH = DATA_DIR / "crop_risk_reference.csv"
HISTORY_PATH = DATA_DIR / "historical_weather.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SOIL_FEATURES = ["N", "P", "K", "ph"]
CLIMATE_FEATURES = ["temperature", "humidity", "rainfall"]

INPUT_LIMITS = {
    "N": ("Nitrogen (N)", 0, 500),
    "P": ("Phosphorus (P)", 0, 500),
    "K": ("Potassium (K)", 0, 500),
    "temperature": ("Temperature", -10, 60),
    "humidity": ("Humidity", 0, 100),
    "ph": ("Soil pH", 0, 14),
    "rainfall": ("Rainfall", 0, 5000),
}

DEFAULT_HORIZON = 6
