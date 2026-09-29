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
