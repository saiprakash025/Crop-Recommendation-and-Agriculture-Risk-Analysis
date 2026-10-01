import logging
import math
import pickle
import threading

import pandas as pd

from src.config import (CLIMATE_FEATURES, DEFAULT_HORIZON, FEATURES, INPUT_LIMITS,
                        MODEL_DIR, SOIL_FEATURES)
from src.errors import ModelUnavailableError, UserInputError
from src.forecast import MAX_HORIZON, forecast_weather, load_history
from src.recommender import recommend
from src.risk_assessment import assess_risk, load_risk_reference

log = logging.getLogger(__name__)

_lock = threading.Lock()
_artifacts = {}


def load_artifacts(train_if_missing=False):
    with _lock:
        if _artifacts:
            return _artifacts["model"], _artifacts["label_encoder"], _artifacts["risk_ref"]

        model_path = MODEL_DIR / "model.pkl"
        encoder_path = MODEL_DIR / "label_encoder.pkl"

        if not model_path.exists() or not encoder_path.exists():
            if not train_if_missing:
                raise ModelUnavailableError(
                    f"Trained model files not found in {MODEL_DIR}. "
                    "Run 'python -m src.train_model' during the build step."
                )
            from src.train_model import train
            train()

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        _artifacts["model"] = model
        _artifacts["label_encoder"] = label_encoder
        _artifacts["risk_ref"] = load_risk_reference()
        log.info("Model artifacts loaded from %s (%s)", MODEL_DIR, type(model).__name__)
        return model, label_encoder, _artifacts["risk_ref"]


def parse_inputs(form, fields):
    values = {}
    for field in fields:
        label, low, high = INPUT_LIMITS[field]
        raw = (form.get(field) or "").strip()
        if raw == "":
            raise UserInputError(f"{label} is required.")
        try:
            number = float(raw)
        except ValueError:
            raise UserInputError(f"{label} must be a number.")
        if not math.isfinite(number):
            raise UserInputError(f"{label} must be a valid number.")
        if number < low or number > high:
            raise UserInputError(f"{label} must be between {low} and {high}.")
        values[field] = number
    return values


def parse_horizon(form):
    raw = (form.get("horizon") or "").strip() or str(DEFAULT_HORIZON)
    try:
        horizon = int(raw)
    except ValueError:
        raise UserInputError("Forecast horizon must be a whole number of months.")
    if not 1 <= horizon <= MAX_HORIZON:
        raise UserInputError(f"Forecast horizon must be between 1 and {MAX_HORIZON} months.")
    return horizon


def run_current(values):
    model, label_encoder, risk_ref = load_artifacts()
    frame = pd.DataFrame([values])[FEATURES].astype(float)
    top3 = recommend(model, label_encoder, frame)[0]
    crop = top3[0][0]
    risk = assess_risk(values, crop, risk_ref)
    if risk["flags"]:
        reason = "Outside the suitable range: " + ", ".join(risk["flags"]) + "."
    else:
        reason = "All input parameters are within the suitable range for this crop."
    return {
        "crop": crop,
        "top3": top3,
        "risk_level": risk["risk_level"],
        "flags": risk["flags"],
        "reason": reason,
        "details": risk["details"],
        "water_requirement": risk["water_requirement"],
    }


def run_future(soil, horizon, source=None):
    history = load_history(source)
    forecast, errors = forecast_weather(history, horizon)

    model, label_encoder, risk_ref = load_artifacts()

    frame = forecast[CLIMATE_FEATURES].copy()
    for field in SOIL_FEATURES:
        frame[field] = soil[field]
    frame = frame[FEATURES].astype(float)
    if frame.isna().any().any():
        raise ValueError("Forecast produced missing values.")

    predictions = recommend(model, label_encoder, frame)

    rows = []
    for (_, row), top in zip(forecast.iterrows(), predictions):
        values = {**soil, **{c: float(row[c]) for c in CLIMATE_FEATURES}}
        crop, confidence = top[0]
        risk = assess_risk(values, crop, risk_ref)
        rows.append({
            "month": row["month"],
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "rainfall": row["rainfall"],
            "crop": crop,
            "confidence": confidence,
            "risk_level": risk["risk_level"],
            "flags": risk["flags"],
            "water_requirement": risk["water_requirement"],
        })

    crops = pd.Series([r["crop"] for r in rows])
    best_crop = crops.value_counts().index[0]
    return {
        "rows": rows,
        "first": rows[0],
        "best_crop": best_crop,
        "best_crop_months": int((crops == best_crop).sum()),
        "high_risk_months": sum(1 for r in rows if r["risk_level"] == "High"),
        "history_months": len(history),
        "history_from": history.index[0].strftime("%b %Y"),
        "history_to": history.index[-1].strftime("%b %Y"),
        "errors": errors,
    }
