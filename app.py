import pickle

import pandas as pd
from flask import Flask, render_template, request

from src.config import CLIMATE_FEATURES, FEATURES, MODEL_DIR, SOIL_FEATURES
from src.forecast import MAX_HORIZON, forecast_weather, load_history
from src.recommender import recommend
from src.risk_assessment import assess_risk, load_risk_reference
from src.train_model import train

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

_artifacts = {}


def get_artifacts():
    if not _artifacts:
        if not (MODEL_DIR / "model.pkl").exists() or not (MODEL_DIR / "label_encoder.pkl").exists():
            train()
        with open(MODEL_DIR / "model.pkl", "rb") as f:
            _artifacts["model"] = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            _artifacts["label_encoder"] = pickle.load(f)
        _artifacts["risk_ref"] = load_risk_reference()
    return _artifacts["model"], _artifacts["label_encoder"], _artifacts["risk_ref"]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {f: "" for f in FEATURES}

    if request.method == "POST":
        try:
            form_values = {f: request.form.get(f, "") for f in FEATURES}
            values = {f: float(request.form[f]) for f in FEATURES}

            model, le, risk_ref = get_artifacts()
            X = pd.DataFrame([values])[FEATURES]
            top3 = recommend(model, le, X)[0]
            crop = top3[0][0]
            risk = assess_risk(values, crop, risk_ref)

            result = {
                "crop": crop,
                "top3": top3,
                "risk_level": risk["risk_level"],
                "flags": risk["flags"],
                "details": risk["details"],
                "water_requirement": risk["water_requirement"],
            }
        except Exception as exc:
            error = f"Could not generate a recommendation: {exc}"

    return render_template("index.html", result=result, error=error, values=form_values, page="current")


@app.route("/future", methods=["GET", "POST"])
def future():
    result = None
    error = None
    form_values = {f: "" for f in SOIL_FEATURES}
    form_values["horizon"] = "6"

    if request.method == "POST":
        try:
            form_values = {f: request.form.get(f, "") for f in SOIL_FEATURES}
            form_values["horizon"] = request.form.get("horizon", "6")
            soil = {f: float(request.form[f]) for f in SOIL_FEATURES}
            horizon = int(form_values["horizon"])

            upload = request.files.get("history")
            source = upload if upload is not None and upload.filename else None
            history = load_history(source)
            forecast, errors = forecast_weather(history, horizon)

            model, le, risk_ref = get_artifacts()
            rows = []
            for _, row in forecast.iterrows():
                values = {**soil, **{c: float(row[c]) for c in CLIMATE_FEATURES}}
                X = pd.DataFrame([values])[FEATURES]
                top3 = recommend(model, le, X)[0]
                crop, confidence = top3[0]
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
            result = {
                "rows": rows,
                "best_crop": best_crop,
                "best_crop_months": int((crops == best_crop).sum()),
                "high_risk_months": sum(1 for r in rows if r["risk_level"] == "High"),
                "history_months": len(history),
                "history_from": history.index[0].strftime("%b %Y"),
                "history_to": history.index[-1].strftime("%b %Y"),
                "errors": errors,
            }
        except Exception as exc:
            error = f"Could not generate a forecast: {exc}"

    return render_template(
        "future.html",
        result=result,
        error=error,
        values=form_values,
        max_horizon=MAX_HORIZON,
        page="future",
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
