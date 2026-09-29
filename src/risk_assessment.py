import pandas as pd

from src.config import RISK_REF_PATH

FACTOR_CHECKS = [
    ("temperature", "temp_min", "temp_max", "Temperature"),
    ("humidity", "humidity_min", "humidity_max", "Humidity"),
    ("ph", "ph_min", "ph_max", "Soil pH"),
    ("rainfall", "rainfall_min", "rainfall_max", "Rainfall"),
    ("N", "n_min", "n_max", "Nitrogen"),
    ("P", "p_min", "p_max", "Phosphorus"),
    ("K", "k_min", "k_max", "Potassium"),
]


def load_risk_reference():
    return pd.read_csv(RISK_REF_PATH)


def assess_risk(inputs, crop, risk_ref=None):
    if risk_ref is None:
        risk_ref = load_risk_reference()

    row = risk_ref[risk_ref["crop"].str.lower() == crop.lower()]
    if row.empty:
        return {
            "risk_level": "Unknown",
            "flags": [],
            "details": [f"No risk-reference profile found for '{crop}'."],
            "water_requirement": "N/A",
        }
    row = row.iloc[0]

    out_of_range = []
    details = []
    for field, min_col, max_col, label in FACTOR_CHECKS:
        value = inputs.get(field)
        if value is None or min_col not in row.index or max_col not in row.index:
            continue
        value = round(float(value), 2)
        lo, hi = row[min_col], row[max_col]
        if value < lo or value > hi:
            out_of_range.append(label)
            details.append(f"{label} ({value}) is outside the suitable range [{lo}, {hi}] for {crop}.")
        else:
            details.append(f"{label} ({value}) is within the suitable range [{lo}, {hi}] for {crop}.")

    n_flags = len(out_of_range)
    if n_flags == 0:
        risk_level = "Low"
    elif n_flags <= 2:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_level": risk_level,
        "flags": out_of_range,
        "details": details,
        "water_requirement": row["water_requirement"],
    }
