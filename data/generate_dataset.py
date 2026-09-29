import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATASET_PATH, HISTORY_PATH

rng = np.random.default_rng(42)

CROPS = ["Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", "Mothbeans", "Mungbean",
         "Blackgram", "Lentil", "Pomegranate", "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon",
         "Apple", "Orange", "Papaya", "Coconut", "Cotton", "Jute", "Coffee", "Wheat", "Barley", "Millet",
         "Sorghum", "Groundnut", "Soybean", "Sunflower", "Sugarcane", "Tobacco", "Tea", "Rubber",
         "Cashew", "Areca nut", "Turmeric", "Ginger", "Cardamom", "Blackpepper", "Chilli", "Onion",
         "Garlic", "Potato", "Tomato", "Brinjal", "Cauliflower", "Cabbage", "Okra", "Peas", "Carrot", "Beetroot"]


def build_profiles():
    profiles = {}
    for c in CROPS:
        profiles[c] = dict(
            N=rng.uniform(10, 140), P=rng.uniform(5, 140), K=rng.uniform(5, 140),
            temperature=rng.uniform(15, 35), humidity=rng.uniform(30, 95),
            ph=rng.uniform(4.5, 8.5), rainfall=rng.uniform(40, 300),
        )
    return profiles


def generate_training_data(profiles, n_per_crop=400):
    rows = []
    for c in CROPS:
        p = profiles[c]
        for _ in range(n_per_crop):
            rows.append(dict(
                N=max(0, rng.normal(p["N"], 12)),
                P=max(0, rng.normal(p["P"], 12)),
                K=max(0, rng.normal(p["K"], 12)),
                temperature=rng.normal(p["temperature"], 3.5),
                humidity=np.clip(rng.normal(p["humidity"], 8), 10, 100),
                ph=np.clip(rng.normal(p["ph"], 0.4), 3.5, 9.5),
                rainfall=max(0, rng.normal(p["rainfall"], 35)),
                label=c,
            ))
    df = pd.DataFrame(rows)
    dup = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    for col in ["humidity", "rainfall"]:
        idx = df.sample(frac=0.005, random_state=2).index
        df.loc[idx, col] = np.nan
    df.to_csv(DATASET_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {DATASET_PATH}")


def generate_historical_weather(start="2016-01-01", end="2026-08-01"):
    dates = pd.date_range(start, end, freq="MS")
    month = dates.month.to_numpy()
    years = (dates.year - dates.year[0]).to_numpy() + month / 12

    temperature = 28 + 4.5 * np.sin(2 * np.pi * (month - 2) / 12) + 0.05 * years + rng.normal(0, 0.8, len(dates))
    humidity = 68 + 12 * np.sin(2 * np.pi * (month - 8) / 12) + rng.normal(0, 3, len(dates))
    seasonal_rain = (
        30
        + 140 * np.exp(-((month - 10.5) ** 2) / (2 * 1.5 ** 2))
        + 60 * np.exp(-((month - 7.5) ** 2) / (2 * 1.2 ** 2))
    )
    rainfall = seasonal_rain * rng.lognormal(0, 0.25, len(dates))

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "temperature": np.round(temperature, 2),
        "humidity": np.round(np.clip(humidity, 20, 100), 2),
        "rainfall": np.round(np.clip(rainfall, 0, None), 2),
    })
    df.to_csv(HISTORY_PATH, index=False)
    print(f"Wrote {len(df)} monthly records -> {HISTORY_PATH}")


if __name__ == "__main__":
    profiles = build_profiles()
    generate_training_data(profiles)
    generate_historical_weather()
