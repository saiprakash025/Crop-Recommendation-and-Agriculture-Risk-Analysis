import pandas as pd

from src.config import DATASET_PATH, RISK_REF_PATH

LOW_QUANTILE = 0.05
HIGH_QUANTILE = 0.95

COLUMN_NAMES = {
    "temperature": ("temp_min", "temp_max"),
    "humidity": ("humidity_min", "humidity_max"),
    "ph": ("ph_min", "ph_max"),
    "rainfall": ("rainfall_min", "rainfall_max"),
    "N": ("n_min", "n_max"),
    "P": ("p_min", "p_max"),
    "K": ("k_min", "k_max"),
}


def build():
    df = pd.read_csv(DATASET_PATH).drop_duplicates().dropna()

    rows = []
    for crop, group in df.groupby("label"):
        row = {"crop": crop}
        for feature, (lo_col, hi_col) in COLUMN_NAMES.items():
            row[lo_col] = round(float(group[feature].quantile(LOW_QUANTILE)), 2)
            row[hi_col] = round(float(group[feature].quantile(HIGH_QUANTILE)), 2)
        row["median_rainfall"] = float(group["rainfall"].median())
        rows.append(row)

    ref = pd.DataFrame(rows)
    ranks = ref["median_rainfall"].rank(method="first")
    ref["water_requirement"] = pd.qcut(ranks, 3, labels=["Low", "Medium", "High"]).astype(str)
    ref = ref.drop(columns="median_rainfall")

    ref.to_csv(RISK_REF_PATH, index=False)
    print(f"Wrote {len(ref)} crop profiles -> {RISK_REF_PATH}")


if __name__ == "__main__":
    build()
