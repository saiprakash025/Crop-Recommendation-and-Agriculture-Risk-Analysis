import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import CLIMATE_FEATURES, HISTORY_PATH
from src.errors import UserInputError

REQUIRED_COLUMNS = ["date"] + CLIMATE_FEATURES
MIN_MONTHS = 36
HOLDOUT_MONTHS = 12
MAX_HORIZON = 12


def load_history(source=None):
    try:
        df = pd.read_csv(source if source is not None else HISTORY_PATH)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        if source is None:
            raise
        raise UserInputError("The uploaded file could not be read as a CSV file.") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise UserInputError(
            "Historical weather file is missing column(s): "
            + ", ".join(missing)
            + ". Required columns: "
            + ", ".join(REQUIRED_COLUMNS)
            + "."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in CLIMATE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))
        .agg(temperature=("temperature", "mean"), humidity=("humidity", "mean"), rainfall=("rainfall", "sum"))
        .sort_index()
    )

    if len(monthly) < MIN_MONTHS:
        raise UserInputError(
            f"Need at least {MIN_MONTHS} months of historical data, found {len(monthly)}."
        )
    return monthly


def _design_matrix(periods):
    months = np.array([p.month for p in periods])
    ordinal = np.array([p.year * 12 + p.month for p in periods], dtype=float)
    trend = (ordinal - 24000.0) / 120.0
    angle = 2 * np.pi * months / 12
    return np.column_stack([trend, np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)])


def _clip(variable, values):
    values = np.asarray(values, dtype=float)
    if variable == "humidity":
        return np.clip(values, 0, 100)
    if variable == "rainfall":
        return np.clip(values, 0, None)
    return values


def _fit(monthly):
    X = _design_matrix(list(monthly.index))
    return {v: LinearRegression().fit(X, monthly[v].to_numpy()) for v in CLIMATE_FEATURES}


def backtest(monthly):
    train, test = monthly.iloc[:-HOLDOUT_MONTHS], monthly.iloc[-HOLDOUT_MONTHS:]
    models = _fit(train)
    X_test = _design_matrix(list(test.index))
    errors = {}
    for v in CLIMATE_FEATURES:
        pred = _clip(v, models[v].predict(X_test))
        errors[v] = round(float(np.mean(np.abs(pred - test[v].to_numpy()))), 2)
    return errors


def forecast_weather(monthly, horizon):
    if not 1 <= horizon <= MAX_HORIZON:
        raise UserInputError(f"Forecast horizon must be between 1 and {MAX_HORIZON} months.")

    models = _fit(monthly)
    last = monthly.index[-1]
    future_periods = [last + i for i in range(1, horizon + 1)]
    X_future = _design_matrix(future_periods)

    future = pd.DataFrame({"period": future_periods})
    for v in CLIMATE_FEATURES:
        future[v] = np.round(_clip(v, models[v].predict(X_future)), 2)
    future["month"] = [p.strftime("%b %Y") for p in future_periods]
    return future[["month"] + CLIMATE_FEATURES], backtest(monthly)
