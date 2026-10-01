# Machine Learning Based Crop Recommendation, Agricultural Risk Analysis and Future Suitability Prediction

A Flask web app with two parts:

1. **Current conditions** - from N, P, K, temperature, humidity, pH and rainfall it recommends a crop (with top-3 candidates) and classifies agricultural risk as **Low / Medium / High**, checking all seven parameters (including humidity) against crop-specific ranges.
2. **Future prediction** - from historical monthly temperature, humidity and rainfall it forecasts the coming months, then gives a crop recommendation and risk level for each forecast month using the soil values you enter.

## Project structure

```
.
├── app.py
├── gunicorn.conf.py
├── requirements.txt
├── Procfile
├── data/
│   ├── generate_dataset.py
│   ├── crop_dataset.csv
│   ├── crop_risk_reference.csv
│   └── historical_weather.csv
├── src/
│   ├── config.py
│   ├── train_model.py
│   ├── build_risk_reference.py
│   ├── recommender.py
│   ├── risk_assessment.py
│   └── forecast.py
├── model/
├── templates/
│   ├── base.html
│   ├── index.html
│   └── future.html
└── static/
    └── style.css
```

## What changed

- **Humidity in risk analysis** - `risk_assessment.py` now checks temperature, humidity, pH, rainfall, N, P and K. Risk is Low with no out-of-range factors, Medium with 1-2, High with 3 or more.
- **Risk reference built from the dataset** - `src/build_risk_reference.py` computes each crop's suitable range (5th to 95th percentile of that crop's rows) for all seven parameters, and derives Low/Medium/High water requirement from each crop's median rainfall. Replace the dataset, re-run it, and the risk ranges follow the real data.
- **Future prediction** - `src/forecast.py` fits a linear trend plus yearly seasonality per variable (temperature, humidity, rainfall) on monthly history, forecasts 1-12 months, and reports mean absolute error on the last 12 held-out months. The `/future` page feeds the forecast into the same crop model and risk assessment. You can upload your own historical CSV on that page.
- **Model files** - the trained Random Forest is too large to ship in a ZIP, so the app trains automatically on first use if `model/` has no trained model. You can also train it ahead of time (below).
- **Comments removed** from all code files.

## Before final submission: use real data

`data/crop_dataset.csv` and `data/historical_weather.csv` in this ZIP are **synthetic placeholders** produced by `data/generate_dataset.py` so the project runs end to end. Do not present them as real data.

1. **Crop dataset** - replace `data/crop_dataset.csv` with a real public crop recommendation dataset. Keep the columns `N, P, K, temperature, humidity, ph, rainfall, label`.
2. **Historical weather** - replace `data/historical_weather.csv` with real observations for your region (for example from IMD, NASA POWER or Open-Meteo). Columns: `date, temperature, humidity, rainfall`; daily or monthly rows; at least 36 months. Daily rainfall is summed per month, temperature and humidity are averaged.
3. Re-run the two commands below so the risk reference and model match the new data.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data/generate_dataset.py
python -m src.build_risk_reference
python -m src.train_model

python app.py
```

Skip `generate_dataset.py` once you have added real data. Then open http://127.0.0.1:5000 (current conditions) or http://127.0.0.1:5000/future (future prediction).

## Deploy (Render)

1. Push the project to GitHub.
2. Render: New, Web Service, connect the repo.
3. Build command: `pip install -r requirements.txt && python -m src.build_risk_reference && python -m src.train_model`
4. Start command: `gunicorn app:app`

Gunicorn settings (1 worker with 4 threads, 120 s timeout, model preloaded at startup) live in `gunicorn.conf.py`, which gunicorn reads automatically from the project root, so the start command does not need extra flags. They can be overridden with the `WEB_CONCURRENCY`, `GUNICORN_THREADS` and `GUNICORN_TIMEOUT` environment variables.

The model must be trained in the build step. The server never trains inside a web request; if the model files are missing it returns a clear "model not available" message and logs the reason.

The `/health` route can be used as the Render health check path.

## Results panel

Results appear in a sidebar next to the form (on screens narrower than 960 px it becomes a bottom sheet with a Close button). The home page shows current recommendation, risk analysis and future prediction together; the Future Prediction page shows the future prediction. The forms still work without JavaScript, in which case results render on the reloaded page.

## Limitations

- The forecast is a simple statistical model (trend plus seasonality). It suits short horizons and is a decision-support estimate, not a weather forecast.
- Soil values (N, P, K, pH) are assumed constant over the forecast period.
- No authentication, database or rate limiting; add those before real-world use.
