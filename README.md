# Autoscope — Car Price Intelligence

A Streamlit dashboard built around your Linear / Ridge / Lasso regression notebook for
predicting used-car prices.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## What's inside

- **Overview** — headline metrics, an R² gauge for the best model, and a raw data sample.
- **Explore the Data** — the Price vs. log(Price) distributions, the EngineV outlier
  cleanup (the famous "99.99 litre engine"), a correlation heatmap, and brand/mileage charts.
- **Model Lab** — side-by-side R² and RMSE for Linear, Ridge and Lasso, an actual-vs-predicted
  plot, and a feature-coefficient comparison (great for showing how Lasso zeroes out features).
- **Predict a Price** — a form (brand, body type, engine, mileage, year, registration) that
  runs the best model live and returns an estimated price with a confidence range.

## Using your own data

The app ships with a synthetic dataset that mimics the shape and quirks of the original
"used car sales" CSV (same columns, missing values, and an EngineV data-entry glitch) so it
runs immediately with no external downloads. To use your real dataset instead, upload a CSV
from the sidebar with these columns:

```
Brand, Price, Body, Mileage, EngineV, Engine Type, Registration, Year
```

(`Model` is optional — it's dropped before modeling, same as in the notebook.)

## Notes

- The pipeline mirrors your notebook exactly: drop missing `Price`/`EngineV` → drop
  `EngineV > 10` outliers → log-transform `Price` → label-encode categoricals → train Linear,
  Ridge and Lasso → compare R² and RMSE.
- The "best model" shown throughout is whichever of the three scores highest on the held-out
  test set (usually Linear or Ridge, unless you tune Lasso's alpha).
