# Stock Direction ML Demo

A small end-to-end machine learning project that pulls live stock market data,
engineers predictive features, trains an XGBoost classifier to predict
next-day price direction (up/down), and presents the results in an
interactive dashboard.

**This is an educational demo, not investment advice.** The model predicts
short-horizon price *direction*, not price targets, and its accuracy should
be interpreted against a naive baseline (see Results below).

## What's in this repo

- `data/` — Jupyter notebook used to explore the data, build and test the
  feature engineering pipeline, and train/evaluate the model step by step.
- `dashboard.py` — Standalone Streamlit app that reproduces the same
  pipeline and presents live data, model predictions, accuracy metrics, and
  feature importance in a browser-based UI.
- `README.md` — this file.

## Dependencies

Install with pip:

```bash
pip install pandas numpy matplotlib yfinance xgboost scikit-learn streamlit
```

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and feature engineering (rolling windows, returns, shifts) |
| `numpy` | Numerical operations |
| `matplotlib` | Static charts in the notebook |
| `yfinance` | Live and historical stock market data from Yahoo Finance |
| `xgboost` | Gradient-boosted tree classifier used for the prediction model |
| `scikit-learn` | Train/test evaluation metrics (accuracy, confusion matrix) |
| `streamlit` | Interactive browser-based dashboard |

Tested with Python 3.x in a virtual environment (`.venv`).

## Running the notebook

Open `data/*.ipynb` in Jupyter or PyCharm and run all cells top to bottom
(Kernel → Restart & Run All is recommended to avoid stale-variable issues).

## Running the dashboard

From the project root, with the virtual environment activated:

```bash
streamlit run dashboard.py
```

This opens a browser tab at `http://localhost:8501` showing:

- Live price and volume charts for a chosen ticker
- Model accuracy vs. a naive baseline
- Confusion matrix and feature importance
- A next-session direction prediction (for illustration, not trading use)

## Methodology notes

- **Target**: whether the next trading day's close is higher than the
  current day's close (binary classification).
- **Features**: 5/10/20-day returns, 20-day volume change, close-open
  difference, and intraday high-low range as a percentage of close.
- **Train/test split**: chronological (not shuffled) — the most recent ~20%
  of trading days are held out as the test set, since shuffling time-series
  data would let the model "see" future information during training.
- **Evaluation**: model accuracy is compared against a naive baseline
  (always predicting the majority class) rather than judged in isolation,
  since short-horizon price direction is famously close to unpredictable
  from price/volume history alone.

## Credits

This project was built collaboratively with **Claude** (Anthropic), which
assisted with:

- Planning the overall feature engineering and modeling approach
- Explaining core ML/pandas concepts (rolling windows, lag features,
  target construction, train/test splitting) from first principles
- Debugging issues in the notebook (variable scope errors, accidental
  column overwrites, incorrect lookback periods, a stray `dask` import,
  and an invalid XGBoost parameter)
- Diagnosing a network/SSL error from the Yahoo Finance API
- Building the Streamlit dashboard and troubleshooting its launch
  (PyCharm's Run button vs. the `streamlit run` command, a deprecated
  Streamlit parameter, and file-path issues)

All modeling decisions, data interpretation, and final presentation were
reviewed and directed by the project author.
