import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Stock Direction ML Demo", layout="wide")

FEATURE_COLS = [
    "return_5d", "return_10d", "return_20d",
    "Volume_change", "Close_open_diff", "high_low_pct",
]


# ---------- Data + feature pipeline ----------

@st.cache_data(ttl=300)  # refresh from Yahoo every 5 minutes
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="2y")
    df = df.dropna(subset=["Close"])

    df["return_5d"] = df["Close"].pct_change(periods=5) * 100
    df["return_10d"] = df["Close"].pct_change(periods=10) * 100
    df["return_20d"] = df["Close"].pct_change(periods=20) * 100
    df["Volume_change"] = df["Volume"].pct_change(periods=20) * 100
    df["Close_open_diff"] = df["Close"] - df["Open"]
    df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df


@st.cache_resource(ttl=300)
def train_model(df: pd.DataFrame):
    clean = df.dropna()
    split_idx = int(len(clean) * 0.8)
    train, test = clean.iloc[:split_idx], clean.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    cm = confusion_matrix(y_test, preds)
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)

    return model, acc, baseline, cm, importances, test, preds


# ---------- UI ----------

st.title("📈 Live Stock Direction ML Demo")
st.caption("Educational demo — not investment advice. Predicting daily up/down direction, not price targets.")

ticker = st.sidebar.text_input("Ticker", value="MSFT").upper().strip()
st.sidebar.markdown("---")
st.sidebar.markdown(
    "This app pulls live data from Yahoo Finance, engineers the same six "
    "features used in the demo notebook, and trains an XGBoost classifier "
    "to predict whether tomorrow's close will be higher than today's."
)

with st.spinner(f"Fetching live data for {ticker}..."):
    try:
        data = load_data(ticker)
    except Exception as e:
        st.error(f"Couldn't fetch data for '{ticker}': {e}")
        st.stop()

if data.empty or len(data) < 60:
    st.error("Not enough data returned for this ticker. Try a different symbol.")
    st.stop()

model, acc, baseline, cm, importances, test, preds = train_model(data)

# --- Live price section ---
col1, col2, col3 = st.columns(3)
latest = data.iloc[-1]
prev = data.iloc[-2]
change = latest["Close"] - prev["Close"]
pct_change = (change / prev["Close"]) * 100

col1.metric(
    f"{ticker} Last Close",
    f"${latest['Close']:.2f}",
    f"{change:+.2f} ({pct_change:+.2f}%)",
)
col2.metric("Volume (latest session)", f"{latest['Volume']:,.0f}")
col3.metric("Data as of", str(data.index[-1].date()))

st.subheader(f"{ticker} — Close price, last 2 years")
st.line_chart(data["Close"])

st.subheader(f"{ticker} — Trading volume, last 2 years")
st.bar_chart(data["Volume"])

st.markdown("---")

# --- Model results section ---
st.header("🤖 Model Results")

mcol1, mcol2 = st.columns(2)
mcol1.metric("Model accuracy (test set)", f"{acc:.1%}")
mcol2.metric("Naive baseline (always predict majority class)", f"{baseline:.1%}")

gap = acc - baseline
if gap > 0.05:
    st.success(
        f"Model beats the baseline by {gap:.1%} — worth digging into whether this "
        "holds up out of sample, or is a lucky split."
    )
else:
    st.info(
        f"Model is within {abs(gap):.1%} of the naive baseline — this is the "
        "expected result for short-horizon price direction from price/volume "
        "history alone, and matches well-documented market efficiency research."
    )

rcol1, rcol2 = st.columns(2)

with rcol1:
    st.subheader("Confusion matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Down", "Actual: Up"],
        columns=["Predicted: Down", "Predicted: Up"],
    )
    st.dataframe(cm_df, width="stretch")

with rcol2:
    st.subheader("Feature importance")
    st.bar_chart(importances.sort_values(ascending=True))

st.subheader("Model prediction for the next session")
next_features = data[FEATURE_COLS].iloc[[-1]]
next_pred = model.predict(next_features)[0]
next_proba = model.predict_proba(next_features)[0]
direction = "📈 UP" if next_pred == 1 else "📉 DOWN"
confidence = next_proba[next_pred]
st.metric(f"{ticker} predicted direction for next session", direction, f"{confidence:.0%} confidence")
st.caption(
    "Reminder: given the accuracy numbers above, treat this prediction as an "
    "illustration of the pipeline, not a trading signal."
)
