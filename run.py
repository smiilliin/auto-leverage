import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from features import make_features
from target import make_target
from walk_forward import walk_forward_predict
from strategy import make_strategy_returns

# ========================================
# Settings
# ========================================

MODEL_PATH = "volatility_model.pkl"
BACKTEST_PATH = "backtest_results.csv"
LEVERAGE_PATH = "leverage.json"

TARGET_VOLATILITY = 0.25
ALPHA = 1.0

MIN_LEVERAGE = 0.5
MAX_LEVERAGE = 2.0

EMA_SPAN = 10

TRAIN_SIZE = 756
TEST_SIZE = 63

FEATURES = [
    "rv_20",
    "rv_60",
    "rv_ratio",
    "vix",
    "vix_change",
]


# ========================================
# Load model
# ========================================

model = joblib.load(MODEL_PATH)


# ========================================
# Download data
# ========================================

print("Loading market data...")

data = yf.download(
    ["QQQ", "^VIX"],
    start="2010-01-01",
    auto_adjust=True,
    progress=False,
)


# ========================================
# Build features
# ========================================

features = make_features(data)


# ========================================
# Current prediction
# ========================================

# target과 완전히 독립적으로 최신 feature를 사용
latest_features = features[FEATURES].dropna()


predicted_volatility = pd.Series(
    model.predict(latest_features),
    index=latest_features.index,
)


# ========================================
# Build target for backtest
# ========================================

target = make_target(
    data,
    horizon=21,
)


# ========================================
# Build backtest dataset
# ========================================

dataset = pd.concat(
    [features, target],
    axis=1,
).dropna()


X = dataset[FEATURES]
y = dataset["future_volatility"]


# ========================================
# Walk-forward backtest
# ========================================

print("Running walk-forward backtest...")

prediction = walk_forward_predict(
    X,
    y,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
)

prediction.name = "predicted_volatility"


qqq = data["Close"]["QQQ"].reindex(prediction.index)


# ========================================
# Backtest strategy
# ========================================

results = make_strategy_returns(
    qqq,
    prediction,
)


# ========================================
# Save backtest
# ========================================

results.to_csv(BACKTEST_PATH)

print(f"Backtest saved to: {BACKTEST_PATH}")


# ========================================
# Raw leverage
# ========================================

raw_leverage = (TARGET_VOLATILITY / predicted_volatility) ** ALPHA


raw_leverage = raw_leverage.clip(
    MIN_LEVERAGE,
    MAX_LEVERAGE,
)


# ========================================
# EMA leverage
# ========================================

smooth_leverage = raw_leverage.ewm(
    span=EMA_SPAN,
    adjust=False,
).mean()


# ========================================
# Latest values
# ========================================

latest_date = predicted_volatility.index[-1]

latest_predicted_volatility = predicted_volatility.iloc[-1]

latest_raw_leverage = raw_leverage.iloc[-1]

latest_smooth_leverage = smooth_leverage.iloc[-1]


# ========================================
# Save leverage
# ========================================

output = {
    "data_date": latest_date.date().isoformat(),
    "predicted_21d_volatility": float(latest_predicted_volatility),
    "target_volatility": float(TARGET_VOLATILITY),
    "alpha": float(ALPHA),
    "raw_leverage": float(latest_raw_leverage),
    "ema_leverage": float(latest_smooth_leverage),
}


with open(
    LEVERAGE_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    json.dumps(
        output,
        indent=2,
        ensure_ascii=False,
    )
)
