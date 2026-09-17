import joblib
import pandas as pd

from data import load_data
from features import make_features
from target import make_target
from model import create_model

# ========================================
# Load data
# ========================================

data = load_data()

features = make_features(data)

target = make_target(
    data,
    horizon=21,
)

dataset = pd.concat(
    [features, target],
    axis=1,
).dropna()


# ========================================
# Features
# ========================================

volatility_features = [
    "rv_20",
    "rv_60",
    "rv_ratio",
    "vix",
    "vix_change",
]

X = dataset[volatility_features]

y = dataset["future_volatility"]


# ========================================
# Train final model
# ========================================

model = create_model()

model.fit(
    X,
    y,
)


# ========================================
# Save
# ========================================

joblib.dump(
    model,
    "volatility_model.pkl",
)

print("Model saved to volatility_model.pkl")
