import joblib
import pandas as pd
import yfinance as yf
from arch import arch_model

from garch_model import (
    create_garch_premium_model,
    make_direct_leverage_features,
    make_direct_leverage_target,
)

MODEL_PATH = "garch_regression_model.pkl"
RIDGE_ALPHA = 10.0
INVERSE_VOL_ALPHA = 4.1
TARGET_VOLATILITY = 0.18
TARGET_SHARPE = None
TARGET_MAX_DRAWDOWN = -0.35
MIN_LEVERAGE = 0.5
MAX_LEVERAGE = 1.0
EMA_SPAN = 7


def get_series(data, ticker):
    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        return close[ticker].dropna()

    return close.dropna()


def load_garch_inputs(period="20y"):
    market = yf.download(
        "^GSPC",
        period=period,
        auto_adjust=True,
        progress=False,
    )
    vix = yf.download(
        "^VIX",
        period=period,
        auto_adjust=True,
        progress=False,
    )

    sp_close = get_series(market, "^GSPC")
    vix_close = get_series(vix, "^VIX")

    sp_returns_decimal = sp_close.pct_change().dropna()
    sp_returns_percent = sp_returns_decimal * 100

    garch = arch_model(
        sp_returns_percent,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
    )
    fitted = garch.fit(disp="off")

    garch_variance = fitted.conditional_volatility**2
    garch_parameters = {
        "omega": float(fitted.params["omega"]),
        "alpha": float(fitted.params["alpha[1]"]),
        "beta": float(fitted.params["beta[1]"]),
    }
    premium = (vix_close**2) / 252 - garch_variance

    return (
        sp_close,
        sp_returns_decimal,
        sp_returns_percent,
        garch_variance,
        premium,
        garch_parameters,
        vix_close,
    )


def train_and_save():
    (
        sp_close,
        sp_returns_decimal,
        sp_returns_percent,
        garch_variance,
        premium,
        garch_parameters,
        vix_close,
    ) = load_garch_inputs(
        period="20y",
    )

    features = make_direct_leverage_features(
        garch_variance,
        premium,
        vix_close,
    )
    target_leverage = make_direct_leverage_target(
        sp_returns_decimal,
        target_volatility=TARGET_VOLATILITY,
        horizon=21,
        alpha=INVERSE_VOL_ALPHA,
        min_leverage=MIN_LEVERAGE,
        max_leverage=MAX_LEVERAGE,
    )

    regression_data = pd.concat(
        [features, target_leverage],
        axis=1,
        sort=False,
    ).dropna()

    feature_columns = ["garch_variance", "premium", "vix", "vix_change"]
    X = regression_data[feature_columns]
    y = regression_data["target_leverage"]

    model = create_garch_premium_model(alpha=RIDGE_ALPHA)
    model.fit(X, y)

    model_artifact = {
        "model": model,
        "inverse_vol_alpha": INVERSE_VOL_ALPHA,
        "feature_columns": feature_columns,
        "trained_through": sp_close.index[-1],
        "training_samples": len(regression_data),
        "garch_parameters": garch_parameters,
        "model_type": "direct_vix_premium_leverage",
        "ema_span": EMA_SPAN,
        "target_volatility": TARGET_VOLATILITY,
        "target_max_drawdown": TARGET_MAX_DRAWDOWN,
        "min_leverage": MIN_LEVERAGE,
        "max_leverage": MAX_LEVERAGE,
    }

    joblib.dump(
        model_artifact,
        MODEL_PATH,
    )

    print("GARCH Ridge model saved:", MODEL_PATH)
    print(
        pd.Series(
            {
                "ridge_alpha": RIDGE_ALPHA,
                "training_samples": len(regression_data),
                "trained_through": sp_close.index[-1],
                "target_volatility": TARGET_VOLATILITY,
                "target_max_drawdown": TARGET_MAX_DRAWDOWN,
                "min_leverage": MIN_LEVERAGE,
                "max_leverage": MAX_LEVERAGE,
                "ema_span": EMA_SPAN,
            }
        ).to_string()
    )


if __name__ == "__main__":
    train_and_save()
