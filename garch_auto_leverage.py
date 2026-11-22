import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import json

from garch_model import (
    calculate_auto_leverage_performance,
    garch_variance_from_parameters,
    make_direct_leverage_features,
    make_direct_leverage_results,
)

MODEL_PATH = "garch_regression_model.pkl"


def get_series(data, ticker):
    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        return close[ticker].dropna()

    return close.dropna()


def load_garch_inputs(
    period="20y",
):
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

    return (
        sp_close,
        sp_returns_decimal,
        sp_returns_percent,
        vix_close,
    )


def make_model_prediction(model_artifact, garch_variance, premium, vix):
    features = make_direct_leverage_features(
        garch_variance,
        premium,
        vix,
    )
    model = model_artifact["model"]
    feature_columns = model_artifact["feature_columns"]

    prediction_values = model.predict(features[feature_columns])

    return pd.Series(
        prediction_values,
        index=features.index,
        name="predicted_future_variance",
    )


def run():
    model_artifact = joblib.load(MODEL_PATH)

    (
        sp_close,
        sp_returns_decimal,
        sp_returns_percent,
        vix_close,
    ) = load_garch_inputs()

    garch_variance = garch_variance_from_parameters(
        sp_returns_percent,
        model_artifact["garch_parameters"],
    )
    premium = (vix_close**2) / 252 - garch_variance

    predicted_leverage = make_model_prediction(
        model_artifact,
        garch_variance,
        premium,
        vix_close,
    )

    if predicted_leverage.empty:
        raise RuntimeError("예측값을 생성하지 못했습니다.")

    results = make_direct_leverage_results(
        sp_returns_decimal,
        predicted_leverage,
        min_leverage=model_artifact["min_leverage"],
        max_leverage=model_artifact["max_leverage"],
        ema_span=model_artifact["ema_span"],
    )
    performance = calculate_auto_leverage_performance(results["strategy_return"])

    print("Direct leverage model:")
    print(
        pd.Series(
            {
                "model_type": model_artifact["model_type"],
                "target_volatility": model_artifact["target_volatility"],
                "target_max_drawdown": model_artifact.get(
                    "target_max_drawdown",
                    -0.35,
                ),
                "min_leverage": model_artifact["min_leverage"],
                "max_leverage": model_artifact["max_leverage"],
                "ema_span": model_artifact["ema_span"],
            }
        ).to_string()
    )
    print("\nS&P 500 auto-leverage performance:")
    print(pd.Series(performance).to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nLatest leverage:")
    print(results["leverage"].dropna().tail(1).to_string())
    print(
        "\nActive strategy period:"
        f"{results['strategy_return'].dropna().index.min().date()} "
        f"to {results['strategy_return'].dropna().index.max().date()} "
        f"({len(results['strategy_return'].dropna())} days)"
    )

    results.to_csv(
        "garch_auto_leverage_backtest.csv",
        encoding="utf-8",
    )

    latest_leverage = results["leverage"].dropna().iloc[-1]
    latest_raw_leverage = results["leverage_raw"].dropna().iloc[-1]
    latest_predicted_leverage = results["predicted_leverage"].dropna().iloc[-1]
    full_leverage_history = [
        {
            "date": date.date().isoformat(),
            "leverage": float(value),
            "raw_leverage": float(results["leverage_raw"].get(date, value)),
            "predicted_leverage": float(results["predicted_leverage"].get(date, 0.0)),
        }
        for date, value in results["leverage"].dropna().tail(1260).items()
    ]

    output = {
        "model": "VIX premium direct leverage",
        "data_date": predicted_leverage.dropna().index[-1].date().isoformat(),
        "predicted_leverage": float(latest_predicted_leverage),
        "target_volatility": float(model_artifact["target_volatility"]),
        "target_max_drawdown": float(model_artifact.get("target_max_drawdown", -0.35)),
        "inverse_vol_alpha": float(model_artifact.get("inverse_vol_alpha", 1.0)),
        "ema_span": int(model_artifact["ema_span"]),
        "raw_leverage": float(latest_predicted_leverage),
        "leverage": float(latest_leverage),
        "ema_leverage": float(latest_leverage),
        "performance": {key: float(value) for key, value in performance.items()},
        "active_start": results["strategy_return"]
        .dropna()
        .index.min()
        .date()
        .isoformat(),
        "active_end": results["strategy_return"]
        .dropna()
        .index.max()
        .date()
        .isoformat(),
    }

    with open(
        "leverage.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        "leverage-history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "history": full_leverage_history,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    plt.figure(figsize=(12, 6))
    plt.plot(results["leverage"], label="Leverage")
    plt.title("GARCH Auto-Leverage Strategy Leverage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Leverage")
    plt.legend()
    plt.grid()
    plt.savefig("garch_auto_leverage.png")

    # make return to equity plot
    equity = (1 + results["strategy_return"]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="Equity")
    plt.title("GARCH Auto-Leverage Strategy Equity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.savefig("garch_auto_leverage_performance.png")

    return results, model_artifact, performance


if __name__ == "__main__":
    run()
