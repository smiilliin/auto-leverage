import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_data
from features import make_features
from target import make_target
from model import create_model

from strategy import make_strategy_returns
from backtest import calculate_performance
from walk_forward import walk_forward_predict

EMA_SPAN = 10


def make_fixed_strategy(qqq, leverage):
    """
    고정 레버리지 전략
    """
    qqq_return = qqq.pct_change()

    strategy_return = leverage * qqq_return

    return pd.DataFrame(
        {
            "qqq_return": qqq_return,
            "leverage": leverage,
            "strategy_return": strategy_return,
        }
    )


def apply_ema(results, span=10):
    """
    레버리지에 EMA smoothing 적용
    """
    results = results.copy()

    results["leverage_raw"] = results["leverage"]

    results["leverage"] = results["leverage_raw"].ewm(span=span, adjust=False).mean()

    results["strategy_return"] = results["leverage"] * results["qqq_return"]

    return results


def print_performance(name, returns):
    """
    전략 성과 출력
    """
    performance = calculate_performance(returns)

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    for key, value in performance.items():
        print(f"{key:<15}: {value:.4f}")


def print_leverage_statistics(name, results):
    """
    레버리지 통계 출력
    """
    leverage = results["leverage"].dropna()

    print()
    print(name)

    print(f"Mean:   {leverage.mean():.4f}")
    print(f"Median: {leverage.median():.4f}")
    print(f"Min:    {leverage.min():.4f}")
    print(f"Max:    {leverage.max():.4f}")
    print(f"Std:    {leverage.std():.4f}")
    print(f"Change: {leverage.diff().abs().mean():.4f}")


def validate_leverage_vs_future_vol(
    results,
    data,
    horizon=21,
):
    """
    AUTO 레버리지와 실제 미래 변동성의 관계를 검증한다.
    """

    qqq = data["Close"]["QQQ"]

    # 실제 미래 21일 변동성
    daily_return = qqq.pct_change()

    future_volatility = daily_return.shift(-1).rolling(horizon).std() * np.sqrt(252)

    validation = pd.DataFrame(
        {
            "leverage": results["leverage"],
            "predicted_volatility": results["predicted_volatility"],
            "future_volatility": future_volatility,
        }
    ).dropna()

    # --------------------------------
    # 1. 상관관계
    # --------------------------------

    correlation = validation[["leverage", "future_volatility"]].corr().iloc[0, 1]

    prediction_correlation = (
        validation[["predicted_volatility", "future_volatility"]].corr().iloc[0, 1]
    )

    print()
    print("=" * 50)
    print("LEVERAGE VALIDATION")
    print("=" * 50)

    print(f"Leverage vs Future Vol : " f"{correlation:.4f}")

    print(f"Prediction vs Future Vol : " f"{prediction_correlation:.4f}")

    # --------------------------------
    # 2. 레버리지 구간별 실제 미래 변동성
    # --------------------------------

    bins = [
        0.5,
        0.8,
        1.1,
        1.4,
        1.7,
        2.01,
    ]

    labels = [
        "0.5~0.8x",
        "0.8~1.1x",
        "1.1~1.4x",
        "1.4~1.7x",
        "1.7~2.0x",
    ]

    validation["leverage_bucket"] = pd.cut(
        validation["leverage"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    grouped = validation.groupby(
        "leverage_bucket",
        observed=True,
    ).agg(
        samples=("future_volatility", "count"),
        mean_future_vol=(
            "future_volatility",
            "mean",
        ),
        median_future_vol=(
            "future_volatility",
            "median",
        ),
        mean_predicted_vol=(
            "predicted_volatility",
            "mean",
        ),
        mean_leverage=(
            "leverage",
            "mean",
        ),
    )

    print()
    print("Leverage Bucket Analysis")
    print()

    print(grouped.to_string(float_format=lambda x: f"{x:.4f}"))

    # --------------------------------
    # 3. 분위수 기반 분석
    # --------------------------------

    validation["leverage_quartile"] = pd.qcut(
        validation["leverage"],
        q=4,
        labels=[
            "Q1 Low",
            "Q2",
            "Q3",
            "Q4 High",
        ],
        duplicates="drop",
    )

    quartile = validation.groupby(
        "leverage_quartile",
        observed=True,
    ).agg(
        samples=("future_volatility", "count"),
        mean_leverage=(
            "leverage",
            "mean",
        ),
        mean_future_vol=(
            "future_volatility",
            "mean",
        ),
        mean_predicted_vol=(
            "predicted_volatility",
            "mean",
        ),
    )

    print()
    print("Leverage Quartile Analysis")
    print()

    print(quartile.to_string(float_format=lambda x: f"{x:.4f}"))

    return validation, grouped, quartile


def validate_volatility_calibration(
    results,
    data,
    horizon=21,
    bins=10,
):
    """
    예측 변동성과 실제 미래 변동성의 calibration을 검증한다.
    """

    qqq = data["Close"]["QQQ"]

    daily_return = qqq.pct_change()

    future_volatility = daily_return.shift(-1).rolling(horizon).std() * np.sqrt(252)

    validation = pd.DataFrame(
        {
            "predicted_volatility": results["predicted_volatility"],
            "future_volatility": future_volatility,
        }
    ).dropna()

    # 예측 변동성 기준 분위수 구간
    validation["volatility_bucket"] = pd.qcut(
        validation["predicted_volatility"],
        q=bins,
        labels=False,
        duplicates="drop",
    )

    calibration = validation.groupby("volatility_bucket").agg(
        samples=("future_volatility", "count"),
        mean_predicted_vol=(
            "predicted_volatility",
            "mean",
        ),
        mean_actual_vol=(
            "future_volatility",
            "mean",
        ),
        median_actual_vol=(
            "future_volatility",
            "median",
        ),
    )

    calibration["error"] = (
        calibration["mean_actual_vol"] - calibration["mean_predicted_vol"]
    )

    calibration["ratio"] = (
        calibration["mean_actual_vol"] / calibration["mean_predicted_vol"]
    )

    print()
    print("=" * 50)
    print("VOLATILITY CALIBRATION")
    print("=" * 50)

    print()
    print(calibration.to_string(float_format=lambda x: f"{x:.4f}"))

    # 전체 bias
    mean_prediction = validation["predicted_volatility"].mean()

    mean_actual = validation["future_volatility"].mean()

    bias = mean_actual - mean_prediction
    ratio = mean_actual / mean_prediction

    print()
    print(f"Mean predicted vol : {mean_prediction:.4f}")
    print(f"Mean actual vol    : {mean_actual:.4f}")
    print(f"Bias               : {bias:.4f}")
    print(f"Actual / Predicted : {ratio:.4f}")

    return validation, calibration


def main():

    # =========================
    # 1. Data
    # =========================

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

    # =========================
    # 2. Features
    # =========================

    volatility_features = [
        "rv_20",
        "rv_60",
        "rv_ratio",
        "vix",
        "vix_change",
    ]

    X = dataset[volatility_features]

    y = dataset["future_volatility"]

    # =========================
    # 3. Walk Forward Prediction
    # =========================

    prediction = walk_forward_predict(
        X,
        y,
        train_size=756,
        test_size=63,
    )

    prediction.name = "predicted_volatility"

    # =========================
    # 4. Strategy
    # =========================

    qqq = data["Close"]["QQQ"].reindex(prediction.index)

    # -------------------------
    # Raw Dynamic Strategy
    # -------------------------

    results_raw = make_strategy_returns(
        qqq,
        prediction,
    )

    # -------------------------
    # EMA Smoothed Strategy
    # -------------------------

    results = apply_ema(
        results_raw,
        span=EMA_SPAN,
    )

    # =========================
    # 5. Fixed Strategies
    # =========================

    results_117 = make_fixed_strategy(
        qqq,
        1.17,
    )

    results_133 = make_fixed_strategy(
        qqq,
        1.33,
    )

    results_150 = make_fixed_strategy(
        qqq,
        1.50,
    )

    # =========================
    # 6. Equity Curves
    # =========================

    qqq_equity = (1 + qqq.pct_change().fillna(0)).cumprod()

    auto_equity = (1 + results["strategy_return"].fillna(0)).cumprod()

    fixed_117_equity = (1 + results_117["strategy_return"].fillna(0)).cumprod()

    fixed_133_equity = (1 + results_133["strategy_return"].fillna(0)).cumprod()

    fixed_150_equity = (1 + results_150["strategy_return"].fillna(0)).cumprod()

    # =========================
    # 7. Plot
    # =========================

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(
        qqq_equity.index,
        qqq_equity,
        label="QQQ",
    )

    ax1.plot(
        auto_equity.index,
        auto_equity,
        label="AUTO",
    )

    ax1.plot(
        fixed_117_equity.index,
        fixed_117_equity,
        label="Fixed 1.17x",
    )

    ax1.plot(
        fixed_133_equity.index,
        fixed_133_equity,
        label="Fixed 1.33x",
    )

    ax1.plot(
        fixed_150_equity.index,
        fixed_150_equity,
        label="Fixed 1.50x",
    )

    ax1.set_ylabel("Growth of $1")

    ax1.set_title(f"QQQ vs Dynamic and Fixed Leverage " f"(EMA {EMA_SPAN})")

    ax1.legend(loc="upper left")

    # =========================
    # 8. Dynamic Leverage
    # =========================

    ax2 = ax1.twinx()

    ax2.plot(
        results.index,
        results["leverage"],
        label="AUTO Leverage",
    )

    ax2.set_ylabel("Leverage")

    ax2.set_ylim(
        0,
        2.1,
    )

    ax2.legend(loc="upper right")

    fig.tight_layout()

    plt.savefig(
        "strategy-comparison-ema.png",
        dpi=150,
    )

    plt.show()

    # =========================
    # 9. Momentum vs Leverage
    # =========================

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(
        results.index,
        results["qqq_return"].rolling(21).mean(),
        label="QQQ 21D Return",
    )

    ax1.set_ylabel("QQQ Return")

    ax2 = ax1.twinx()

    ax2.plot(
        results.index,
        results["leverage"],
        label="AUTO Leverage",
    )

    ax2.set_ylabel("Leverage")

    ax2.set_ylim(
        0,
        2.1,
    )

    ax1.set_title(f"QQQ Momentum vs AUTO Leverage " f"(EMA {EMA_SPAN})")

    ax1.legend(loc="upper left")

    ax2.legend(loc="upper right")

    fig.tight_layout()

    plt.savefig(
        "strategy-comparison-momentum-ema.png",
        dpi=150,
    )

    plt.show()

    # =========================
    # 10. Performance
    # =========================

    print_performance(
        "QQQ",
        qqq.pct_change().dropna(),
    )

    print_performance(
        "AUTO",
        results["strategy_return"].dropna(),
    )

    print_performance(
        "FIXED 1.17x",
        results_117["strategy_return"].dropna(),
    )

    print_performance(
        "FIXED 1.33x",
        results_133["strategy_return"].dropna(),
    )

    print_performance(
        "FIXED 1.50x",
        results_150["strategy_return"].dropna(),
    )

    # =========================
    # 11. Leverage Statistics
    # =========================

    print()
    print("=" * 50)
    print(f"DYNAMIC LEVERAGE (EMA {EMA_SPAN})")
    print("=" * 50)

    print_leverage_statistics(
        "AUTO",
        results,
    )
    # =========================
    # 12. Validation
    # =========================

    validate_leverage_vs_future_vol(
        results,
        data,
        horizon=21,
    )
    # =========================
    # 13. Volatility Calibration
    # =========================

    validate_volatility_calibration(
        results,
        data,
        horizon=21,
        bins=10,
    )


if __name__ == "__main__":
    main()
