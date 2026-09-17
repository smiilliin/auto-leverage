import numpy as np
import pandas as pd


def volatility_to_leverage(
    predicted_volatility,
    target_volatility=0.25,
    alpha=1.0,
    min_leverage=0.5,
    max_leverage=2.0,
    max_daily_change=0.1,
):
    predicted_volatility = pd.Series(predicted_volatility).copy()

    # 예측 변동성이 너무 작아서 레버리지가 폭발하는 것 방지
    predicted_volatility = predicted_volatility.clip(lower=0.05)

    # Target Volatility 방식
    raw_leverage = (target_volatility / predicted_volatility) ** alpha

    # 레버리지 범위 제한
    target_leverage = raw_leverage.clip(
        min_leverage,
        max_leverage,
    )

    # 하루 레버리지 변화량 제한
    leverage = pd.Series(
        np.nan,
        index=predicted_volatility.index,
    )

    previous_leverage = 1.0

    for i in range(len(target_leverage)):
        target = target_leverage.iloc[i]

        if pd.isna(target):
            leverage.iloc[i] = previous_leverage
            continue

        change = target - previous_leverage

        change = np.clip(
            change,
            -max_daily_change,
            max_daily_change,
        )

        current_leverage = previous_leverage + change

        current_leverage = np.clip(
            current_leverage,
            min_leverage,
            max_leverage,
        )

        leverage.iloc[i] = current_leverage
        previous_leverage = current_leverage

    return leverage


def make_strategy_returns(
    qqq,
    predicted_volatility,
):
    leverage = volatility_to_leverage(
        predicted_volatility,
        target_volatility=0.25,
        alpha=1.0,
        min_leverage=0.5,
        max_leverage=2.0,
        max_daily_change=0.1,
    )

    # EMA smoothing
    leverage = leverage.ewm(
        span=10,
        adjust=False,
    ).mean()

    # 다음 날부터 적용
    leverage = leverage.shift(1)

    qqq_return = qqq.pct_change()

    strategy_return = leverage * qqq_return

    return pd.DataFrame(
        {
            "qqq_return": qqq_return,
            "predicted_volatility": predicted_volatility,
            "leverage": leverage,
            "strategy_return": strategy_return,
        }
    )
