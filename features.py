import pandas as pd
import numpy as np


def make_features(data):
    qqq = data["Close"]["QQQ"]
    spy = data["Close"]["SPY"]

    # =========================
    # 1. Trend
    # =========================

    return_5 = qqq.pct_change(5)
    return_21 = qqq.pct_change(21)
    return_63 = qqq.pct_change(63)

    ma20_distance = qqq / qqq.rolling(20).mean() - 1
    ma200_distance = qqq / qqq.rolling(200).mean() - 1

    # =========================
    # 2. Volatility
    # =========================

    daily_return = qqq.pct_change()

    # 252는 미국 시장 1년 거래일 수
    rv_20 = daily_return.rolling(20).std() * np.sqrt(252)
    rv_60 = daily_return.rolling(60).std() * np.sqrt(252)

    rv_ratio = rv_20 / rv_60

    vix = data["Close"]["^VIX"]

    vix_change = vix.pct_change(5)

    features = pd.DataFrame(
        {
            "return_5": return_5,
            # "return_21": return_21,
            # "return_63": return_63,
            "ma20_distance": ma20_distance,
            # "ma200_distance": ma200_distance,
            "rv_20": rv_20,
            "rv_60": rv_60,
            "rv_ratio": rv_ratio,
            "vix": vix,
            "vix_change": vix_change,
        }
    )

    return features
