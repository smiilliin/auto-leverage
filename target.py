import pandas as pd
import numpy as np


def make_target(data, horizon=21):

    qqq = data["Close"]["QQQ"]

    daily_return = qqq.pct_change()

    future_volatility = daily_return.rolling(horizon).std().shift(-horizon) * np.sqrt(
        252
    )

    return future_volatility.rename("future_volatility")
