import numpy as np
import pandas as pd


def calculate_performance(
    returns,
    trading_days=252,
):
    """
    일간 수익률을 받아 성과 지표를 계산한다.
    """

    returns = returns.dropna()

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(returns) / trading_days

    cagr = equity.iloc[-1] ** (1 / years) - 1

    volatility = returns.std() * np.sqrt(trading_days)

    sharpe = returns.mean() / returns.std() * np.sqrt(trading_days)

    rolling_max = equity.cummax()

    drawdown = equity / rolling_max - 1

    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }
