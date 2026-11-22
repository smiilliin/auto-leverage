import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_garch_premium_features(
    garch_variance,
    premium,
):
    garch_variance = pd.Series(garch_variance).rename("garch_variance")
    premium = pd.Series(premium).rename("premium")

    features = pd.concat(
        [garch_variance, premium],
        axis=1,
    ).dropna()

    return features


def make_direct_leverage_features(
    garch_variance,
    premium,
    vix,
):
    garch_variance = pd.Series(garch_variance).rename("garch_variance")
    premium = pd.Series(premium).rename("premium")
    vix = pd.Series(vix).rename("vix")
    vix_change = vix.pct_change(5).rename("vix_change")

    features = pd.concat(
        [
            garch_variance,
            premium,
            vix,
            vix_change,
        ],
        axis=1,
    ).dropna()

    return features


def make_direct_leverage_target(
    sp500_returns,
    target_volatility=0.18,
    horizon=21,
    alpha=1.0,
    min_leverage=0.5,
    max_leverage=1.5,
):
    sp500_returns = pd.Series(sp500_returns).astype(float)

    future_volatility = (
        sp500_returns
        .rolling(horizon)
        .std()
        .shift(-horizon)
        .mul(np.sqrt(252))
    )

    target_leverage = (
        target_volatility / future_volatility.clip(lower=0.05)
    ) ** alpha

    return target_leverage.clip(
        min_leverage,
        max_leverage,
    ).rename("target_leverage")


def make_future_variance(
    returns,
    horizon=21,
):
    returns = pd.Series(returns)

    future_variance = returns.pow(2).rolling(horizon).sum().shift(-horizon)

    return future_variance.rename("future_variance")


def create_garch_premium_model(
    alpha=10.0,
):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def garch_variance_from_parameters(
    returns_percent,
    garch_parameters,
):
    returns_percent = pd.Series(returns_percent).astype(float).dropna()

    omega = float(garch_parameters["omega"])
    alpha = float(garch_parameters["alpha"])
    beta = float(garch_parameters["beta"])

    persistence = alpha + beta
    if persistence >= 1:
        raise ValueError("GARCH persistence must be below 1.")

    residual = returns_percent - returns_percent.mean()
    variance = pd.Series(
        np.nan,
        index=returns_percent.index,
        dtype=float,
    )
    variance.iloc[0] = omega / (1 - persistence)

    for i in range(1, len(variance)):
        variance.iloc[i] = (
            omega
            + alpha * residual.iloc[i - 1] ** 2
            + beta * variance.iloc[i - 1]
        )

    return variance


def walk_forward_garch_premium_predict(
    X,
    y,
    train_size=756,
    test_size=63,
    alpha=10.0,
):
    dataset = pd.concat(
        [X, y.rename("y")],
        axis=1,
    ).dropna()

    X = dataset[X.columns]
    y = dataset["y"]

    predictions = []
    start = train_size

    while start < len(X):
        train_end = start
        train_start = train_end - train_size
        test_end = min(start + test_size, len(X))

        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[start:test_end]

        model = create_garch_premium_model(alpha=alpha)
        model.fit(X_train, y_train)

        predictions.append(
            pd.Series(
                model.predict(X_test),
                index=X_test.index,
                name="predicted_future_variance",
            )
        )

        start = test_end

    if not predictions:
        return pd.Series(
            dtype=float,
            name="predicted_future_variance",
        )

    return pd.concat(predictions).sort_index()


def predicted_variance_to_annual_volatility(
    predicted_variance,
    horizon=21,
):
    predicted_variance = pd.Series(predicted_variance).astype(float)

    predicted_daily_variance = predicted_variance.clip(lower=0) / horizon

    return np.sqrt(predicted_daily_variance * 252).div(100)


def variance_to_leverage(
    predicted_variance,
    target_volatility=0.25,
    alpha=1.0,
    min_leverage=0.5,
    max_leverage=2.0,
    max_daily_change=0.1,
    ema_span=10,
):
    predicted_volatility = predicted_variance_to_annual_volatility(predicted_variance)
    predicted_volatility = predicted_volatility.clip(lower=0.05)

    raw_leverage = (target_volatility / predicted_volatility) ** alpha
    raw_leverage = raw_leverage.clip(
        min_leverage,
        max_leverage,
    )

    leverage = pd.Series(
        np.nan,
        index=raw_leverage.index,
        dtype=float,
    )
    previous_leverage = 1.0
    prediction_started = False

    for i in range(len(raw_leverage)):
        target = raw_leverage.iloc[i]

        if pd.isna(target):
            if not prediction_started:
                continue

            leverage.iloc[i] = previous_leverage
            continue

        prediction_started = True
        change = target - previous_leverage
        change = np.clip(
            change,
            -max_daily_change,
            max_daily_change,
        )

        current_leverage = np.clip(
            previous_leverage + change,
            min_leverage,
            max_leverage,
        )
        leverage.iloc[i] = current_leverage
        previous_leverage = current_leverage

    leverage = leverage.ewm(
        span=ema_span,
        adjust=False,
    ).mean()

    return leverage


def make_garch_auto_leverage_results(
    sp500_returns,
    predicted_variance,
    target_volatility=0.25,
    alpha=1.0,
    min_leverage=0.5,
    max_leverage=2.0,
    max_daily_change=0.1,
    ema_span=10,
):
    sp500_returns = pd.Series(sp500_returns).astype(float)

    leverage = variance_to_leverage(
        predicted_variance.reindex(sp500_returns.index),
        target_volatility=target_volatility,
        alpha=alpha,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        max_daily_change=max_daily_change,
        ema_span=ema_span,
    )

    applied_leverage = leverage.shift(1)
    strategy_return = applied_leverage * sp500_returns

    return pd.DataFrame(
        {
            "sp500_return": sp500_returns,
            "predicted_variance": predicted_variance.reindex(sp500_returns.index),
            "leverage_raw": leverage,
            "leverage": applied_leverage,
            "strategy_return": strategy_return,
        }
    )


def make_direct_leverage_results(
    sp500_returns,
    predicted_leverage,
    min_leverage=0.5,
    max_leverage=1.5,
    max_daily_change=0.1,
    ema_span=7,
):
    sp500_returns = pd.Series(sp500_returns).astype(float)
    predicted_leverage = pd.Series(predicted_leverage).astype(float)

    raw_leverage = predicted_leverage.reindex(
        sp500_returns.index
    ).clip(
        min_leverage,
        max_leverage,
    )

    constrained_leverage = pd.Series(
        np.nan,
        index=raw_leverage.index,
        dtype=float,
    )
    previous_leverage = 1.0

    for i in range(len(raw_leverage)):
        target = raw_leverage.iloc[i]

        if pd.isna(target):
            continue

        change = np.clip(
            target - previous_leverage,
            -max_daily_change,
            max_daily_change,
        )
        current_leverage = np.clip(
            previous_leverage + change,
            min_leverage,
            max_leverage,
        )
        constrained_leverage.iloc[i] = current_leverage
        previous_leverage = current_leverage

    smoothed_leverage = constrained_leverage.ewm(
        span=ema_span,
        adjust=False,
    ).mean()

    applied_leverage = smoothed_leverage.shift(1)
    strategy_return = applied_leverage * sp500_returns

    return pd.DataFrame(
        {
            "sp500_return": sp500_returns,
            "predicted_leverage": raw_leverage,
            "leverage_raw": smoothed_leverage,
            "leverage": applied_leverage,
            "strategy_return": strategy_return,
        }
    )


def calculate_auto_leverage_performance(
    strategy_return,
    trading_days=252,
):
    strategy_return = pd.Series(strategy_return).dropna()

    if len(strategy_return) < 2:
        return {
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
        }

    equity = (1 + strategy_return).cumprod()
    years = len(strategy_return) / trading_days

    annual_volatility = strategy_return.std(ddof=1) * np.sqrt(trading_days)
    sharpe = (
        strategy_return.mean() / strategy_return.std(ddof=1) * np.sqrt(trading_days)
    )
    cagr = equity.iloc[-1] ** (1 / years) - 1
    drawdown = equity / equity.cummax() - 1

    return {
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": drawdown.min(),
    }


def optimize_garch_leverage_parameters(
    sp500_returns,
    predicted_variance,
    target_volatility=0.25,
    target_sharpe=None,
    target_max_drawdown=-0.25,
    min_leverage=0.5,
    max_leverage=2.0,
    alpha_bounds=(0.5, 2.0),
    ema_span_bounds=(5, 30),
    max_daily_change=0.1,
    seed=42,
):
    def objective(parameters):
        alpha, ema_span = parameters
        ema_span = int(round(ema_span))

        results = make_garch_auto_leverage_results(
            sp500_returns,
            predicted_variance,
            target_volatility=target_volatility,
            alpha=alpha,
            min_leverage=min_leverage,
            max_leverage=max_leverage,
            max_daily_change=max_daily_change,
            ema_span=ema_span,
        )
        performance = calculate_auto_leverage_performance(results["strategy_return"])

        volatility_error = (
            performance["annual_volatility"] - target_volatility
        ) / target_volatility

        objective_value = volatility_error**2

        # if target_sharpe is not None, we consider the sharpe ratio as part of the objective function
        if target_sharpe is not None:
            sharpe_error = (performance["sharpe"] - target_sharpe) / max(
                abs(target_sharpe), 1.0
            )
            objective_value += 0.5 * sharpe_error**2
        elif np.isfinite(performance["sharpe"]):
            objective_value -= 0.15 * performance["sharpe"]

        if np.isfinite(performance["max_drawdown"]):
            mdd_limit = abs(target_max_drawdown)
            mdd_excess = max(
                0.0,
                abs(performance["max_drawdown"]) - mdd_limit,
            ) / mdd_limit
            objective_value += 3.0 * mdd_excess + 8.0 * mdd_excess**2

        if not np.isfinite(objective_value):
            return 1e6

        return objective_value

    bounds = [
        alpha_bounds,
        ema_span_bounds,
    ]
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        polish=True,
    )

    alpha = float(result.x[0])
    ema_span = int(round(result.x[1]))

    return {
        "alpha": alpha,
        "ema_span": ema_span,
        "target_volatility": target_volatility,
        "target_sharpe": target_sharpe,
        "target_max_drawdown": target_max_drawdown,
        "objective": float(result.fun),
    }
