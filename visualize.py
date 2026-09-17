import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================================
# Settings
# ========================================

CSV_PATH = "backtest_results.csv"
HORIZON = 21


# ========================================
# Load data
# ========================================

results = pd.read_csv(
    CSV_PATH,
    index_col=0,
    parse_dates=True,
)


# ========================================
# Basic calculations
# ========================================

qqq_return = results["qqq_return"]

auto_return = results["strategy_return"]

leverage = results["leverage"]

predicted_volatility = results["predicted_volatility"]


# ========================================
# Equity curves
# ========================================

qqq_equity = (1 + qqq_return.fillna(0)).cumprod()

auto_equity = (1 + auto_return.fillna(0)).cumprod()


# ========================================
# 1. Equity Curve
# ========================================

plt.figure(figsize=(14, 7))

plt.plot(
    qqq_equity.index,
    qqq_equity,
    label="QQQ",
)

plt.plot(
    auto_equity.index,
    auto_equity,
    label="AUTO",
)

plt.title("QQQ vs AUTO Leverage")

plt.xlabel("Date")
plt.ylabel("Growth of $1")

plt.legend()

plt.tight_layout()

plt.savefig(
    "visual-equity.png",
    dpi=150,
)

plt.show()


# ========================================
# 2. Leverage & Predicted Volatility
# ========================================

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(
    leverage.index,
    leverage,
    label="AUTO Leverage",
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Leverage")

ax1.set_ylim(
    0,
    2.1,
)

ax2 = ax1.twinx()

ax2.plot(
    predicted_volatility.index,
    predicted_volatility,
    label="Predicted Volatility",
)

ax2.set_ylabel("Predicted 21D Volatility")

ax1.set_title("AUTO Leverage vs Predicted Volatility")

ax1.legend(loc="upper left")

ax2.legend(loc="upper right")

fig.tight_layout()

plt.savefig(
    "visual-leverage-volatility.png",
    dpi=150,
)

plt.show()


# ========================================
# 3. Leverage vs Future Volatility
# ========================================

daily_return = qqq_return

future_volatility = daily_return.shift(-1).rolling(HORIZON).std() * np.sqrt(252)


validation = pd.DataFrame(
    {
        "leverage": leverage,
        "future_volatility": future_volatility,
    }
).dropna()


plt.figure(figsize=(10, 7))

plt.scatter(
    validation["leverage"],
    validation["future_volatility"],
    alpha=0.25,
    s=12,
)

plt.xlabel("AUTO Leverage")
plt.ylabel("Actual Future 21D Volatility")

plt.title("AUTO Leverage vs Future Volatility")

plt.tight_layout()

plt.savefig(
    "visual-leverage-vs-future-volatility.png",
    dpi=150,
)

plt.show()


# ========================================
# Summary
# ========================================

correlation = validation[["leverage", "future_volatility"]].corr().iloc[0, 1]


print()
print("=" * 50)
print("AUTO-LEVERAGE VISUALIZATION")
print("=" * 50)

print(
    f"Data range : "
    f"{results.index.min().date()} "
    f"~ "
    f"{results.index.max().date()}"
)

print(f"Observations : " f"{len(results)}")

print(f"Leverage / Future Vol correlation : " f"{correlation:.4f}")

print()
print("Saved:")
print("  visual-equity.png")
print("  visual-leverage-volatility.png")
print("  visual-leverage-vs-future-volatility.png")
print("=" * 50)
