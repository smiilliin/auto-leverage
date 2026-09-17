import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

CSV_PATH = "backtest_results.csv"

# =========================
# Load backtest
# =========================

results = pd.read_csv(
    CSV_PATH,
    index_col=0,
    parse_dates=True,
)

qqq_return = results["qqq_return"]
auto_return = results["strategy_return"]

# =========================
# Load QLD
# =========================

print("Loading QLD data...")

qld = yf.download(
    "QLD",
    start=results.index.min().strftime("%Y-%m-%d"),
    end=(results.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
)

qld_close = qld["Close"]

# yfinance 버전에 따라 MultiIndex일 수 있음
if isinstance(qld_close, pd.DataFrame):
    qld_close = qld_close.iloc[:, 0]

qld_return = qld_close.pct_change()

# 같은 날짜만 사용
qld_return = qld_return.reindex(results.index)

# =========================
# Equity curves
# =========================

qqq_equity = (1 + qqq_return.fillna(0)).cumprod()

qld_equity = (1 + qld_return.fillna(0)).cumprod()

auto_equity = (1 + auto_return.fillna(0)).cumprod()

# =========================
# Normalize
# =========================

comparison = pd.DataFrame(
    {
        "QQQ": qqq_equity,
        "QLD": qld_equity,
        "AUTO": auto_equity,
    }
).dropna()

# =========================
# Plot
# =========================

BG = "#f3f0e9"  # 페이지 배경
PAPER = "#faf8f3"  # 그래프 영역
INK = "#24231f"  # 글자
MUTED = "#77736a"  # 보조 글자
LINE = "#d8d3c8"  # 선 / 그리드

plt.figure(
    figsize=(14, 7),
    facecolor=BG,
)

ax = plt.gca()
ax.set_facecolor(PAPER)


ax = plt.gca()
ax.set_facecolor(PAPER)

ax.tick_params(colors=INK)
ax.xaxis.label.set_color(INK)
ax.yaxis.label.set_color(INK)
ax.title.set_color(INK)

for spine in ax.spines.values():
    spine.set_color(LINE)

ax.grid(
    True,
    color=LINE,
    alpha=0.7,
)

plt.plot(
    comparison.index,
    comparison["QQQ"],
    label="QQQ",
)

plt.plot(
    comparison.index,
    comparison["QLD"],
    label="QLD",
)

plt.plot(
    comparison.index,
    comparison["AUTO"],
    label="AUTO",
)

plt.title("QQQ vs QLD vs AUTO")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()

plt.tight_layout()

plt.savefig(
    "visual-qqq-qld-auto.png",
    dpi=150,
    facecolor=BG,
    edgecolor=BG,
)

plt.show()

# =========================
# Final values
# =========================

print()
print("=" * 55)
print("             QQQ vs QLD vs AUTO")
print("=" * 55)

print()
print(
    f"Period : "
    f"{comparison.index.min().date()} ~ "
    f"{comparison.index.max().date()}"
)

print()

for name in ["QQQ", "QLD", "AUTO"]:
    final_value = comparison[name].iloc[-1]
    total_return = final_value - 1

    print(
        f"{name:<5} "
        f"Final : {final_value:.4f}    "
        f"Total Return : {total_return:.2%}"
    )

print()
print("Saved:")
print("  visual-qqq-qld-auto.png")

print("=" * 55)
