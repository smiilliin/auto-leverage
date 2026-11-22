import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

CSV_PATH = "garch_auto_leverage_backtest.csv"
OUTPUT_PATH = "visual-sp500-garch-auto.png"


results = pd.read_csv(
    CSV_PATH,
    index_col=0,
    parse_dates=True,
)

start = results.index.min().strftime("%Y-%m-%d")
end = (results.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

benchmarks = yf.download(
    ["SPY", "SSO"],
    start=start,
    end=end,
    auto_adjust=True,
    progress=False,
)

close = benchmarks["Close"]
if isinstance(close, pd.DataFrame):
    close = close.reindex(columns=["SPY", "SSO"])

benchmark_returns = close.pct_change().reindex(results.index)

auto_returns = results["strategy_return"]
auto_equity = (1 + auto_returns.fillna(0)).cumprod()
spy_equity = (1 + benchmark_returns["SPY"].fillna(0)).cumprod()
sso_equity = (1 + benchmark_returns["SSO"].fillna(0)).cumprod()

comparison = pd.DataFrame(
    {
        "SPY": spy_equity,
        "SSO": sso_equity,
        "GARCH AUTO": auto_equity,
    }
).dropna()

BG = "#f3f0e9"
PAPER = "#faf8f3"
INK = "#24231f"
MUTED = "#77736a"
LINE = "#d8d3c8"
ACCENT = "#68705f"

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(14, 10),
    sharex=True,
    facecolor=BG,
)

for ax in (ax1, ax2):
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.grid(True, color=LINE, alpha=0.7)

ax1.plot(comparison.index, comparison["SPY"], label="SPY", color=MUTED)
ax1.plot(comparison.index, comparison["SSO"], label="SSO", color="#a07c52")
ax1.plot(
    comparison.index,
    comparison["GARCH AUTO"],
    label="GARCH AUTO",
    color=ACCENT,
    linewidth=2,
)
ax1.set_title("S&P 500: SPY vs SSO vs DIRECT VIX PREMIUM AUTO")
ax1.set_ylabel("Growth of $1")
ax1.legend()

leverage = results["leverage"]
ax2.plot(
    leverage.index,
    leverage,
    label="GARCH AUTO leverage",
    color=ACCENT,
    linewidth=1.5,
)
ax2.axhline(
    y=1.0,
    color=MUTED,
    linestyle="--",
    linewidth=1.2,
    alpha=0.9,
    label="1x Reference",
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Leverage")
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.tick_params(axis="y", labelleft=False, left=False, labelright=True, right=True)
ax2.spines["left"].set_visible(False)
ax2.set_ylim(0.4, 1.1)
ax2.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_PATH,
    dpi=150,
    facecolor=BG,
    edgecolor=BG,
)

print(f"Saved visualization: {OUTPUT_PATH}")
