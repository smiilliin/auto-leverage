import yfinance as yf


def load_data():
    tickers = ["SPY", "QQQ", "TQQQ", "SQQQ", "^VIX"]

    data = yf.download(
        tickers,
        start="2010-01-01",
        auto_adjust=True,
        progress=False,
    )

    return data
