import yfinance as yf
import numpy as np


def load_universe(markets):

    u = []

    if "SP500" in markets:
        u += ["AAPL","MSFT","NVDA","AMZN"]

    if "Nasdaq" in markets:
        u += ["META","AMD","AVGO"]

    if "FTSE" in markets:
        u += ["ENI.MI","ISP.MI","UCG.MI"]

    return list(set(u))


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100/(1+rs))


def scan_ticker(ticker, *args):

    try:
        data = yf.download(ticker, period="6mo",
                           progress=False,
                           auto_adjust=True)

        if data.empty:
            return None, None

        close = data["Close"]
        volume = data["Volume"]

        price = float(close.iloc[-1])

        rsi = float(compute_rsi(close).iloc[-1])
        vol_today = int(volume.iloc[-1])
        vol_avg = int(volume.tail(7).mean())

        base = {
            "Ticker": ticker,
            "Nome": ticker,
            "Prezzo": price,
            "Vol_Today": vol_today,
            "Vol_7d_Avg": vol_avg,
            "RSI": round(rsi,2),
        }

        ep = base.copy()
        ep["Stato_Early"] = "EARLY"

        rea = base.copy()
        rea["Stato"] = "HOT"

        return ep, rea

    except Exception:
        return None, None
