import yfinance as yf
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# LOAD UNIVERSE
# -----------------------------------------------------------------------------
def load_universe(markets):

    universe = []

    if "SP500" in markets:
        universe += [
            "AAPL", "MSFT", "NVDA", "AMZN",
            "META", "GOOGL", "TSLA"
        ]

    if "Nasdaq" in markets:
        universe += [
            "AMD", "INTC", "ADBE", "AVGO"
        ]

    if "FTSE" in markets:
        universe += [
            "ENI.MI", "ISP.MI", "UCG.MI", "STM.MI"
        ]

    return sorted(list(set(universe)))


# -----------------------------------------------------------------------------
# RSI
# -----------------------------------------------------------------------------
def compute_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


# -----------------------------------------------------------------------------
# SCAN TICKER
# -----------------------------------------------------------------------------
def scan_ticker(
    ticker,
    e_h,
    p_rmin,
    p_rmax,
    r_poc,
    vol_ratio_hot,
):

    try:
        data = yf.download(
            ticker,
            period="6mo",
            progress=False,
            auto_adjust=True,
        )

        if data.empty or len(data) < 30:
            return None, None

        close = data["Close"]
        volume = data["Volume"]

        price = float(close.iloc[-1])

        # Indicators
        ema20 = close.ewm(span=20).mean().iloc[-1]
        rsi = compute_rsi(close).iloc[-1]

        vol_today = int(volume.iloc[-1])
        vol_avg7 = int(volume.tail(7).mean())

        vol_ratio = vol_today / vol_avg7 if vol_avg7 else 0

        early_score = abs(price - ema20) / ema20
        pro_score = rsi

        base = {
            "Ticker": ticker,
            "Nome": ticker,
            "Prezzo": round(price, 2),
            "MarketCap": 0,
            "Currency": "USD",
            "Vol_Today": vol_today,
            "Vol_7d_Avg": vol_avg7,
            "RSI": round(float(rsi), 2),
            "Vol_Ratio": round(vol_ratio, 2),
            "Early_Score": round(early_score, 4),
            "Pro_Score": round(pro_score, 2),
        }

        # EARLY SIGNAL
        ep = None
        if early_score <= e_h:
            ep = base.copy()
            ep["Stato_Early"] = "EARLY"

        # REA HOT SIGNAL
        rea = None
        if vol_ratio >= vol_ratio_hot:
            rea = base.copy()
            rea["Stato"] = "HOT"

        return ep, rea

    except Exception:
        # sicurezza totale scanner
        return None, None
