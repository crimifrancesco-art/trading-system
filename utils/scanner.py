# scanner.py — Trading Scanner PRO v35
# Multithread + Yahoo batch download

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCAN_ERRORS = []

# -----------------------------
# Indicatori tecnici
# -----------------------------

def compute_indicators(df):

    c = df["Close"]
    v = df["Volume"]

    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    ema200 = c.ewm(span=200).mean()

    delta = c.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    vol_ratio = v.iloc[-1] / v.rolling(20).mean().iloc[-1]

    return {
        "Close": c.iloc[-1],
        "EMA20": ema20.iloc[-1],
        "EMA50": ema50.iloc[-1],
        "EMA200": ema200.iloc[-1],
        "RSI": rsi.iloc[-1],
        "Vol_Ratio": vol_ratio,
    }


# -----------------------------
# Scan singolo ticker
# -----------------------------

def scan_ticker(ticker):

    try:

        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False,
            threads=False
        )

        if df is None or len(df) < 50:
            return None

        ind = compute_indicators(df)

        trend = (
            ind["EMA20"] > ind["EMA50"]
            and ind["EMA50"] > ind["EMA200"]
        )

        momentum = ind["RSI"] > 55

        volume = ind["Vol_Ratio"] > 1.5

        score = (
            (trend * 40)
            + (momentum * 30)
            + (volume * 30)
        )

        return {
            "Ticker": ticker,
            "Close": ind["Close"],
            "RSI": round(ind["RSI"], 2),
            "Vol_Ratio": round(ind["Vol_Ratio"], 2),
            "Score": score
        }

    except Exception as e:

        _SCAN_ERRORS.append((ticker, str(e)))
        return None


# -----------------------------
# Scan universo
# -----------------------------

def scan_universe(tickers, workers=8):

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = {
            executor.submit(scan_ticker, t): t
            for t in tickers
        }

        for future in as_completed(futures):

            res = future.result()

            if res is not None:
                results.append(res)

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    df = df.sort_values("Score", ascending=False)

    return df


def get_scan_errors():
    return _SCAN_ERRORS
