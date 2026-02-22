# ==================================================
# RUN_SCAN.PY — SCANNER ENGINE V11
# ==================================================

import yfinance as yf
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# LOAD RUNTIME UNIVERSE (tickers scelti sidebar)
# --------------------------------------------------

runtime = Path("data/runtime_universe.json")

if runtime.exists():
    TICKERS = json.loads(runtime.read_text())["tickers"]
else:
    TICKERS = ["AAPL", "MSFT"]

RESULT_PATH = Path("data/scan_results.json")
RESULT_PATH.parent.mkdir(exist_ok=True)

# --------------------------------------------------
# INDICATORS
# --------------------------------------------------

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line, signal

def atr(df, period=14):
    tr = pd.concat([
        df.High - df.Low,
        abs(df.High - df.Close.shift()),
        abs(df.Low - df.Close.shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --------------------------------------------------
# FILTERS (AFFIDABILITÀ ↑)
# --------------------------------------------------

def analyze_ticker(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)

    if df.empty or len(df) < 200:
        return None

    df["EMA50"] = ema(df.Close, 50)
    df["EMA200"] = ema(df.Close, 200)
    df["RSI"] = rsi(df.Close)
    df["MACD"], df["MACD_SIGNAL"] = macd(df.Close)
    df["ATR"] = atr(df)

    trend = df.EMA50.iloc[-1] > df.EMA200.iloc[-1]

    rsi_sig = (
        df.RSI.iloc[-2] < 35
        and df.RSI.iloc[-1] > df.RSI.iloc[-2]
    )

    macd_ok = df.MACD.iloc[-1] > df.MACD_SIGNAL.iloc[-1]

    volume_ok = (
        df.Volume.iloc[-1] >
        df.Volume.rolling(20).mean().iloc[-1]
    )

    atr_pct = df.ATR.iloc[-1] / df.Close.iloc[-1]
    volatility_ok = 0.01 < atr_pct < 0.08

    checks = [
        trend,
        rsi_sig,
        macd_ok,
        volume_ok,
        volatility_ok
    ]

    score = sum(checks)

    signal = "NONE"
    if score >= 4:
        signal = "BUY"
    if score >= 5:
        signal = "STRONG BUY"

    return {
        "ticker": ticker,
        "signal": signal,
        "score": int(score),
        "price": float(df.Close.iloc[-1]),
        "timestamp": datetime.utcnow().isoformat()
    }

# --------------------------------------------------
# RUN SCAN
# --------------------------------------------------

def run_scan():
    results = []

    # DEBUG: prova con pochi ticker noti
    tickers_to_scan = TICKERS or ["AAPL", "MSFT"]

    for t in tickers_to_scan:
        try:
            r = analyze_ticker(t)
            if r:
                results.append(r)
        except Exception as e:
            print("Errore su ticker", t, e)

    try:
        RESULT_PATH.write_text(json.dumps(results, indent=2))
    except Exception as e:
        print("Errore salvataggio risultati:", e)

    print("Scan completed, risultati:", len(results))

# --------------------------------------------------

if __name__ == "__main__":
    run_scan()
