# ==================================================
# RUN_SCAN.PY — SCANNER ENGINE V11
# ==================================================
import yfinance as yf
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

RESULT_PATH = Path("data/scan_results.json")
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --------------------------------------------------
# ANALYZE TICKER
# --------------------------------------------------
def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        df["EMA50"] = ema(close, 50)
        df["RSI"] = rsi(close)
        df["MACD"], df["MACD_SIGNAL"] = macd(close)
        df["ATR"] = atr(df)

        # 1. Trend: EMA50 sopra il prezzo di 5 giorni fa
        trend = bool(df["EMA50"].iloc[-1] > close.iloc[-5])

        # 2. RSI: valore attuale > valore precedente (momentum in ripresa)
        rsi_val = float(df["RSI"].iloc[-1])
        rsi_prev = float(df["RSI"].iloc[-2])
        rsi_sig = bool(rsi_val > rsi_prev and rsi_val < 70)

        # 3. MACD: linea sopra signal
        macd_ok = bool(df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1])

        # 4. Volume: volume odierno sopra media 20 giorni
        vol_mean = float(volume.rolling(20).mean().iloc[-1])
        volume_ok = bool(float(volume.iloc[-1]) > vol_mean)

        # 5. Volatilita ATR: contenuta tra 0.5% e 10%
        atr_pct = float(df["ATR"].iloc[-1]) / float(close.iloc[-1])
        volatility_ok = bool(0.005 < atr_pct < 0.10)

        checks = [trend, rsi_sig, macd_ok, volume_ok, volatility_ok]
        score = sum(checks)

        signal = "NONE"
        if score >= 3:
            signal = "BUY"
        if score >= 5:
            signal = "STRONG BUY"

        return {
            "ticker": ticker,
            "signal": signal,
            "score": int(score),
            "price": round(float(close.iloc[-1]), 2),
            "rsi": round(rsi_val, 1),
            "ema50": round(float(df["EMA50"].iloc[-1]), 2),
            "macd_ok": macd_ok,
            "vol_ok": volume_ok,
            "trend_ok": trend,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Errore analyze_ticker {ticker}: {e}")
        return None

# --------------------------------------------------
# RUN SCAN
# --------------------------------------------------
def run_scan():
    # Leggi tickers dal runtime universe (aggiornato dalla dashboard)
    runtime = Path("data/runtime_universe.json")
    if runtime.exists():
        try:
            tickers_to_scan = json.loads(runtime.read_text())["tickers"]
        except Exception:
            tickers_to_scan = ["AAPL", "MSFT", "NVDA"]
    else:
        tickers_to_scan = ["AAPL", "MSFT", "NVDA"]

    if not tickers_to_scan:
        tickers_to_scan = ["AAPL", "MSFT", "NVDA"]

    results = []
    for t in tickers_to_scan:
        try:
            r = analyze_ticker(t)
            if r:
                results.append(r)
        except Exception as e:
            print(f"Errore su ticker {t}: {e}")

    try:
        RESULT_PATH.write_text(json.dumps(results, indent=2))
        print(f"Scan completato. Risultati: {len(results)}")
    except Exception as e:
        print(f"Errore salvataggio: {e}")

# --------------------------------------------------
if __name__ == "__main__":
    run_scan()
