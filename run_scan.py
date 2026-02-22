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
        # Aggiungo auto_adjust=True per coerenza e info per il nome
        yt = yf.Ticker(ticker)
        df = yt.history(period="6mo")
        
        if df.empty or len(df) < 50:
            return None
            
        info = yt.info
        name = info.get("longName", ticker)
        market_cap = info.get("marketCap", 0)
        currency = info.get("currency", "USD")
        
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        df["EMA50"] = ema(close, 50)
        df["RSI"] = rsi(close)
        df["MACD"], df["MACD_SIGNAL"] = macd(close)
        df["ATR"] = atr(df)

        # 1. Trend: EMA50 sopra il prezzo di 5 giorni fa
        trend = bool(df["EMA50"].iloc[-1] > close.iloc[-5])

        # 2. RSI Momentum
        rsi_val = float(df["RSI"].iloc[-1])
        rsi_prev = float(df["RSI"].iloc[-2])
        rsi_sig = bool(rsi_val > rsi_prev and rsi_val < 70)

        # 3. MACD Cross
        macd_ok = bool(df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1])

        # 4. Volume Confirm
        vol_mean = float(volume.rolling(20).mean().iloc[-1])
        vol_7d_avg = float(volume.tail(7).mean())
        volume_ok = bool(float(volume.iloc[-1]) > vol_mean)

        # 5. Volatility (ATR)
        atr_pct = float(df["ATR"].iloc[-1]) / float(close.iloc[-1])
        volatility_ok = bool(0.005 < atr_pct < 0.10)
        
        # OBV Trend simplified
        obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
        obv_trend = "UP" if obv.iloc[-1] > obv.iloc[-5] else "DOWN"

        checks = [trend, rsi_sig, macd_ok, volume_ok, volatility_ok]
        score = sum(checks)

        signal = "NONE"
        if score >= 3:
            signal = "BUY"
        if score >= 5:
            signal = "STRONG BUY"

        return {
            "name": name,
            "ticker": ticker,
            "signal": signal,
            "score": int(score),
            "price": round(float(close.iloc[-1]), 2),
            "market_cap": market_cap,
            "vol_today": int(volume.iloc[-1]),
            "vol_7d_avg": int(vol_7d_avg),
            "rsi": round(rsi_val, 1),
            "vol_ratio": round(float(volume.iloc[-1] / vol_mean), 2),
            "obv_trend": obv_trend,
            "atr": round(float(df["ATR"].iloc[-1]), 2),
            "atr_exp": bool(atr_pct > 0.03), # Example expansion threshold
            "currency": currency,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Errore su {ticker}: {e}")
        return None

# --------------------------------------------------
# RUN SCAN
# --------------------------------------------------
def run_scan():
    runtime = Path("data/runtime_universe.json")
    if runtime.exists():
        try:
            tickers_to_scan = json.loads(runtime.read_text())["tickers"]
        except:
            tickers_to_scan = ["AAPL", "MSFT"]
    else:
        tickers_to_scan = ["AAPL", "MSFT"]

    results = []
    for t in tickers_to_scan:
        r = analyze_ticker(t)
        if r:
            results.append(r)

    try:
        RESULT_PATH.write_text(json.dumps(results, indent=2))
        print(f"Scan completato. Risultati: {len(results)}")
    except Exception as e:
        print(f"Errore salvataggio: {e}")

if __name__ == "__main__":
    run_scan()
