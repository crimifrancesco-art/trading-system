import numpy as np
import pandas as pd
import yfinance as yf
import traceback
from pathlib import Path


def calc_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def load_index_from_csv(filename: str):
    """Carica ticker da CSV nella cartella data/."""
    path = Path("data") / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    for col in ["ticker", "Simbolo", "simbolo", "Ticker", "Symbol", "symbol"]:
        if col in df.columns:
            return df[col].dropna().astype(str).unique().tolist()
    return []


def load_universe(markets: list) -> list:
    """Costruisce la lista di ticker dai mercati selezionati."""
    t = []
    if "SP500" in markets:
        t += load_index_from_csv("sp500.csv")
    if "Eurostoxx" in markets:
        t += load_index_from_csv("eurostoxx600.csv")
    if "FTSE" in markets:
        t += load_index_from_csv("ftsemib.csv")
    if "Nasdaq" in markets:
        t += load_index_from_csv("nasdaq100.csv")
    if "Dow" in markets:
        t += load_index_from_csv("dowjones.csv")
    if "Russell" in markets:
        t += load_index_from_csv("russell2000.csv")
    if "StoxxEmerging" in markets:
        t += load_index_from_csv("stoxx emerging market 50.csv")
    if "USSmallCap" in markets:
        t += load_index_from_csv("us small cap 2000.csv")
    return list(dict.fromkeys(t))


def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot=1.5):
    """Analizza un singolo ticker e restituisce (res_ep, res_rea) o (None, None)."""
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data) < 40:
            return None, None

        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]

        yt = yf.Ticker(ticker)
        info = yt.info
        name = info.get("longName", info.get("shortName", ticker))[:50]
        price = float(c.iloc[-1])
        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        market_cap = info.get("marketCap", np.nan)
        vol_today = float(v.iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())
        currency = info.get("currency", "USD")

        # EARLY
        dist_ema = abs(price - ema20) / ema20
        early_score = 8 if dist_ema < e_h else 0

        # PRO
        pro_score = 3 if price > ema20 else 0
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14
