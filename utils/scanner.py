import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

def calc_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def load_index_from_csv(filename: str):
    path = Path("data") / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    for col in ["ticker", "Simbolo", "simbolo", "Ticker", "Symbol", "symbol"]:
        if col in df.columns:
            return df[col].dropna().astype(str).unique().tolist()
    return []

def load_universe(markets: list) -> list:
    t = []
    if "SP500" in markets: t += load_index_from_csv("sp500.csv")
    if "Eurostoxx" in markets: t += load_index_from_csv("eurostoxx600.csv")
    if "FTSE" in markets: t += load_index_from_csv("ftsemib.csv")
    if "Nasdaq" in markets: t += load_index_from_csv("nasdaq100.csv")
    if "Dow" in markets: t += load_index_from_csv("dowjones.csv")
    if "Russell" in markets: t += load_index_from_csv("russell2000.csv")
    return list(dict.fromkeys(t))

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot=1.5):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data) < 40: return None, None
        
        c, h, l, v = data["Close"], data["High"], data["Low"], data["Volume"]
        yt = yf.Ticker(ticker)
        info = yt.info
        name = info.get("longName", info.get("shortName", ticker))[:50]
        price = float(c.iloc[-1])
        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        market_cap = info.get("marketCap", np.nan)
        vol_today = float(v.iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())
        currency = info.get("currency", "USD")

        # Logica punteggi
        dist_ema = abs(price - ema20) / ema20
        early_score = 8 if dist_ema < e_h else 0
        
        pro_score = 3 if price > ema20 else 0
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        rsi_val = float(rsi.iloc[-1])
        if p_rmin < rsi_val < p_rmax: pro_score += 3
        
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
        if vol_ratio > 1.2: pro_score += 2
        
        stato_early = "EARLY" if early_score >= 8 else "-"
        stato_pro = "PRO" if pro_score >= 8 else "-"
        
        res_ep = {
            "Nome": name, "Ticker": ticker, "Prezzo": round(price, 2),
            "MarketCap": market_cap, "Vol_Today": int(vol_today), "Vol_7d_Avg": int(vol_7d_avg),
            "Currency": currency, "Early_Score": early_score, "Pro_Score": pro_score,
            "RSI": round(rsi_val, 1), "Vol_Ratio": round(vol_ratio, 2), "Stato_Early": stato_early, "Stato_Pro": stato_pro
        }
        
        # REA-QUANT
        tp = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dist_poc = abs(price - poc) / poc
        rea_score = 7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0
        
        res_rea = {
            "Nome": name, "Ticker": ticker, "Prezzo": round(price, 2), "Stato": "HOT" if rea_score >= 7 else "-"
        } if rea_score >= 7 else None
        
        return res_ep if (stato_early != "-" or stato_pro != "-") else None, res_rea
    except:
        return None, None
