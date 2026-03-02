import threading
import concurrent.futures
import time
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

_SCAN_ERRORS: list = []

def calc_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    return tr.rolling(period).mean()

def calc_bollinger(close, period=20, std_dev=2):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ma + std_dev * std, ma, ma - std_dev * std

def calc_keltner(close, high, low, period=20, atr_mult=1.5):
    ema = close.ewm(span=period).mean()
    atr = calc_atr(high, low, close, period)
    return ema + atr_mult * atr, ema, ema - atr_mult * atr

def detect_squeeze(close, high, low):
    bb_up, _, bb_dn = calc_bollinger(close)
    kc_up, _, kc_dn = calc_keltner(close, high, low)
    return bool(bb_up.iloc[-1] < kc_up.iloc[-1] and bb_dn.iloc[-1] > kc_dn.iloc[-1])

def detect_rsi_divergence(close, rsi_series, lookback=20):
    c = close.tail(lookback); r = rsi_series.tail(lookback)
    if len(c) < lookback: return None
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.9: return "BEARISH"
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.1: return "BULLISH"
    return None

def calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    s = 0
    if vol_ratio > 1.5: s += 2
    if obv_trend == "UP": s += 2
    if atr_expansion: s += 1
    if 45 <= rsi_val <= 65: s += 3
    if price > ema20: s += 2
    if price > ema50: s += 2
    return s

def load_index_from_csv(filename: str):
    for base in [Path("data"), Path(".")]:
        path = base / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in ["Simbolo", "simbolo", "ticker", "Ticker", "Symbol"]:
                    if col in df.columns: return df[col].dropna().astype(str).str.strip().unique().tolist()
            except Exception: pass
    return []

def load_universe(markets: list) -> list:
    tickers = []
    if "SP500" in markets: tickers += load_index_from_csv("sp500.csv")
    if "Nasdaq" in markets: tickers += load_index_from_csv("nasdaq100.csv")
    if "FTSE" in markets: tickers += [t if t.endswith(".MI") else t+".MI" for t in load_index_from_csv("ftsemib.csv")]
    if "Eurostoxx" in markets: tickers += load_index_from_csv("eurostoxx600.csv")
    return list(dict.fromkeys(tickers))

def scan_ticker(ticker: str, e_h: float, p_rmin: int, p_rmax: int, r_poc: float, vol_ratio_hot: float = 1.5):
    try:
        data = None
        for _ in range(2):
            data = yf.Ticker(ticker).history(period="1y", timeout=20)
            if data is not None and len(data) >= 20: break
            time.sleep(1)
        if data is None or len(data) < 20: return None, None
        
        c, h, l, v = data["Close"], data["High"], data["Low"], data["Volume"]
        price = float(c.iloc[-1])
        if price <= 0: return None, None
        
        vt = float(v.iloc[-1]); av20 = float(v.rolling(20).mean().iloc[-1]); v7 = float(v.tail(7).mean())
        vr = float(vt / av20) if av20 > 0 else 0.0
        e20s = c.ewm(span=20).mean(); e50s = c.ewm(span=50).mean()
        e20, e50 = float(e20s.iloc[-1]), float(e50s.iloc[-1])
        sma200 = c.rolling(200).mean().iloc[-1]
        e200 = float(sma200) if not np.isnan(sma200) else None
        
        rsi_s = calc_rsi(c); rsi_v = float(rsi_s.iloc[-1])
        obv = calc_obv(c, v); obv_t = "UP" if obv.diff().rolling(5).mean().iloc[-1] > 0 else "DOWN"
        atr_s = calc_atr(h, l, c); atr_v = float(atr_s.iloc[-1]); atr50 = atr_s.rolling(50).mean().iloc[-1]
        atr_exp = bool(atr_v / atr50 > 1.2) if (atr50 and not np.isnan(atr50)) else False
        
        sq = detect_squeeze(c, h, l); rdiv = detect_rsi_divergence(c, rsi_s)
        qs = calc_quality_score(price, e20, e50, vr, obv_t, atr_exp, rsi_v)
        
        wb = None
        try:
            wc = c.resample('W').last()
            if len(wc) >= 5: wb = bool(wc.iloc[-1] > wc.ewm(span=20).mean().iloc[-1])
        except Exception: pass
        
        t60 = data.tail(60)
        chart = {
            "dates": t60.index.strftime("%Y-%m-%d").tolist(), "open": t60["Open"].round(2).tolist(),
            "high": t60["High"].round(2).tolist(), "low": t60["Low"].round(2).tolist(), "close": t60["Close"].round(2).tolist(),
            "volume": t60["Volume"].astype(int).tolist(), "ema20": e20s.tail(60).round(2).tolist(), "ema50": e50s.tail(60).round(2).tolist(),
            "bb_up": calc_bollinger(c)[0].tail(60).round(2).tolist(), "bb_dn": calc_bollinger(c)[2].tail(60).round(2).tolist()
        }
        
        dist_e = abs(price - e20) / e20
        es = round(max(0.0, (1.0 - dist_e / e_h) * 10.0), 1) if dist_e < e_h else 0.0
        se = "EARLY" if es > 0 else "-"
        ps = 0
        if price > e20: ps += 3
        if p_rmin < rsi_v < p_rmax: ps += 3
        if vr > 1.2: ps += 2
        sp = "PRO" if ps >= 4 else "-"
        
        poc, dpoc, srea = price, 0.0, "-"
        try:
            bins = np.linspace(float(l.min()), float(h.max()), 50)
            pbins = pd.cut((h + l + c) / 3, bins, labels=bins[:-1])
            vp = pd.DataFrame({"P": pbins, "V": v}).groupby("P")["V"].sum()
            poc = float(vp.idxmax()); dpoc = abs(price - poc) / poc
            if dpoc < r_poc and vr > vol_ratio_hot * 0.3: srea = "HOT"
        except Exception: pass
        
        common = {
            "Nome": ticker, "Ticker": ticker, "Prezzo": round(price, 2), "Vol_Today": int(vt), "Vol_7d_Avg": int(v7), "Avg_Vol_20": int(av20),
            "Vol_Ratio": round(vr, 2), "RSI": round(rsi_v, 1), "OBV_Trend": obv_t, "ATR": round(atr_v, 2), "Squeeze": sq,
            "RSI_Div": rdiv if rdiv else "-", "Weekly_Bull": wb, "EMA20": round(e20, 2), "EMA50": round(e50, 2), "EMA200": e200,
            "Quality_Score": qs, "_chart_data": chart, "_quality_components": {"VolRatio": min(vr/3, 1), "OBV": 1.0 if obv_t == "UP" else 0.0}
        }
        return {**common, "Early_Score": es, "Pro_Score": ps, "Stato_Early": se, "Stato_Pro": sp}, ({**common, "Rea_Score": 7, "POC": round(poc, 2), "Dist_POC_%": round(dpoc*100, 1), "Stato": srea} if srea=="HOT" else None)
    except Exception as e:
        _SCAN_ERRORS.append(f"{ticker}: {type(e).__name__}")
        return None, None

def scan_universe(universe: list, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot=1.5, **kwargs):
    global _SCAN_ERRORS; _SCAN_ERRORS = []; rep, rrea = [], []; lock = threading.Lock(); c = [0]; t0 = time.time()
    def _one(t):
        e, r = scan_ticker(t, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
        with lock:
            c[0] += 1
            if kwargs.get("progress_callback"): kwargs["progress_callback"](c[0], len(universe), t)
        return e, r
    nw = min(kwargs.get("n_workers", 8), 12)
    with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(_one, t): t for t in universe}
        for f in concurrent.futures.as_completed(futs):
            try:
                e, r = f.result()
                if e: rep.append(e)
                if r: rrea.append(r)
            except Exception: pass
    return pd.DataFrame(rep), pd.DataFrame(rrea), {"elapsed_s": round(time.time()-t0, 1), "total": len(universe), "n_errors": len(_SCAN_ERRORS), "errors": _SCAN_ERRORS[:10], "downloaded": c[0], "cache_hits": 0}
