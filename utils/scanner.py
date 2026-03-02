import threading
import concurrent.futures
import time
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

_SCAN_ERRORS: list = []

# -------------------------------------------------------------------------
# INDICATORI TECNICI
# -------------------------------------------------------------------------
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
    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift()), abs(low - close.shift()))
    )
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
    return bool(
        bb_up.iloc[-1] < kc_up.iloc[-1] and bb_dn.iloc[-1] > kc_dn.iloc[-1]
    )

def detect_rsi_divergence(close, rsi_series, lookback=20):
    c = close.tail(lookback)
    r = rsi_series.tail(lookback)
    if len(c) < lookback: return None
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.9:
        return "BEARISH"
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.1:
        return "BULLISH"
    return None

def calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    score = 0
    if vol_ratio > 1.5: score += 2
    if obv_trend == "UP": score += 2
    if atr_expansion: score += 1
    if 45 <= rsi_val <= 65: score += 3
    if price > ema20: score += 2
    if price > ema50: score += 2
    return score

def calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    return {
        "VolRatio": min(vol_ratio / 3.0, 1.0),
        "OBV": 1.0 if obv_trend == "UP" else 0.0,
        "ATRExp": 1.0 if atr_expansion else 0.0,
        "RSI Zone": max(0.0, 1.0 - abs(rsi_val - 55) / 25.0),
        "EMA20 Bull": 1.0 if price > ema20 else 0.0,
        "EMA50 Bull": 1.0 if price > ema50 else 0.0,
    }

# -------------------------------------------------------------------------
# CARICAMENTO UNIVERSE
# -------------------------------------------------------------------------
def load_index_from_csv(filename: str):
    for base in [Path("data"), Path(".")]:
        path = base / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in ["Simbolo", "simbolo", "ticker", "Ticker", "Symbol", "symbol"]:
                    if col in df.columns:
                        return df[col].dropna().astype(str).str.strip().unique().tolist()
            except Exception: pass
    return []

def load_universe(markets: list) -> list:
    tickers = []
    if "SP500" in markets: tickers += load_index_from_csv("sp500.csv")
    if "Nasdaq" in markets: tickers += load_index_from_csv("nasdaq100.csv")
    if "FTSE" in markets: 
        for t in load_index_from_csv("ftsemib.csv"):
            tickers.append(t if t.endswith(".MI") else t + ".MI")
    if "Dow" in markets: tickers += load_index_from_csv("dowjones.csv")
    if "Russell" in markets: tickers += load_index_from_csv("russell2000.csv")
    return list(dict.fromkeys(tickers))

# -------------------------------------------------------------------------
# SCAN TICKER — v28.3 (Fix naming + skip yf.info)
# -------------------------------------------------------------------------
def scan_ticker(ticker: str, e_h: float, p_rmin: int, p_rmax: int, r_poc: float, vol_ratio_hot: float = 1.5):
    try:
        data = yf.Ticker(ticker).history(period="6mo", timeout=15)
        if data is None or len(data) < 20: return None, None

        c, h, l, v = data["Close"], data["High"], data["Low"], data["Volume"]
        price = float(c.iloc[-1])
        if price <= 0: return None, None

        # Info base senza chiamare yf.info (lento e blocca)
        name = ticker
        currency = "USD"
        
        vol_today = float(v.iloc[-1])
        avg_vol_20 = float(v.rolling(20).mean().iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())
        vol_ratio = float(vol_today / avg_vol_20) if avg_vol_20 > 0 else 0.0

        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        ema50 = float(c.ewm(span=50).mean().iloc[-1])
        sma200_ser = c.rolling(200).mean()
        ema200 = float(sma200_ser.iloc[-1]) if not np.isnan(sma200_ser.iloc[-1]) else None

        rsi_series = calc_rsi(c)
        rsi_val = float(rsi_series.iloc[-1])
        obv = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"

        atr_series = calc_atr(h, l, c)
        atr_val = float(atr_series.iloc[-1])
        atr_50 = atr_series.rolling(50).mean().iloc[-1]
        atr_expansion = bool(atr_val / atr_50 > 1.2) if (atr_50 and not np.isnan(atr_50)) else False

        in_squeeze = detect_squeeze(c, h, l)
        rsi_divergence = detect_rsi_divergence(c, rsi_series)
        
        quality_score = calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)
        quality_comps = calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)

        # Weekly trend (opzionale)
        weekly_bullish = None
        try:
            w_data = yf.Ticker(ticker).history(period="6mo", interval="1wk", timeout=10)
            if len(w_data) >= 5:
                w_c = w_data["Close"]
                weekly_bullish = bool(w_c.iloc[-1] > w_c.ewm(span=20).mean().iloc[-1])
        except Exception: pass

        # Dati per grafici
        tail60 = data.tail(60)
        chart_data = {
            "dates": tail60.index.strftime("%Y-%m-%d").tolist(),
            "open": [round(float(x), 2) for x in tail60["Open"]],
            "high": [round(float(x), 2) for x in tail60["High"]],
            "low": [round(float(x), 2) for x in tail60["Low"]],
            "close": [round(float(x), 2) for x in tail60["Close"]],
            "volume": [int(x) for x in tail60["Volume"]],
            "ema20": [round(float(x), 2) for x in c.ewm(span=20).mean().tail(60)],
            "ema50": [round(float(x), 2) for x in c.ewm(span=50).mean().tail(60)],
            "bb_up": [round(float(x), 2) for x in calc_bollinger(c)[0].tail(60)],
            "bb_dn": [round(float(x), 2) for x in calc_bollinger(c)[2].tail(60)],
        }

        dist_ema = abs(price - ema20) / ema20
        early_score = round(max(0.0, (1.0 - dist_ema / e_h) * 10.0), 1) if dist_ema < e_h else 0.0
        stato_early = "EARLY" if early_score > 0 else "-"

        pro_score = 0
        if price > ema20: pro_score += 3
        if p_rmin < rsi_val < p_rmax: pro_score += 3
        if vol_ratio > 1.2: pro_score += 2
        stato_pro = "PRO" if pro_score >= 4 else "-"

        # Point of Control (POC)
        poc, dist_poc, stato_rea = price, 0.0, "-"
        try:
            bins = np.linspace(float(l.min()), float(h.max()), 50)
            pbins = pd.cut((h + l + c) / 3, bins, labels=bins[:-1])
            vp = pd.DataFrame({"P": pbins, "V": v}).groupby("P")["V"].sum()
            poc = float(vp.idxmax())
            dist_poc = abs(price - poc) / poc
            if dist_poc < r_poc and vol_ratio > vol_ratio_hot * 0.3:
                stato_rea = "HOT"
        except Exception: pass

        # Nomi colonne normalizzati (Snake_Case come richiesto da Dashboard_pro V_28)
        common = {
            "Nome": name,
            "Ticker": ticker,
            "Prezzo": round(price, 2),
            "Vol_Today": int(vol_today),
            "Vol_7d_Avg": int(vol_7d_avg),
            "Avg_Vol_20": int(avg_vol_20),
            "Vol_Ratio": round(vol_ratio, 2),
            "RSI": round(rsi_val, 1),
            "OBV_Trend": obv_trend,
            "ATR": round(atr_val, 2),
            "Squeeze": in_squeeze,
            "RSI_Div": rsi_divergence if rsi_divergence else "-",
            "Weekly_Bull": weekly_bullish,
            "EMA20": round(ema20, 2),
            "EMA50": round(ema50, 2),
            "EMA200": round(ema200, 2) if ema200 else None,
            "Quality_Score": quality_score,
            "_quality_components": quality_comps,
            "_chart_data": chart_data,
        }

        res_ep = {**common, "Early_Score": early_score, "Pro_Score": pro_score, "Stato_Early": stato_early, "Stato_Pro": stato_pro}
        res_rea = {**common, "Rea_Score": 7, "POC": round(poc, 2), "Dist_POC_%": round(dist_poc * 100, 1), "Stato": stato_rea} if stato_rea == "HOT" else None

        return res_ep, res_rea

    except Exception as e:
        _SCAN_ERRORS.append(f"{ticker}: {type(e).__name__}: {e}")
        return None, None

def scan_universe(universe: list, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot=1.5, cache_enabled=True, finviz_enabled=False, n_workers=8, progress_callback=None):
    global _SCAN_ERRORS
    _SCAN_ERRORS = []
    rep, rrea = [], []
    lock = threading.Lock()
    counter = [0]
    t0 = time.time()
    
    def _one(tkr):
        ep, rea = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
        with lock:
            counter[0] += 1
            if progress_callback:
                progress_callback(counter[0], len(universe), tkr)
        return ep, rea

    nw = min(max(n_workers, 1), 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as ex:
        futures = {ex.submit(_one, t): t for t in universe}
        for fut in concurrent.futures.as_completed(futures):
            try:
                ep, rea = fut.result()
                if ep: rep.append(ep)
                if rea: rrea.append(rea)
            except Exception: pass

    return pd.DataFrame(rep), pd.DataFrame(rrea), {
        "elapsed_s": round(time.time() - t0, 1),
        "cache_hits": 0,
        "downloaded": len(universe),
        "workers": nw,
        "total": len(universe),
        "n_errors": len(_SCAN_ERRORS),
        "errors": _SCAN_ERRORS[:20]
    }
