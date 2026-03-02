import threading
import concurrent.futures
import time

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


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
    tr = np.maximum(high - low, np.maximum(
        abs(high - close.shift()), abs(low - close.shift())))
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
    c = close.tail(lookback)
    r = rsi_series.tail(lookback)
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.9:
        return "BEARISH"
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.1:
        return "BULLISH"
    return None


def calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    score = 0
    if vol_ratio > 1.5:      score += 2
    if obv_trend == "UP":    score += 2
    if atr_expansion:        score += 1
    if 45 <= rsi_val <= 65:  score += 3
    if price > ema20:        score += 2
    if price > ema50:        score += 2
    return score


def calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    return {
        "Vol_Ratio":  min(vol_ratio / 3.0, 1.0),
        "OBV":        1.0 if obv_trend == "UP" else 0.0,
        "ATR_Exp":    1.0 if atr_expansion else 0.0,
        "RSI Zone":   max(0.0, 1.0 - abs(rsi_val - 55) / 25.0),
        "EMA20 Bull": 1.0 if price > ema20 else 0.0,
        "EMA50 Bull": 1.0 if price > ema50 else 0.0,
    }


# -------------------------------------------------------------------------
# CARICAMENTO UNIVERSE
# -------------------------------------------------------------------------

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
    if "SP500"         in markets: t += load_index_from_csv("sp500.csv")
    if "Eurostoxx"     in markets: t += load_index_from_csv("eurostoxx600.csv")
    if "FTSE"          in markets: t += load_index_from_csv("ftsemib.csv")
    if "Nasdaq"        in markets: t += load_index_from_csv("nasdaq100.csv")
    if "Dow"           in markets: t += load_index_from_csv("dowjones.csv")
    if "Russell"       in markets: t += load_index_from_csv("russell2000.csv")
    if "StoxxEmerging" in markets: t += load_index_from_csv("stoxx emerging market 50.csv")
    if "USSmallCap"    in markets: t += load_index_from_csv("us small cap 2000.csv")
    return list(dict.fromkeys(t))


# -------------------------------------------------------------------------
# SCAN TICKER — v28.1
# Fix principali rispetto v22:
#   - pro_score soglia: 8 -> 6  (era praticamente irraggiungibile)
#   - early: include ticker anche se solo EARLY (non richiede anche PRO)
#   - aggiunge EMA20, Avg_Vol_20, Rel_Vol, Ser_OK/Score, FV_OK/Score
#   - gestione errori piu' granulare (non swallowa tutto in except)
# -------------------------------------------------------------------------

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot=1.5):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data) < 50:
            return None, None

        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]

        # Info base
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}

        name       = str(info.get("longName", info.get("shortName", ticker)))[:50]
        price      = float(c.iloc[-1])
        currency   = info.get("currency", "USD")
        market_cap = info.get("marketCap", np.nan)
        vol_today  = float(v.iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())
        avg_vol_20 = float(v.rolling(20).mean().iloc[-1])

        # Medie mobili
        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        ema50 = float(c.ewm(span=50).mean().iloc[-1])
        sma200_series = c.rolling(200).mean()
        ema200 = float(sma200_series.iloc[-1]) if not np.isnan(sma200_series.iloc[-1]) else None

        # Indicatori
        rsi_series  = calc_rsi(c)
        rsi_val     = float(rsi_series.iloc[-1])
        vol_ratio   = float(v.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 0.0
        obv         = calc_obv(c, v)
        obv_slope   = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend   = "UP" if obv_slope > 0 else "DOWN"
        atr_series      = calc_atr(h, l, c)
        atr_val         = float(atr_series.iloc[-1])
        atr_50avg       = atr_series.rolling(50).mean().iloc[-1]
        atr_expansion   = bool((atr_val / atr_50avg) > 1.2) if not np.isnan(atr_50avg) else False
        in_squeeze      = detect_squeeze(c, h, l)
        rsi_divergence  = detect_rsi_divergence(c, rsi_series)
        quality_score   = calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)
        quality_comps   = calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)

        # Multi-timeframe settimanale
        try:
            data_w = yf.Ticker(ticker).history(period="6mo", interval="1wk")
            c_w    = data_w["Close"]
            if len(data_w) >= 5:
                ema20_w        = float(c_w.ewm(span=20).mean().iloc[-1])
                weekly_bullish = float(c_w.iloc[-1]) > ema20_w
            else:
                weekly_bullish = None
        except Exception:
            weekly_bullish = None

        # Chart data (60 giorni)
        tail60    = data.tail(60).copy()
        ema20_ser = c.ewm(span=20).mean()
        ema50_ser = c.ewm(span=50).mean()
        bb_up, _, bb_dn = calc_bollinger(c)
        chart_data = {
            "dates":  tail60.index.strftime("%Y-%m-%d").tolist(),
            "open":   [round(x, 2) for x in tail60["Open"].tolist()],
            "high":   [round(x, 2) for x in tail60["High"].tolist()],
            "low":    [round(x, 2) for x in tail60["Low"].tolist()],
            "close":  [round(x, 2) for x in tail60["Close"].tolist()],
            "volume": [int(x) for x in tail60["Volume"].tolist()],
            "ema20":  [round(x, 2) for x in ema20_ser.tail(60).tolist()],
            "ema50":  [round(x, 2) for x in ema50_ser.tail(60).tolist()],
            "bb_up":  [round(x, 2) for x in bb_up.tail(60).tolist()],
            "bb_dn":  [round(x, 2) for x in bb_dn.tail(60).tolist()],
        }

        # ------------------------------------------------------------------
        # SCORING
        # ------------------------------------------------------------------

        # EARLY: prezzo vicino EMA20 (dentro la banda e_h)
        dist_ema    = abs(price - ema20) / ema20
        early_score = round(max(0.0, (1.0 - dist_ema / e_h) * 10.0), 1) if dist_ema < e_h else 0.0
        stato_early = "EARLY" if early_score > 0 else "-"

        # PRO: soglia 6/8 (era 8/8, quasi impossibile)
        #   +3 prezzo sopra EMA20
        #   +3 RSI nel range [p_rmin, p_rmax]
        #   +2 volume sopra media
        pro_score = 0
        if price > ema20:              pro_score += 3
        if p_rmin < rsi_val < p_rmax:  pro_score += 3
        if vol_ratio > 1.2:            pro_score += 2
        stato_pro = "PRO" if pro_score >= 6 else "-"   # FIX: era >= 8

        # HOT (REA): volume anomalo vicino al POC
        try:
            tp   = (h + l + c) / 3
            bins = np.linspace(float(l.min()), float(h.max()), 50)
            price_bins = pd.cut(tp, bins, labels=bins[:-1])
            vp   = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
            poc      = float(vp.idxmax())
            dist_poc = abs(price - poc) / poc
            rea_score = 7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0
            stato_rea = "HOT" if rea_score >= 7 else "-"
        except Exception:
            poc = price; dist_poc = 0.0; rea_score = 0; stato_rea = "-"

        # SERAFINI (6 criteri): RSI>50, P>EMA20, EMA20>EMA50, OBV_UP, Vol>1, no earnings
        ser_c1 = rsi_val > 50
        ser_c2 = price > ema20
        ser_c3 = ema20 > ema50
        ser_c4 = obv_trend == "UP"
        ser_c5 = vol_ratio > 1.0
        ser_c6 = True   # earnings_soon non disponibile in v22 — default True
        ser_score = sum([ser_c1, ser_c2, ser_c3, ser_c4, ser_c5, ser_c6])
        ser_ok    = all([ser_c1, ser_c2, ser_c3, ser_c4, ser_c5, ser_c6])

        # FINVIZ (criteri base): Price>10, AvgVol>500K, RelVol>1, P>EMA20, P>EMA50
        fv_c1 = price > 10
        fv_c2 = avg_vol_20 > 500_000
        fv_c3 = vol_ratio > 1.0
        fv_c4 = price > ema20
        fv_c5 = price > ema50
        fv_score = sum([fv_c1, fv_c2, fv_c3, fv_c4, fv_c5])
        fv_ok    = fv_c1 and fv_c2 and fv_c3 and fv_c4 and fv_c5

        # ------------------------------------------------------------------
        # RECORD
        # ------------------------------------------------------------------
        common = {
            "Nome":        name,
            "Ticker":      ticker,
            "Prezzo":      round(price, 2),
            "MarketCap":   market_cap,
            "Vol_Today":   int(vol_today),
            "Vol_7d_Avg":  int(vol_7d_avg),
            "Avg_Vol_20":  int(avg_vol_20),
            "Rel_Vol":     round(vol_ratio, 2),
            "Currency":    currency,
            "RSI":         round(rsi_val, 1),
            "Vol_Ratio":   round(vol_ratio, 2),
            "OBV_Trend":   obv_trend,
            "ATR":         round(atr_val, 2),
            "ATR_Exp":     atr_expansion,
            "Squeeze":     in_squeeze,
            "RSI_Div":     rsi_divergence if rsi_divergence else "-",
            "Weekly_Bull": weekly_bullish,
            "EMA20":       round(ema20, 2),
            "EMA50":       round(ema50, 2),
            "EMA200":      round(ema200, 2) if ema200 else None,
            "Quality_Score": quality_score,
            "Ser_OK":      ser_ok,
            "Ser_Score":   ser_score,
            "FV_OK":       fv_ok,
            "FV_Score":    fv_score,
            "_quality_components": quality_comps,
            "_chart_data":         chart_data,
        }

        # ep: includi se EARLY oppure PRO (non richiedere entrambi)
        if stato_early == "-" and stato_pro == "-":
            res_ep = None
        else:
            res_ep = {
                **common,
                "Early_Score": early_score,
                "Pro_Score":   pro_score,
                "Stato":       stato_pro if stato_pro != "-" else stato_early,
                "Stato_Early": stato_early,
                "Stato_Pro":   stato_pro,
            }

        res_rea = None if stato_rea == "-" else {
            **common,
            "Rea_Score":  rea_score,
            "POC":        round(poc, 2),
            "Dist_POC_%": round(dist_poc * 100, 1),
            "Pro_Score":  pro_score,
            "Stato":      stato_rea,
        }

        return res_ep, res_rea

    except Exception:
        return None, None


# -------------------------------------------------------------------------
# SCAN UNIVERSE — parallelo con ThreadPoolExecutor
# -------------------------------------------------------------------------

def scan_universe(universe: list, e_h, p_rmin, p_rmax, r_poc,
                  vol_ratio_hot=1.5, cache_enabled=True, finviz_enabled=False,
                  n_workers=8, progress_callback=None):
    """
    Scansione parallela dell'universo.
    Parametri extra (cache_enabled, finviz_enabled) accettati per
    compatibilita' con il dashboard v28 ma ignorati in questa versione.
    """
    rep, rrea = [], []
    lock      = threading.Lock()
    counter   = [0]
    t0        = time.time()
    tot       = len(universe)

    def _one(tkr):
        ep, rea = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
        with lock:
            counter[0] += 1
            if progress_callback:
                progress_callback(counter[0], tot, tkr)
        return ep, rea

    nw = min(max(n_workers, 1), 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as ex:
        futures = {ex.submit(_one, t): t for t in universe}
        for fut in concurrent.futures.as_completed(futures):
            try:
                ep, rea = fut.result()
                if ep:  rep.append(ep)
                if rea: rrea.append(rea)
            except Exception:
                pass

    df_ep  = pd.DataFrame(rep)  if rep  else pd.DataFrame()
    df_rea = pd.DataFrame(rrea) if rrea else pd.DataFrame()
    stats  = {
        "elapsed_s":  round(time.time() - t0, 1),
        "cache_hits": 0,
        "downloaded": tot,
        "workers":    nw,
        "total":      tot,
        "ep_found":   len(rep),
        "rea_found":  len(rrea),
        "finviz":     False,
    }
    return df_ep, df_rea, stats

