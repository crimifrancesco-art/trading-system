"""
utils/scanner.py  —  v28.0
===========================
Novità:
  #1  Cache SQLite  — storica su Streamlit Cloud (history, info, finviz)
  #5  finvizfinance — dati fondamentali reali (EPS growth, short float, optionable)
      con fallback automatico a yfinance se scraping fallisce
  Parallelo ThreadPoolExecutor mantenuto (8 worker default)
"""

import time
import random
import threading
import concurrent.futures

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# Import db per cache — fallback se non disponibile
try:
    from utils.db import cache_get, cache_set, df_to_cache_json, cache_json_to_df
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    def cache_get(*a, **k): return None
    def cache_set(*a, **k): pass
    def df_to_cache_json(df): return []
    def cache_json_to_df(d, **k): return pd.DataFrame()

# finvizfinance — fallback gracile se non installata
try:
    from finvizfinance.quote import finvizfinance as fvf
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False


# =========================================================================
# INDICATORI TECNICI
# =========================================================================

def calc_obv(close, volume):
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()

def calc_rsi(close, period=14):
    d    = close.diff()
    gain = d.where(d > 0, 0).rolling(period).mean()
    loss = -d.where(d < 0, 0).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))

def calc_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(
        (high - close.shift()).abs(), (low - close.shift()).abs()))
    return tr.rolling(period).mean()

def calc_bollinger(close, period=20, std_dev=2):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ma + std_dev * sd, ma, ma - std_dev * sd

def calc_keltner(close, high, low, period=20, atr_mult=1.5):
    ema = close.ewm(span=period).mean()
    atr = calc_atr(high, low, close, period)
    return ema + atr_mult * atr, ema, ema - atr_mult * atr

def detect_squeeze(close, high, low):
    bb_up, _, bb_dn = calc_bollinger(close)
    kc_up, _, kc_dn = calc_keltner(close, high, low)
    return bool(bb_up.iloc[-1] < kc_up.iloc[-1] and bb_dn.iloc[-1] > kc_dn.iloc[-1])

def detect_rsi_divergence(close, rsi, lookback=20):
    c = close.tail(lookback); r = rsi.tail(lookback)
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.9: return "BEARISH"
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.1: return "BULLISH"
    return None

def calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_exp, rsi):
    s  = 0
    if vol_ratio > 1.5:    s += 2
    if obv_trend == "UP":  s += 2
    if atr_exp:            s += 1
    if 45 <= rsi <= 65:    s += 3
    if price > ema20:      s += 2
    if price > ema50:      s += 2
    return s

def calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend, atr_exp, rsi):
    return {
        "Vol_Ratio":  min(vol_ratio / 3.0, 1.0),
        "OBV":        1.0 if obv_trend == "UP" else 0.0,
        "ATR_Exp":    1.0 if atr_exp else 0.0,
        "RSI Zone":   max(0.0, 1.0 - abs(rsi - 55) / 25.0),
        "EMA20 Bull": 1.0 if price > ema20 else 0.0,
        "EMA50 Bull": 1.0 if price > ema50 else 0.0,
    }

def calc_serafini_score(price, ema20, ema50, rsi, obv_trend, vol_ratio, earnings_soon):
    c1 = rsi > 50; c2 = price > ema20; c3 = ema20 > ema50
    c4 = obv_trend == "UP"; c5 = vol_ratio > 1.0; c6 = not earnings_soon
    return (sum([c1,c2,c3,c4,c5,c6]),
            all([c1,c2,c3,c4,c5,c6]),
            {"RSI>50":c1,"P>EMA20":c2,"EMA20>EMA50":c3,
             "OBV_UP":c4,"VolRatio>1":c5,"No_Earnings":c6})

def calc_finviz_score(price, avg_vol_20, rel_vol, ema20, ema50, ema200,
                       eps_ny, eps_5y, optionable):
    c1 = price > 10
    c2 = avg_vol_20 > 1_000_000
    c3 = rel_vol    > 1.0
    c4 = bool(price > ema20)  if ema20  else False
    c5 = bool(price > ema50)  if ema50  else False
    c6 = bool(price > ema200) if ema200 else False
    c7 = (eps_ny  is not None and eps_ny  > 0.10)
    c8 = (eps_5y  is not None and eps_5y  > 0.15)
    return (sum([c1,c2,c3,c4,c5,c6,c7,c8]),
            all([c1,c2,c3,c4,c5,c6]),
            {"Price>10":c1,"AvgVol>1M":c2,"RelVol>1":c3,
             "P>SMA20":c4,"P>SMA50":c5,"P>SMA200":c6,
             "EPS_NY>10%":c7,"EPS_5Y>15%":c8})


# =========================================================================
# CACHE WRAPPERS  (SQLite via db.py)
# =========================================================================

def _get_history(ticker: str, period: str = "9mo",
                 interval: str = "1d", force: bool = False) -> pd.DataFrame:
    kind = f"history_{period}_{'weekly' if interval != '1d' else 'daily'}"
    if not force:
        cached = cache_get(ticker, kind)
        if cached:
            try:
                return cache_json_to_df(cached, index_col="Date")
            except Exception:
                pass
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if not df.empty:
            cache_set(ticker, kind, df_to_cache_json(df))
        return df
    except Exception:
        return pd.DataFrame()


def _get_info(ticker: str, force: bool = False) -> dict:
    if not force:
        cached = cache_get(ticker, "info")
        if cached:
            return cached
    try:
        raw = yf.Ticker(ticker).info
        clean = {}
        for k, v in raw.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                clean[k] = v
            elif isinstance(v, np.integer):  clean[k] = int(v)
            elif isinstance(v, np.floating): clean[k] = float(v)
        cache_set(ticker, "info", clean)
        return clean
    except Exception:
        return {}


def _get_calendar(ticker: str, force: bool = False):
    if not force:
        cached = cache_get(ticker, "calendar")
        if cached:
            try:
                return pd.DataFrame(cached)
            except Exception:
                pass
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None and not cal.empty:
            cache_set(ticker, "calendar", df_to_cache_json(cal))
        return cal
    except Exception:
        return None


# =========================================================================
# FINVIZ SCRAPER  (Upgrade #5)
# Rate-limited: max 1 req/sec + random jitter + retry
# =========================================================================

_fv_lock = threading.Semaphore(2)   # max 2 richieste Finviz simultanee
_fv_last  = [0.0]                    # timestamp ultima richiesta

def _get_finviz(ticker: str, force: bool = False) -> dict:
    """
    Scarica dati Finviz via finvizfinance (scraping gratuito).
    Ritorna dict con chiavi standardizzate.
    Fallback a {} se non disponibile o errore.

    Dati restituiti:
        eps_growth_ny  : EPS Growth Next Year (float, es. 0.15 = 15%)
        eps_growth_5y  : EPS Growth Next 5 Years
        short_float    : Short Float %
        optionable     : bool
        shortable      : bool
        insider_own    : Insider Ownership %
        inst_own       : Institutional Ownership %
        pe             : P/E ratio
        fwd_pe         : Forward P/E
        peg            : PEG ratio
        roe            : Return on Equity
        roa            : Return on Assets
        debt_eq        : Debt/Equity
        gross_margin   : Gross Margin
        op_margin      : Operating Margin
        profit_margin  : Profit Margin
    """
    if not FINVIZ_AVAILABLE:
        return {}

    # Check cache prima
    if not force:
        cached = cache_get(ticker, "finviz")
        if cached:
            return cached

    # Rate limiting: non più di 1 req/sec globale
    with _fv_lock:
        elapsed = time.time() - _fv_last[0]
        if elapsed < 1.2:
            time.sleep(1.2 - elapsed + random.uniform(0.1, 0.4))
        _fv_last[0] = time.time()

        for attempt in range(3):
            try:
                stock = fvf(ticker)
                raw   = stock.TickerFundamentals()

                def _pct(key):
                    """Converte '15.23%' → 0.1523, '-' → None"""
                    val = raw.get(key, "-")
                    if val in ("-", "", None): return None
                    try: return float(str(val).replace("%", "").replace(",", "")) / 100
                    except: return None

                def _float(key):
                    val = raw.get(key, "-")
                    if val in ("-", "", None): return None
                    try: return float(str(val).replace(",", ""))
                    except: return None

                result = {
                    "eps_growth_ny": _pct("EPS next Y"),
                    "eps_growth_5y": _pct("EPS next 5Y"),
                    "short_float":   _pct("Short Float"),
                    "optionable":    raw.get("Optionable", "-") == "Yes",
                    "shortable":     raw.get("Shortable",  "-") == "Yes",
                    "insider_own":   _pct("Insider Own"),
                    "inst_own":      _pct("Inst Own"),
                    "pe":            _float("P/E"),
                    "fwd_pe":        _float("Forward P/E"),
                    "peg":           _float("PEG"),
                    "roe":           _pct("ROE"),
                    "roa":           _pct("ROA"),
                    "debt_eq":       _float("Debt/Eq"),
                    "gross_margin":  _pct("Gross Margin"),
                    "op_margin":     _pct("Oper. Margin"),
                    "profit_margin": _pct("Profit Margin"),
                }
                cache_set(ticker, "finviz", result)
                return result

            except Exception as e:
                wait = (attempt + 1) * 2 + random.uniform(0, 1)
                if attempt < 2:
                    time.sleep(wait)
                else:
                    print(f"[finviz] {ticker} fallito dopo 3 tentativi: {e}")
        return {}


# =========================================================================
# CARICAMENTO UNIVERSE  (invariato — CSV da trading-system/data/)
# =========================================================================

def load_index_from_csv(filename: str):
    path = Path("data") / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    for col in ["ticker","Simbolo","simbolo","Ticker","Symbol","symbol"]:
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


# =========================================================================
# SCAN TICKER  —  v28.0
# =========================================================================

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc,
                vol_ratio_hot=1.5, cache_enabled=True,
                finviz_enabled=True):
    """
    Analizza un singolo ticker.
    cache_enabled  → usa SQLite cache per yfinance (salva chiamate API)
    finviz_enabled → integra dati fondamentali reali da Finviz scraping
    """
    try:
        # ── Dati OHLCV ───────────────────────────────────────────────────
        data = _get_history(ticker, "9mo", "1d", force=not cache_enabled)
        if len(data) < 60: return None, None
        c = data["Close"]; h = data["High"]; l = data["Low"]; v = data["Volume"]

        # ── Info base yfinance ────────────────────────────────────────────
        info   = _get_info(ticker, force=not cache_enabled)
        name   = info.get("longName", info.get("shortName", ticker))[:50]
        price  = float(c.iloc[-1])
        curr   = info.get("currency", "USD")
        mcap   = info.get("marketCap", np.nan)
        vol_td = float(v.iloc[-1])
        vol_7d = float(v.tail(7).mean())

        # ── Medie mobili ─────────────────────────────────────────────────
        ema20  = float(c.ewm(span=20).mean().iloc[-1])
        ema50  = float(c.ewm(span=50).mean().iloc[-1])
        sma200 = c.rolling(200).mean()
        ema200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None

        # ── Indicatori ───────────────────────────────────────────────────
        rsi_s  = calc_rsi(c); rsi_v = float(rsi_s.iloc[-1])
        avg20  = float(v.rolling(20).mean().iloc[-1])
        volr   = float(v.iloc[-1] / avg20) if avg20 > 0 else 0.0
        obv    = calc_obv(c, v)
        obv_t  = "UP" if obv.diff().rolling(5).mean().iloc[-1] > 0 else "DOWN"
        atr_s  = calc_atr(h, l, c)
        atr_v  = float(atr_s.iloc[-1])
        atr_ex = (atr_v / atr_s.rolling(50).mean().iloc[-1]) > 1.2
        squeeze= detect_squeeze(c, h, l)
        rsi_dv = detect_rsi_divergence(c, rsi_s)

        # ── Fondamentali: prima Finviz, fallback yfinance ─────────────────
        fv_data = {}
        if finviz_enabled and FINVIZ_AVAILABLE:
            fv_data = _get_finviz(ticker, force=not cache_enabled)

        # EPS Growth Next Year
        eps_ny = fv_data.get("eps_growth_ny")
        if eps_ny is None:
            eps_fwd  = info.get("forwardEps",  None)
            eps_trl  = info.get("trailingEps", None)
            eps_ny   = info.get("earningsGrowth", None)
            if eps_fwd and eps_trl and eps_trl != 0:
                eps_ny = (eps_fwd - eps_trl) / abs(eps_trl)

        # EPS Growth 5Y
        eps_5y = fv_data.get("eps_growth_5y")
        if eps_5y is None:
            eps_5y = info.get("revenueGrowth", None) \
                     or info.get("earningsQuarterlyGrowth", None)

        # Altri fondamentali: preferisci Finviz (più accurato) poi yfinance
        pe_r     = fv_data.get("pe")       or info.get("trailingPE",         None)
        fwd_pe   = fv_data.get("fwd_pe")   or info.get("forwardPE",          None)
        peg      = fv_data.get("peg")
        roe      = fv_data.get("roe")      or info.get("returnOnEquity",      None)
        roa      = fv_data.get("roa")
        debt_eq  = fv_data.get("debt_eq")  or info.get("debtToEquity",        None)
        gm       = fv_data.get("gross_margin") or info.get("grossMargins",    None)
        op_m     = fv_data.get("op_margin")    or info.get("operatingMargins",None)
        pm       = fv_data.get("profit_margin")
        sf       = fv_data.get("short_float") or info.get("shortPercentOfFloat", None)
        opt      = fv_data.get("optionable", False) or bool(
                       info.get("exchange", "") in ["NMS","NYQ","ASE","PCX","CBT"])
        ins_own  = fv_data.get("insider_own")
        inst_own = fv_data.get("inst_own")

        # ── Earnings imminenti ────────────────────────────────────────────
        earn_soon = False
        try:
            cal = _get_calendar(ticker, force=not cache_enabled)
            if cal is not None and not cal.empty:
                ed = cal.get("Earnings Date")
                if ed is not None:
                    ed_val = pd.to_datetime(ed.iloc[0] if hasattr(ed,"iloc") else ed)
                    days   = (ed_val.tz_localize(None) - pd.Timestamp.now()).days
                    earn_soon = 0 <= days <= 14
        except Exception:
            pass

        # ── Quality / Multi-TF ────────────────────────────────────────────
        qs = calc_quality_score(price, ema20, ema50, volr, obv_t, atr_ex, rsi_v)
        qc = calc_quality_components(price, ema20, ema50, volr, obv_t, atr_ex, rsi_v)

        weekly_bull = None
        try:
            dw = _get_history(ticker, "6mo", "1wk", force=not cache_enabled)
            if len(dw) >= 5:
                cw     = dw["Close"]
                ema20w = float(cw.ewm(span=20).mean().iloc[-1])
                weekly_bull = float(cw.iloc[-1]) > ema20w
        except Exception:
            pass

        # ── Scores ───────────────────────────────────────────────────────
        ser_score, ser_ok, ser_crit = calc_serafini_score(
            price, ema20, ema50, rsi_v, obv_t, volr, earn_soon)
        fv_score, fv_ok, fv_crit = calc_finviz_score(
            price, avg20, volr, ema20, ema50, ema200, eps_ny, eps_5y, opt)

        # ── Chart data ────────────────────────────────────────────────────
        t60   = data.tail(60).copy()
        e20s  = c.ewm(span=20).mean()
        e50s  = c.ewm(span=50).mean()
        bb_u, _, bb_d = calc_bollinger(c)
        chart_data = {
            "dates":  t60.index.strftime("%Y-%m-%d").tolist(),
            "open":   [round(x,2) for x in t60["Open"].tolist()],
            "high":   [round(x,2) for x in t60["High"].tolist()],
            "low":    [round(x,2) for x in t60["Low"].tolist()],
            "close":  [round(x,2) for x in t60["Close"].tolist()],
            "volume": [int(x)     for x in t60["Volume"].tolist()],
            "ema20":  [round(x,2) for x in e20s.tail(60).tolist()],
            "ema50":  [round(x,2) for x in e50s.tail(60).tolist()],
            "bb_up":  [round(x,2) for x in bb_u.tail(60).tolist()],
            "bb_dn":  [round(x,2) for x in bb_d.tail(60).tolist()],
        }

        # ── Scoring setup ─────────────────────────────────────────────────
        dist_ema    = abs(price - ema20) / ema20
        early_score = round(max(0.0,(1.0 - dist_ema/e_h)*10.0),1) if dist_ema < e_h else 0.0
        stato_early = "EARLY" if early_score > 0 else "-"

        pro_score = 3 if price > ema20 else 0
        if p_rmin < rsi_v < p_rmax: pro_score += 3
        if volr > 1.2:              pro_score += 2
        stato_pro = "PRO" if pro_score >= 8 else "-"

        tp         = (h + l + c) / 3
        bins       = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp         = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc        = float(vp.idxmax())
        dist_poc   = abs(price - poc) / poc
        rea_score  = 7 if (dist_poc < r_poc and volr > vol_ratio_hot) else 0
        stato_rea  = "HOT" if rea_score >= 7 else "-"

        # ── Record comune ─────────────────────────────────────────────────
        common = {
            "Nome":          name,         "Ticker":       ticker,
            "Prezzo":        round(price,2),"MarketCap":    mcap,
            "Vol_Today":     int(vol_td),  "Vol_7d_Avg":   int(vol_7d),
            "Currency":      curr,
            "RSI":           round(rsi_v,1),"Vol_Ratio":   round(volr,2),
            "Rel_Vol":       round(volr,2), "Avg_Vol_20":  int(avg20),
            "OBV_Trend":     obv_t,         "ATR":         round(atr_v,2),
            "ATR_Exp":       atr_ex,        "Squeeze":     squeeze,
            "RSI_Div":       rsi_dv or "-", "Weekly_Bull": weekly_bull,
            "EMA20":         round(ema20,2),"EMA50":       round(ema50,2),
            "EMA200":        round(ema200,2) if ema200 else None,
            "Quality_Score": qs,
            # Fondamentali (Finviz > yfinance)
            "PE":          _r(pe_r,2),  "Fwd_PE":   _r(fwd_pe,2),
            "PEG":         _r(peg,2),   "ROE":      _r(roe,4),
            "ROA":         _r(roa,4),   "Debt_Eq":  _r(debt_eq,2),
            "Gross_Mgn":   _r(gm,4),    "Op_Mgn":   _r(op_m,4),
            "Profit_Mgn":  _r(pm,4),    "Short_Float": _r(sf,4),
            "Insider_Own": _r(ins_own,4),"Inst_Own": _r(inst_own,4),
            "EPS_NY_Gr":   _r(eps_ny,4),"EPS_5Y_Gr":_r(eps_5y,4),
            "Earnings_Soon": earn_soon,  "Optionable": opt,
            "FV_Source":   "finviz" if fv_data else "yfinance",
            # Serafini
            "Ser_Score":   ser_score,   "Ser_OK":    ser_ok,
            "_ser_criteri":ser_crit,
            # Finviz
            "FV_Score":    fv_score,    "FV_OK":     fv_ok,
            "_fv_criteri": fv_crit,
            # Grafici (non serializzati nel DB)
            "_quality_components": qc,
            "_chart_data":         chart_data,
        }

        res_ep = None
        if stato_early != "-" or stato_pro != "-":
            res_ep = {**common,
                      "Early_Score": early_score, "Pro_Score":  pro_score,
                      "Stato":    stato_pro if stato_pro != "-" else stato_early,
                      "Stato_Early": stato_early, "Stato_Pro": stato_pro}

        res_rea = None if stato_rea == "-" else {
            **common,
            "Rea_Score":  rea_score,  "POC":        round(poc,2),
            "Dist_POC_%": round(dist_poc*100,1), "Pro_Score": pro_score,
            "Stato":      stato_rea}

        return res_ep, res_rea

    except Exception as e:
        return None, None


def _r(v, n):
    """round sicuro con None check"""
    if v is None: return None
    try: return round(float(v), n)
    except: return None


# =========================================================================
# SCAN UNIVERSE  —  orchestratore (parallelo + cache + progress + stats)
# =========================================================================

def scan_universe(universe: list, e_h, p_rmin, p_rmax, r_poc,
                  vol_ratio_hot=1.5, cache_enabled=True, finviz_enabled=True,
                  n_workers=8, progress_callback=None):
    """
    Esegue scansione completa in parallelo con cache e Finviz.

    Returns: (df_ep, df_rea, stats_dict)
    stats_dict include: elapsed_s, cache_hits, downloaded, workers
    """
    n_workers  = min(max(n_workers, 1), 16)
    rep, rrea  = [], []
    lock       = threading.Lock()
    counter    = [0]
    start      = time.time()

    # Conta cache hits prima (stima)
    cache_hits = [0]
    if cache_enabled and CACHE_AVAILABLE:
        for t in universe:
            if cache_get(t, "history_9mo_daily") is not None:
                cache_hits[0] += 1

    def _one(tkr):
        ep, rea = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc,
                               vol_ratio_hot, cache_enabled, finviz_enabled)
        with lock:
            counter[0] += 1
            if progress_callback:
                progress_callback(counter[0], len(universe), tkr)
        return ep, rea

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_one, t): t for t in universe}
        for fut in concurrent.futures.as_completed(futs):
            try:
                ep, rea = fut.result()
                if ep:  rep.append(ep)
                if rea: rrea.append(rea)
            except Exception:
                pass

    elapsed = round(time.time() - start, 1)
    df_ep   = pd.DataFrame(rep)  if rep  else pd.DataFrame()
    df_rea  = pd.DataFrame(rrea) if rrea else pd.DataFrame()
    stats   = {
        "total":      len(universe),
        "ep_found":   len(rep),
        "rea_found":  len(rrea),
        "elapsed_s":  elapsed,
        "cache_hits": cache_hits[0],
        "downloaded": len(universe) - cache_hits[0],
        "workers":    n_workers,
        "finviz":     finviz_enabled and FINVIZ_AVAILABLE,
    }
    return df_ep, df_rea, stats
