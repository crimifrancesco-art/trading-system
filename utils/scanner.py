# -*- coding: utf-8 -*-
"""
scanner.py — Trading Scanner PRO 28.0
ARCHITETTURA: sequenziale (come v22 che funzionava) + API Yahoo diretta
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import requests as _req
    _HAS_REQ = True
except ImportError:
    _HAS_REQ = False

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

_SCAN_ERRORS: list = []

# ── Yahoo Finance API v8 diretta (requests puro) ──────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}

# Session globale — riusa connessioni HTTP, più veloce e meno blocchi
_SESSION = None

_CRUMB: str = ""

def _init_crumb(sess) -> str:
    """Ottieni crumb Yahoo Finance — necessario per API autenticate."""
    global _CRUMB
    if _CRUMB:
        return _CRUMB
    try:
        # Step 1: visita la homepage per ottenere cookie
        sess.get("https://finance.yahoo.com", timeout=10)
        # Step 2: ottieni crumb
        r = sess.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
        if r.status_code == 200 and r.text and r.text != "":
            _CRUMB = r.text.strip()
    except Exception:
        pass
    return _CRUMB

def _get_session():
    global _SESSION
    if _SESSION is None and _HAS_REQ:
        _SESSION = _req.Session()
        _SESSION.headers.update(_HEADERS)
        _init_crumb(_SESSION)
    return _SESSION


# Cache meta globale: {ticker: {name, currency, market_cap}}
_META_CACHE: dict = {}

def _yahoo_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Chiama Yahoo Finance API v8. Ritorna DataFrame OHLCV.
    Come effetto collaterale salva meta in _META_CACHE[ticker].
    """
    sess = _get_session()
    if sess is None:
        return pd.DataFrame()
    try:
        for host in ["query2", "query1"]:
            url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
            chart_params = {"interval": interval, "range": period, "includePrePost": "false"}
            if _CRUMB:
                chart_params["crumb"] = _CRUMB
            resp = sess.get(url, params=chart_params, timeout=25)
            if resp.status_code == 200:
                break
        else:
            return pd.DataFrame()

        data   = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()

        r    = result[0]
        ts   = r.get("timestamp", [])
        q    = r.get("indicators", {}).get("quote", [{}])[0]
        adjc = r.get("indicators", {}).get("adjclose", [{}])

        if not ts or not q:
            return pd.DataFrame()

        # ── Estrai meta da chart response (sempre disponibile) ──────────
        meta = r.get("meta", {})
        _name = (meta.get("longName") or meta.get("shortName") or ticker)[:50]
        _curr = meta.get("currency", "USD") or "USD"
        _mcap = meta.get("regularMarketCap") or meta.get("marketCap")
        # Aggiorna cache: NON sovrascrivere market_cap se già presente e valido
        _existing = _META_CACHE.get(ticker, {})
        _mcap_final = float(_mcap) if _mcap else float("nan")
        _existing_mcap = _existing.get("market_cap", float("nan"))
        _keep_mcap = (isinstance(_existing_mcap, float)
                      and _existing_mcap == _existing_mcap  # not nan
                      and _existing_mcap > 0)
        _META_CACHE[ticker] = {
            "name":       str(_name),
            "currency":   str(_curr),
            "market_cap": _existing_mcap if _keep_mcap else _mcap_final,
        }

        closes = (adjc[0].get("adjclose") if adjc and adjc[0] else None) or q.get("close", [])
        opens  = q.get("open",   [None] * len(ts))
        highs  = q.get("high",   [None] * len(ts))
        lows   = q.get("low",    [None] * len(ts))
        vols   = q.get("volume", [0]    * len(ts))

        idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
        df  = pd.DataFrame({"Open": opens, "High": highs,
                             "Low": lows,  "Close": closes,
                             "Volume": vols}, index=idx)
        df  = df.dropna(subset=["Close"])
        df["Volume"] = df["Volume"].fillna(0).astype(float)
        df  = df.ffill()
        return df if len(df) >= 10 else pd.DataFrame()

    except Exception:
        return pd.DataFrame()


def _raw_val(v):
    """Estrai valore numerico da {raw, fmt} o numero diretto. 0/None → None."""
    r = v.get("raw") if isinstance(v, dict) else v
    return r if r else None


def fetch_bulk_meta(tickers: list) -> dict:
    """Yahoo Finance /v7/finance/quote — nome+currency+marketCap per tutti i ticker.
    Una sola chiamata HTTP per batch. Molto più affidabile di v10/quoteSummary.
    """
    if not tickers:
        return {}
    sess = _get_session()
    if sess is None:
        return {}
    result = {}
    for i in range(0, len(tickers), 80):
        batch = tickers[i:i+80]
        for host in ["query2", "query1"]:
            try:
                params = {"symbols": ",".join(batch),
                          "fields": "longName,shortName,currency,marketCap,regularMarketPrice"}
                if _CRUMB:
                    params["crumb"] = _CRUMB
                resp = sess.get(f"https://{host}.finance.yahoo.com/v7/finance/quote",
                                params=params, timeout=20)
                if resp.status_code != 200:
                    continue
                for q in resp.json().get("quoteResponse", {}).get("result", []):
                    tkr  = q.get("symbol", "")
                    if not tkr:
                        continue
                    name = q.get("longName") or q.get("shortName") or tkr
                    mcap = q.get("marketCap") or 0
                    result[tkr] = {
                        "name":       str(name)[:50],
                        "currency":   str(q.get("currency", "USD") or "USD"),
                        "market_cap": float(mcap) if mcap and mcap > 0 else float("nan"),
                    }
                break
            except Exception:
                continue
    return result


def _fetch_meta_v10(ticker: str) -> dict:
    """MarketCap e metadati via yfinance fast_info (primary) + quoteSummary (fallback)."""
    # ── 1. yfinance fast_info: usa internamente API v8 chart, molto affidabile ──
    if _HAS_YF:
        try:
            fi   = yf.Ticker(ticker).fast_info
            mcap = getattr(fi, "market_cap", None)
            curr = getattr(fi, "currency", "USD") or "USD"
            if mcap and mcap > 0:
                return {
                    "name":       "",           # fast_info non ha longName
                    "currency":   str(curr),
                    "market_cap": float(mcap),
                }
        except Exception:
            pass

    # ── 2. Fallback: v10/quoteSummary (può essere bloccato su Streamlit Cloud) ──
    sess = _get_session()
    if sess is None:
        return {}
    for host in ["query2", "query1"]:
        for ver in ["v10", "v6"]:
            try:
                url  = f"https://{host}.finance.yahoo.com/{ver}/finance/quoteSummary/{ticker}"
                params_v = {"modules": "price,defaultKeyStatistics"}
                if _CRUMB:
                    params_v["crumb"] = _CRUMB
                resp = sess.get(url, params=params_v, timeout=10)
                if resp.status_code != 200:
                    continue
                result = resp.json().get("quoteSummary", {}).get("result", [])
                if not result:
                    continue
                d         = result[0]
                price_mod = d.get("price", {})
                stats_mod = d.get("defaultKeyStatistics", {})
                name = price_mod.get("longName") or price_mod.get("shortName") or ""
                curr = price_mod.get("currency") or "USD"
                mcap = _raw_val(price_mod.get("marketCap"))
                if not mcap:
                    shares    = _raw_val(stats_mod.get("sharesOutstanding"))
                    reg_price = _raw_val(price_mod.get("regularMarketPrice"))
                    if shares and reg_price:
                        mcap = float(shares) * float(reg_price)
                if mcap and mcap > 0:
                    return {
                        "name":       str(name)[:50] if name else "",
                        "currency":   str(curr),
                        "market_cap": float(mcap),
                    }
            except Exception:
                continue
    return {}
def _download_ohlcv_meta(ticker: str, period: str = "6mo") -> tuple:
    """Ritorna (DataFrame OHLCV, dict meta con name/currency/market_cap).
    Priorità: chart v8 meta → quoteSummary v10 → yfinance
    """
    df = _yahoo_ohlcv(ticker, period=period, interval="1d")

    # Prendi quello che il chart v8 ha già messo in cache
    meta = _META_CACHE.get(ticker, {
        "name": ticker, "currency": "USD", "market_cap": float("nan")
    })

    _name_ok = meta.get("name","") and meta["name"] != ticker
    _mcap_v  = meta.get("market_cap", float("nan"))
    _mcap_ok = (isinstance(_mcap_v, float) and _mcap_v == _mcap_v and _mcap_v > 0)

    # Se nome o marketcap mancano, chiama v10/quoteSummary
    if not _name_ok or not _mcap_ok:
        v10 = _fetch_meta_v10(ticker)
        if v10:
            if not _name_ok and v10.get("name"):
                meta["name"] = v10["name"]
            if not _mcap_ok and v10.get("market_cap") and float(v10["market_cap"]) > 0:
                meta["market_cap"] = v10["market_cap"]
            if v10.get("currency"):
                meta["currency"] = v10["currency"]
            _META_CACHE[ticker] = meta
            _name_ok = bool(meta["name"] and meta["name"] != ticker)
            _mcap_ok = meta["market_cap"] == meta["market_cap"]

    # Ultimo fallback: yfinance
    if (not _name_ok or not _mcap_ok) and _HAS_YF:
        try:
            info = yf.Ticker(ticker).info
            if not _name_ok:
                name = info.get("longName") or info.get("shortName") or meta["name"]
                meta["name"] = str(name)[:50]
            if not _mcap_ok:
                mcap = info.get("marketCap") or info.get("enterpriseValue")
                meta["market_cap"] = float(mcap) if mcap else float("nan")
            if info.get("currency"):
                meta["currency"] = str(info["currency"])
            _META_CACHE[ticker] = meta
        except Exception:
            pass

    # Rileggi da cache per ottenere market_cap da bulk se ancora mancante
    if not _mcap_ok:
        _cached = _META_CACHE.get(ticker, {})
        _cv = _cached.get("market_cap", float("nan"))
        if isinstance(_cv, float) and _cv == _cv and _cv > 0:
            meta["market_cap"] = _cv
    return df, meta


def _yahoo_info(ticker: str) -> dict:
    """Metadati ticker — usa _META_CACHE popolata da _yahoo_ohlcv (gratuito, nessuna chiamata extra)."""
    # Prima controlla cache (popolata dalla chiamata chart già fatta)
    if ticker in _META_CACHE:
        return _META_CACHE[ticker]
    # Fallback: prova API v8 chart con range minimo solo per i meta
    df = _yahoo_ohlcv(ticker, period="5d", interval="1d")
    if ticker in _META_CACHE:
        return _META_CACHE[ticker]
    return {"name": ticker, "currency": "USD", "market_cap": float("nan")}


def _download_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """API diretta → yfinance fallback."""
    df = _yahoo_ohlcv(ticker, period=period, interval="1d")
    if not df.empty:
        return df
    if _HAS_YF:
        try:
            data = yf.Ticker(ticker).history(period=period, timeout=20, auto_adjust=True)
            if data is not None and not data.empty and len(data) >= 10:
                return data.ffill()
        except Exception:
            pass
    return pd.DataFrame()


def _download_weekly(ticker: str) -> pd.DataFrame:
    df = _yahoo_ohlcv(ticker, period="6mo", interval="1wk")
    if not df.empty:
        return df
    if _HAS_YF:
        try:
            data = yf.Ticker(ticker).history(period="6mo", interval="1wk", timeout=12)
            if data is not None and not data.empty:
                return data
        except Exception:
            pass
    return pd.DataFrame()


# ── Indicatori tecnici ─────────────────────────────────────────────────────────
def calc_obv(close, volume):
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = -delta.where(delta < 0, 0).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    tr = np.maximum(high - low,
         np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    return tr.rolling(period).mean()

def calc_bollinger(close, period=20, std_dev=2):
    ma  = close.rolling(period).mean()
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
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.9: return "BEARISH"
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.1: return "BULLISH"
    return None

def calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    s = 0
    if vol_ratio > 1.5:     s += 2
    if obv_trend == "UP":   s += 2
    if atr_expansion:       s += 1
    if 45 <= rsi_val <= 65: s += 3
    if price > ema20:       s += 2
    if price > ema50:       s += 2
    return s

def calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    return {
        "Vol_Ratio":  min(vol_ratio / 3.0, 1.0),
        "OBV":        1.0 if obv_trend == "UP" else 0.0,
        "ATR_Exp":    1.0 if atr_expansion else 0.0,
        "RSI Zone":   max(0.0, 1.0 - abs(rsi_val - 55) / 25.0),
        "EMA20 Bull": 1.0 if price > ema20 else 0.0,
        "EMA50 Bull": 1.0 if price > ema50 else 0.0,
    }


# ── Universe ───────────────────────────────────────────────────────────────────
def load_index_from_csv(filename: str):
    for base in [Path("data"), Path(".")]:
        path = base / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in ["Simbolo","simbolo","ticker","Ticker",
                             "Symbol","symbol","TICKER","SYMBOL"]:
                    if col in df.columns:
                        tickers = df[col].dropna().astype(str).str.strip().unique().tolist()
                        return [t for t in tickers if t and len(t) <= 12 and not t.isdigit()]
            except Exception:
                pass
    return []

_CURRENCY_SUFFIX = {
    "GBX":".L","GBP":".L","CHF":".SW","SEK":".ST","DKK":".CO","NOK":".OL",
    "PLN":".WA","HKD":".HK","INR":".NS","KRW":".KS","TWD":".TW","MXN":".MX",
    "BRL":".SA","IDR":".JK","THB":".BK","ZAC":".JO","HUF":".BD",
    "CNY":".SS","CNH":".SS",
}
_US_LISTED = {"ASML","AZN","SAP","NVO","HSBC","SHELL","RDS","UBS","CS",
              "SAN","BBVA","ING","AEG"}

def _add_suffix(ticker, currency, market):
    if ticker in _US_LISTED: return ticker
    if market == "FTSE":     return ticker + ".MI"
    return ticker + _CURRENCY_SUFFIX[currency] if currency in _CURRENCY_SUFFIX else ticker

def load_universe(markets: list) -> list:
    tickers = []
    if "SP500"      in markets: tickers += load_index_from_csv("sp500.csv")
    if "Nasdaq"     in markets: tickers += load_index_from_csv("nasdaq100.csv")
    if "Dow"        in markets: tickers += load_index_from_csv("dowjones.csv")
    if "Russell"    in markets: tickers += load_index_from_csv("russell2000.csv")
    if "USSmallCap" in markets:
        for fname in ["us_small_cap_2000.csv","us small cap 2000.csv"]:
            t = load_index_from_csv(fname); 
            if t: tickers += t; break
    if "FTSE" in markets:
        for raw in load_index_from_csv("ftsemib.csv"):
            tickers.append(raw + ".MI")
    if "Eurostoxx" in markets:
        for fname in ["eurostoxx600.csv"]:
            for path in [Path("data")/fname, Path(".")/fname]:
                if path.exists():
                    df = pd.read_csv(path)
                    for _, row in df.iterrows():
                        tkr = str(row.get("Simbolo","")).strip()
                        cur = str(row.get("Prezzo - Valuta","")).strip()
                        if not tkr or tkr.isdigit() or len(tkr)>12: continue
                        tickers.append(_add_suffix(tkr, cur, "Eurostoxx"))
                    break
    if "StoxxEmerging" in markets:
        for fname in ["stoxx_emerging_market_50.csv","stoxx emerging market 50.csv"]:
            for path in [Path("data")/fname, Path(".")/fname]:
                if path.exists():
                    df = pd.read_csv(path)
                    for _, row in df.iterrows():
                        tkr = str(row.get("Simbolo","")).strip()
                        cur = str(row.get("Prezzo - Valuta","")).strip()
                        if not tkr or tkr.isdigit() or len(tkr)>12: continue
                        tickers.append(_add_suffix(tkr, cur, "StoxxEmerging"))
                    break
    return list(dict.fromkeys(tickers))


# ── Scan singolo ticker ────────────────────────────────────────────────────────
def scan_ticker(ticker: str, e_h: float, p_rmin: int, p_rmax: int,
                r_poc: float, vol_ratio_hot: float = 1.5):
    try:
        data, _meta = _download_ohlcv_meta(ticker)
        if data is None or data.empty or len(data) < 10:
            return None, None

        c = data["Close"]; h = data["High"]; l = data["Low"]; v = data["Volume"]
        price = float(c.iloc[-1])
        if price <= 0: return None, None

        name       = _meta["name"]
        currency   = _meta["currency"]
        market_cap = _meta["market_cap"]

        vol_today  = float(v.iloc[-1]) if not v.empty else 0.0
        avg_vol_20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
        vol_7d_avg = float(v.tail(7).mean())
        vol_ratio  = float(vol_today / avg_vol_20) if avg_vol_20 > 0 else 0.0

        ema20    = float(c.ewm(span=20).mean().iloc[-1])
        ema50    = float(c.ewm(span=50).mean().iloc[-1])
        # EMA200 con span adattivo — funziona anche con solo 126 barre (6 mesi)
        _ema200_s = c.ewm(span=min(200, len(c)), adjust=False).mean()
        ema200    = float(_ema200_s.iloc[-1]) if not _ema200_s.empty else None

        rsi_series    = calc_rsi(c)
        rsi_val       = float(rsi_series.iloc[-1])
        obv           = calc_obv(c, v)
        obv_slope     = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend     = "UP" if obv_slope > 0 else "DOWN"
        atr_series    = calc_atr(h, l, c)
        atr_val       = float(atr_series.iloc[-1])
        atr_50        = atr_series.rolling(50).mean().iloc[-1]
        atr_expansion = bool(atr_val / atr_50 > 1.2) if (atr_50 and not np.isnan(float(atr_50))) else False
        in_squeeze    = detect_squeeze(c, h, l)
        rsi_div       = detect_rsi_divergence(c, rsi_series)
        quality_score = calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)
        quality_comps = calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)

        weekly_bullish = None
        try:
            dw = _download_weekly(ticker)
            if not dw.empty and "Close" in dw.columns and len(dw) >= 5:
                cw = dw["Close"]
                weekly_bullish = float(cw.iloc[-1]) > float(cw.ewm(span=20).mean().iloc[-1])
        except Exception:
            pass

        # Usa TUTTE le barre disponibili (max 252 = ~1 anno) per avere EMA200 significativa
        n_bars    = min(len(data), 252)
        tail_data = data.tail(n_bars).copy()
        ema20_ser = c.ewm(span=20).mean()
        ema50_ser = c.ewm(span=50).mean()
        ema200_ser = c.ewm(span=200, adjust=False).mean()
        bb_up, _, bb_dn = calc_bollinger(c)
        open_col = tail_data["Open"] if "Open" in tail_data.columns else tail_data["Close"]

        def _safe(series, n):
            vals = series.tail(n).tolist()
            return [round(float(v), 2) if v == v else None for v in vals]

        chart_data = {
            "dates":  tail_data.index.strftime("%Y-%m-%d").tolist(),
            "open":   _safe(open_col, n_bars),
            "high":   _safe(tail_data["High"], n_bars),
            "low":    _safe(tail_data["Low"],  n_bars),
            "close":  _safe(tail_data["Close"], n_bars),
            "volume": [int(x) for x in tail_data["Volume"]],
            "ema20":  _safe(ema20_ser,  n_bars),
            "ema50":  _safe(ema50_ser,  n_bars),
            "ema200": _safe(ema200_ser, n_bars),
            "bb_up":  _safe(bb_up,     n_bars),
            "bb_dn":  _safe(bb_dn,     n_bars),
        }

        dist_ema    = abs(price - ema20) / ema20
        early_score = round(max(0.0, (1.0 - dist_ema / e_h) * 10.0), 1) if dist_ema < e_h else 0.0
        stato_early = "EARLY" if early_score > 0 else "-"

        pro_score = 0
        if price > ema20:             pro_score += 3
        if p_rmin < rsi_val < p_rmax: pro_score += 3
        if vol_ratio > 1.2:           pro_score += 2
        stato_pro = "PRO" if pro_score >= 4 else "-"

        poc, dist_poc, rea_score, stato_rea = price, 0.0, 0, "-"
        try:
            tp    = (h + l + c) / 3
            bins  = np.linspace(float(l.min()), float(h.max()), 50)
            pbins = pd.cut(tp, bins, labels=bins[:-1])
            vp    = pd.DataFrame({"P": pbins, "V": v}).groupby("P")["V"].sum()
            poc      = float(vp.idxmax())
            dist_poc = abs(price - poc) / poc
            if dist_poc < r_poc and vol_ratio > vol_ratio_hot:
                rea_score, stato_rea = 7, "HOT"
        except Exception:
            pass

        ser_score = sum([rsi_val>50, price>ema20, ema20>ema50,
                         obv_trend=="UP", vol_ratio>1.0, True])
        ser_ok    = all([rsi_val>50, price>ema20, ema20>ema50,
                         obv_trend=="UP", vol_ratio>1.0, True])
        f1,f2,f3  = price>10, avg_vol_20>500_000, vol_ratio>1.0
        f4,f5     = price>ema20, price>ema50
        fv_score  = sum([f1,f2,f3,f4,f5])
        fv_ok     = all([f1,f2,f3,f4,f5])

        common = {
            "Nome": name, "Ticker": ticker, "Prezzo": round(price, 2),
            "MarketCap":  market_cap if (market_cap and market_cap == market_cap and market_cap > 0) else float("nan"),
            "Vol_Today":  int(vol_today),
            "Vol_7d_Avg": int(vol_7d_avg),
            "Avg_Vol_20": int(avg_vol_20),
            "Rel_Vol":    round(vol_ratio, 2),
            "Currency":   currency,
            "RSI":        round(rsi_val, 1),
            "Vol_Ratio":  round(vol_ratio, 2),
            "OBV_Trend":  obv_trend,
            "ATR":        round(atr_val, 2),
            "ATR_Exp":    atr_expansion,
            "Squeeze":    in_squeeze,
            "RSI_Div":    rsi_div if rsi_div else "-",
            "Weekly_Bull":    weekly_bullish,
            "EMA20":      round(ema20, 2),
            "EMA50":      round(ema50, 2),
            "EMA200":     round(ema200, 2) if ema200 else None,
            "Quality_Score":  quality_score,
            "Ser_OK":     ser_ok,
            "Ser_Score":  ser_score,
            "FV_OK":      fv_ok,
            "FV_Score":   fv_score,
            "_quality_components": quality_comps,
            "_chart_data":         chart_data,
        }
        res_ep = {
            **common,
            "Early_Score": early_score,
            "Pro_Score":   pro_score,
            "Stato":       stato_pro if stato_pro != "-" else stato_early,
            "Stato_Early": stato_early,
            "Stato_Pro":   stato_pro,
        }
        res_rea = None
        if stato_rea != "-":
            res_rea = {
                **common,
                "Rea_Score":  rea_score,
                "POC":        round(poc, 2),
                "Dist_POC_%": round(dist_poc * 100, 1),
                "Pro_Score":  pro_score,
                "Stato":      stato_rea,
            }
        return res_ep, res_rea

    except Exception as _e:
        _SCAN_ERRORS.append(f"{ticker}: {type(_e).__name__}: {_e}")
        return None, None


# ── scan_universe: SEQUENZIALE con callback ────────────────────────────────────
def scan_universe(universe: list, e_h, p_rmin, p_rmax, r_poc,
                  vol_ratio_hot=1.5, cache_enabled=True, finviz_enabled=False,
                  n_workers=8, progress_callback=None):
    """Loop sequenziale — più stabile su Streamlit Cloud di ThreadPoolExecutor."""
    global _SCAN_ERR
    # Pre-carica nome+marketcap per TUTTI i ticker in una sola chiamata bulk
    try:
        _bulk = fetch_bulk_meta(universe)
        for _tkr, _m in _bulk.items():
            if _tkr not in _META_CACHE:
                _META_CACHE[_tkr] = _m
            else:
                if not (_META_CACHE[_tkr].get("market_cap", 0) > 0):
                    _META_CACHE[_tkr]["market_cap"] = _m["market_cap"]
                if _META_CACHE[_tkr].get("name", _tkr) == _tkr and _m.get("name", _tkr) != _tkr:
                    _META_CACHE[_tkr]["name"] = _m["name"]
    except Exception:
        passORS
    _SCAN_ERRORS = []
    rep, rrea = [], []
    t0  = time.time()
    tot = len(universe)

    for i, tkr in enumerate(universe, 1):
        ep, rea = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
        if ep:  rep.append(ep)
        if rea: rrea.append(rea)
        if progress_callback:
            progress_callback(i, tot, tkr)

    df_ep  = pd.DataFrame(rep)  if rep  else pd.DataFrame()
    df_rea = pd.DataFrame(rrea) if rrea else pd.DataFrame()
    return df_ep, df_rea, {
        "elapsed_s":  round(time.time() - t0, 1),
        "cache_hits": 0,
        "downloaded": tot,
        "workers":    1,
        "total":      tot,
        "ep_found":   len(rep),
        "rea_found":  len(rrea),
        "finviz":     False,
        "errors":     _SCAN_ERRORS[:20],
        "n_errors":   len(_SCAN_ERRORS),
    }
