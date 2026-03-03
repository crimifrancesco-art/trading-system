import threading
import concurrent.futures
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ── Backend dati ─────────────────────────────────────────────────────────────
try:
    from yahooquery import Ticker as YQTicker
    _HAS_YQ = True
except ImportError:
    _HAS_YQ = False

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

_SCAN_ERRORS: list = []

# ── Download OHLCV ────────────────────────────────────────────────────────────
def _download_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """yahooquery (primario) → yfinance (fallback)"""
    if _HAS_YQ:
        try:
            df = YQTicker(ticker).history(period=period, interval="1d")
            if df is not None and not df.empty and len(df) >= 10:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)
                df.index = pd.to_datetime(df.index)
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if cl == "open":   col_map[c] = "Open"
                    elif cl == "high": col_map[c] = "High"
                    elif cl == "low":  col_map[c] = "Low"
                    elif cl in ("close", "adjclose", "adj close"): col_map[c] = "Close"
                    elif cl == "volume": col_map[c] = "Volume"
                df = df.rename(columns=col_map)
                needed = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in df.columns for c in needed):
                    df = df[needed].copy()
                    df["Volume"] = df["Volume"].fillna(0)
                    df = df.ffill().dropna(subset=["Close"])
                    if len(df) >= 10:
                        return df
        except Exception:
            pass

    if _HAS_YF:
        try:
            data = yf.download(ticker, period=period, progress=False,
                               auto_adjust=True, threads=False, timeout=30)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data["Volume"] = data["Volume"].fillna(0)
                data = data.ffill().dropna(subset=["Close"])
                if len(data) >= 10:
                    return data
        except Exception:
            pass
        try:
            data = yf.Ticker(ticker).history(period=period, timeout=30)
            if data is not None and not data.empty and len(data) >= 10:
                return data.ffill()
        except Exception:
            pass

    return pd.DataFrame()


def _get_info(ticker: str) -> dict:
    if _HAS_YQ:
        try:
            p = YQTicker(ticker).price
            if isinstance(p, dict) and ticker in p and isinstance(p[ticker], dict):
                d = p[ticker]
                return {
                    "name":       str(d.get("longName", d.get("shortName", ticker)))[:50],
                    "currency":   str(d.get("currency", "USD")),
                    "market_cap": float(d.get("marketCap", float("nan")))
                }
        except Exception:
            pass
    if _HAS_YF:
        try:
            info = yf.Ticker(ticker).info
            return {
                "name":       str(info.get("longName", info.get("shortName", ticker)))[:50],
                "currency":   str(info.get("currency", "USD")),
                "market_cap": float(info.get("marketCap", float("nan")))
            }
        except Exception:
            pass
    return {"name": ticker, "currency": "USD", "market_cap": float("nan")}


def _download_weekly(ticker: str) -> pd.DataFrame:
    if _HAS_YQ:
        try:
            df = YQTicker(ticker).history(period="6mo", interval="1wk")
            if df is not None and not df.empty:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)
                df.index = pd.to_datetime(df.index)
                for c in df.columns:
                    if c.lower() in ("close", "adjclose"):
                        df = df.rename(columns={c: "Close"}); break
                if "Close" in df.columns and len(df) >= 5:
                    return df
        except Exception:
            pass
    if _HAS_YF:
        try:
            data = yf.Ticker(ticker).history(period="6mo", interval="1wk", timeout=15)
            if data is not None and not data.empty and len(data) >= 5:
                return data
        except Exception:
            pass
    return pd.DataFrame()


# ── Indicatori tecnici ────────────────────────────────────────────────────────
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


# ── CSV / Universe ────────────────────────────────────────────────────────────
def load_index_from_csv(filename: str):
    for base in [Path("data"), Path(".")]:
        path = base / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in ["Simbolo", "simbolo", "ticker", "Ticker",
                             "Symbol", "symbol", "TICKER", "SYMBOL"]:
                    if col in df.columns:
                        tickers = df[col].dropna().astype(str).str.strip().unique().tolist()
                        return [t for t in tickers if t and len(t) <= 12 and not t.isdigit()]
            except Exception:
                pass
    return []

_CURRENCY_SUFFIX = {
    "GBX": ".L",  "GBP": ".L",  "CHF": ".SW", "SEK": ".ST",
    "DKK": ".CO", "NOK": ".OL", "PLN": ".WA", "HKD": ".HK",
    "INR": ".NS", "KRW": ".KS", "TWD": ".TW", "MXN": ".MX",
    "BRL": ".SA", "IDR": ".JK", "THB": ".BK", "ZAC": ".JO",
    "HUF": ".BD", "CNY": ".SS", "CNH": ".SS",
}
_US_LISTED = {"ASML","AZN","SAP","NVO","HSBC","SHELL","RDS","UBS","CS","SAN","BBVA","ING","AEG"}

def _add_suffix(ticker: str, currency: str, market: str) -> str:
    if ticker in _US_LISTED: return ticker
    if market == "FTSE": return ticker + ".MI"
    return ticker + _CURRENCY_SUFFIX[currency] if currency in _CURRENCY_SUFFIX else ticker

def load_universe(markets: list) -> list:
    tickers = []
    if "SP500"      in markets: tickers += load_index_from_csv("sp500.csv")
    if "Nasdaq"     in markets: tickers += load_index_from_csv("nasdaq100.csv")
    if "Dow"        in markets: tickers += load_index_from_csv("dowjones.csv")
    if "Russell"    in markets: tickers += load_index_from_csv("russell2000.csv")
    if "USSmallCap" in markets:
        for fname in ["us_small_cap_2000.csv", "us small cap 2000.csv"]:
            t = load_index_from_csv(fname)
            if t: tickers += t; break
    if "FTSE" in markets:
        for raw in load_index_from_csv("ftsemib.csv"):
            tickers.append(raw + ".MI")
    if "Eurostoxx" in markets:
        for fname in ["eurostoxx600.csv"]:
            for path in [Path("data") / fname, Path(".") / fname]:
                if path.exists():
                    df = pd.read_csv(path)
                    for _, row in df.iterrows():
                        tkr = str(row.get("Simbolo", "")).strip()
                        cur = str(row.get("Prezzo - Valuta", "")).strip()
                        if not tkr or tkr.isdigit() or len(tkr) > 12: continue
                        tickers.append(_add_suffix(tkr, cur, "Eurostoxx"))
                    break
    if "StoxxEmerging" in markets:
        for fname in ["stoxx_emerging_market_50.csv", "stoxx emerging market 50.csv"]:
            for path in [Path("data") / fname, Path(".") / fname]:
                if path.exists():
                    df = pd.read_csv(path)
                    for _, row in df.iterrows():
                        tkr = str(row.get("Simbolo", "")).strip()
                        cur = str(row.get("Prezzo - Valuta", "")).strip()
                        if not tkr or tkr.isdigit() or len(tkr) > 12: continue
                        tickers.append(_add_suffix(tkr, cur, "StoxxEmerging"))
                    break
    return list(dict.fromkeys(tickers))


# ── Scan singolo ticker ───────────────────────────────────────────────────────
def scan_ticker(ticker: str, e_h: float, p_rmin: int, p_rmax: int,
                r_poc: float, vol_ratio_hot: float = 1.5):
    try:
        data = _download_ohlcv(ticker)
        if data is None or data.empty or len(data) < 10:
            return None, None

        c = data["Close"]; h = data["High"]; l = data["Low"]; v = data["Volume"]
        price = float(c.iloc[-1])
        if price <= 0: return None, None

        info       = _get_info(ticker)
        name       = info["name"]
        currency   = info["currency"]
        market_cap = info["market_cap"]

        vol_today  = float(v.iloc[-1]) if not v.empty else 0.0
        avg_vol_20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
        vol_7d_avg = float(v.tail(7).mean())
        vol_ratio  = float(vol_today / avg_vol_20) if avg_vol_20 > 0 else 0.0

        ema20    = float(c.ewm(span=20).mean().iloc[-1])
        ema50    = float(c.ewm(span=50).mean().iloc[-1])
        sma200_s = c.rolling(200).mean()
        ema200   = float(sma200_s.iloc[-1]) if len(sma200_s.dropna()) > 0 else None

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

        tail60    = data.tail(60).copy()
        ema20_ser = c.ewm(span=20).mean()
        ema50_ser = c.ewm(span=50).mean()
        bb_up, _, bb_dn = calc_bollinger(c)
        open_col  = tail60["Open"] if "Open" in tail60.columns else tail60["Close"]
        chart_data = {
            "dates":  tail60.index.strftime("%Y-%m-%d").tolist(),
            "open":   [round(float(x), 2) for x in open_col],
            "high":   [round(float(x), 2) for x in tail60["High"]],
            "low":    [round(float(x), 2) for x in tail60["Low"]],
            "close":  [round(float(x), 2) for x in tail60["Close"]],
            "volume": [int(x)             for x in tail60["Volume"]],
            "ema20":  [round(float(x), 2) for x in ema20_ser.tail(60)],
            "ema50":  [round(float(x), 2) for x in ema50_ser.tail(60)],
            "bb_up":  [round(float(x), 2) for x in bb_up.tail(60)],
            "bb_dn":  [round(float(x), 2) for x in bb_dn.tail(60)],
        }

        # ── Scoring ──────────────────────────────────────────────────────
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
            poc       = float(vp.idxmax())
            dist_poc  = abs(price - poc) / poc
            if dist_poc < r_poc and vol_ratio > vol_ratio_hot:
                rea_score, stato_rea = 7, "HOT"
        except Exception:
            pass

        ser_score = sum([rsi_val > 50, price > ema20, ema20 > ema50,
                         obv_trend == "UP", vol_ratio > 1.0, True])
        ser_ok    = all([rsi_val > 50, price > ema20, ema20 > ema50,
                         obv_trend == "UP", vol_ratio > 1.0, True])
        f1, f2, f3 = price > 10, avg_vol_20 > 500_000, vol_ratio > 1.0
        f4, f5     = price > ema20, price > ema50
        fv_score   = sum([f1, f2, f3, f4, f5])
        fv_ok      = all([f1, f2, f3, f4, f5])

        common = {
            "Nome": name, "Ticker": ticker, "Prezzo": round(price, 2),
            "MarketCap":  market_cap,
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


# ── Scan universe ─────────────────────────────────────────────────────────────
def scan_universe(universe: list, e_h, p_rmin, p_rmax, r_poc,
                  vol_ratio_hot=1.5, cache_enabled=True, finviz_enabled=False,
                  n_workers=8, progress_callback=None):
    global _SCAN_ERRORS
    _SCAN_ERRORS = []
    rep, rrea = [], []
    lock    = threading.Lock()
    counter = [0]
    t0      = time.time()
    tot     = len(universe)

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
                with lock:
                    if ep:  rep.append(ep)
                    if rea: rrea.append(rea)
            except Exception:
                pass

    df_ep  = pd.DataFrame(rep)  if rep  else pd.DataFrame()
    df_rea = pd.DataFrame(rrea) if rrea else pd.DataFrame()
    return df_ep, df_rea, {
        "elapsed_s":  round(time.time() - t0, 1),
        "cache_hits": 0,
        "downloaded": tot,
        "workers":    nw,
        "total":      tot,
        "ep_found":   len(rep),
        "rea_found":  len(rrea),
        "finviz":     False,
        "errors":     _SCAN_ERRORS[:20],
        "n_errors":   len(_SCAN_ERRORS),
    }

