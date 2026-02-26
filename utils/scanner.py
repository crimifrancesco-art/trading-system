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
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    return tr.rolling(period).mean()


def calc_bollinger(close, period=20, std_dev=2):
    """Restituisce (banda_sup, media, banda_inf)."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ma + std_dev * std, ma, ma - std_dev * std


def calc_keltner(close, high, low, period=20, atr_mult=1.5):
    """Restituisce (banda_sup, media, banda_inf) Keltner Channels."""
    ema = close.ewm(span=period).mean()
    atr = calc_atr(high, low, close, period)
    return ema + atr_mult * atr, ema, ema - atr_mult * atr


def detect_squeeze(close, high, low):
    """True se le Bande di Bollinger sono dentro le Keltner Channels (Squeeze)."""
    bb_up, _, bb_dn = calc_bollinger(close)
    kc_up, _, kc_dn = calc_keltner(close, high, low)
    return bool(bb_up.iloc[-1] < kc_up.iloc[-1] and bb_dn.iloc[-1] > kc_dn.iloc[-1])


def detect_rsi_divergence(close, rsi_series, lookback=20):
    """
    Bearish divergence: prezzo fa nuovo max ma RSI no.
    Bullish divergence: prezzo fa nuovo min ma RSI no.
    Restituisce 'BEARISH', 'BULLISH', o None.
    """
    c = close.tail(lookback)
    r = rsi_series.tail(lookback)
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.9:
        return "BEARISH"
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.1:
        return "BULLISH"
    return None


def calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val):
    """
    Score di qualità composito 0-12:
    - Vol_Ratio > 1.5  → +2
    - OBV_Trend UP     → +2
    - ATR expansion    → +1
    - RSI in [45-65]   → +3
    - Prezzo > EMA20   → +2
    - Prezzo > EMA50   → +2
    """
    score = 0
    if vol_ratio > 1.5:
        score += 2
    if obv_trend == "UP":
        score += 2
    if atr_expansion:
        score += 1
    if 45 <= rsi_val <= 65:
        score += 3
    if price > ema20:
        score += 2
    if price > ema50:
        score += 2
    return score


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


# -------------------------------------------------------------------------
# SCAN TICKER — v21.0
# -------------------------------------------------------------------------

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot=1.5):
    """
    Analizza un singolo ticker.
    Novità v21.0:
    - Early_Score continuo (0-10) invece di binario
    - EMA50 per Quality_Score
    - Squeeze Bollinger/Keltner
    - Divergenza RSI/Prezzo
    - Quality_Score composito
    - Multi-timeframe: trend settimanale allineato
    Restituisce (res_ep, res_rea) o (None, None).
    """
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data) < 50:
            return None, None

        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]

        info = yf.Ticker(ticker).info
        name = info.get("longName", info.get("shortName", ticker))[:50]
        price = float(c.iloc[-1])
        currency = info.get("currency", "USD")
        market_cap = info.get("marketCap", np.nan)
        vol_today = float(v.iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())

        # EMA20, EMA50
        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        ema50 = float(c.ewm(span=50).mean().iloc[-1])

        # RSI
        rsi_series = calc_rsi(c)
        rsi_val = float(rsi_series.iloc[-1])

        # Vol Ratio (rispetto a media 20gg)
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])

        # OBV
        obv = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"

        # ATR
        atr_series = calc_atr(h, l, c)
        atr_val = float(atr_series.iloc[-1])
        atr_expansion = (atr_val / atr_series.rolling(50).mean().iloc[-1]) > 1.2

        # Squeeze
        in_squeeze = detect_squeeze(c, h, l)

        # Divergenza RSI
        rsi_divergence = detect_rsi_divergence(c, rsi_series)

        # Quality Score composito
        quality_score = calc_quality_score(
            price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val
        )

        # Multi-Timeframe: trend settimanale
        try:
            data_w = yf.Ticker(ticker).history(period="6mo", interval="1wk")
            if len(data_w) >= 5:
                c_w = data_w["Close"]
                ema20_w = float(c_w.ewm(span=20).mean().iloc[-1])
                weekly_bullish = float(c_w.iloc[-1]) > ema20_w
            else:
                weekly_bullish = None
        except Exception:
            weekly_bullish = None

        # ---------------------------------------------------------------
        # EARLY — prezzo vicino all'EMA20 (score CONTINUO 0-10)
        # ---------------------------------------------------------------
        dist_ema = abs(price - ema20) / ema20
        if dist_ema < e_h:
            early_score = round(max(0.0, (1.0 - dist_ema / e_h) * 10.0), 1)
        else:
            early_score = 0.0
        stato_early = "EARLY" if early_score > 0 else "-"

        # ---------------------------------------------------------------
        # PRO — trend + RSI + volume
        # ---------------------------------------------------------------
        pro_score = 3 if price > ema20 else 0
        if p_rmin < rsi_val < p_rmax:
            pro_score += 3
        if vol_ratio > 1.2:
            pro_score += 2
        stato_pro = "PRO" if pro_score >= 8 else "-"

        # ---------------------------------------------------------------
        # REA-QUANT (Volume Profile / POC)
        # ---------------------------------------------------------------
        tp = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dist_poc = abs(price - poc) / poc
        rea_score = 7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0
        stato_rea = "HOT" if rea_score >= 7 else "-"

        # ---------------------------------------------------------------
        # Costruzione record
        # ---------------------------------------------------------------
        if stato_early == "-" and stato_pro == "-":
            res_ep = None
        else:
            stato_ep = stato_pro if stato_pro != "-" else stato_early
            res_ep = {
                "Nome": name,
                "Ticker": ticker,
                "Prezzo": round(price, 2),
                "MarketCap": market_cap,
                "Vol_Today": int(vol_today),
                "Vol_7d_Avg": int(vol_7d_avg),
                "Currency": currency,
                "Early_Score": early_score,
                "Pro_Score": pro_score,
                "Quality_Score": quality_score,
                "RSI": round(rsi_val, 1),
                "Vol_Ratio": round(vol_ratio, 2),
                "OBV_Trend": obv_trend,
                "ATR": round(atr_val, 2),
                "ATR_Exp": atr_expansion,
                "Squeeze": in_squeeze,
                "RSI_Div": rsi_divergence if rsi_divergence else "-",
                "Weekly_Bull": weekly_bullish,
                "EMA50": round(ema50, 2),
                "Stato": stato_ep,
                "Stato_Early": stato_early,
                "Stato_Pro": stato_pro,
            }

        res_rea = None if stato_rea == "-" else {
            "Nome": name,
            "Ticker": ticker,
            "Prezzo": round(price, 2),
            "MarketCap": market_cap,
            "Vol_Today": int(vol_today),
            "Vol_7d_Avg": int(vol_7d_avg),
            "Currency": currency,
            "Rea_Score": rea_score,
            "Quality_Score": quality_score,
            "POC": round(poc, 2),
            "Dist_POC_%": round(dist_poc * 100, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "Stato": stato_rea,
            "Pro_Score": pro_score,
            "RSI": round(rsi_val, 1),
            "OBV_Trend": obv_trend,
            "ATR": round(atr_val, 2),
            "ATR_Exp": atr_expansion,
            "Squeeze": in_squeeze,
            "RSI_Div": rsi_divergence if rsi_divergence else "-",
            "Weekly_Bull": weekly_bullish,
        }

        return res_ep, res_rea

    except Exception:
        return None, None

