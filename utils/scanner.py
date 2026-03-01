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
    return bool(
        bb_up.iloc[-1] < kc_up.iloc[-1] and
        bb_dn.iloc[-1] > kc_dn.iloc[-1]
    )


def detect_rsi_divergence(close, rsi_series, lookback=20):
    c = close.tail(lookback)
    r = rsi_series.tail(lookback)
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.9:
        return "BEARISH"
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.1:
        return "BULLISH"
    return None


def calc_quality_score(price, ema20, ema50, vol_ratio, obv_trend,
                        atr_expansion, rsi_val):
    score = 0
    if vol_ratio > 1.5:       score += 2
    if obv_trend == "UP":     score += 2
    if atr_expansion:         score += 1
    if 45 <= rsi_val <= 65:   score += 3
    if price > ema20:         score += 2
    if price > ema50:         score += 2
    return score


def calc_quality_components(price, ema20, ema50, vol_ratio, obv_trend,
                              atr_expansion, rsi_val):
    return {
        "Vol_Ratio":  min(vol_ratio / 3.0, 1.0),
        "OBV":        1.0 if obv_trend == "UP" else 0.0,
        "ATR_Exp":    1.0 if atr_expansion else 0.0,
        "RSI Zone":   max(0.0, 1.0 - abs(rsi_val - 55) / 25.0),
        "EMA20 Bull": 1.0 if price > ema20 else 0.0,
        "EMA50 Bull": 1.0 if price > ema50 else 0.0,
    }


# -------------------------------------------------------------------------
# SERAFINI SCORE  (criteri Stefano Serafini)
# RSI > 50, prezzo > EMA20 > EMA50, OBV crescente,
# no earnings imminenti (entro 14 gg), Vol_Ratio crescente
# -------------------------------------------------------------------------

def calc_serafini_score(price, ema20, ema50, rsi_val, obv_trend,
                         vol_ratio, earnings_soon: bool) -> tuple:
    """
    Ritorna (serafini_score int 0-6, serafini_ok bool, criteri dict).
    Tutti e 5 i criteri devono essere True per serafini_ok=True.
    """
    c1 = rsi_val > 50
    c2 = price > ema20
    c3 = ema20  > ema50
    c4 = obv_trend == "UP"
    c5 = vol_ratio > 1.0
    c6 = not earnings_soon           # no earnings entro 14 gg

    score = sum([c1, c2, c3, c4, c5, c6])
    ok    = all([c1, c2, c3, c4, c5, c6])

    criteri = {
        "RSI>50":        c1,
        "P>EMA20":       c2,
        "EMA20>EMA50":   c3,
        "OBV_UP":        c4,
        "VolRatio>1":    c5,
        "No_Earnings":   c6,
    }
    return score, ok, criteri


# -------------------------------------------------------------------------
# FINVIZ-STYLE SCORE  (replica filtri Finviz da immagine)
# -------------------------------------------------------------------------

def calc_finviz_score(price, avg_vol_20, rel_vol, ema20, ema50, ema200,
                       eps_growth_next_y, eps_growth_5y,
                       optionable: bool) -> tuple:
    """
    Ritorna (finviz_score int 0-8, finviz_ok bool, criteri dict).
    """
    c1 = price > 10                          # Price > $10
    c2 = avg_vol_20 > 1_000_000              # Avg Volume > 1M
    c3 = rel_vol    > 1.0                    # Relative Volume > 1
    c4 = price > ema20  if ema20  else False # Price above 20-SMA
    c5 = price > ema50  if ema50  else False # Price above 50-SMA
    c6 = price > ema200 if ema200 else False # Price above 200-SMA
    c7 = (eps_growth_next_y is not None and
          eps_growth_next_y > 0.10)          # EPS Growth Next Year > 10%
    c8 = (eps_growth_5y is not None and
          eps_growth_5y > 0.15)              # EPS Growth Next 5Y > 15%
    # c_opt = optionable  (non filtrabile via yfinance, tenuto come info)

    score = sum([c1, c2, c3, c4, c5, c6, c7, c8])
    ok    = all([c1, c2, c3, c4, c5, c6])   # fondamentali opzionali

    criteri = {
        "Price>10":      c1,
        "AvgVol>1M":     c2,
        "RelVol>1":      c3,
        "P>SMA20":       c4,
        "P>SMA50":       c5,
        "P>SMA200":      c6,
        "EPS_NY>10%":    c7,
        "EPS_5Y>15%":    c8,
    }
    return score, ok, criteri


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
# SCAN TICKER — v27.0
# Novità: dati fondamentali, Serafini score, Finviz score,
#          SMA200, rel_vol, earnings_soon
# -------------------------------------------------------------------------

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot=1.5):
    try:
        tk   = yf.Ticker(ticker)
        data = tk.history(period="9mo")       # più lungo per SMA200
        if len(data) < 60:
            return None, None

        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]

        # ── Info base ────────────────────────────────────────────────────
        info       = tk.info
        name       = info.get("longName", info.get("shortName", ticker))[:50]
        price      = float(c.iloc[-1])
        currency   = info.get("currency", "USD")
        market_cap = info.get("marketCap", np.nan)
        vol_today  = float(v.iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())

        # ── Medie mobili ─────────────────────────────────────────────────
        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        ema50 = float(c.ewm(span=50).mean().iloc[-1])
        sma200_ser = c.rolling(200).mean()
        ema200 = float(sma200_ser.iloc[-1]) if not np.isnan(sma200_ser.iloc[-1]) else None

        # ── RSI ──────────────────────────────────────────────────────────
        rsi_series = calc_rsi(c)
        rsi_val    = float(rsi_series.iloc[-1])

        # ── Volume ───────────────────────────────────────────────────────
        avg_vol_20 = float(v.rolling(20).mean().iloc[-1])
        vol_ratio  = float(v.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 0.0
        rel_vol    = vol_ratio   # alias Finviz naming

        # ── OBV ──────────────────────────────────────────────────────────
        obv       = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"

        # ── ATR / Squeeze ─────────────────────────────────────────────────
        atr_series    = calc_atr(h, l, c)
        atr_val       = float(atr_series.iloc[-1])
        atr_expansion = (atr_val / atr_series.rolling(50).mean().iloc[-1]) > 1.2
        in_squeeze    = detect_squeeze(c, h, l)
        rsi_div       = detect_rsi_divergence(c, rsi_series)

        # ── Fondamentali da yfinance.info ────────────────────────────────
        # EPS growth
        eps_growth_ny = info.get("earningsGrowth", None)   # YoY (proxy)
        eps_fwd       = info.get("forwardEps", None)
        eps_trail     = info.get("trailingEps", None)
        if eps_fwd and eps_trail and eps_trail != 0:
            eps_growth_ny = (eps_fwd - eps_trail) / abs(eps_trail)

        # EPS Growth 5Y: yfinance non ha dato diretto → usa revenueGrowth come proxy
        eps_growth_5y = info.get("revenueGrowth", None)    # proxy (annuo)
        # Se disponibile, usa earningsQuarterlyGrowth moltiplicato (stima grossolana)
        eq_growth = info.get("earningsQuarterlyGrowth", None)
        if eq_growth is not None and eps_growth_5y is None:
            eps_growth_5y = eq_growth

        # Altri fondamentali
        pe_ratio      = info.get("trailingPE", None)
        fwd_pe        = info.get("forwardPE",  None)
        roe           = info.get("returnOnEquity", None)
        debt_eq       = info.get("debtToEquity", None)
        gross_margin  = info.get("grossMargins", None)
        op_margin     = info.get("operatingMargins", None)
        short_float   = info.get("shortPercentOfFloat", None)
        optionable    = bool(info.get("exchange", "") in
                             ["NMS","NYQ","ASE","PCX","CBT"])  # proxy

        # ── Earnings imminenti (entro 14 giorni) ─────────────────────────
        earnings_soon = False
        try:
            cal = tk.calendar
            if cal is not None and not cal.empty:
                ed = cal.get("Earnings Date")
                if ed is not None:
                    ed_val = pd.to_datetime(ed.iloc[0] if hasattr(ed,"iloc") else ed)
                    days_to = (ed_val.tz_localize(None) -
                               pd.Timestamp.now()).days
                    earnings_soon = 0 <= days_to <= 14
        except Exception:
            pass

        # ── Quality / Components ──────────────────────────────────────────
        quality_score = calc_quality_score(
            price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)
        quality_components = calc_quality_components(
            price, ema20, ema50, vol_ratio, obv_trend, atr_expansion, rsi_val)

        # ── Multi-timeframe ───────────────────────────────────────────────
        weekly_bullish = None
        try:
            dw     = tk.history(period="6mo", interval="1wk")
            cw     = dw["Close"]
            ema20w = float(cw.ewm(span=20).mean().iloc[-1]) if len(dw)>=5 else None
            weekly_bullish = float(cw.iloc[-1]) > ema20w if ema20w else None
        except Exception:
            pass

        # ── Serafini Score ────────────────────────────────────────────────
        ser_score, ser_ok, ser_criteri = calc_serafini_score(
            price, ema20, ema50, rsi_val, obv_trend, vol_ratio, earnings_soon)

        # ── Finviz Score ──────────────────────────────────────────────────
        fv_score, fv_ok, fv_criteri = calc_finviz_score(
            price, avg_vol_20, rel_vol, ema20, ema50, ema200,
            eps_growth_ny, eps_growth_5y, optionable)

        # ── Chart data (ultimi 60 gg) ─────────────────────────────────────
        tail60    = data.tail(60).copy()
        ema20_ser = c.ewm(span=20).mean()
        ema50_ser = c.ewm(span=50).mean()
        bb_up, bb_mid, bb_dn = calc_bollinger(c)

        chart_data = {
            "dates":  tail60.index.strftime("%Y-%m-%d").tolist(),
            "open":   [round(x, 2) for x in tail60["Open"].tolist()],
            "high":   [round(x, 2) for x in tail60["High"].tolist()],
            "low":    [round(x, 2) for x in tail60["Low"].tolist()],
            "close":  [round(x, 2) for x in tail60["Close"].tolist()],
            "volume": [int(x)      for x in tail60["Volume"].tolist()],
            "ema20":  [round(x, 2) for x in ema20_ser.tail(60).tolist()],
            "ema50":  [round(x, 2) for x in ema50_ser.tail(60).tolist()],
            "bb_up":  [round(x, 2) for x in bb_up.tail(60).tolist()],
            "bb_dn":  [round(x, 2) for x in bb_dn.tail(60).tolist()],
        }

        # ── Scoring esistente ─────────────────────────────────────────────
        dist_ema    = abs(price - ema20) / ema20
        early_score = round(
            max(0.0, (1.0 - dist_ema / e_h) * 10.0), 1
        ) if dist_ema < e_h else 0.0
        stato_early = "EARLY" if early_score > 0 else "-"

        pro_score = 3 if price > ema20 else 0
        if p_rmin < rsi_val < p_rmax: pro_score += 3
        if vol_ratio > 1.2:           pro_score += 2
        stato_pro = "PRO" if pro_score >= 8 else "-"

        tp   = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp   = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc  = float(vp.idxmax())
        dist_poc  = abs(price - poc) / poc
        rea_score = 7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0
        stato_rea = "HOT" if rea_score >= 7 else "-"

        # ── Record comune ─────────────────────────────────────────────────
        common = {
            # Base
            "Nome":        name,
            "Ticker":      ticker,
            "Prezzo":      round(price, 2),
            "MarketCap":   market_cap,
            "Vol_Today":   int(vol_today),
            "Vol_7d_Avg":  int(vol_7d_avg),
            "Currency":    currency,
            # Tecnici
            "RSI":         round(rsi_val, 1),
            "Vol_Ratio":   round(vol_ratio, 2),
            "Rel_Vol":     round(rel_vol,   2),
            "Avg_Vol_20":  int(avg_vol_20),
            "OBV_Trend":   obv_trend,
            "ATR":         round(atr_val, 2),
            "ATR_Exp":     atr_expansion,
            "Squeeze":     in_squeeze,
            "RSI_Div":     rsi_div if rsi_div else "-",
            "Weekly_Bull": weekly_bullish,
            "EMA20":       round(ema20, 2),
            "EMA50":       round(ema50, 2),
            "EMA200":      round(ema200, 2) if ema200 else None,
            "Quality_Score": quality_score,
            # Fondamentali
            "PE":          round(pe_ratio,   2) if pe_ratio   else None,
            "Fwd_PE":      round(fwd_pe,     2) if fwd_pe     else None,
            "ROE":         round(roe,        4) if roe        else None,
            "Debt_Eq":     round(debt_eq,    2) if debt_eq    else None,
            "Gross_Mgn":   round(gross_margin,4) if gross_margin else None,
            "Op_Mgn":      round(op_margin,  4) if op_margin  else None,
            "Short_Float": round(short_float,4) if short_float else None,
            "EPS_NY_Gr":   round(eps_growth_ny,4) if eps_growth_ny else None,
            "EPS_5Y_Gr":   round(eps_growth_5y,4) if eps_growth_5y else None,
            "Earnings_Soon": earnings_soon,
            "Optionable":  optionable,
            # Serafini
            "Ser_Score":   ser_score,
            "Ser_OK":      ser_ok,
            "_ser_criteri": ser_criteri,
            # Finviz
            "FV_Score":    fv_score,
            "FV_OK":       fv_ok,
            "_fv_criteri": fv_criteri,
            # Grafici
            "_quality_components": quality_components,
            "_chart_data":         chart_data,
        }

        if stato_early == "-" and stato_pro == "-":
            res_ep = None
        else:
            res_ep = {
                **common,
                "Early_Score": early_score, "Pro_Score": pro_score,
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
