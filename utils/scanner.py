import numpy as np
import pandas as pd
import yfinance as yf
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
    # Rimuove duplicati mantenendo l'ordine
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

        # EARLY — prezzo vicino all'EMA20 (indipendente da PRO)
        dist_ema = abs(price - ema20) / ema20
        early_score = 8 if dist_ema < e_h else 0

        # PRO — trend + RSI nella finestra + volume sopra media
        pro_score = 3 if price > ema20 else 0
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        rsi_val = float(rsi.iloc[-1])
        if p_rmin < rsi_val < p_rmax:
            pro_score += 3
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
        if vol_ratio > 1.2:
            pro_score += 2

        # OBV
        obv = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"

        # ATR
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
        atr = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])
        atr_expansion = (atr_val / atr.rolling(50).mean().iloc[-1]) > 1.2

        # Stato EARLY e PRO sono ora INDIPENDENTI:
        # - EARLY: prezzo vicino EMA20 (early_score >= 8)
        # - PRO:   tutti i criteri trend+RSI+volume soddisfatti (pro_score >= 8)
        # Un ticker puo' soddisfare entrambe le condizioni
        stato_early = "EARLY" if early_score >= 8 else "-"
        stato_pro = "PRO" if pro_score >= 8 else "-"

        # REA-QUANT (Volume Profile / POC)
        tp = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dist_poc = abs(price - poc) / poc
        rea_score = 7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0
        stato_rea = "HOT" if rea_score >= 7 else "-"

        # res_ep: contiene sia EARLY sia PRO (usa il campo Stato per discriminare)
        # Restituisce None se nessun segnale EARLY o PRO
        if stato_early == "-" and stato_pro == "-":
            res_ep = None
        else:
            # Se il ticker e' sia EARLY sia PRO, usa PRO come stato principale
            # ma viene filtrato correttamente nel tab grazie a early_score e pro_score
            stato_ep = stato_pro if stato_pro != "-" else stato_early
            # Se e' entrambi, crea due record separati per far apparire in entrambi i tab
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
                "RSI": round(rsi_val, 1),
                "Vol_Ratio": round(vol_ratio, 2),
                "OBV_Trend": obv_trend,
                "ATR": round(atr_val, 2),
                "ATR_Exp": atr_expansion,
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
            "POC": round(poc, 2),
            "Dist_POC_%": round(dist_poc * 100, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "Stato": stato_rea,
            "Pro_Score": pro_score,
            "RSI": round(rsi_val, 1),
            "OBV_Trend": obv_trend,
            "ATR": round(atr_val, 2),
            "ATR_Exp": atr_expansion,
        }

        return res_ep, res_rea
    except Exception:
        return None, None