# -*- coding: utf-8 -*-
"""
bluechip_dip.py  —  💎 Blue Chip Dip Screener  v31.1
══════════════════════════════════════════════════════
Monitora le 60 maggiori aziende mondiali per market cap.
Per ognuna calcola:
  • Drawdown % dal massimo 52 settimane
  • RSI(14) corrente
  • Distanza % da EMA200 (proxy oversold strutturale)
  • Quality Score tecnico
  • Trend EMA20/50/200
  • Volume anomalia (Vol_Ratio)

Logica "Dip Score" (0-100):
  40% drawdown da 52w high (più è profondo = opportunità)
  30% RSI < 45 (zona potenziale rimbalzo)
  30% prezzo vicino o sotto EMA200

Filtro: solo aziende con market cap > $50B e drawdown > 10%
══════════════════════════════════════════════════════
"""

import urllib.request, json, time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Palette TV ────────────────────────────────────
TV_BG     = "#131722"
TV_PANEL  = "#1e222d"
TV_BORDER = "#2a2e39"
TV_BLUE   = "#2962ff"
TV_GREEN  = "#26a69a"
TV_RED    = "#ef5350"
TV_GOLD   = "#ffd700"
TV_CYAN   = "#50c4e0"
TV_GRAY   = "#787b86"
TV_TEXT   = "#d1d4dc"
TV_ORANGE = "#ff9800"

# ── Universe: Top 100 Blue Chip globali ───────────
BLUE_CHIPS = [
    # USA Mega Cap
    ("AAPL",  "Apple"),           ("MSFT",  "Microsoft"),
    ("NVDA",  "NVIDIA"),          ("AMZN",  "Amazon"),
    ("GOOGL", "Alphabet A"),      ("META",  "Meta"),
    ("BRK-B", "Berkshire Hath."), ("LLY",   "Eli Lilly"),
    ("TSLA",  "Tesla"),           ("AVGO",  "Broadcom"),
    ("JPM",   "JPMorgan Chase"),  ("V",     "Visa"),
    ("MA",    "Mastercard"),      ("UNH",   "UnitedHealth"),
    ("XOM",   "ExxonMobil"),      ("JNJ",   "Johnson & Johnson"),
    ("WMT",   "Walmart"),         ("PG",    "Procter & Gamble"),
    ("ORCL",  "Oracle"),          ("HD",    "Home Depot"),
    ("COST",  "Costco"),          ("BAC",   "Bank of America"),
    ("NFLX",  "Netflix"),         ("KO",    "Coca-Cola"),
    ("CRM",   "Salesforce"),      ("AMD",   "AMD"),
    ("MRK",   "Merck"),           ("CVX",   "Chevron"),
    ("PEP",   "PepsiCo"),         ("ABBV",  "AbbVie"),
    ("TMO",   "Thermo Fisher"),   ("LIN",   "Linde"),
    ("ACN",   "Accenture"),       ("MCD",   "McDonald's"),
    ("PM",    "Philip Morris"),   ("GE",    "GE Aerospace"),
    ("NOW",   "ServiceNow"),      ("CAT",   "Caterpillar"),
    ("IBM",   "IBM"),             ("GS",    "Goldman Sachs"),
    ("AMGN",  "Amgen"),           ("T",     "AT&T"),
    ("MS",    "Morgan Stanley"),  ("AXP",   "American Express"),
    ("SPGI",  "S&P Global"),      ("BLK",   "BlackRock"),
    ("RTX",   "RTX Corp"),        ("HON",   "Honeywell"),
    ("DE",    "John Deere"),      ("MMM",   "3M"),
    ("PFE",   "Pfizer"),          ("GILD",  "Gilead Sciences"),
    ("BSX",   "Boston Scientific"),("ISRG", "Intuitive Surgical"),
    ("PANW",  "Palo Alto Net."),  ("SNOW",  "Snowflake"),
    ("ADBE",  "Adobe"),           ("INTU",  "Intuit"),
    ("QCOM",  "Qualcomm"),        ("TXN",   "Texas Instruments"),
    # Europa
    ("NESN.SW","Nestlé"),         ("NOVN.SW","Novartis"),
    ("ROG.SW", "Roche"),          ("ASML",   "ASML"),
    ("SAP",    "SAP"),            ("LVMH.PA","LVMH"),
    ("TTE",    "TotalEnergies"),  ("SIE.DE", "Siemens"),
    ("AIR.PA", "Airbus"),         ("OR.PA",  "L'Oréal"),
    ("BAYN.DE","Bayer"),          ("BAS.DE", "BASF"),
    ("ALV.DE", "Allianz"),        ("HSBA.L", "HSBC"),
    ("AZN",    "AstraZeneca"),    ("GSK",    "GSK"),
    ("BP",     "BP"),             ("ENEL.MI","Enel"),
    # Asia / Altri
    ("TSM",   "TSMC"),            ("TM",    "Toyota"),
    ("BABA",  "Alibaba"),         ("NVO",   "Novo Nordisk"),
    ("SONY",  "Sony"),            ("UL",    "Unilever"),
    ("BTI",   "BAT"),             ("DEO",   "Diageo"),
    ("RIO",   "Rio Tinto"),       ("BHP",   "BHP Group"),
    ("VALE",  "Vale"),            ("SHOP",  "Shopify"),
]

# ── Settori S&P500 per ticker ─────────────────────
SECTOR_MAP = {
    # Technology
    "AAPL":"Technology",  "MSFT":"Technology",  "NVDA":"Technology",
    "AVGO":"Technology",  "ORCL":"Technology",  "CRM":"Technology",
    "AMD":"Technology",   "NOW":"Technology",   "IBM":"Technology",
    "ACN":"Technology",   "SAP":"Technology",   "ASML":"Technology",
    "TSM":"Technology",   "SIE.DE":"Technology",
    # Communication
    "GOOGL":"Communication", "META":"Communication", "NFLX":"Communication",
    # Consumer Discretionary
    "AMZN":"Cons. Discret.", "TSLA":"Cons. Discret.", "HD":"Cons. Discret.",
    "MCD":"Cons. Discret.",  "COST":"Cons. Discret.", "TM":"Cons. Discret.",
    "BABA":"Cons. Discret.", "SONY":"Cons. Discret.", "AIR.PA":"Cons. Discret.",
    "LVMH.PA":"Cons. Discret.", "OR.PA":"Cons. Discret.",
    # Consumer Staples
    "WMT":"Cons. Staples", "PG":"Cons. Staples",  "KO":"Cons. Staples",
    "PEP":"Cons. Staples", "PM":"Cons. Staples",  "UL":"Cons. Staples",
    "BTI":"Cons. Staples", "DEO":"Cons. Staples", "NESN.SW":"Cons. Staples",
    # Financials
    "JPM":"Financials", "V":"Financials",   "MA":"Financials",
    "BAC":"Financials", "GS":"Financials",  "BRK-B":"Financials",
    # Healthcare
    "LLY":"Healthcare", "UNH":"Healthcare", "JNJ":"Healthcare",
    "MRK":"Healthcare", "ABBV":"Healthcare","TMO":"Healthcare",
    "NVO":"Healthcare", "NOVN.SW":"Healthcare", "ROG.SW":"Healthcare",
    # Energy
    "XOM":"Energy", "CVX":"Energy", "TTE":"Energy", "BP":"Energy",
    "RIO":"Materials",
    # Industrials
    "GE":"Industrials",  "CAT":"Industrials", "LIN":"Materials",
    # USA extra
    "AMGN":"Healthcare",  "T":"Communication",   "MS":"Financials",
    "AXP":"Financials",   "SPGI":"Financials",   "BLK":"Financials",
    "RTX":"Industrials",  "HON":"Industrials",   "DE":"Industrials",
    "MMM":"Industrials",  "PFE":"Healthcare",    "GILD":"Healthcare",
    "BSX":"Healthcare",   "ISRG":"Healthcare",   "PANW":"Technology",
    "SNOW":"Technology",  "ADBE":"Technology",   "INTU":"Technology",
    "QCOM":"Technology",  "TXN":"Technology",
    # Europa extra
    "BAYN.DE":"Healthcare","BAS.DE":"Materials",  "ALV.DE":"Financials",
    "HSBA.L":"Financials", "AZN":"Healthcare",    "GSK":"Healthcare",
    "ENEL.MI":"Utilities",
    # Asia / Altri extra
    "BHP":"Materials",    "VALE":"Materials",    "SHOP":"Cons. Discret.",
    # Materials / Other
}

def _get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Other")

# ── Fetch ─────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)   # cache 30 min
def _fetch_ticker(symbol: str) -> dict:
    """Scarica OHLCV 1 anno + metadati per calcolare tutti gli indicatori."""
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range=1y")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        meta   = result["meta"]
        ts     = result["timestamp"]
        q      = result["indicators"]["quote"][0]

        closes  = q.get("close",  [])
        volumes = q.get("volume", [])
        highs   = q.get("high",   [])
        lows    = q.get("low",    [])
        opens   = q.get("open",   [])

        # Pulisci None
        closes  = [c for c in closes  if c is not None]
        volumes = [v for v in volumes if v is not None]
        highs   = [h for h in highs   if h is not None]

        if len(closes) < 20:
            return {"ok": False}

        c  = np.array(closes,  dtype=float)
        v  = np.array(volumes, dtype=float)
        h  = np.array(highs,   dtype=float)

        price     = c[-1]
        high_52w  = np.nanmax(h) if len(h) > 0 else price
        drawdown  = (high_52w - price) / high_52w * 100 if high_52w > 0 else 0

        # EMA
        def ema(arr, n):
            s = pd.Series(arr)
            return float(s.ewm(span=n, adjust=False).mean().iloc[-1])

        ema20  = ema(c, 20)
        ema50  = ema(c, 50)
        ema200 = ema(c, min(200, len(c)))

        # RSI
        s  = pd.Series(c)
        d  = s.diff()
        g  = d.clip(lower=0).rolling(14).mean()
        l  = (-d.clip(upper=0)).rolling(14).mean()
        rs = g / l.replace(0, np.nan)
        rsi_series = 100 - 100 / (1 + rs)
        rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

        # Volume ratio
        vol_today = v[-1] if len(v) > 0 else 0
        avg_vol   = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v)
        vol_ratio = vol_today / avg_vol if avg_vol > 0 else 1.0

        # Dist EMA200
        dist_ema200 = (price - ema200) / ema200 * 100 if ema200 > 0 else 0

        # Quality score (semplificato)
        obv_trend = "UP" if len(c) >= 2 and c[-1] > c[-2] else "DOWN"
        quality = 0
        if price > ema20:  quality += 2
        if price > ema50:  quality += 2
        if price > ema200: quality += 2
        if vol_ratio > 1.2: quality += 2
        if obv_trend == "UP": quality += 2
        if 40 < rsi < 65: quality += 2

        # Dip Score 0-100
        # 40% drawdown component (max utilità a 30%+ drawdown)
        dd_score = min(drawdown / 30 * 100, 100) * 0.40
        # 30% RSI component (più basso = più punteggio, range 20-50)
        rsi_score = max(0, min((50 - rsi) / 30 * 100, 100)) * 0.30
        # 30% EMA200 component (sotto EMA200 = massimo)
        ema_score = max(0, min(-dist_ema200 / 15 * 100, 100)) * 0.30
        dip_score = round(dd_score + rsi_score + ema_score, 1)

        # Market cap da meta (spesso disponibile)
        mktcap = meta.get("marketCap", 0) or 0

        # ── Momentum Score multi-segnale ──────────────
        # Ogni segnale contribuisce +1 (bull) o -1 (bear) o 0 (neutro)
        mom_signals = []

        # 1. Trend EMA: prezzo vs EMA20/50/200
        mom_signals.append(1 if price > ema20  else -1)
        mom_signals.append(1 if price > ema50  else -1)
        mom_signals.append(1 if price > ema200 else -1)
        mom_signals.append(1 if ema20 > ema50  else -1)   # allineamento EMA

        # 2. RSI momentum
        mom_signals.append(1 if rsi > 55 else (-1 if rsi < 45 else 0))

        # 3. RSI slope (ultimi 5 periodi)
        rsi_arr = rsi_series.dropna().values
        if len(rsi_arr) >= 5:
            rsi_slope = rsi_arr[-1] - rsi_arr[-5]
            mom_signals.append(1 if rsi_slope > 2 else (-1 if rsi_slope < -2 else 0))
        else:
            mom_signals.append(0)

        # 4. MACD (12,26,9)
        s_pd = pd.Series(c)
        macd_line  = s_pd.ewm(span=12).mean() - s_pd.ewm(span=26).mean()
        signal_line= macd_line.ewm(span=9).mean()
        macd_hist  = macd_line - signal_line
        if len(macd_hist.dropna()) >= 2:
            m_last = float(macd_hist.dropna().iloc[-1])
            m_prev = float(macd_hist.dropna().iloc[-2])
            mom_signals.append(1 if m_last > 0 else -1)
            mom_signals.append(1 if m_last > m_prev else (-1 if m_last < m_prev else 0))
        else:
            mom_signals.extend([0, 0])

        # 5. Volume trend (media 5gg vs media 20gg)
        if len(v) >= 20:
            v_short = np.mean(v[-5:])
            v_long  = np.mean(v[-20:])
            mom_signals.append(1 if v_short > v_long * 1.1 else
                               (-1 if v_short < v_long * 0.9 else 0))
        else:
            mom_signals.append(0)

        # 6. Price momentum: close vs 20gg fa e 60gg fa
        if len(c) >= 20:
            mom_signals.append(1 if c[-1] > c[-20] else -1)
        if len(c) >= 60:
            mom_signals.append(1 if c[-1] > c[-60] else -1)

        # Score finale: da -10 a +10 → normalizza 0-100
        n_sig     = len(mom_signals)
        mom_raw   = sum(mom_signals)                        # range [-n, +n]
        mom_score = round((mom_raw / n_sig + 1) / 2 * 100) # 0-100

        # Etichetta e colore
        if mom_score >= 72:   mom_label, mom_color = "🚀 FORTE RIALZO",  "#26a69a"
        elif mom_score >= 58: mom_label, mom_color = "📈 RIALZISTA",     "#66bb6a"
        elif mom_score >= 43: mom_label, mom_color = "➡️ NEUTRO",        "#ffd700"
        elif mom_score >= 28: mom_label, mom_color = "📉 RIBASSISTA",    "#ff9800"
        else:                 mom_label, mom_color = "🔻 FORTE RIBASSO", "#ef5350"

        # MACD values per chart
        macd_val  = round(float(macd_line.iloc[-1]), 3)  if not macd_line.empty  else 0
        signal_val= round(float(signal_line.iloc[-1]),3) if not signal_line.empty else 0
        hist_val  = round(float(macd_hist.iloc[-1]), 3)  if not macd_hist.empty  else 0

        return {
            "ok":          True,
            "price":       round(price, 2),
            "high_52w":    round(high_52w, 2),
            "drawdown":    round(drawdown, 1),
            "rsi":         round(rsi, 1),
            "ema20":       round(ema20, 2),
            "ema50":       round(ema50, 2),
            "ema200":      round(ema200, 2),
            "dist_ema200": round(dist_ema200, 1),
            "vol_ratio":   round(vol_ratio, 2),
            "quality":     quality,
            "dip_score":   dip_score,
            "mktcap":      mktcap,
            "currency":    meta.get("currency", "USD"),
            "name":        meta.get("longName") or meta.get("shortName", ""),
            # Momentum
            "mom_score":   mom_score,
            "mom_label":   mom_label,
            "mom_color":   mom_color,
            "mom_signals": mom_signals,
            "macd":        macd_val,
            "macd_signal": signal_val,
            "macd_hist":   hist_val,
        }
    except Exception as e:
        return {"ok": False, "err": str(e)}


# ── Scan all blue chips ───────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _scan_all() -> pd.DataFrame:
    rows = []
    for sym, label in BLUE_CHIPS:
        d = _fetch_ticker(sym)
        if not d.get("ok"):
            continue
        rows.append({
            "Ticker":      sym,
            "Nome":        label,
            "Prezzo":      d["price"],
            "Max 52w":     d["high_52w"],
            "Drawdown %":  -d["drawdown"],   # negativo per visualizzazione
            "RSI":         d["rsi"],
            "EMA200":      d["ema200"],
            "Dist EMA200%":d["dist_ema200"],
            "Vol×":        d["vol_ratio"],
            "Quality":     d["quality"],
            "Dip Score":   d["dip_score"],
            "Momentum":    d.get("mom_score", 50),
            "Mom Label":   d.get("mom_label", "➡️ NEUTRO"),
            "Mom Color":   d.get("mom_color", "#ffd700"),
            "MACD":        d.get("macd", 0),
            "MACD Signal": d.get("macd_signal", 0),
            "MACD Hist":   d.get("macd_hist", 0),
            "Currency":    d["currency"],
            "_dd_raw":     d["drawdown"],
            "_ema200_raw": d["ema200"],
            "Sector":      _get_sector(sym),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("Dip Score", ascending=False).reset_index(drop=True)
    return df


# ── Sparkline mini chart per top N ───────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_closes(symbol: str) -> list:
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range=6mo")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        q = data["chart"]["result"][0]["indicators"]["quote"][0]
        return [c for c in q.get("close", []) if c is not None]
    except Exception:
        return []


def _sparkline(closes: list, color: str) -> go.Figure:
    if not closes:
        return go.Figure()
    norm = [(c / closes[0] - 1) * 100 for c in closes]
    fig = go.Figure(go.Scatter(
        y=norm, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=60, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig



# ── Momentum Gauge semicircolare ──────────────────

def _momentum_gauge(score: int, label: str, color: str,
                    title: str = "", height: int = 200) -> go.Figure:
    """
    Gauge semicircolare 0-100:
      0-28  → Forte Ribasso  (rosso)
      28-43 → Ribassista     (arancio)
      43-58 → Neutro         (giallo)
      58-72 → Rialzista      (verde chiaro)
      72-100→ Forte Rialzo   (verde)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 28, "color": color}, "suffix": ""},
        title={"text": f"<b>{title}</b><br><span style='font-size:0.85em;color:{color}'>{label}</span>",
               "font": {"size": 11, "color": "#d1d4dc"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 28, 43, 58, 72, 100],
                "ticktext": ["", "Ribasso", "Neutro", "", "Rialzo", ""],
                "tickfont": {"size": 8, "color": "#787b86"},
                "tickcolor": "#2a2e39",
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e222d",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   28],  "color": "rgba(239,83,80,0.25)"},
                {"range": [28,  43],  "color": "rgba(255,152,0,0.20)"},
                {"range": [43,  58],  "color": "rgba(255,215,0,0.15)"},
                {"range": [58,  72],  "color": "rgba(102,187,106,0.20)"},
                {"range": [72,  100], "color": "rgba(38,166,154,0.25)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=height,
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color="#d1d4dc"),
    )
    return fig


def _momentum_bar(score: int, color: str) -> str:
    """HTML barra direzionale momentum -100% a +100%."""
    # Converti score 0-100 in posizione -50/+50 per la barra
    pct    = score - 50          # -50 a +50
    width  = abs(pct) * 2        # 0-100%
    left   = pct < 0
    bg     = color
    side   = "right" if left else "left"
    return (
        f'<div style="background:#2a2e39;border-radius:4px;height:8px;'
        f'position:relative;margin:4px 0">'
        f'<div style="position:absolute;top:0;bottom:0;{side}:50%;'
        f'width:{width/2:.0f}%;background:{bg};border-radius:4px"></div>'
        f'<div style="position:absolute;top:-1px;bottom:-1px;left:50%;'
        f'width:2px;background:#787b86"></div>'
        f'</div>'
    )


# ── Momentum Dashboard globale ────────────────────

def _render_momentum_dashboard(df: pd.DataFrame) -> None:
    """
    Vista dedicata momentum: gauge griglia per ogni titolo filtrato.
    Mostra anche distribuzione Bull/Neutro/Bear e heatmap momentum.
    """
    st.markdown(
        f'<div style="background:#1e222d;border-left:3px solid #2962ff;'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:12px">'
        f'<span style="color:#2962ff;font-weight:700">📡 MOMENTUM DASHBOARD</span>'
        f'<span style="color:#787b86;font-size:0.78rem;margin-left:10px">'
        f'Analisi direzionale multi-segnale: EMA · RSI · MACD · Volume · Price Momentum</span>'
        f'</div>', unsafe_allow_html=True
    )

    # ── Distribuzione Bull/Neutro/Bear ────────────
    bull  = (df["Momentum"] >= 58).sum()
    bear  = (df["Momentum"] <  43).sum()
    neut  = len(df) - bull - bear
    total = len(df)

    pct_bull = bull / total * 100
    pct_bear = bear / total * 100
    pct_neut = neut / total * 100

    # Sentiment di mercato aggregato
    avg_mom = df["Momentum"].mean()
    if avg_mom >= 65:   mkt_label, mkt_color = "🚀 MERCATO RIALZISTA",  "#26a69a"
    elif avg_mom >= 55: mkt_label, mkt_color = "📈 LIEVE RIALZO",       "#66bb6a"
    elif avg_mom >= 45: mkt_label, mkt_color = "➡️ MERCATO NEUTRO",     "#ffd700"
    elif avg_mom >= 35: mkt_label, mkt_color = "📉 LIEVE RIBASSO",      "#ff9800"
    else:               mkt_label, mkt_color = "🔻 MERCATO RIBASSISTA", "#ef5350"

    # Banner sentiment
    st.markdown(
        f'<div style="background:#1e222d;border:1px solid #2a2e39;'
        f'border-radius:8px;padding:14px;text-align:center;margin-bottom:12px;'
        f'border-top:3px solid {mkt_color}">'
        f'<div style="font-size:1.3rem;font-weight:800;color:{mkt_color}">'
        f'{mkt_label}</div>'
        f'<div style="color:#787b86;font-size:0.8rem;margin-top:4px">'
        f'Momentum medio Blue Chip: <b style="color:#d1d4dc">{avg_mom:.0f}/100</b> '
        f'su {total} titoli analizzati</div>'
        f'</div>', unsafe_allow_html=True
    )

    # Barre distribuzione
    c1, c2, c3 = st.columns(3)
    for col, label, count, pct, color in [
        (c1, "🟢 RIALZISTI",  bull, pct_bull, "#26a69a"),
        (c2, "🟡 NEUTRI",     neut, pct_neut, "#ffd700"),
        (c3, "🔴 RIBASSISTI", bear, pct_bear, "#ef5350"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:#1e222d;border:1px solid #2a2e39;'
                f'border-radius:6px;padding:10px;text-align:center">'
                f'<div style="color:#787b86;font-size:0.72rem">{label}</div>'
                f'<div style="color:{color};font-size:1.6rem;font-weight:700">{count}</div>'
                f'<div style="color:#787b86;font-size:0.75rem">{pct:.0f}% del totale</div>'
                f'<div style="background:#2a2e39;border-radius:3px;height:5px;margin-top:6px">'
                f'<div style="background:{color};width:{pct:.0f}%;height:5px;border-radius:3px">'
                f'</div></div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge griglia ─────────────────────────────
    st.markdown(
        f'<span style="color:#787b86;font-size:0.8rem">'
        f'Gauge momentum per titolo — ordinati per score</span>',
        unsafe_allow_html=True
    )

    df_sorted = df.sort_values("Momentum", ascending=False).reset_index(drop=True)
    cols_per_row = 4
    tickers_list = list(df_sorted.iterrows())

    for row_start in range(0, len(tickers_list), cols_per_row):
        chunk = tickers_list[row_start:row_start + cols_per_row]
        cols  = st.columns(cols_per_row)
        for col, (ri, (_, row)) in zip(cols, enumerate(chunk)):
            sym   = row["Ticker"]
            nome  = row["Nome"][:16]
            score = int(row["Momentum"])
            label = row["Mom Label"]
            color = row["Mom Color"]
            rsi   = row["RSI"]
            macd_h= row["MACD Hist"]
            dd    = row["_dd_raw"]
            tv_url= f"https://it.tradingview.com/chart/?symbol={sym.split('.')[0]}"
            # Key stabile: posizione assoluta nel dataset ordinato, mai duplicata
            abs_pos = row_start + ri

            with col:
                fig = _momentum_gauge(score, label, color,
                                      title=f"{sym}", height=175)
                # Inietta ticker nell'id figura per renderla unica a Streamlit
                fig.update_layout(meta={"ticker": sym, "pos": abs_pos})
                st.plotly_chart(fig, use_container_width=True,
                                key=f"bcd_mg_{abs_pos}_{sym.replace('-','_').replace('.','_')}")

                # Mini dettagli sotto il gauge
                macd_color = "#26a69a" if macd_h >= 0 else "#ef5350"
                st.markdown(
                    f'<div style="background:#1e222d;border-radius:4px;'
                    f'padding:4px 8px;font-size:0.72rem;margin-top:-8px">'
                    f'<a href="{tv_url}" target="_blank" style="color:#50c4e0;'
                    f'text-decoration:none;font-weight:700">{sym}</a>'
                    f'<span style="color:#787b86"> {nome}</span><br>'
                    f'RSI <b style="color:{"#26a69a" if rsi<45 else "#787b86"}">{rsi:.0f}</b>'
                    f' · MACD <b style="color:{macd_color}">{"▲" if macd_h>=0 else "▼"}</b>'
                    f' · DD <b style="color:#ef5350">{dd:.0f}%</b>'
                    f'{_momentum_bar(score, color)}'
                    f'</div>',
                    unsafe_allow_html=True
                )


# ── Render card per top ticker ────────────────────

def _render_card(row: pd.Series, rank: int):
    sym       = row["Ticker"]
    nome      = row["Nome"]
    price     = row["Prezzo"]
    dd        = row["_dd_raw"]
    rsi       = row["RSI"]
    dip       = row["Dip Score"]
    qual      = row["Quality"]
    dist200   = row["Dist EMA200%"]
    vol       = row["Vol×"]
    currency  = row["Currency"]
    mom_score = int(row.get("Momentum", 50))
    mom_label = row.get("Mom Label", "➡️ NEUTRO")
    mom_color = row.get("Mom Color", "#ffd700")
    curr_sym  = "€" if currency == "EUR" else ("£" if currency == "GBP" else "$")
    max52     = row["Max 52w"]

    # Colori
    dd_color  = TV_RED if dd > 25 else (TV_ORANGE if dd > 15 else TV_GOLD)
    rsi_color = TV_GREEN if rsi < 35 else (TV_CYAN if rsi < 45 else TV_GRAY)
    dip_color = TV_GREEN if dip >= 60 else (TV_GOLD if dip >= 35 else TV_GRAY)

    # TV link
    tv_sym = sym.replace(".SW", "").replace(".PA", "").replace(".DE", "")
    tv_url = f"https://it.tradingview.com/chart/?symbol={sym.split('.')[0]}"

    # Medaglia rank
    medal = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")

    st.markdown(
        f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
        f'border-radius:8px;padding:12px 16px;margin-bottom:8px;'
        f'border-left:4px solid {dip_color}">'

        # Header
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div>'
        f'<span style="color:{TV_GRAY};font-size:0.85rem;margin-right:6px">{medal}</span>'
        f'<a href="{tv_url}" target="_blank" style="text-decoration:none">'
        f'<span style="color:{TV_CYAN};font-weight:700;font-size:1rem">{sym}</span></a>'
        f'<span style="color:{TV_GRAY};font-size:0.8rem;margin-left:8px">{nome}</span>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<span style="font-size:1.1rem;font-weight:700;color:{TV_TEXT}">'
        f'{curr_sym}{price:,.2f}</span>'
        f'<span style="color:{TV_RED};font-size:0.85rem;margin-left:10px">'
        f'▼{dd:.1f}% dal max</span>'
        f'</div></div>'

        # Metriche
        f'<div style="display:flex;gap:20px;margin-top:10px;flex-wrap:wrap">'

        f'<div style="text-align:center">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem">DIP SCORE</div>'
        f'<div style="color:{dip_color};font-weight:700;font-size:1.1rem">{dip:.0f}/100</div>'
        f'</div>'

        f'<div style="text-align:center">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem">RSI(14)</div>'
        f'<div style="color:{rsi_color};font-weight:700;font-size:1.1rem">{rsi:.1f}</div>'
        f'</div>'

        f'<div style="text-align:center">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem">vs EMA200</div>'
        f'<div style="color:{"#ef5350" if dist200<0 else "#26a69a"};font-weight:700;font-size:1.1rem">'
        f'{"▼" if dist200<0 else "▲"}{abs(dist200):.1f}%</div>'
        f'</div>'

        f'<div style="text-align:center">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem">Max 52w</div>'
        f'<div style="color:{TV_TEXT};font-weight:600;font-size:0.95rem">'
        f'{curr_sym}{max52:,.2f}</div>'
        f'</div>'

        f'<div style="text-align:center">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem">Vol×</div>'
        f'<div style="color:{"#50c4e0" if vol>1.5 else TV_GRAY};font-weight:600;font-size:0.95rem">'
        f'{vol:.2f}x</div>'
        f'</div>'

        f'<div style="text-align:center">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem">Quality</div>'
        f'<div style="color:{TV_TEXT};font-weight:600;font-size:0.95rem">{qual}/12</div>'
        f'</div>'

        f'</div>'

        # Barra drawdown
        f'<div style="margin-top:10px">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem;margin-bottom:3px">'
        f'Drawdown dal massimo 52 settimane</div>'
        f'<div style="background:{TV_BORDER};border-radius:3px;height:5px">'
        f'<div style="background:{dd_color};width:{min(dd,50)/50*100:.0f}%;'
        f'height:5px;border-radius:3px"></div></div>'
        f'</div>'

        # Momentum bar
        f'<div style="margin-top:8px">'
        f'<div style="color:{TV_GRAY};font-size:0.68rem;margin-bottom:2px">'
        f'Momentum: <b style="color:{mom_color}">{mom_label}</b> ({mom_score}/100)</div>'
        f'{_momentum_bar(mom_score, mom_color)}'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════
# MODULO 4 — 🔥 Heatmap Settoriale
# ══════════════════════════════════════════════════

def _render_sector_heatmap(df: pd.DataFrame) -> None:
    """
    Griglia interattiva stile TradingView: settori S&P500 colorati
    per momentum medio. Click settore → dettaglio ticker con segnali.
    """
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_ORANGE};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:12px">'
        f'<span style="color:{TV_ORANGE};font-weight:700">🔥 HEATMAP SETTORIALE</span>'
        f'<span style="color:{TV_GRAY};font-size:0.78rem;margin-left:10px">'
        f'Momentum medio per settore · Click settore → dettaglio ticker</span>'
        f'</div>', unsafe_allow_html=True
    )

    # Aggiungi settore al dataframe se non presente
    if "Sector" not in df.columns:
        df = df.copy()
        df["Sector"] = df["Ticker"].apply(_get_sector)

    # Aggrega per settore
    sector_stats = (
        df.groupby("Sector")
        .agg(
            mom_avg=("Momentum", "mean"),
            mom_min=("Momentum", "min"),
            mom_max=("Momentum", "max"),
            count=("Ticker", "count"),
            dip_avg=("Dip Score", "mean"),
            rsi_avg=("RSI", "mean"),
            dd_avg=("_dd_raw", "mean"),
        )
        .reset_index()
        .sort_values("mom_avg", ascending=False)
    )

    def _sector_color(score: float) -> str:
        if score >= 72:  return "#26a69a"
        if score >= 58:  return "#66bb6a"
        if score >= 43:  return "#ffd700"
        if score >= 28:  return "#ff9800"
        return "#ef5350"

    def _sector_bg(score: float) -> str:
        if score >= 72:  return "rgba(38,166,154,0.18)"
        if score >= 58:  return "rgba(102,187,106,0.15)"
        if score >= 43:  return "rgba(255,215,0,0.12)"
        if score >= 28:  return "rgba(255,152,0,0.15)"
        return "rgba(239,83,80,0.18)"

    # ── Heatmap Plotly treemap ─────────────────────
    fig_hm = go.Figure(go.Treemap(
        labels=sector_stats["Sector"].tolist(),
        parents=[""] * len(sector_stats),
        values=sector_stats["count"].tolist(),
        customdata=np.column_stack([
            sector_stats["mom_avg"].round(0).tolist(),
            sector_stats["dip_avg"].round(1).tolist(),
            sector_stats["rsi_avg"].round(1).tolist(),
            sector_stats["dd_avg"].round(1).tolist(),
            sector_stats["count"].tolist(),
        ]),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Momentum: <b>%{customdata[0]:.0f}/100</b><br>"
            "Dip Score medio: %{customdata[1]:.1f}<br>"
            "RSI medio: %{customdata[2]:.1f}<br>"
            "Drawdown medio: %{customdata[3]:.1f}%<br>"
            "Titoli: %{customdata[4]}<extra></extra>"
        ),
        marker=dict(
            colors=sector_stats["mom_avg"].tolist(),
            colorscale=[
                [0.0,  "#ef5350"],
                [0.28, "#ff9800"],
                [0.43, "#ffd700"],
                [0.58, "#66bb6a"],
                [1.0,  "#26a69a"],
            ],
            cmin=0, cmax=100,
            colorbar=dict(
                title="Momentum",
                tickvals=[0, 28, 43, 58, 72, 100],
                ticktext=["Forte Ribasso","Ribasso","Neutro","Rialzo","Forte Rialzo",""],
                tickfont=dict(size=9, color=TV_TEXT),
                bgcolor=TV_PANEL,
                bordercolor=TV_BORDER,
                len=0.8,
            ),
            pad=dict(t=4),
        ),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.0f}",
        textfont=dict(size=13, color="#ffffff"),
        pathbar_visible=False,
    ))
    fig_hm.update_layout(
        height=380,
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_BG,
        margin=dict(l=4, r=4, t=10, b=4),
        font=dict(color=TV_TEXT),
    )
    st.plotly_chart(fig_hm, use_container_width=True, key="bcd_sector_treemap")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cards per settore con barra momentum ──────
    st.markdown(
        f'<span style="color:{TV_GRAY};font-size:0.8rem">Seleziona un settore per il dettaglio dei ticker</span>',
        unsafe_allow_html=True
    )

    sectors_list = sector_stats["Sector"].tolist()
    selected_sector = st.selectbox(
        "Settore",
        options=["— Tutti —"] + sectors_list,
        key="bcd_sector_select",
        label_visibility="collapsed"
    )

    # ── Griglia settori cards ──────────────────────
    cols_s = st.columns(min(4, len(sector_stats)))
    for i, (_, srow) in enumerate(sector_stats.iterrows()):
        with cols_s[i % len(cols_s)]:
            sc = srow["mom_avg"]
            sc_color = _sector_color(sc)
            sc_bg    = _sector_bg(sc)
            bar_w    = f"{sc:.0f}%"
            st.markdown(
                f'<div style="background:{sc_bg};border:1px solid {TV_BORDER};'
                f'border-top:3px solid {sc_color};border-radius:6px;'
                f'padding:8px 10px;margin-bottom:8px;cursor:pointer">'
                f'<div style="color:{TV_TEXT};font-weight:700;font-size:0.85rem">{srow["Sector"]}</div>'
                f'<div style="color:{sc_color};font-size:1.3rem;font-weight:800">{sc:.0f}</div>'
                f'<div style="color:{TV_GRAY};font-size:0.68rem">'
                f'{int(srow["count"])} titoli · RSI {srow["rsi_avg"]:.0f} · DD {srow["dd_avg"]:.1f}%</div>'
                f'<div style="background:{TV_BORDER};border-radius:3px;height:4px;margin-top:5px">'
                f'<div style="background:{sc_color};width:{bar_w};height:4px;border-radius:3px"></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

    # ── Dettaglio ticker del settore selezionato ──
    if selected_sector != "— Tutti —":
        df_sec = df[df["Sector"] == selected_sector].sort_values("Momentum", ascending=False)
    else:
        df_sec = df.sort_values("Momentum", ascending=False)

    st.markdown(
        f'<div style="color:{TV_GOLD};font-weight:700;font-size:0.9rem;'
        f'margin:12px 0 6px 0">📋 Ticker — {selected_sector} '
        f'({len(df_sec)} titoli)</div>',
        unsafe_allow_html=True
    )

    # Tabella interattiva con segnali per ticker
    for _, trow in df_sec.iterrows():
        sym   = trow["Ticker"]
        nome  = trow["Nome"]
        mom   = int(trow["Momentum"])
        mom_c = trow["Mom Color"]
        mom_l = trow["Mom Label"]
        rsi   = trow["RSI"]
        dd    = trow["_dd_raw"]
        dip   = trow["Dip Score"]
        macdh = trow["MACD Hist"]
        sector= trow.get("Sector", _get_sector(sym))
        tv_url= f"https://it.tradingview.com/chart/?symbol={sym.split('.')[0]}"
        macd_color = TV_GREEN if macdh >= 0 else TV_RED
        rsi_color  = TV_GREEN if rsi < 35 else (TV_CYAN if rsi < 45 else TV_GRAY)
        dip_color  = TV_GREEN if dip >= 60 else (TV_GOLD if dip >= 35 else TV_GRAY)

        bull_signals = sum([
            mom >= 58,
            rsi < 45,
            macdh > 0,
            dd > 15,
            dip >= 40,
        ])
        signal_icons = "🟢" * bull_signals + "⚪" * (5 - bull_signals)

        st.markdown(
            f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
            f'border-left:4px solid {mom_c};border-radius:6px;'
            f'padding:8px 12px;margin-bottom:6px;'
            f'display:flex;align-items:center;justify-content:space-between">'
            # Sinistra: ticker + nome
            f'<div style="min-width:140px">'
            f'<a href="{tv_url}" target="_blank" style="color:{TV_CYAN};'
            f'font-weight:700;text-decoration:none;font-size:0.9rem">{sym}</a>'
            f'<span style="color:{TV_GRAY};font-size:0.75rem;margin-left:6px">{nome[:18]}</span>'
            f'</div>'
            # Centro: metriche
            f'<div style="display:flex;gap:16px;flex:1;justify-content:center;flex-wrap:wrap">'
            f'<div style="text-align:center">'
            f'<div style="color:{TV_GRAY};font-size:0.65rem">MOM</div>'
            f'<div style="color:{mom_c};font-weight:700">{mom}</div></div>'
            f'<div style="text-align:center">'
            f'<div style="color:{TV_GRAY};font-size:0.65rem">RSI</div>'
            f'<div style="color:{rsi_color};font-weight:700">{rsi:.0f}</div></div>'
            f'<div style="text-align:center">'
            f'<div style="color:{TV_GRAY};font-size:0.65rem">DIP</div>'
            f'<div style="color:{dip_color};font-weight:700">{dip:.0f}</div></div>'
            f'<div style="text-align:center">'
            f'<div style="color:{TV_GRAY};font-size:0.65rem">DD%</div>'
            f'<div style="color:{TV_RED};font-weight:700">{dd:.1f}%</div></div>'
            f'<div style="text-align:center">'
            f'<div style="color:{TV_GRAY};font-size:0.65rem">MACD</div>'
            f'<div style="color:{macd_color};font-weight:700">{"▲" if macdh>=0 else "▼"}</div></div>'
            f'</div>'
            # Destra: segnali + label
            f'<div style="text-align:right;min-width:130px">'
            f'<div style="font-size:0.75rem">{signal_icons}</div>'
            f'<div style="color:{mom_c};font-size:0.7rem;font-weight:700">{mom_l}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════
# MODULO 6 — 📈 Backtest Avanzato
# ══════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ohlcv_2y(symbol: str) -> pd.DataFrame:
    """Scarica 2 anni di dati OHLCV per backtest."""
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range=2y")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        q  = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s"),
            "open":   q.get("open",   []),
            "high":   q.get("high",   []),
            "low":    q.get("low",    []),
            "close":  q.get("close",  []),
            "volume": q.get("volume", []),
        }).dropna(subset=["close"]).reset_index(drop=True)
        df["date"] = df["date"].dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def _compute_indicators(df_ohlcv: pd.DataFrame) -> dict:
    """Calcola tutti gli indicatori necessari per i backtest."""
    c = df_ohlcv["close"].values.astype(float)
    v = df_ohlcv["volume"].fillna(0).values.astype(float)
    dates = df_ohlcv["date"].values
    s = pd.Series(c)

    ema20  = s.ewm(span=20,  adjust=False).mean().values
    ema50  = s.ewm(span=50,  adjust=False).mean().values
    ema200 = s.ewm(span=200, adjust=False).mean().values

    d_   = s.diff()
    g_   = d_.clip(lower=0).rolling(14).mean()
    l_   = (-d_.clip(upper=0)).rolling(14).mean()
    rsi_s = (100 - 100 / (1 + g_ / l_.replace(0, np.nan))).values

    macd_line   = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist   = (macd_line - signal_line).values
    macd_arr    = macd_line.values
    signal_arr  = signal_line.values

    # VWAP approssimato (rolling 20 giorni: somma(close*vol)/somma(vol))
    cv = pd.Series(c * v)
    vwap_s = cv.rolling(20).sum() / pd.Series(v).rolling(20).sum()
    vwap   = vwap_s.values

    # ADX (14 periodi, approssimazione)
    h = df_ohlcv["high"].fillna(method="ffill").values.astype(float)
    lo = df_ohlcv["low"].fillna(method="ffill").values.astype(float)
    tr  = np.maximum(h[1:] - lo[1:],
          np.maximum(np.abs(h[1:] - c[:-1]),
                     np.abs(lo[1:] - c[:-1])))
    dm_pos = np.where((h[1:] - h[:-1]) > (lo[:-1] - lo[1:]),
                       np.maximum(h[1:] - h[:-1], 0), 0)
    dm_neg = np.where((lo[:-1] - lo[1:]) > (h[1:] - h[:-1]),
                       np.maximum(lo[:-1] - lo[1:], 0), 0)
    n = 14
    atr_s, dmp_s, dmn_s = [0.0]*n, [0.0]*n, [0.0]*n
    if len(tr) >= n:
        atr_s = [np.mean(tr[:n])]
        dmp_s = [np.mean(dm_pos[:n])]
        dmn_s = [np.mean(dm_neg[:n])]
        for k in range(n, len(tr)):
            atr_s.append(atr_s[-1] - atr_s[-1]/n + tr[k])
            dmp_s.append(dmp_s[-1] - dmp_s[-1]/n + dm_pos[k])
            dmn_s.append(dmn_s[-1] - dmn_s[-1]/n + dm_neg[k])
    atr_a = np.array(atr_s)
    dmp_a = np.array(dmp_s)
    dmn_a = np.array(dmn_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        di_pos = np.where(atr_a > 0, 100 * dmp_a / atr_a, 0)
        di_neg = np.where(atr_a > 0, 100 * dmn_a / atr_a, 0)
        dx     = np.where((di_pos + di_neg) > 0,
                           100 * np.abs(di_pos - di_neg) / (di_pos + di_neg), 0)
    adx_list = []
    if len(dx) >= n:
        adx_list = [np.mean(dx[:n])]
        for k in range(n, len(dx)):
            adx_list.append((adx_list[-1] * (n-1) + dx[k]) / n)
    # Allinea adx alla lunghezza di c (pad con 0 all'inizio)
    adx_full = np.zeros(len(c))
    offset = len(c) - len(adx_list)
    if len(adx_list) > 0:
        adx_full[offset:] = adx_list

    return dict(c=c, v=v, dates=dates, ema20=ema20, ema50=ema50, ema200=ema200,
                rsi=rsi_s, macd=macd_arr, macd_signal=signal_arr,
                macd_hist=macd_hist, vwap=vwap, adx=adx_full,
                di_pos=np.concatenate([np.zeros(offset), di_pos]) if len(di_pos) else np.zeros(len(c)),
                di_neg=np.concatenate([np.zeros(offset), di_neg]) if len(di_neg) else np.zeros(len(c)))


def _run_backtest(df_ohlcv: pd.DataFrame, strategy: str = "DipScore") -> dict:
    """
    Simula strategia su dati storici.
    Strategie:
      DipScore  — RSI<45 + prezzo sotto EMA200
      Momentum  — MACD+ + EMA20>EMA50
      RSI+VWAP  — RSI incrocia sopra 30 + prezzo sopra VWAP (Long)
                  RSI incrocia sotto 70 + prezzo sotto VWAP (Exit)
      ADX+EMA   — EMA20>EMA50 + ADX>25 (Long); EMA20<EMA50 + ADX<25 (Exit)
    """
    if df_ohlcv.empty or len(df_ohlcv) < 60:
        return {"ok": False}

    ind  = _compute_indicators(df_ohlcv)
    c    = ind["c"]
    dates= ind["dates"]
    rsi  = ind["rsi"]
    ema20= ind["ema20"]; ema50=ind["ema50"]; ema200=ind["ema200"]
    macd_hist = ind["macd_hist"]
    vwap = ind["vwap"]
    adx  = ind["adx"]

    in_trade    = False
    entry_price = 0.0
    entry_idx   = 0
    trades      = []
    equity      = [100.0]
    equity_dates= [dates[0]]
    hold_days   = 20
    stop_loss   = -0.08
    take_profit = 0.15

    # Segnali di ingresso precedente periodo (per crossover)
    rsi_prev_above30 = False
    rsi_prev_below70 = False

    for i in range(60, len(c) - 1):
        rsi_i    = rsi[i]   if not np.isnan(rsi[i])   else 50
        rsi_prev = rsi[i-1] if not np.isnan(rsi[i-1]) else 50
        vwap_i   = vwap[i]  if not np.isnan(vwap[i])  else c[i]

        if strategy == "DipScore":
            entry_signal = (rsi_i < 45) and (c[i] < ema200[i])
            exit_signal  = False  # usa SL/TP/Time

        elif strategy == "Momentum":
            entry_signal = (macd_hist[i] > 0) and (ema20[i] > ema50[i]) and (c[i] > ema50[i])
            exit_signal  = (macd_hist[i] < 0) or (ema20[i] < ema50[i])

        elif strategy == "RSI+VWAP":
            # Entry: RSI attraversa sopra 30 (da sotto) + prezzo > VWAP
            rsi_cross_30 = (rsi_prev < 30) and (rsi_i >= 30)
            entry_signal = rsi_cross_30 and (c[i] > vwap_i)
            # Exit: RSI attraversa sotto 70 (da sopra) + prezzo < VWAP
            rsi_cross_70 = (rsi_prev > 70) and (rsi_i <= 70)
            exit_signal  = rsi_cross_70 or (c[i] < vwap_i and rsi_i > 65)

        else:  # ADX+EMA
            entry_signal = (ema20[i] > ema50[i]) and (adx[i] > 25) and (ema20[i-1] <= ema50[i-1])
            exit_signal  = (ema20[i] < ema50[i]) or (adx[i] < 25)

        if not in_trade and entry_signal:
            in_trade    = True
            entry_price = c[i + 1]
            entry_idx   = i + 1

        elif in_trade:
            pct       = (c[i] - entry_price) / entry_price
            days_held = i - entry_idx
            exit_reason = None

            if pct <= stop_loss:          exit_reason = "SL"
            elif pct >= take_profit:      exit_reason = "TP"
            elif days_held >= hold_days:  exit_reason = "Time"
            elif exit_signal:             exit_reason = "Signal"

            if exit_reason:
                trades.append({
                    "entry_date":  dates[entry_idx],
                    "exit_date":   dates[i],
                    "entry_price": entry_price,
                    "exit_price":  c[i],
                    "pct":         pct * 100,
                    "days":        days_held,
                    "reason":      exit_reason,
                    "win":         pct > 0,
                })
                equity.append(equity[-1] * (1 + pct))
                equity_dates.append(dates[i])
                in_trade = False

    daily_equity = [100.0]
    for i in range(1, len(c)):
        daily_equity.append(daily_equity[-1] * (c[i] / c[i-1]))

    if not trades:
        return {"ok": False, "reason": "no_trades"}

    df_trades = pd.DataFrame(trades)

    wins     = df_trades["win"].sum()
    total_t  = len(df_trades)
    win_rate = wins / total_t * 100 if total_t > 0 else 0

    eq_arr  = np.array(equity)
    returns = np.diff(eq_arr) / eq_arr[:-1]
    sharpe  = (np.mean(returns) / np.std(returns) * np.sqrt(252)
               if np.std(returns) > 0 else 0)
    peak     = np.maximum.accumulate(eq_arr)
    drawdowns= (eq_arr - peak) / peak * 100
    max_dd   = float(np.min(drawdowns))
    total_return = (eq_arr[-1] / eq_arr[0] - 1) * 100

    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])
    df_trades["month"]      = df_trades["entry_date"].dt.month
    df_trades["year"]       = df_trades["entry_date"].dt.year
    monthly = df_trades.groupby(["year","month"])["pct"].sum().reset_index()

    return {
        "ok": True, "trades": df_trades, "indicators": ind,
        "equity": eq_arr.tolist(),
        "equity_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in equity_dates],
        "bnh_equity": daily_equity,
        "bnh_dates":  [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "win_rate": win_rate, "sharpe": sharpe, "max_dd": max_dd,
        "total_return": total_return, "total_trades": total_t,
        "monthly": monthly,
    }


def _plot_strategy_chart(res: dict, df_ohlcv: pd.DataFrame,
                         strategy: str, sel_ticker: str, sel_name: str) -> None:
    """
    Grafico specifico per ogni strategia con i suoi indicatori dedicati.
    Completamente separato dall'equity curve.
    """
    ind   = res["indicators"]
    c     = ind["c"]
    dates_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in ind["dates"]]
    df_t  = res["trades"]

    # ── Colori e config per strategia ─────────────
    strategy_colors = {
        "DipScore":  {"accent": TV_CYAN,   "sub_color": "#9c27b0"},
        "Momentum":  {"accent": TV_BLUE,   "sub_color": TV_GREEN},
        "RSI+VWAP":  {"accent": "#e91e63", "sub_color": "#9c27b0"},
        "ADX+EMA":   {"accent": TV_ORANGE, "sub_color": TV_GREEN},
    }
    sc = strategy_colors.get(strategy, {"accent": TV_BLUE, "sub_color": TV_GREEN})

    if strategy in ("DipScore", "Momentum"):
        # 2 righe: candele+EMA / MACD o RSI
        rows, row_h = 2, [0.68, 0.32]
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
        subplot_titles = [
            f"{sel_ticker} — {sel_name}  |  EMA 20/50/200",
            "MACD (12,26,9)" if strategy == "Momentum" else "RSI (14)",
        ]
    elif strategy == "RSI+VWAP":
        # 2 righe: candele+VWAP / RSI
        rows, row_h = 2, [0.65, 0.35]
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
        subplot_titles = [
            f"{sel_ticker} — {sel_name}  |  VWAP (rolling 20d)",
            "RSI (14)  ·  Zona oversold 30 / overbought 70",
        ]
    else:  # ADX+EMA
        # 2 righe: candele+EMA20/50 / ADX
        rows, row_h = 2, [0.65, 0.35]
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
        subplot_titles = [
            f"{sel_ticker} — {sel_name}  |  EMA 20 / EMA 50",
            "ADX (14)  ·  Soglia trend = 25",
        ]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_h,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    # ── Candele ────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates_str,
        open=df_ohlcv["open"].values,
        high=df_ohlcv["high"].values,
        low=df_ohlcv["low"].values,
        close=c,
        name="Price",
        increasing=dict(fillcolor=TV_GREEN, line=dict(color=TV_GREEN, width=1)),
        decreasing=dict(fillcolor=TV_RED,   line=dict(color=TV_RED,   width=1)),
        showlegend=False,
    ), row=1, col=1)

    # ── Indicatori pannello superiore ─────────────
    if strategy in ("DipScore", "Momentum", "ADX+EMA"):
        fig.add_trace(go.Scatter(x=dates_str, y=ind["ema20"],
            mode="lines", name="EMA 20", line=dict(color="#26c6da", width=1.2)),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=dates_str, y=ind["ema50"],
            mode="lines", name="EMA 50", line=dict(color="#7e57c2", width=1.2)),
            row=1, col=1)
    if strategy == "DipScore":
        fig.add_trace(go.Scatter(x=dates_str, y=ind["ema200"],
            mode="lines", name="EMA 200", line=dict(color=TV_GOLD, width=1.5, dash="dot")),
            row=1, col=1)
    if strategy == "RSI+VWAP":
        fig.add_trace(go.Scatter(x=dates_str, y=ind["vwap"],
            mode="lines", name="VWAP", line=dict(color=TV_ORANGE, width=2)),
            row=1, col=1)

    # ── Entry / Exit markers sulle candele ─────────
    entry_dates = [pd.Timestamp(d).strftime("%Y-%m-%d")
                   for d in df_t["entry_date"]]
    exit_dates  = [pd.Timestamp(d).strftime("%Y-%m-%d")
                   for d in df_t["exit_date"]]
    entry_prices= df_t["entry_price"].tolist()
    exit_prices = df_t["exit_price"].tolist()
    pcts        = df_t["pct"].tolist()

    fig.add_trace(go.Scatter(
        x=entry_dates, y=entry_prices,
        mode="markers",
        name="Entry",
        marker=dict(symbol="triangle-up", size=10,
                    color=TV_GREEN, line=dict(color="#ffffff", width=1)),
        hovertemplate="<b>ENTRY</b><br>%{x}<br>%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    win_exits  = [(d, p) for d, p, pct in zip(exit_dates, exit_prices, pcts) if pct > 0]
    loss_exits = [(d, p) for d, p, pct in zip(exit_dates, exit_prices, pcts) if pct <= 0]
    if win_exits:
        fig.add_trace(go.Scatter(
            x=[w[0] for w in win_exits], y=[w[1] for w in win_exits],
            mode="markers", name="Exit ✅",
            marker=dict(symbol="triangle-down", size=10,
                        color=TV_GREEN, line=dict(color="#ffffff", width=1)),
            hovertemplate="<b>EXIT WIN</b><br>%{x}<br>%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if loss_exits:
        fig.add_trace(go.Scatter(
            x=[l[0] for l in loss_exits], y=[l[1] for l in loss_exits],
            mode="markers", name="Exit ❌",
            marker=dict(symbol="triangle-down", size=10,
                        color=TV_RED, line=dict(color="#ffffff", width=1)),
            hovertemplate="<b>EXIT LOSS</b><br>%{x}<br>%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Pannello inferiore: indicatore dedicato ────
    if strategy == "Momentum":
        # MACD
        colors_hist = [TV_GREEN if v >= 0 else TV_RED for v in ind["macd_hist"]]
        fig.add_trace(go.Bar(x=dates_str, y=ind["macd_hist"],
            name="MACD Hist", marker_color=colors_hist, opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates_str, y=ind["macd"],
            mode="lines", name="MACD", line=dict(color=TV_BLUE, width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates_str, y=ind["macd_signal"],
            mode="lines", name="Signal", line=dict(color=TV_ORANGE, width=1.2)), row=2, col=1)
        fig.add_hline(y=0, line=dict(color=TV_BORDER, width=1), row=2, col=1)

    elif strategy == "DipScore":
        # RSI
        fig.add_trace(go.Scatter(x=dates_str, y=ind["rsi"],
            mode="lines", name="RSI(14)",
            line=dict(color="#9c27b0", width=1.8),
            fill="tozeroy", fillcolor="rgba(156,33,176,0.05)"), row=2, col=1)
        fig.add_hline(y=45, line=dict(color=TV_CYAN, dash="dot", width=1), row=2, col=1)
        fig.add_hrect(y0=0, y1=45, fillcolor="rgba(38,166,154,0.06)", line_width=0, row=2, col=1)
        fig.add_annotation(text="Zona Entry RSI<45", x=dates_str[80], y=22,
                           font=dict(size=9, color=TV_CYAN), showarrow=False, row=2, col=1)

    elif strategy == "RSI+VWAP":
        # RSI con zone 30/70
        fig.add_trace(go.Scatter(x=dates_str, y=ind["rsi"],
            mode="lines", name="RSI(14)",
            line=dict(color="#9c27b0", width=1.8)), row=2, col=1)
        fig.add_hline(y=70, line=dict(color=TV_RED,   dash="dot", width=1.2), row=2, col=1)
        fig.add_hline(y=50, line=dict(color=TV_GRAY,  dash="dot", width=0.8), row=2, col=1)
        fig.add_hline(y=30, line=dict(color=TV_GREEN, dash="dot", width=1.2), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.10)",  line_width=0, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.10)", line_width=0, row=2, col=1)
        fig.add_annotation(text="Overbought → EXIT", x=dates_str[80], y=85,
                           font=dict(size=9, color=TV_RED),   showarrow=False, row=2, col=1)
        fig.add_annotation(text="Oversold → ENTRY", x=dates_str[80], y=15,
                           font=dict(size=9, color=TV_GREEN), showarrow=False, row=2, col=1)

    else:  # ADX+EMA
        fig.add_trace(go.Scatter(x=dates_str[:len(ind["adx"])], y=ind["adx"],
            mode="lines", name="ADX(14)",
            line=dict(color=TV_RED, width=2)), row=2, col=1)
        fig.add_hline(y=25, line=dict(color=TV_GOLD, dash="dot", width=1.5), row=2, col=1)
        fig.add_hrect(y0=25, y1=100, fillcolor="rgba(255,152,0,0.08)", line_width=0, row=2, col=1)
        fig.add_annotation(text="Trend forte (ADX>25)", x=dates_str[80], y=40,
                           font=dict(size=9, color=TV_GOLD), showarrow=False, row=2, col=1)

    fig.update_layout(
        height=600,
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_PANEL,
        legend=dict(bgcolor=TV_PANEL, bordercolor=TV_BORDER,
                    font=dict(size=9, color=TV_TEXT), orientation="h",
                    yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color=TV_TEXT, size=10),
        hovermode="x unified",
    )
    for i in range(1, rows+1):
        fig.update_xaxes(showgrid=True, gridcolor=TV_BORDER, zeroline=False, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=TV_BORDER, zeroline=False, row=i, col=1)

    key_safe = strategy.replace("+","_").replace(" ","_")
    st.plotly_chart(fig, use_container_width=True, key=f"bt_strategy_chart_{key_safe}")


def _plot_equity_and_dd(res: dict, strategy: str, sel_ticker: str,
                        df_ohlcv: pd.DataFrame) -> None:
    """Equity Curve + Drawdown: grafici identici al backtest originale."""
    # Equity curve
    fig_eq = go.Figure()
    bnh_norm = [100 * v / df_ohlcv["close"].iloc[0] for v in df_ohlcv["close"].values]
    fig_eq.add_trace(go.Scatter(
        x=res["bnh_dates"][:len(bnh_norm)], y=bnh_norm,
        mode="lines", name="Buy & Hold",
        line=dict(color=TV_GRAY, width=1.5, dash="dot"),
        hovertemplate="B&H: %{y:.1f}<extra></extra>",
    ))
    fig_eq.add_trace(go.Scatter(
        x=res["equity_dates"], y=res["equity"],
        mode="lines+markers", name=f"Strategia {strategy}",
        line=dict(color=TV_BLUE, width=2.5),
        marker=dict(size=5, color=TV_BLUE),
        fill="tonexty" if res["total_return"] > 0 else None,
        fillcolor="rgba(41,98,255,0.08)",
        hovertemplate="Equity: %{y:.1f}<extra></extra>",
    ))
    fig_eq.add_hline(y=100, line=dict(color=TV_BORDER, dash="dash", width=1))
    fig_eq.update_layout(
        title=dict(text=f"📈 <b>Equity Curve</b> — {sel_ticker} · {strategy}",
                   font=dict(size=13, color=TV_TEXT), x=0.01),
        height=320,
        paper_bgcolor=TV_BG, plot_bgcolor=TV_PANEL,
        legend=dict(bgcolor=TV_PANEL, bordercolor=TV_BORDER, font=dict(size=10, color=TV_TEXT)),
        xaxis=dict(showgrid=True, gridcolor=TV_BORDER, zeroline=False),
        yaxis=dict(title="Equity (base 100)", showgrid=True, gridcolor=TV_BORDER, zeroline=False),
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(color=TV_TEXT, size=10), hovermode="x unified",
    )
    key_safe = strategy.replace("+","_").replace(" ","_")
    st.plotly_chart(fig_eq, use_container_width=True, key=f"bt_equity_{key_safe}")

    # Drawdown
    eq_arr = np.array(res["equity"])
    peak   = np.maximum.accumulate(eq_arr)
    dd_arr = (eq_arr - peak) / peak * 100
    fig_dd = go.Figure(go.Scatter(
        x=res["equity_dates"], y=dd_arr,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(239,83,80,0.15)",
        line=dict(color=TV_RED, width=1.5),
        hovertemplate="DD: %{y:.1f}%<extra></extra>", name="Drawdown",
    ))
    fig_dd.update_layout(
        title=dict(text="📉 <b>Drawdown Strategia</b>",
                   font=dict(size=12, color=TV_TEXT), x=0.01),
        height=190, paper_bgcolor=TV_BG, plot_bgcolor=TV_PANEL,
        xaxis=dict(showgrid=True, gridcolor=TV_BORDER, zeroline=False),
        yaxis=dict(title="DD%", showgrid=True, gridcolor=TV_BORDER, ticksuffix="%"),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color=TV_TEXT, size=10), showlegend=False,
    )
    st.plotly_chart(fig_dd, use_container_width=True, key=f"bt_drawdown_{key_safe}")


def _plot_monthly_heatmap(res: dict, strategy: str) -> None:
    """Heatmap performance mensile identica al backtest originale."""
    monthly = res["monthly"]
    if monthly.empty:
        return
    years  = sorted(monthly["year"].unique())
    months = list(range(1, 13))
    month_names = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"]
    z_data = []
    for yr in years:
        row_data = []
        for mo in months:
            val = monthly[(monthly["year"]==yr) & (monthly["month"]==mo)]["pct"]
            row_data.append(float(val.values[0]) if len(val) > 0 else None)
        z_data.append(row_data)
    fig_hm = go.Figure(go.Heatmap(
        z=z_data, x=month_names, y=[str(y) for y in years],
        colorscale=[[0.0,"#ef5350"],[0.5,"#1e222d"],[1.0,"#26a69a"]],
        zmid=0,
        text=[[f"{v:+.1f}%" if v is not None else "—" for v in row] for row in z_data],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="Anno: %{y}<br>Mese: %{x}<br>PnL: %{z:+.1f}%<extra></extra>",
        colorbar=dict(title="PnL%", tickfont=dict(size=9, color=TV_TEXT),
                      bgcolor=TV_PANEL, bordercolor=TV_BORDER),
    ))
    fig_hm.update_layout(
        title=dict(text="🗓️ <b>Performance mensile (PnL% somma trade)</b>",
                   font=dict(size=12, color=TV_TEXT), x=0.01),
        height=max(180, 80 + len(years) * 55),
        paper_bgcolor=TV_BG, plot_bgcolor=TV_PANEL,
        xaxis=dict(side="top", tickfont=dict(size=10, color=TV_TEXT)),
        yaxis=dict(tickfont=dict(size=10, color=TV_TEXT)),
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color=TV_TEXT),
    )
    key_safe = strategy.replace("+","_").replace(" ","_")
    st.plotly_chart(fig_hm, use_container_width=True, key=f"bt_monthly_{key_safe}")


def _render_backtest(df: pd.DataFrame) -> None:
    """Modulo 6 — Backtest Avanzato."""
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_GREEN};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:12px">'
        f'<span style="color:{TV_GREEN};font-weight:700">📈 BACKTEST AVANZATO</span>'
        f'<span style="color:{TV_GRAY};font-size:0.78rem;margin-left:10px">'
        f'Equity curve · Sharpe · Max Drawdown · Win Rate · Heatmap mensile</span>'
        f'</div>', unsafe_allow_html=True
    )

    # ── Selectbox ticker: Nome (TICKER), ordinato alfabeticamente per nome
    ticker_nome = sorted(
        [(row["Ticker"], row["Nome"]) for _, row in df.iterrows()],
        key=lambda x: x[1].lower()
    )
    ticker_options  = [f"{n}  ({t})" for t, n in ticker_nome]
    ticker_map      = {f"{n}  ({t})": t for t, n in ticker_nome}
    name_map        = {f"{n}  ({t})": n for t, n in ticker_nome}

    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        sel_option = st.selectbox(
            "Azienda (ticker)",
            options=ticker_options,
            key="bt_ticker",
            help=f"{len(ticker_options)} aziende · ordinate alfabeticamente per nome"
        )
        sel_ticker = ticker_map[sel_option]
        sel_name   = name_map[sel_option]

    with c2:
        strategy = st.radio(
            "Strategia",
            ["DipScore", "Momentum", "RSI+VWAP", "ADX+EMA"],
            horizontal=True,
            key="bt_strategy",
        )
    with c3:
        st.write("")
        st.write("")
        run_bt = st.button("▶ Esegui", key="bt_run", use_container_width=True)

    # Info strategia selezionata
    strategy_info = {
        "DipScore":  ("💎", TV_CYAN,   "RSI < 45  +  Prezzo sotto EMA200",
                      "SL -8% · TP +15% · Hold max 20gg"),
        "Momentum":  ("⚡", TV_BLUE,   "MACD+ & EMA20 > EMA50 & Prezzo > EMA50",
                      "Exit su inversione MACD o EMA · SL -8% · TP +15%"),
        "RSI+VWAP":  ("📊", "#e91e63", "RSI attraversa 30 dal basso  +  Prezzo > VWAP",
                      "Exit: RSI attraversa 70 dall'alto o Prezzo < VWAP · SL -8%"),
        "ADX+EMA":   ("📈", TV_ORANGE, "EMA20 incrocia sopra EMA50  +  ADX > 25",
                      "Exit: EMA20 < EMA50 o ADX < 25 · SL -8% · TP +15%"),
    }
    icon, color, entry_txt, exit_txt = strategy_info[strategy]
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
            f'border-left:4px solid {color};border-radius:6px;padding:8px 12px;margin-bottom:10px">'
            f'<div style="color:{TV_GRAY};font-size:0.68rem">ENTRY {icon}</div>'
            f'<div style="color:{TV_TEXT};font-size:0.82rem;font-weight:600">{entry_txt}</div>'
            f'</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown(
            f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
            f'border-left:4px solid {TV_RED};border-radius:6px;padding:8px 12px;margin-bottom:10px">'
            f'<div style="color:{TV_GRAY};font-size:0.68rem">EXIT / PARAMETRI</div>'
            f'<div style="color:{TV_TEXT};font-size:0.82rem;font-weight:600">{exit_txt}</div>'
            f'</div>', unsafe_allow_html=True)

    if not run_bt:
        st.info("👆 Seleziona ticker e strategia, poi clicca **▶ Esegui**")
        return

    with st.spinner(f"⏳ Scaricando 2 anni di dati per {sel_ticker}..."):
        df_ohlcv = _fetch_ohlcv_2y(sel_ticker)

    if df_ohlcv.empty:
        st.error(f"⚠️ Nessun dato disponibile per {sel_ticker}")
        return

    with st.spinner("🔄 Esecuzione backtest..."):
        res = _run_backtest(df_ohlcv, strategy)

    if not res.get("ok"):
        st.warning(f"Nessun trade generato per {sel_ticker} con strategia {strategy}. "
                   f"Prova un altro ticker o cambia strategia.")
        return

    # ── KPI metriche ──────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    metrics = [
        (k1, "Win Rate",      f"{res['win_rate']:.1f}%",
         TV_GREEN if res['win_rate'] >= 50 else TV_RED),
        (k2, "Sharpe Ratio",  f"{res['sharpe']:.2f}",
         TV_GREEN if res['sharpe'] >= 1 else (TV_GOLD if res['sharpe'] >= 0 else TV_RED)),
        (k3, "Max Drawdown",  f"{res['max_dd']:.1f}%",
         TV_RED if res['max_dd'] < -15 else TV_GOLD),
        (k4, "Return Totale", f"{res['total_return']:+.1f}%",
         TV_GREEN if res['total_return'] > 0 else TV_RED),
        (k5, "N° Trade",      str(res['total_trades']), TV_CYAN),
    ]
    for col, label, val, color in metrics:
        with col:
            st.markdown(
                f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
                f'border-top:3px solid {color};border-radius:6px;'
                f'padding:10px;text-align:center">'
                f'<div style="color:{TV_GRAY};font-size:0.68rem">{label}</div>'
                f'<div style="color:{color};font-size:1.4rem;font-weight:800">{val}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grafico strategia dedicato (con indicatori propri) ──
    st.markdown(
        f'<div style="color:{TV_GRAY};font-size:0.8rem;margin-bottom:6px">'
        f'📉 <b style="color:{TV_TEXT}">Grafico {strategy}</b> — '
        f'Candele con indicatori · Entry ▲ · Exit ▼</div>',
        unsafe_allow_html=True)
    _plot_strategy_chart(res, df_ohlcv, strategy, sel_ticker, sel_name)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity curve + Drawdown ────────────────────
    _plot_equity_and_dd(res, strategy, sel_ticker, df_ohlcv)

    # ── Heatmap mensile ────────────────────────────
    _plot_monthly_heatmap(res, strategy)

    # ── Lista trade ───────────────────────────────
    with st.expander("📋 Lista trade dettagliata"):
        df_t = res["trades"].copy()
        df_t["entry_date"] = pd.to_datetime(df_t["entry_date"]).dt.strftime("%d/%m/%Y")
        df_t["exit_date"]  = pd.to_datetime(df_t["exit_date"]).dt.strftime("%d/%m/%Y")
        df_t["pct"]        = df_t["pct"].round(2)
        df_t["win"]        = df_t["win"].map({True:"✅", False:"❌"})
        df_t.columns = ["Entrata","Uscita","Prezzo In","Prezzo Out",
                        "PnL%","Giorni","Ragione","Esito","Anno","Mese"]
        df_t = df_t[["Entrata","Uscita","Prezzo In","Prezzo Out",
                     "PnL%","Giorni","Ragione","Esito"]].reset_index(drop=True)

        def _color_pnl(v):
            return f"color: {'#26a69a' if v > 0 else '#ef5350'}; font-weight:700"

        styled_t = (df_t.style
            .applymap(_color_pnl, subset=["PnL%"])
            .format({"Prezzo In":"${:.2f}", "Prezzo Out":"${:.2f}", "PnL%":"{:+.2f}%"})
            .set_properties(**{"background-color": TV_PANEL, "color": TV_TEXT})
        )
        st.dataframe(styled_t, use_container_width=True)


# ── Entry point ───────────────────────────────────

def render_bluechip_dip():
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_GOLD};'
        f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:6px">'
        f'<span style="color:{TV_GOLD};font-weight:700;font-size:1rem">'
        f'💎 BLUE CHIP DIP SCREENER</span>'
        f'<span style="color:{TV_GRAY};font-size:0.8rem;margin-left:12px">'
        f'Top 100 aziende mondiali · Opportunità di rientro · v31.1</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="color:{TV_GRAY};font-size:0.8rem;margin-bottom:14px;'
        f'padding:8px 12px;background:{TV_PANEL};border-radius:4px;'
        f'border:1px solid {TV_BORDER}">'
        f'📌 <b style="color:{TV_TEXT}">Dip Score</b> = 40% drawdown 52w + 30% RSI oversold + 30% distanza EMA200. '
        f'Più alto = potenziale opportunità di rientro su aziende di qualità. '
        f'<b style="color:{TV_RED}">Non è un segnale di acquisto</b> — è un radar per identificare candidati da analizzare.'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Controlli ─────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        min_dd = st.slider("Drawdown minimo %", 5, 40, 10, 5,
                           key="bcd_min_dd",
                           help="Mostra solo aziende con drawdown >= questa soglia")
    with c2:
        max_rsi = st.slider("RSI massimo", 30, 75, 60, 5,
                            key="bcd_max_rsi",
                            help="Filtra via titoli in ipercomprato")
    with c3:
        top_n = st.slider("Top N risultati", 5, 100, 20, 5,
                          key="bcd_top_n")
    with c4:
        st.write("")
        st.write("")
        refresh = st.button("🔄 Aggiorna", key="bcd_refresh",
                            use_container_width=True)
        if refresh:
            st.cache_data.clear()
            st.rerun()

    # ── Scan ──────────────────────────────────────
    with st.spinner(f"📡 Scansione {len(BLUE_CHIPS)} Blue Chip globali... (cache 30 min)"):
        df = _scan_all()

    if df.empty:
        st.error("⚠️ Nessun dato disponibile. Controlla la connessione.")
        return

    # ── Filtri ────────────────────────────────────
    df_f = df[
        (df["_dd_raw"] >= min_dd) &
        (df["RSI"]     <= max_rsi)
    ].head(top_n).copy()

    if df_f.empty:
        st.warning(f"Nessun titolo con drawdown ≥{min_dd}% e RSI ≤{max_rsi}. Allarga i filtri.")
        return

    # ── Metriche sommario ─────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Titoli analizzati", len(df))
    m2.metric("Titoli filtrati",   len(df_f))
    m3.metric("Drawdown medio",    f"{df_f['_dd_raw'].mean():.1f}%")
    m4.metric("RSI medio",         f"{df_f['RSI'].mean():.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Vista ─────────────────────────────────────
    view = st.radio("Vista", ["📡 Momentum", "🔥 Heatmap Settoriale", "📈 Backtest", "🃏 Cards", "📋 Tabella", "📊 Scatter"],
                    horizontal=True, key="bcd_view")

    if view == "📡 Momentum":
        _render_momentum_dashboard(df_f)

    elif view == "🔥 Heatmap Settoriale":
        _render_sector_heatmap(df_f)

    elif view == "📈 Backtest":
        _render_backtest(df_f)

    elif view == "🃏 Cards":
        col_a, col_b = st.columns(2)
        for i, (_, row) in enumerate(df_f.iterrows()):
            with (col_a if i % 2 == 0 else col_b):
                _render_card(row, i + 1)

    elif view == "📋 Tabella":
        disp = df_f[[
            "Ticker","Nome","Prezzo","_dd_raw","RSI",
            "Dist EMA200%","Vol×","Quality","Dip Score"
        ]].copy()
        disp.columns = [
            "Ticker","Nome","Prezzo $","Drawdown %","RSI",
            "Dist EMA200%","Vol×","Quality","Dip Score"
        ]
        disp.index = range(1, len(disp)+1)

        def _color_dd(v):
            if v > 25: return "color: #ef5350; font-weight:700"
            if v > 15: return "color: #ff9800; font-weight:600"
            return "color: #ffd700"

        def _color_rsi(v):
            if v < 30: return "color: #26a69a; font-weight:700"
            if v < 45: return "color: #50c4e0"
            return "color: #787b86"

        styled = (disp.style
            .applymap(_color_dd,  subset=["Drawdown %"])
            .applymap(_color_rsi, subset=["RSI"])
            .format({
                "Prezzo $":    "${:.2f}",
                "Drawdown %":  "{:.1f}%",
                "RSI":         "{:.1f}",
                "Dist EMA200%":"{:.1f}%",
                "Vol×":        "{:.2f}x",
                "Dip Score":   "{:.1f}",
            })
            .set_properties(**{"background-color": TV_PANEL, "color": TV_TEXT})
        )
        st.dataframe(styled, use_container_width=True)

    else:  # Scatter
        fig = go.Figure()
        for _, row in df_f.iterrows():
            dd   = row["_dd_raw"]
            rsi  = row["RSI"]
            dip  = row["Dip Score"]
            sym  = row["Ticker"]
            nome = row["Nome"]
            color = (TV_GREEN if dip >= 60 else
                     TV_GOLD  if dip >= 35 else TV_GRAY)
            fig.add_trace(go.Scatter(
                x=[dd], y=[rsi],
                mode="markers+text",
                marker=dict(
                    size=max(8, dip * 0.3),
                    color=color,
                    line=dict(color=TV_BORDER, width=1),
                    opacity=0.85,
                ),
                text=[sym],
                textposition="top center",
                textfont=dict(size=9, color=TV_TEXT),
                hovertemplate=(
                    f"<b>{sym}</b> — {nome}<br>"
                    f"Drawdown: {dd:.1f}%<br>"
                    f"RSI: {rsi:.1f}<br>"
                    f"Dip Score: {dip:.0f}/100<br>"
                    f"<extra></extra>"
                ),
                showlegend=False,
            ))

        # Quadranti
        fig.add_vline(x=20, line=dict(color=TV_BORDER, dash="dot", width=1))
        fig.add_hline(y=40, line=dict(color=TV_BORDER, dash="dot", width=1))
        fig.add_annotation(x=35, y=25, text="🎯 Opportunità",
                           showarrow=False, font=dict(color=TV_GREEN, size=11))
        fig.add_annotation(x=8,  y=25, text="😴 Poco interessante",
                           showarrow=False, font=dict(color=TV_GRAY,  size=10))
        fig.add_annotation(x=35, y=65, text="⚡ Oversold profondo",
                           showarrow=False, font=dict(color=TV_ORANGE,size=10))

        fig.update_layout(
            title=dict(
                text="📊 <b>Scatter: Drawdown vs RSI</b>"
                     "  <span style='color:#787b86;font-size:0.85em'>"
                     "(dimensione bolla = Dip Score)</span>",
                font=dict(size=13, color=TV_TEXT), x=0.01
            ),
            height=500,
            paper_bgcolor=TV_BG,
            plot_bgcolor=TV_PANEL,
            xaxis=dict(title="Drawdown dal max 52w (%)", showgrid=True,
                       gridcolor=TV_BORDER, zeroline=False, ticksuffix="%"),
            yaxis=dict(title="RSI(14)", showgrid=True,
                       gridcolor=TV_BORDER, zeroline=False,
                       range=[0, 85]),
            margin=dict(l=10, r=10, t=50, b=10),
            font=dict(color=TV_TEXT, size=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="bcd_scatter")

    # ── Strategy Chart widget (presente in tutte le viste) ────────────────
    if not df_f.empty and "Ticker" in df_f.columns:
        try:
            from utils.backtest_tab import strategy_chart_widget as _scw
            _bcd_tkrs_sorted = [
                n for n, t in sorted(
                    [(row["Nome"], row["Ticker"]) for _, row in df_f.iterrows()],
                    key=lambda x: x[0].lower()
                )
            ]
            # Costruisci options "Nome  (TICKER)" nello stesso formato del Backtest
            _bcd_opts = sorted(
                [(row["Nome"], row["Ticker"]) for _, row in df_f.iterrows()],
                key=lambda x: x[0].lower()
            )
            _bcd_tickers = [t for _n, t in _bcd_opts]
            _bcd_labels  = {t: f"{n}  ({t})" for n, t in _bcd_opts}
            st.markdown("---")
            _scw(tickers=_bcd_tickers, key_suffix="BCD", ticker_labels=_bcd_labels)
        except Exception:
            pass

    # ── Footer ────────────────────────────────────
    st.markdown(
        f'<div style="color:{TV_GRAY};font-size:0.72rem;text-align:center;'
        f'margin-top:16px;padding-top:8px;border-top:1px solid {TV_BORDER}">'
        f'Dati: Yahoo Finance · Cache 30 min · Universe: {len(BLUE_CHIPS)} Blue Chip globali · '
        f'v31.1 · Aggiornato: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        f'</div>',
        unsafe_allow_html=True
    )

