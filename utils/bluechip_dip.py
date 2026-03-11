# -*- coding: utf-8 -*-
"""
bluechip_dip.py  —  💎 Blue Chip Dip Screener  v30.0
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

# ── Universe: Top 60 Blue Chip globali ───────────
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
    # Europa
    ("NESN.SW","Nestlé"),         ("NOVN.SW","Novartis"),
    ("ROG.SW", "Roche"),          ("ASML",   "ASML"),
    ("SAP",    "SAP"),            ("LVMH.PA","LVMH"),
    ("TTE",    "TotalEnergies"),  ("SIE.DE", "Siemens"),
    ("AIR.PA", "Airbus"),         ("OR.PA",  "L'Oréal"),
    # Asia / Altri
    ("TSM",   "TSMC"),            ("TM",    "Toyota"),
    ("BABA",  "Alibaba"),         ("NVO",   "Novo Nordisk"),
    ("SONY",  "Sony"),            ("UL",    "Unilever"),
    ("BTI",   "BAT"),             ("DEO",   "Diageo"),
    ("RIO",   "Rio Tinto"),       ("BP",    "BP"),
]

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
        for col, (abs_idx, row) in zip(cols, [(row_start + i, r) for i, (_, r) in enumerate(chunk)]):
            sym   = row["Ticker"]
            nome  = row["Nome"][:16]
            score = int(row["Momentum"])
            label = row["Mom Label"]
            color = row["Mom Color"]
            rsi   = row["RSI"]
            macd_h= row["MACD Hist"]
            dd    = row["_dd_raw"]
            tv_url= f"https://it.tradingview.com/chart/?symbol={sym.split('.')[0]}"

            with col:
                fig = _momentum_gauge(score, label, color,
                                      title=f"{sym}", height=175)
                st.plotly_chart(fig, use_container_width=True,
                                key=f"gauge_{abs_idx}")

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


# ── Entry point ───────────────────────────────────

def render_bluechip_dip():
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_GOLD};'
        f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:6px">'
        f'<span style="color:{TV_GOLD};font-weight:700;font-size:1rem">'
        f'💎 BLUE CHIP DIP SCREENER</span>'
        f'<span style="color:{TV_GRAY};font-size:0.8rem;margin-left:12px">'
        f'Top 60 aziende mondiali · Opportunità di rientro · v30.0</span>'
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
        top_n = st.slider("Top N risultati", 5, 60, 20, 5,
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
    view = st.radio("Vista", ["📡 Momentum", "🃏 Cards", "📋 Tabella", "📊 Scatter"],
                    horizontal=True, key="bcd_view")

    if view == "📡 Momentum":
        _render_momentum_dashboard(df_f)

    if view == "📡 Momentum":
        _render_momentum_dashboard(df_f)

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

    # ── Footer ────────────────────────────────────
    st.markdown(
        f'<div style="color:{TV_GRAY};font-size:0.72rem;text-align:center;'
        f'margin-top:16px;padding-top:8px;border-top:1px solid {TV_BORDER}">'
        f'Dati: Yahoo Finance · Cache 30 min · Universe: {len(BLUE_CHIPS)} Blue Chip globali · '
        f'Aggiornato: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        f'</div>',
        unsafe_allow_html=True
    )
