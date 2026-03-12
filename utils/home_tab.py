# -*- coding: utf-8 -*-
"""
home_tab.py  —  🏠 Market Intelligence Home  v30.0
════════════════════════════════════════════════════
Prima cosa che vedi aprendo l'app.

Sezioni:
  1. Indici Globali   — S&P500, NASDAQ, BTC, Gold, VIX live (Yahoo)
  2. Fear & Greed     — proxy RSI/Volatilità
  3. Market Breadth   — % titoli scannerizzati sopra EMA200
  4. Top 5 Segnali    — migliori setup del giorno dagli scanner
  5. Heatmap Settori  — performance settoriale (ETF proxy)

Tutti i dati vengono da Yahoo Finance (già dipendenza del progetto).
Nessuna API esterna aggiuntiva.
════════════════════════════════════════════════════
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Colori TV ─────────────────────────────────────
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

# ── Fetch Yahoo Finance ───────────────────────────

@st.cache_data(ttl=300, show_spinner=False)   # cache 5 min
def _fetch_quote(symbol: str) -> dict:
    """Scarica ultimo prezzo + variazione % da Yahoo Finance API v8."""
    try:
        import urllib.request, json
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range=2d")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        meta   = result["meta"]
        price  = meta.get("regularMarketPrice", 0)
        prev   = meta.get("chartPreviousClose", price)
        chg    = ((price - prev) / prev * 100) if prev else 0
        name   = meta.get("longName") or meta.get("shortName") or symbol
        # Storico 2 giorni per sparkline
        closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        closes = [c for c in closes if c is not None]
        return {
            "symbol": symbol, "name": name,
            "price": price, "chg": chg,
            "closes": closes, "ok": True,
        }
    except Exception as e:
        return {"symbol": symbol, "name": symbol, "price": 0,
                "chg": 0, "closes": [], "ok": False, "err": str(e)}


@st.cache_data(ttl=600, show_spinner=False)   # cache 10 min
def _fetch_history(symbol: str, days: int = 60) -> pd.DataFrame:
    """Storico OHLCV per sparkline e calcoli breadth."""
    try:
        import urllib.request, json
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range={days}d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts     = result["timestamp"]
        q      = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s"),
            "open":   q.get("open",  []),
            "high":   q.get("high",  []),
            "low":    q.get("low",   []),
            "close":  q.get("close", []),
            "volume": q.get("volume",[]),
        }).dropna(subset=["close"])
        return df
    except Exception:
        return pd.DataFrame()


# ── EMA helper ────────────────────────────────────

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _rsi(series: pd.Series, n: int = 14) -> float:
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    vals  = rsi.dropna()
    return round(float(vals.iloc[-1]), 1) if not vals.empty else 50.0


# ── Sezione 1 — Indici Globali ────────────────────

INDICES = [
    {"sym": "^GSPC",  "label": "S&P 500",  "icon": "🇺🇸"},
    {"sym": "^IXIC",  "label": "NASDAQ",   "icon": "💻"},
    {"sym": "^VIX",   "label": "VIX",      "icon": "😨"},
    {"sym": "BTC-USD","label": "Bitcoin",  "icon": "₿"},
    {"sym": "GC=F",   "label": "Gold",     "icon": "🥇"},
    {"sym": "DX-Y.NYB","label":"DXY",      "icon": "💵"},
]


def _render_indices():
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_BLUE};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:12px">'
        f'<span style="color:{TV_BLUE};font-weight:700">📊 MERCATI LIVE</span>'
        f'<span style="color:{TV_GRAY};font-size:0.78rem;margin-left:10px">'
        f'Aggiornato: {datetime.now().strftime("%H:%M:%S")}</span>'
        f'</div>', unsafe_allow_html=True
    )

    cols = st.columns(len(INDICES))
    for col, idx in zip(cols, INDICES):
        q = _fetch_quote(idx["sym"])
        price = q["price"]
        chg   = q["chg"]
        color = TV_GREEN if chg >= 0 else TV_RED
        arrow = "▲" if chg >= 0 else "▼"

        # Formato prezzo
        if idx["sym"] == "BTC-USD":
            price_str = f"${price:,.0f}"
        elif idx["sym"] == "GC=F":
            price_str = f"${price:,.1f}"
        elif idx["sym"] == "^VIX":
            price_str = f"{price:.2f}"
        else:
            price_str = f"{price:,.2f}"

        with col:
            st.markdown(
                f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
                f'border-radius:6px;padding:10px 12px;text-align:center;'
                f'border-top:2px solid {color}">'
                f'<div style="color:{TV_GRAY};font-size:0.72rem;margin-bottom:2px">'
                f'{idx["icon"]} {idx["label"]}</div>'
                f'<div style="color:{TV_TEXT};font-size:1.1rem;font-weight:700">'
                f'{price_str}</div>'
                f'<div style="color:{color};font-size:0.82rem;font-weight:600">'
                f'{arrow} {chg:+.2f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ── Sezione 2 — Fear & Greed proxy ───────────────

def _fear_greed_score(vix_chg: float, sp500_rsi: float,
                       breadth_pct: float) -> tuple:
    """
    Calcola un Fear & Greed proxy 0-100 basato su:
    - VIX (20%), RSI S&P500 (40%), Market Breadth (40%)
    """
    # VIX component: alto VIX = paura
    vix_score = max(0, min(100, 100 - (vix_chg + 20) * 2))
    # RSI S&P500: direttamente mappato
    rsi_score = sp500_rsi
    # Breadth: % sopra EMA200
    breadth_score = breadth_pct

    score = round(0.20 * vix_score + 0.40 * rsi_score + 0.40 * breadth_score)
    score = max(0, min(100, score))

    if score >= 75:   label, color = "🤑 Extreme Greed", TV_GREEN
    elif score >= 55: label, color = "😊 Greed",         "#66bb6a"
    elif score >= 45: label, color = "😐 Neutral",       TV_GOLD
    elif score >= 25: label, color = "😟 Fear",          "#ffa726"
    else:             label, color = "😱 Extreme Fear",  TV_RED

    return score, label, color


def _render_fear_greed(score: int, label: str, color: str):
    pct = score
    # Gauge arco SVG semplice
    st.markdown(
        f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
        f'border-radius:8px;padding:16px;text-align:center">'
        f'<div style="color:{TV_GRAY};font-size:0.75rem;margin-bottom:6px">'
        f'FEAR & GREED INDEX (proxy)</div>'
        f'<div style="font-size:2rem;font-weight:800;color:{color}">{score}</div>'
        f'<div style="font-size:0.9rem;color:{color};margin:4px 0">{label}</div>'
        f'<div style="background:{TV_BORDER};border-radius:4px;height:6px;margin-top:8px">'
        f'<div style="background:{color};width:{pct}%;height:6px;'
        f'border-radius:4px;transition:width 0.3s"></div></div>'
        f'<div style="color:{TV_GRAY};font-size:0.7rem;margin-top:4px">'
        f'Basato su VIX · RSI S&P500 · Market Breadth</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ── Sezione 3 — Market Breadth ────────────────────

def _calc_breadth(df_scanner: Optional[pd.DataFrame]) -> dict:
    """
    Calcola market breadth dagli scanner data.
    """
    result = {
        "above_ema200": 0, "total": 0, "pct": 0.0,
        "rsi_avg": 50.0, "squeeze_pct": 0.0,
        "bull_weekly_pct": 0.0,
    }
    if df_scanner is None or df_scanner.empty:
        return result

    df = df_scanner.copy()
    total = len(df)
    if total == 0:
        return result

    result["total"] = total

    # % sopra EMA200
    if "EMA200" in df.columns and "Prezzo" in df.columns:
        above = (df["Prezzo"] > df["EMA200"]).sum()
        result["above_ema200"] = int(above)
        result["pct"] = round(above / total * 100, 1)

    # RSI medio
    if "RSI" in df.columns:
        result["rsi_avg"] = round(float(df["RSI"].dropna().mean()), 1)

    # % in squeeze
    if "Squeeze" in df.columns:
        sqz = df["Squeeze"].astype(bool).sum()
        result["squeeze_pct"] = round(sqz / total * 100, 1)

    # % Weekly Bull
    if "Weekly_Bull" in df.columns:
        wb = df["Weekly_Bull"].astype(bool).sum()
        result["bull_weekly_pct"] = round(wb / total * 100, 1)

    return result


def _render_breadth(breadth: dict):
    pct  = breadth["pct"]
    tot  = breadth["total"]
    ab   = breadth["above_ema200"]
    color = TV_GREEN if pct >= 60 else (TV_GOLD if pct >= 40 else TV_RED)

    if pct >= 70:   signal = "🟢 Mercato Rialzista"
    elif pct >= 55: signal = "🟡 Lieve rialzista"
    elif pct >= 45: signal = "⚪ Neutrale"
    elif pct >= 30: signal = "🟠 Lieve ribassista"
    else:           signal = "🔴 Mercato Ribassista"

    st.markdown(
        f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
        f'border-radius:8px;padding:16px">'
        f'<div style="color:{TV_GRAY};font-size:0.75rem;margin-bottom:8px">'
        f'📊 MARKET BREADTH</div>'

        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:8px">'
        f'<span style="color:{TV_TEXT};font-size:1.6rem;font-weight:800;'
        f'color:{color}">{pct}%</span>'
        f'<span style="color:{TV_GRAY};font-size:0.8rem">{ab}/{tot} titoli<br>'
        f'sopra EMA200</span></div>'

        f'<div style="background:{TV_BORDER};border-radius:4px;height:8px;margin:8px 0">'
        f'<div style="background:{color};width:{min(pct,100)}%;height:8px;'
        f'border-radius:4px"></div></div>'

        f'<div style="color:{color};font-size:0.85rem;font-weight:600">'
        f'{signal}</div>'

        f'<div style="display:flex;gap:12px;margin-top:10px">'
        f'<div style="color:{TV_GRAY};font-size:0.75rem">'
        f'RSI medio: <b style="color:{TV_TEXT}">{breadth["rsi_avg"]}</b></div>'
        f'<div style="color:{TV_GRAY};font-size:0.75rem">'
        f'Squeeze: <b style="color:{TV_CYAN}">{breadth["squeeze_pct"]}%</b></div>'
        f'<div style="color:{TV_GRAY};font-size:0.75rem">'
        f'Weekly+: <b style="color:{TV_GREEN}">{breadth["bull_weekly_pct"]}%</b></div>'
        f'</div></div>',
        unsafe_allow_html=True
    )


# ── Sezione 4 — Top 5 Segnali ─────────────────────

def _render_top5(df_ep: Optional[pd.DataFrame],
                 df_rea: Optional[pd.DataFrame]):
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_GOLD};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:10px">'
        f'<span style="color:{TV_GOLD};font-weight:700">🏆 TOP 5 SEGNALI DEL GIORNO</span>'
        f'</div>', unsafe_allow_html=True
    )

    dfs = []
    if df_ep is not None and not df_ep.empty:
        df_ep2 = df_ep.copy(); df_ep2["_src"] = "EARLY"
        dfs.append(df_ep2)
    if df_rea is not None and not df_rea.empty:
        df_r2 = df_rea.copy(); df_r2["_src"] = "HOT"
        dfs.append(df_r2)

    if not dfs:
        st.info("🔍 Esegui lo scanner per vedere i top segnali.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Score composito: Early_Score + Quality_Score + Weekly_Bull bonus
    for col in ["Early_Score", "Quality_Score", "Pro_Score"]:
        if col not in df_all.columns:
            df_all[col] = 0
    if "Weekly_Bull" not in df_all.columns:
        df_all["Weekly_Bull"] = False
    if "Squeeze" not in df_all.columns:
        df_all["Squeeze"] = False

    df_all["_composite"] = (
        df_all["Early_Score"].fillna(0) * 2 +
        df_all["Quality_Score"].fillna(0) +
        df_all["Weekly_Bull"].astype(bool) * 3 +
        df_all["Squeeze"].astype(bool) * 2
    )

    top5 = df_all.nlargest(5, "_composite")

    for _, row in top5.iterrows():
        tkr   = row.get("Ticker", "")
        nome  = row.get("Nome", "")[:22]
        prezzo= row.get("Prezzo", "")
        rsi   = row.get("RSI", "")
        qual  = row.get("Quality_Score", "")
        early = row.get("Early_Score", "")
        src   = row.get("_src", "")
        sqz   = "🔥" if row.get("Squeeze") else ""
        wb    = "📅" if row.get("Weekly_Bull") else ""
        src_color = TV_GOLD if src == "EARLY" else TV_RED

        # Converti simbolo Yahoo → TradingView
        def _to_tv(sym):
            if sym.endswith(".MI"):  return "MIL:"  + sym[:-3]
            if sym.endswith(".L"):   return "LSE:"  + sym[:-2]
            if sym.endswith(".PA"):  return "EURONEXT:" + sym[:-3]
            if sym.endswith(".DE"):  return "XETRA:" + sym[:-3]
            if sym.endswith(".AS"):  return "EURONEXT:" + sym[:-3]
            return sym  # US ticker — nessuna modifica
        tv_url = f"https://it.tradingview.com/chart/?symbol={_to_tv(tkr)}"
        st.markdown(
            f'<div style="background:{TV_PANEL};border:1px solid {TV_BORDER};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;'
            f'border-left:3px solid {src_color}">'

            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center">'
            f'<div>'
            f'<a href="{tv_url}" target="_blank" style="text-decoration:none">'
            f'<span style="color:{TV_CYAN};font-weight:700;font-size:0.95rem;'
            f'cursor:pointer" title="Apri su TradingView">'
            f'{tkr} 🔗</span></a>'
            f'<span style="color:{TV_GRAY};font-size:0.78rem;margin-left:8px">'
            f'{nome}</span>'
            f'<span style="font-size:0.8rem;margin-left:6px">{sqz}{wb}</span>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<span style="color:{TV_TEXT};font-weight:600">${prezzo}</span>'
            f'<span style="background:{src_color};color:#fff;font-size:0.65rem;'
            f'padding:2px 6px;border-radius:3px;margin-left:8px">{src}</span>'
            f'</div></div>'

            f'<div style="display:flex;gap:16px;margin-top:6px">'
            f'<span style="color:{TV_GRAY};font-size:0.75rem">'
            f'RSI <b style="color:{TV_TEXT}">{rsi}</b></span>'
            f'<span style="color:{TV_GRAY};font-size:0.75rem">'
            f'Quality <b style="color:{TV_CYAN}">{qual}/12</b></span>'
            f'<span style="color:{TV_GRAY};font-size:0.75rem">'
            f'Early <b style="color:{TV_GOLD}">{early}</b></span>'
            f'</div></div>',
            unsafe_allow_html=True
        )


# ── Sezione 5 — Heatmap Settoriale ───────────────

SECTOR_ETFS = [
    ("XLK",  "Tech"),       ("XLF",  "Finance"),
    ("XLV",  "Healthcare"), ("XLE",  "Energy"),
    ("XLI",  "Industrial"), ("XLY",  "Consumer D."),
    ("XLP",  "Consumer S."),("XLB",  "Materials"),
    ("XLRE", "Real Estate"),("XLU",  "Utilities"),
    ("XLC",  "Comm. Serv."),
]


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_sector_perf() -> list:
    results = []
    for sym, label in SECTOR_ETFS:
        q = _fetch_quote(sym)
        results.append({
            "label": label, "sym": sym,
            "chg": q["chg"], "ok": q["ok"]
        })
    return results


def _render_sector_heatmap(sectors: list):
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_CYAN};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:10px">'
        f'<span style="color:{TV_CYAN};font-weight:700">🔥 HEATMAP SETTORIALE</span>'
        f'<span style="color:{TV_GRAY};font-size:0.78rem;margin-left:10px">'
        f'Performance giornaliera ETF settoriali USA</span>'
        f'</div>', unsafe_allow_html=True
    )

    def _color(chg):
        if chg >= 2:    return "#1b5e20", TV_GREEN
        elif chg >= 1:  return "#2e7d32", "#66bb6a"
        elif chg >= 0:  return "#1b3a2e", "#81c784"
        elif chg >= -1: return "#4a1a1a", "#ef9a9a"
        elif chg >= -2: return "#6a1a1a", TV_RED
        else:           return "#8b0000", "#ff5252"

    # Link TradingView Heatmap settoriale Italia
    tv_heatmap_url = "https://it.tradingview.com/heatmap/stock/#%7B%22dataSource%22%3A%22SPX500%22%2C%22blockColor%22%3A%22change%22%2C%22blockSize%22%3A%22market_cap_basic%22%2C%22grouping%22%3A%22sector%22%7D"
    st.markdown(
        f'<div style="text-align:right;margin-bottom:6px">'
        f'<a href="{tv_heatmap_url}" target="_blank" '
        f'style="color:{TV_CYAN};font-size:0.78rem;text-decoration:none">'
        f'🔗 Apri Heatmap su TradingView →</a></div>',
        unsafe_allow_html=True
    )

    cols = st.columns(len(sectors))
    for col, s in zip(cols, sectors):
        chg = s["chg"]
        bg, fg = _color(chg)
        arrow = "▲" if chg >= 0 else "▼"
        sym   = s["sym"]
        tv_url = f"https://it.tradingview.com/chart/?symbol={sym}"
        with col:
            st.markdown(
                f'<a href="{tv_url}" target="_blank" style="text-decoration:none">'
                f'<div style="background:{bg};border-radius:6px;'
                f'padding:10px 4px;text-align:center;'
                f'border:1px solid {TV_BORDER};cursor:pointer;'
                f'transition:border-color 0.2s" '
                f'onmouseover="this.style.borderColor=\'{fg}\'" '
                f'onmouseout="this.style.borderColor=\'{TV_BORDER}\'">'
                f'<div style="color:{fg};font-size:0.68rem;font-weight:600'
                f';margin-bottom:2px">{s["label"]}</div>'
                f'<div style="color:{fg};font-size:0.85rem;font-weight:700">'
                f'{arrow}{abs(chg):.1f}%</div>'
                f'</div></a>',
                unsafe_allow_html=True
            )


# ── Grafico sparkline indici ──────────────────────

def _render_sparklines():
    """
    Mini chart S&P500 / NASDAQ / BTC — 90 giorni.
    Row 1: linea close normalizzata % (fill area) + EMA20 punteggiata
    Row 2: istogramma MACD per il momentum
    """
    symbols = [
        ("^GSPC",   "S&P 500", TV_GREEN),
        ("^IXIC",   "NASDAQ",  TV_BLUE),
        ("BTC-USD", "Bitcoin", TV_GOLD),
    ]

    def _ema_s(s, n):
        return s.ewm(span=n, adjust=False).mean()

    def _macd_h(s):
        ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
        return ml - ml.ewm(span=9).mean()

    fig = make_subplots(
        rows=2, cols=3,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
        subplot_titles=[s[1] for s in symbols] + ["", "", ""],
    )

    for i, (sym, label, color) in enumerate(symbols, 1):
        df = _fetch_history(sym, days=90)
        if df.empty:
            continue

        c     = df["close"]
        dates = df["date"].tolist()
        base  = float(c.dropna().iloc[0])
        norm  = ((c / base - 1) * 100).tolist()
        chg   = norm[-1]
        chg_c = TV_GREEN if chg >= 0 else TV_RED

        mh = _macd_h(c)
        hist_colors = [TV_GREEN if v >= 0 else TV_RED for v in mh]

        # Row 1 — linea normalizzata + EMA20
        try:
            r, g, b_ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill_c = f"rgba({r},{g},{b_},0.10)"
        except Exception:
            fill_c = "rgba(38,166,154,0.10)"

        fig.add_trace(go.Scatter(
            x=dates, y=norm, mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=fill_c,
            name=label, showlegend=False,
            hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
        ), row=1, col=i)

        ema20_norm = ((_ema_s(c, 20) / base - 1) * 100).tolist()
        fig.add_trace(go.Scatter(
            x=dates, y=ema20_norm, mode="lines",
            line=dict(color=TV_GRAY, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=i)

        fig.add_annotation(
            text=f"{'▲' if chg>=0 else '▼'}{abs(chg):.1f}%",
            xref=f"x{'' if i==1 else i} domain",
            yref=f"y{'' if i==1 else i} domain",
            x=0.04, y=0.92,
            showarrow=False,
            font=dict(size=12, color=chg_c, family="monospace"),
            xanchor="left",
        )

        # Row 2 — MACD histogram
        fig.add_trace(go.Bar(
            x=dates, y=mh.tolist(),
            marker_color=hist_colors, marker_line_width=0,
            opacity=0.85, showlegend=False,
            hovertemplate="MACD: %{y:.3f}<extra></extra>",
        ), row=2, col=i)
        fig.add_hline(y=0, row=2, col=i,
            line=dict(color=TV_BORDER, width=1))

    fig.update_layout(
        height=240,
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_PANEL,
        margin=dict(l=0, r=0, t=28, b=0),
        showlegend=False,
        font=dict(color=TV_TEXT, size=10),
        bargap=0.1,
    )
    fig.update_xaxes(showgrid=False, showticklabels=False,
                     zeroline=False, linecolor=TV_BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=TV_BORDER,
                     zeroline=False, showticklabels=False)
    fig.update_annotations(font_color=TV_GRAY)
    st.plotly_chart(fig, use_container_width=True, key="home_sparklines")


# ── Entry point ───────────────────────────────────

def render_home(df_ep=None, df_rea=None):
    """Renderizza il tab Home completo."""

    # Refresh button
    col_title, col_ref = st.columns([8, 1])
    with col_title:
        st.markdown(
            f'<div style="background:{TV_PANEL};border-left:3px solid {TV_BLUE};'
            f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:14px">'
            f'<span style="color:{TV_BLUE};font-weight:700;font-size:1rem">'
            f'🏠 MARKET INTELLIGENCE</span>'
            f'<span style="color:{TV_GRAY};font-size:0.8rem;margin-left:12px">'
            f'Dashboard mercati in tempo reale · v30.0</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_ref:
        st.write("")
        if st.button("🔄", key="home_refresh", help="Aggiorna dati"):
            st.cache_data.clear()
            st.rerun()

    # ── Row 1: Indici live ─────────────────────────
    _render_indices()

    # ── Row 2: Sparklines 60gg ────────────────────
    _render_sparklines()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Fear&Greed + Breadth + Top5 ────────
    col_fg, col_br, col_top = st.columns([1, 1.4, 2])

    # Fear & Greed
    with col_fg:
        vix_q   = _fetch_quote("^VIX")
        sp_hist = _fetch_history("^GSPC", days=30)
        sp_rsi  = _rsi(sp_hist["close"]) if not sp_hist.empty else 50.0
        breadth = _calc_breadth(
            pd.concat([df for df in [df_ep, df_rea]
                       if df is not None and not df.empty], ignore_index=True)
            if any(df is not None and not df.empty for df in [df_ep, df_rea])
            else None
        )
        fg_score, fg_label, fg_color = _fear_greed_score(
            vix_q["chg"], sp_rsi, breadth["pct"]
        )
        _render_fear_greed(fg_score, fg_label, fg_color)

    # Market Breadth
    with col_br:
        _render_breadth(breadth)

    # Top 5 Segnali
    with col_top:
        _render_top5(df_ep, df_rea)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4: Heatmap Settoriale ─────────────────
    sectors = _fetch_sector_perf()
    _render_sector_heatmap(sectors)

    # ── Footer ────────────────────────────────────
    st.markdown(
        f'<div style="color:{TV_GRAY};font-size:0.72rem;text-align:center;'
        f'margin-top:20px;padding-top:10px;border-top:1px solid {TV_BORDER}">'
        f'Dati: Yahoo Finance · Aggiornamento automatico ogni 5 min · '
        f'Ultimo refresh: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        f'</div>',
        unsafe_allow_html=True
    )
