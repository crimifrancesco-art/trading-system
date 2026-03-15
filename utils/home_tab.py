# -*- coding: utf-8 -*-
"""
home_tab.py  --  Market Intelligence Home  v32.0
=================================================
Dashboard pre-trade professionale. Prima cosa da leggere ogni mattina.

Sezioni (ordine operativo da trader):
  0. REGIME BAR      -- VIX regime alert + risk-on/off (SEMPRE visibile)
  1. INDICI LIVE     -- S&P500, NASDAQ, Dow, Russell, BTC, Gold, Oil, DXY, VIX
  2. SPARKLINES      -- S&P500 / NASDAQ / BTC 90gg con MACD momentum
  3. BREADTH ROW     -- Fear&Greed | Market Breadth | RSI Distribution
  4. TOP SEGNALI     -- Top 8 setup con score composito, ATR stop, R:R calcolato
  5. CALENDARIO      -- Fed, CPI, NFP, Earnings -- regole operative
  6. ROTAZIONE       -- Heatmap settori ordinati per forza + bar chart
  7. CORRELAZIONI    -- Matrice correlazione risk-on vs risk-off (30gg)

Fonte dati: Yahoo Finance puro (urllib, nessuna dipendenza aggiuntiva).
"""

import json
import urllib.request
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ══════════════════════════════════════════════════════════════
# PALETTE TradingView Dark
# ══════════════════════════════════════════════════════════════
BG     = "#131722"
PANEL  = "#1e222d"
BORDER = "#2a2e39"
BLUE   = "#2962ff"
GREEN  = "#26a69a"
RED    = "#ef5350"
GOLD   = "#ffd700"
CYAN   = "#50c4e0"
GRAY   = "#787b86"
TEXT   = "#d1d4dc"
ORANGE = "#ff9800"
PURPLE = "#9c27b0"
LIME   = "#00e676"


# ══════════════════════════════════════════════════════════════
# DATA FETCH  (cache aggressiva, fallback robusto)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=180, show_spinner=False)
def _fetch_quote(symbol: str) -> dict:
    """Prezzo live + variazione % dal giorno precedente."""
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range=5d")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        meta   = result["meta"]
        price  = float(meta.get("regularMarketPrice") or 0)
        prev   = float(meta.get("chartPreviousClose") or price)
        chg    = (price - prev) / prev * 100 if prev else 0.0
        name   = meta.get("longName") or meta.get("shortName") or symbol
        closes = [c for c in
                  (result.get("indicators", {}).get("quote", [{}])[0].get("close", []) or [])
                  if c is not None]
        return {"symbol": symbol, "name": name, "price": price,
                "chg": round(chg, 2), "closes": closes, "ok": True}
    except Exception as e:
        return {"symbol": symbol, "name": symbol, "price": 0.0,
                "chg": 0.0, "closes": [], "ok": False, "err": str(e)}


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """OHLCV storico per grafici e calcoli."""
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range={days}d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        q  = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s"),
            "open":   q.get("open",  []),
            "high":   q.get("high",  []),
            "low":    q.get("low",   []),
            "close":  q.get("close", []),
            "volume": q.get("volume", []),
        }).dropna(subset=["close"])
        df["date"] = df["date"].dt.tz_localize(None)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# INDICATORI TECNICI
# ══════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi_last(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    g = d.clip(lower=0).rolling(n).mean()
    lo = (-d.clip(upper=0)).rolling(n).mean()
    r  = g / lo.replace(0, np.nan)
    v  = (100 - 100 / (1 + r)).dropna()
    return round(float(v.iloc[-1]), 1) if not v.empty else 50.0


def _rsi_series(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    lo = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / lo.replace(0, np.nan))


def _atr_last(df: pd.DataFrame, n: int = 14) -> float:
    if df.empty or len(df) < n + 2:
        return 0.0
    h = df["high"]; lo = df["low"]; c = df["close"]
    tr = pd.concat([h - lo,
                    (h - c.shift(1)).abs(),
                    (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    return round(float(tr.ewm(span=n, adjust=False).mean().iloc[-1]), 4)


def _macd_hist(s: pd.Series) -> pd.Series:
    ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    return ml - ml.ewm(span=9).mean()


# ══════════════════════════════════════════════════════════════
# SEZIONE 0 -- REGIME BAR  (VIX-based)
# ══════════════════════════════════════════════════════════════

def _vix_regime(vix: float) -> tuple:
    """
    Regime di mercato basato sul VIX -- soglie istituzionali standard.
    Ritorna (label, hex_color, emoji, advice_operativo)
    """
    if vix >= 40:
        return "CRISI / PANIC",  RED,    "🔴", \
               "STOP operativita' attiva. Riduci size -75%. Solo hedge e cash."
    elif vix >= 30:
        return "ALTA VOL",       RED,    "🟠", \
               "Size -50%. Allarga stop ATR×2. Evita nuove aperture."
    elif vix >= 20:
        return "VOL ELEVATA",    GOLD,   "🟡", \
               "ATR stop standard. Attenzione a falsi breakout. No pyramiding."
    elif vix >= 15:
        return "NORMALE",        GREEN,  "🟢", \
               "Condizioni ideali. Sizing pieno. Swing e momentum attivi."
    else:
        return "BASSA VOL",      CYAN,   "💙", \
               "VIX basso: possibile compiacenza. Squeeze e breakout favoriti."


def _render_regime_bar(vix_price: float, sp500_chg: float, btc_chg: float):
    regime, color, em, advice = _vix_regime(vix_price)

    # Risk mode: risk-on se equity+crypto entrambi positivi
    if sp500_chg > 0 and btc_chg > 0:
        risk_signal, risk_color = "RISK-ON",  GREEN
    elif sp500_chg < 0 and btc_chg < 0:
        risk_signal, risk_color = "RISK-OFF", RED
    else:
        risk_signal, risk_color = "MISTO",    GOLD

    sp_arr  = "▲" if sp500_chg >= 0 else "▼"
    btc_arr = "▲" if btc_chg  >= 0 else "▼"
    sp_col  = GREEN if sp500_chg >= 0 else RED
    btc_col = GREEN if btc_chg  >= 0 else RED

    st.markdown(
        f'<div style="background:{PANEL};border:1px solid {color}44;'
        f'border-left:4px solid {color};border-radius:0 6px 6px 0;'
        f'padding:10px 18px;margin-bottom:10px;'
        f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'

        f'<div style="display:flex;align-items:center;gap:14px">'
        f'<div style="text-align:center">'
        f'<div style="color:{GRAY};font-size:0.62rem;font-weight:600;letter-spacing:0.05em">VIX</div>'
        f'<div style="color:{color};font-size:1.6rem;font-weight:800;'
        f'font-family:Courier New;line-height:1.1">{vix_price:.2f}</div>'
        f'</div>'
        f'<div>'
        f'<span style="background:{color}1a;color:{color};font-size:0.82rem;font-weight:700;'
        f'padding:4px 12px;border-radius:4px;border:1px solid {color}44">'
        f'{em} REGIME: {regime}</span>'
        f'<div style="color:{GRAY};font-size:0.71rem;margin-top:4px">{advice}</div>'
        f'</div>'
        f'</div>'

        f'<div style="display:flex;gap:22px;align-items:center">'
        f'<div style="text-align:center">'
        f'<div style="color:{GRAY};font-size:0.62rem;letter-spacing:0.05em">RISK MODE</div>'
        f'<div style="color:{risk_color};font-weight:700;font-size:0.92rem">{risk_signal}</div>'
        f'</div>'
        f'<div style="color:{sp_col};font-size:0.88rem;font-weight:600">'
        f'S&P500 {sp_arr} {abs(sp500_chg):.2f}%</div>'
        f'<div style="color:{btc_col};font-size:0.88rem;font-weight:600">'
        f'BTC {btc_arr} {abs(btc_chg):.2f}%</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
# SEZIONE 1 -- INDICI LIVE (9 asset chiave)
# ══════════════════════════════════════════════════════════════

INDICES = [
    {"sym": "^GSPC",     "label": "S&P 500",   "icon": "🇺🇸", "fmt": ",.2f"},
    {"sym": "^IXIC",     "label": "NASDAQ",    "icon": "💻",  "fmt": ",.2f"},
    {"sym": "^DJI",      "label": "Dow Jones", "icon": "🏦",  "fmt": ",.0f"},
    {"sym": "^RUT",      "label": "Russell2K", "icon": "📊",  "fmt": ",.2f"},
    {"sym": "^VIX",      "label": "VIX",       "icon": "😨",  "fmt": ".2f"},
    {"sym": "BTC-USD",   "label": "Bitcoin",   "icon": "₿",   "fmt": ",.0f", "prefix": "$"},
    {"sym": "GC=F",      "label": "Gold",      "icon": "🥇",  "fmt": ",.1f", "prefix": "$"},
    {"sym": "CL=F",      "label": "Oil WTI",   "icon": "🛢️", "fmt": ",.2f", "prefix": "$"},
    {"sym": "DX-Y.NYB",  "label": "DXY",       "icon": "💵",  "fmt": ".2f"},
]


def _render_indices():
    st.markdown(
        f'<div style="background:{PANEL};border-left:3px solid {BLUE};'
        f'padding:6px 14px;border-radius:0 4px 4px 0;margin-bottom:8px">'
        f'<span style="color:{BLUE};font-weight:700">📊 MERCATI LIVE</span>'
        f'<span style="color:{GRAY};font-size:0.74rem;margin-left:12px">'
        f'{datetime.now().strftime("%d/%m/%Y  %H:%M")}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    cols = st.columns(len(INDICES))
    for col, idx in zip(cols, INDICES):
        q     = _fetch_quote(idx["sym"])
        p     = q["price"]
        chg   = q["chg"]
        color = GREEN if chg >= 0 else RED
        arrow = "▲" if chg >= 0 else "▼"
        prefix = idx.get("prefix", "")
        price_str = f"{prefix}{p:{idx['fmt']}}"
        border_top = _vix_regime(p)[1] if idx["sym"] == "^VIX" else color

        with col:
            st.markdown(
                f'<div style="background:{PANEL};border:1px solid {BORDER};'
                f'border-top:2px solid {border_top};border-radius:6px;'
                f'padding:8px 8px;text-align:center">'
                f'<div style="color:{GRAY};font-size:0.62rem">{idx["icon"]} {idx["label"]}</div>'
                f'<div style="color:{TEXT};font-size:1.0rem;font-weight:700;'
                f'font-family:Courier New;white-space:nowrap">{price_str}</div>'
                f'<div style="color:{color};font-size:0.76rem;font-weight:600">'
                f'{arrow} {chg:+.2f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════
# SEZIONE 2 -- SPARKLINES (S&P / NASDAQ / BTC + MACD)
# ══════════════════════════════════════════════════════════════

def _render_sparklines():
    """90 giorni normalizzati % + EMA20 + MACD histogram."""
    symbols = [
        ("^GSPC",   "S&P 500", GREEN),
        ("^IXIC",   "NASDAQ",  BLUE),
        ("BTC-USD", "Bitcoin", GOLD),
    ]
    fig = make_subplots(
        rows=2, cols=3, shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.03, horizontal_spacing=0.05,
        subplot_titles=[s[1] for s in symbols] + ["", "", ""],
    )
    for i, (sym, label, color) in enumerate(symbols, 1):
        df = _fetch_history(sym, 90)
        if df.empty:
            continue
        c     = df["close"]
        dates = df["date"].tolist()
        base  = float(c.dropna().iloc[0])
        norm  = (c / base - 1) * 100
        chg   = float(norm.iloc[-1])
        chg_c = GREEN if chg >= 0 else RED
        mh    = _macd_hist(c)
        e20   = (_ema(c, 20) / base - 1) * 100

        try:
            r_, g_, b_ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill_c = f"rgba({r_},{g_},{b_},0.10)"
        except Exception:
            fill_c = "rgba(38,166,154,0.10)"

        fig.add_trace(go.Scatter(
            x=dates, y=norm, mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=fill_c,
            name=label, showlegend=False,
            hovertemplate=f"<b>{label}</b>: %{{y:.2f}}%<extra></extra>",
        ), row=1, col=i)

        fig.add_trace(go.Scatter(
            x=dates, y=e20, mode="lines",
            line=dict(color=GRAY, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=i)

        atr_v = _atr_last(df)
        fig.add_annotation(
            text=f"{'▲' if chg>=0 else '▼'}{abs(chg):.1f}%  ATR:{atr_v:.1f}",
            xref=f"x{'' if i==1 else i} domain",
            yref=f"y{'' if i==1 else i} domain",
            x=0.04, y=0.93, showarrow=False,
            font=dict(size=11, color=chg_c, family="Courier New"),
            xanchor="left",
        )

        hist_colors = [GREEN if v >= 0 else RED for v in mh]
        fig.add_trace(go.Bar(
            x=dates, y=mh.tolist(),
            marker_color=hist_colors, marker_line_width=0,
            opacity=0.8, showlegend=False,
            hovertemplate="MACD: %{y:.3f}<extra></extra>",
        ), row=2, col=i)
        fig.add_hline(y=0, row=2, col=i, line=dict(color=BORDER, width=1))

    fig.update_layout(
        height=255, paper_bgcolor=BG, plot_bgcolor=PANEL,
        margin=dict(l=0, r=0, t=26, b=0),
        showlegend=False, font=dict(color=TEXT, size=9), bargap=0.1,
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, showticklabels=False)
    st.plotly_chart(fig, use_container_width=True, key="home_sparklines_v32")


# ══════════════════════════════════════════════════════════════
# SEZIONE 3 -- BREADTH ROW
# ══════════════════════════════════════════════════════════════

def _calc_breadth(df: Optional[pd.DataFrame]) -> dict:
    r = {"above_ema200": 0, "total": 0, "pct": 0.0,
         "rsi_avg": 50.0, "rsi_above60": 0.0,
         "squeeze_pct": 0.0, "bull_weekly_pct": 0.0,
         "strong_pct": 0.0, "liq_pct": 0.0}
    if df is None or df.empty:
        return r
    total = len(df)
    r["total"] = total
    if total == 0:
        return r

    if "EMA200" in df.columns and "Prezzo" in df.columns:
        above = (pd.to_numeric(df["Prezzo"], errors="coerce") >
                 pd.to_numeric(df["EMA200"],  errors="coerce")).sum()
        r["above_ema200"] = int(above)
        r["pct"] = round(above / total * 100, 1)

    if "RSI" in df.columns:
        rsi_s = pd.to_numeric(df["RSI"], errors="coerce").dropna()
        if not rsi_s.empty:
            r["rsi_avg"]     = round(float(rsi_s.mean()), 1)
            r["rsi_above60"] = round(float((rsi_s > 60).mean() * 100), 1)

    if "Squeeze" in df.columns:
        r["squeeze_pct"] = round(float(df["Squeeze"].astype(bool).mean() * 100), 1)
    if "Weekly_Bull" in df.columns:
        r["bull_weekly_pct"] = round(float(df["Weekly_Bull"].astype(bool).mean() * 100), 1)
    if "Stato_Pro" in df.columns:
        r["strong_pct"] = round(float((df["Stato_Pro"] == "STRONG").mean() * 100), 1)
    if "Liq_OK" in df.columns:
        r["liq_pct"] = round(float(df["Liq_OK"].astype(bool).mean() * 100), 1)
    return r


def _fear_greed(vix: float, sp_rsi: float, breadth_pct: float) -> tuple:
    vix_score = max(0, 100 - (max(10, min(80, vix)) - 10) * (100 / 70))
    score = int(0.25 * vix_score + 0.40 * sp_rsi + 0.35 * breadth_pct)
    score = max(0, min(100, score))
    if score >= 75:   label, color = "🤑 Extreme Greed", GREEN
    elif score >= 55: label, color = "😊 Greed",         "#66bb6a"
    elif score >= 45: label, color = "😐 Neutral",       GOLD
    elif score >= 25: label, color = "😟 Fear",          ORANGE
    else:             label, color = "😱 Extreme Fear",  RED
    return score, label, color


def _render_fear_greed_card(score: int, label: str, color: str):
    st.markdown(
        f'<div style="background:{PANEL};border:1px solid {BORDER};'
        f'border-radius:8px;padding:16px">'
        f'<div style="color:{GRAY};font-size:0.7rem;margin-bottom:6px;'
        f'font-weight:600;letter-spacing:0.05em">FEAR & GREED INDEX</div>'
        f'<div style="font-size:2.5rem;font-weight:800;color:{color};'
        f'font-family:Courier New;line-height:1">{score}</div>'
        f'<div style="font-size:0.86rem;color:{color};margin:5px 0">{label}</div>'
        f'<div style="background:{BORDER};border-radius:4px;height:7px;margin-top:8px">'
        f'<div style="background:{color};width:{score}%;height:7px;border-radius:4px"></div>'
        f'</div>'
        f'<div style="color:{GRAY};font-size:0.68rem;margin-top:6px">'
        f'VIX + RSI S&P + Breadth</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def _render_breadth_card(breadth: dict):
    pct   = breadth["pct"]
    color = GREEN if pct >= 60 else (GOLD if pct >= 40 else RED)
    if pct >= 70:   sig = "🟢 BULL dominante"
    elif pct >= 55: sig = "🟡 BULL moderato"
    elif pct >= 45: sig = "⚪ Neutrale"
    elif pct >= 30: sig = "🟠 BEAR moderato"
    else:           sig = "🔴 BEAR dominante"

    st.markdown(
        f'<div style="background:{PANEL};border:1px solid {BORDER};border-radius:8px;padding:16px">'
        f'<div style="color:{GRAY};font-size:0.7rem;margin-bottom:5px;'
        f'font-weight:600;letter-spacing:0.05em">MARKET BREADTH</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<span style="color:{color};font-size:2rem;font-weight:800;font-family:Courier New">'
        f'{pct:.1f}%</span>'
        f'<span style="color:{GRAY};font-size:0.72rem">'
        f'{breadth["above_ema200"]}/{breadth["total"]} &gt; EMA200</span>'
        f'</div>'
        f'<div style="background:{BORDER};border-radius:4px;height:6px;margin:7px 0">'
        f'<div style="background:{color};width:{min(pct,100):.1f}%;height:6px;border-radius:4px"></div>'
        f'</div>'
        f'<div style="color:{color};font-size:0.82rem;font-weight:700;margin-bottom:10px">{sig}</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">'
        f'<span style="color:{GRAY};font-size:0.69rem">RSI avg: <b style="color:{TEXT}">{breadth["rsi_avg"]}</b></span>'
        f'<span style="color:{GRAY};font-size:0.69rem">RSI>60: <b style="color:{CYAN}">{breadth["rsi_above60"]}%</b></span>'
        f'<span style="color:{GRAY};font-size:0.69rem">Squeeze: <b style="color:{ORANGE}">{breadth["squeeze_pct"]}%</b></span>'
        f'<span style="color:{GRAY};font-size:0.69rem">Weekly+: <b style="color:{GREEN}">{breadth["bull_weekly_pct"]}%</b></span>'
        f'<span style="color:{GRAY};font-size:0.69rem">STRONG: <b style="color:{GOLD}">{breadth["strong_pct"]}%</b></span>'
        f'<span style="color:{GRAY};font-size:0.69rem">Liq OK: <b style="color:{LIME}">{breadth["liq_pct"]}%</b></span>'
        f'</div></div>',
        unsafe_allow_html=True
    )


def _render_rsi_distribution(df: Optional[pd.DataFrame]):
    """Istogramma distribuzione RSI -- snapshot visivo del mercato."""
    if df is None or df.empty or "RSI" not in df.columns:
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-radius:8px;padding:30px;text-align:center;'
            f'color:{GRAY};font-size:0.8rem">'
            f'Esegui lo scanner per vedere<br>la distribuzione RSI del mercato</div>',
            unsafe_allow_html=True
        )
        return

    rsi_vals = pd.to_numeric(df["RSI"], errors="coerce").dropna()
    if rsi_vals.empty:
        return

    bins    = list(range(0, 110, 10))
    counts, _ = np.histogram(rsi_vals, bins=bins)
    centers   = [(bins[i] + bins[i + 1]) // 2 for i in range(len(bins) - 1)]
    bar_colors = []
    for c in centers:
        if c <= 30:   bar_colors.append("rgba(38,166,154,0.85)")   # verde  -- oversold
        elif c >= 70: bar_colors.append("rgba(239,83,80,0.85)")    # rosso  -- overbought
        elif c >= 50: bar_colors.append("rgba(80,196,224,0.55)")   # cyan   -- bull zone
        else:         bar_colors.append("rgba(120,123,134,0.50)")  # grigio -- neutrale

    fig = go.Figure(go.Bar(
        x=centers, y=counts,
        marker_color=bar_colors, marker_line_width=0,
        hovertemplate="RSI %{x}±5: <b>%{y}</b> titoli<extra></extra>",
    ))
    for xv, clr, lbl in [(30, GREEN, "OS"), (70, RED, "OB")]:
        fig.add_vline(x=xv, line=dict(color=clr, width=1.5, dash="dot"),
                      annotation_text=f" {lbl}",
                      annotation_font=dict(color=clr, size=9))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        height=160, margin=dict(l=0, r=0, t=28, b=0),
        title=dict(text="Distribuzione RSI  (scanner corrente)",
                   font=dict(color=GRAY, size=11), x=0.02),
        xaxis=dict(range=[0, 100], dtick=10, gridcolor=BORDER,
                   title="RSI", tickfont=dict(size=9)),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(size=9)),
        font=dict(color=TEXT, size=9), bargap=0.06,
    )
    st.plotly_chart(fig, use_container_width=True, key="rsi_dist_v32")


# ══════════════════════════════════════════════════════════════
# SEZIONE 4 -- TOP SEGNALI OPERATIVI  (ATR stop + R:R)
# ══════════════════════════════════════════════════════════════

def _to_tv(sym: str) -> str:
    """Ticker Yahoo → formato TradingView."""
    if sym.endswith(".MI"):  return "MIL:"       + sym[:-3]
    if sym.endswith(".L"):   return "LSE:"       + sym[:-2]
    if sym.endswith(".PA"):  return "EURONEXT:"  + sym[:-3]
    if sym.endswith(".DE"):  return "XETRA:"     + sym[:-3]
    if sym.endswith(".AS"):  return "EURONEXT:"  + sym[:-3]
    if sym.endswith(".MC"):  return "BME:"       + sym[:-3]
    return sym


def _render_top_signals(df_ep: Optional[pd.DataFrame],
                         df_rea: Optional[pd.DataFrame],
                         n: int = 8):
    """
    Top N segnali con score composito professionale:
      Score = Pro*3 + Quality*2 + Early*1.5
            + WeeklyBull*4 + Squeeze*3 + STRONG_bonus*5
            + DollarVol_tier (0-3)
    Mostra: price, ATR stop (-1.5×ATR), T1 (+1.5×ATR R:1), T2 (+3×ATR R:2).
    Esclude titoli con earnings imminenti.
    """
    st.markdown(
        f'<div style="background:{PANEL};border-left:3px solid {GOLD};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:8px">'
        f'<span style="color:{GOLD};font-weight:700">🏆 TOP {n} SEGNALI OPERATIVI</span>'
        f'<span style="color:{GRAY};font-size:0.74rem;margin-left:10px">'
        f'Score composito · ATR stop · R:R calcolato · click ticker → TradingView</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    dfs = []
    if df_ep  is not None and not df_ep.empty:
        d = df_ep.copy();  d["_src"] = "EARLY/PRO"; dfs.append(d)
    if df_rea is not None and not df_rea.empty:
        d = df_rea.copy(); d["_src"] = "HOT";       dfs.append(d)

    if not dfs:
        st.info("🔍 Esegui lo scanner per vedere i segnali operativi.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Assicura colonne
    for col, val in [("Early_Score", 0), ("Quality_Score", 0), ("Pro_Score", 0),
                      ("Weekly_Bull", False), ("Squeeze", False),
                      ("Stato_Pro", ""), ("Dollar_Vol", 0),
                      ("ATR_pct", 0), ("ATR", 0), ("Prezzo", 0),
                      ("Earnings_Soon", False)]:
        if col not in df.columns:
            df[col] = val

    def _dvs(v):
        v = float(v) if pd.notna(v) else 0
        return 3 if v >= 50 else 2 if v >= 20 else 1 if v >= 5 else 0

    df["_dvs"]   = df["Dollar_Vol"].apply(_dvs)
    df["_score"] = (
        pd.to_numeric(df["Pro_Score"],     errors="coerce").fillna(0) * 3.0 +
        pd.to_numeric(df["Quality_Score"], errors="coerce").fillna(0) * 2.0 +
        pd.to_numeric(df["Early_Score"],   errors="coerce").fillna(0) * 1.5 +
        df["Weekly_Bull"].astype(bool) * 4.0 +
        df["Squeeze"].astype(bool)     * 3.0 +
        (df["Stato_Pro"] == "STRONG")  * 5.0 +
        df["_dvs"]
    )

    # Escludi earnings imminenti (gate obbligatorio)
    df_safe = df[~df["Earnings_Soon"].astype(bool)].copy()
    if df_safe.empty:
        df_safe = df.copy()

    top = df_safe.nlargest(n, "_score")

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        tkr     = str(row.get("Ticker", ""))
        nome    = str(row.get("Nome", ""))[:26]
        price   = float(row.get("Prezzo", 0) or 0)
        rsi     = row.get("RSI", "—")
        qual    = row.get("Quality_Score", "—")
        pro     = row.get("Pro_Score", "—")
        stato   = str(row.get("Stato_Pro", ""))
        atr     = float(row.get("ATR", 0) or 0)
        atr_pct = float(row.get("ATR_pct", 0) or 0)
        dvol    = float(row.get("Dollar_Vol", 0) or 0)
        src     = str(row.get("_src", ""))
        score   = float(row.get("_score", 0))
        sq      = bool(row.get("Squeeze", False))
        wb      = bool(row.get("Weekly_Bull", False))

        # ATR stop + targets
        if price > 0 and atr > 0:
            sl   = round(price - 1.5 * atr, 2)
            t1   = round(price + 1.5 * atr, 2)
            t2   = round(price + 3.0 * atr, 2)
            slp  = (sl - price) / price * 100
            t1p  = (t1 - price) / price * 100
            t2p  = (t2 - price) / price * 100
            risk_html = (
                f'<span style="color:{RED};font-size:0.71rem">'
                f'SL ${sl:.2f} ({slp:+.1f}%)</span>'
                f'&ensp;'
                f'<span style="color:{ORANGE};font-size:0.71rem">'
                f'T1 ${t1:.2f} ({t1p:+.1f}%) R:1</span>'
                f'&ensp;'
                f'<span style="color:{GREEN};font-size:0.71rem">'
                f'T2 ${t2:.2f} ({t2p:+.1f}%) R:2</span>'
            )
        else:
            risk_html = f'<span style="color:{GRAY};font-size:0.71rem">ATR non calcolato</span>'

        # Badge stato
        if stato == "STRONG":
            badge = (f'<span style="background:{GOLD}22;color:{GOLD};font-size:0.63rem;'
                     f'padding:2px 8px;border-radius:3px;border:1px solid {GOLD}55">★ STRONG</span>')
        elif stato == "PRO":
            badge = (f'<span style="background:{GREEN}22;color:{GREEN};font-size:0.63rem;'
                     f'padding:2px 8px;border-radius:3px;border:1px solid {GREEN}55">✦ PRO</span>')
        else:
            badge = ""

        src_color  = GOLD if "EARLY" in src else (RED if "HOT" in src else BLUE)
        rank_color = GOLD if rank == 1 else (TEXT if rank <= 3 else GRAY)
        dvol_str   = f"${dvol:.1f}M" if dvol >= 1 else f"${dvol*1000:.0f}K"
        dvol_col   = GREEN if dvol >= 20 else (ORANGE if dvol >= 5 else RED)
        atr_col    = GREEN if 1.5 <= atr_pct <= 6.0 else GOLD
        sq_b = f'<span style="color:{ORANGE}"> 🔥</span>' if sq else ""
        wb_b = f'<span style="color:{GREEN}"> 📈</span>'  if wb else ""
        tv_url = f"https://it.tradingview.com/chart/?symbol={_to_tv(tkr)}"

        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-left:3px solid {src_color};border-radius:6px;'
            f'padding:10px 14px;margin-bottom:5px">'

            # Riga 1: rank + ticker + nome + badges
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="color:{rank_color};font-size:0.88rem;font-weight:800;'
            f'font-family:Courier New;min-width:22px">#{rank}</span>'
            f'<a href="{tv_url}" target="_blank" style="text-decoration:none">'
            f'<span style="color:{CYAN};font-weight:700;font-size:0.95rem">{tkr} 🔗</span></a>'
            f'<span style="color:{GRAY};font-size:0.76rem">{nome}</span>'
            f'{sq_b}{wb_b}'
            f'</div>'
            f'<div style="display:flex;gap:6px;align-items:center">'
            f'{badge}'
            f'<span style="background:{src_color}2a;color:{src_color};font-size:0.62rem;'
            f'padding:2px 7px;border-radius:3px">{src}</span>'
            f'<span style="color:{TEXT};font-weight:700;font-family:Courier New;font-size:0.95rem">'
            f'${price:.2f}</span>'
            f'</div>'
            f'</div>'

            # Riga 2: metriche
            f'<div style="display:flex;gap:14px;margin-top:5px;flex-wrap:wrap">'
            f'<span style="color:{GRAY};font-size:0.71rem">RSI <b style="color:{TEXT}">{rsi}</b></span>'
            f'<span style="color:{GRAY};font-size:0.71rem">Q <b style="color:{CYAN}">{qual}/12</b></span>'
            f'<span style="color:{GRAY};font-size:0.71rem">Pro <b style="color:{GREEN}">{pro}</b></span>'
            f'<span style="color:{GRAY};font-size:0.71rem">DolVol <b style="color:{dvol_col}">{dvol_str}</b></span>'
            f'<span style="color:{GRAY};font-size:0.71rem">ATR% <b style="color:{atr_col}">{atr_pct:.1f}%</b></span>'
            f'<span style="color:{GRAY};font-size:0.71rem">Score <b style="color:{GOLD}">{score:.0f}</b></span>'
            f'</div>'

            # Riga 3: risk levels
            f'<div style="margin-top:5px">{risk_html}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
# SEZIONE 5 -- CALENDARIO MACRO
# ══════════════════════════════════════════════════════════════

def _render_macro_calendar():
    st.markdown(
        f'<div style="background:{PANEL};border-left:3px solid {PURPLE};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:8px">'
        f'<span style="color:{PURPLE};font-weight:700">📅 CALENDARIO MACRO</span>'
        f'<span style="color:{GRAY};font-size:0.74rem;margin-left:10px">'
        f'Regole operative per eventi ad alto impatto</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    events = [
        ("Ogni ~6 sett.",  "FOMC Fed Decision",     "🔴 ALTO",
         "Non entrare nelle 24h precedenti. Volatilita' spike. Chiudi posizioni aperte."),
        ("1° Ven. mese",   "NFP Jobs Report",        "🔴 ALTO",
         "Non-Farm Payrolls — muove S&P, DXY e bond. Stop allargati giorno prima."),
        ("~10-15 mese",    "CPI / Inflazione USA",   "🔴 ALTO",
         "Consumer Price Index — impatta growth stocks e Nasdaq. Evita grandi long."),
        ("Trimestrale",    "Earnings Season",         "🟠 MEDIO",
         "Q1:Apr | Q2:Lug | Q3:Ott | Q4:Gen — No posizioni sui singoli titoli in earnings week."),
        ("Ogni 2 settimane","ECB / Banca Centrale EU","🟠 MEDIO",
         "Impatta Euro Stoxx e titoli europei. Attenzione a .MI .PA .DE .AS"),
        ("Mensile",        "Retail Sales + PPI",     "🟡 BASSO",
         "Consumer spending + Producer Prices — segnali secondari. Ridurre size +20%."),
    ]

    rows_html = ""
    for freq, ev, imp, note in events:
        ic = RED if "ALTO" in imp else (ORANGE if "MEDIO" in imp else GOLD)
        rows_html += (
            f'<tr>'
            f'<td style="color:{GRAY};font-size:0.72rem;padding:5px 8px;'
            f'border-bottom:1px solid {BORDER};white-space:nowrap">{freq}</td>'
            f'<td style="color:{TEXT};font-size:0.76rem;font-weight:600;'
            f'padding:5px 8px;border-bottom:1px solid {BORDER};white-space:nowrap">{ev}</td>'
            f'<td style="color:{ic};font-size:0.7rem;padding:5px 8px;'
            f'border-bottom:1px solid {BORDER};white-space:nowrap">{imp}</td>'
            f'<td style="color:{GRAY};font-size:0.7rem;padding:5px 8px;'
            f'border-bottom:1px solid {BORDER}">{note}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="background:{PANEL};border:1px solid {BORDER};'
        f'border-radius:6px;padding:10px;overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr>'
        f'<th style="color:{GRAY};font-size:0.66rem;text-align:left;padding:4px 8px;'
        f'border-bottom:1px solid {BORDER}">FREQ.</th>'
        f'<th style="color:{GRAY};font-size:0.66rem;text-align:left;padding:4px 8px;'
        f'border-bottom:1px solid {BORDER}">EVENTO</th>'
        f'<th style="color:{GRAY};font-size:0.66rem;text-align:left;padding:4px 8px;'
        f'border-bottom:1px solid {BORDER}">IMPATTO</th>'
        f'<th style="color:{GRAY};font-size:0.66rem;text-align:left;padding:4px 8px;'
        f'border-bottom:1px solid {BORDER}">REGOLA OPERATIVA</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
        f'<div style="color:{GRAY};font-size:0.67rem;margin-top:8px">'
        f'⚠️ Regola generale: eventi ALTO → -50% size o flat. '
        f'Mai entrare nelle 2h precedenti un release macro critico.</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
# SEZIONE 6 -- HEATMAP SETTORI (con ranking + bar chart)
# ══════════════════════════════════════════════════════════════

SECTOR_ETFS = [
    ("XLK",  "Tech"),        ("XLF",  "Finance"),
    ("XLV",  "Healthcare"),  ("XLE",  "Energy"),
    ("XLI",  "Industrial"),  ("XLY",  "Cons.Discr"),
    ("XLP",  "Cons.Stpl"),   ("XLB",  "Materials"),
    ("XLRE", "Real Estate"), ("XLU",  "Utilities"),
    ("XLC",  "Comm.Srv"),    ("XBI",  "Biotech"),
]


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_sector_perf() -> list:
    results = []
    for sym, label in SECTOR_ETFS:
        q = _fetch_quote(sym)
        results.append({"label": label, "sym": sym,
                        "chg": q["chg"], "price": q["price"], "ok": q["ok"]})
    return sorted(results, key=lambda x: x["chg"], reverse=True)


def _render_sector_heatmap(sectors: list):
    st.markdown(
        f'<div style="background:{PANEL};border-left:3px solid {CYAN};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin-bottom:8px">'
        f'<span style="color:{CYAN};font-weight:700">🔥 ROTAZIONE SETTORIALE</span>'
        f'<span style="color:{GRAY};font-size:0.74rem;margin-left:10px">'
        f'ETF USA ordinati per forza — performance giornaliera</span>'
        f'<a href="https://it.tradingview.com/heatmap/stock/#%7B%22dataSource%22%3A%22SPX500%22%2C%22blockColor%22%3A%22change%22%2C%22blockSize%22%3A%22market_cap_basic%22%2C%22grouping%22%3A%22sector%22%7D"'
        f' target="_blank" style="color:{CYAN};font-size:0.71rem;margin-left:14px;'
        f'text-decoration:none">🔗 TradingView heatmap →</a>'
        f'</div>',
        unsafe_allow_html=True
    )

    def _bg_fg(chg):
        if chg >= 2:    return "#1a4a2a", GREEN
        elif chg >= 1:  return "#1e3d1e", "#66bb6a"
        elif chg >= 0:  return "#1b3330", "#81c784"
        elif chg >= -1: return "#3a1f1f", "#ef9a9a"
        elif chg >= -2: return "#4a1a1a", RED
        else:           return "#6a0000", "#ff5252"

    cols = st.columns(len(sectors))
    for col, s in zip(cols, sectors):
        bg, fg = _bg_fg(s["chg"])
        arr = "▲" if s["chg"] >= 0 else "▼"
        tv_url = f"https://it.tradingview.com/chart/?symbol={s['sym']}"
        with col:
            st.markdown(
                f'<a href="{tv_url}" target="_blank" style="text-decoration:none">'
                f'<div style="background:{bg};border:1px solid {BORDER};'
                f'border-radius:5px;padding:8px 3px;text-align:center;cursor:pointer">'
                f'<div style="color:{fg};font-size:0.62rem;font-weight:600">{s["label"]}</div>'
                f'<div style="color:{fg};font-size:0.82rem;font-weight:700">'
                f'{arr}{abs(s["chg"]):.1f}%</div>'
                f'</div></a>',
                unsafe_allow_html=True
            )

    # Bar chart espandibile
    with st.expander("📊 Ranking settori — bar chart", expanded=False):
        sdf = pd.DataFrame(sectors)
        if not sdf.empty:
            bar_colors = [GREEN if c >= 0 else RED for c in sdf["chg"]]
            fig = go.Figure(go.Bar(
                y=sdf["label"], x=sdf["chg"], orientation="h",
                marker_color=bar_colors, marker_line_width=0,
                hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",
            ))
            fig.add_vline(x=0, line=dict(color=BORDER, width=1))
            fig.update_layout(
                paper_bgcolor=BG, plot_bgcolor=PANEL,
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor=BORDER, ticksuffix="%", tickfont=dict(size=9)),
                yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
                font=dict(color=TEXT, size=9),
            )
            st.plotly_chart(fig, use_container_width=True, key="sector_bar_v32")


# ══════════════════════════════════════════════════════════════
# SEZIONE 7 -- CORRELAZIONI ASSET (30gg)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_corr_matrix() -> pd.DataFrame:
    assets = {
        "S&P500":  "^GSPC",
        "NASDAQ":  "^IXIC",
        "Bitcoin": "BTC-USD",
        "Gold":    "GC=F",
        "Oil":     "CL=F",
        "DXY":     "DX-Y.NYB",
        "VIX":     "^VIX",
    }
    closes = {}
    for name, sym in assets.items():
        df = _fetch_history(sym, 35)
        if not df.empty:
            closes[name] = df.set_index("date")["close"]
    if len(closes) < 3:
        return pd.DataFrame()
    return pd.DataFrame(closes).pct_change().dropna().corr().round(2)


def _render_correlations():
    with st.expander("🔗 Correlazioni Asset — 30 giorni", expanded=False):
        corr = _fetch_corr_matrix()
        if corr.empty:
            st.info("Dati correlazione non disponibili — attendi il caricamento.")
            return

        labels = list(corr.columns)
        z      = corr.values.tolist()

        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels,
            colorscale=[
                [0.0, "rgba(239,83,80,0.9)"],
                [0.5, "rgba(30,34,45,0.9)"],
                [1.0, "rgba(38,166,154,0.9)"],
            ],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=10, color=TEXT),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b>: %{z:.2f}<extra></extra>",
            showscale=False,
        ))
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PANEL,
            height=290, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(tickfont=dict(size=10, color=TEXT)),
            yaxis=dict(tickfont=dict(size=10, color=TEXT)),
            font=dict(color=TEXT, size=9),
        )
        st.plotly_chart(fig, use_container_width=True, key="corr_matrix_v32")
        st.markdown(
            f'<div style="color:{GRAY};font-size:0.7rem">'
            f'<span style="color:{GREEN}">▬ +1 = si muovono assieme (rischio correlato)</span> &nbsp;·&nbsp; '
            f'<span style="color:{RED}">▬ -1 = hedge naturale (si muovono opposti)</span> &nbsp;·&nbsp; '
            f'0 = scorrelati. Periodo: 30gg giornaliero.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def render_home(df_ep=None, df_rea=None):
    """Renderizza il tab Home completo -- v32.0."""

    c_title, c_ref = st.columns([9, 1])
    with c_title:
        st.markdown(
            f'<div style="background:{PANEL};border-left:3px solid {BLUE};'
            f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:10px">'
            f'<span style="color:{BLUE};font-weight:700;font-size:1rem">'
            f'🏠 MARKET INTELLIGENCE PRO</span>'
            f'<span style="color:{GRAY};font-size:0.8rem;margin-left:12px">'
            f'Dashboard pre-trade · v32.0 · {datetime.now().strftime("%d/%m/%Y")}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c_ref:
        st.write("")
        if st.button("🔄", key="home_refresh_v32", help="Svuota cache e aggiorna tutti i dati"):
            st.cache_data.clear()
            st.rerun()

    # 0 -- REGIME BAR
    vix_q = _fetch_quote("^VIX")
    sp_q  = _fetch_quote("^GSPC")
    btc_q = _fetch_quote("BTC-USD")
    _render_regime_bar(vix_q["price"], sp_q["chg"], btc_q["chg"])

    # 1 -- INDICI LIVE
    _render_indices()

    # 2 -- SPARKLINES
    _render_sparklines()

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # 3 -- BREADTH ROW
    dfs_scan = [d for d in [df_ep, df_rea] if d is not None and not d.empty]
    df_all   = pd.concat(dfs_scan, ignore_index=True) if dfs_scan else None

    sp_hist = _fetch_history("^GSPC", 30)
    sp_rsi  = _rsi_last(sp_hist["close"]) if not sp_hist.empty else 50.0
    breadth = _calc_breadth(df_all)
    fg_score, fg_label, fg_color = _fear_greed(vix_q["price"], sp_rsi, breadth["pct"])

    bc1, bc2, bc3 = st.columns([1, 1.2, 1.8])
    with bc1: _render_fear_greed_card(fg_score, fg_label, fg_color)
    with bc2: _render_breadth_card(breadth)
    with bc3: _render_rsi_distribution(df_all)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # 4+5 -- TOP SEGNALI  |  CALENDARIO MACRO
    sc, cc = st.columns([1.55, 1])
    with sc: _render_top_signals(df_ep, df_rea, n=8)
    with cc: _render_macro_calendar()

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # 6 -- HEATMAP SETTORIALE
    sectors = _fetch_sector_perf()
    _render_sector_heatmap(sectors)

    # 7 -- CORRELAZIONI
    _render_correlations()

    # FOOTER
    st.markdown(
        f'<div style="color:{GRAY};font-size:0.69rem;text-align:center;'
        f'margin-top:16px;padding-top:10px;border-top:1px solid {BORDER}">'
        f'Dati: Yahoo Finance · cache indici 3min · settori 10min · correlazioni 30min · '
        f'Ultimo aggiornamento: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        f'</div>',
        unsafe_allow_html=True
    )
