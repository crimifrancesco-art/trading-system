 # -*- coding: utf-8 -*-
"""
orderflow_tab.py  —  🔬 Order Flow  v31.1
Stile TradingView Dark · semplice e leggibile
"""

import json
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Colori TradingView Dark ───────────────────────────────────────────────────
BG     = "#131722"
PANEL  = "#1e222d"
BORDER = "#2a2e39"
GREEN  = "#26a69a"
RED    = "#ef5350"
GOLD   = "#f0b90b"
BLUE   = "#2962ff"
CYAN   = "#50c4e0"
GRAY   = "#787b86"
TEXT   = "#d1d4dc"
ORANGE = "#ff9800"
VWAP_C = "#ff6d00"
G_DARK = "rgba(38,166,154,0.15)"
R_DARK = "rgba(239,83,80,0.15)"

# ── Mappa Ticker → Nome completo ──────────────────────────────────────────────
NAMES = {
    # USA Large Cap
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMZN":"Amazon",
    "GOOGL":"Alphabet","META":"Meta","TSLA":"Tesla","AVGO":"Broadcom",
    "BRK-B":"Berkshire","LLY":"Eli Lilly","JPM":"JPMorgan","V":"Visa",
    "MA":"Mastercard","UNH":"UnitedHealth","XOM":"ExxonMobil",
    "JNJ":"J&J","WMT":"Walmart","PG":"Procter&Gamble","ORCL":"Oracle",
    "HD":"Home Depot","COST":"Costco","BAC":"Bank of America",
    "NFLX":"Netflix","KO":"Coca-Cola","CRM":"Salesforce","AMD":"AMD",
    "MRK":"Merck","CVX":"Chevron","PEP":"PepsiCo","ABBV":"AbbVie",
    "TMO":"Thermo Fisher","LIN":"Linde","ACN":"Accenture",
    "MCD":"McDonald's","PM":"Philip Morris","GE":"GE Aerospace",
    "NOW":"ServiceNow","CAT":"Caterpillar","IBM":"IBM","GS":"Goldman",
    "AMGN":"Amgen","T":"AT&T","MS":"Morgan Stanley","AXP":"Amex",
    "SPGI":"S&P Global","BLK":"BlackRock","RTX":"RTX","HON":"Honeywell",
    "DE":"John Deere","PFE":"Pfizer","ADBE":"Adobe","INTU":"Intuit",
    "QCOM":"Qualcomm","TXN":"Texas Instr.","PANW":"Palo Alto",
    # ETF
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF","IWM":"Russell 2000 ETF",
    "DIA":"Dow Jones ETF","GLD":"Gold ETF","SLV":"Silver ETF",
    "TLT":"20yr Bond ETF","HYG":"High Yield ETF","GDX":"Gold Miners ETF",
    # Futures
    "ES=F":"S&P 500 Fut","NQ=F":"Nasdaq Fut","YM=F":"Dow Fut",
    "RTY=F":"Russell Fut","CL=F":"Crude Oil Fut","GC=F":"Gold Fut",
    "SI=F":"Silver Fut","ZB=F":"Bond Fut",
    # Crypto
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana",
    # Forex
    "EUR=X":"EUR/USD","JPY=X":"USD/JPY","GBP=X":"GBP/USD",
    "EURUSD=X":"EUR/USD","GBPUSD=X":"GBP/USD",
    # Europa
    "ASML":"ASML","SAP":"SAP","NESN.SW":"Nestlé","NOVN.SW":"Novartis",
    "ROG.SW":"Roche","LVMH.PA":"LVMH","TTE":"TotalEnergies",
    "SIE.DE":"Siemens","AIR.PA":"Airbus","OR.PA":"L'Oréal",
    "AZN":"AstraZeneca","GSK":"GSK","BP":"BP",
    # Asia
    "TSM":"TSMC","TM":"Toyota","BABA":"Alibaba","NVO":"Novo Nordisk",
    "SONY":"Sony","UL":"Unilever",
}

# Ticker predefiniti nella selectbox (nome — ticker, ordinati per nome)
_DEFAULT_TICKERS = [
    "ES=F","NQ=F","SPY","QQQ","BTC-USD","ETH-USD",
    "AAPL","MSFT","NVDA","TSLA","META","AMZN","GOOGL",
    "GLD","GC=F","CL=F","EUR=X",
]

# Timeframe
TF_MAP = {
    "15min": ("2m",  "15min", "3d"),
    "30min": ("2m",  "30min", "5d"),
    "1h":    ("5m",  "60min", "10d"),
    "4h":    ("15m", "240min","30d"),
    "Daily": ("60m", "1D",    "180d"),
}


def _name(ticker: str) -> str:
    return NAMES.get(ticker, ticker)


def _label(ticker: str) -> str:
    """Restituisce 'Nome (TICKER)' oppure solo TICKER se nome non noto."""
    n = NAMES.get(ticker)
    return f"{n}  ({ticker})" if n else ticker


# ═════════════════════════════════════════════════════════════════════════════
# DATI
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _fetch(symbol: str, interval: str, range_: str) -> pd.DataFrame:
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval={interval}&range={range_}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        res = data["chart"]["result"][0]
        ts  = res["timestamp"]
        q   = res["indicators"]["quote"][0]
        df  = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s", utc=True).tz_localize(None),
            "open":   q.get("open",  [None]*len(ts)),
            "high":   q.get("high",  [None]*len(ts)),
            "low":    q.get("low",   [None]*len(ts)),
            "close":  q.get("close", [None]*len(ts)),
            "volume": q.get("volume",[0]*len(ts)),
        }).dropna(subset=["open","high","low","close"]).reset_index(drop=True)
        df["volume"] = df["volume"].fillna(0).astype(float)
        return df
    except Exception:
        return pd.DataFrame()


def _add_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stima buy/sell volume da OHLCV con Candle Body Ratio.
    Metodo standard usato da TradingView per replay storico quando
    i dati tick non sono disponibili.
    """
    df = df.copy()
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    ratio = ((df["close"] - df["low"]) / hl).fillna(0.5).clip(0, 1)
    df["buy_vol"]  = (df["volume"] * ratio).round(0)
    df["sell_vol"] = (df["volume"] - df["buy_vol"]).round(0)
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    return df


def _resample(df_sub: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggrega le sub-bar nel timeframe principale."""
    df = _add_delta(df_sub)
    df["bar"] = df["date"].dt.floor(freq)
    agg = df.groupby("bar", sort=True).agg(
        open    =("open",    "first"),
        high    =("high",    "max"),
        low     =("low",     "min"),
        close   =("close",   "last"),
        volume  =("volume",  "sum"),
        buy_vol =("buy_vol", "sum"),
        sell_vol=("sell_vol","sum"),
        delta   =("delta",   "sum"),
    ).reset_index().rename(columns={"bar": "date"})
    agg["cum_delta"] = agg["delta"].cumsum()
    agg["delta_pct"] = (agg["delta"] / agg["volume"].replace(0, np.nan) * 100).round(1).fillna(0)
    return agg


def _vwap(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP con bande ±1σ e ±2σ, reset giornaliero."""
    df = df.copy()
    df["tp"]  = (df["high"] + df["low"] + df["close"]) / 3
    df["day"] = df["date"].dt.date
    g = df.groupby("day", group_keys=False)
    df["cum_tpv"] = g.apply(lambda x: (x["tp"] * x["volume"]).cumsum())
    df["cum_vol"] = g["volume"].cumsum()
    df["vwap"]    = df["cum_tpv"] / df["cum_vol"].replace(0, np.nan)
    df["var"]     = g.apply(
        lambda x: ((x["tp"] - df.loc[x.index, "vwap"]) ** 2 * x["volume"]).cumsum()
                  / x["volume"].cumsum()
    )
    df["std"]     = np.sqrt(df["var"].clip(lower=0))
    df["vwap_1u"] = df["vwap"] + df["std"]
    df["vwap_1d"] = df["vwap"] - df["std"]
    df["vwap_2u"] = df["vwap"] + 2 * df["std"]
    df["vwap_2d"] = df["vwap"] - 2 * df["std"]
    return df


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI 14, EMA 20/50, MACD per pannello indicatori."""
    df = df.copy()
    # RSI
    d = df["close"].diff()
    gain = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # EMA
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    # MACD
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = e12 - e26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"]   = df["macd"] - df["signal"]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# GRAFICO PRINCIPALE — Candele + VWAP + Volume Profile + Delta
# ═════════════════════════════════════════════════════════════════════════════

def _build_main_chart(df: pd.DataFrame, df_vwap: pd.DataFrame,
                      symbol: str, show_vwap: bool,
                      show_ema: bool, show_vp: bool) -> go.Figure:
    """
    Layout TradingView classico:
      Row 1 (65%): Candele + VWAP + EMA + Volume Profile laterale
      Row 2 (20%): Delta bar (buy verde / sell rosso) + CVD line
      Row 3 (15%): Volume bar
    """
    n = len(df)
    if n == 0:
        return go.Figure()

    x    = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    op   = df["open"].tolist()
    hi   = df["high"].tolist()
    lo   = df["low"].tolist()
    cl   = df["close"].tolist()
    vol  = df["volume"].tolist()
    bvol = df["buy_vol"].tolist()
    svol = df["sell_vol"].tolist()
    dlt  = df["delta"].tolist()
    cvd  = df["cum_delta"].tolist()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.64, 0.22, 0.14],
        vertical_spacing=0.01,
    )

    # ── Row 1: Candele ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x, open=op, high=hi, low=lo, close=cl,
        name=symbol,
        increasing=dict(fillcolor=GREEN, line=dict(color=GREEN, width=1)),
        decreasing=dict(fillcolor=RED,   line=dict(color=RED,   width=1)),
        showlegend=False,
    ), row=1, col=1)

    # ── EMA 20 / 50 ───────────────────────────────────────────────────────
    if show_ema and "ema20" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["ema20"].tolist(), mode="lines",
            line=dict(color=ORANGE, width=1.5),
            name="EMA 20", showlegend=True, opacity=0.9,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["ema50"].tolist(), mode="lines",
            line=dict(color=BLUE, width=1.5, dash="dot"),
            name="EMA 50", showlegend=True, opacity=0.9,
        ), row=1, col=1)

    # ── VWAP ──────────────────────────────────────────────────────────────
    if show_vwap and not df_vwap.empty:
        xv = df_vwap["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        fig.add_trace(go.Scatter(
            x=xv, y=df_vwap["vwap"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=2),
            name="VWAP", showlegend=True, opacity=0.95,
        ), row=1, col=1)
        # Banda ±1σ (fill tra le due)
        fig.add_trace(go.Scatter(
            x=xv + xv[::-1],
            y=df_vwap["vwap_1u"].tolist() + df_vwap["vwap_1d"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(255,109,0,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="VWAP ±1σ", showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)
        # Linee ±2σ
        fig.add_trace(go.Scatter(
            x=xv, y=df_vwap["vwap_2u"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=1, dash="dash"),
            name="VWAP +2σ", showlegend=False, opacity=0.5,
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=xv, y=df_vwap["vwap_2d"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=1, dash="dash"),
            name="VWAP -2σ", showlegend=False, opacity=0.5,
            hoverinfo="skip",
        ), row=1, col=1)

    # ── Volume Profile (barre orizzontali a dx) ───────────────────────────
    if show_vp:
        # Calcola distribuzione volume per livello di prezzo
        price_min = min(lo); price_max = max(hi)
        n_bins = 20
        bins   = np.linspace(price_min, price_max, n_bins + 1)
        vp_buy  = np.zeros(n_bins)
        vp_sell = np.zeros(n_bins)

        for i in range(len(x)):
            for j in range(n_bins):
                p_lo, p_hi = bins[j], bins[j+1]
                # Overlap candle con bucket
                overlap = min(hi[i], p_hi) - max(lo[i], p_lo)
                if overlap <= 0: continue
                span = max(hi[i] - lo[i], 1e-10)
                frac = overlap / span
                vp_buy[j]  += bvol[i] * frac
                vp_sell[j] += svol[i] * frac

        vp_total = vp_buy + vp_sell
        poc_idx  = int(np.argmax(vp_total))
        poc_price= float((bins[poc_idx] + bins[poc_idx+1]) / 2)
        prices_c = [(bins[j]+bins[j+1])/2 for j in range(n_bins)]

        # Normalizza per larghezza display (max 8% del range x)
        max_v = max(vp_total) if max(vp_total) > 0 else 1
        # Proietta sulle x come offset dalle ultime barre
        scale = (price_max - price_min) * 0.12 / max_v

        # Barre buy (verde)
        for j in range(n_bins):
            p_c = prices_c[j]
            bar_h = (bins[j+1] - bins[j]) * 0.85
            col = GOLD if j == poc_idx else (G_DARK if vp_buy[j] > vp_sell[j] else R_DARK)
            lw  = 2 if j == poc_idx else 0
            fig.add_shape(type="rect",
                x0=x[-1], x1=x[-1],   # verrà spostato in paper coords
                y0=p_c - bar_h/2, y1=p_c + bar_h/2,
                xref="x", yref="y",
                fillcolor=col,
                line=dict(color=GOLD if j==poc_idx else "rgba(0,0,0,0)", width=lw),
                row=1, col=1)

        # POC line
        fig.add_hline(
            y=poc_price, row=1, col=1,
            line=dict(color=GOLD, width=1.5, dash="dot"),
            annotation_text=f" POC {poc_price:.2f}",
            annotation_font_color=GOLD,
            annotation_font_size=10,
            annotation_position="right",
        )

        # VAH / VAL (Value Area 70%)
        total = float(np.sum(vp_total))
        target = total * 0.70
        acc = vp_total[poc_idx]; lo_i = hi_i = poc_idx
        while acc < target and (lo_i > 0 or hi_i < n_bins-1):
            al = vp_total[lo_i-1] if lo_i>0 else 0
            ah = vp_total[hi_i+1] if hi_i<n_bins-1 else 0
            if ah >= al and hi_i < n_bins-1: hi_i+=1; acc+=vp_total[hi_i]
            elif lo_i > 0: lo_i-=1; acc+=vp_total[lo_i]
            else: break
        vah = float((bins[hi_i]+bins[hi_i+1])/2)
        val = float((bins[lo_i]+bins[lo_i+1])/2)
        for y_val, lbl in [(vah,"VAH"),(val,"VAL")]:
            fig.add_hline(y=y_val, row=1, col=1,
                line=dict(color=BLUE, width=1, dash="dot"),
                annotation_text=f" {lbl} {y_val:.2f}",
                annotation_font_color=BLUE,
                annotation_font_size=9,
                annotation_position="right")

    # ── Row 2: Delta bar + CVD ────────────────────────────────────────────
    # Barre delta colorate (verde = buy dominant, rosso = sell dominant)
    d_colors = [GREEN if d >= 0 else RED for d in dlt]
    fig.add_trace(go.Bar(
        x=x, y=dlt,
        marker_color=d_colors,
        marker_line_width=0,
        name="Delta",
        showlegend=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Delta: %{y:+,.0f}<extra></extra>"
        ),
    ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1,
                  line=dict(color=BORDER, width=1))

    # CVD sovrapposta (asse secondario simulato con normalizzazione)
    d_min, d_max = min(cvd), max(cvd)
    delta_range  = [min(dlt)*1.5, max(dlt)*1.5] if max(dlt) != min(dlt) else [-1,1]
    if d_max != d_min:
        cvd_scaled = [
            delta_range[0] + (v - d_min) / (d_max - d_min)
            * (delta_range[1] - delta_range[0])
            for v in cvd
        ]
        cvd_color = [GREEN if cvd[i] >= cvd[i-1] else RED
                     for i in range(len(cvd))]
        # Usa scatter con colore per segmenti CVD
        fig.add_trace(go.Scatter(
            x=x, y=cvd_scaled, mode="lines",
            line=dict(color=CYAN, width=1.5),
            name="CVD",
            showlegend=True,
            hovertemplate="CVD: %{customdata:+,.0f}<extra></extra>",
            customdata=cvd,
            opacity=0.85,
        ), row=2, col=1)

    # ── Row 3: Volume (buy verde / sell rosso stacked) ────────────────────
    fig.add_trace(go.Bar(
        x=x, y=bvol,
        marker_color=GREEN,
        marker_line_width=0,
        name="Buy Vol",
        showlegend=False,
        opacity=0.85,
        hovertemplate="Buy: %{y:,.0f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=x, y=svol,
        marker_color=RED,
        marker_line_width=0,
        name="Sell Vol",
        showlegend=False,
        opacity=0.85,
        hovertemplate="Sell: %{y:,.0f}<extra></extra>",
    ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    last_close = cl[-1]
    first_open = op[0]
    chg   = (last_close / first_open - 1) * 100 if first_open != 0 else 0
    arrow = "▲" if chg >= 0 else "▼"
    chg_c = GREEN if chg >= 0 else RED

    name_str = _name(symbol)
    title_str = (
        f"<b style='color:{CYAN}'>{symbol}</b>"
        + (f"  <span style='color:{GRAY}'>{name_str}</span>" if name_str != symbol else "")
        + f"  <span style='color:{chg_c}'>{arrow} {abs(chg):.2f}%</span>"
        + f"  <span style='color:{GRAY};font-size:0.8em'>"
        + f"  VWAP · Volume Profile · Delta · CVD</span>"
    )

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=14, color=TEXT), x=0.01),
        height=680,
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=60, t=50, b=8),
        font=dict(color=TEXT, size=10),
        hovermode="x unified",
        barmode="stack",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=TEXT),
            orientation="h",
            x=0.01, y=1.02,
        ),
    )
    # Stile assi
    axis_style = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                      linecolor=BORDER, tickfont=dict(size=9, color=GRAY))
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=True, tickangle=-30, row=3, col=1)
    # Label pannelli
    fig.update_yaxes(title_text="Prezzo", title_font=dict(size=9, color=GRAY),
                     row=1, col=1)
    fig.update_yaxes(title_text="Delta", title_font=dict(size=9, color=GRAY),
                     row=2, col=1)
    fig.update_yaxes(title_text="Volume", title_font=dict(size=9, color=GRAY),
                     row=3, col=1)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# GRAFICO CVD DETTAGLIATO
# ═════════════════════════════════════════════════════════════════════════════

def _build_cvd_chart(df: pd.DataFrame, df_vwap: pd.DataFrame,
                     symbol: str, show_vwap: bool) -> go.Figure:
    """
    Prezzo + CVD sovrapposto (normalizzato) + barre delta.
    Evidenzia le divergenze prezzo/CVD in arancione.
    """
    if len(df) == 0:
        return go.Figure()

    x   = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    cl  = df["close"].tolist()
    dlt = df["delta"].tolist()
    cvd = df["cum_delta"].tolist()

    # Normalizza CVD sulla scala del prezzo
    p_min, p_max = min(cl), max(cl)
    d_min, d_max = min(cvd), max(cvd)
    if d_max != d_min and p_max != p_min:
        cvd_n = [p_min + (v-d_min)/(d_max-d_min)*(p_max-p_min) for v in cvd]
    else:
        cvd_n = cl[:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.01)

    # Prezzo
    fig.add_trace(go.Scatter(
        x=x, y=cl, mode="lines",
        line=dict(color=CYAN, width=2),
        name=f"{symbol} · Close",
        showlegend=True,
    ), row=1, col=1)

    # VWAP
    if show_vwap and not df_vwap.empty:
        xv = df_vwap["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        fig.add_trace(go.Scatter(
            x=xv, y=df_vwap["vwap"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=1.5, dash="dot"),
            name="VWAP", showlegend=True, opacity=0.9,
        ), row=1, col=1)

    # CVD normalizzato
    fig.add_trace(go.Scatter(
        x=x, y=cvd_n, mode="lines",
        line=dict(color=ORANGE, width=2),
        name="CVD (scala prezzo)",
        customdata=cvd,
        hovertemplate="CVD: %{customdata:+,.0f}<extra></extra>",
    ), row=1, col=1)

    # Evidenzia divergenze
    n_div = 0
    for i in range(1, len(x)):
        p_up  = cl[i]   > cl[i-1]
        cd_up = cvd[i]  > cvd[i-1]
        if p_up != cd_up:
            n_div += 1
            fig.add_vrect(
                x0=x[i-1], x1=x[i],
                fillcolor="rgba(255,152,0,0.12)",
                line_width=0,
                annotation_text="div" if n_div == 1 else "",
                annotation_font_color=ORANGE,
                annotation_font_size=8,
                row=1, col=1,
            )

    # Delta bar
    fig.add_trace(go.Bar(
        x=x, y=dlt,
        marker_color=[GREEN if d >= 0 else RED for d in dlt],
        marker_line_width=0,
        name="Delta/bar",
        showlegend=False,
        hovertemplate="Δ %{y:+,.0f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1,
                  line=dict(color=BORDER, width=1))

    name_str = _name(symbol)
    fig.update_layout(
        title=dict(
            text=(f"<b style='color:{CYAN}'>{symbol}</b>"
                  + (f"  <span style='color:{GRAY}'>{name_str}</span>"
                     if name_str != symbol else "")
                  + f"  <span style='color:{GRAY}'>Cumulative Volume Delta</span>"
                  + f"  <span style='color:{ORANGE};font-size:0.85em'>"
                  + f"  {n_div} divergenze</span>"),
            font=dict(size=14, color=TEXT), x=0.01,
        ),
        height=500,
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        margin=dict(l=60, r=60, t=50, b=8),
        font=dict(color=TEXT, size=10),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT),
                    orientation="h", x=0.01, y=1.02),
    )
    axis_s = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                  linecolor=BORDER, tickfont=dict(size=9, color=GRAY))
    fig.update_xaxes(**axis_s)
    fig.update_yaxes(**axis_s)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=True, tickangle=-30, row=2, col=1)
    fig.update_yaxes(title_text="Prezzo", title_font=dict(size=9, color=GRAY), row=1, col=1)
    fig.update_yaxes(title_text="Delta",  title_font=dict(size=9, color=GRAY), row=2, col=1)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# GRAFICO INDICATORI (RSI + MACD)
# ═════════════════════════════════════════════════════════════════════════════

def _build_indicators_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Candele + EMA + RSI + MACD — layout classico TradingView.
    """
    if len(df) == 0:
        return go.Figure()

    x  = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    op = df["open"].tolist(); hi = df["high"].tolist()
    lo = df["low"].tolist();  cl = df["close"].tolist()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.22, 0.23],
                        vertical_spacing=0.01)

    # Candele
    fig.add_trace(go.Candlestick(
        x=x, open=op, high=hi, low=lo, close=cl,
        name=symbol, showlegend=False,
        increasing=dict(fillcolor=GREEN, line=dict(color=GREEN, width=1)),
        decreasing=dict(fillcolor=RED,   line=dict(color=RED,   width=1)),
    ), row=1, col=1)

    # EMA
    if "ema20" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["ema20"].tolist(), mode="lines",
            line=dict(color=ORANGE, width=1.5), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["ema50"].tolist(), mode="lines",
            line=dict(color=BLUE, width=1.5, dash="dot"), name="EMA 50"),
            row=1, col=1)

    # RSI
    if "rsi" in df.columns:
        rsi = df["rsi"].tolist()
        fig.add_trace(go.Scatter(x=x, y=rsi, mode="lines",
            line=dict(color=CYAN, width=1.5), name="RSI 14",
            hovertemplate="RSI: %{y:.1f}<extra></extra>"),
            row=2, col=1)
        fig.add_hline(y=70, row=2, col=1,
                      line=dict(color=RED, width=1, dash="dot"))
        fig.add_hline(y=30, row=2, col=1,
                      line=dict(color=GREEN, width=1, dash="dot"))
        fig.add_hline(y=50, row=2, col=1,
                      line=dict(color=BORDER, width=1))
        fig.add_hrect(y0=30, y1=70, row=2, col=1,
                      fillcolor="rgba(80,196,224,0.04)", line_width=0)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # MACD
    if "macd" in df.columns:
        hist = df["hist"].tolist()
        fig.add_trace(go.Bar(x=x, y=hist,
            marker_color=[GREEN if h >= 0 else RED for h in hist],
            marker_line_width=0, name="MACD hist",
            hovertemplate="Hist: %{y:.4f}<extra></extra>"),
            row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["macd"].tolist(), mode="lines",
            line=dict(color=CYAN, width=1.5), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["signal"].tolist(), mode="lines",
            line=dict(color=ORANGE, width=1.5), name="Signal"), row=3, col=1)
        fig.add_hline(y=0, row=3, col=1, line=dict(color=BORDER, width=1))

    name_str = _name(symbol)
    fig.update_layout(
        title=dict(
            text=(f"<b style='color:{CYAN}'>{symbol}</b>"
                  + (f"  <span style='color:{GRAY}'>{name_str}</span>"
                     if name_str != symbol else "")
                  + f"  <span style='color:{GRAY}'>RSI 14 · EMA 20/50 · MACD</span>"),
            font=dict(size=14, color=TEXT), x=0.01,
        ),
        height=600,
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=60, t=50, b=8),
        font=dict(color=TEXT, size=10),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT),
                    orientation="h", x=0.01, y=1.02),
    )
    axis_s = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                  linecolor=BORDER, tickfont=dict(size=9, color=GRAY))
    fig.update_xaxes(**axis_s)
    fig.update_yaxes(**axis_s)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=True, tickangle=-30, row=3, col=1)
    fig.update_yaxes(title_text="Prezzo", title_font=dict(size=9, color=GRAY), row=1, col=1)
    fig.update_yaxes(title_text="RSI",    title_font=dict(size=9, color=GRAY), row=2, col=1)
    fig.update_yaxes(title_text="MACD",   title_font=dict(size=9, color=GRAY), row=3, col=1)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS UI
# ═════════════════════════════════════════════════════════════════════════════

def _fv(v: float) -> str:
    v = abs(v)
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    if v >= 1e6:  return f"{v/1e6:.1f}M"
    if v >= 1e3:  return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def _kpi_html(label: str, value: str, color: str = TEXT, sub: str = "") -> str:
    return (
        f'<div style="background:{PANEL};border:1px solid {BORDER};'
        f'border-left:4px solid {color};border-radius:6px;'
        f'padding:10px 12px;text-align:center">'
        f'<div style="color:{GRAY};font-size:0.65rem;font-weight:600;'
        f'letter-spacing:.06em;text-transform:uppercase">{label}</div>'
        f'<div style="color:{color};font-size:1.15rem;font-weight:700;'
        f'margin:3px 0">{value}</div>'
        + (f'<div style="color:{GRAY};font-size:0.72rem">{sub}</div>' if sub else "")
        + '</div>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def render_orderflow_tab(df_scanner=None):
    """
    Renderizza il tab Order Flow.
    df_scanner: DataFrame dallo scanner (opzionale) con colonna Ticker.
    """

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{PANEL};border-left:3px solid {ORANGE};'
        f'padding:10px 18px;border-radius:0 6px 6px 0;margin-bottom:16px">'
        f'<span style="color:{ORANGE};font-weight:700;font-size:1.05rem">'
        f'🔬 ORDER FLOW</span>'
        f'<span style="color:{GRAY};font-size:0.8rem;margin-left:12px">'
        f'Candele · Volume Profile · VWAP ±σ · Delta · CVD · Divergenze · v31.1'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # ── Nota metodologica (collassata) ───────────────────────────────────
    with st.expander("ℹ️ Come vengono calcolati i dati", expanded=False):
        st.markdown("""
**Fonte dati:** Yahoo Finance — OHLCV intraday (gratuito, senza dati tick).

**Delta Buy/Sell:** stimato con il metodo *Candle Body Ratio*
`buy_vol ≈ volume × (close − low) / (high − low)`
Questo è lo stesso metodo usato da TradingView e Bookmap per il replay storico
quando i dati tick Level-2 non sono disponibili.
Accuratezza ~70-80% su strumenti molto liquidi (SPY, NQ=F, BTC-USD).

**VWAP:** Volume Weighted Average Price con bande ±1σ e ±2σ, reset ogni giorno.

**Volume Profile:** distribuzione del volume per fascia di prezzo sull'intero periodo.
POC = Point of Control (fascia con massimo volume). Value Area = 70% del volume.
""")

    # ── Costruisci lista ticker (scanner + default) ───────────────────────
    sc_tickers: list = []
    if df_scanner is not None and not df_scanner.empty:
        tc = "Ticker" if "Ticker" in df_scanner.columns else "ticker"
        if tc in df_scanner.columns:
            sc_tickers = df_scanner[tc].dropna().unique().tolist()[:30]

    # Merge: scanner ticker prima, poi default; niente duplicati
    merged = list(dict.fromkeys(sc_tickers + _DEFAULT_TICKERS))
    # Costruisci opzioni "Nome  (TICKER)" ordinate per nome
    opts_display = sorted(
        [_label(t) for t in merged],
        key=lambda s: s.lower()
    )
    # Mappa display → ticker
    disp_to_tk = {_label(t): t for t in merged}

    # ── Controlli ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])

    with col1:
        sel_display = st.selectbox(
            "Strumento",
            opts_display,
            key="of_ticker_sel",
            help="Scegli il ticker da analizzare",
        )
        symbol = disp_to_tk.get(sel_display, sel_display.split("(")[-1].rstrip(")").strip())
        # Override manuale
        manual = st.text_input(
            "Oppure inserisci ticker Yahoo Finance",
            placeholder="es. EURUSD=X · BTC-USD · ES=F",
            key="of_manual",
        ).strip().upper()
        if manual:
            symbol = manual

    with col2:
        tf_label = st.selectbox("Timeframe", list(TF_MAP.keys()),
                                index=2, key="of_tf")
        sub_iv, main_freq, range_ = TF_MAP[tf_label]

    with col3:
        vista = st.radio(
            "Vista",
            ["📊 Principale", "📈 CVD", "📉 Indicatori"],
            key="of_vista",
        )

    with col4:
        st.write(""); st.write("")
        show_vwap = st.checkbox("VWAP ±σ",   value=True,  key="of_vwap")
        show_ema  = st.checkbox("EMA 20/50", value=True,  key="of_ema")
        show_vp   = st.checkbox("Vol Profile",value=True,  key="of_vp")
        st.write("")
        run = st.button("▶ Carica", key="of_run",
                        use_container_width=True, type="primary")
        if st.button("🔄", key="of_ref", help="Svuota cache"):
            st.cache_data.clear(); st.rerun()

    if not run:
        # Placeholder
        st.markdown(
            f'<div style="background:{PANEL};border:1px dashed {BORDER};'
            f'border-radius:8px;padding:60px;text-align:center;margin-top:10px">'
            f'<div style="font-size:2.5rem">📊</div>'
            f'<div style="color:{TEXT};font-size:1.05rem;font-weight:600;margin-top:10px">'
            f'Seleziona lo strumento e clicca '
            f'<b style="color:{ORANGE}">▶ Carica</b></div>'
            f'<div style="color:{GRAY};font-size:0.85rem;margin-top:6px">'
            f'Candele · VWAP ±1σ/±2σ · Volume Profile POC/VAH/VAL · '
            f'Delta · CVD · RSI · MACD</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Caricamento dati ──────────────────────────────────────────────────
    name_display = _name(symbol)
    spinner_lbl  = (f"{symbol} — {name_display}" if name_display != symbol
                    else symbol)

    with st.spinner(f"⏳ Caricamento {spinner_lbl} [{tf_label}]…"):
        df_sub  = _fetch(symbol, sub_iv,  range_)
        df_main = _fetch(symbol, main_freq.replace("min","m").replace("D","d"),
                         range_)
        # Fallback: usa df_sub direttamente se main non torna dati
        if df_sub.empty:
            st.error(
                f"❌ Dati non disponibili per **{symbol}**.\n\n"
                "Verifica che il simbolo sia valido su Yahoo Finance:\n"
                "es. `AAPL`, `BTC-USD`, `ES=F`, `EUR=X`, `SPY`"
            )
            return

        # Ricampiona al timeframe principale
        df = _resample(df_sub, main_freq)
        if df.empty:
            st.error("❌ Impossibile aggregare i dati. Prova un timeframe diverso.")
            return

        df = _indicators(df)
        df_vwap_data = _vwap(df_sub) if show_vwap else pd.DataFrame()

    # ── KPI Bar ───────────────────────────────────────────────────────────
    last  = df.iloc[-1]
    first = df.iloc[0]
    chg   = (last["close"] / first["open"] - 1) * 100 if first["open"] != 0 else 0
    tot_buy  = float(df["buy_vol"].sum())
    tot_sell = float(df["sell_vol"].sum())
    tot_delta= float(df["delta"].sum())
    buy_pct  = tot_buy / (tot_buy + tot_sell) * 100 if (tot_buy + tot_sell) > 0 else 50
    dom      = "BUY" if tot_delta >= 0 else "SELL"
    dom_c    = GREEN if tot_delta >= 0 else RED
    chg_c    = GREEN if chg >= 0 else RED
    vwap_val = (float(df_vwap_data["vwap"].iloc[-1])
                if not df_vwap_data.empty and "vwap" in df_vwap_data.columns
                else 0)
    vs_vwap  = ("▲ sopra" if last["close"] > vwap_val and vwap_val > 0
                else "▼ sotto" if vwap_val > 0 else "–")

    k_cols = st.columns(6)
    kpis = [
        ("Ticker",     f"{symbol}",         CYAN,   name_display if name_display != symbol else ""),
        ("Close",      f"${last['close']:.2f}",
         GREEN if last["close"] >= last["open"] else RED,
         f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%"),
        ("VWAP",       f"${vwap_val:.2f}" if vwap_val else "–", VWAP_C, vs_vwap),
        ("Delta tot",  f"{'+' if tot_delta>=0 else ''}{_fv(tot_delta)}", dom_c, dom),
        ("Buy %",      f"{buy_pct:.0f}%",   GREEN,  _fv(tot_buy)),
        ("Sell %",     f"{100-buy_pct:.0f}%", RED,  _fv(tot_sell)),
    ]
    for (lbl, val, col, sub), kcol in zip(kpis, k_cols):
        with kcol:
            st.markdown(_kpi_html(lbl, val, col, sub), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Grafico ───────────────────────────────────────────────────────────
    if vista == "📊 Principale":
        fig = _build_main_chart(df, df_vwap_data, symbol,
                                show_vwap, show_ema, show_vp)
        st.plotly_chart(fig, use_container_width=True, key="of_main_chart")
        # Legenda pannelli
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-radius:6px;padding:7px 14px;font-size:0.76rem;'
            f'display:flex;gap:18px;flex-wrap:wrap">'
            f'<b style="color:{GRAY}">Pannelli:</b>'
            f'<span style="color:{TEXT}">① Prezzo</span>'
            f'<span style="color:{VWAP_C}">━ VWAP</span>'
            f'<span style="color:{ORANGE}">━ EMA 20</span>'
            f'<span style="color:{BLUE}">┄ EMA 50</span>'
            f'<span style="color:{GOLD}">◆ POC</span>'
            f'<span style="color:{BLUE}">── VAH/VAL</span>'
            f' &nbsp;|&nbsp; '
            f'<span style="color:{TEXT}">② Delta</span>'
            f'<span style="color:{GREEN}">▌ Buy aggressor</span>'
            f'<span style="color:{RED}">▌ Sell aggressor</span>'
            f'<span style="color:{CYAN}">━ CVD</span>'
            f' &nbsp;|&nbsp; '
            f'<span style="color:{TEXT}">③ Volume</span>'
            f'<span style="color:{GREEN}">▌ Buy</span>'
            f'<span style="color:{RED}">▌ Sell</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    elif vista == "📈 CVD":
        fig = _build_cvd_chart(df, df_vwap_data, symbol, show_vwap)
        st.plotly_chart(fig, use_container_width=True, key="of_cvd_chart")
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-radius:6px;padding:7px 14px;font-size:0.76rem;'
            f'display:flex;gap:18px;flex-wrap:wrap">'
            f'<span style="color:{CYAN}">━ Prezzo Close</span>'
            f'<span style="color:{VWAP_C}">┄ VWAP</span>'
            f'<span style="color:{ORANGE}">━ CVD normalizzato sulla scala prezzo</span>'
            f'<span style="color:{ORANGE}">░ Zona di divergenza prezzo / CVD</span>'
            f'<span style="color:{GRAY}">— Le divergenze indicano potenziale inversione</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:  # Indicatori
        fig = _build_indicators_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True, key="of_ind_chart")
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-radius:6px;padding:7px 14px;font-size:0.76rem;'
            f'display:flex;gap:18px;flex-wrap:wrap">'
            f'<span style="color:{ORANGE}">━ EMA 20</span>'
            f'<span style="color:{BLUE}">┄ EMA 50</span>'
            f'<span style="color:{CYAN}">━ RSI 14</span>'
            f'<span style="color:{RED}">┄ RSI 70</span>'
            f'<span style="color:{GREEN}">┄ RSI 30</span>'
            f'<span style="color:{CYAN}">━ MACD line</span>'
            f'<span style="color:{ORANGE}">━ Signal</span>'
            f'<span style="color:{GREEN}">▌ Hist +</span>'
            f'<span style="color:{RED}">▌ Hist –</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Tabella riassuntiva (collassata) ──────────────────────────────────
    with st.expander("📋 Dati candle", expanded=False):
        show_cols = ["date", "open", "high", "low", "close",
                     "volume", "buy_vol", "sell_vol", "delta", "delta_pct", "cum_delta"]
        df_show = df[show_cols].tail(30).copy()
        df_show["date"]      = df_show["date"].dt.strftime("%Y-%m-%d %H:%M")
        df_show["volume"]    = df_show["volume"].apply(lambda v: _fv(v))
        df_show["buy_vol"]   = df_show["buy_vol"].apply(lambda v: _fv(v))
        df_show["sell_vol"]  = df_show["sell_vol"].apply(lambda v: _fv(v))
        df_show["delta"]     = df_show["delta"].apply(lambda v: f"{'+' if v>=0 else ''}{_fv(v)}")
        df_show["delta_pct"] = df_show["delta_pct"].apply(lambda v: f"{v:+.1f}%")
        df_show["cum_delta"] = df_show["cum_delta"].apply(lambda v: _fv(v))
        df_show.columns = ["Data","Open","High","Low","Close",
                           "Volume","Buy Vol","Sell Vol","Delta","Δ%","CVD"]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{GRAY};font-size:0.70rem;text-align:center;'
        f'margin-top:14px;padding-top:8px;border-top:1px solid {BORDER}">'
        f'Dati: Yahoo Finance OHLCV · '
        f'Delta stimato: Candle Body Ratio (~70-80% accuratezza) · '
        f'Cache 5 min · {datetime.now().strftime("%d/%m/%Y %H:%M")}'
        f'</div>',
        unsafe_allow_html=True,
    )
