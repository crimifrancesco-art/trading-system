# -*- coding: utf-8 -*-
"""
orderflow_tab.py  —  🔬 Order Flow  v31.1
Stile TradingView Dark · legende embedded per ogni vista
"""

import json, base64, urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Palette TradingView Dark ────────────────────────────────────────────────
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
PURPLE = "#9c27b0"
VWAP_C = "#ff6d00"
G_DARK = "rgba(38,166,154,0.15)"
R_DARK = "rgba(239,83,80,0.15)"

# ─── Slide legende per vista ─────────────────────────────────────────────────
# Cartella base dove cercare le immagini (outputs di Streamlit Cloud = assets/)
_ASSETS = Path("/mnt/user-data/outputs")   # locale
_ASSETS_ALT = Path("assets")               # Streamlit Cloud

def _img(filename: str) -> str:
    """Restituisce base64 dell'immagine o stringa vuota se non trovata."""
    for base in [_ASSETS, _ASSETS_ALT]:
        p = base / filename
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return ""

# Mappa vista → lista slide (filename, titolo)
SLIDES = {
    "principale": [
        ("leg_rsi_vwap.png",      "RSI & VWAP — Long/Short"),
        ("leg_rsi_vwap2.png",     "RSI & VWAP — Intraday"),
    ],
    "cvd": [
        ("leg_bb_rsi.png",        "Bollinger Bands & RSI"),
    ],
    "indicatori": [
        ("leg_rsi_vwap.png",      "RSI & VWAP"),
        ("leg_sma_rsi.png",       "9 & 21 SMA & RSI"),
        ("leg_sma_rsi2.png",      "9 & 21 SMA & RSI (Short)"),
        ("leg_bb_rsi.png",        "Bollinger Bands & RSI"),
        ("leg_adx_ema.png",       "ADX & EMA"),
        ("leg_keltner_macd.png",  "Keltner Channel & MACD"),
        ("leg_sar.png",           "Parabolic SAR & Chop Zone"),
        ("leg_alligator.png",     "Alligator & Vortex"),
    ],
}

# ─── Ticker → Nome ───────────────────────────────────────────────────────────
NAMES = {
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMZN":"Amazon",
    "GOOGL":"Alphabet","META":"Meta","TSLA":"Tesla","AVGO":"Broadcom",
    "BRK-B":"Berkshire","LLY":"Eli Lilly","JPM":"JPMorgan","V":"Visa",
    "MA":"Mastercard","UNH":"UnitedHealth","XOM":"ExxonMobil",
    "JNJ":"J&J","WMT":"Walmart","PG":"P&G","ORCL":"Oracle",
    "HD":"Home Depot","COST":"Costco","BAC":"BofA","NFLX":"Netflix",
    "KO":"Coca-Cola","CRM":"Salesforce","AMD":"AMD","MRK":"Merck",
    "CVX":"Chevron","PEP":"PepsiCo","ABBV":"AbbVie","TMO":"Thermo Fisher",
    "LIN":"Linde","ACN":"Accenture","MCD":"McDonald's","PM":"Philip Morris",
    "GE":"GE Aerospace","NOW":"ServiceNow","CAT":"Caterpillar","IBM":"IBM",
    "GS":"Goldman Sachs","AMGN":"Amgen","T":"AT&T","MS":"Morgan Stanley",
    "AXP":"Amex","SPGI":"S&P Global","BLK":"BlackRock","RTX":"RTX",
    "HON":"Honeywell","PFE":"Pfizer","ADBE":"Adobe","INTU":"Intuit",
    "QCOM":"Qualcomm","TXN":"Texas Instr.","PANW":"Palo Alto",
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF","IWM":"Russell 2000 ETF",
    "DIA":"Dow Jones ETF","GLD":"Gold ETF","SLV":"Silver ETF",
    "TLT":"20yr Bond ETF","HYG":"High Yield ETF","GDX":"Gold Miners ETF",
    "ES=F":"S&P 500 Fut","NQ=F":"Nasdaq Fut","YM=F":"Dow Fut",
    "RTY=F":"Russell Fut","CL=F":"Crude Oil Fut","GC=F":"Gold Fut",
    "SI=F":"Silver Fut","ZB=F":"Bond Fut",
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana",
    "EUR=X":"EUR/USD","JPY=X":"USD/JPY","GBP=X":"GBP/USD",
    "EURUSD=X":"EUR/USD","GBPUSD=X":"GBP/USD",
    "ASML":"ASML","SAP":"SAP","AZN":"AstraZeneca","GSK":"GSK","BP":"BP",
    "TSM":"TSMC","TM":"Toyota","BABA":"Alibaba","NVO":"Novo Nordisk",
}

_DEFAULT_TKS = [
    "ES=F","NQ=F","SPY","QQQ","BTC-USD","ETH-USD",
    "AAPL","MSFT","NVDA","TSLA","META","AMZN","GOOGL",
    "GLD","GC=F","CL=F","EUR=X",
]

TF_MAP = {
    "15min": ("2m",  "15min", "3d"),
    "30min": ("2m",  "30min", "5d"),
    "1h":    ("5m",  "60min", "10d"),
    "4h":    ("15m", "240min","30d"),
    "Daily": ("60m", "1D",    "180d"),
}

def _name(t: str) -> str: return NAMES.get(t, t)
def _label(t: str) -> str:
    n = NAMES.get(t); return f"{n}  ({t})" if n else t
def _fv(v: float) -> str:
    v = abs(v)
    if v >= 1e9: return f"{v/1e9:.2f}B"
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.0f}K"
    return f"{v:.0f}"

# ─── UI helpers ──────────────────────────────────────────────────────────────
def _kpi(label, value, color=TEXT, sub=""):
    return (
        f'<div style="background:{PANEL};border:1px solid {BORDER};'
        f'border-left:4px solid {color};border-radius:6px;'
        f'padding:10px 12px;text-align:center">'
        f'<div style="color:{GRAY};font-size:.64rem;font-weight:600;'
        f'letter-spacing:.06em;text-transform:uppercase">{label}</div>'
        f'<div style="color:{color};font-size:1.12rem;font-weight:700;margin:3px 0">{value}</div>'
        + (f'<div style="color:{GRAY};font-size:.72rem">{sub}</div>' if sub else "")
        + "</div>"
    )

def _slide_block(vista_key: str):
    """Renderizza le slide legenda appropriate per la vista corrente."""
    slides = SLIDES.get(vista_key, [])
    if not slides:
        return
    available = [(fn, title) for fn, title in slides if _img(fn)]
    if not available:
        return
    with st.expander(
        f"📖 Legenda — come leggere questo grafico  ({len(available)} slide)",
        expanded=False
    ):
        cols = st.columns(len(available))
        for col, (fn, title) in zip(cols, available):
            b64 = _img(fn)
            with col:
                st.markdown(
                    f'<div style="border:1px solid {BORDER};border-radius:6px;overflow:hidden">'
                    f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block">'
                    f'</div>'
                    f'<p style="color:{GRAY};font-size:.68rem;text-align:center;margin:4px 0 0">'
                    f'{title}</p>',
                    unsafe_allow_html=True
                )

def _legend_strip(items: list):
    """Striscia colorata di legenda sotto ogni grafico."""
    parts = "".join(
        f'<span style="color:{c}">{sym} {lbl}</span>'
        for sym, lbl, c in items
    )
    st.markdown(
        f'<div style="background:{PANEL};border:1px solid {BORDER};'
        f'border-radius:6px;padding:6px 14px;font-size:.76rem;'
        f'display:flex;gap:16px;flex-wrap:wrap;margin-top:4px">'
        f'{parts}</div>',
        unsafe_allow_html=True
    )

# ─── Dati ────────────────────────────────────────────────────────────────────
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

def _delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    r  = ((df["close"] - df["low"]) / hl).fillna(0.5).clip(0, 1)
    df["buy_vol"]  = (df["volume"] * r).round(0)
    df["sell_vol"] = (df["volume"] - df["buy_vol"]).round(0)
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    return df

def _resample(df_sub: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = _delta(df_sub)
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
    ).reset_index().rename(columns={"bar":"date"})
    agg["cum_delta"] = agg["delta"].cumsum()
    agg["delta_pct"] = (agg["delta"] / agg["volume"].replace(0, np.nan) * 100).round(1).fillna(0)
    return agg

def _vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tp"]  = (df["high"] + df["low"] + df["close"]) / 3
    df["day"] = df["date"].dt.date
    g = df.groupby("day", group_keys=False)
    df["cum_tpv"] = g.apply(lambda x: (x["tp"] * x["volume"]).cumsum())
    df["cum_vol"] = g["volume"].cumsum()
    df["vwap"]    = df["cum_tpv"] / df["cum_vol"].replace(0, np.nan)
    df["var"]     = g.apply(
        lambda x: ((x["tp"] - df.loc[x.index,"vwap"])**2 * x["volume"]).cumsum()
                  / x["volume"].cumsum()
    )
    df["std"]     = np.sqrt(df["var"].clip(lower=0))
    df["vwap_1u"] = df["vwap"] + df["std"]
    df["vwap_1d"] = df["vwap"] - df["std"]
    df["vwap_2u"] = df["vwap"] + 2*df["std"]
    df["vwap_2d"] = df["vwap"] - 2*df["std"]
    return df

def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # RSI 14
    d    = df["close"].diff()
    gain = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi"]  = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # EMA 20 / 50
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    # SMA 9 / 21
    df["sma9"]  = df["close"].rolling(9).mean()
    df["sma21"] = df["close"].rolling(21).mean()
    # MACD 12-26-9
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = e12 - e26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"]   = df["macd"] - df["signal"]
    # Bollinger 20,2
    rm   = df["close"].rolling(20).mean()
    rstd = df["close"].rolling(20).std()
    df["bb_mid"] = rm
    df["bb_up"]  = rm + 2*rstd
    df["bb_dn"]  = rm - 2*rstd
    # ADX 14
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr    = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    dm_p  = (hi - hi.shift()).clip(lower=0)
    dm_m  = (lo.shift() - lo).clip(lower=0)
    dm_p  = np.where(dm_p > dm_m, dm_p, 0)
    dm_m  = np.where(pd.Series(dm_m) > pd.Series(dm_p), dm_m, 0)
    atr14 = pd.Series(tr).ewm(com=13, adjust=False).mean()
    di_p  = 100 * pd.Series(dm_p).ewm(com=13, adjust=False).mean() / atr14.replace(0,np.nan)
    di_m  = 100 * pd.Series(dm_m).ewm(com=13, adjust=False).mean() / atr14.replace(0,np.nan)
    dx    = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0,np.nan))
    df["adx"]  = dx.ewm(com=13, adjust=False).mean()
    df["di_p"] = di_p.values
    df["di_m"] = di_m.values
    return df

# ─── Chart builders ──────────────────────────────────────────────────────────
_AX = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
           linecolor=BORDER, tickfont=dict(size=9, color=GRAY))

def _layout(fig, title, height):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=TEXT), x=0.01),
        height=height,
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=60, t=48, b=8),
        font=dict(color=TEXT, size=10),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT),
                    orientation="h", x=0.01, y=1.02),
    )
    fig.update_xaxes(**_AX)
    fig.update_yaxes(**_AX)

# ── VISTA PRINCIPALE: Candele + VWAP + Volume Profile + Delta + CVD ───────────
def _chart_main(df, df_vwap, symbol, show_vwap, show_ema, show_vp):
    if df.empty: return go.Figure()
    x   = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    op  = df["open"].tolist(); hi = df["high"].tolist()
    lo  = df["low"].tolist();  cl = df["close"].tolist()
    bv  = df["buy_vol"].tolist(); sv = df["sell_vol"].tolist()
    dlt = df["delta"].tolist(); cvd = df["cum_delta"].tolist()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.63, 0.22, 0.15],
                        vertical_spacing=0.01)

    # ── Candele ──
    fig.add_trace(go.Candlestick(
        x=x, open=op, high=hi, low=lo, close=cl, name=symbol,
        showlegend=False,
        increasing=dict(fillcolor=GREEN, line=dict(color=GREEN, width=1)),
        decreasing=dict(fillcolor=RED,   line=dict(color=RED,   width=1)),
    ), row=1, col=1)

    # ── EMA 20 / 50 ──
    if show_ema and "ema20" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["ema20"].tolist(), mode="lines",
            line=dict(color=ORANGE, width=1.5), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["ema50"].tolist(), mode="lines",
            line=dict(color=BLUE, width=1.5, dash="dot"), name="EMA 50"), row=1, col=1)

    # ── VWAP + bande ──
    if show_vwap and not df_vwap.empty:
        xv = df_vwap["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        fig.add_trace(go.Scatter(x=xv, y=df_vwap["vwap"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=2), name="VWAP"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=xv + xv[::-1],
            y=df_vwap["vwap_1u"].tolist() + df_vwap["vwap_1d"].tolist()[::-1],
            fill="toself", fillcolor="rgba(255,109,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="VWAP ±1σ",
            hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xv, y=df_vwap["vwap_2u"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=1, dash="dash"), showlegend=False,
            opacity=0.45, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xv, y=df_vwap["vwap_2d"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=1, dash="dash"), showlegend=False,
            opacity=0.45, hoverinfo="skip"), row=1, col=1)

    # ── Volume Profile (POC / VAH / VAL) ──
    if show_vp:
        n_bins = 20
        p_min, p_max = min(lo), max(hi)
        bins = np.linspace(p_min, p_max, n_bins+1)
        vp_buy = np.zeros(n_bins); vp_sell = np.zeros(n_bins)
        for i in range(len(x)):
            for j in range(n_bins):
                ov = min(hi[i], bins[j+1]) - max(lo[i], bins[j])
                if ov <= 0: continue
                sp = max(hi[i]-lo[i], 1e-10)
                vp_buy[j]  += bv[i] * ov/sp
                vp_sell[j] += sv[i] * ov/sp
        vp_tot = vp_buy + vp_sell
        poc_i  = int(np.argmax(vp_tot))
        poc_p  = float((bins[poc_i]+bins[poc_i+1])/2)
        # Value Area 70%
        tv = vp_tot.sum(); acc = vp_tot[poc_i]; li = hi_i = poc_i
        while acc < tv*0.70 and (li>0 or hi_i<n_bins-1):
            al = vp_tot[li-1] if li>0 else 0
            ah = vp_tot[hi_i+1] if hi_i<n_bins-1 else 0
            if ah >= al and hi_i < n_bins-1: hi_i+=1; acc+=vp_tot[hi_i]
            elif li > 0: li-=1; acc+=vp_tot[li]
            else: break
        vah = float((bins[hi_i]+bins[hi_i+1])/2)
        val = float((bins[li]+bins[li+1])/2)
        for yv, lbl, c in [(poc_p,"POC",GOLD),(vah,"VAH",BLUE),(val,"VAL",BLUE)]:
            fig.add_hline(y=yv, row=1, col=1,
                line=dict(color=c, width=1.5 if lbl=="POC" else 1, dash="dot"),
                annotation_text=f" {lbl} {yv:.2f}",
                annotation_font_color=c, annotation_font_size=9,
                annotation_position="right")

    # ── Delta bar + CVD ──
    fig.add_trace(go.Bar(x=x, y=dlt,
        marker_color=[GREEN if d>=0 else RED for d in dlt],
        marker_line_width=0, name="Delta", showlegend=False,
        hovertemplate="Δ %{y:+,.0f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line=dict(color=BORDER, width=1))
    # CVD scalato sul range delta
    d_min, d_max = min(cvd), max(cvd)
    dr = [min(dlt)*1.4, max(dlt)*1.4] if max(dlt) != min(dlt) else [-1,1]
    cvd_s = ([dr[0]+(v-d_min)/(d_max-d_min)*(dr[1]-dr[0]) for v in cvd]
             if d_max != d_min else dlt[:])
    fig.add_trace(go.Scatter(x=x, y=cvd_s, mode="lines",
        line=dict(color=CYAN, width=1.5), name="CVD",
        customdata=cvd,
        hovertemplate="CVD: %{customdata:+,.0f}<extra></extra>"), row=2, col=1)

    # ── Volume stacked buy/sell ──
    fig.add_trace(go.Bar(x=x, y=bv, marker_color=GREEN, marker_line_width=0,
        name="Buy Vol", showlegend=False, opacity=0.8,
        hovertemplate="Buy: %{y:,.0f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=sv, marker_color=RED, marker_line_width=0,
        name="Sell Vol", showlegend=False, opacity=0.8,
        hovertemplate="Sell: %{y:,.0f}<extra></extra>"), row=3, col=1)

    chg = (cl[-1]/op[0]-1)*100 if op[0] else 0
    n   = _name(symbol)
    ttl = (f"<b style='color:{CYAN}'>{symbol}</b>"
           + (f"  <span style='color:{GRAY}'>{n}</span>" if n != symbol else "")
           + f"  <span style='color:{GREEN if chg>=0 else RED}'>"
           + f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%</span>"
           + f"  <span style='color:{GRAY};font-size:.82em'>"
           + "  VWAP · Volume Profile · Delta · CVD</span>")
    _layout(fig, ttl, 690)
    fig.update_layout(barmode="stack")
    for r, show_tl in [(1,False),(2,False),(3,True)]:
        n_ = "" if r==1 else str(r)
        fig.update_layout(**{f"xaxis{n_}":dict(**_AX, showticklabels=show_tl,
                                               tickangle=-30 if show_tl else 0)})
    fig.update_yaxes(title_text="Prezzo", title_font=dict(size=9,color=GRAY), row=1, col=1)
    fig.update_yaxes(title_text="Delta",  title_font=dict(size=9,color=GRAY), row=2, col=1)
    fig.update_yaxes(title_text="Volume", title_font=dict(size=9,color=GRAY), row=3, col=1)
    return fig

# ── VISTA CVD: Prezzo + CVD normalizzato + divergenze ─────────────────────────
def _chart_cvd(df, df_vwap, symbol, show_vwap):
    if df.empty: return go.Figure()
    x   = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    cl  = df["close"].tolist()
    dlt = df["delta"].tolist()
    cvd = df["cum_delta"].tolist()

    p_min,p_max = min(cl), max(cl)
    d_min,d_max = min(cvd), max(cvd)
    cvd_n = ([p_min+(v-d_min)/(d_max-d_min)*(p_max-p_min) for v in cvd]
             if d_max!=d_min and p_max!=p_min else cl[:])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.01)

    fig.add_trace(go.Scatter(x=x, y=cl, mode="lines",
        line=dict(color=CYAN, width=2), name="Close"), row=1, col=1)

    if show_vwap and not df_vwap.empty and "vwap" in df_vwap.columns:
        xv = df_vwap["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        fig.add_trace(go.Scatter(x=xv, y=df_vwap["vwap"].tolist(), mode="lines",
            line=dict(color=VWAP_C, width=1.5, dash="dot"),
            name="VWAP", opacity=0.9), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=cvd_n, mode="lines",
        line=dict(color=ORANGE, width=2), name="CVD (norm)",
        customdata=cvd,
        hovertemplate="CVD: %{customdata:+,.0f}<extra></extra>"), row=1, col=1)

    # Divergenze
    n_div = 0
    for i in range(1, len(x)):
        if (cl[i]>cl[i-1]) != (cvd[i]>cvd[i-1]):
            n_div += 1
            fig.add_vrect(x0=x[i-1], x1=x[i],
                fillcolor="rgba(255,152,0,0.13)", line_width=0, row=1, col=1)

    fig.add_trace(go.Bar(x=x, y=dlt,
        marker_color=[GREEN if d>=0 else RED for d in dlt],
        marker_line_width=0, name="Delta/bar", showlegend=False,
        hovertemplate="Δ %{y:+,.0f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line=dict(color=BORDER, width=1))

    n = _name(symbol)
    ttl = (f"<b style='color:{CYAN}'>{symbol}</b>"
           + (f"  <span style='color:{GRAY}'>{n}</span>" if n!=symbol else "")
           + f"  <span style='color:{GRAY}'>Cumulative Volume Delta</span>"
           + f"  <span style='color:{ORANGE};font-size:.85em'>  {n_div} divergenze</span>")
    _layout(fig, ttl, 500)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=True, tickangle=-30, row=2, col=1)
    fig.update_yaxes(title_text="Prezzo", title_font=dict(size=9,color=GRAY), row=1, col=1)
    fig.update_yaxes(title_text="Delta",  title_font=dict(size=9,color=GRAY), row=2, col=1)
    return fig

# ── VISTA INDICATORI: Candele + RSI + MACD + ADX ──────────────────────────────
def _chart_indicators(df, symbol, ind_sel):
    if df.empty: return go.Figure()
    x  = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    op = df["open"].tolist(); hi = df["high"].tolist()
    lo = df["low"].tolist();  cl = df["close"].tolist()

    # Layout dinamico in base agli indicatori selezionati
    row_h  = [0.48]
    labels = ["Prezzo"]
    if "RSI 14" in ind_sel:    row_h.append(0.17); labels.append("RSI")
    if "MACD"   in ind_sel:    row_h.append(0.18); labels.append("MACD")
    if "ADX 14" in ind_sel:    row_h.append(0.17); labels.append("ADX")
    total  = sum(row_h)
    row_h  = [v/total for v in row_h]
    n_rows = len(row_h)

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=row_h, vertical_spacing=0.01)

    # ── Candele ──
    fig.add_trace(go.Candlestick(
        x=x, open=op, high=hi, low=lo, close=cl, name=symbol, showlegend=False,
        increasing=dict(fillcolor=GREEN, line=dict(color=GREEN, width=1)),
        decreasing=dict(fillcolor=RED,   line=dict(color=RED,   width=1)),
    ), row=1, col=1)

    # ── SMA 9 / 21 ──
    if "SMA 9/21" in ind_sel and "sma9" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["sma9"].tolist(), mode="lines",
            line=dict(color=PURPLE, width=1.5), name="SMA 9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["sma21"].tolist(), mode="lines",
            line=dict(color=GOLD, width=1.5), name="SMA 21"), row=1, col=1)

    # ── EMA 20 / 50 ──
    if "EMA 20/50" in ind_sel and "ema20" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["ema20"].tolist(), mode="lines",
            line=dict(color=ORANGE, width=1.5), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["ema50"].tolist(), mode="lines",
            line=dict(color=BLUE, width=1.5, dash="dot"), name="EMA 50"), row=1, col=1)

    # ── Bollinger Bands ──
    if "Bollinger" in ind_sel and "bb_up" in df.columns:
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=df["bb_up"].tolist() + df["bb_dn"].tolist()[::-1],
            fill="toself", fillcolor="rgba(41,98,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="BB band",
            hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["bb_up"].tolist(), mode="lines",
            line=dict(color=BLUE, width=1), name="BB upper", showlegend=False,
            opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["bb_dn"].tolist(), mode="lines",
            line=dict(color=BLUE, width=1), name="BB lower", showlegend=False,
            opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["bb_mid"].tolist(), mode="lines",
            line=dict(color=CYAN, width=1, dash="dot"), name="BB mid",
            opacity=0.6), row=1, col=1)

    cur_row = 2
    # ── RSI ──
    if "RSI 14" in ind_sel and "rsi" in df.columns:
        rsi = df["rsi"].tolist()
        fig.add_trace(go.Scatter(x=x, y=rsi, mode="lines",
            line=dict(color=PURPLE, width=1.5), name="RSI 14",
            hovertemplate="RSI: %{y:.1f}<extra></extra>"), row=cur_row, col=1)
        fig.add_hline(y=70, row=cur_row, col=1,
                      line=dict(color=RED, width=1, dash="dot"))
        fig.add_hline(y=30, row=cur_row, col=1,
                      line=dict(color=GREEN, width=1, dash="dot"))
        fig.add_hline(y=50, row=cur_row, col=1,
                      line=dict(color=BORDER, width=1))
        fig.add_hrect(y0=70, y1=100, row=cur_row, col=1,
                      fillcolor="rgba(239,83,80,0.05)", line_width=0)
        fig.add_hrect(y0=0,  y1=30,  row=cur_row, col=1,
                      fillcolor="rgba(38,166,154,0.05)", line_width=0)
        fig.update_yaxes(range=[0,100], row=cur_row, col=1)
        fig.update_yaxes(title_text="RSI", title_font=dict(size=9,color=GRAY),
                         row=cur_row, col=1)
        cur_row += 1

    # ── MACD ──
    if "MACD" in ind_sel and "macd" in df.columns:
        hist = df["hist"].tolist()
        fig.add_trace(go.Bar(x=x, y=hist,
            marker_color=[GREEN if h>=0 else RED for h in hist],
            marker_line_width=0, name="MACD hist",
            hovertemplate="Hist: %{y:.4f}<extra></extra>"), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["macd"].tolist(), mode="lines",
            line=dict(color=CYAN, width=1.5), name="MACD"), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["signal"].tolist(), mode="lines",
            line=dict(color=ORANGE, width=1.5), name="Signal"), row=cur_row, col=1)
        fig.add_hline(y=0, row=cur_row, col=1, line=dict(color=BORDER, width=1))
        fig.update_yaxes(title_text="MACD", title_font=dict(size=9,color=GRAY),
                         row=cur_row, col=1)
        cur_row += 1

    # ── ADX ──
    if "ADX 14" in ind_sel and "adx" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["adx"].tolist(), mode="lines",
            line=dict(color=GOLD, width=2), name="ADX 14",
            hovertemplate="ADX: %{y:.1f}<extra></extra>"), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["di_p"].tolist(), mode="lines",
            line=dict(color=GREEN, width=1, dash="dot"), name="+DI",
            opacity=0.8), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["di_m"].tolist(), mode="lines",
            line=dict(color=RED, width=1, dash="dot"), name="-DI",
            opacity=0.8), row=cur_row, col=1)
        fig.add_hline(y=25, row=cur_row, col=1,
                      line=dict(color=GRAY, width=1, dash="dot"))
        fig.update_yaxes(title_text="ADX", title_font=dict(size=9,color=GRAY),
                         row=cur_row, col=1)

    n = _name(symbol)
    inds_str = " · ".join(ind_sel)
    ttl = (f"<b style='color:{CYAN}'>{symbol}</b>"
           + (f"  <span style='color:{GRAY}'>{n}</span>" if n!=symbol else "")
           + f"  <span style='color:{GRAY}'>{inds_str}</span>")
    _layout(fig, ttl, 620)
    for r in range(1, n_rows+1):
        n_ = "" if r==1 else str(r)
        tl = (r == n_rows)
        fig.update_layout(**{f"xaxis{n_}": dict(**_AX, showticklabels=tl,
                                                tickangle=-30 if tl else 0)})
    fig.update_yaxes(title_text="Prezzo", title_font=dict(size=9,color=GRAY), row=1, col=1)
    return fig

# ─── Entry point ─────────────────────────────────────────────────────────────
def render_orderflow_tab(df_scanner=None):
    # ── Header ──
    st.markdown(
        f'<div style="background:{PANEL};border-left:3px solid {ORANGE};'
        f'padding:10px 18px;border-radius:0 6px 6px 0;margin-bottom:10px">'
        f'<span style="color:{ORANGE};font-weight:700;font-size:1.05rem">'
        f'🔬 ORDER FLOW ANALYZER</span>'
        f'<span style="color:{GRAY};font-size:.78rem;margin-left:12px">'
        f'Scegli un ticker e una vista — il grafico si carica automaticamente</span>'
        f'</div>',
        unsafe_allow_html=True)

    # ── Ticker list ──
    sc_tickers = []
    if df_scanner is not None and not df_scanner.empty:
        tc = "Ticker" if "Ticker" in df_scanner.columns else "ticker"
        if tc in df_scanner.columns:
            sc_tickers = df_scanner[tc].dropna().unique().tolist()[:30]
    # v34: costruisci "Nome Azienda  (TICKER)" anche per ticker dallo scanner
    _sc_names: dict = {}  # ticker → nome da df_scanner
    if df_scanner is not None and not df_scanner.empty:
        _tnm = "Ticker" if "Ticker" in df_scanner.columns else "ticker"
        _nnm = "Nome"   if "Nome"   in df_scanner.columns else (
               "name"   if "name"   in df_scanner.columns else None)
        if _tnm in df_scanner.columns and _nnm:
            for _, _sr in df_scanner[[_tnm, _nnm]].dropna(subset=[_tnm]).iterrows():
                _sc_names[str(_sr[_tnm])] = str(_sr[_nnm])

    def _label_v34(t: str) -> str:
        """Nome alfabetico (TICKER) — usa NAMES dict, poi scanner df."""
        n = NAMES.get(t) or _sc_names.get(t)
        return f"{n}  ({t})" if n and n != t else f"({t})"

    merged = list(dict.fromkeys(sc_tickers + _DEFAULT_TKS))
    opts   = sorted([_label_v34(t) for t in merged], key=str.lower)
    d2t    = {_label_v34(t): t for t in merged}

    # ── ROW 1: Ticker + Timeframe + Ticker manuale ───────────────────────
    c1, c2, c3 = st.columns([2.5, 1.5, 2])
    with c1:
        # Evidenzia i ticker dallo scanner (quelli tuoi) rispetto ai default
        group_label = "📡 Scanner" if sc_tickers else "📋 Default"
        sel  = st.selectbox(
            f"Strumento  ({group_label} + default)",
            opts, key="of_sel",
            help="I ticker dallo scanner appaiono prima. Puoi anche digitare un ticker a destra.")
        sym  = d2t.get(sel, sel.split("(")[-1].rstrip(")").strip())
    with c2:
        tf_lbl = st.selectbox(
            "⏱ Timeframe",
            list(TF_MAP.keys()), index=2, key="of_tf",
            help="15m / 1h = intraday  |  1d / 1W = swing")
        sub_iv, main_freq, range_ = TF_MAP[tf_lbl]
    with c3:
        man = st.text_input(
            "✏️ Ticker manuale (opzionale)",
            placeholder="es. ENI.MI · BTC-USD · ES=F · EURUSD=X",
            key="of_man",
            help="Scrivi qualsiasi ticker Yahoo Finance. Sovrascrive la selectbox a sinistra."
        ).strip().upper()
        if man: sym = man

    # ── ROW 2: Vista (orizzontale) + Overlay checkboxes ─────────────────
    st.markdown(
        f'<div style="color:{GRAY};font-size:.74rem;margin:6px 0 2px">📐 Vista grafico:</div>',
        unsafe_allow_html=True)

    v_col, o_col = st.columns([3, 1])
    with v_col:
        vista = st.radio(
            "Vista",
            ["📊 Principale", "📈 CVD + Divergenze", "📉 Indicatori"],
            key="of_vista", horizontal=True,
            help=(
                "Principale: candele + VWAP + Volume Profile + Delta + CVD  |  "
                "CVD: prezzo vs CVD normalizzato con divergenze evidenziate  |  "
                "Indicatori: RSI · MACD · ADX · SMA · Bollinger personalizzabili"
            ),
            label_visibility="collapsed")
    with o_col:
        oc1, oc2, oc3 = st.columns(3)
        with oc1: show_vwap = st.checkbox("VWAP",    value=True,  key="of_vwap",
                                          help="VWAP ±1σ/±2σ — reset giornaliero")
        with oc2: show_ema  = st.checkbox("EMA",     value=True,  key="of_ema",
                                          help="EMA 20 (arancio) + EMA 50 (blu)")
        with oc3: show_vp   = st.checkbox("Vol.Prof",value=True,  key="of_vp",
                                          help="Volume Profile: POC (oro) + VAH/VAL (blu)")

    # ── Indicatori (solo vista Indicatori) ───────────────────────────────
    ind_sel = []
    if vista == "📉 Indicatori":
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-radius:4px;padding:8px 12px;margin:4px 0">'
            f'<span style="color:{GRAY};font-size:.74rem">🔧 Indicatori attivi:</span>'
            f'</div>',
            unsafe_allow_html=True)
        ic1, ic2, ic3, ic4, ic5, ic6 = st.columns(6)
        with ic1: ind_sel += ["RSI 14"]   if st.checkbox("RSI 14",   value=True,  key="of_rsi")  else []
        with ic2: ind_sel += ["MACD"]     if st.checkbox("MACD",     value=True,  key="of_macd") else []
        with ic3: ind_sel += ["EMA 20/50"]if st.checkbox("EMA 20/50",value=True,  key="of_ema2") else []
        with ic4: ind_sel += ["ADX 14"]   if st.checkbox("ADX 14",   value=False, key="of_adx")  else []
        with ic5: ind_sel += ["SMA 9/21"] if st.checkbox("SMA 9/21", value=False, key="of_sma")  else []
        with ic6: ind_sel += ["Bollinger"]if st.checkbox("Bollinger", value=False, key="of_bb")   else []

    # ── Pulsante + Refresh ───────────────────────────────────────────────
    c_run, c_ref = st.columns([6, 1])
    with c_run:
        run = st.button(
            f"▶ Carica  {sym}  [{tf_lbl}]",
            key="of_run", use_container_width=True, type="primary")
    with c_ref:
        if st.button("🔄", key="of_ref", help="Svuota cache Yahoo e ricarica"):
            st.cache_data.clear(); st.rerun()

    # ── Placeholder se non ancora caricato ──────────────────────────────
    if not run:
        # Mostra guida rapida per orientarsi
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-radius:8px;padding:20px 24px;margin-top:10px">'
            f'<div style="color:{TEXT};font-size:.92rem;font-weight:600;margin-bottom:12px">'
            f'📖 Guida rapida Order Flow</div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:.80rem">'
            f'<tr><td style="color:{ORANGE};font-weight:700;padding:4px 12px 4px 0;white-space:nowrap">📊 Principale</td>'
            f'<td style="color:{GRAY}">Candele + VWAP ±1σ/±2σ + Volume Profile (POC/VAH/VAL) + Delta bar + CVD.<br>'
            f'<span style="color:{TEXT}">Usa questa vista per capire dove sta il flusso di volume e i livelli chiave.</span></td></tr>'
            f'<tr><td style="color:{CYAN};font-weight:700;padding:4px 12px 4px 0;white-space:nowrap">📈 CVD</td>'
            f'<td style="color:{GRAY}">Prezzo vs Cumulative Volume Delta normalizzato. Le zone arancioni sono divergenze.<br>'
            f'<span style="color:{TEXT}">CVD sale ma prezzo scende = pressione compratori nascosta (bullish).</span></td></tr>'
            f'<tr><td style="color:{BLUE};font-weight:700;padding:4px 12px 4px 0;white-space:nowrap">📉 Indicatori</td>'
            f'<td style="color:{GRAY}">Scegli liberamente RSI · MACD · ADX · EMA · SMA · Bollinger.<br>'
            f'<span style="color:{TEXT}">Ogni indicatore aggiunge un pannello dedicato sotto le candele.</span></td></tr>'
            f'</table>'
            f'<div style="color:{GRAY};font-size:.72rem;margin-top:10px;border-top:1px solid {BORDER};padding-top:8px">'
            f'💡 VWAP, EMA e Vol.Profile si attivano/disattivano con i checkbox in alto a destra &nbsp;·&nbsp; '
            f'Ticker manuale sovrascrive la selectbox &nbsp;·&nbsp; '
            f'🔄 svuota la cache Yahoo se i dati sembrano vecchi'
            f'</div></div>',
            unsafe_allow_html=True)
        return

    # ── Caricamento ──────────────────────────────────────────────────────
    n_display = _name(sym)
    spin_lbl  = f"{sym} — {n_display}" if n_display != sym else sym
    with st.spinner(f"⏳ Caricamento  {spin_lbl}  [{tf_lbl}]…"):
        df_sub = _fetch(sym, sub_iv, range_)

        if df_sub.empty:
            st.error(
                f"❌ Dati non trovati per **{sym}**.\n\n"
                "Verifica il simbolo Yahoo Finance: `AAPL` · `BTC-USD` · `ES=F` · `EUR=X`")
            return
        df = _resample(df_sub, main_freq)
        if df.empty:
            st.error("❌ Errore nel campionamento. Prova un timeframe diverso."); return
        df       = _indicators(df)
        df_vwap  = _vwap_bands(df_sub) if show_vwap else pd.DataFrame()

    # ── KPI ──
    last = df.iloc[-1]; first = df.iloc[0]
    chg  = (last["close"]/first["open"]-1)*100 if first["open"] else 0
    tb   = float(df["buy_vol"].sum()); ts_ = float(df["sell_vol"].sum())
    td   = float(df["delta"].sum())
    bp   = tb/(tb+ts_)*100 if (tb+ts_)>0 else 50
    dc   = GREEN if td>=0 else RED
    vwap_v = (float(df_vwap["vwap"].iloc[-1])
              if not df_vwap.empty and "vwap" in df_vwap.columns else 0)
    vs_v = ("▲ sopra" if last["close"]>vwap_v and vwap_v>0
            else "▼ sotto" if vwap_v>0 else "–")
    rsi_v = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 0
    rsi_c = (RED if rsi_v>70 else GREEN if rsi_v<30 else GOLD)
    rsi_s = ("Overbought" if rsi_v>70 else "Oversold" if rsi_v<30 else "Neutro")

    k = st.columns(7)
    for (lbl, val, col, sub), kcol in zip([
        ("Ticker",    sym,                    CYAN,   n_display if n_display!=sym else ""),
        ("Close",     f"${last['close']:.2f}",GREEN if chg>=0 else RED,
                                              f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%"),
        ("VWAP",      f"${vwap_v:.2f}" if vwap_v else "–", VWAP_C, vs_v),
        ("RSI 14",    f"{rsi_v:.1f}",         rsi_c,  rsi_s),
        ("Delta tot", f"{'+' if td>=0 else ''}{_fv(td)}", dc,
                                              "Buy dom" if td>=0 else "Sell dom"),
        ("Buy %",     f"{bp:.0f}%",           GREEN,  _fv(tb)),
        ("Sell %",    f"{100-bp:.0f}%",       RED,    _fv(ts_)),
    ], k):
        with kcol:
            st.markdown(_kpi(lbl, val, col, sub), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Grafico + Legenda ──
    if vista == "📊 Principale":
        fig = _chart_main(df, df_vwap, sym, show_vwap, show_ema, show_vp)
        st.plotly_chart(fig, use_container_width=True, key="of_main")
        _legend_strip([
            ("━", "VWAP",      VWAP_C),
            ("░", "±1σ",       VWAP_C),
            ("━", "EMA 20",    ORANGE),
            ("┄", "EMA 50",    BLUE),
            ("◆", "POC",       GOLD),
            ("┄", "VAH/VAL",   BLUE),
            ("▌", "Buy Delta", GREEN),
            ("▌", "Sell Delta",RED),
            ("━", "CVD",       CYAN),
        ])
        _slide_block("principale")

    elif vista == "📈 CVD + Divergenze":
        fig = _chart_cvd(df, df_vwap, sym, show_vwap)
        st.plotly_chart(fig, use_container_width=True, key="of_cvd")
        _legend_strip([
            ("━", "Prezzo Close",         CYAN),
            ("┄", "VWAP",                 VWAP_C),
            ("━", "CVD normalizzato",     ORANGE),
            ("░", "Divergenza pr./CVD",   ORANGE),
            ("▌", "Delta/bar Buy",        GREEN),
            ("▌", "Delta/bar Sell",       RED),
        ])
        _slide_block("cvd")

    else:  # Indicatori
        if not ind_sel:
            st.info("ℹ️ Seleziona almeno un indicatore sopra.")
        else:
            fig = _chart_indicators(df, sym, ind_sel)
            st.plotly_chart(fig, use_container_width=True, key="of_ind")
            items = [("━","EMA 20",ORANGE),("┄","EMA 50",BLUE)]
            if "SMA 9/21"  in ind_sel: items += [("━","SMA 9",PURPLE),("━","SMA 21",GOLD)]
            if "Bollinger" in ind_sel: items += [("░","BB band",BLUE),("┄","BB mid",CYAN)]
            if "RSI 14"    in ind_sel: items += [("━","RSI 14",PURPLE),("┄","OB 70",RED),("┄","OS 30",GREEN)]
            if "MACD"      in ind_sel: items += [("━","MACD",CYAN),("━","Signal",ORANGE),("▌","Hist",GREEN)]
            if "ADX 14"    in ind_sel: items += [("━","ADX",GOLD),("┄","+DI",GREEN),("┄","-DI",RED)]
            _legend_strip(items)
            _slide_block("indicatori")

    # ── Dati candle visuale ───────────────────────────────────────────────
    with st.expander("📋 Dati candle (ultimi 30)", expanded=False):
        scols = ["date","open","high","low","close","volume",
                 "buy_vol","sell_vol","delta","delta_pct","cum_delta"]
        ds = df[scols].tail(30).copy()

        # ── Mini chart candele + delta ────────────────────────────────────
        x_dates = ds["date"].dt.strftime("%H:%M").tolist()
        fig_c = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.04,
        )
        fig_c.add_trace(go.Candlestick(
            x=x_dates,
            open=ds["open"], high=ds["high"],
            low=ds["low"],  close=ds["close"],
            increasing=dict(line=dict(color=GREEN), fillcolor="rgba(38,166,154,0.75)"),
            decreasing=dict(line=dict(color=RED),   fillcolor="rgba(239,83,80,0.75)"),
            name="Price", showlegend=False,
        ), row=1, col=1)
        fig_c.add_trace(go.Bar(
            x=x_dates, y=ds["delta"],
            marker_color=[GREEN if v >= 0 else RED for v in ds["delta"]],
            marker_line_width=0, name="Delta", opacity=0.85,
        ), row=2, col=1)
        fig_c.add_hline(y=0, row=2, col=1, line=dict(color=BORDER, width=1))
        fig_c.update_layout(
            height=320, paper_bgcolor=BG, plot_bgcolor=PANEL,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            font=dict(color=TEXT, size=9),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        )
        fig_c.update_xaxes(showgrid=True, gridcolor=BORDER, tickangle=-30)
        fig_c.update_yaxes(showgrid=True, gridcolor=BORDER,
                           tickfont=dict(size=8))
        fig_c.update_yaxes(title_text="Delta", row=2, col=1,
                           title_font=dict(size=8, color=GRAY))
        st.plotly_chart(fig_c, use_container_width=True, key="of_candle_chart")

        # ── Tabella stilizzata sotto il chart ─────────────────────────────
        ds_disp = ds.copy()
        ds_disp["date"]      = ds_disp["date"].dt.strftime("%Y-%m-%d %H:%M")
        for col in ["volume","buy_vol","sell_vol"]:
            ds_disp[col] = ds_disp[col].apply(_fv)
        ds_disp["delta"]     = ds_disp["delta"].apply(lambda v: f"{'+' if v>=0 else ''}{_fv(v)}")
        ds_disp["delta_pct"] = ds_disp["delta_pct"].apply(lambda v: f"{v:+.1f}%")
        ds_disp["cum_delta"] = ds_disp["cum_delta"].apply(_fv)
        ds_disp["close"]     = ds_disp["close"].apply(lambda v: f"{v:.2f}")
        ds_disp["open"]      = ds_disp["open"].apply(lambda v: f"{v:.2f}")
        ds_disp["high"]      = ds_disp["high"].apply(lambda v: f"{v:.2f}")
        ds_disp["low"]       = ds_disp["low"].apply(lambda v: f"{v:.2f}")
        ds_disp.columns = ["Data","Open","High","Low","Close","Volume",
                           "Buy Vol","Sell Vol","Delta","Δ%","CVD"]

        # Colora Delta e Δ% con HTML
        def _color_cell(val, col_name):
            if col_name in ("Delta", "Δ%"):
                c = GREEN if str(val).startswith("+") else RED
                return f'<span style="color:{c};font-weight:600;font-family:Courier New">{val}</span>'
            return f'<span style="font-family:Courier New;font-size:0.82rem">{val}</span>'

        rows_html = ""
        for _, row_ in ds_disp.iterrows():
            rows_html += "<tr>"
            for cn in ds_disp.columns:
                rows_html += f"<td style='padding:3px 8px;border-bottom:1px solid {BORDER};white-space:nowrap'>{_color_cell(row_[cn], cn)}</td>"
            rows_html += "</tr>"

        headers = "".join(
            f'<th style="color:{CYAN};font-family:Courier New;font-size:0.72rem;'
            f'text-transform:uppercase;letter-spacing:0.5px;padding:5px 8px;'
            f'border-bottom:1px solid {ORANGE};text-align:left">{h}</th>'
            for h in ds_disp.columns
        )
        st.markdown(
            f'<div style="overflow-x:auto;margin-top:10px">'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.80rem;'
            f'background:{PANEL};border-radius:4px">'
            f'<thead><tr>{headers}</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True
        )

    # ── Nota metodologica ──
    with st.expander("ℹ️ Metodologia dati", expanded=False):
        st.markdown(f"""
**Fonte:** Yahoo Finance OHLCV intraday (gratuito).

**Delta Buy/Sell** — *Candle Body Ratio* (standard TradingView replay storico):
`buy_vol ≈ volume × (close − low) / (high − low)` · Accuratezza ~70-80% su strumenti liquidi.

**VWAP** con bande ±1σ / ±2σ, reset giornaliero.
**Volume Profile:** POC (max volume) · Value Area 70% (VAH/VAL).
**CVD:** Cumulative Volume Delta — divergenze con il prezzo segnalano potenziali inversioni.
""")

    # ── Footer ──
    st.markdown(
        f'<div style="color:{GRAY};font-size:.69rem;text-align:center;'
        f'margin-top:12px;padding-top:8px;border-top:1px solid {BORDER}">'
        f'Yahoo Finance OHLCV · Candle Body Ratio · Cache 5min · v34.0 · '
        f'{datetime.now().strftime("%d/%m/%Y %H:%M")}'
        f'</div>',
        unsafe_allow_html=True)
