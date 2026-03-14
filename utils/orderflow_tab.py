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

# ─── Bloomberg Terminal Palette v33 ──────────────────────────────────────────
BG     = "#070b14"
PANEL  = "#0d1117"
PANEL2 = "#111923"
BORDER = "#1c2333"
BORDER2= "#243044"
GREEN  = "#00d4aa"
RED    = "#ff3d57"
GOLD   = "#ffb800"
BLUE   = "#2979ff"
CYAN   = "#00b8d4"
GRAY   = "#5a6478"
TEXT   = "#c8d0e0"
ORANGE = "#ff6d00"
PURPLE = "#7c4dff"
VWAP_C = "#ff6d00"
G_DARK = "rgba(0,212,170,0.12)"
R_DARK = "rgba(255,61,87,0.12)"
MONO   = "'IBM Plex Mono','Courier New',monospace"

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

    return df

# ══════════════════════════════════════════════════════════════════════════════
# v33 UPGRADE — LARGE TRADE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
def _large_trades(df: pd.DataFrame, sigma: float = 2.0) -> pd.DataFrame:
    """
    Identifica barre con volume anomalo (> media + N*sigma).
    Output colonne aggiunte: vol_zscore, is_large, large_side ('BUY'|'SELL'|'NEUTRAL')
    """
    df = df.copy()
    vol_mean = df["volume"].rolling(20, min_periods=5).mean()
    vol_std  = df["volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    df["vol_zscore"] = ((df["volume"] - vol_mean) / vol_std).round(2)
    df["is_large"]   = df["vol_zscore"] >= sigma
    # Lato dominante: se delta > 20% del volume → BUY, < -20% → SELL
    delta_pct = df.get("delta_pct", pd.Series(0, index=df.index))
    df["large_side"] = np.where(
        ~df["is_large"], "",
        np.where(delta_pct >= 20, "BUY",
        np.where(delta_pct <= -20, "SELL", "NEUTRAL"))
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# v33 UPGRADE — PRICE-LEVEL IMBALANCE HEATMAP (Footprint semplificato)
# ══════════════════════════════════════════════════════════════════════════════
def _imbalance_heatmap(df: pd.DataFrame, n_bins: int = 30) -> dict:
    """
    Distribuisce buy_vol e sell_vol per livello di prezzo.
    Restituisce {centers, buy_by_price, sell_by_price, imbalance_pct, poc, vah, val}
    per costruire la heatmap laterale stile footprint.
    """
    if df.empty or len(df) < 5:
        return {}
    try:
        p_min = float(df["low"].min())
        p_max = float(df["high"].max())
        if p_max <= p_min:
            return {}
        bins    = np.linspace(p_min, p_max, n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        buy_acc  = np.zeros(n_bins)
        sell_acc = np.zeros(n_bins)

        for _, row in df.iterrows():
            h, l = float(row["high"]), float(row["low"])
            bv   = float(row.get("buy_vol",  row["volume"] * 0.5))
            sv   = float(row.get("sell_vol", row["volume"] * 0.5))
            span = h - l if h > l else 1e-9
            for b in range(n_bins):
                lo_b = max(bins[b],   l)
                hi_b = min(bins[b+1], h)
                if hi_b <= lo_b:
                    continue
                frac = (hi_b - lo_b) / span
                buy_acc[b]  += bv * frac
                sell_acc[b] += sv * frac

        total = buy_acc + sell_acc
        imb   = np.where(total > 0,
                         (buy_acc - sell_acc) / total * 100, 0)

        # POC & Value Area 70% (sul volume totale)
        poc_i = int(np.argmax(total))
        poc   = float(centers[poc_i])
        tgt   = total.sum() * 0.70
        acc_v = total[poc_i]; lo_i = hi_i = poc_i
        while acc_v < tgt and (lo_i > 0 or hi_i < n_bins - 1):
            add_lo = total[lo_i-1] if lo_i > 0 else 0
            add_hi = total[hi_i+1] if hi_i < n_bins-1 else 0
            if add_hi >= add_lo and hi_i < n_bins-1:
                hi_i += 1; acc_v += add_hi
            elif lo_i > 0:
                lo_i -= 1; acc_v += add_lo
            else:
                hi_i += 1; acc_v += add_hi
        vah = float(centers[hi_i])
        val = float(centers[lo_i])

        return {
            "centers":       centers.tolist(),
            "buy":           buy_acc.tolist(),
            "sell":          sell_acc.tolist(),
            "imbalance_pct": imb.tolist(),
            "poc":  poc, "vah": vah, "val": val,
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# v33 UPGRADE — CHART LARGE TRADES (candele con spike evidenziati)
# ══════════════════════════════════════════════════════════════════════════════
def _chart_large_trades(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Candele normali + overlay Large Trade markers + Volume Z-score bar.
    Barre con Z ≥ 2 → cerchio colorato sul grafico (verde=BUY dom, rosso=SELL dom, giallo=NEUTRAL).
    """
    df = _large_trades(df, sigma=2.0)
    x  = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.25, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # Row 1 — Candele
    fig.add_trace(go.Candlestick(
        x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing=dict(line=dict(color=GREEN), fillcolor=f"rgba(0,212,170,0.7)"),
        decreasing=dict(line=dict(color=RED),   fillcolor=f"rgba(255,61,87,0.7)"),
        name="Price", showlegend=False,
    ), row=1, col=1)

    # Large trade markers
    for side, col_m, sym_m in [("BUY","#00d4aa","circle"), ("SELL","#ff3d57","circle"), ("NEUTRAL","#ffb800","diamond")]:
        mask = (df["is_large"]) & (df["large_side"] == side)
        if mask.any():
            sub = df[mask]
            y_pos = sub["high"] * 1.003 if side != "SELL" else sub["low"] * 0.997
            fig.add_trace(go.Scatter(
                x=[x[i] for i in sub.index],
                y=y_pos.tolist(),
                mode="markers",
                marker=dict(size=10, color=col_m, symbol=sym_m,
                            line=dict(color="#000", width=1)),
                name=f"LT {side}",
                hovertemplate=(
                    f"<b>LARGE {side}</b><br>"
                    "Vol Z: %{customdata[0]:.1f}σ<br>"
                    "Delta%: %{customdata[1]:+.1f}%<extra></extra>"
                ),
                customdata=sub[["vol_zscore","delta_pct"]].values,
            ), row=1, col=1)

    # Row 2 — Delta bars colorati
    dc = [GREEN if v >= 0 else RED for v in df["delta"]]
    fig.add_trace(go.Bar(
        x=x, y=df["delta"],
        marker_color=dc, marker_line_width=0,
        name="Delta", opacity=0.85,
        hovertemplate="Delta: %{y:+,.0f}<extra></extra>",
    ), row=2, col=1)
    # CVD line
    fig.add_trace(go.Scatter(
        x=x, y=df["cum_delta"],
        line=dict(color=CYAN, width=1.5),
        name="CVD", yaxis="y4",
        hovertemplate="CVD: %{y:+,.0f}<extra></extra>",
    ), row=2, col=1)

    # Row 3 — Volume Z-score
    zc = [GREEN if z >= 2 else (ORANGE if z >= 1 else GRAY)
          for z in df["vol_zscore"].fillna(0)]
    fig.add_trace(go.Bar(
        x=x, y=df["vol_zscore"].fillna(0),
        marker_color=zc, marker_line_width=0,
        name="Vol Z-score", opacity=0.9,
        hovertemplate="Z: %{y:.2f}σ<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=2, line=dict(color=ORANGE, width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=0, line=dict(color=BORDER, width=1), row=3, col=1)

    _layout(fig, f"<b>{symbol}</b>  Large Trade Detector  (σ≥2.0)", 540)
    fig.update_yaxes(title_text="Price",   row=1, col=1, **{k:v for k,v in _AX.items()})
    fig.update_yaxes(title_text="Delta",   row=2, col=1, **{k:v for k,v in _AX.items()})
    fig.update_yaxes(title_text="Z-score", row=3, col=1, **{k:v for k,v in _AX.items()})
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# v33 UPGRADE — CHART IMBALANCE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def _chart_imbalance(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    2 colonne: Candele (70%) | Imbalance Heatmap laterale (30%)
    La heatmap mostra per ogni livello di prezzo: % imbalance buy vs sell.
    Verde = buy dominante, rosso = sell dominante.
    """
    imb = _imbalance_heatmap(df)
    if not imb:
        fig = go.Figure()
        fig.update_layout(title="Dati insufficienti", **{k:v for k,v in
            dict(paper_bgcolor=BG, plot_bgcolor=PANEL).items()})
        return fig

    x = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.75, 0.25],
        column_widths=[0.72, 0.28],
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.02, horizontal_spacing=0.01,
        subplot_titles=["", "IMBALANCE MAP", "", ""],
    )

    # Col 1 Row 1 — Candele
    fig.add_trace(go.Candlestick(
        x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing=dict(line=dict(color=GREEN), fillcolor="rgba(0,212,170,0.7)"),
        decreasing=dict(line=dict(color=RED),   fillcolor="rgba(255,61,87,0.7)"),
        name="Price", showlegend=False,
    ), row=1, col=1)

    # POC / VAH / VAL lines
    for lvl, col_l, lbl in [
        (imb["poc"], GOLD,  "POC"),
        (imb["vah"], CYAN,  "VAH"),
        (imb["val"], CYAN,  "VAL"),
    ]:
        fig.add_hline(y=lvl, line=dict(color=col_l, width=1.2, dash="dot"),
                      annotation_text=f" {lbl}", annotation_font_color=col_l,
                      annotation_font_size=9, row=1, col=1)

    # Col 2 Row 1 — Imbalance heatmap (barre orizzontali)
    centers = imb["centers"]
    imb_pct = imb["imbalance_pct"]
    bar_colors = [
        f"rgba(0,212,170,{min(abs(v)/100*0.9+0.1, 0.95):.2f})" if v >= 0
        else f"rgba(255,61,87,{min(abs(v)/100*0.9+0.1, 0.95):.2f})"
        for v in imb_pct
    ]
    bin_w = (centers[1] - centers[0]) * 0.85 if len(centers) > 1 else 1.0
    fig.add_trace(go.Bar(
        x=imb_pct, y=centers,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        width=[bin_w] * len(centers),
        name="Imbalance %",
        hovertemplate="Price: %{y:.2f}<br>Imbalance: %{x:+.1f}%<extra></extra>",
    ), row=1, col=2)

    # Linea zero e POC sull'heatmap
    fig.add_vline(x=0, line=dict(color=GRAY, width=1), row=1, col=2)
    fig.add_hline(y=imb["poc"], line=dict(color=GOLD, width=1.5, dash="dot"), row=1, col=2)

    # Col 1 Row 2 — Volume bar
    vc = [GREEN if c >= o else RED for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=x, y=df["volume"], marker_color=vc, marker_line_width=0,
        opacity=0.7, name="Volume", showlegend=False,
    ), row=2, col=1)

    _layout(fig, f"<b>{symbol}</b>  Footprint Imbalance Heatmap", 500)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text="Vol", row=2, col=1, **{k:v for k,v in _AX.items()})
    return fig

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
    # ── Bloomberg Header ──────────────────────────────────────────────────
    now_str = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(
        f'<div style="background:{PANEL};border-top:2px solid {ORANGE};'
        f'border-bottom:1px solid {BORDER};padding:7px 16px;'
        f'display:flex;align-items:center;gap:10px;margin-bottom:8px">'
        f'<span style="background:{ORANGE};color:#000;font-family:{MONO};'
        f'font-size:0.67rem;font-weight:700;padding:2px 8px;letter-spacing:2px">FLOW</span>'
        f'<span style="color:{ORANGE};font-family:{MONO};font-size:0.67rem;'
        f'font-weight:700;letter-spacing:2px">ORDER FLOW ANALYZER v33</span>'
        f'<span style="color:{GRAY};font-family:{MONO};font-size:0.62rem;'
        f'margin-left:auto">'
        f'VWAP·VP·DELTA·CVD·LARGE TRADES·IMBALANCE  ·  {now_str}'
        f'</span></div>',
        unsafe_allow_html=True)

    # ── Ticker list ──
    sc_tickers = []
    if df_scanner is not None and not df_scanner.empty:
        tc = "Ticker" if "Ticker" in df_scanner.columns else "ticker"
        if tc in df_scanner.columns:
            sc_tickers = df_scanner[tc].dropna().unique().tolist()[:30]
    merged = list(dict.fromkeys(sc_tickers + _DEFAULT_TKS))
    opts   = sorted([_label(t) for t in merged], key=str.lower)
    d2t    = {_label(t): t for t in merged}

    # ── Controlli compatti Bloomberg style ──
    c1, c2, c3, c4 = st.columns([3, 1.5, 2.2, 1])
    with c1:
        sel  = st.selectbox("Strumento", opts, key="of_sel")
        sym  = d2t.get(sel, sel.split("(")[-1].rstrip(")").strip())
        man  = st.text_input("Ticker manuale", placeholder="es. BTC-USD · ES=F · EURUSD=X",
                             key="of_man").strip().upper()
        if man: sym = man
    with c2:
        tf_lbl = st.selectbox("Timeframe", list(TF_MAP.keys()), index=2, key="of_tf")
        sub_iv, main_freq, range_ = TF_MAP[tf_lbl]
    with c3:
        vista = st.radio("Vista",
            ["📊 Principale", "📈 CVD + Divergenze", "📉 Indicatori",
             "🔴 Large Trades v33", "🟥 Imbalance Map v33"],
            key="of_vista", horizontal=False)
    with c4:
        st.write(""); st.write("")
        show_vwap = st.checkbox("VWAP ±σ",    value=True, key="of_vwap")
        show_ema  = st.checkbox("EMA 20/50",  value=True, key="of_ema")
        show_vp   = st.checkbox("Vol Profile", value=True, key="of_vp")
        lt_sigma  = st.slider("Large σ", 1.0, 3.5, 2.0, 0.5, key="of_sigma",
                              help="Soglia Z-score per Large Trade Detector")

    # Indicatori selezionabili solo per vista Indicatori
    ind_sel = []
    if vista == "📉 Indicatori":
        st.markdown(
            f'<div style="color:{GRAY};font-family:{MONO};font-size:0.72rem;margin-bottom:4px">'
            f'INDICATORI ATTIVI:</div>', unsafe_allow_html=True)
        ic = st.columns(4)
        with ic[0]: ind_sel += ["RSI 14"]  if st.checkbox("RSI 14",  value=True,  key="of_rsi")  else []
        with ic[1]: ind_sel += ["MACD"]    if st.checkbox("MACD",    value=True,  key="of_macd") else []
        with ic[2]: ind_sel += ["ADX 14"]  if st.checkbox("ADX 14",  value=False, key="of_adx")  else []
        with ic[3]: pass
        ic2 = st.columns(4)
        with ic2[0]: ind_sel += ["SMA 9/21"]  if st.checkbox("SMA 9/21",  value=False, key="of_sma")  else []
        with ic2[1]: ind_sel += ["EMA 20/50"] if st.checkbox("EMA 20/50", value=True,  key="of_ema2") else []
        with ic2[2]: ind_sel += ["Bollinger"] if st.checkbox("Bollinger",  value=False, key="of_bb")   else []

    c_run, c_ref = st.columns([5, 1])
    with c_run:
        run = st.button("▶ CARICA", key="of_run", use_container_width=True, type="primary")
    with c_ref:
        if st.button("⟳", key="of_ref", help="Svuota cache"):
            st.cache_data.clear(); st.rerun()

    if not run:
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};'
            f'border-top:2px solid {ORANGE};'
            f'border-radius:2px;padding:48px;text-align:center;margin-top:8px">'
            f'<div style="font-family:{MONO};font-size:0.65rem;color:{GRAY};'
            f'letter-spacing:2px;text-transform:uppercase">ORDER FLOW ANALYZER</div>'
            f'<div style="color:{TEXT};font-family:{MONO};font-size:0.9rem;'
            f'font-weight:700;margin:10px 0">'
            f'Seleziona strumento e clicca <span style="color:{ORANGE}">▶ CARICA</span></div>'
            f'<div style="color:{GRAY};font-family:{MONO};font-size:0.68rem;margin-top:6px">'
            f'VWAP ±1σ/±2σ  ·  Volume Profile POC/VAH/VAL  ·  Delta  ·  CVD'
            f'  ·  Large Trade Detector  ·  Imbalance Heatmap'
            f'</div></div>',
            unsafe_allow_html=True)
        return

    # ── Caricamento dati ──────────────────────────────────────────────────
    n_display = _name(sym)
    spin_lbl  = f"{sym} — {n_display}" if n_display != sym else sym
    with st.spinner(f"⏳ {spin_lbl}  [{tf_lbl}]…"):
        df_sub = _fetch(sym, sub_iv, range_)
        if df_sub.empty:
            st.error(
                f"❌ Dati non trovati per **{sym}**.\n\n"
                "Verifica il simbolo Yahoo Finance: `AAPL` · `BTC-USD` · `ES=F` · `EUR=X`")
            return
        df = _resample(df_sub, main_freq)
        if df.empty:
            st.error("❌ Errore nel campionamento. Prova un timeframe diverso."); return
        df      = _indicators(df)
        df_vwap = _vwap_bands(df_sub) if show_vwap else pd.DataFrame()

    # ── KPI Bar Bloomberg style ───────────────────────────────────────────
    last = df.iloc[-1]; first = df.iloc[0]
    chg  = (last["close"]/first["open"]-1)*100 if first["open"] else 0
    tb   = float(df["buy_vol"].sum()); ts_ = float(df["sell_vol"].sum())
    td   = float(df["delta"].sum())
    bp   = tb/(tb+ts_)*100 if (tb+ts_) > 0 else 50
    dc   = GREEN if td >= 0 else RED
    vwap_v = (float(df_vwap["vwap"].iloc[-1])
              if not df_vwap.empty and "vwap" in df_vwap.columns else 0)
    vs_v = ("▲ sopra" if last["close"] > vwap_v and vwap_v > 0
            else "▼ sotto" if vwap_v > 0 else "–")
    rsi_v = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 0
    rsi_c = RED if rsi_v > 70 else (GREEN if rsi_v < 30 else GOLD)
    rsi_s = "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else "Neutro")

    # v33: Large Trade count
    df_lt = _large_trades(df, sigma=lt_sigma)
    n_lt  = int(df_lt["is_large"].sum())
    lt_buy  = int(((df_lt["large_side"] == "BUY")  & df_lt["is_large"]).sum())
    lt_sell = int(((df_lt["large_side"] == "SELL") & df_lt["is_large"]).sum())
    lt_col  = (GREEN if lt_buy > lt_sell else RED if lt_sell > lt_buy else GOLD)
    lt_lbl  = (f"▲{lt_buy}B ▼{lt_sell}S" if n_lt > 0 else "nessuno")

    k = st.columns(8)
    for (lbl, val, col, sub), kcol in zip([
        ("TICKER",    sym,                    ORANGE, n_display if n_display != sym else ""),
        ("CLOSE",     f"${last['close']:.2f}",GREEN if chg >= 0 else RED,
                                              f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%"),
        ("VWAP",      f"${vwap_v:.2f}" if vwap_v else "–", VWAP_C, vs_v),
        ("RSI 14",    f"{rsi_v:.1f}",         rsi_c,  rsi_s),
        ("DELTA TOT", f"{'+' if td>=0 else ''}{_fv(td)}", dc,
                                              "Buy dom" if td >= 0 else "Sell dom"),
        ("BUY %",     f"{bp:.0f}%",           GREEN,  _fv(tb)),
        ("SELL %",    f"{100-bp:.0f}%",       RED,    _fv(ts_)),
        ("LARGE TR",  str(n_lt),              lt_col, lt_lbl),
    ], k):
        with kcol:
            st.markdown(_kpi(lbl, val, col, sub), unsafe_allow_html=True)

    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:8px 0"></div>',
        unsafe_allow_html=True)

    # ── Grafico + Legenda ─────────────────────────────────────────────────
    if vista == "📊 Principale":
        fig = _chart_main(df, df_vwap, sym, show_vwap, show_ema, show_vp)
        st.plotly_chart(fig, use_container_width=True, key="of_main")
        _legend_strip([
            ("━", "VWAP",      VWAP_C), ("░", "±1σ", VWAP_C),
            ("━", "EMA 20",    ORANGE), ("┄", "EMA 50",    BLUE),
            ("◆", "POC",       GOLD),   ("┄", "VAH/VAL",   CYAN),
            ("▌", "Buy Delta", GREEN),  ("▌", "Sell Delta", RED),
            ("━", "CVD",       CYAN),
        ])
        _slide_block("principale")

    elif vista == "📈 CVD + Divergenze":
        fig = _chart_cvd(df, df_vwap, sym, show_vwap)
        st.plotly_chart(fig, use_container_width=True, key="of_cvd")
        _legend_strip([
            ("━", "Prezzo Close", CYAN), ("┄", "VWAP", VWAP_C),
            ("━", "CVD norm",     ORANGE), ("░", "Divergenza", ORANGE),
            ("▌", "Delta Buy",    GREEN),  ("▌", "Delta Sell",  RED),
        ])
        _slide_block("cvd")

    elif vista == "📉 Indicatori":
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

    elif vista == "🔴 Large Trades v33":
        # Usa df_lt già calcolato con sigma sidebar
        fig = _chart_large_trades(df_lt, sym)
        st.plotly_chart(fig, use_container_width=True, key="of_lt")
        _legend_strip([
            ("●", "Large BUY",     GREEN),
            ("●", "Large SELL",    RED),
            ("◆", "Large NEUTRAL", GOLD),
            ("▌", "Delta bar",     CYAN),
            ("━", "CVD",           CYAN),
            ("▌", "Z-score ≥2σ",   ORANGE),
            ("▌", "Z-score ≥1σ",   GRAY),
        ])
        # Tabella large trades
        lt_rows = df_lt[df_lt["is_large"]].copy()
        if not lt_rows.empty:
            with st.expander(f"📋 Large Trades rilevati ({len(lt_rows)})", expanded=True):
                disp = lt_rows[["date","close","volume","vol_zscore","delta","delta_pct","large_side"]].copy()
                disp["date"]      = disp["date"].dt.strftime("%H:%M")
                disp["volume"]    = disp["volume"].apply(_fv)
                disp["vol_zscore"]= disp["vol_zscore"].apply(lambda v: f"{v:.1f}σ")
                disp["delta"]     = disp["delta"].apply(lambda v: f"{v:+,.0f}")
                disp["delta_pct"] = disp["delta_pct"].apply(lambda v: f"{v:+.1f}%")
                disp.columns = ["Ora","Close","Volume","Z-Score","Delta","Δ%","Side"]
                st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info(f"Nessun Large Trade con Z-score ≥ {lt_sigma:.1f}σ nel periodo selezionato.")

    elif vista == "🟥 Imbalance Map v33":
        fig = _chart_imbalance(df, sym)
        st.plotly_chart(fig, use_container_width=True, key="of_imb")
        _legend_strip([
            ("█", "Buy Imbalance",  GREEN),
            ("█", "Sell Imbalance", RED),
            ("◆", "POC",            GOLD),
            ("┄", "VAH/VAL",        CYAN),
        ])
        # Metriche imbalance
        imb_data = _imbalance_heatmap(df)
        if imb_data:
            ia, ib, ic_, id_ = st.columns(4)
            ia.metric("POC",  f"${imb_data['poc']:.2f}", help="Point of Control — max volume")
            ib.metric("VAH",  f"${imb_data['vah']:.2f}", help="Value Area High (70%)")
            ic_.metric("VAL", f"${imb_data['val']:.2f}", help="Value Area Low (70%)")
            # Imbalance dominante (media pesata)
            imb_mean = float(np.mean(imb_data["imbalance_pct"]))
            id_.metric("Imb. medio",
                       f"{imb_mean:+.1f}%",
                       delta=None,
                       help="+= buy dominante globalmente, -= sell dominante")

    # ── Tabella dati candle ──────────────────────────────────────────────
    with st.expander("📋 Dati candle (ultimi 30)", expanded=False):
        scols = ["date","open","high","low","close","volume",
                 "buy_vol","sell_vol","delta","delta_pct","cum_delta"]
        ds = df[scols].tail(30).copy()
        ds["date"]      = ds["date"].dt.strftime("%Y-%m-%d %H:%M")
        for c in ["volume","buy_vol","sell_vol"]:
            ds[c] = ds[c].apply(_fv)
        ds["delta"]     = ds["delta"].apply(lambda v: f"{'+' if v>=0 else ''}{_fv(v)}")
        ds["delta_pct"] = ds["delta_pct"].apply(lambda v: f"{v:+.1f}%")
        ds["cum_delta"] = ds["cum_delta"].apply(_fv)
        ds.columns = ["Data","Open","High","Low","Close","Volume",
                      "Buy Vol","Sell Vol","Delta","Δ%","CVD"]
        st.dataframe(ds, use_container_width=True, hide_index=True)

    # ── Nota metodologica ────────────────────────────────────────────────
    with st.expander("ℹ️ Metodologia dati", expanded=False):
        st.markdown(f"""
**Fonte:** Yahoo Finance OHLCV intraday (gratuito).

**Delta Buy/Sell** — *Candle Body Ratio*:
`buy_vol ≈ volume × (close − low) / (high − low)` · Accuratezza ~70-80% su strumenti liquidi.

**Large Trade Detector v33** — Rileva barre con volume Z-score ≥ soglia impostata (default 2σ).
Il Z-score è calcolato su rolling 20 barre. Il lato dominante è determinato dal delta_pct:
≥+20% → BUY, ≤-20% → SELL, tra i due → NEUTRAL.

**Imbalance Heatmap v33** — Footprint semplificato: distribuisce buy_vol/sell_vol per livello di prezzo
in {30} bin. Il colore indica % di imbalance: verde = buy dominante, rosso = sell dominante.
POC e Value Area (70%) calcolati sul volume totale per livello.

**VWAP** con bande ±1σ / ±2σ, reset giornaliero.
**CVD:** Cumulative Volume Delta — divergenze con il prezzo segnalano potenziali inversioni.
""")

    # ── Footer Bloomberg ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{GRAY};font-family:{MONO};font-size:0.60rem;'
        f'text-align:center;margin-top:10px;padding-top:8px;'
        f'border-top:1px solid {BORDER}">'
        f'YAHOO FINANCE OHLCV  ·  CANDLE BODY RATIO  ·  CACHE 5MIN  ·  v33  ·  '
        f'{datetime.now().strftime("%d/%m/%Y %H:%M")}'
        f'</div>',
        unsafe_allow_html=True)

    # ── Ticker list ──
    sc_tickers = []
    if df_scanner is not None and not df_scanner.empty:
        tc = "Ticker" if "Ticker" in df_scanner.columns else "ticker"
        if tc in df_scanner.columns:
            sc_tickers = df_scanner[tc].dropna().unique().tolist()[:30]
    merged = list(dict.fromkeys(sc_tickers + _DEFAULT_TKS))
    opts   = sorted([_label(t) for t in merged], key=str.lower)
    d2t    = {_label(t): t for t in merged}

    # ── Controlli ──
    c1, c2, c3, c4 = st.columns([3, 1.5, 1.8, 1])
    with c1:
        sel  = st.selectbox("Strumento", opts, key="of_sel")
        sym  = d2t.get(sel, sel.split("(")[-1].rstrip(")").strip())
        man  = st.text_input("Ticker manuale", placeholder="es. BTC-USD · ES=F · EURUSD=X",
                             key="of_man").strip().upper()
        if man: sym = man
    with c2:
        tf_lbl = st.selectbox("Timeframe", list(TF_MAP.keys()), index=2, key="of_tf")
        sub_iv, main_freq, range_ = TF_MAP[tf_lbl]
    with c3:
        vista = st.radio("Vista",
            ["📊 Principale", "📈 CVD + Divergenze", "📉 Indicatori"],
            key="of_vista")
    with c4:
        st.write(""); st.write("")
        show_vwap = st.checkbox("VWAP ±σ",    value=True, key="of_vwap")
        show_ema  = st.checkbox("EMA 20/50",  value=True, key="of_ema")
        show_vp   = st.checkbox("Vol Profile", value=True, key="of_vp")

    # Indicatori selezionabili solo per vista Indicatori
    ind_sel = []
    if vista == "📉 Indicatori":
        st.markdown(
            f'<div style="color:{GRAY};font-size:.78rem;margin-bottom:4px">'
            f'Seleziona gli indicatori da visualizzare:</div>',
            unsafe_allow_html=True)
        ic = st.columns(4)
        with ic[0]: ind_sel += ["RSI 14"] if st.checkbox("RSI 14", value=True, key="of_rsi") else []
        with ic[1]: ind_sel += ["MACD"]   if st.checkbox("MACD",   value=True, key="of_macd") else []
        with ic[2]: ind_sel += ["ADX 14"] if st.checkbox("ADX 14", value=False,key="of_adx") else []
        with ic[3]: pass
        ic2 = st.columns(4)
        with ic2[0]: ind_sel += ["SMA 9/21"]  if st.checkbox("SMA 9/21",  value=False,key="of_sma") else []
        with ic2[1]: ind_sel += ["EMA 20/50"] if st.checkbox("EMA 20/50", value=True, key="of_ema2") else []
        with ic2[2]: ind_sel += ["Bollinger"] if st.checkbox("Bollinger",  value=False,key="of_bb") else []

    c_run, c_ref = st.columns([5, 1])
    with c_run:
        run = st.button("▶ Carica", key="of_run", use_container_width=True, type="primary")
    with c_ref:
        if st.button("🔄", key="of_ref", help="Svuota cache e ricarica"):
            st.cache_data.clear(); st.rerun()

    if not run:
        st.markdown(
            f'<div style="background:{PANEL};border:1px dashed {BORDER};'
            f'border-radius:8px;padding:55px;text-align:center;margin-top:8px">'
            f'<div style="font-size:2.2rem">📊</div>'
            f'<div style="color:{TEXT};font-size:1rem;font-weight:600;margin-top:8px">'
            f'Seleziona strumento e clicca '
            f'<b style="color:{ORANGE}">▶ Carica</b></div>'
            f'<div style="color:{GRAY};font-size:.83rem;margin-top:5px">'
            f'VWAP ±1σ/±2σ · Volume Profile POC/VAH/VAL · Delta · CVD · '
            f'RSI · MACD · ADX · SMA · Bollinger</div>'
            f'</div>',
            unsafe_allow_html=True)
        return

    # ── Caricamento ──
    n_display = _name(sym)
    spin_lbl  = f"{sym} — {n_display}" if n_display != sym else sym
    with st.spinner(f"⏳ {spin_lbl}  [{tf_lbl}]…"):
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

    # ── Tabella dati ──
    with st.expander("📋 Dati candle (ultimi 30)", expanded=False):
        scols = ["date","open","high","low","close","volume",
                 "buy_vol","sell_vol","delta","delta_pct","cum_delta"]
        ds = df[scols].tail(30).copy()
        ds["date"]      = ds["date"].dt.strftime("%Y-%m-%d %H:%M")
        for c in ["volume","buy_vol","sell_vol"]:
            ds[c] = ds[c].apply(_fv)
        ds["delta"]     = ds["delta"].apply(lambda v: f"{'+' if v>=0 else ''}{_fv(v)}")
        ds["delta_pct"] = ds["delta_pct"].apply(lambda v: f"{v:+.1f}%")
        ds["cum_delta"] = ds["cum_delta"].apply(_fv)
        ds.columns = ["Data","Open","High","Low","Close","Volume",
                      "Buy Vol","Sell Vol","Delta","Δ%","CVD"]
        st.dataframe(ds, use_container_width=True, hide_index=True)

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
        f'Yahoo Finance OHLCV · Candle Body Ratio · Cache 5min · v31.1 · '
        f'{datetime.now().strftime("%d/%m/%Y %H:%M")}'
        f'</div>',
        unsafe_allow_html=True)
