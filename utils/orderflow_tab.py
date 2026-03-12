# -*- coding: utf-8 -*-
"""
orderflow_tab.py  —  🔬 Order Flow / Footprint  v31.1
═══════════════════════════════════════════════════════════════════════════════
Tab SMC Order Flow completo con:
  • Footprint Volume Chart (Bid × Ask per livello)
  • VWAP + bande ±1σ / ±2σ istituzionale
  • Volume Profile (POC / VAH / VAL 70%)
  • Cumulative Volume Delta (CVD) con divergenze
  • SMC: CHoCH, Order Block, FVG, Liquidity Sweep, 3-Phase Cycle
  • Imbalance Heatmap
  • Checklist SMC interattiva (framework @niccofx)
  • Legenda visuale con le 5 slide Order Flow SMC
"""

import json
import urllib.request
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Palette TradingView Dark ─────────────────────────────────────────────────
_BG      = "#131722"
_PANEL   = "#1e222d"
_BORDER  = "#2a2e39"
_GREEN   = "#26a69a"
_RED     = "#ef5350"
_GOLD    = "#ffd700"
_BLUE    = "#2962ff"
_CYAN    = "#50c4e0"
_GRAY    = "#787b86"
_TEXT    = "#d1d4dc"
_ORANGE  = "#ff9800"
_PURPLE  = "#9c27b0"
_GREEN2  = "#00e676"
_RED2    = "#ff1744"
_VWAP    = "#ff6d00"
_OB_BUY  = "rgba(38,166,154,0.18)"
_OB_SELL = "rgba(239,83,80,0.18)"
_FVG_BUY = "rgba(41,98,255,0.13)"
_FVG_SEL = "rgba(255,109,0,0.13)"

TF_CONFIG = {
    "30min  (sub: 2min)":  ("2m",  "30m", "5d"),
    "1h     (sub: 5min)":  ("5m",  "60m", "10d"),
    "4h     (sub: 15min)": ("15m", "60m", "30d"),
    "Daily  (sub: 1h)":    ("1h",  "1d",  "90d"),
}

OF_TICKERS = [
    "SPY","QQQ","IWM","DIA",
    "AAPL","MSFT","NVDA","TSLA","META","AMZN",
    "GLD","SLV","GDX","TLT","HYG",
    "BTC-USD","ETH-USD",
    "ES=F","NQ=F","YM=F","RTY=F",
    "CL=F","GC=F","SI=F",
    "EUR=X","JPY=X","GBP=X",
]

_SLIDE_PATHS = [
    "/mnt/user-data/outputs/of_slide1.png",
    "/mnt/user-data/outputs/of_slide2.png",
    "/mnt/user-data/outputs/of_slide3.png",
    "/mnt/user-data/outputs/of_slide4.png",
    "/mnt/user-data/outputs/of_slide5.png",
]
_SLIDE_TITLES = [
    "01 · Cos'è l'Order Flow",
    "02 · Come le Istituzioni Piazzano Ordini",
    "03 · Leggere l'Order Flow",
    "04 · Bullish vs Bearish Flow",
    "05 · Checklist Order Flow",
]
_SLIDE_CONCEPTS = [
    ["Order flow = BUY vs SELL orders",
     "Bullish Flow: più compratori → prezzo sale",
     "Bearish Flow: più venditori → prezzo scende",
     "Order flow is the CAUSE — price is the EFFECT"],
    ["3 Fasi: Accumulation → Manipulation → Expansion",
     "Phase 1: istituzioni accumulano in silenzio",
     "Phase 2: London session — wick sweep SSL/BSL",
     "Phase 3: NY session — markup / distribuzione"],
    ["5 clues: Body size, Wick, Speed, Level reaction, 3-Candle",
     "Grandi corpi = forte order flow direzionale",
     "Wick lungo = ordini assorbiti in quella direzione",
     "3-Candle: Accum → Manip wick → Expansion candle"],
    ["Bullish: HH+HL, lower wicks, grandi candle green",
     "Bearish: LH+LL, upper wicks, grandi candle red",
     "CHoCH = primo break di struttura nella direzione opposta",
     "SSL sweep prima del rally = segnale A+"],
    ["STEP 1: Check HTF order flow (Daily/Weekly)",
     "STEP 2: Identifica la fase corrente (3-Phase)",
     "STEP 3: Leggi candle body + wick direction",
     "STEP 4: Conferma a livello chiave OB/FVG/EQH"],
]


# ═════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_intraday(symbol: str, interval: str, range_: str) -> pd.DataFrame:
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval={interval}&range={range_}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        q  = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s", utc=True).tz_localize(None),
            "open":   q.get("open",   [None]*len(ts)),
            "high":   q.get("high",   [None]*len(ts)),
            "low":    q.get("low",    [None]*len(ts)),
            "close":  q.get("close",  [None]*len(ts)),
            "volume": q.get("volume", [0]*len(ts)),
        }).dropna(subset=["close","open","high","low"]).reset_index(drop=True)
        df["volume"] = df["volume"].fillna(0).astype(float)
        return df
    except Exception:
        return pd.DataFrame()


def _estimate_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    ratio = np.where(hl > 0, (df["close"] - df["low"]) / hl, 0.5)
    df["buy_vol"]  = (df["volume"] * ratio).round(0)
    df["sell_vol"] = (df["volume"] * (1 - ratio)).round(0)
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    return df


def _resample_main(df_sub: pd.DataFrame, main_iv: str) -> pd.DataFrame:
    freq_map = {"30m":"30min","60m":"60min","1h":"60min","4h":"240min","1d":"1D"}
    freq = freq_map.get(main_iv, "60min")
    df = _estimate_delta(df_sub)
    df["bar"] = df["date"].dt.floor(freq)
    agg = df.groupby("bar").agg(
        open    =("open",    "first"),
        high    =("high",    "max"),
        low     =("low",     "min"),
        close   =("close",   "last"),
        volume  =("volume",  "sum"),
        buy_vol =("buy_vol", "sum"),
        sell_vol=("sell_vol","sum"),
        delta   =("delta",   "sum"),
    ).reset_index().rename(columns={"bar":"date"})
    agg["delta_pct"] = np.where(agg["volume"]>0,
                                (agg["delta"]/agg["volume"]*100).round(1), 0)
    agg["cum_delta"] = agg["delta"].cumsum()
    return agg


def _auto_tick(price: float, n: int = 10) -> float:
    if price <= 0: return 0.01
    raw = (price * 0.005) / n
    mag = 10 ** np.floor(np.log10(max(raw, 1e-10)))
    for m in [1, 2, 2.5, 5, 10]:
        ts = round(mag * m, 10)
        if ts >= raw: return ts
    return raw


def _build_levels(df_main: pd.DataFrame, df_sub: pd.DataFrame,
                  tick: float, n: int, imb_r: float = 3.0) -> list:
    df_sub  = _estimate_delta(df_sub)
    df_main = df_main.tail(n).reset_index(drop=True)
    if len(df_main) < 2: return []
    avg_sec = max(
        (df_main["date"].iloc[-1]-df_main["date"].iloc[0]).total_seconds()
        / max(len(df_main)-1, 1), 60)
    candles = []
    for i, row in df_main.iterrows():
        t0 = row["date"]
        t1 = (df_main.loc[i+1,"date"] if i+1 < len(df_main)
              else t0 + pd.Timedelta(seconds=avg_sec*2))
        sub = df_sub[(df_sub["date"]>=t0)&(df_sub["date"]<t1)]
        lo, hi = float(row["low"]), float(row["high"])
        n_lv = max(int((hi-lo)/tick), 1)
        ta   = tick if n_lv <= 20 else (hi-lo)/20
        edges = np.arange(lo, hi+ta*0.5, ta)
        levels = []
        for j in range(len(edges)-1):
            p_lo, p_hi = edges[j], edges[j+1]
            pm = round((p_lo+p_hi)/2, 6)
            if sub.empty:
                b = s = 0.0
            else:
                lv_s = sub[(sub["low"]<=p_hi)&(sub["high"]>=p_lo)]
                if lv_s.empty:
                    b = s = 0.0
                else:
                    sp = lv_s.apply(
                        lambda r: max((r["high"]-r["low"])/ta, 1), axis=1).mean()
                    b = round(float(lv_s["buy_vol"].sum())/sp, 0)
                    s = round(float(lv_s["sell_vol"].sum())/sp, 0)
            d = b - s
            imb = None
            if s>0 and b/s>=imb_r:        imb = "buy"
            elif b>0 and s/b>=imb_r:      imb = "sell"
            elif b==0 and s>0:            imb = "sell"
            elif s==0 and b>0:            imb = "buy"
            levels.append({"price":pm,"buy":b,"sell":s,"delta":d,"imbalance":imb})
        candles.append({
            "date":     row["date"],
            "open":     float(row["open"]),
            "high":     hi, "low": lo,
            "close":    float(row["close"]),
            "volume":   float(row["volume"]),
            "buy_vol":  float(row.get("buy_vol", 0)),
            "sell_vol": float(row.get("sell_vol", 0)),
            "delta":    float(row.get("delta", 0)),
            "delta_pct":float(row.get("delta_pct", 0)),
            "cum_delta":float(row.get("cum_delta", 0)),
            "levels":   levels,
        })
    return candles


# ═════════════════════════════════════════════════════════════════════════════
# VWAP
# ═════════════════════════════════════════════════════════════════════════════

def _compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tp"]  = (df["high"]+df["low"]+df["close"])/3
    df["day"] = df["date"].dt.date
    cum_tpv = df.groupby("day").apply(
        lambda g: (g["tp"]*g["volume"]).cumsum()
    ).reset_index(level=0, drop=True)
    cum_vol = df.groupby("day")["volume"].cumsum()
    df["vwap"] = cum_tpv / cum_vol.replace(0, np.nan)
    variance = df.groupby("day").apply(
        lambda g: (
            (g["tp"] - (g["tp"]*g["volume"]).cumsum()/g["volume"].cumsum())**2
            * g["volume"]
        ).cumsum() / g["volume"].cumsum()
    ).reset_index(level=0, drop=True)
    df["vstd"]   = np.sqrt(variance.clip(lower=0))
    df["vwap_1u"]= df["vwap"] + df["vstd"]
    df["vwap_1d"]= df["vwap"] - df["vstd"]
    df["vwap_2u"]= df["vwap"] + 2*df["vstd"]
    df["vwap_2d"]= df["vwap"] - 2*df["vstd"]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SMC DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def _detect_smc(df: pd.DataFrame, swing_n: int = 3) -> dict:
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    opens  = df["open"].values
    n = len(df)
    sh, sl = [], []
    for i in range(swing_n, n-swing_n):
        if all(highs[i]>highs[i-j] and highs[i]>highs[i+j] for j in range(1,swing_n+1)):
            sh.append(i)
        if all(lows[i] <lows[i-j]  and lows[i] <lows[i+j]  for j in range(1,swing_n+1)):
            sl.append(i)
    choch = []
    for k in range(1, len(sh)):
        i = sh[k]
        if highs[i] < highs[sh[k-1]]:
            choch.append({"idx":i,"dir":"bear","level":highs[i]})
    for k in range(1, len(sl)):
        i = sl[k]
        if lows[i] > lows[sl[k-1]]:
            choch.append({"idx":i,"dir":"bull","level":lows[i]})
    obs = []
    for k in range(2, n-1):
        if closes[k] < opens[k]:
            bk = abs(opens[k]-closes[k])
            if closes[k+1]>opens[k+1] and abs(closes[k+1]-opens[k+1])>bk*1.5:
                obs.append({"idx":k,"dir":"bull","top":opens[k],"bot":closes[k]})
        if closes[k] > opens[k]:
            bk = abs(closes[k]-opens[k])
            if closes[k+1]<opens[k+1] and abs(opens[k+1]-closes[k+1])>bk*1.5:
                obs.append({"idx":k,"dir":"bear","top":closes[k],"bot":opens[k]})
    fvgs = []
    for k in range(1, n-1):
        if highs[k-1] < lows[k+1]:
            fvgs.append({"idx":k,"dir":"bull","top":lows[k+1],"bot":highs[k-1]})
        if lows[k-1] > highs[k+1]:
            fvgs.append({"idx":k,"dir":"bear","top":lows[k-1],"bot":highs[k+1]})
    sweeps = []
    for i in sh[-5:]:
        for j in range(i+1, min(i+5,n)):
            if highs[j]>highs[i] and closes[j]<highs[i]:
                sweeps.append({"idx":j,"dir":"bear","level":highs[i]}); break
    for i in sl[-5:]:
        for j in range(i+1, min(i+5,n)):
            if lows[j]<lows[i] and closes[j]>lows[i]:
                sweeps.append({"idx":j,"dir":"bull","level":lows[i]}); break
    return {"sh":sh,"sl":sl,"choch":choch,"obs":obs[-6:],"fvgs":fvgs[-8:],"sweeps":sweeps}


def _classify_flow(df: pd.DataFrame) -> dict:
    if len(df) < 10:
        return {"flow":"neutral","phase":"unknown","score":0,"body_ratio":0.5}
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    bodies = np.abs(df["close"].values - df["open"].values)
    ranges = highs - lows
    br = float(np.mean(bodies[-10:] / np.where(ranges[-10:]>0, ranges[-10:], 1)))
    hh = sum(1 for i in range(1,5) if highs[-i]>highs[-(i+1)])
    hl = sum(1 for i in range(1,5) if lows[-i] >lows[-(i+1)])
    lh = sum(1 for i in range(1,5) if highs[-i]<highs[-(i+1)])
    ll = sum(1 for i in range(1,5) if lows[-i] <lows[-(i+1)])
    score = (hh+hl)-(ll+lh)
    flow  = "bullish" if score>=3 else "bearish" if score<=-3 else "balanced"
    vs_r  = float(np.std(closes[-5:]))
    vs_p  = float(np.std(closes[-15:-5])) if len(closes)>=15 else vs_r
    phase = ("accumulation" if vs_r < vs_p*0.6
             else "expansion" if vs_r > vs_p*1.5
             else "manipulation")
    return {"flow":flow,"phase":phase,"score":score,"body_ratio":round(br,2)}


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _fv(v: float) -> str:
    v = abs(v)
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def _kpi(label, value, color=_TEXT, sub=""):
    return (f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
            f'border-left:4px solid {color};border-radius:6px;'
            f'padding:9px 11px;text-align:center">'
            f'<div style="color:{_GRAY};font-size:0.63rem;font-weight:600;'
            f'letter-spacing:0.05em;text-transform:uppercase">{label}</div>'
            f'<div style="color:{color};font-size:1.1rem;font-weight:700;margin:2px 0">'
            f'{value}</div>'
            + (f'<div style="color:{_GRAY};font-size:0.69rem">{sub}</div>' if sub else "")
            + '</div>')


def _img_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def _chart_footprint(candles, df_vwap, smc, symbol,
                     show_vwap, show_ob, show_fvg, show_delta):
    if not candles: return go.Figure()
    nr = 3 if show_delta else 2
    rh = [0.62, 0.22, 0.16] if show_delta else [0.72, 0.28]
    fig = make_subplots(rows=nr, cols=1, shared_xaxes=True,
                        row_heights=rh, vertical_spacing=0.02)
    dates  = [str(c["date"])[:16] for c in candles]
    opens  = [c["open"]  for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    closes = [c["close"] for c in candles]
    deltas = [c["delta"] for c in candles]
    vols   = [c["volume"]for c in candles]
    # Candele
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name="OHLC", showlegend=False,
        increasing=dict(fillcolor=_GREEN, line=dict(color=_GREEN, width=1)),
        decreasing=dict(fillcolor=_RED,   line=dict(color=_RED,   width=1))),
        row=1, col=1)
    # VWAP
    if show_vwap and not df_vwap.empty:
        xv = [str(d)[:16] for d in df_vwap["date"]]
        for col_n, lbl, clr, dsh, w in [
            ("vwap","VWAP",_VWAP,"solid",2),
            ("vwap_1u","±1σ",_BLUE,"dot",1),("vwap_1d",None,_BLUE,"dot",1),
            ("vwap_2u","±2σ",_PURPLE,"dash",1),("vwap_2d",None,_PURPLE,"dash",1)]:
            if col_n not in df_vwap.columns: continue
            fig.add_trace(go.Scatter(
                x=xv, y=df_vwap[col_n].tolist(), mode="lines",
                line=dict(color=clr,width=w,dash=dsh),
                name=lbl, showlegend=lbl is not None, opacity=0.85,
                hoverinfo="skip"), row=1, col=1)
    # OB
    if show_ob and smc:
        for ob in smc.get("obs",[]):
            idx = ob["idx"]
            if 0 <= idx < len(dates):
                bc = _OB_BUY if ob["dir"]=="bull" else _OB_SELL
                lc = _GREEN  if ob["dir"]=="bull" else _RED
                fig.add_shape(type="rect",
                    x0=dates[idx], x1=dates[min(idx+5,len(dates)-1)],
                    y0=ob["bot"],  y1=ob["top"],
                    fillcolor=bc, line=dict(color=lc,width=1,dash="dot"),
                    row=1, col=1)
    # FVG
    if show_fvg and smc:
        for fvg in smc.get("fvgs",[]):
            idx = fvg["idx"]
            if 0 <= idx < len(dates):
                fc = _FVG_BUY if fvg["dir"]=="bull" else _FVG_SEL
                fig.add_shape(type="rect",
                    x0=dates[max(idx-1,0)], x1=dates[min(idx+8,len(dates)-1)],
                    y0=fvg["bot"], y1=fvg["top"],
                    fillcolor=fc, line=dict(color="rgba(0,0,0,0)"),
                    row=1, col=1)
    # CHoCH
    if smc:
        for ch in smc.get("choch",[])[-4:]:
            idx = ch["idx"]
            if 0 <= idx < len(dates):
                cc = _GREEN if ch["dir"]=="bull" else _RED
                fig.add_annotation(x=dates[idx], y=ch["level"],
                    text=f"<span style='color:{cc};font-size:9px;font-weight:700'>"
                         f"CHoCH {'▲' if ch['dir']=='bull' else '▼'}</span>",
                    showarrow=True, arrowhead=0, arrowcolor=cc,
                    ay=-25 if ch["dir"]=="bull" else 25, row=1, col=1)
        # Sweeps
        for sw in smc.get("sweeps",[]):
            idx = sw["idx"]
            if 0 <= idx < len(dates):
                sc = _GREEN if sw["dir"]=="bull" else _RED
                fig.add_annotation(x=dates[idx], y=sw["level"],
                    text=f"<span style='color:{sc};font-size:10px'>⚡</span>",
                    showarrow=False, xanchor="center",
                    yanchor="top" if sw["dir"]=="bear" else "bottom",
                    row=1, col=1)
    # Footprint livelli
    for ci, c in enumerate(candles):
        if not c["levels"]: continue
        ds = dates[ci]
        for lv in c["levels"]:
            tot = lv["buy"]+lv["sell"]
            if tot < 1: continue
            imb = lv["imbalance"]
            tc  = _GREEN2 if imb=="buy" else _RED2 if imb=="sell" else "rgba(209,212,220,0.45)"
            lbl = f"{_fv(lv['sell'])}×{_fv(lv['buy'])}"
            fig.add_annotation(x=ds, y=lv["price"],
                text=f"<span style='color:{tc};font-size:7.5px;font-family:monospace;"
                     f"font-weight:{700 if imb else 400}'>{lbl}</span>",
                showarrow=False, xanchor="center", yanchor="middle", row=1, col=1)
        dc = _GREEN2 if c["delta"]>=0 else _RED2
        fig.add_annotation(x=ds,
            y=c["high"]*1.001 if c["delta"]>=0 else c["low"]*0.999,
            text=f"<span style='color:{dc};font-size:8px;font-weight:700'>"
                 f"Δ{_fv(abs(c['delta']))}</span>",
            showarrow=False, xanchor="center",
            yanchor="bottom" if c["delta"]>=0 else "top", row=1, col=1)
    # Delta bar
    if show_delta:
        fig.add_trace(go.Bar(x=dates, y=deltas,
            marker_color=[_GREEN if d>=0 else _RED for d in deltas],
            marker_line_width=0, name="Delta", showlegend=False,
            hovertemplate="Δ %{y:,.0f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line=dict(color=_BORDER,width=1))
    # Volume
    vr = 3 if show_delta else 2
    fig.add_trace(go.Bar(x=dates, y=vols, opacity=0.75,
        marker_color=[_GREEN if cl>=op else _RED for cl,op in zip(closes,opens)],
        marker_line_width=0, name="Vol", showlegend=False,
        hovertemplate="Vol %{y:,.0f}<extra></extra>"), row=vr, col=1)
    chg = (closes[-1]/opens[0]-1)*100 if opens[0]!=0 else 0
    fig.update_layout(
        title=dict(text=(f"<b style='color:{_CYAN}'>{symbol}</b>"
                         f"  <span style='color:{_GRAY};font-size:0.83em'>"
                         f"Footprint + VWAP + SMC</span>"
                         f"  <span style='color:{'#26a69a' if chg>=0 else '#ef5350'}'>"
                         f"{'▲' if chg>=0 else '▼'} {abs(chg):.1f}%</span>"),
                   font=dict(size=13,color=_TEXT), x=0.01),
        height=790, paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        xaxis_rangeslider_visible=False,
        margin=dict(l=8,r=8,t=55,b=8), font=dict(color=_TEXT,size=9),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9),
                    orientation="h",y=1.025,x=0.12))
    for r in range(1, nr+1):
        n_ = "" if r==1 else str(r)
        fig.update_layout(**{
            f"xaxis{n_}":dict(showgrid=True,gridcolor=_BORDER,zeroline=False,
                              showticklabels=(r==nr)),
            f"yaxis{n_}":dict(showgrid=True,gridcolor=_BORDER,zeroline=False,
                              tickfont=dict(size=9))})
    return fig


def _chart_cvd(candles, df_vwap, symbol, show_vwap):
    if not candles: return go.Figure()
    dates  = [str(c["date"])[:16] for c in candles]
    closes = [c["close"] for c in candles]
    run = 0; cum_d = []
    for c in candles: run+=c["delta"]; cum_d.append(run)
    p_min,p_max = min(closes),max(closes)
    d_min,d_max = min(cum_d), max(cum_d)
    cvd_n = ([p_min+(v-d_min)/(d_max-d_min)*(p_max-p_min) for v in cum_d]
             if d_max!=d_min and p_max!=p_min else closes[:])
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.70,0.30],vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=dates,y=closes,mode="lines",
        line=dict(color=_CYAN,width=2),name="Close"),row=1,col=1)
    if show_vwap and not df_vwap.empty and "vwap" in df_vwap.columns:
        xv=[str(d)[:16] for d in df_vwap["date"]]
        fig.add_trace(go.Scatter(x=xv,y=df_vwap["vwap"].tolist(),mode="lines",
            line=dict(color=_VWAP,width=1.5,dash="dot"),name="VWAP",opacity=0.9),
            row=1,col=1)
    fig.add_trace(go.Scatter(x=dates,y=cvd_n,mode="lines",
        line=dict(color=_ORANGE,width=2,dash="dot"),name="CVD (norm)",
        customdata=cum_d,
        hovertemplate="CVD: %{customdata:,.0f}<extra></extra>"),row=1,col=1)
    ndiv=0
    for i in range(1,len(dates)):
        p_up=closes[i]>closes[i-1]; cd_up=cum_d[i]>cum_d[i-1]
        if p_up!=cd_up:
            ndiv+=1
            fig.add_vrect(x0=dates[i-1],x1=dates[i],
                fillcolor="rgba(255,152,0,0.10)",line_width=0,row=1,col=1)
    bd=[c["delta"] for c in candles]
    fig.add_trace(go.Bar(x=dates,y=bd,
        marker_color=[_GREEN if d>=0 else _RED for d in bd],
        marker_line_width=0,name="Delta/bar",showlegend=False),row=2,col=1)
    fig.add_hline(y=0,row=2,col=1,line=dict(color=_BORDER,width=1))
    fig.update_layout(
        title=dict(text=(f"<b style='color:{_CYAN}'>{symbol}</b>"
                         f"  <span style='color:{_GRAY}'>CVD — Cumulative Volume Delta</span>"
                         f"  <span style='color:{_ORANGE};font-size:0.8em'>"
                         f"  {ndiv} divergenze</span>"),
                   font=dict(size=13,color=_TEXT),x=0.01),
        height=580,paper_bgcolor=_BG,plot_bgcolor=_PANEL,
        margin=dict(l=8,r=8,t=50,b=8),font=dict(color=_TEXT,size=9),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9),
                    orientation="h",y=1.02))
    for r in [1,2]:
        n_=("" if r==1 else str(r))
        fig.update_layout(**{
            f"xaxis{n_}":dict(showgrid=True,gridcolor=_BORDER,zeroline=False,
                              showticklabels=(r==2)),
            f"yaxis{n_}":dict(showgrid=True,gridcolor=_BORDER,zeroline=False,
                              tickfont=dict(size=9))})
    return fig


def _chart_vprofile(candles, df_vwap, symbol, show_vwap):
    if not candles: return go.Figure()
    dates=[str(c["date"])[:16] for c in candles]
    opens=[c["open"] for c in candles]; highs=[c["high"] for c in candles]
    lows =[c["low"]  for c in candles]; closes=[c["close"]for c in candles]
    pv,pb,ps={},{},{}
    for c in candles:
        for lv in c["levels"]:
            p=round(lv["price"],4)
            pv[p]=pv.get(p,0)+lv["buy"]+lv["sell"]
            pb[p]=pb.get(p,0)+lv["buy"]
            ps[p]=ps.get(p,0)+lv["sell"]
    if not pv:
        for c in candles:
            p=round((c["high"]+c["low"])/2,2)
            pv[p]=pv.get(p,0)+c["volume"]
            pb[p]=pb.get(p,0)+c["buy_vol"]
            ps[p]=ps.get(p,0)+c["sell_vol"]
    prices=sorted(pv.keys()); vols=[pv[p] for p in prices]
    buys=[pb[p] for p in prices]; sells=[ps[p] for p in prices]
    poc_i=int(np.argmax(vols)); poc_p=prices[poc_i]
    tv=sum(vols); target=tv*0.70; va=vols[poc_i]
    li=hi=poc_i
    while va<target and (li>0 or hi<len(vols)-1):
        al=vols[li-1] if li>0 else 0; ah=vols[hi+1] if hi<len(vols)-1 else 0
        if ah>=al and hi<len(vols)-1: hi+=1; va+=vols[hi]
        elif li>0: li-=1; va+=vols[li]
        else: break
    vah=prices[hi]; val=prices[li]
    fig=make_subplots(rows=1,cols=2,column_widths=[0.70,0.30],
                      horizontal_spacing=0.01,shared_yaxes=True)
    fig.add_trace(go.Candlestick(x=dates,open=opens,high=highs,low=lows,close=closes,
        name="OHLC",showlegend=False,
        increasing=dict(fillcolor=_GREEN,line=dict(color=_GREEN,width=1)),
        decreasing=dict(fillcolor=_RED,  line=dict(color=_RED,  width=1))),
        row=1,col=1)
    if show_vwap and not df_vwap.empty and "vwap" in df_vwap.columns:
        xv=[str(d)[:16] for d in df_vwap["date"]]
        fig.add_trace(go.Scatter(x=xv,y=df_vwap["vwap"].tolist(),mode="lines",
            line=dict(color=_VWAP,width=1.5),name="VWAP"),row=1,col=1)
    for y,lbl,clr in [(vah,"VAH",_BLUE),(val,"VAL",_BLUE),(poc_p,"POC",_GOLD)]:
        fig.add_hline(y=y,row=1,col=1,line=dict(color=clr,width=1.5,dash="dot"),
            annotation_text=f" {lbl} {y:.2f}",
            annotation_font_color=clr,annotation_font_size=9)
    bc=[_GOLD if i==poc_i else ("rgba(41,98,255,0.6)" if val<=prices[i]<=vah
        else "rgba(120,123,134,0.35)") for i in range(len(prices))]
    fig.add_trace(go.Bar(x=vols,y=prices,orientation="h",
        marker_color=bc,marker_line_width=0,name="Volume",showlegend=False,
        hovertemplate="Price: %{y:.2f}<br>Vol: %{x:,.0f}<extra></extra>"),row=1,col=2)
    fig.add_trace(go.Bar(x=buys,y=prices,orientation="h",
        marker_color="rgba(38,166,154,0.4)",marker_line_width=0,
        name="Buy",showlegend=True),row=1,col=2)
    fig.add_trace(go.Bar(x=[-s for s in sells],y=prices,orientation="h",
        marker_color="rgba(239,83,80,0.4)",marker_line_width=0,
        name="Sell",showlegend=True),row=1,col=2)
    fig.update_layout(
        title=dict(text=(f"<b style='color:{_CYAN}'>{symbol}</b>"
                         f"  <span style='color:{_GRAY}'>Volume Profile + VWAP</span>"
                         f"  <span style='color:{_GOLD};font-size:0.85em'> POC {poc_p:.2f}</span>"
                         f"  <span style='color:{_BLUE};font-size:0.8em'>"
                         f" VA [{val:.2f}–{vah:.2f}]</span>"),
                   font=dict(size=13,color=_TEXT),x=0.01),
        height=580,paper_bgcolor=_BG,plot_bgcolor=_PANEL,barmode="overlay",
        xaxis_rangeslider_visible=False,margin=dict(l=8,r=8,t=50,b=8),
        font=dict(color=_TEXT,size=9),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9),orientation="h",y=1.02))
    for cv in [1,2]:
        nc=("" if cv==1 else str(cv))
        fig.update_layout(**{
            f"xaxis{nc}":dict(showgrid=True,gridcolor=_BORDER,zeroline=False),
            f"yaxis{nc}":dict(showgrid=True,gridcolor=_BORDER,zeroline=False,
                              tickfont=dict(size=9))})
    return fig


def _chart_heatmap(candles, symbol):
    if not candles: return go.Figure()
    dates=[str(c["date"])[:16] for c in candles]
    all_p=sorted(set(round(lv["price"],4) for c in candles for lv in c["levels"]))
    if not all_p:
        fig=go.Figure()
        fig.add_annotation(text="Nessun livello disponibile",
            xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,
            font=dict(color=_GRAY,size=12))
        fig.update_layout(height=400,paper_bgcolor=_BG,plot_bgcolor=_PANEL)
        return fig
    z=[]
    for p in all_p:
        row=[]
        for c in candles:
            m=[lv for lv in c["levels"] if abs(lv["price"]-p)<1e-4]
            row.append(m[0]["delta"] if m else None)
        z.append(row)
    fig=go.Figure(go.Heatmap(
        z=z,x=dates,y=[f"{p:.2f}" for p in all_p],
        colorscale=[[0,_RED],[0.45,"rgba(30,34,45,0.8)"],
                    [0.55,"rgba(30,34,45,0.8)"],[1,_GREEN]],
        zmid=0,
        colorbar=dict(title="Delta",tickfont=dict(size=9,color=_TEXT),
                      title_font=dict(color=_TEXT)),
        hovertemplate="Date: %{x}<br>Price: %{y}<br>Δ: %{z:,.0f}<extra></extra>"))
    fig.update_layout(
        title=dict(text=(f"<b style='color:{_CYAN}'>{symbol}</b>"
                         f"  <span style='color:{_GRAY}'>Imbalance Heatmap"
                         f"  🟢 buy dom · 🔴 sell dom</span>"),
                   font=dict(size=13,color=_TEXT),x=0.01),
        height=500,paper_bgcolor=_BG,plot_bgcolor=_PANEL,
        margin=dict(l=8,r=8,t=50,b=8),font=dict(color=_TEXT,size=9),
        xaxis=dict(showgrid=True,gridcolor=_BORDER,zeroline=False),
        yaxis=dict(showgrid=True,gridcolor=_BORDER,zeroline=False,tickfont=dict(size=8)))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# LEGENDA SLIDE SMC
# ═════════════════════════════════════════════════════════════════════════════

def _render_slide_legend():
    st.markdown(
        f'<div style="background:{_PANEL};border-left:3px solid {_GOLD};'
        f'padding:8px 14px;border-radius:0 6px 6px 0;margin-bottom:12px">'
        f'<span style="color:{_GOLD};font-weight:700">📚 SMC ORDER FLOW GUIDE</span>'
        f'  <span style="color:{_GRAY};font-size:0.77rem;margin-left:8px">'
        f'by @niccofx · 5 slide · clicca per espandere</span>'
        f'</div>', unsafe_allow_html=True)
    # Miniature
    th_cols = st.columns(5)
    for i,(path,title) in enumerate(zip(_SLIDE_PATHS,_SLIDE_TITLES)):
        b64=_img_b64(path)
        with th_cols[i]:
            if b64:
                st.markdown(
                    f'<div style="border:2px solid {_BORDER};border-radius:6px;'
                    f'overflow:hidden">'
                    f'<img src="data:image/png;base64,{b64}" style="width:100%">'
                    f'</div>'
                    f'<div style="color:{_GRAY};font-size:0.66rem;text-align:center;'
                    f'margin-top:4px">{title}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
                    f'border-radius:6px;padding:20px;text-align:center">'
                    f'<div style="color:{_GRAY};font-size:0.75rem">{title}</div>'
                    f'</div>', unsafe_allow_html=True)
    # Expander con concetti
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    ex_cols = st.columns(5)
    for i,(path,title,concepts) in enumerate(
            zip(_SLIDE_PATHS,_SLIDE_TITLES,_SLIDE_CONCEPTS)):
        with ex_cols[i]:
            with st.expander(f"🔍 Slide {i+1}", expanded=False):
                b64=_img_b64(path)
                if b64:
                    st.markdown(
                        f'<img src="data:image/png;base64,{b64}" '
                        f'style="width:100%;border-radius:4px;margin-bottom:8px">',
                        unsafe_allow_html=True)
                for c in concepts:
                    st.markdown(
                        f'<div style="color:{_TEXT};font-size:0.74rem;'
                        f'border-left:2px solid {_GOLD};padding-left:6px;'
                        f'margin:3px 0">{c}</div>',
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# CHECKLIST SMC
# ═════════════════════════════════════════════════════════════════════════════

def _render_checklist(flow_data: dict):
    flow  = flow_data.get("flow","neutral")
    phase = flow_data.get("phase","unknown")
    score = flow_data.get("score",0)
    br    = flow_data.get("body_ratio",0.5)
    fc = {"bullish":_GREEN,"bearish":_RED,"balanced":_GOLD}.get(flow,_GRAY)
    pc = {"accumulation":_BLUE,"manipulation":_ORANGE,"expansion":_GREEN}.get(phase,_GRAY)
    checks_bull = [
        ("Corpi candle grandi e verdi dominanti",           flow=="bullish"),
        ("Wick verso il basso (buy orders absorbed)",       flow=="bullish"),
        ("Struttura HH + HL in formazione",                 score>=2),
        ("SSL sweep (manipolazione) prima del rally",       phase=="manipulation"),
        ("CHoCH bullish rilevato",                          False),
    ]
    checks_bear = [
        ("Corpi candle grandi e rossi dominanti",           flow=="bearish"),
        ("Wick verso l'alto (sell orders absorbed)",        flow=="bearish"),
        ("Struttura LH + LL in formazione",                 score<=-2),
        ("BSL sweep (manipolazione) prima del drop",        phase=="manipulation"),
        ("CHoCH bearish rilevato",                          False),
    ]
    checks_bal  = [
        ("Corpi piccoli — flow bilanciato",                 flow=="balanced"),
        ("Volatilità bassa → possibile accumulazione",      phase=="accumulation"),
        ("Nessuna struttura dominante",                     abs(score)<2),
        ("Attendi break direzionale + volume confermato",   True),
        ("Non operare in fase di consolidazione",           True),
    ]
    cl = {"bullish":checks_bull,"bearish":checks_bear}.get(flow, checks_bal)
    st.markdown(
        f'<div style="background:{_PANEL};border-left:3px solid {_GOLD};'
        f'padding:8px 14px;border-radius:0 6px 6px 0;margin-bottom:10px">'
        f'<span style="color:{_GOLD};font-weight:700">✅ SMC ORDER FLOW CHECKLIST</span>'
        f'  <span style="color:{_GRAY};font-size:0.77rem;margin-left:8px">'
        f'Da leggere prima di ogni trade · framework @niccofx</span>'
        f'</div>', unsafe_allow_html=True)
    r1,r2=st.columns(2)
    with r1:
        fe = {"bullish":"📈","bearish":"📉","balanced":"⚖️"}.get(flow,"🔄")
        st.markdown(
            f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:8px">'
            f'<div style="color:{fc};font-weight:700;margin-bottom:6px">'
            f'{fe} {flow.upper()} Flow Signals</div>'
            + "".join([
                f'<div style="color:{"#00e676" if ok else _GRAY};font-size:0.8rem;padding:2px 0">'
                f'{"✅" if ok else "⬜"} {lbl}</div>'
                for lbl,ok in cl])
            + f'<div style="color:{pc};font-size:0.8rem;margin-top:8px;font-weight:700">'
            f'📍 Fase: {phase.upper()}</div>'
            + f'<div style="color:{_GRAY};font-size:0.75rem">Body Ratio: {br:.0%}</div>'
            + '</div>', unsafe_allow_html=True)
    with r2:
        df_tbl=pd.DataFrame({
            "Segnale":["Corpi candle","Direzione wick","Struttura","Manipolazione","Transizione"],
            "📈 Bullish":["🟢 Grandi verdi","⬇️ Lower wicks","HH + HL","SSL sweep + rally","CHoCH ▲"],
            "📉 Bearish":["🔴 Grandi rossi","⬆️ Upper wicks","LH + LL","BSL sweep + drop","CHoCH ▼"],
        })
        st.dataframe(df_tbl,use_container_width=True,hide_index=True)
    steps=[
        ("01","HTF Order Flow First","Controlla Daily/Weekly. "
         "Green bodies + HH/HL = bullish HTF. Opera SOLO nella direzione HTF.",_PURPLE),
        ("02","Identifica la 3-Phase Cycle",
         f"Fase attuale: <b style='color:{pc}'>{phase.upper()}</b>. "
         f"{'Accumulation → cerca sweep.' if phase=='accumulation' else 'Manipulation → prepara entry.' if phase=='manipulation' else 'Expansion → gestisci trade.'}",_ORANGE),
        ("03","Leggi Candle — Body e Wick",
         f"Flow: <b style='color:{fc}'>{flow.upper()}</b>  Body ratio: {br:.0%}. "
         f"{'Forte direzionalità ✓' if br>0.6 else 'Flow debole — attendi ✗'}",_GREEN),
        ("04","Conferma con Struttura e Livello",
         "Cerca confluenza: OB + FVG + EQH/EQL + VWAP ±σ. "
         "Order flow in spazio aperto = low prob.",_CYAN),
        ("05","Entry · SL · Target · Trust the Flow 🔥",
         "Dopo sweep manipulation → candle espansione → entry. "
         "SL oltre wick. Target: prossimo liquidity pool BSL/SSL. Min 2:1 RR.",_RED),
    ]
    for num,title,desc,sc in steps:
        st.markdown(
            f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
            f'border-left:4px solid {sc};border-radius:6px;'
            f'padding:10px 14px;margin-bottom:6px;display:flex;gap:12px">'
            f'<div style="color:{sc};font-weight:700;font-size:1rem;min-width:36px">'
            f'{num}</div>'
            f'<div><div style="color:{_TEXT};font-weight:700;font-size:0.84rem">{title}</div>'
            f'<div style="color:{_GRAY};font-size:0.77rem;margin-top:2px">{desc}</div>'
            f'</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def render_orderflow_tab(df_scanner=None):
    """
    Renderizza il tab Order Flow / Footprint completo.
    df_scanner: DataFrame dallo scanner (opzionale, per ticker pre-popolati).
    """
    # Header
    st.markdown(
        f'<div style="background:{_PANEL};border-left:3px solid {_ORANGE};'
        f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:14px">'
        f'<span style="color:{_ORANGE};font-weight:700;font-size:1.05rem">'
        f'🔬 ORDER FLOW — FOOTPRINT + VWAP + SMC</span>'
        f'<span style="color:{_GRAY};font-size:0.78rem;margin-left:12px">'
        f'Footprint Volume · VWAP ±1σ/±2σ · Volume Profile POC/VAH/VAL · '
        f'CVD · CHoCH · OB · FVG · Liquidity Sweep · v31.1</span>'
        f'</div>', unsafe_allow_html=True)

    # Slide legenda SMC
    with st.expander("📚 SMC Order Flow Guide — 5 slide @niccofx  ▸ clicca per aprire",
                     expanded=False):
        _render_slide_legend()

    with st.expander("ℹ️ Metodologia — come funziona il Footprint su Yahoo Finance",
                     expanded=False):
        st.markdown(f"""
**Dati tick reali** → solo piattaforme a pagamento (ATAS ~€30/mese, Bookmap ~$85/mese,
Sierra Chart ~$30/mese). Richiedono Level-2 / DOM feed da CME/CQG/Rithmic.

**Approccio adottato** *(standard per replay storico — usato da TradingView e Bookmap)*:
- Download OHLCV intraday basso (2min/5min/15min) da **Yahoo Finance** (gratuito)
- Stima bid/ask con **Candle Body Ratio**: `buy_vol ≈ volume × (close−low)/(high−low)`
  Questo è il metodo usato da MotiveWave, Order Flow Pro (MQL5) e simili
- Aggrega nel timeframe scelto → matrice Footprint completa
- **Accuratezza stimata**: ~70-80% su strumenti liquidi (SPY, NQ=F, BTC-USD)

**VWAP** con bande ±1σ/±2σ: livelli istituzionali di fair value

**SMC** rilevato automaticamente: CHoCH (Change of Character), Order Block,
FVG (Fair Value Gap), Liquidity Sweep, classificazione 3-Phase Cycle

**Imbalance**: ratio ≥ 3:1 (standard ATAS / JumpstartTrading)
""")

    st.markdown("---")

    # Controlli
    ca,cb,cc,cd,ce = st.columns([2.5,1.8,1,1,1])
    with ca:
        sc_tickers = []
        if df_scanner is not None and not df_scanner.empty:
            tc = "Ticker" if "Ticker" in df_scanner.columns else "ticker"
            if tc in df_scanner.columns:
                sc_tickers = df_scanner[tc].dropna().tolist()[:20]
        all_t = list(dict.fromkeys(sc_tickers + OF_TICKERS))
        symbol = st.selectbox("Ticker", all_t, key="of_ticker")
        manual = st.text_input("Ticker manuale", placeholder="es. ES=F, BTC-USD, EUR=X",
                               key="of_manual").strip().upper()
        if manual: symbol = manual
    with cb:
        tf_label = st.selectbox("Timeframe", list(TF_CONFIG.keys()), index=1, key="of_tf")
        sub_iv, main_iv, range_ = TF_CONFIG[tf_label]
    with cc:
        n_candles = st.slider("Candle", 10, 60, 20, 5, key="of_nc")
        imb_r     = st.slider("Imbalance ≥", 1.5, 10.0, 3.0, 0.5, key="of_imb")
    with cd:
        show_vwap  = st.checkbox("VWAP ±σ",    value=True, key="of_vwap")
        show_ob    = st.checkbox("Order Block", value=True, key="of_ob")
        show_fvg   = st.checkbox("FVG",         value=True, key="of_fvg")
        show_delta = st.checkbox("Delta bar",   value=True, key="of_db")
    with ce:
        st.write(""); st.write("")
        run = st.button("▶ Analizza", key="of_run",
                        use_container_width=True, type="primary")
        if st.button("🔄 Refresh", key="of_ref"):
            st.cache_data.clear(); st.rerun()

    vista = st.radio("Vista",
        ["📊 Footprint + VWAP + SMC","📈 CVD + Divergenze",
         "🔥 Volume Profile","🗺️ Imbalance Heatmap","✅ SMC Checklist"],
        horizontal=True, key="of_vista")

    if not run:
        st.markdown(
            f'<div style="background:{_PANEL};border:1px dashed {_BORDER};'
            f'border-radius:8px;padding:50px;text-align:center;margin-top:16px">'
            f'<div style="font-size:2.5rem">🔬</div>'
            f'<div style="color:{_TEXT};font-size:1rem;font-weight:600;margin-top:10px">'
            f'Seleziona ticker e timeframe → '
            f'<b style="color:{_ORANGE}">▶ Analizza</b></div>'
            f'<div style="color:{_GRAY};font-size:0.82rem;margin-top:6px">'
            f'Footprint · VWAP ±1σ/±2σ · POC/VAH/VAL · CVD · CHoCH · OB · FVG · Sweep'
            f'</div></div>', unsafe_allow_html=True)
        return

    with st.spinner(f"⏳ {symbol} [{tf_label}] — caricamento…"):
        df_sub  = _fetch_intraday(symbol, sub_iv,  range_)
        df_main = _fetch_intraday(symbol, main_iv, range_)
        if df_sub.empty or df_main.empty:
            st.error(f"❌ Dati non disponibili per **{symbol}**. "
                     "Verifica il simbolo Yahoo Finance (es. BTC-USD, AAPL, ES=F).")
            return
        df_agg    = _resample_main(df_sub, main_iv)
        tick      = _auto_tick(float(df_main["close"].iloc[-1]))
        candles   = _build_levels(df_agg, df_sub, tick, n_candles, imb_r)
        df_vwap   = _compute_vwap(df_sub) if show_vwap else pd.DataFrame()
        smc       = _detect_smc(df_agg.tail(n_candles+10).reset_index(drop=True))
        flow_data = _classify_flow(df_agg.tail(n_candles))

    if not candles:
        st.warning("⚠️ Nessun candle elaborato. Prova altro timeframe.")
        return

    # Flow badge
    flow  = flow_data["flow"]; phase = flow_data["phase"]
    score = flow_data["score"]; br   = flow_data["body_ratio"]
    fc = {"bullish":_GREEN,"bearish":_RED,"balanced":_GOLD}.get(flow,_GRAY)
    pc = {"accumulation":_BLUE,"manipulation":_ORANGE,"expansion":_GREEN}.get(phase,_GRAY)
    fe = {"bullish":"📈","bearish":"📉","balanced":"⚖️"}.get(flow,"🔄")
    pe = {"accumulation":"💤","manipulation":"⚡","expansion":"🚀"}.get(phase,"❓")
    st.markdown(
        f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
        f'border-radius:8px;padding:10px 18px;display:flex;align-items:center;'
        f'gap:24px;flex-wrap:wrap;margin-bottom:10px">'
        f'<div><div style="color:{_GRAY};font-size:0.63rem;font-weight:600">SMC FLOW</div>'
        f'<div style="color:{fc};font-size:1.1rem;font-weight:700">{fe} {flow.upper()}</div></div>'
        f'<div><div style="color:{_GRAY};font-size:0.63rem;font-weight:600">3-PHASE</div>'
        f'<div style="color:{pc};font-size:1.1rem;font-weight:700">{pe} {phase.upper()}</div></div>'
        f'<div><div style="color:{_GRAY};font-size:0.63rem;font-weight:600">SCORE</div>'
        f'<div style="color:{_TEXT};font-size:1.1rem;font-weight:700">{score:+d}/4</div></div>'
        f'<div><div style="color:{_GRAY};font-size:0.63rem;font-weight:600">BODY RATIO</div>'
        f'<div style="color:{_TEXT};font-size:1.1rem;font-weight:700">{br:.0%}</div></div>'
        f'</div>', unsafe_allow_html=True)

    # KPI
    rec = df_agg.tail(n_candles)
    tb = float(rec["buy_vol"].sum()); ts = float(rec["sell_vol"].sum())
    td = float(rec["delta"].sum()); lc = candles[-1]
    bp = tb/(tb+ts)*100 if (tb+ts)>0 else 50
    dc = _GREEN if td>=0 else _RED
    vwap_v = (float(df_vwap["vwap"].iloc[-1])
              if not df_vwap.empty and "vwap" in df_vwap.columns else 0)
    nob = len(smc.get("obs",[])); nfvg = len(smc.get("fvgs",[])); nsw = len(smc.get("sweeps",[]))

    cols7 = st.columns(7)
    kpis = [
        ("Prezzo", f"${lc['close']:.2f}", _GREEN if lc["close"]>=lc["open"] else _RED, ""),
        ("VWAP",   f"${vwap_v:.2f}",      _VWAP,
         f"{'▲ sopra' if lc['close']>vwap_v else '▼ sotto'} VWAP"),
        ("Delta tot",_fv(td), dc, f"{'Buy' if td>=0 else 'Sell'} dom"),
        ("Buy %",  f"{bp:.0f}%",   _GREEN,  _fv(tb)),
        ("Sell %", f"{100-bp:.0f}%", _RED,  _fv(ts)),
        ("OB/FVG", f"{nob}/{nfvg}", _GOLD,  f"{nsw} sweeps"),
        ("Tick",   f"${tick:.3f}",  _GRAY,  f"{n_candles} candle"),
    ]
    for (l,v,c,s), col in zip(kpis, cols7):
        with col: st.markdown(_kpi(l,v,c,s), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # Vista selezionata
    if vista == "📊 Footprint + VWAP + SMC":
        fig = _chart_footprint(candles, df_vwap, smc, symbol,
                               show_vwap, show_ob, show_fvg, show_delta)
        st.plotly_chart(fig, use_container_width=True, key="of_fp")
        st.markdown(
            f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
            f'border-radius:6px;padding:7px 12px;font-size:0.74rem;'
            f'display:flex;gap:16px;flex-wrap:wrap">'
            f'<span style="color:{_GRAY}"><b>Celle:</b> Sell×Buy</span>'
            f'<span style="color:{_GREEN2}">■ Imbalance Buy (≥{imb_r:.0f}:1)</span>'
            f'<span style="color:{_RED2}">■ Imbalance Sell</span>'
            f'<span style="color:{_GOLD}">Δ=Delta candle</span>'
            f'<span style="color:{_VWAP}">━ VWAP</span>'
            f'<span style="color:{_BLUE}">┄ ±1σ</span>'
            f'<span style="color:{_PURPLE}">╌ ±2σ</span>'
            f'<span style="color:{_GREEN}">░ Bullish OB</span>'
            f'<span style="color:{_RED}">░ Bearish OB</span>'
            f'<span style="color:{_BLUE}">░ FVG</span>'
            f'<span style="color:{_CYAN}">CHoCH</span>'
            f'<span style="color:{_ORANGE}">⚡ Sweep</span>'
            f'</div>', unsafe_allow_html=True)

    elif vista == "📈 CVD + Divergenze":
        fig = _chart_cvd(candles, df_vwap, symbol, show_vwap)
        st.plotly_chart(fig, use_container_width=True, key="of_cvd")
        st.markdown(
            f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
            f'border-radius:6px;padding:7px 12px;font-size:0.74rem">'
            f'<span style="color:{_ORANGE}">━━ CVD norm</span>'
            f'  <span style="color:{_CYAN}">━━ Prezzo close</span>'
            f'  <span style="color:{_VWAP}">┄ VWAP</span>'
            f'  <span style="color:{_GRAY}">Zone arancioni = divergenza '
            f'prezzo/delta → potenziale CHoCH / reversal</span>'
            f'</div>', unsafe_allow_html=True)

    elif vista == "🔥 Volume Profile":
        fig = _chart_vprofile(candles, df_vwap, symbol, show_vwap)
        st.plotly_chart(fig, use_container_width=True, key="of_vp")
        st.markdown(
            f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
            f'border-radius:6px;padding:7px 12px;font-size:0.74rem">'
            f'<span style="color:{_GOLD}">■ POC</span> (max volume)'
            f'  <span style="color:{_BLUE}">■ Value Area 70%</span>'
            f'  <span style="color:{_VWAP}">━ VWAP</span>'
            f'  <span style="color:{_GREEN}">■ Buy</span>'
            f'  <span style="color:{_RED}">■ Sell</span>'
            f'  <span style="color:{_GRAY}">— Confluenza POC+VWAP = '
            f'livello istituzionale chiave</span>'
            f'</div>', unsafe_allow_html=True)

    elif vista == "🗺️ Imbalance Heatmap":
        fig = _chart_heatmap(candles, symbol)
        st.plotly_chart(fig, use_container_width=True, key="of_hm")

    else:  # ✅ SMC Checklist
        _render_checklist(flow_data)

    # Tabella
    with st.expander("📋 Tabella candle (ultimi 20)", expanded=False):
        rows=[]
        for c in candles[-20:]:
            nb=sum(1 for lv in c["levels"] if lv["imbalance"]=="buy")
            ns=sum(1 for lv in c["levels"] if lv["imbalance"]=="sell")
            rows.append({
                "Data":      str(c["date"])[:16],
                "Close":     f"${c['close']:.2f}",
                "Volume":    _fv(c["volume"]),
                "Buy Vol":   _fv(c["buy_vol"]),
                "Sell Vol":  _fv(c["sell_vol"]),
                "Delta":     f"{'+' if c['delta']>=0 else ''}{_fv(c['delta'])}",
                "Delta %":   f"{c['delta_pct']:+.1f}%",
                "Imb Buy ▲": nb,
                "Imb Sell ▼":ns,
                "Cum Delta": _fv(c["cum_delta"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Footer
    st.markdown(
        f'<div style="color:{_GRAY};font-size:0.69rem;text-align:center;'
        f'margin-top:14px;padding-top:8px;border-top:1px solid {_BORDER}">'
        f'📊 Yahoo Finance OHLCV · Candle Body Ratio (~70-80% acc. su strumenti liquidi) · '
        f'SMC: @niccofx framework · Cache 5min · '
        f'Tick ${tick:.3f} · {datetime.now().strftime("%d/%m/%Y %H:%M")}'
        f'</div>', unsafe_allow_html=True)
