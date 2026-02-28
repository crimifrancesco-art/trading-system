import io
import time
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

from utils.formatting import add_formatted_cols, add_links, prepare_display_df
from utils.db import (
    init_db, reset_watchlist_db, add_to_watchlist, load_watchlist,
    DB_PATH, save_scan_history, load_scan_history, load_scan_snapshot,
    delete_from_watchlist, move_watchlist_rows, rename_watchlist,
)
from utils.scanner import load_universe, scan_ticker

# =========================================================================
# DARK THEME CSS
# =========================================================================
DARK_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #0a0e1a !important; color: #c9d1d9 !important;
}
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1f2937 !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
h1 { color: #00ff88 !important; font-family: 'Courier New', monospace !important;
     letter-spacing: 2px; text-shadow: 0 0 20px #00ff8855; }
h2, h3 { color: #58a6ff !important; font-family: 'Courier New', monospace !important; }
.stCaption, small { color: #6b7280 !important; }
[data-testid="stTabs"] button {
    background: #0d1117 !important; color: #8b949e !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Courier New', monospace !important; font-size: 0.82rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00ff88 !important; border-bottom: 2px solid #00ff88 !important;
}
[data-testid="stMetric"] {
    background: #0d1117 !important; border: 1px solid #1f2937 !important;
    border-radius: 8px !important; padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #00ff88 !important; font-size: 1.6rem !important;
                                 font-family: 'Courier New', monospace !important; }
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0d1117, #1a2233) !important;
    color: #00ff88 !important; border: 1px solid #00ff8855 !important;
    border-radius: 6px !important; font-family: 'Courier New', monospace !important;
    transition: all 0.2s;
}
[data-testid="stButton"] > button:hover {
    border-color: #00ff88 !important; color: #ffffff !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00401f, #006633) !important;
    border-color: #00ff88 !important; color: #00ff88 !important; font-size: 1rem !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: linear-gradient(135deg, #1a0a0a, #2d1010) !important;
    color: #ef4444 !important; border: 1px solid #ef444455 !important;
}
[data-testid="stDownloadButton"] > button {
    background: #0d1117 !important; color: #58a6ff !important;
    border: 1px solid #1f3a5f !important; border-radius: 6px !important;
}
[data-testid="stExpander"] {
    background: #0d1117 !important; border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #58a6ff !important; }
hr { border-color: #1f2937 !important; }

/* AgGrid */
.ag-root-wrapper { background: #0d1117 !important; border: 1px solid #1f2937 !important; }
.ag-header { background: #0a0e1a !important; border-bottom: 1px solid #1f2937 !important; }
.ag-header-cell-label { color: #58a6ff !important; font-family: 'Courier New', monospace !important;
                         font-size: 0.78rem !important; letter-spacing: 1px; }
.ag-header-cell-resize { background: #374151 !important; }
.ag-row { background: #0d1117 !important; border-bottom: 1px solid #1a2233 !important; }
.ag-row:hover { background: #131d2e !important; }
.ag-row-selected { background: #0d2d1e !important; }
.ag-cell { color: #c9d1d9 !important; font-family: 'Courier New', monospace !important;
           font-size: 0.82rem !important; }
.ag-paging-panel { background: #0a0e1a !important; color: #6b7280 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }

.section-pill {
    display: inline-block; background: linear-gradient(90deg, #003320, #001a10);
    border: 1px solid #00ff8844; border-radius: 20px; padding: 4px 16px;
    font-family: 'Courier New', monospace; font-size: 0.8rem; color: #00ff88;
    letter-spacing: 2px; margin-bottom: 12px;
}

/* Watchlist card style */
.wl-card {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1f2937; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 8px;
    transition: border-color 0.2s;
}
.wl-card:hover { border-color: #374151; }
.wl-card-ticker {
    font-family: 'Courier New', monospace; font-size: 1.05rem;
    font-weight: bold; color: #00ff88; letter-spacing: 1px;
}
.wl-card-name { color: #8b949e; font-size: 0.82rem; margin-top: 2px; }
.wl-card-badge {
    display: inline-block; border-radius: 10px; padding: 2px 8px;
    font-size: 0.72rem; font-weight: bold; margin-right: 4px;
}
.badge-green  { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid #00ff8844; }
.badge-orange { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b44; }
.badge-red    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef444444; }
.badge-blue   { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid #58a6ff44; }
.badge-gray   { background: rgba(107,114,128,0.15);color: #6b7280; border: 1px solid #6b728044; }

/* Legend table */
.legend-table { width:100%; border-collapse:collapse; font-family:'Courier New',monospace; font-size:0.82rem; }
.legend-table th { color:#58a6ff; border-bottom:1px solid #1f2937; padding:6px 10px; text-align:left; }
.legend-table td { color:#c9d1d9; border-bottom:1px solid #1a2233; padding:5px 10px; }
.legend-table tr:hover td { background:#131d2e; }
.legend-col-name  { color:#00ff88; font-weight:bold; }
.legend-col-range { color:#f59e0b; }
</style>
"""

PLOTLY_DARK = dict(
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Courier New"),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
)

# =========================================================================
# INDICATORI TECNICI
# =========================================================================
def _sma(arr, n):
    return pd.Series(arr).rolling(n).mean().tolist()

def _ema_calc(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean().tolist()

def _rsi_calc(arr, n=14):
    s = pd.Series(arr); d = s.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(com=n-1, adjust=False).mean() / dn.ewm(com=n-1, adjust=False).mean()
    return (100 - 100/(1+rs)).tolist()

def _macd_calc(arr, fast=12, slow=26, sig=9):
    s = pd.Series(arr)
    macd = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sign = macd.ewm(span=sig, adjust=False).mean()
    return macd.tolist(), sign.tolist(), (macd - sign).tolist()

def _parabolic_sar(highs, lows, af_start=0.02, af_max=0.2):
    h = list(highs); l = list(lows); n = len(h)
    if n < 2: return [None]*n, [0]*n
    sar = [0.0]*n; bull = [True]*n; ep = h[0]; af = af_start; sar[0] = l[0]
    for i in range(1, n):
        pb = bull[i-1]; ps = sar[i-1]
        if pb:
            ns = min(ps + af*(ep - ps), l[i-1], l[i-2] if i>=2 else l[i-1])
            if l[i] < ns: bull[i]=False; sar[i]=ep; ep=l[i]; af=af_start
            else:
                bull[i]=True; sar[i]=ns
                if h[i]>ep: ep=h[i]; af=min(af+af_start, af_max)
        else:
            ns = max(ps + af*(ep - ps), h[i-1], h[i-2] if i>=2 else h[i-1])
            if h[i] > ns: bull[i]=True; sar[i]=ep; ep=h[i]; af=af_start
            else:
                bull[i]=False; sar[i]=ns
                if l[i]<ep: ep=l[i]; af=min(af+af_start, af_max)
    return sar, [1 if b else -1 for b in bull]

# =========================================================================
# BUILD FULL CHART
# =========================================================================
def build_full_chart(row: pd.Series, indicators: list) -> go.Figure:
    cd = row.get("_chart_data")
    if not cd or not isinstance(cd, dict): return None
    dates  = cd.get("dates", []); opens  = cd.get("open",  [])
    highs  = cd.get("high",  []); lows   = cd.get("low",   [])
    closes = cd.get("close", []); vols   = cd.get("volume",[])
    ema20  = cd.get("ema20", []); ema50  = cd.get("ema50", [])
    bb_up  = cd.get("bb_up", []); bb_dn  = cd.get("bb_dn", [])
    if not dates or not closes: return None

    show_sma  = "SMA 9 & 21 + RSI" in indicators
    show_macd = "MACD" in indicators
    show_sar  = "Parabolic SAR" in indicators

    cur = 2; row_rsi = None; row_macd = None
    if show_sma:  row_rsi  = cur; cur += 1
    if show_macd: row_macd = cur; cur += 1
    row_vol = cur; n_rows = cur

    ht = {2:[0.65,0.15], 3:[0.52,0.18,0.13], 4:[0.44,0.17,0.17,0.13]}
    heights = ht.get(n_rows, [0.38,0.15,0.15,0.17,0.13])[:n_rows]
    s = sum(heights); heights = [h/s for h in heights]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=heights, vertical_spacing=0.025)

    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color="#22c55e", increasing_fillcolor="rgba(34,197,94,0.33)",
        decreasing_line_color="#ef4444", decreasing_fillcolor="rgba(239,68,68,0.33)",
        name="Prezzo", showlegend=False,
    ), row=1, col=1)

    if bb_up and bb_dn:
        fig.add_trace(go.Scatter(x=dates+dates[::-1], y=bb_up+bb_dn[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.06)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="BB"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bb_up,
            line=dict(color="#58a6ff", width=1, dash="dot"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bb_dn,
            line=dict(color="#58a6ff", width=1, dash="dot"), showlegend=False), row=1, col=1)
    if ema20:
        fig.add_trace(go.Scatter(x=dates, y=ema20,
            line=dict(color="#f59e0b", width=1.5), name="EMA20"), row=1, col=1)
    if ema50:
        fig.add_trace(go.Scatter(x=dates, y=ema50,
            line=dict(color="#a78bfa", width=1.5), name="EMA50"), row=1, col=1)

    if show_sma:
        sma9 = _sma(closes, 9); sma21 = _sma(closes, 21)
        fig.add_trace(go.Scatter(x=dates, y=sma9,
            line=dict(color="#c084fc", width=1.5, dash="dash"), name="SMA9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=sma21,
            line=dict(color="#fb923c", width=1.5, dash="dash"), name="SMA21"), row=1, col=1)
        for i in range(1, len(closes)):
            if any(v is None for v in [sma9[i], sma21[i], sma9[i-1], sma21[i-1]]): continue
            if sma9[i-1] <= sma21[i-1] and sma9[i] > sma21[i]:
                fig.add_annotation(x=dates[i], y=lows[i]*0.995, text="▲ ENTRY",
                    font=dict(color="#00ff88", size=10), showarrow=True,
                    arrowhead=2, arrowcolor="#00ff88", ay=30, ax=0, row=1, col=1)
            elif sma9[i-1] >= sma21[i-1] and sma9[i] < sma21[i]:
                fig.add_annotation(x=dates[i], y=highs[i]*1.005, text="▼ EXIT",
                    font=dict(color="#ef4444", size=10), showarrow=True,
                    arrowhead=2, arrowcolor="#ef4444", ay=-30, ax=0, row=1, col=1)

    if show_sar:
        sv, sd = _parabolic_sar(highs, lows)
        fig.add_trace(go.Scatter(x=dates,
            y=[sv[i] if sd[i]==1 else None for i in range(len(sv))],
            mode="markers", marker=dict(color="#00ff88", size=4), name="SAR ↑"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates,
            y=[sv[i] if sd[i]==-1 else None for i in range(len(sv))],
            mode="markers", marker=dict(color="#ef4444", size=4), name="SAR ↓"), row=1, col=1)
        for i in range(1, len(sd)):
            if sd[i]==1 and sd[i-1]==-1:
                fig.add_annotation(x=dates[i], y=lows[i]*0.991,
                    text="◆ SAR BUY", font=dict(color="#00ff88", size=8),
                    showarrow=False, row=1, col=1)
            elif sd[i]==-1 and sd[i-1]==1:
                fig.add_annotation(x=dates[i], y=highs[i]*1.009,
                    text="◆ SAR SELL", font=dict(color="#ef4444", size=8),
                    showarrow=False, row=1, col=1)

    if show_sma and row_rsi:
        rv = _rsi_calc(closes)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)",
                      line_width=0, row=row_rsi, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.08)",
                      line_width=0, row=row_rsi, col=1)
        fig.add_trace(go.Scatter(x=dates, y=rv,
            line=dict(color="#60a5fa", width=1.5), name="RSI"), row=row_rsi, col=1)
        for lvl, col in [(70,"#ef4444"),(50,"#6b7280"),(30,"#00ff88")]:
            fig.add_hline(y=lvl, line=dict(color=col, width=1, dash="dot"),
                          row=row_rsi, col=1)
        for i in range(1, len(rv)):
            if rv[i] is None or rv[i-1] is None: continue
            if rv[i-1]<=30 and rv[i]>30:
                fig.add_annotation(x=dates[i], y=32, text="RSI↑ ENTRY",
                    font=dict(color="#00ff88",size=8), showarrow=True,
                    arrowhead=1, arrowcolor="#00ff88", ay=-20, ax=0, row=row_rsi, col=1)
            if rv[i-1]>=70 and rv[i]<70:
                fig.add_annotation(x=dates[i], y=68, text="RSI↓ EXIT",
                    font=dict(color="#ef4444",size=8), showarrow=True,
                    arrowhead=1, arrowcolor="#ef4444", ay=20, ax=0, row=row_rsi, col=1)
        fig.update_yaxes(title_text="RSI", range=[0,100],
                         tickfont=dict(size=9), row=row_rsi, col=1)

    if show_macd and row_macd:
        ml, ms, mh = _macd_calc(closes)
        fig.add_trace(go.Bar(x=dates, y=mh,
            marker_color=["rgba(0,255,136,0.7)" if v>=0 else "rgba(239,68,68,0.7)" for v in mh],
            name="MACD Hist", showlegend=False), row=row_macd, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ml,
            line=dict(color="#60a5fa", width=1.5), name="MACD"), row=row_macd, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ms,
            line=dict(color="#f97316", width=1.5), name="Signal"), row=row_macd, col=1)
        fig.add_hline(y=0, line=dict(color="#6b7280", width=1, dash="dot"),
                      row=row_macd, col=1)
        for i in range(1, len(ml)):
            if None in (ml[i], ms[i], ml[i-1], ms[i-1]): continue
            if ml[i-1]<=ms[i-1] and ml[i]>ms[i]:
                fig.add_annotation(x=dates[i], y=ml[i], text="▲ MACD",
                    font=dict(color="#00ff88",size=8), showarrow=True,
                    arrowhead=1, arrowcolor="#00ff88", ay=-20, ax=0, row=row_macd, col=1)
            elif ml[i-1]>=ms[i-1] and ml[i]<ms[i]:
                fig.add_annotation(x=dates[i], y=ml[i], text="▼ MACD",
                    font=dict(color="#ef4444",size=8), showarrow=True,
                    arrowhead=1, arrowcolor="#ef4444", ay=20, ax=0, row=row_macd, col=1)
        fig.update_yaxes(title_text="MACD", tickfont=dict(size=9), row=row_macd, col=1)

    if vols:
        fig.add_trace(go.Bar(x=dates, y=vols,
            marker_color=["rgba(0,255,136,0.4)" if c>=o else "rgba(239,68,68,0.4)"
                          for c,o in zip(closes,opens)],
            name="Volume", showlegend=False), row=row_vol, col=1)
        fig.update_yaxes(title_text="Vol", tickfont=dict(size=8), row=row_vol, col=1)

    ticker = row.get("Ticker",""); sq = "  🔥 SQ" if row.get("Squeeze") else ""
    ind_str = " · ".join([i.split(" ")[0] for i in indicators]) if indicators else "Base"
    fig.update_layout(**PLOTLY_DARK,
        title=dict(
            text=f"<b>{ticker}</b> — {row.get('Nome','')}  |  {row.get('Prezzo','')}  |  RSI {row.get('RSI','')}{sq}"
                 f"  <span style='color:#6b7280;font-size:11px'>[ {ind_str} ]</span>",
            font=dict(color="#00ff88", size=13)),
        height=160+180*n_rows, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=55, b=0), hovermode="x unified")
    for r in range(1, n_rows+1):
        fig.update_xaxes(gridcolor="#1f2937", row=r, col=1)
        fig.update_yaxes(gridcolor="#1f2937", row=r, col=1)
    return fig

def build_radar(row: pd.Series) -> go.Figure:
    qc = row.get("_quality_components")
    if not qc or not isinstance(qc, dict): return None
    keys = list(qc.keys()); vals = list(qc.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=keys+[keys[0]], fill="toself",
        fillcolor="rgba(0,255,136,0.15)", line=dict(color="#00ff88", width=2), name="Quality"))
    fig.update_layout(**PLOTLY_DARK,
        polar=dict(bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9,color="#6b7280"),
                gridcolor="#1f2937", linecolor="#1f2937"),
            angularaxis=dict(tickfont=dict(size=11,color="#c9d1d9"),
                gridcolor="#1f2937", linecolor="#1f2937")),
        title=dict(
            text=f"<b>{row.get('Ticker','')}</b>  Quality: <b style='color:#00ff88'>{row.get('Quality_Score',0)}/12</b>",
            font=dict(color="#58a6ff", size=13)),
        height=340, margin=dict(l=40,r=40,t=55,b=20), showlegend=False)
    return fig

def show_charts(row_full: pd.Series, key_suffix: str = ""):
    ticker = row_full.get("Ticker", "")
    st.markdown("---")
    ind_opts = ["SMA 9 & 21 + RSI", "MACD", "Parabolic SAR"]

    ctl1, ctl2 = st.columns([4, 1])
    with ctl1:
        indicators = st.multiselect(
            "🔧 Indicatori",
            options=ind_opts,
            default=st.session_state.get("active_indicators", ind_opts),
            key=f"ind_{ticker}_{key_suffix}",
        )
        st.session_state["active_indicators"] = indicators
    with ctl2:
        st.write("")
        if st.button("🔄 Aggiorna", key=f"ref_{ticker}_{key_suffix}"):
            st.rerun()

    st.markdown(f'<div class="section-pill">📊 ANALISI — {ticker}</div>', unsafe_allow_html=True)
    fig = build_full_chart(row_full, indicators)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"full_{ticker}_{key_suffix}")
    else:
        st.info("Dati grafici non disponibili. Riesegui lo scanner.")

    fig_r = build_radar(row_full)
    if fig_r:
        _, c2, _ = st.columns([1,1,1])
        with c2:
            st.markdown('<div class="section-pill" style="text-align:center">🧭 QUALITY RADAR</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_r, use_container_width=True, key=f"radar_{ticker}_{key_suffix}")

# =========================================================================
# JS RENDERERS
# =========================================================================
name_dblclick_renderer = JsCode("""
class NameDblClick {
    init(p) {
        this.eGui = document.createElement('span');
        this.eGui.innerText = p.value || '';
        const t = p.data.Ticker || p.data.ticker;
        if (!t) return;
        this.eGui.style.cursor='pointer';
        this.eGui.title='Doppio click → TradingView';
        this.eGui.ondblclick = () =>
            window.open("https://www.tradingview.com/chart/?symbol="+String(t).split(".")[0],"_blank");
    }
    getGui() { return this.eGui; }
}""")

rsi_renderer = JsCode("""
class RsiR {
    init(p) {
        this.eGui = document.createElement('span');
        const v = parseFloat(p.value);
        this.eGui.innerText = isNaN(v) ? '-' : v.toFixed(1);
        this.eGui.style.fontWeight='bold'; this.eGui.style.fontFamily='Courier New';
        if      (v<30)  this.eGui.style.color='#60a5fa';
        else if (v<40)  this.eGui.style.color='#93c5fd';
        else if (v<=65) this.eGui.style.color='#00ff88';
        else if (v<=70) this.eGui.style.color='#f59e0b';
        else            this.eGui.style.color='#ef4444';
    }
    getGui() { return this.eGui; }
}""")

vol_ratio_renderer = JsCode("""
class VolR {
    init(p) {
        this.eGui = document.createElement('span');
        const v = parseFloat(p.value);
        this.eGui.innerText = isNaN(v) ? '-' : v.toFixed(2)+'x';
        this.eGui.style.fontFamily='Courier New'; this.eGui.style.fontWeight='bold';
        if      (v<1)   this.eGui.style.color='#6b7280';
        else if (v<2)   this.eGui.style.color='#00ff88';
        else if (v<3)   this.eGui.style.color='#f59e0b';
        else          { this.eGui.style.color='#ef4444'; this.eGui.style.textShadow='0 0 6px #ef4444'; }
    }
    getGui() { return this.eGui; }
}""")

early_score_renderer = JsCode("""
class ESR {
    init(p) {
        this.eGui = document.createElement('span');
        const v = parseFloat(p.value||0);
        this.eGui.innerText = v.toFixed(1);
        this.eGui.style.fontFamily='Courier New'; this.eGui.style.fontWeight='bold';
        if      (v>=8) { this.eGui.style.color='#00ff88'; this.eGui.style.textShadow='0 0 8px #00ff88'; }
        else if (v>=5)   this.eGui.style.color='#f59e0b';
        else if (v>0)    this.eGui.style.color='#9ca3af';
        else             this.eGui.style.color='#374151';
    }
    getGui() { return this.eGui; }
}""")

quality_renderer = JsCode("""
class QR {
    init(p) {
        this.eGui = document.createElement('div');
        this.eGui.style.cssText='display:flex;align-items:center;gap:6px';
        const v = parseInt(p.value||0);
        const pct = Math.round((v/12)*100);
        const c = v>=9?'#00ff88':v>=6?'#f59e0b':'#6b7280';
        this.eGui.innerHTML=`<span style="font-family:Courier New;font-weight:bold;color:${c};min-width:20px">${v}</span>
            <div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
            <div style="width:${pct}%;background:${c};height:6px;border-radius:3px"></div></div>`;
    }
    getGui() { return this.eGui; }
}""")

squeeze_renderer = JsCode("""
class SqR {
    init(p) {
        this.eGui = document.createElement('span');
        const v = p.value;
        if (v===true||v==='True'||v==='true')
            { this.eGui.innerText='🔥 SQ'; this.eGui.style.color='#f97316'; this.eGui.style.fontWeight='bold'; }
        else { this.eGui.innerText='—'; this.eGui.style.color='#374151'; }
    }
    getGui() { return this.eGui; }
}""")

rsi_div_renderer = JsCode("""
class RDR {
    init(p) {
        this.eGui = document.createElement('span');
        const v = p.value;
        if      (v==='BEARISH') { this.eGui.innerText='⚠️ BEAR'; this.eGui.style.color='#ef4444'; }
        else if (v==='BULLISH') { this.eGui.innerText='✅ BULL'; this.eGui.style.color='#00ff88'; }
        else                    { this.eGui.innerText='—';       this.eGui.style.color='#374151'; }
    }
    getGui() { return this.eGui; }
}""")

weekly_renderer = JsCode("""
class WR {
    init(p) {
        this.eGui = document.createElement('span');
        const v = p.value;
        if      (v===true||v==='True'||v==='true')    { this.eGui.innerText='📈 W+'; this.eGui.style.color='#00ff88'; }
        else if (v===false||v==='False'||v==='false') { this.eGui.innerText='📉 W—'; this.eGui.style.color='#ef4444'; }
        else                                          { this.eGui.innerText='—';     this.eGui.style.color='#374151'; }
    }
    getGui() { return this.eGui; }
}""")

price_renderer = JsCode("""
class PR {
    init(p) {
        this.eGui = document.createElement('span');
        this.eGui.innerText = p.value ?? '-';
        this.eGui.style.fontFamily='Courier New'; this.eGui.style.color='#e2e8f0';
        this.eGui.style.fontWeight='bold';
    }
    getGui() { return this.eGui; }
}""")

# =========================================================================
# EXPORT HELPERS
# =========================================================================
def to_excel_bytes(sheets_dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for name, df in sheets_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()

def make_tv_csv(df, tab_name):
    tmp = df[["Ticker"]].copy(); tmp.insert(0, "Tab", tab_name)
    return tmp.to_csv(index=False).encode("utf-8")

def csv_btn(df, filename, key):
    st.download_button("📥 CSV", df.to_csv(index=False).encode(), filename, "text/csv", key=key)

# =========================================================================
# PRESETS
# =========================================================================
PRESETS = {
    "⚡ Aggressivo":    dict(eh=0.01,prmin=45,prmax=65,rpoc=0.01,vol_ratio_hot=1.2,top=20,
                             min_early_score=2.0,min_quality=3,min_pro_score=2.0),
    "⚖️ Bilanciato":    dict(eh=0.02,prmin=40,prmax=70,rpoc=0.02,vol_ratio_hot=1.5,top=15,
                             min_early_score=4.0,min_quality=5,min_pro_score=4.0),
    "🛡️ Conservativo":  dict(eh=0.04,prmin=35,prmax=75,rpoc=0.04,vol_ratio_hot=2.0,top=10,
                             min_early_score=6.0,min_quality=7,min_pro_score=6.0),
    "🔓 Nessun Filtro": dict(eh=0.05,prmin=10,prmax=90,rpoc=0.05,vol_ratio_hot=0.3,top=100,
                             min_early_score=0.0,min_quality=0,min_pro_score=0.0),
}

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(page_title="Trading Scanner PRO 26.0", layout="wide", page_icon="🧠")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 🧠 Trading Scanner PRO 26.0")
st.markdown('<div class="section-pill">DARK · SMA · MACD · SAR · MULTI-WATCHLIST · v26.0</div>',
            unsafe_allow_html=True)
init_db()

# =========================================================================
# SESSION STATE
# =========================================================================
defaults = dict(
    mSP500=True, mNasdaq=True, mFTSE=True, mEurostoxx=False,
    mDow=False, mRussell=False, mStoxxEmerging=False, mUSSmallCap=False,
    eh=0.02, prmin=40, prmax=70, rpoc=0.02, vol_ratio_hot=1.5, top=15,
    min_early_score=2.0, min_quality=3, min_pro_score=2.0,
    current_list_name="DEFAULT", last_active_tab="EARLY",
    active_indicators=["SMA 9 & 21 + RSI", "MACD", "Parabolic SAR"],
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# =========================================================================
# KPI BAR
# =========================================================================
def render_kpi_bar(df_ep, df_rea):
    hist = load_scan_history(2)
    p_e=p_p=p_h=p_c=0
    if len(hist)>=2:
        pr=hist.iloc[1]
        p_e=int(pr.get("n_early",0)); p_p=int(pr.get("n_pro",0))
        p_h=int(pr.get("n_rea",0));   p_c=int(pr.get("n_confluence",0))
    n_e=int((df_ep.get("Stato_Early",pd.Series())=="EARLY").sum()) if not df_ep.empty else 0
    n_p=int((df_ep.get("Stato_Pro",  pd.Series())=="PRO"  ).sum()) if not df_ep.empty else 0
    n_h=len(df_rea) if not df_rea.empty else 0
    n_c=0
    if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
        n_c=int(((df_ep["Stato_Early"]=="EARLY")&(df_ep["Stato_Pro"]=="PRO")).sum())
    k1,k2,k3,k4=st.columns(4)
    k1.metric("📡 EARLY",     n_e, delta=n_e-p_e if p_e else None)
    k2.metric("💪 PRO",       n_p, delta=n_p-p_p if p_p else None)
    k3.metric("🔥 REA-HOT",   n_h, delta=n_h-p_h if p_h else None)
    k4.metric("⭐ CONFLUENCE", n_c, delta=n_c-p_c if p_c else None)

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.title("⚙️ Configurazione")

with st.sidebar.expander("🎯 Preset Rapidi", expanded=False):
    for pname, pvals in PRESETS.items():
        if st.button(pname, use_container_width=True, key=f"preset_{pname}"):
            for k,v in pvals.items(): st.session_state[k]=v
            st.rerun()

with st.sidebar.expander("🌍 Mercati", expanded=True):
    msp500   = st.checkbox("S&P 500",          st.session_state.mSP500)
    mnasdaq  = st.checkbox("Nasdaq 100",        st.session_state.mNasdaq)
    mftse    = st.checkbox("FTSE MIB",          st.session_state.mFTSE)
    meuro    = st.checkbox("Eurostoxx 600",     st.session_state.mEurostoxx)
    mdow     = st.checkbox("Dow Jones",         st.session_state.mDow)
    mrussell = st.checkbox("Russell 2000",      st.session_state.mRussell)
    mstoxxem = st.checkbox("Stoxx Emerging 50", st.session_state.mStoxxEmerging)
    mussmall = st.checkbox("US Small Cap 2000", st.session_state.mUSSmallCap)

sel = [mkt for flag,mkt in [
    (msp500,"SP500"),(mnasdaq,"Nasdaq"),(mftse,"FTSE"),(meuro,"Eurostoxx"),
    (mdow,"Dow"),(mrussell,"Russell"),(mstoxxem,"StoxxEmerging"),(mussmall,"USSmallCap"),
] if flag]

(st.session_state.mSP500,st.session_state.mNasdaq,st.session_state.mFTSE,
 st.session_state.mEurostoxx,st.session_state.mDow,st.session_state.mRussell,
 st.session_state.mStoxxEmerging,st.session_state.mUSSmallCap) = (
    msp500,mnasdaq,mftse,meuro,mdow,mrussell,mstoxxem,mussmall)

with st.sidebar.expander("🎛️ Parametri Scanner", expanded=False):
    eh            = st.slider("EARLY EMA20 %", 0.0,10.0,float(st.session_state.eh*100),0.5)/100
    prmin         = st.slider("PRO RSI min",   0,100,int(st.session_state.prmin),5)
    prmax         = st.slider("PRO RSI max",   0,100,int(st.session_state.prmax),5)
    rpoc          = st.slider("REA POC %",     0.0,10.0,float(st.session_state.rpoc*100),0.5)/100
    vol_ratio_hot = st.number_input("VolRatio HOT",0.0,10.0,float(st.session_state.vol_ratio_hot),0.1)
    top           = st.number_input("TOP N",   5,200,int(st.session_state.top),5)

(st.session_state.eh,st.session_state.prmin,st.session_state.prmax,
 st.session_state.rpoc,st.session_state.vol_ratio_hot,st.session_state.top) = (
    eh,prmin,prmax,rpoc,vol_ratio_hot,top)

with st.sidebar.expander("🔬 Soglie Filtri (live)", expanded=True):
    st.caption("⬇️ Abbassa per vedere più segnali  |  0 = nessun filtro")
    min_early_score = st.slider("Early Score ≥", 0.0,10.0,float(st.session_state.min_early_score),0.5)
    min_quality     = st.slider("Quality ≥",     0,12,   int(st.session_state.min_quality),1)
    min_pro_score   = st.slider("Pro Score ≥",   0.0,10.0,float(st.session_state.min_pro_score),0.5)
    st.session_state.min_early_score = min_early_score
    st.session_state.min_quality     = min_quality
    st.session_state.min_pro_score   = min_pro_score

with st.sidebar.expander("📊 Indicatori Grafici", expanded=False):
    ind_opts_all = ["SMA 9 & 21 + RSI", "MACD", "Parabolic SAR"]
    active_ind = st.multiselect("Attivi di default", options=ind_opts_all,
        default=[x for x in st.session_state.active_indicators if x in ind_opts_all],
        key="global_indicators")
    st.session_state.active_indicators = active_ind

st.sidebar.divider()
st.sidebar.subheader("📋 Watchlist")

df_wl_all    = load_watchlist()
list_options = sorted(df_wl_all["list_name"].unique().tolist()) if not df_wl_all.empty else []
if "DEFAULT" not in list_options: list_options.append("DEFAULT")
list_options = sorted(list_options)

active_list = st.sidebar.selectbox("Lista Attiva", list_options,
    index=list_options.index(st.session_state.current_list_name)
    if st.session_state.current_list_name in list_options else 0,
    key="active_list")
st.session_state.current_list_name = active_list

new_list = st.sidebar.text_input("➕ Nuova lista")
if st.sidebar.button("Crea lista") and new_list.strip():
    name_clean = new_list.strip()
    st.session_state.current_list_name = name_clean
    # Force create by adding a dummy record that we'll never see (list becomes visible only with real tickers)
    st.sidebar.success(f"Lista '{name_clean}' pronta.")
    st.rerun()

if st.sidebar.button("⚠️ Reset Watchlist DB", key="rst_wl"):
    reset_watchlist_db(); st.rerun()

st.sidebar.divider()
if st.sidebar.button("🗑️ Reset Storico", key="reset_hist_sidebar"):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM scan_history"); conn.commit(); conn.close()
        st.sidebar.success("Storico cancellato."); st.rerun()
    except Exception as e:
        st.sidebar.error(f"Errore: {e}")

only_watchlist = st.sidebar.checkbox("Solo Watchlist", False)

# =========================================================================
# SCANNER
# =========================================================================
if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 26.0", type="primary", use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            rep,rrea=[],[]
            pb=st.progress(0); status=st.empty(); tot=len(universe)
            for i,tkr in enumerate(universe,1):
                status.text(f"Analisi {i}/{tot}: {tkr}")
                ep,rea = scan_ticker(tkr,eh,prmin,prmax,rpoc,vol_ratio_hot)
                if ep:  rep.append(ep)
                if rea: rrea.append(rea)
                pb.progress(i/tot)
            df_ep_new  = pd.DataFrame(rep)
            df_rea_new = pd.DataFrame(rrea)
            st.session_state.df_ep     = df_ep_new
            st.session_state.df_rea    = df_rea_new
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
            save_scan_history(sel, df_ep_new, df_rea_new)
            n_h=len(df_rea_new); n_c=0
            if not df_ep_new.empty and "Stato_Early" in df_ep_new.columns:
                n_c=int(((df_ep_new["Stato_Early"]=="EARLY")&(df_ep_new["Stato_Pro"]=="PRO")).sum())
            if n_h>=5: st.toast(f"🔥 {n_h} segnali HOT!",icon="🔥")
            if n_c>=3: st.toast(f"⭐ {n_c} CONFLUENCE!",icon="⭐")
            st.rerun()

df_ep  = st.session_state.get("df_ep",  pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

if "last_scan" in st.session_state:
    st.caption(f"⏱️ Ultima scansione: {st.session_state.last_scan}")

render_kpi_bar(df_ep, df_rea)
st.markdown("---")

# =========================================================================
# AGGRID BUILDER  — resize manuale abilitato
# =========================================================================
def build_aggrid(df_disp: pd.DataFrame, grid_key: str, height: int = 480):
    gb = GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(
        sortable=True, resizable=True, filterable=True, editable=False,
        wrapText=False, suppressSizeToFit=False,
    )
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)

    col_w = {"Ticker":80,"Nome":160,"Prezzo_fmt":90,"MarketCap_fmt":110,
             "Early_Score":95,"Pro_Score":80,"Quality_Score":130,
             "RSI":65,"Vol_Ratio":85,"Squeeze":70,"RSI_Div":85,
             "Weekly_Bull":80,"Stato_Early":85,"Stato_Pro":80}
    for c,w in col_w.items():
        if c in df_disp.columns: gb.configure_column(c, width=w)

    rmap = {"Nome":name_dblclick_renderer,"RSI":rsi_renderer,
            "Vol_Ratio":vol_ratio_renderer,"Early_Score":early_score_renderer,
            "Quality_Score":quality_renderer,"Squeeze":squeeze_renderer,
            "RSI_Div":rsi_div_renderer,"Weekly_Bull":weekly_renderer,
            "Prezzo_fmt":price_renderer}
    for c,r in rmap.items():
        if c in df_disp.columns: gb.configure_column(c, cellRenderer=r)

    if "Ticker" in df_disp.columns: gb.configure_column("Ticker", pinned="left")
    if "Nome"   in df_disp.columns: gb.configure_column("Nome",   pinned="left")

    go_opts = gb.build()
    # sizeColumnsToFit al primo render, poi l'utente può ridimensionare
    go_opts["onFirstDataRendered"] = JsCode("function(p){p.api.sizeColumnsToFit();}")

    return AgGrid(df_disp, gridOptions=go_opts, height=height,
                  update_mode=GridUpdateMode.SELECTION_CHANGED,
                  data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                  fit_columns_on_grid_load=True, theme="streamlit",
                  allow_unsafe_jscode=True, key=grid_key)

# =========================================================================
# LEGENDE  — complete e dettagliate
# =========================================================================
LEGENDS = {
    "EARLY": {
        "desc": "Titoli dove il prezzo è **vicino o sopra la EMA20** — zona di potenziale rimbalzo o continuazione. Ideale per ingressi anticipati prima del breakout.",
        "cols": [
            ("Early_Score","0–10","Punteggio prossimità EMA20. ≥8 = ottimo, 5–7 = buono, <5 = marginale"),
            ("RSI","0–100","Momentum. Blu <30 (oversold), Verde 40–65 (neutro-bull), Rosso >70 (overbought)"),
            ("EMA20","prezzo","Media mobile 20 giorni. Supporto/resistenza dinamico principale"),
            ("Squeeze","🔥/—","Bollinger dentro Keltner: esplosione imminente di volatilità"),
            ("Weekly_Bull","📈/📉","Trend settimanale: verde = above EMA20 weekly"),
        ],
        "filters": "Stato_Early = 'EARLY'  AND  Early_Score ≥ soglia_E",
        "sort": "Early_Score DESC, RSI ASC",
    },
    "PRO": {
        "desc": "Titoli con **trend confermato** (prezzo > EMA20 > EMA50), RSI in zona neutro-rialzista e volume sopra media. Setup completo per posizioni long.",
        "cols": [
            ("Pro_Score","0–8","Score tecnico: +3 trend, +3 RSI range, +2 volume. ≥8 = PRO confermato"),
            ("Quality_Score","0–12","Score composito: Vol_Ratio+OBV+ATR+RSI+EMA20+EMA50. ≥9 alta qualità"),
            ("RSI","40–70","Range ideale PRO: momentum attivo ma non esaurito"),
            ("Vol_Ratio","x","Volume oggi/media 20gg. >1.5x = partecipazione, >3x = anomalia"),
            ("RSI_Div","BULL/BEAR","Divergenza RSI-prezzo: BULL = segnale long, BEAR = attenzione"),
        ],
        "filters": "Stato_Pro = 'PRO'  AND  Pro_Score ≥ soglia_P  AND  Quality ≥ soglia_Q",
        "sort": "Quality_Score DESC, Pro_Score DESC, RSI ASC",
    },
    "REA-HOT": {
        "desc": "Titoli con **volumi anomali vicini al POC** (Point of Control del Volume Profile). Segnala interesse istituzionale e potenziali movimenti bruschi.",
        "cols": [
            ("Vol_Ratio","x","Volume/media 20gg. >vol_ratio_hot = trigger HOT"),
            ("Dist_POC_%","%","Distanza % dal POC. Minore = prezzo al livello di massimo volume storico"),
            ("POC","prezzo","Point of Control: livello con maggior volume scambiato"),
            ("Rea_Score","0–7","Score REA. 7 = dist_poc < soglia AND vol_ratio > soglia_hot"),
        ],
        "filters": "dist_poc < rpoc  AND  Vol_Ratio > vol_ratio_hot",
        "sort": "Vol_Ratio DESC, Dist_POC_% ASC",
    },
    "⭐ CONFLUENCE": {
        "desc": "Titoli che soddisfano **contemporaneamente EARLY e PRO** — setup ad altissima probabilità. Combinazione ideale di timing (early) e forza (pro).",
        "cols": [
            ("Early_Score","0–10","Prossimità EMA20 — timing d'ingresso"),
            ("Pro_Score","0–8","Forza del trend confermato"),
            ("Quality_Score","0–12","Qualità complessiva del setup"),
        ],
        "filters": "Stato_Early='EARLY' AND Stato_Pro='PRO' AND Early≥soglia_E AND Quality≥soglia_Q",
        "sort": "Quality_Score DESC, Early_Score DESC, Pro_Score DESC",
    },
    "Regime Momentum": {
        "desc": "Titoli PRO ordinati per **Momentum composito** = Pro_Score×10 + RSI. Evidenzia i titoli con la maggiore forza relativa in questo momento.",
        "cols": [
            ("Momentum","calc","Pro_Score×10 + RSI. Più alto = momentum più forte"),
            ("Pro_Score","0–8","Forza trend"),
            ("RSI","0–100","Contributo momentum RSI"),
        ],
        "filters": "Stato_Pro = 'PRO'  AND  Pro_Score ≥ soglia_P",
        "sort": "Momentum DESC",
    },
    "Multi-Timeframe": {
        "desc": "Titoli PRO con **allineamento multi-timeframe**: trend rialzista sia giornaliero (EMA20/50 daily) che settimanale (EMA20 weekly). Massima coerenza di trend.",
        "cols": [
            ("Weekly_Bull","📈 W+","Prezzo sopra EMA20 weekly = trend settimanale rialzista"),
            ("Quality_Score","0–12","Qualità setup daily"),
            ("Pro_Score","0–8","Forza trend daily"),
        ],
        "filters": "Stato_Pro='PRO' AND Weekly_Bull=True AND Pro_Score≥soglia_P",
        "sort": "Quality_Score DESC, Pro_Score DESC",
    },
    "Finviz": {
        "desc": "Titoli PRO filtrati per **capitalizzazione ≥ mediana** e volume anomalo (Vol_Ratio > 1.2). Focus su titoli liquidi e istituzionali con segnale confermato.",
        "cols": [
            ("MarketCap","$","Cap ≥ mediana del campione — titoli di qualità"),
            ("Vol_Ratio","x",">1.2x = partecipazione superiore alla media"),
            ("Quality_Score","0–12","Qualità complessiva"),
        ],
        "filters": "PRO  AND  MarketCap ≥ median(MarketCap)  AND  Vol_Ratio > 1.2",
        "sort": "Quality_Score DESC, Pro_Score DESC",
    },
}

def show_legend(title):
    info = LEGENDS.get(title)
    if not info:
        return
    with st.expander(f"📖 Come funziona: {title}", expanded=False):
        st.markdown(info["desc"])
        st.markdown(f"""
<table class="legend-table">
<tr><th>Colonna</th><th>Range</th><th>Significato</th></tr>
{"".join(f'<tr><td class="legend-col-name">{c}</td><td class="legend-col-range">{r}</td><td>{d}</td></tr>' for c,r,d in info["cols"])}
</table>
<br>
<span style="color:#6b7280;font-size:0.78rem">
🔬 <b>Filtro applicato:</b> <code>{info["filters"]}</code><br>
📊 <b>Ordinamento:</b> <code>{info["sort"]}</code>
</span>
""", unsafe_allow_html=True)

# =========================================================================
# RENDER SCAN TAB  — filtri corretti
# =========================================================================
def render_scan_tab(df, status_filter, sort_cols, ascending, title):
    if df.empty:
        st.info(f"Nessun dato. Esegui lo scanner per popolare la tab **{title}**.")
        return

    s_e = float(st.session_state.min_early_score)
    s_q = int(st.session_state.min_quality)
    s_p = float(st.session_state.min_pro_score)

    st.caption(
        f"🔬 Filtri attivi → Early Score ≥ **{s_e}** | Quality ≥ **{s_q}** | Pro Score ≥ **{s_p}**  "
        f"&nbsp;&nbsp;_(modifica: sidebar → 🔬 Soglie Filtri)_"
    )

    # ── Filtro stato + soglie ────────────────────────────────────────────
    if status_filter == "EARLY":
        if "Stato_Early" not in df.columns:
            st.warning("Colonna Stato_Early non trovata."); return
        df_f = df[df["Stato_Early"] == "EARLY"].copy()
        if "Early_Score" in df_f.columns and s_e > 0:
            df_f = df_f[df_f["Early_Score"] >= s_e]

    elif status_filter == "PRO":
        if "Stato_Pro" not in df.columns:
            st.warning("Colonna Stato_Pro non trovata."); return
        df_f = df[df["Stato_Pro"] == "PRO"].copy()
        # Pro_Score >= 8 è il criterio dello scanner — la soglia UI è aggiuntiva
        if "Pro_Score"     in df_f.columns and s_p > 0:
            df_f = df_f[df_f["Pro_Score"]     >= s_p]
        if "Quality_Score" in df_f.columns and s_q > 0:
            df_f = df_f[df_f["Quality_Score"] >= s_q]

    elif status_filter == "HOT":
        if "Stato" not in df.columns:
            st.warning("Colonna Stato non trovata."); return
        df_f = df[df["Stato"] == "HOT"].copy()

    elif status_filter == "CONFLUENCE":
        if "Stato_Early" not in df.columns or "Stato_Pro" not in df.columns:
            st.warning("Colonne Stato_Early/Stato_Pro non trovate."); return
        df_f = df[(df["Stato_Early"]=="EARLY") & (df["Stato_Pro"]=="PRO")].copy()
        if "Early_Score"   in df_f.columns and s_e > 0:
            df_f = df_f[df_f["Early_Score"]   >= s_e]
        if "Quality_Score" in df_f.columns and s_q > 0:
            df_f = df_f[df_f["Quality_Score"] >= s_q]

    elif status_filter == "REGIME":
        if "Stato_Pro" not in df.columns:
            df_f = df.copy()
        else:
            df_f = df[df["Stato_Pro"] == "PRO"].copy()
        if "Pro_Score" in df_f.columns and s_p > 0:
            df_f = df_f[df_f["Pro_Score"] >= s_p]
        if "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
            df_f["Momentum"] = df_f["Pro_Score"] * 10 + df_f["RSI"]
            sort_cols = ["Momentum"]; ascending = [False]

    elif status_filter == "MTF":
        if "Stato_Pro" not in df.columns:
            df_f = df.copy()
        else:
            df_f = df[df["Stato_Pro"] == "PRO"].copy()
        if "Pro_Score"   in df_f.columns and s_p > 0:
            df_f = df_f[df_f["Pro_Score"] >= s_p]
        if "Weekly_Bull" in df_f.columns:
            df_f = df_f[df_f["Weekly_Bull"].isin([True, "True", "true", 1])]

    else:
        df_f = df.copy()

    if df_f.empty:
        st.warning(
            f"⚠️ Nessun segnale **{title}** trovato con le soglie attuali.  \n"
            f"💡 Prova il preset **🔓 Nessun Filtro** oppure abbassa le soglie nella sidebar."
        )
        return

    # ── Ordina e tronca ──────────────────────────────────────────────────
    valid_sort = [c for c in sort_cols if c in df_f.columns]
    if valid_sort:
        df_f = df_f.sort_values(valid_sort, ascending=ascending[:len(valid_sort)])
    df_f = df_f.head(int(st.session_state.top))

    # ── Mini KPI ─────────────────────────────────────────────────────────
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Titoli trovati", len(df_f))
    if "Squeeze" in df_f.columns:
        m2.metric("🔥 Squeeze", int(df_f["Squeeze"].apply(
            lambda x: x is True or str(x).lower()=="true").sum()))
    if "Weekly_Bull" in df_f.columns:
        m3.metric("📈 Weekly+", int(df_f["Weekly_Bull"].apply(
            lambda x: x is True or str(x).lower()=="true").sum()))
    if "RSI_Div" in df_f.columns:
        m4.metric("⚠️ Div RSI", int((df_f["RSI_Div"] != "-").sum()))

    # ── Formatta per AgGrid ───────────────────────────────────────────────
    df_fmt  = add_formatted_cols(df_f)
    df_disp = prepare_display_df(df_fmt)
    cols = list(df_disp.columns)
    base = [c for c in ["Ticker","Nome"] if c in cols]
    df_disp = df_disp[base + [c for c in cols if c not in base]].reset_index(drop=True)

    ce1,ce2 = st.columns([1,3])
    with ce1: csv_btn(df_f, f"{title.lower().replace(' ','_')}.csv", f"exp_{title}")
    with ce2: st.caption(
        f"Seleziona riga/e → **➕ Aggiungi** alla lista `{st.session_state.current_list_name}`.  "
        "Doppio click su Nome → TradingView.")

    grid_resp   = build_aggrid(df_disp, f"grid_{title}")
    selected_df = pd.DataFrame(grid_resp["selected_rows"])

    if st.button(f"➕ Aggiungi a '{st.session_state.current_list_name}'", key=f"btn_{title}"):
        if not selected_df.empty and "Ticker" in selected_df.columns:
            tickers = selected_df["Ticker"].tolist()
            names   = selected_df.get("Nome", selected_df["Ticker"]).tolist()
            add_to_watchlist(tickers, names, title, "Scanner", "LONG",
                             st.session_state.current_list_name)
            st.success(f"✅ Aggiunti {len(tickers)} titoli a '{st.session_state.current_list_name}'.")
            time.sleep(0.8); st.rerun()
        else:
            st.warning("Seleziona almeno una riga dalla tabella.")

    if not selected_df.empty:
        ticker_sel = selected_df.iloc[0].get("Ticker","")
        match = df_f[df_f["Ticker"]==ticker_sel]
        if not match.empty:
            show_charts(match.iloc[0], key_suffix=title)

# =========================================================================
# TABS
# =========================================================================
tabs = st.tabs(["📡 EARLY","💪 PRO","🔥 REA-HOT","⭐ CONFLUENCE",
                "🚀 Momentum","🌐 Multi-TF","🔎 Finviz",
                "📋 Watchlist","📜 Storico"])
(tab_e,tab_p,tab_r,tab_conf,tab_regime,tab_mtf,
 tab_finviz,tab_w,tab_hist) = tabs

with tab_e:
    st.session_state.last_active_tab="EARLY"
    show_legend("EARLY")
    render_scan_tab(df_ep,"EARLY",["Early_Score","RSI"],[False,True],"EARLY")

with tab_p:
    st.session_state.last_active_tab="PRO"
    show_legend("PRO")
    render_scan_tab(df_ep,"PRO",["Quality_Score","Pro_Score","RSI"],[False,False,True],"PRO")

with tab_r:
    st.session_state.last_active_tab="REA-HOT"
    show_legend("REA-HOT")
    render_scan_tab(df_rea,"HOT",["Vol_Ratio","Dist_POC_%"],[False,True],"REA-HOT")

with tab_conf:
    st.session_state.last_active_tab="CONFLUENCE"
    show_legend("⭐ CONFLUENCE")
    render_scan_tab(df_ep,"CONFLUENCE",
                    ["Quality_Score","Early_Score","Pro_Score"],[False,False,False],"CONFLUENCE")

with tab_regime:
    show_legend("Regime Momentum")
    render_scan_tab(df_ep,"REGIME",["Pro_Score"],[False],"Regime Momentum")

with tab_mtf:
    show_legend("Multi-Timeframe")
    render_scan_tab(df_ep,"MTF",["Quality_Score","Pro_Score"],[False,False],"Multi-Timeframe")

with tab_finviz:
    show_legend("Finviz")
    sp = df_ep.get("Stato_Pro")
    df_fv = df_ep[sp=="PRO"].copy() if sp is not None and not df_ep.empty else df_ep.copy()
    if not df_fv.empty:
        if "MarketCap" in df_fv.columns: df_fv = df_fv[df_fv["MarketCap"]>=df_fv["MarketCap"].median()]
        if "Vol_Ratio" in df_fv.columns: df_fv = df_fv[df_fv["Vol_Ratio"]>1.2]
    render_scan_tab(df_fv,"PRO",["Quality_Score","Pro_Score"],[False,False],"Finviz")

# =========================================================================
# WATCHLIST  — multi-lista con gestione completa
# =========================================================================
with tab_w:
    st.markdown(
        f'<div class="section-pill">📋 WATCHLIST MANAGER — lista attiva: {st.session_state.current_list_name}</div>',
        unsafe_allow_html=True)

    df_wl_full = load_watchlist()

    # ── Pannello gestione liste ──────────────────────────────────────────
    with st.expander("⚙️ Gestione Liste", expanded=True):
        all_lists = sorted(df_wl_full["list_name"].unique().tolist()) if not df_wl_full.empty else ["DEFAULT"]
        if not all_lists: all_lists = ["DEFAULT"]

        gc1,gc2,gc3,gc4 = st.columns(4)
        with gc1:
            st.markdown("**📂 Liste esistenti**")
            for ln in all_lists:
                n_items = len(df_wl_full[df_wl_full["list_name"]==ln]) if not df_wl_full.empty else 0
                active_marker = " ✅" if ln==st.session_state.current_list_name else ""
                if st.button(f"{ln}  ({n_items}){active_marker}", key=f"switch_{ln}",
                             use_container_width=True):
                    st.session_state.current_list_name = ln; st.rerun()

        with gc2:
            st.markdown("**✏️ Rinomina lista**")
            rename_src = st.selectbox("Lista da rinominare", all_lists, key="ren_src")
            rename_dst = st.text_input("Nuovo nome", key="ren_dst")
            if st.button("✏️ Rinomina", key="do_rename") and rename_dst.strip():
                rename_watchlist(rename_src, rename_dst.strip())
                if st.session_state.current_list_name == rename_src:
                    st.session_state.current_list_name = rename_dst.strip()
                st.rerun()

        with gc3:
            st.markdown("**📋 Copia lista in**")
            copy_src = st.selectbox("Copia da", all_lists, key="copy_src")
            copy_dst = st.text_input("Destinazione (nome nuovo o esistente)", key="copy_dst")
            if st.button("📋 Copia", key="do_copy") and copy_dst.strip():
                df_src = df_wl_full[df_wl_full["list_name"]==copy_src]
                if not df_src.empty:
                    tcol = "Ticker" if "Ticker" in df_src.columns else "ticker"
                    ncol = "Nome"   if "Nome"   in df_src.columns else "name"
                    tks  = df_src[tcol].tolist()
                    nms  = df_src[ncol].tolist() if ncol in df_src.columns else tks
                    add_to_watchlist(tks, nms, "Copia", f"da {copy_src}", "LONG", copy_dst.strip())
                    st.success(f"✅ {len(tks)} ticker copiati in '{copy_dst.strip()}'.")
                    st.rerun()

        with gc4:
            st.markdown("**🗑️ Elimina lista**")
            del_list = st.selectbox("Lista da eliminare", all_lists, key="del_list_sel")
            if st.button("🗑️ Elimina lista", key="do_del_list", type="secondary"):
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute("DELETE FROM watchlist WHERE list_name=?", (del_list,))
                conn.commit(); conn.close()
                if st.session_state.current_list_name == del_list:
                    remaining = [l for l in all_lists if l != del_list]
                    st.session_state.current_list_name = remaining[0] if remaining else "DEFAULT"
                st.rerun()

    # ── Contenuto lista attiva ───────────────────────────────────────────
    df_wl = df_wl_full[df_wl_full["list_name"]==st.session_state.current_list_name].copy() \
            if not df_wl_full.empty else pd.DataFrame()

    st.markdown(
        f'<div class="section-pill">📌 {st.session_state.current_list_name} — {len(df_wl)} titoli</div>',
        unsafe_allow_html=True)

    if df_wl.empty:
        st.info("Lista vuota. Aggiungi ticker dallo scanner o usa Copia lista.")
    else:
        tcol_wl = "Ticker" if "Ticker" in df_wl.columns else "ticker"
        ncol_wl = "Nome"   if "Nome"   in df_wl.columns else "name"

        # Merge colonne extra da scanner
        extra_cols = ["RSI","Vol_Ratio","Quality_Score","OBV_Trend","Weekly_Bull","Squeeze","Early_Score","Pro_Score"]
        df_wl_disp = df_wl.copy()
        if not df_ep.empty and "Ticker" in df_ep.columns:
            for ec in extra_cols:
                if ec in df_ep.columns:
                    mm = df_ep[["Ticker",ec]].drop_duplicates("Ticker")
                    df_wl_disp = df_wl_disp.merge(mm, left_on=tcol_wl, right_on="Ticker",
                                                    how="left", suffixes=("","_ep"))
                    if "Ticker_ep" in df_wl_disp.columns:
                        df_wl_disp.drop(columns=["Ticker_ep"], inplace=True)

        # ── Azioni in massa ──────────────────────────────────────────────
        wa1,wa2,wa3 = st.columns(3)
        with wa1:
            csv_btn(df_wl, f"watchlist_{st.session_state.current_list_name}.csv","exp_wl_dl")
        with wa2:
            move_dest = st.selectbox("Sposta selezionati in →",
                [l for l in all_lists if l!=st.session_state.current_list_name] or ["DEFAULT"],
                key="mass_move_dest")
        with wa3:
            copy_dest2 = st.selectbox("Copia selezionati in →",
                [l for l in all_lists if l!=st.session_state.current_list_name] or ["DEFAULT"],
                key="mass_copy_dest")

        # ── Cards ticker ─────────────────────────────────────────────────
        st.markdown("---")
        selected_ids = []

        for _, wrow in df_wl_disp.iterrows():
            rid     = wrow.get("id","")
            tkr     = wrow.get(tcol_wl,"")
            nom     = wrow.get(ncol_wl,"")
            rsi_v   = wrow.get("RSI",None)
            vr_v    = wrow.get("Vol_Ratio",None)
            qs_v    = wrow.get("Quality_Score",None)
            sq_v    = wrow.get("Squeeze",False)
            wb_v    = wrow.get("Weekly_Bull",None)
            origine = wrow.get("origine","")
            created = wrow.get("created_at","")

            # Badge RSI
            if rsi_v is not None and not (isinstance(rsi_v, float) and np.isnan(rsi_v)):
                rsi_num = float(rsi_v)
                rsi_cls = "badge-blue" if rsi_num<40 else "badge-green" if rsi_num<=65 else "badge-orange" if rsi_num<=70 else "badge-red"
                rsi_badge = f'<span class="wl-card-badge {rsi_cls}">RSI {rsi_num:.1f}</span>'
            else:
                rsi_badge = ""

            # Badge Vol_Ratio
            if vr_v is not None and not (isinstance(vr_v, float) and np.isnan(vr_v)):
                vr_num = float(vr_v)
                vr_cls = "badge-gray" if vr_num<1 else "badge-green" if vr_num<2 else "badge-orange" if vr_num<3 else "badge-red"
                vr_badge = f'<span class="wl-card-badge {vr_cls}">Vol {vr_num:.1f}x</span>'
            else:
                vr_badge = ""

            # Badge Quality
            if qs_v is not None and not (isinstance(qs_v, float) and np.isnan(qs_v)):
                qs_num = int(float(qs_v))
                qs_cls = "badge-green" if qs_num>=9 else "badge-orange" if qs_num>=6 else "badge-gray"
                qs_badge = f'<span class="wl-card-badge {qs_cls}">Q {qs_num}/12</span>'
            else:
                qs_badge = ""

            sq_badge  = '<span class="wl-card-badge badge-orange">🔥 SQ</span>' if sq_v is True or str(sq_v).lower()=="true" else ""
            wb_badge  = '<span class="wl-card-badge badge-green">📈 W+</span>' if wb_v is True or str(wb_v).lower()=="true" else \
                        '<span class="wl-card-badge badge-red">📉 W—</span>'   if wb_v is False or str(wb_v).lower()=="false" else ""

            row_cols = st.columns([0.3, 3, 1])
            with row_cols[0]:
                checked = st.checkbox("", key=f"chk_{rid}", label_visibility="collapsed")
                if checked: selected_ids.append(rid)
            with row_cols[1]:
                st.markdown(f"""
<div class="wl-card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span class="wl-card-ticker">{tkr}</span>
      <span class="wl-card-name"> &nbsp;{nom}</span>
    </div>
    <div style="color:#374151;font-size:0.72rem">{origine} · {created[:10] if created else ''}</div>
  </div>
  <div style="margin-top:8px">{rsi_badge}{vr_badge}{qs_badge}{sq_badge}{wb_badge}</div>
</div>""", unsafe_allow_html=True)
            with row_cols[2]:
                st.write("")
                if st.button("🗑️", key=f"del_{rid}", help=f"Elimina {tkr}"):
                    delete_from_watchlist([rid]); st.rerun()

        # ── Azioni sui selezionati ───────────────────────────────────────
        if selected_ids:
            st.markdown(f"**{len(selected_ids)} selezionati**")
            ac1,ac2,ac3 = st.columns(3)
            with ac1:
                if st.button(f"➡️ Sposta in '{move_dest}'", key="do_mass_move"):
                    move_watchlist_rows(selected_ids, move_dest); st.rerun()
            with ac2:
                if st.button(f"📋 Copia in '{copy_dest2}'", key="do_mass_copy"):
                    rows_sel = df_wl_disp[df_wl_disp["id"].isin(selected_ids)]
                    tks = rows_sel[tcol_wl].tolist()
                    nms = rows_sel[ncol_wl].tolist() if ncol_wl in rows_sel.columns else tks
                    add_to_watchlist(tks, nms, "Copia", "da selezione", "LONG", copy_dest2)
                    st.success(f"✅ Copiati {len(tks)} ticker."); st.rerun()
            with ac3:
                if st.button("🗑️ Elimina selezionati", key="do_mass_del", type="secondary"):
                    delete_from_watchlist(selected_ids); st.rerun()

        # ── Grafici ticker selezionato ───────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-pill">📊 ANALISI TICKER</div>', unsafe_allow_html=True)

        tickers_wl = df_wl[tcol_wl].dropna().unique().tolist()
        if tickers_wl:
            sel_wl = st.selectbox("🔍 Seleziona ticker",
                                   options=tickers_wl, key="wl_ticker_sel")
            row_wl = None
            for src in [df_ep, df_rea]:
                if src.empty or "Ticker" not in src.columns: continue
                m = src[src["Ticker"]==sel_wl]
                if not m.empty: row_wl=m.iloc[0]; break
            if row_wl is not None:
                show_charts(row_wl, key_suffix="wl")
            else:
                st.info(f"📭 Dati grafici non disponibili per **{sel_wl}**.  "
                        "Esegui lo scanner per caricarli.")

    if st.button("🔄 Refresh Watchlist", key="wl_ref"): st.rerun()

# =========================================================================
# STORICO
# =========================================================================
with tab_hist:
    st.markdown('<div class="section-pill">📜 STORICO SCANSIONI</div>', unsafe_allow_html=True)
    col_t, col_rst = st.columns([4,1])
    with col_rst:
        if st.button("🗑️ Reset Storico", key="reset_hist_tab", type="secondary"):
            try:
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute("DELETE FROM scan_history"); conn.commit(); conn.close()
                st.success("Storico cancellato!"); st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")

    df_hist = load_scan_history(20)
    if df_hist.empty:
        st.info("Nessuna scansione salvata.")
    else:
        st.dataframe(df_hist, use_container_width=True)
        st.markdown("---")
        st.subheader("🔍 Confronto Snapshot")
        hc1,hc2 = st.columns(2)
        with hc1: id_a = st.selectbox("Scansione A", df_hist["id"].tolist(), key="snap_a")
        with hc2: id_b = st.selectbox("Scansione B", df_hist["id"].tolist(),
                                       index=min(1,len(df_hist)-1), key="snap_b")
        if st.button("🔍 Confronta"):
            ep_a,_=load_scan_snapshot(id_a); ep_b,_=load_scan_snapshot(id_b)
            if ep_a.empty or ep_b.empty: st.warning("Dati non disponibili.")
            else:
                ta=set(ep_a.get("Ticker",[])); tb=set(ep_b.get("Ticker",[]))
                sc1,sc2,sc3=st.columns(3)
                sc1.metric("🆕 Nuovi",      len(tb-ta))
                sc2.metric("❌ Usciti",      len(ta-tb))
                sc3.metric("✅ Persistenti", len(ta&tb))
                if tb-ta: st.markdown("**🆕** "+", ".join(sorted(tb-ta)))
                if ta-tb: st.markdown("**❌** "+", ".join(sorted(ta-tb)))

# =========================================================================
# EXPORT GLOBALI
# =========================================================================
st.markdown("---")
st.markdown('<div class="section-pill">💾 EXPORT GLOBALI</div>', unsafe_allow_html=True)

df_conf_exp = pd.DataFrame()
if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
    df_conf_exp = df_ep[(df_ep["Stato_Early"]=="EARLY")&(df_ep["Stato_Pro"]=="PRO")].copy()

df_wl_exp = load_watchlist()
df_wl_exp = df_wl_exp[df_wl_exp["list_name"]==st.session_state.current_list_name]

all_exp = {"EARLY":df_ep,"PRO":df_ep,"REA-HOT":df_rea,
           "CONFLUENCE":df_conf_exp,"Watchlist":df_wl_exp}
cur_tab = st.session_state.get("last_active_tab","EARLY")
df_cur  = all_exp.get(cur_tab, pd.DataFrame())

ec1,ec2,ec3,ec4 = st.columns(4)
with ec1:
    st.download_button("📊 XLSX Tutti", to_excel_bytes(all_exp),
        "TradingScanner_v26_Tutti.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xlsx_all")
with ec2:
    tv_rows=[]
    for n,df_t in all_exp.items():
        if isinstance(df_t,pd.DataFrame) and not df_t.empty and "Ticker" in df_t.columns:
            tks=df_t["Ticker"].tolist()
            tv_rows.append(pd.DataFrame({"Tab":[n]*len(tks),"Ticker":tks}))
    if tv_rows:
        df_tv=pd.concat(tv_rows,ignore_index=True).drop_duplicates(subset=["Ticker"])
        st.download_button("📈 CSV TV Tutti", df_tv.to_csv(index=False).encode(),
            "TradingScanner_v26_TV.csv","text/csv",key="csv_tv_all")
with ec3:
    st.download_button(f"📊 XLSX {cur_tab}", to_excel_bytes({cur_tab:df_cur}),
        f"TradingScanner_v26_{cur_tab}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="xlsx_curr")
with ec4:
    if not df_cur.empty and "Ticker" in df_cur.columns:
        st.download_button(f"📈 CSV TV {cur_tab}", make_tv_csv(df_cur,cur_tab),
            f"TradingScanner_v26_{cur_tab}_TV.csv","text/csv",key="csv_tv_curr")
