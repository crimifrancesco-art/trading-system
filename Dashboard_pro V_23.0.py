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
# DARK THEME CSS  (Bloomberg / TradingView terminal style)
# =========================================================================

DARK_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #0a0e1a !important;
    color: #c9d1d9 !important;
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
    font-family: 'Courier New', monospace !important;
    font-size: 0.82rem !important; letter-spacing: 1px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00ff88 !important; border-bottom: 2px solid #00ff88 !important;
    background: #0d1117 !important;
}

[data-testid="stMetric"] {
    background: #0d1117 !important; border: 1px solid #1f2937 !important;
    border-radius: 8px !important; padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #00ff88 !important; font-size: 1.6rem !important;
                                 font-family: 'Courier New', monospace !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0d1117, #1a2233) !important;
    color: #00ff88 !important; border: 1px solid #00ff8855 !important;
    border-radius: 6px !important; font-family: 'Courier New', monospace !important;
    letter-spacing: 1px; transition: all 0.2s;
}
[data-testid="stButton"] > button:hover {
    border-color: #00ff88 !important; box-shadow: 0 0 12px #00ff8833 !important;
    color: #ffffff !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00401f, #006633) !important;
    border-color: #00ff88 !important; color: #00ff88 !important;
    font-size: 1rem !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: linear-gradient(135deg, #1a0a0a, #2d1010) !important;
    color: #ef4444 !important; border: 1px solid #ef444455 !important;
}

[data-testid="stDownloadButton"] > button {
    background: #0d1117 !important; color: #58a6ff !important;
    border: 1px solid #1f3a5f !important; border-radius: 6px !important;
}

[data-testid="stSelectbox"] select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #0d1117 !important; color: #c9d1d9 !important;
    border: 1px solid #1f2937 !important; border-radius: 6px !important;
}

[data-testid="stExpander"] {
    background: #0d1117 !important; border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #58a6ff !important; }

[data-testid="stAlert"] {
    background: #0d1117 !important; border-left: 3px solid #00ff88 !important;
}

hr { border-color: #1f2937 !important; }

/* AgGrid */
.ag-root-wrapper { background: #0d1117 !important; border: 1px solid #1f2937 !important; }
.ag-header { background: #0a0e1a !important; border-bottom: 1px solid #1f2937 !important; }
.ag-header-cell-label { color: #58a6ff !important;
                         font-family: 'Courier New', monospace !important;
                         font-size: 0.78rem !important; letter-spacing: 1px; }
.ag-row { background: #0d1117 !important; border-bottom: 1px solid #1a2233 !important; }
.ag-row:hover { background: #131d2e !important; }
.ag-row-selected { background: #0d2d1e !important; }
.ag-cell { color: #c9d1d9 !important;
           font-family: 'Courier New', monospace !important; font-size: 0.82rem !important; }
.ag-paging-panel { background: #0a0e1a !important; color: #6b7280 !important; }
.ag-side-bar { background: #0d1117 !important; }
.ag-filter { background: #0d1117 !important; color: #c9d1d9 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #374151; }

.kpi-card {
    background: linear-gradient(135deg, #0d1117, #111827);
    border: 1px solid #1f2937; border-radius: 10px;
    padding: 16px 20px; text-align: center; font-family: 'Courier New', monospace;
}
.kpi-value { font-size: 2rem; font-weight: bold; color: #00ff88; }
.kpi-label { font-size: 0.72rem; color: #6b7280; letter-spacing: 1px; margin-top: 4px; }

.section-pill {
    display: inline-block;
    background: linear-gradient(90deg, #003320, #001a10);
    border: 1px solid #00ff8844; border-radius: 20px;
    padding: 4px 16px; font-family: 'Courier New', monospace;
    font-size: 0.8rem; color: #00ff88; letter-spacing: 2px; margin-bottom: 12px;
}

.filter-panel {
    background: #0d1117; border: 1px solid #1f2937;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;
}
</style>
"""

# =========================================================================
# PLOTLY DARK TEMPLATE
# =========================================================================

PLOTLY_DARK = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Courier New"),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
)


# =========================================================================
# CANDLESTICK
# =========================================================================

def build_candlestick(row: pd.Series) -> go.Figure:
    cd = row.get("_chart_data")
    if not cd or not isinstance(cd, dict):
        return None

    dates  = cd.get("dates", [])
    opens  = cd.get("open",  [])
    highs  = cd.get("high",  [])
    lows   = cd.get("low",   [])
    closes = cd.get("close", [])
    vols   = cd.get("volume",[])
    ema20  = cd.get("ema20", [])
    ema50  = cd.get("ema50", [])
    bb_up  = cd.get("bb_up", [])
    bb_dn  = cd.get("bb_dn", [])

    if not dates or not closes:
        return None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color="#22c55e",
        increasing_fillcolor="rgba(34, 197, 94, 0.33)",
        decreasing_line_color="#ef4444",
        decreasing_fillcolor="rgba(239, 68, 68, 0.33)",
        name="Prezzo",
    ), row=1, col=1)

    if bb_up and bb_dn:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=bb_up + bb_dn[::-1],
            fill="toself",
            fillcolor="rgba(88,166,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bollinger Band", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=bb_up, line=dict(color="#58a6ff", width=1, dash="dot"),
            name="BB Up", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=bb_dn, line=dict(color="#58a6ff", width=1, dash="dot"),
            name="BB Dn", showlegend=False,
        ), row=1, col=1)

    if ema20:
        fig.add_trace(go.Scatter(
            x=dates, y=ema20, line=dict(color="#f59e0b", width=1.5), name="EMA20",
        ), row=1, col=1)
    if ema50:
        fig.add_trace(go.Scatter(
            x=dates, y=ema50, line=dict(color="#a78bfa", width=1.5), name="EMA50",
        ), row=1, col=1)

    if vols:
        colors_vol = ["rgba(0,255,136,0.4)" if c >= o else "rgba(239,68,68,0.4)"
                      for c, o in zip(closes, opens)]
        fig.add_trace(go.Bar(
            x=dates, y=vols, marker_color=colors_vol,
            name="Volume", showlegend=False,
        ), row=2, col=1)

    ticker   = row.get("Ticker", "")
    name_lbl = row.get("Nome", "")
    price    = row.get("Prezzo", "")
    rsi_val  = row.get("RSI", "")
    squeeze  = row.get("Squeeze", False)
    sq_lbl   = "  🔥 SQUEEZE" if squeeze else ""

    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(
            text=f"<b>{ticker}</b> — {name_lbl}  |  Prezzo: {price}  |  RSI: {rsi_val}{sq_lbl}",
            font=dict(color="#00ff88", size=14),
        ),
        height=500,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.update_yaxes(tickfont=dict(size=10), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=9),  row=2, col=1)
    return fig


# =========================================================================
# RADAR CHART
# =========================================================================

def build_radar(row: pd.Series) -> go.Figure:
    qc = row.get("_quality_components")
    if not qc or not isinstance(qc, dict):
        return None

    keys   = list(qc.keys())
    vals   = list(qc.values())
    categories = keys + [keys[0]]
    values     = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor="rgba(0,255,136,0.15)",
        line=dict(color="#00ff88", width=2),
        name="Quality",
    ))

    ticker = row.get("Ticker", "")
    qs     = row.get("Quality_Score", 0)

    fig.update_layout(
        **PLOTLY_DARK,
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=9, color="#6b7280"),
                            gridcolor="#1f2937", linecolor="#1f2937"),
            angularaxis=dict(tickfont=dict(size=11, color="#c9d1d9"),
                             gridcolor="#1f2937", linecolor="#1f2937"),
        ),
        title=dict(
            text=f"<b>{ticker}</b>  Quality Score: <b style='color:#00ff88'>{qs}/12</b>",
            font=dict(color="#58a6ff", size=13),
        ),
        height=380,
        margin=dict(l=40, r=40, t=60, b=20),
        showlegend=False,
    )
    return fig


# =========================================================================
# HELPER: mostra candlestick + radar per una riga
# =========================================================================

def show_charts(row_full: pd.Series, key_suffix: str = ""):
    st.markdown("---")
    c_left, c_right = st.columns([2, 1])
    ticker = row_full.get("Ticker", "")
    with c_left:
        st.markdown(f'<div class="section-pill">📊 CANDLESTICK — {ticker}</div>',
                    unsafe_allow_html=True)
        fig_c = build_candlestick(row_full)
        if fig_c:
            st.plotly_chart(fig_c, use_container_width=True, key=f"candle_{ticker}_{key_suffix}")
        else:
            st.info("Dati candlestick non disponibili.")
    with c_right:
        st.markdown(f'<div class="section-pill">🧭 QUALITY RADAR — {ticker}</div>',
                    unsafe_allow_html=True)
        fig_r = build_radar(row_full)
        if fig_r:
            st.plotly_chart(fig_r, use_container_width=True, key=f"radar_{ticker}_{key_suffix}")
        else:
            st.info("Dati radar non disponibili.")


# =========================================================================
# JS RENDERERS AGGRID
# =========================================================================

name_dblclick_renderer = JsCode("""
class NameDoubleClickRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        this.eGui.innerText = params.value || '';
        const ticker = params.data.Ticker || params.data.ticker;
        if (!ticker) return;
        this.eGui.style.cursor = 'pointer';
        this.eGui.title = 'Doppio click → TradingView';
        this.eGui.ondblclick = function() {
            window.open("https://www.tradingview.com/chart/?symbol=" + String(ticker).split(".")[0], "_blank");
        };
    }
    getGui() { return this.eGui; }
}
""")

rsi_renderer = JsCode("""
class RsiRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const v = parseFloat(params.value);
        this.eGui.innerText = isNaN(v) ? '-' : v.toFixed(1);
        this.eGui.style.fontWeight = 'bold';
        this.eGui.style.fontFamily = 'Courier New';
        if (v < 30)       { this.eGui.style.color = '#60a5fa'; }
        else if (v < 40)  { this.eGui.style.color = '#93c5fd'; }
        else if (v <= 65) { this.eGui.style.color = '#00ff88'; }
        else if (v <= 70) { this.eGui.style.color = '#f59e0b'; }
        else              { this.eGui.style.color = '#ef4444'; }
    }
    getGui() { return this.eGui; }
}
""")

vol_ratio_renderer = JsCode("""
class VolRatioRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const v = parseFloat(params.value);
        this.eGui.innerText = isNaN(v) ? '-' : v.toFixed(2) + 'x';
        this.eGui.style.fontFamily = 'Courier New';
        this.eGui.style.fontWeight = 'bold';
        if (v < 1.0)      { this.eGui.style.color = '#6b7280'; }
        else if (v < 2.0) { this.eGui.style.color = '#00ff88'; }
        else if (v < 3.0) { this.eGui.style.color = '#f59e0b'; }
        else              { this.eGui.style.color = '#ef4444';
                            this.eGui.style.textShadow = '0 0 6px #ef4444'; }
    }
    getGui() { return this.eGui; }
}
""")

early_score_renderer = JsCode("""
class EarlyScoreRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const v = parseFloat(params.value || 0);
        this.eGui.innerText = v.toFixed(1);
        this.eGui.style.fontFamily = 'Courier New';
        this.eGui.style.fontWeight = 'bold';
        if (v >= 8)      { this.eGui.style.color = '#00ff88'; this.eGui.style.textShadow = '0 0 8px #00ff88'; }
        else if (v >= 5) { this.eGui.style.color = '#f59e0b'; }
        else if (v > 0)  { this.eGui.style.color = '#9ca3af'; }
        else             { this.eGui.style.color = '#374151'; }
    }
    getGui() { return this.eGui; }
}
""")

quality_renderer = JsCode("""
class QualityRenderer {
    init(params) {
        this.eGui = document.createElement('div');
        this.eGui.style.display = 'flex';
        this.eGui.style.alignItems = 'center';
        this.eGui.style.gap = '6px';
        const v = parseInt(params.value || 0);
        const pct = Math.round((v / 12) * 100);
        let color = v >= 9 ? '#00ff88' : v >= 6 ? '#f59e0b' : '#6b7280';
        this.eGui.innerHTML = `
            <span style="font-family:Courier New;font-weight:bold;color:${color};min-width:20px">${v}</span>
            <div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
                <div style="width:${pct}%;background:${color};height:6px;border-radius:3px"></div>
            </div>`;
    }
    getGui() { return this.eGui; }
}
""")

squeeze_renderer = JsCode("""
class SqueezeRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const val = params.value;
        if (val === true || val === 'True' || val === 'true') {
            this.eGui.innerText = '🔥 SQ';
            this.eGui.style.color = '#f97316'; this.eGui.style.fontWeight = 'bold';
        } else {
            this.eGui.innerText = '—'; this.eGui.style.color = '#374151';
        }
    }
    getGui() { return this.eGui; }
}
""")

rsi_div_renderer = JsCode("""
class RsiDivRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const val = params.value;
        if (val === 'BEARISH')      { this.eGui.innerText = '⚠️ BEAR'; this.eGui.style.color = '#ef4444'; }
        else if (val === 'BULLISH') { this.eGui.innerText = '✅ BULL'; this.eGui.style.color = '#00ff88'; }
        else                        { this.eGui.innerText = '—';       this.eGui.style.color = '#374151'; }
    }
    getGui() { return this.eGui; }
}
""")

weekly_renderer = JsCode("""
class WeeklyRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const val = params.value;
        if (val === true || val === 'True' || val === 'true')
            { this.eGui.innerText = '📈 W+'; this.eGui.style.color = '#00ff88'; }
        else if (val === false || val === 'False' || val === 'false')
            { this.eGui.innerText = '📉 W—'; this.eGui.style.color = '#ef4444'; }
        else
            { this.eGui.innerText = '—'; this.eGui.style.color = '#374151'; }
    }
    getGui() { return this.eGui; }
}
""")

price_renderer = JsCode("""
class PriceRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const v = params.value;
        this.eGui.innerText = v !== undefined && v !== null ? v : '-';
        this.eGui.style.fontFamily = 'Courier New';
        this.eGui.style.color = '#e2e8f0'; this.eGui.style.fontWeight = 'bold';
    }
    getGui() { return this.eGui; }
}
""")


# =========================================================================
# EXPORT HELPERS
# =========================================================================

def to_excel_bytes(sheets_dict: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for name, df in sheets_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


def make_tv_csv(df: pd.DataFrame, tab_name: str) -> bytes:
    tmp = df[["Ticker"]].copy()
    tmp.insert(0, "Tab", tab_name)
    return tmp.to_csv(index=False).encode("utf-8")


def csv_btn(df, filename, key):
    st.download_button("📥 CSV", df.to_csv(index=False).encode(), filename, "text/csv", key=key)


# =========================================================================
# PRESET
# =========================================================================

PRESETS = {
    "⚡ Aggressivo":   dict(eh=0.01, prmin=45, prmax=65, rpoc=0.01, vol_ratio_hot=1.2, top=20,
                            min_early_score=3.0, min_quality=4, min_pro_score=3.0),
    "⚖️ Bilanciato":   dict(eh=0.02, prmin=40, prmax=70, rpoc=0.02, vol_ratio_hot=1.5, top=15,
                            min_early_score=5.0, min_quality=6, min_pro_score=5.0),
    "🛡️ Conservativo": dict(eh=0.04, prmin=35, prmax=75, rpoc=0.04, vol_ratio_hot=2.0, top=10,
                            min_early_score=7.0, min_quality=8, min_pro_score=7.0),
    "🔓 Nessun Filtro": dict(eh=0.05, prmin=20, prmax=85, rpoc=0.05, vol_ratio_hot=0.5, top=50,
                             min_early_score=0.0, min_quality=0, min_pro_score=0.0),
}


# =========================================================================
# PAGE CONFIG
# =========================================================================

st.set_page_config(page_title="Trading Scanner PRO 23.0", layout="wide", page_icon="🧠")
st.markdown(DARK_CSS, unsafe_allow_html=True)

st.markdown("# 🧠 Trading Scanner PRO 23.0")
st.markdown(
    '<div class="section-pill">DARK · CANDLESTICK · RADAR · COLORED CELLS · CONFLUENCE · v23.0</div>',
    unsafe_allow_html=True,
)

init_db()

# =========================================================================
# SESSION STATE DEFAULTS
# =========================================================================

defaults = dict(
    mSP500=True, mNasdaq=True, mFTSE=True, mEurostoxx=False,
    mDow=False, mRussell=False, mStoxxEmerging=False, mUSSmallCap=False,
    eh=0.02, prmin=40, prmax=70, rpoc=0.02, vol_ratio_hot=1.5, top=15,
    min_early_score=3.0, min_quality=3, min_pro_score=3.0,
    current_list_name="DEFAULT", last_active_tab="EARLY",
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# =========================================================================
# KPI BAR
# =========================================================================

def render_kpi_bar(df_ep: pd.DataFrame, df_rea: pd.DataFrame):
    hist = load_scan_history(2)
    prev_early = prev_pro = prev_hot = prev_conf = 0
    if len(hist) >= 2:
        prev_row   = hist.iloc[1]
        prev_early = int(prev_row.get("n_early", 0))
        prev_pro   = int(prev_row.get("n_pro",   0))
        prev_hot   = int(prev_row.get("n_rea",   0))
        prev_conf  = int(prev_row.get("n_confluence", 0))

    n_early = int((df_ep.get("Stato_Early", pd.Series()) == "EARLY").sum()) if not df_ep.empty else 0
    n_pro   = int((df_ep.get("Stato_Pro",   pd.Series()) == "PRO"  ).sum()) if not df_ep.empty else 0
    n_hot   = len(df_rea) if not df_rea.empty else 0
    n_conf  = 0
    if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
        n_conf = int(((df_ep["Stato_Early"] == "EARLY") & (df_ep["Stato_Pro"] == "PRO")).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📡 EARLY",     n_early, delta=n_early - prev_early if prev_early else None)
    k2.metric("💪 PRO",       n_pro,   delta=n_pro   - prev_pro   if prev_pro   else None)
    k3.metric("🔥 REA-HOT",   n_hot,   delta=n_hot   - prev_hot   if prev_hot   else None)
    k4.metric("⭐ CONFLUENCE", n_conf,  delta=n_conf  - prev_conf  if prev_conf  else None)


# =========================================================================
# SIDEBAR
# =========================================================================

st.sidebar.title("⚙️ Configurazione")

with st.sidebar.expander("🎯 Preset Rapidi", expanded=False):
    for pname, pvals in PRESETS.items():
        if st.button(pname, use_container_width=True, key=f"preset_{pname}"):
            for k, v in pvals.items():
                st.session_state[k] = v
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

sel = []
for flag, mkt in [
    (msp500,"SP500"), (mnasdaq,"Nasdaq"), (mftse,"FTSE"), (meuro,"Eurostoxx"),
    (mdow,"Dow"), (mrussell,"Russell"), (mstoxxem,"StoxxEmerging"), (mussmall,"USSmallCap"),
]:
    if flag: sel.append(mkt)

(st.session_state.mSP500, st.session_state.mNasdaq, st.session_state.mFTSE,
 st.session_state.mEurostoxx, st.session_state.mDow, st.session_state.mRussell,
 st.session_state.mStoxxEmerging, st.session_state.mUSSmallCap) = (
    msp500, mnasdaq, mftse, meuro, mdow, mrussell, mstoxxem, mussmall)

with st.sidebar.expander("🎛️ Parametri Scanner", expanded=False):
    eh            = st.slider("EARLY EMA20 %",   0.0, 10.0, float(st.session_state.eh*100), 0.5) / 100
    prmin         = st.slider("PRO RSI min",      0, 100, int(st.session_state.prmin), 5)
    prmax         = st.slider("PRO RSI max",      0, 100, int(st.session_state.prmax), 5)
    rpoc          = st.slider("REA POC %",        0.0, 10.0, float(st.session_state.rpoc*100), 0.5) / 100
    vol_ratio_hot = st.number_input("VolRatio HOT", 0.0, 10.0, float(st.session_state.vol_ratio_hot), 0.1)
    top           = st.number_input("TOP N",       5, 200, int(st.session_state.top), 5)

(st.session_state.eh, st.session_state.prmin, st.session_state.prmax,
 st.session_state.rpoc, st.session_state.vol_ratio_hot, st.session_state.top) = (
    eh, prmin, prmax, rpoc, vol_ratio_hot, top)

# ── SOGLIE FILTRI (nuove, in tempo reale) ──────────────────────────────────
with st.sidebar.expander("🔬 Soglie Filtri", expanded=True):
    st.caption("Abbassa le soglie per vedere più segnali")
    min_early_score = st.slider(
        "Early Score minimo",  0.0, 10.0,
        float(st.session_state.min_early_score), 0.5,
        help="0 = mostra tutti i ticker EARLY senza filtro score",
    )
    min_quality = st.slider(
        "Quality Score minimo", 0, 12,
        int(st.session_state.min_quality), 1,
        help="0 = mostra tutti senza filtro quality",
    )
    min_pro_score = st.slider(
        "Pro Score minimo", 0.0, 10.0,
        float(st.session_state.min_pro_score), 0.5,
        help="0 = mostra tutti i ticker PRO senza filtro score",
    )
    st.session_state.min_early_score = min_early_score
    st.session_state.min_quality     = min_quality
    st.session_state.min_pro_score   = min_pro_score

st.sidebar.divider()
st.sidebar.subheader("📋 Watchlist")

df_wl_all    = load_watchlist()
list_options = sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
if "DEFAULT" not in list_options: list_options.append("DEFAULT")

active_list = st.sidebar.selectbox(
    "Lista Attiva", list_options,
    index=list_options.index(st.session_state.current_list_name)
    if st.session_state.current_list_name in list_options else 0,
    key="active_list",
)
st.session_state.current_list_name = active_list

new_list = st.sidebar.text_input("Crea lista")
if st.sidebar.button("Crea") and new_list.strip():
    st.session_state.current_list_name = new_list.strip()
    st.rerun()

if st.sidebar.button("⚠️ Reset Watchlist DB"):
    reset_watchlist_db(); st.rerun()

# Reset storico nella sidebar
st.sidebar.divider()
if st.sidebar.button("🗑️ Reset Storico Scansioni", key="reset_hist_sidebar"):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM scan_history")
        conn.commit(); conn.close()
        st.sidebar.success("Storico cancellato.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Errore: {e}")

only_watchlist = st.sidebar.checkbox("Solo Watchlist", False)


# =========================================================================
# SCANNER
# =========================================================================

if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 23.0", type="primary", use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            rep, rrea = [], []
            pb     = st.progress(0)
            status = st.empty()
            tot    = len(universe)
            for i, tkr in enumerate(universe, 1):
                status.text(f"Analisi {i}/{tot}: {tkr}")
                ep, rea = scan_ticker(tkr, eh, prmin, prmax, rpoc, vol_ratio_hot)
                if ep:  rep.append(ep)
                if rea: rrea.append(rea)
                pb.progress(i / tot)

            df_ep_new  = pd.DataFrame(rep)
            df_rea_new = pd.DataFrame(rrea)
            st.session_state.df_ep    = df_ep_new
            st.session_state.df_rea   = df_rea_new
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
            save_scan_history(sel, df_ep_new, df_rea_new)

            n_hot  = len(df_rea_new)
            n_conf = 0
            if not df_ep_new.empty and "Stato_Early" in df_ep_new.columns:
                n_conf = int(((df_ep_new["Stato_Early"] == "EARLY") &
                              (df_ep_new["Stato_Pro"]   == "PRO"  )).sum())
            if n_hot  >= 5: st.toast(f"🔥 {n_hot} segnali HOT!", icon="🔥")
            if n_conf >= 3: st.toast(f"⭐ {n_conf} CONFLUENCE!", icon="⭐")
            st.rerun()

df_ep  = st.session_state.get("df_ep",  pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

if "last_scan" in st.session_state:
    st.caption(f"⏱️ Ultima scansione: {st.session_state.last_scan}")

render_kpi_bar(df_ep, df_rea)
st.markdown("---")


# =========================================================================
# AGGRID BUILDER — auto-fit colonne
# =========================================================================

def build_aggrid(df_disp: pd.DataFrame, grid_key: str, height: int = 520):
    gb = GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(
        sortable=True, resizable=True, filterable=True, editable=False,
        autoHeight=False, wrapText=False,
    )
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)

    # Larghezze compatte per auto-fit visivo
    col_widths = {
        "Ticker": 80, "Nome": 160,
        "Prezzo_fmt": 90, "MarketCap_fmt": 100,
        "Early_Score": 95, "Pro_Score": 80, "Quality_Score": 120,
        "RSI": 65, "Vol_Ratio": 80,
        "Squeeze": 70, "RSI_Div": 80, "Weekly_Bull": 75,
        "Stato_Early": 80, "Stato_Pro": 75,
    }
    for col, w in col_widths.items():
        if col in df_disp.columns:
            gb.configure_column(col, width=w)

    renderer_map = {
        "Nome":          name_dblclick_renderer,
        "RSI":           rsi_renderer,
        "Vol_Ratio":     vol_ratio_renderer,
        "Early_Score":   early_score_renderer,
        "Quality_Score": quality_renderer,
        "Squeeze":       squeeze_renderer,
        "RSI_Div":       rsi_div_renderer,
        "Weekly_Bull":   weekly_renderer,
        "Prezzo_fmt":    price_renderer,
    }
    for col, renderer in renderer_map.items():
        if col in df_disp.columns:
            gb.configure_column(col, cellRenderer=renderer)

    # Pinned columns: Ticker e Nome sempre visibili
    if "Ticker" in df_disp.columns:
        gb.configure_column("Ticker", pinned="left")
    if "Nome" in df_disp.columns:
        gb.configure_column("Nome", pinned="left")

    go_opts = gb.build()
    # Auto-size all columns on first render
    go_opts["onFirstDataRendered"] = JsCode("""
        function(params) { params.api.sizeColumnsToFit(); }
    """)

    return AgGrid(
        df_disp,
        gridOptions=go_opts,
        height=height,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        allow_unsafe_jscode=True,
        key=grid_key,
    )


# =========================================================================
# RENDER TAB — con filtri soglia in tempo reale
# =========================================================================

def render_scan_tab(df: pd.DataFrame, status_filter: str, sort_cols, ascending, title: str):
    if df.empty:
        st.info(f"Nessun dato {title}. Esegui lo scanner.")
        return

    # ── Badge soglie attive ──────────────────────────────────────────────
    s_early = st.session_state.min_early_score
    s_qual  = st.session_state.min_quality
    s_pro   = st.session_state.min_pro_score
    st.caption(
        f"🔬 Soglie attive → Early Score ≥ **{s_early}** | "
        f"Quality ≥ **{s_qual}** | Pro Score ≥ **{s_pro}**  "
        f"_(modifica nella sidebar → 🔬 Soglie Filtri)_"
    )

    # ── Filtro stato ─────────────────────────────────────────────────────
    if status_filter == "EARLY" and "Stato_Early" in df.columns:
        df_f = df[df["Stato_Early"] == "EARLY"].copy()
        # Applica soglia Early_Score
        if "Early_Score" in df_f.columns and s_early > 0:
            df_f = df_f[df_f["Early_Score"] >= s_early]

    elif status_filter == "PRO" and "Stato_Pro" in df.columns:
        df_f = df[df["Stato_Pro"] == "PRO"].copy()
        if "Pro_Score" in df_f.columns and s_pro > 0:
            df_f = df_f[df_f["Pro_Score"] >= s_pro]
        if "Quality_Score" in df_f.columns and s_qual > 0:
            df_f = df_f[df_f["Quality_Score"] >= s_qual]

    elif status_filter == "HOT" and "Stato" in df.columns:
        df_f = df[df["Stato"] == "HOT"].copy()

    elif status_filter == "CONFLUENCE":
        if "Stato_Early" in df.columns and "Stato_Pro" in df.columns:
            df_f = df[(df["Stato_Early"] == "EARLY") & (df["Stato_Pro"] == "PRO")].copy()
            if "Early_Score" in df_f.columns and s_early > 0:
                df_f = df_f[df_f["Early_Score"] >= s_early]
            if "Quality_Score" in df_f.columns and s_qual > 0:
                df_f = df_f[df_f["Quality_Score"] >= s_qual]
        else:
            df_f = pd.DataFrame()

    elif status_filter == "REGIME":
        sp   = df.get("Stato_Pro")
        df_f = df[sp == "PRO"].copy() if sp is not None else df.copy()
        if "Pro_Score" in df_f.columns and s_pro > 0:
            df_f = df_f[df_f["Pro_Score"] >= s_pro]
        if "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
            df_f["Momentum"] = df_f["Pro_Score"] * 10 + df_f["RSI"]
            sort_cols = ["Momentum"]; ascending = [False]

    elif status_filter == "MTF":
        sp   = df.get("Stato_Pro")
        df_f = df[sp == "PRO"].copy() if sp is not None else df.copy()
        if "Weekly_Bull" in df_f.columns:
            df_f = df_f[df_f["Weekly_Bull"] == True]

    else:
        df_f = df.copy()

    if df_f.empty:
        st.warning(
            f"⚠️ Nessun segnale **{title}** trovato con le soglie attuali. "
            "Prova ad abbassare le soglie nella sidebar → 🔬 Soglie Filtri."
        )
        return

    # ── Ordinamento ──────────────────────────────────────────────────────
    valid_sort = [c for c in sort_cols if c in df_f.columns]
    if valid_sort:
        df_f = df_f.sort_values(valid_sort, ascending=ascending[:len(valid_sort)])
    df_f = df_f.head(int(st.session_state.top))

    # ── Metriche rapide ──────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Titoli trovati", len(df_f))
    if "Squeeze" in df_f.columns:
        m2.metric("🔥 Squeeze", int(df_f["Squeeze"].apply(
            lambda x: x is True or str(x).lower() == "true").sum()))
    if "Weekly_Bull" in df_f.columns:
        m3.metric("📈 Weekly+", int(df_f["Weekly_Bull"].apply(
            lambda x: x is True or str(x).lower() == "true").sum()))
    if "RSI_Div" in df_f.columns:
        m4.metric("⚠️ Div RSI", int((df_f["RSI_Div"] != "-").sum()))

    # ── Formatta per AgGrid ───────────────────────────────────────────────
    df_fmt  = add_formatted_cols(df_f)
    df_disp = prepare_display_df(df_fmt)

    cols      = list(df_disp.columns)
    base_cols = [c for c in ["Ticker", "Nome"] if c in cols]
    df_disp   = df_disp[base_cols + [c for c in cols if c not in base_cols]]
    df_disp   = df_disp.reset_index(drop=True)

    # ── Export ───────────────────────────────────────────────────────────
    ce1, ce2 = st.columns([1, 3])
    with ce1:
        csv_btn(df_f, f"{title.lower().replace(' ','_')}.csv", f"exp_{title}")
    with ce2:
        st.caption(
            f"Seleziona righe → **Aggiungi** alla lista `{st.session_state.current_list_name}`. "
            "Doppio click su Nome → TradingView."
        )

    # ── AgGrid ───────────────────────────────────────────────────────────
    grid_resp   = build_aggrid(df_disp, f"grid_{title}")
    selected_df = pd.DataFrame(grid_resp["selected_rows"])

    if st.button(f"➕ Aggiungi a '{st.session_state.current_list_name}'", key=f"btn_{title}"):
        if not selected_df.empty and "Ticker" in selected_df.columns:
            tickers = selected_df["Ticker"].tolist()
            names   = selected_df.get("Nome", selected_df["Ticker"]).tolist()
            add_to_watchlist(tickers, names, title, "Scanner", "LONG",
                             st.session_state.current_list_name)
            st.success(f"✅ Aggiunti {len(tickers)} titoli.")
            time.sleep(0.8); st.rerun()
        else:
            st.warning("Nessuna riga selezionata.")

    # ── Candlestick + Radar ───────────────────────────────────────────────
    if not selected_df.empty:
        ticker_sel = selected_df.iloc[0].get("Ticker", "")
        row_match  = df_f[df_f["Ticker"] == ticker_sel]
        if not row_match.empty:
            show_charts(row_match.iloc[0], key_suffix=title)


# =========================================================================
# LEGENDA
# =========================================================================

def show_legend(title: str):
    with st.expander(f"📖 Legenda {title}", expanded=False):
        legends = {
            "EARLY":           "Titoli vicini alla EMA20. **Early_Score 0–10** continuo (verde ≥8).",
            "PRO":             "Trend + RSI neutrale-rialzista + volume sopra media. **Quality_Score** 0–12.",
            "REA-HOT":         "Volumi anomali vicini al POC. Vol_Ratio = volume odierno / media 20gg.",
            "⭐ CONFLUENCE":   "Titoli che soddisfano **sia EARLY sia PRO** contemporaneamente.",
            "Regime Momentum": "PRO ordinati per Momentum = Pro_Score×10 + RSI.",
            "Multi-Timeframe": "PRO con trend settimanale rialzista (Weekly_Bull = 📈 W+).",
            "Finviz":          "PRO filtrati per MarketCap ≥ mediana e Vol_Ratio > 1.2.",
        }
        st.markdown(legends.get(title, f"Segnali scanner **{title}**."))


# =========================================================================
# TABS
# =========================================================================

tabs = st.tabs([
    "EARLY", "PRO", "REA-HOT", "⭐ CONFLUENCE",
    "Regime Momentum", "Multi-Timeframe", "Finviz",
    "📋 Watchlist", "📜 Storico",
])
(tab_e, tab_p, tab_r, tab_conf, tab_regime, tab_mtf,
 tab_finviz, tab_w, tab_hist) = tabs

with tab_e:
    st.session_state.last_active_tab = "EARLY"
    show_legend("EARLY")
    render_scan_tab(df_ep, "EARLY", ["Early_Score", "RSI"], [False, True], "EARLY")

with tab_p:
    st.session_state.last_active_tab = "PRO"
    show_legend("PRO")
    render_scan_tab(df_ep, "PRO", ["Quality_Score", "Pro_Score", "RSI"], [False, False, True], "PRO")

with tab_r:
    st.session_state.last_active_tab = "REA-HOT"
    show_legend("REA-HOT")
    render_scan_tab(df_rea, "HOT", ["Vol_Ratio", "Dist_POC_%"], [False, True], "REA-HOT")

with tab_conf:
    st.session_state.last_active_tab = "CONFLUENCE"
    show_legend("⭐ CONFLUENCE")
    render_scan_tab(df_ep, "CONFLUENCE",
                    ["Quality_Score", "Early_Score", "Pro_Score"], [False, False, False],
                    "CONFLUENCE")

with tab_regime:
    show_legend("Regime Momentum")
    render_scan_tab(df_ep, "REGIME", ["Pro_Score"], [False], "Regime Momentum")

with tab_mtf:
    show_legend("Multi-Timeframe")
    render_scan_tab(df_ep, "MTF", ["Quality_Score", "Pro_Score"], [False, False], "Multi-Timeframe")

with tab_finviz:
    show_legend("Finviz")
    sp    = df_ep.get("Stato_Pro")
    df_fv = df_ep[sp == "PRO"].copy() if sp is not None and not df_ep.empty else df_ep.copy()
    if not df_fv.empty:
        if "MarketCap" in df_fv.columns:
            df_fv = df_fv[df_fv["MarketCap"] >= df_fv["MarketCap"].median()]
        if "Vol_Ratio" in df_fv.columns:
            df_fv = df_fv[df_fv["Vol_Ratio"] > 1.2]
    render_scan_tab(df_fv, "PRO", ["Quality_Score", "Pro_Score"], [False, False], "Finviz")


# =========================================================================
# WATCHLIST  (con candlestick + radar)
# =========================================================================

with tab_w:
    st.markdown(
        f'<div class="section-pill">📋 WATCHLIST — {st.session_state.current_list_name}</div>',
        unsafe_allow_html=True,
    )
    df_wl = load_watchlist()
    df_wl = df_wl[df_wl["list_name"] == st.session_state.current_list_name]

    if df_wl.empty:
        st.info("Watchlist vuota.")
    else:
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            csv_btn(df_wl, f"watchlist_{st.session_state.current_list_name}.csv", "exp_wl")
        with wc2:
            move_tgt = st.selectbox("Sposta in", list_options, key="move_tgt")
            ids_move = st.multiselect("ID da spostare", df_wl["id"].tolist(), key="ids_move")
            if st.button("Sposta"):
                move_watchlist_rows(ids_move, move_tgt); st.rerun()
        with wc3:
            ids_del = st.multiselect("ID da eliminare", df_wl["id"].tolist(), key="ids_del")
            if st.button("🗑️ Elimina"):
                delete_from_watchlist(ids_del); st.rerun()
        with wc4:
            new_name = st.text_input("Rinomina lista", key="ren_wl")
            if st.button("✏️ Rinomina") and new_name.strip():
                rename_watchlist(st.session_state.current_list_name, new_name.strip())
                st.session_state.current_list_name = new_name.strip(); st.rerun()

        df_wv = add_links(prepare_display_df(add_formatted_cols(df_wl)))
        st.write(df_wv.to_html(escape=False, index=False), unsafe_allow_html=True)

        # ── Grafici per ticker selezionato dalla watchlist ────────────────
        st.markdown("---")
        st.markdown('<div class="section-pill">📊 ANALISI TICKER WATCHLIST</div>',
                    unsafe_allow_html=True)

        tickers_wl = df_wl["ticker"].dropna().unique().tolist() if "ticker" in df_wl.columns \
                     else df_wl["Ticker"].dropna().unique().tolist() if "Ticker" in df_wl.columns \
                     else []

        if tickers_wl:
            sel_wl_ticker = st.selectbox(
                "Seleziona ticker per analisi", tickers_wl, key="wl_ticker_sel"
            )

            # Cerca il ticker nei dati scanner se disponibili
            row_wl = None
            if not df_ep.empty and "Ticker" in df_ep.columns:
                match = df_ep[df_ep["Ticker"] == sel_wl_ticker]
                if not match.empty:
                    row_wl = match.iloc[0]
            if row_wl is None and not df_rea.empty and "Ticker" in df_rea.columns:
                match = df_rea[df_rea["Ticker"] == sel_wl_ticker]
                if not match.empty:
                    row_wl = match.iloc[0]

            if row_wl is not None:
                show_charts(row_wl, key_suffix="watchlist")
            else:
                st.info(
                    f"Nessun dato di analisi disponibile per **{sel_wl_ticker}**. "
                    "Esegui lo scanner per caricare i dati grafici."
                )
        else:
            st.info("Nessun ticker in watchlist.")

    if st.button("🔄 Refresh Watchlist"): st.rerun()


# =========================================================================
# STORICO  (con pulsante reset)
# =========================================================================

with tab_hist:
    st.markdown('<div class="section-pill">📜 STORICO SCANSIONI</div>', unsafe_allow_html=True)

    # Pulsante reset storico anche qui
    col_title, col_reset = st.columns([4, 1])
    with col_reset:
        if st.button("🗑️ Reset Storico", key="reset_hist_tab", type="secondary"):
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM scan_history")
                conn.commit(); conn.close()
                st.success("Storico cancellato!")
                st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")

    df_hist = load_scan_history(20)

    if df_hist.empty:
        st.info("Nessuna scansione salvata. Esegui lo scanner.")
    else:
        st.dataframe(df_hist, use_container_width=True)
        st.markdown("---")
        st.subheader("🔍 Confronto Snapshot")
        hc1, hc2 = st.columns(2)
        with hc1: id_a = st.selectbox("Scansione A", df_hist["id"].tolist(), key="snap_a")
        with hc2: id_b = st.selectbox("Scansione B", df_hist["id"].tolist(),
                                       index=min(1, len(df_hist)-1), key="snap_b")
        if st.button("🔍 Confronta"):
            ep_a, _ = load_scan_snapshot(id_a)
            ep_b, _ = load_scan_snapshot(id_b)
            if ep_a.empty or ep_b.empty:
                st.warning("Dati non disponibili.")
            else:
                ta = set(ep_a.get("Ticker", [])); tb = set(ep_b.get("Ticker", []))
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🆕 Nuovi",      len(tb - ta))
                sc2.metric("❌ Usciti",      len(ta - tb))
                sc3.metric("✅ Persistenti", len(ta & tb))
                if tb - ta: st.markdown("**🆕 Nuovi:** "   + ", ".join(sorted(tb - ta)))
                if ta - tb: st.markdown("**❌ Usciti:** "  + ", ".join(sorted(ta - tb)))


# =========================================================================
# EXPORT GLOBALI
# =========================================================================

st.markdown("---")
st.markdown('<div class="section-pill">💾 EXPORT GLOBALI</div>', unsafe_allow_html=True)

df_conf_exp = pd.DataFrame()
if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
    df_conf_exp = df_ep[
        (df_ep["Stato_Early"] == "EARLY") & (df_ep["Stato_Pro"] == "PRO")
    ].copy()

df_wl_exp = load_watchlist()
df_wl_exp = df_wl_exp[df_wl_exp["list_name"] == st.session_state.current_list_name]

all_tabs_export = {
    "EARLY": df_ep, "PRO": df_ep, "REA-HOT": df_rea,
    "CONFLUENCE": df_conf_exp, "Watchlist": df_wl_exp,
}

ec1, ec2, ec3, ec4 = st.columns(4)
current_tab = st.session_state.get("last_active_tab", "EARLY")
df_current  = all_tabs_export.get(current_tab, pd.DataFrame())

with ec1:
    st.download_button(
        "📊 XLSX Tutti", to_excel_bytes(all_tabs_export),
        "TradingScanner_v23_Tutti.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xlsx_all",
    )
with ec2:
    tv_rows = []
    for n, df_t in all_tabs_export.items():
        if isinstance(df_t, pd.DataFrame) and not df_t.empty and "Ticker" in df_t.columns:
            tickers = df_t["Ticker"].tolist()
            tv_rows.append(pd.DataFrame({"Tab": [n] * len(tickers), "Ticker": tickers}))
    if tv_rows:
        df_tv = pd.concat(tv_rows, ignore_index=True).drop_duplicates(subset=["Ticker"])
        st.download_button(
            "📈 CSV TradingView Tutti", df_tv.to_csv(index=False).encode(),
            "TradingScanner_v23_TV.csv", "text/csv", key="csv_tv_all",
        )
with ec3:
    st.download_button(
        f"📊 XLSX {current_tab}",
        to_excel_bytes({current_tab: df_current}),
        f"TradingScanner_v23_{current_tab}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xlsx_curr",
    )
with ec4:
    if not df_current.empty and "Ticker" in df_current.columns:
        st.download_button(
            f"📈 CSV TV {current_tab}",
            make_tv_csv(df_current, current_tab),
            f"TradingScanner_v23_{current_tab}_TV.csv",
            "text/csv", key="csv_tv_curr",
        )
