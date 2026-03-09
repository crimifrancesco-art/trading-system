"""
Dashboard_pro-V_29.0.py — Trading Scanner PRO 29.0
Upgrade completo: Crisi&Inflazione + checkbox raw + persistenza AgGrid + grafici avanzati
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import json
import sqlite3
from datetime import datetime

# Import moduli locali
try:
    from utils import scanner, formatting, db  # scanner.py, formatting.py, db.py
    from utils.backtest_tab import render_backtest_tab
    MODULES_OK = True
except ImportError:
    st.error("❌ Moduli utils non trovati. Assicurati di avere scanner.py, formatting.py, db.py, backtest_tab.py")
    MODULES_OK = False

# ── PLOTLY DARK THEME (unificato) ──────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="#050812",
    plot_bgcolor="#0d1117",
    font=dict(color="#e5e7eb", family="Courier New, monospace"),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
)

# ── INIT SESSION STATE ─────────────────────────────────────────────────────
if "df_ep" not in st.session_state:
    st.session_state.df_ep = pd.DataFrame()
if "df_rea" not in st.session_state:
    st.session_state.df_rea = pd.DataFrame()
if "last_scan" not in st.session_state:
    st.session_state.last_scan = ""
if "current_list_name" not in st.session_state:
    st.session_state.current_list_name = "DEFAULT"
if "min_early_score" not in st.session_state:
    st.session_state.min_early_score = 5.0
if "min_pro_score" not in st.session_state:
    st.session_state.min_pro_score = 4.0
if "min_quality" not in st.session_state:
    st.session_state.min_quality = 6
if "top" not in st.session_state:
    st.session_state.top = 50
if "aggrid_state" not in st.session_state:
    st.session_state.aggrid_state = {}
if "last_active_tab" not in st.session_state:
    st.session_state.last_active_tab = "EARLY"

# ── INIT DB e load layouts ─────────────────────────────────────────────────
if MODULES_OK:
    db.init_db()
    # Carica layout AgGrid persistenti
    try:
        conn = sqlite3.connect(db.DB_PATH)
        layout_df = pd.read_sql_query(
            "CREATE TABLE IF NOT EXISTS grid_layouts (grid_key TEXT PRIMARY KEY, column_state TEXT);"
            "SELECT * FROM grid_layouts;", conn)
        for _, row in layout_df.iterrows():
            st.session_state.aggrid_state[row["grid_key"]] = json.loads(row["column_state"])
        conn.close()
    except:
        pass

# ── CONFIG PAGE ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Scanner PRO 29.0",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background-color: #050812; padding: 1rem; border-radius: 10px; border-left: 5px solid #00ff88;'>
<h1 style='color: #e5e7eb; margin: 0;'>📡 Trading Scanner PRO 29.0</h1>
<p style='color: #9ca3af; margin: 0;'>Upgrade: Crisi&Inflazione + Checkbox Raw + AgGrid Persistente + Volume Profile + Ichimoku</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR SCANNER ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚀 Scanner")
    markets = st.multiselect(
        "Mercati", ["SP500", "Nasdaq", "Dow", "Russell", "USSmallCap", "FTSE", "Eurostoxx", "StoxxEmerging"],
        default=["SP500", "Nasdaq", "FTSE"]
    )
    
    st.markdown("### 🔬 Soglie")
    st.session_state.min_early_score = st.slider("Min Early_Score", 0.0, 10.0, 5.0)
    st.session_state.min_pro_score = st.slider("Min Pro_Score", 0.0, 10.0, 4.0)
    st.session_state.min_quality = st.slider("Min Quality", 0, 12, 6)
    st.session_state.top = st.slider("Top N risultati", 10, 200, 50)
    
    if st.button("🔍 AVVIA SCAN", type="primary", use_container_width=True):
        with st.spinner("Scanning..."):
            df_ep, df_rea, stats = scanner.scan_universe(
                scanner.load_universe(markets),
                e_h=st.session_state.min_early_score,
                p_rmin=st.session_state.min_pro_score,
                p_rmax=100-st.session_state.min_pro_score,
                r_poc=0.02,
                vol_ratio_hot=1.5
            )
            # SALVA NEL DB (29.0)
            scan_id = db.save_scan_history(markets, df_ep, df_rea, stats["elapsed_s"], stats["cache_hits"])
            db.save_signals(scan_id, df_ep, df_rea, markets)
            
            st.session_state.df_ep = df_ep
            st.session_state.df_rea = df_rea
            st.session_state.last_scan = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        st.success(f"✅ Scan completato in {stats['elapsed_s']}s | {stats['ep_found']} EARLY | {stats['rea_found']} HOT")
        st.rerun()

# ── RENDER FUNZIONI UTILITY ─────────────────────────────────────────────────
def build_aggrid(df, grid_key):
    """AgGrid con persistenza colonne (29.0)"""
    column_state = st.session_state.aggrid_state.get(grid_key)
    grid_options = {
        "columnState": column_state,
        "enableRangeSelection": True,
        "animateRows": True,
        "defaultColDef": {"resizable": True, "sortable": True}
    }
    
    response = AgGrid(df, gridOptions=grid_options, key=grid_key)
    
    # Salva stato aggiornato
    if "columnState" in response["grid_state"]:
        st.session_state.aggrid_state[grid_key] = response["grid_state"]["columnState"]
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("💾 Salva layout", key=f"save_{grid_key}"):
            conn = sqlite3.connect(db.DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO grid_layouts (grid_key, column_state) VALUES (?, ?)",
                (grid_key, json.dumps(st.session_state.aggrid_state[grid_key]))
            )
            conn.commit()
            conn.close()
            st.success("✅ Layout salvato!")
    with col2:
        if st.button("♻️ Reset layout", key=f"reset_{grid_key}"):
            st.session_state.aggrid_state.pop(grid_key, None)
            st.rerun()
    
    return response

def render_scan_tab(df, status_filter, sort_cols, ascending, title, mode_raw=False):
    """Render tab con checkbox raw mode (29.0)"""
    if df.empty:
        st.info(f"📭 Nessun dato in **{title}**. Avvia scanner.")
        return
    
    s_e = st.session_state.min_early_score
    s_q = st.session_state.min_quality
    s_p = st.session_state.min_pro_score
    st.caption(f"🔬 Soglie globali: Early≥{s_e} | Quality≥{s_q} | Pro≥{s_p}")
    
    # CHECKBOX RAW MODE (29.0)
    raw_key = f"chk_raw_{status_filter.lower()}"
    raw_label = {
        "EARLY": "Mostra tutti EARLY (ignora Early_Score)",
        "PRO": "Mostra tutti PRO (ignora Pro/Quality)",
        "HOT": "Applica filtro severo Vol/POC",
        "CONFLUENCE": "Mostra tutti CONFLUENCE (ignora soglie)",
        "MTF": "Mostra tutti PRO+Weekly (ignora Pro_Score)",
        "SERAFINI": "Applica Quality≥min_quality",
        "FINVIZ_PRO": "Applica Quality≥min_quality"
    }.get(status_filter, "Raw mode")
    
    mode_raw = st.checkbox(raw_label, value=mode_raw, key=raw_key)
    if mode_raw:
        st.caption("*Raw mode attivo: ignorate soglie globali punteggio*")
    
    # LOGICA FILTRI (estesa per raw mode)
    if status_filter == "EARLY":
        df_f = df[df["Stato_Early"] == "EARLY"].copy()
        if not mode_raw and "Early_Score" in df_f.columns and s_e > 0:
            df_f = df_f[df_f["Early_Score"] >= s_e]
    
    elif status_filter == "PRO":
        df_f = df[df["Stato_Pro"] == "PRO"].copy()
        if not mode_raw:
            if "Pro_Score" in df_f.columns and s_p > 0:
                df_f = df_f[df_f["Pro_Score"] >= s_p]
            if "Quality_Score" in df_f.columns and s_q > 0:
                df_f = df_f[df_f["Quality_Score"] >= s_q]
    
    elif status_filter == "HOT":
        df_f = df[df["Stato"] == "HOT"].copy()
    
    elif status_filter == "CONFLUENCE":
        df_f = df[(df["Stato_Early"] == "EARLY") & (df["Stato_Pro"] == "PRO")].copy()
        if not mode_raw:
            if "Early_Score" in df_f.columns and s_e > 0:
                df_f = df_f[df_f["Early_Score"] >= s_e]
            if "Quality_Score" in df_f.columns and s_q > 0:
                df_f = df_f[df_f["Quality_Score"] >= s_q]
    
    elif status_filter == "MTF":
        df_f = df[df["Stato_Pro"] == "PRO"].copy()
        if "Weekly_Bull" in df_f.columns:
            df_f = df_f[df_f["Weekly_Bull"].isin([True, "True", "true", 1])]
        if not mode_raw and "Pro_Score" in df_f.columns and s_p > 0:
            df_f = df_f[df_f["Pro_Score"] >= s_p]
    
    elif status_filter == "SERAFINI":
        df_f = df[df["Ser_OK"].isin([True, "True", "true"])].copy()
        if not mode_raw and "Quality_Score" in df_f.columns and s_q > 0:
            df_f = df_f[df_f["Quality_Score"] >= s_q]
    
    elif status_filter == "FINVIZ_PRO":
        df_f = df[df["FV_OK"].isin([True, "True", "true"])].copy()
        if not mode_raw and "Quality_Score" in df_f.columns and s_q > 0:
            df_f = df_f[df_f["Quality_Score"] >= s_q]
    
    else:
        df_f = df.copy()
    
    if df_f.empty:
        st.warning(f"⚠️ **{title}**: nessun risultato con filtri correnti.")
        return
    
    # SORT e LIMIT
    valid_sort = [c for c in sort_cols if c in df_f.columns]
    if valid_sort:
        df_f = df_f.sort_values(valid_sort, ascending=ascending[:len(valid_sort)])
    df_f = df_f.head(st.session_state.top)
    
    # METRICHE
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Titoli", len(df_f))
    if "Squeeze" in df_f.columns:
        m2.metric("🔥 Squeeze", int(df_f["Squeeze"].apply(lambda x: x is True or str(x).lower() == "true").sum()))
    if "Weekly_Bull" in df_f.columns:
        m3.metric("📈 Weekly+", int(df_f["Weekly_Bull"].apply(lambda x: x is True or str(x).lower() == "true").sum()))
    if "RSI_Div" in df_f.columns:
        m4.metric("⚠️ Div RSI", int((df_f["RSI_Div"] != "-").sum()))
    
    # PREPARA DF DISPLAY
    df_fmt = formatting.add_formatted_cols(df_f)
    df_disp = formatting.prepare_display_df(df_fmt)
    drop_cols = [c for c in df_disp.columns if c.startswith("_")]
    df_disp = df_disp.drop(columns=drop_cols, errors="ignore")
    
    # PRIORITY COLUMNS
    cols = list(df_disp.columns)
    priority = ["Ticker", "Nome", "Prezzo_fmt", "MarketCap_fmt", "Early_Score", "Pro_Score",
                "RSI", "Vol_Ratio", "Quality_Score", "Stato_Early", "Stato_Pro", "EMA200_fmt"]
    base = [c for c in priority if c in cols]
    rest = [c for c in cols if c not in base]
    df_disp = df_disp[base + rest].reset_index(drop=True)
    
    # AGGRID PERSISTENTE (29.0)
    grid_resp = build_aggrid(df_disp, f"grid_{title.replace(' ', '_').replace('-', '_')}")
    selected_df = pd.DataFrame(grid_resp["selected_rows"])
    
    # BUTTON AGGIUNGI
    if st.button(f"➕ Aggiungi a '{st.session_state.current_list_name}'", key=f"btn_{title}"):
        if not selected_df.empty and "Ticker" in selected_df.columns:
            tickers = selected_df["Ticker"].tolist()
            names = selected_df.get("Nome", selected_df["Ticker"]).tolist()
            db.add_to_watchlist(tickers, names, title, "Scanner", "LONG", st.session_state.current_list_name)
            st.success(f"✅ Aggiunti {len(tickers)} titoli!")
            time.sleep(0.8)
            st.rerun()
    
    # GRAFICO DETTAGLIO SE SELEZIONATO
    if not selected_df.empty:
        ticker_sel = selected_df.iloc[0].get("Ticker", "")
        match = df_f[df_f["Ticker"] == ticker_sel]
        if not match.empty:
            show_charts_v29(match.iloc[0])  # Versione 29.0 con volume profile + Ichimoku

# ── GRAFICO AVANZATO V29 (Volume Profile + Ichimoku) ───────────────────────
def show_charts_v29(row_data, key_suffix=""):
    """Grafico 29.0: candlestick + volume profile + tab Ichimoku S/R"""
    ticker = row_data.get("Ticker", "")
    chart_data = row_data.get("_chart_data", {})
    
    if not chart_data:
        st.warning("Dati grafico non disponibili.")
        return
    
    # GRAFICO PRINCIPALE CON VOLUME PROFILE
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{ticker} - Candlestick + EMA", "Volume Profile"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        column_widths=[0.7, 0.3]
    )
    
    # Candlestick + EMA
    dates, open_, high, low, close = chart_data["dates"], chart_data["open"], chart_data["high"], chart_data["low"], chart_data["close"]
    fig.add_candlestick(
        x=dates, open=open_, high=high, low=low, close=close,
        name="Candles", increasing_line_color="#00ff88", decreasing_line_color="#ef4444"
    )
    if "ema20" in chart_data:
        fig.add_scatter(x=dates, y=chart_data["ema20"], name="EMA20", line=dict(color="#3b82f6"))
    if "ema50" in chart_data:
        fig.add_scatter(x=dates, y=chart_data["ema50"], name="EMA50", line=dict(color="#f59e0b"))
    if "ema200" in chart_data:
        fig.add_scatter(x=dates, y=chart_data["ema200"], name="EMA200", line=dict(color="#8b5cf6"))
    
    # VOLUME PROFILE (29.0)
    highs = np.array(chart_data["high"])
    lows = np.array(chart_data["low"])
    volumes = np.array(chart_data["volume"])
    
    # Bins prezzo + volume per bin
    price_bins = np.linspace(lows.min(), highs.max(), 30)
    vol_profile = np.zeros(len(price_bins)-1)
    for i in range(len(highs)):
        bin_idx = np.digitize((highs[i] + lows[i]) / 2, price_bins) - 1
        if 0 <= bin_idx < len(vol_profile):
            vol_profile[bin_idx] += volumes[i]
    
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    fig.add_bar(
        x=vol_profile, y=bin_centers, orientation="h",
        name="Vol Profile", marker_color="#10b981", opacity=0.7,
        row=1, col=2
    )
    
    fig.update_layout(
        **PLOTLY_DARK,
        title=f"{ticker} - Analisi Completa 29.0",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # TAB ICHIMOKU + S/R (29.0)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Ichimoku + S/R", key=f"ichimoku_{ticker}_{key_suffix}"):
            show_ichimoku_sr(ticker)
    
def show_ichimoku_sr(ticker):
    """Ichimoku Cloud + Supporti/Resistenze"""
    data = scanner._download_ohlcv(ticker, period="1y")
    if data.empty:
        st.error("Dati non disponibili.")
        return
    
    h, l, c = data["High"], data["Low"], data["Close"]
    
    # ICHIMOKU
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    chikou = c.shift(-26)
    
    fig = go.Figure()
    fig.add_candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"])
    fig.add_scatter(x=data.index, y=tenkan, name="Tenkan", line=dict(color="#3b82f6"))
    fig.add_scatter(x=data.index, y=kijun, name="Kijun", line=dict(color="#f59e0b"))
    fig.add_scatter(x=data.index, y=senkou_a, name="Senkou A", line=dict(color="#10b981"))
    fig.add_scatter(x=data.index, y=senkou_b, name="Senkou B", line=dict(color="#ef4444"))
    
    # CLOUD FILL
    fig.add_scatter(x=data.index, y=senkou_a, fill="tonexty", fillcolor="rgba(16,185,129,0.2)", line=dict(color="rgba(0,0,0,0)"), name="Cloud")
    
    # SUPPORTI/RESISTENZE SEMPLICI (29.0)
    for window in [20, 50]:
        sr_high = h.rolling(window).max()
        sr_low = l.rolling(window).min()
        fig.add_scatter(x=data.index, y=sr_high, mode="lines", line=dict(color="#ef4444", width=1, dash="dash"), name=f"R {window}", showlegend=False)
        fig.add_scatter(x=data.index, y=sr_low, mode="lines", line=dict(color="#10b981", width=1, dash="dash"), name=f"S {window}", showlegend=False)
    
    fig.update_layout(**PLOTLY_DARK, title=f"{ticker} - Ichimoku + S/R", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── TABS PRINCIPALI ─────────────────────────────────────────────────────────
st.markdown("---")
tabs = st.tabs([
    "📡 EARLY", "💪 PRO", "🔥 REA-HOT", "⭐ CONFLUENCE", "🌐 Multi-TF",
    "🎯 Serafini", "🔎 Finviz Pro", "🛡 Crisi&Inflazione",
    "📋 Watchlist", "📈 Backtest", "📜 Storico"
])

# Assegna variabili tab
(tab_e, tab_p, tab_r, tab_conf, tab_mtf, tab_ser, tab_fv, tab_crisi,
 tab_w, tab_bt, tab_hist) = tabs

# ── TAB EARLY ──────────────────────────────────────────────────────────────
with tab_e:
    st.session_state.last_active_tab = "EARLY"
    render_scan_tab(st.session_state.df_ep, "EARLY", ["Early_Score", "RSI"], [False, True], "EARLY")

# ── TAB PRO ────────────────────────────────────────────────────────────────
with tab_p:
    st.session_state.last_active_tab = "PRO"
    pro_sort = st.radio("Ordina per", ["Quality", "Momentum"], horizontal=True, key="pro_sort")
    if pro_sort == "Momentum":
        df_pro = st.session_state.df_ep.copy()
        if not df_pro.empty and "Pro_Score" in df_pro.columns and "RSI" in df_pro.columns:
            df_pro["_Momentum"] = df_pro["Pro_Score"].fillna(0) * 10 + df_pro["RSI"].fillna(0)
        render_scan_tab(df_pro, "PRO", ["_Momentum", "Quality_Score"], [False, False], "PRO Momentum")
    else:
        render_scan_tab(st.session_state.df_ep, "PRO", ["Quality_Score", "Pro_Score", "RSI"], [False, False, True], "PRO")

# ── TAB REA-HOT ────────────────────────────────────────────────────────────
with tab_r:
    st.session_state.last_active_tab = "REA-HOT"
    render_scan_tab(st.session_state.df_rea, "HOT", ["Vol_Ratio", "Dist_POC_%"], [False, True], "REA-HOT")

# ── TAB CONFLUENCE ─────────────────────────────────────────────────────────
with tab_conf:
    st.session_state.last_active_tab = "CONFLUENCE"
    render_scan_tab(st.session_state.df_ep, "CONFLUENCE", ["Quality_Score", "Early_Score", "Pro_Score"], [False, False, False], "CONFLUENCE")

# ── TAB MTF ────────────────────────────────────────────────────────────────
with tab_mtf:
    render_scan_tab(st.session_state.df_ep, "MTF", ["Quality_Score", "Pro_Score"], [False, False], "Multi-Timeframe")

# ── TAB SERAFINI ───────────────────────────────────────────────────────────
with tab_ser:
    with st.expander("✅ Criteri Serafini", expanded=False):
        st.markdown("""
        | # | Criterio | Soglia |
        |---|----------|--------|
        | 1 | RSI(14) > 50 | Sì |
        | 2 | Prezzo > EMA20 | Sì |
        | 3 | EMA20 > EMA50 | Sì |
        | 4 | OBV crescente | Sì |
        | 5 | Vol > media | Sì |
        | 6 | No earnings | Sì |
        """)
    render_scan_tab(st.session_state.df_ep, "SERAFINI", ["Ser_Score", "Quality_Score", "RSI"], [False, False, True], "Serafini")

# ── TAB FINVIZ ─────────────────────────────────────────────────────────────
with tab_fv:
    with st.expander("✅ Filtri Finviz replicati", expanded=False):
        st.markdown("""
        | Filtro | Replica | Soglia |
        |--------|---------|--------|
        | Price | Close > $10 | Sì |
        | Avg Vol | >1M | Sì |
        | Rel Vol | >1.0 | Sì |
        | SMA20/50/200 | Prezzo sopra | Sì |
        """)
    render_scan_tab(st.session_state.df_ep, "FINVIZ_PRO", ["FV_Score", "Quality_Score"], [False, False], "Finviz Pro")

# ── NUOVO TAB CRISI & INFLAZIONE (29.0) ────────────────────────────────────
with tab_crisi:
    st.markdown("### 🛡 Crisi & Inflazione — Asset Difensivi USA + Europa")
    
    # LEGENDA
    with st.expander("📋 Spiegazione categorie", expanded=True):
        st.markdown("""
        | Categoria | Perché? | USA/Global | Europa (.MI) |
        |-----------|---------|------------|--------------|
        | **Oro** | Hedge inflazione/geopolitica | GLD, IAU, GDX | PHAU.MI, SGLD.MI |
        | **Difesa** | Boom militare | LMT, RTX, NOC | DFEU.MI, WDEF.MI |
        | **Utilities** | Beta bassa, cashflow | XLU, XLP | STUX.MI, XS6R.MI |
        | **Energia** | Inflazione energetica | XLE, CVX | ESIE.MI |
        """)
    
    # LISTA TICKER CRISI (USA + Europa)
    crisi_tickers = {
        "Oro": ["GLD", "IAU", "GDX", "GDXJ", "SLV", "PHAU.MI", "SGLD.MI"],
        "Difesa": ["LMT", "RTX", "NOC", "GD", "BA", "DFEU.MI", "WDEF.MI"],
        "Utilities": ["XLU", "XLP", "VDC", "STUX.MI", "XS6R.MI"],
        "Energia": ["XLE", "CVX", "XOM", "ESIE.MI"]
    }
    
    # SCANNER SU QUESTI TICKER
    if st.button("🔍 Analizza Crisi (15s)", type="primary"):
        with st.spinner("Scanning asset difensivi..."):
            results = []
            for cat, tickers in crisi_tickers.items():
                for tkr in tickers:
                    ep, rea = scanner.scan_ticker(tkr, 10.0, 4.0, 96.0, 0.02, 1.5)
                    if ep:
                        ep["Categoria"] = cat
                        results.append(ep)
            
            df_crisi = pd.DataFrame(results)
            st.session_state.df_crisi = df_crisi
        
        st.success(f"✅ Analizzati {len(st.session_state.df_crisi)} asset difensivi!")
    
    if "df_crisi" in st.session_state and not st.session_state.df_crisi.empty:
        # METRICHE
        df_c = st.session_state.df_crisi
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Oro RSI<30", len(df_c[(df_c["Categoria"]=="Oro") & (df_c["RSI"]<30)]))
        c2.metric("Difesa Vol>1.5x", len(df_c[(df_c["Categoria"]=="Difesa") & (df_c["Vol_Ratio"]>1.5)]))
        c3.metric("Utilities Weekly+", len(df_c[(df_c["Categoria"]=="Utilities") & (df_c["Weekly_Bull"]==True)]))
        c4.metric("Energia >EMA200", len(df_c[(df_c["Categoria"]=="Energia") & (df_c["Close"]>df_c["EMA200"])]))
        
        # AGGRID
        df_fmt = formatting.add_formatted_cols(df_c)
        df_disp = formatting.prepare_display_df(df_fmt)
        df_disp = df_disp[["Categoria", "Ticker", "Nome", "Prezzo_fmt", "MarketCap_fmt", 
                          "RSI", "Vol_Ratio", "Quality_Score", "Weekly_Bull", "EMA200_fmt"]].reset_index(drop=True)
        
        grid_resp = build_aggrid(df_disp, "grid_crisi")
        selected = pd.DataFrame(grid_resp["selected_rows"])
        
        if st.button("➕ Aggiungi a Watchlist Crisi"):
            if not selected.empty:
                tickers = selected["Ticker"].tolist()
                names = selected.get("Nome", tickers).tolist()
                db.add_to_watchlist(tickers, names, "Crisi&Inflazione", "Difensivi", "LONG", "CRISI")
                st.success(f"✅ {len(tickers)} asset salvati!")
    
    else:
        st.info("👆 Clicca 'Analizza Crisi' per popolare il tab.")

# ── TAB WATCHLIST CON DEBUG ─────────────────────────────────────────────────
with tab_w:
    st.markdown("### 📋 Watchlist")
    
    # DEBUG DB (29.0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DB Path", str(db.DB_PATH))
    with col2:
        df_w = db.load_watchlist()
        st.metric("N Titoli", len(df_w))
    with col3:
        csv = df_w.to_csv(index=False).encode()
        st.download_button("⬇️ Esporta CSV", csv, "watchlist.csv", "text/csv")
    
    if not df_w.empty:
        grid_resp = build_aggrid(df_w[["Ticker", "Nome", "origine", "note", "trend"]], "grid_watchlist")
    else:
        st.info("Watchlist vuota. Aggiungi titoli dai tab scanner!")

# ── TAB BACKTEST ───────────────────────────────────────────────────────────
with tab_bt:
    if MODULES_OK:
        render_backtest_tab()
    else:
        st.error("Backtest richiede backtest_tab.py")

# ── TAB STORICO CON LEGENDA MIGLIORATA ─────────────────────────────────────
with tab_hist:
    st.markdown("### 📜 Storico Scansioni")
    
    with st.expander("📋 Legenda (29.0)", expanded=True):
        st.markdown("""
        | Colonna | Significato |
        |---------|-------------|
        | scanned_at | Data/ora scan |
        | markets | Mercati analizzati |
        | n_early | Segnali EARLY trovati |
        | n_pro | Segnali PRO |
        | n_rea | Segnali HOT |
        | n_confluence | EARLY+PRO |
        | elapsed_s | Durata scan |
        | cache_hits | Cache efficiency |
        
        *Clicca riga → carica snapshot nei tab scanner*
        """)
    
    df_hist = db.load_scan_history(20)
    if not df_hist.empty:
        grid_resp = build_aggrid(df_hist, "grid_storico")
        if "selected_rows" in grid_resp and grid_resp["selected_rows"]:
            scan_id = grid_resp["selected_rows"][0]["id"]
            ep, rea = db.load_scan_snapshot(scan_id)
            st.session_state.df_ep = ep
            st.session_state.df_rea = rea
            st.success(f"✅ Snapshot {scan_id} caricato!")
            st.rerun()
