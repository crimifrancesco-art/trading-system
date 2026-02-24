"""
Trading Scanner PRO 20.0 - COMPLETAMENTE AUTOCONTENUTO
Nessun file utils richiesto - Tutto integrato!
"""

import io
import time
import sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# =============================================================================
# UTILITY FUNCTIONS INTEGRATE (NO IMPORT ESTERNI)
# =============================================================================
def fmt_currency(x):
    return f"€{x:,.2f}" if pd.notna(x) else ""

def fmt_int(x):
    return f"{int(x):,}" if pd.notna(x) else ""

def fmt_marketcap(x):
    if pd.isna(x): return ""
    return f"{x/1e9:.1f}B" if x > 1e9 else f"{x/1e6:.0f}M"

def add_formatted_cols(df):
    """Aggiunge colonne formattate"""
    df_fmt = df.copy()
    for col in df.columns:
        if col.endswith('_fmt'):
            continue
        if df[col].dtype in ['int64', 'float64']:
            if 'MarketCap' in col:
                df_fmt[f'{col}_fmt'] = df_fmt[col].apply(fmt_marketcap)
            elif 'Vol' in col or 'Volume' in col:
                df_fmt[f'{col}_fmt'] = df_fmt[col].apply(fmt_int)
            elif any unit in col for unit in ['Price', 'Prezzo']):
                df_fmt[f'{col}_fmt'] = df_fmt[col].apply(lambda x: f"{x:.2f}")
    return df_fmt

def add_links(df):
    """Aggiunge link Yahoo/TradingView"""
    df_links = df.copy()
    if 'Ticker' in df_links.columns:
        df_links['Yahoo'] = df_links['Ticker'].apply(lambda t: f"[📈](https://finance.yahoo.com/quote/{t})")
        df_links['TradingView'] = df_links['Ticker'].apply(lambda t: f"[📊](https://www.tradingview.com/symbols/{t}/)")
    return df_links

def prepare_display_df(df):
    """Prepara DF per visualizzazione"""
    display_cols = ['Ticker', 'Nome', 'Prezzo', 'RSI', 'Vol_Ratio', 'Early_Score', 'Pro_Score', 'Yahoo', 'TradingView']
    available_cols = [col for col in display_cols if col in df.columns]
    return df[available_cols]

# Database functions integrate
DB_PATH = "watchlist_pro20.db"

def init_db():
    """Inizializza database watchlist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, nome TEXT, signal TEXT, source TEXT, 
        direction TEXT, list_name TEXT, note TEXT, created_at TEXT
    )''')
    conn.commit()
    conn.close()

def add_to_watchlist(tickers, names, signal, source, direction, list_name):
    """Aggiunge ticker a watchlist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for tkr, name in zip(tickers, names):
        c.execute("INSERT INTO watchlist (ticker, nome, signal, source, direction, list_name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (tkr, name, signal, source, direction, list_name, now))
    conn.commit()
    conn.close()

def load_watchlist():
    """Carica watchlist dal DB"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def reset_watchlist_db():
    """Reset completo DB"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM watchlist")
    conn.commit()
    conn.close()

# Mock scanner functions (sostituire con reali)
def load_universe(markets):
    """Carica universo titoli mock"""
    universes = {
        "SP500": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX"],
        "Nasdaq": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "FTSE": ["ISP.MI", "UCG.MI", "ENI.MI", "STM.MI"],
        "Eurostoxx": ["ASML.AS", "SAP.DE", "NESN.SW"],
        "Dow": ["AAPL", "MSFT", "UNH", "GS", "JPM"],
        "Russell": ["ABC", "ACAD", "ACLS"],
        "StoxxEmerging": ["INFY", "BABA", "PDD"],
        "USSmallCap": ["TURN", "BLBD", "TASK"]
    }
    all_tickers = []
    for market in markets:
        all_tickers.extend(universes.get(market, []))
    return list(set(all_tickers))[:50]  # Max 50 per performance

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc, vol_ratio):
    """Scanner mock - sostituisci con yfinance reale"""
    # Simula dati realistici
    np.random.seed(hash(ticker) % 1000)
    
    early_signals = []
    rea_signals = []
    
    # EARLY signal (2% vicinanza EMA20, RSI 40-70)
    if np.random.random() > 0.8:
        early_signals.append({
            'Ticker': ticker,
            'Nome': f"{ticker} Corp",
            'Prezzo': round(100 + np.random.randn()*20, 2),
            'RSI': np.random.uniform(p_rmin, p_rmax),
            'Early_Score': round(np.random.uniform(70, 95), 1),
            'Pro_Score': round(np.random.uniform(60, 85), 1),
            'MarketCap': np.random.uniform(10e9, 500e9),
            'Vol_Today': np.random.uniform(1e6, 50e6),
            'Vol_7d_Avg': np.random.uniform(0.5e6, 10e6),
            'Stato_Early': 'EARLY',
            'Stato_Pro': np.random.choice(['PRO', 'HOLD'], p=[0.3, 0.7])
        })
    
    # REA-HOT signal (vol ratio > threshold)
    if np.random.random() > 0.85:
        vol_ratio_sim = np.random.uniform(vol_ratio, 5.0)
        rea_signals.append({
            'Ticker': ticker,
            'Nome': f"{ticker} Corp",
            'Prezzo': round(100 + np.random.randn()*20, 2),
            'Vol_Ratio': round(vol_ratio_sim, 2),
            'Dist_POC_%': round(np.random.uniform(0, r_poc*100), 2),
            'MarketCap': np.random.uniform(5e9, 200e9),
            'Stato': 'HOT'
        })
    
    return early_signals, rea_signals

# =============================================================================
# CONFIGURAZIONE BASE
# =============================================================================
st.set_page_config(page_title="Trading Scanner PRO 20.0", layout="wide", page_icon="📊")
st.title("📊 Trading Scanner – PRO 20.0")
st.caption("✅ AUTOCONTENUTO • EARLY • PRO • REA-HOT • Watchlist DB")

# Init DB
init_db()

# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        "m_SP500": True, "m_Nasdaq": True, "m_FTSE": True,
        "e_h": 0.02, "p_rmin": 40, "p_rmax": 70, "r_poc": 0.02, 
        "vol_ratio_hot": 1.5, "top": 15, "current_list_name": "DEFAULT"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("⚙️ Config")

with st.sidebar.expander("📈 Mercati", expanded=True):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state["m_SP500"] = st.checkbox("🇺🇸 S&P500", st.session_state["m_SP500"])
        st.session_state["m_FTSE"] = st.checkbox("🇮🇹 FTSE", st.session_state["m_FTSE"])
    with col2:
        st.session_state["m_Nasdaq"] = st.checkbox("🇺🇸 Nasdaq", st.session_state["m_Nasdaq"])
        st.session_state["m_Eurostoxx"] = st.checkbox("🇪🇺 Eurostoxx", False)

with st.sidebar.expander("🎛️ Parametri"):
    st.session_state["e_h"] = st.slider("EARLY %", 0.0, 0.1, st.session_state["e_h"], 0.005)
    st.session_state["p_rmin"] = st.slider("RSI min", 30, 60, st.session_state["p_rmin"])
    st.session_state["p_rmax"] = st.slider("RSI max", 60, 80, st.session_state["p_rmax"])
    st.session_state["vol_ratio_hot"] = st.slider("Vol Ratio", 1.0, 5.0, st.session_state["vol_ratio_hot"], 0.1)

st.session_state["top"] = st.sidebar.slider("Top N", 5, 50, st.session_state["top"], 5)

# Watchlist sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Watchlist")
df_wl = load_watchlist()
lists = sorted(df_wl["list_name"].unique()) if not df_wl.empty else ["DEFAULT"]
st.session_state["current_list_name"] = st.sidebar.selectbox("Lista", lists + ["DEFAULT"])

# =============================================================================
# MAIN SCANNER
# =============================================================================
def run_scanner():
    markets = [k for k, v in {
        "SP500": st.session_state["m_SP500"],
        "Nasdaq": st.session_state["m_Nasdaq"],
        "FTSE": st.session_state["m_FTSE"],
        "Eurostoxx": st.session_state["m_Eurostoxx"]
    }.items() if v]
    
    universe = load_universe(markets)
    progress = st.progress(0)
    
    early, rea = [], []
    for i, ticker in enumerate(universe):
        e, r = scan_ticker(ticker, st.session_state["e_h"], st.session_state["p_rmin"], 
                          st.session_state["p_rmax"], st.session_state["r_poc"], 
                          st.session_state["vol_ratio_hot"])
        early.extend(e)
        rea.extend(r)
        progress.progress((i+1)/len(universe))
        st.empty()
    
    st.session_state["early_df"] = pd.DataFrame(early)
    st.session_state["rea_df"] = pd.DataFrame(rea)
    st.success(f"✅ {len(early)} EARLY | {len(rea)} REA-HOT")

if st.button("🚀 SCANNER PRO 20.0", type="primary"):
    run_scanner()

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🟢 EARLY", "🟣 PRO", "🔥 REA-HOT", "📋 Watchlist"])

df_early = st.session_state.get("early_df", pd.DataFrame())
df_rea = st.session_state.get("rea_df", pd.DataFrame())

def render_table(df, title):
    if df.empty:
        st.info(f"Nessun segnale {title}")
        return
    
    df_display = add_links(add_formatted_cols(prepare_display_df(df)))
    
    col1, col2 = st.columns([1,3])
    with col1:
        st.download_button("📥 CSV", df.to_csv(index=False), f"{title}.csv")
    
    with col2:
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_default_column(sortable=True, filter=True)
        gb.configure_selection("multiple", use_checkbox=True)
        gridOptions = gb.build()
        
        AgGrid(df_display, gridOptions=gridOptions, height=500)

with tab1:
    render_table(df_early[df_early.get("Stato_Early") == "EARLY"], "EARLY")

with tab2:
    render_table(df_early[df_early.get("Stato_Pro") == "PRO"], "PRO")

with tab3:
    render_table(df_rea[df_rea.get("Stato") == "HOT"], "REA-HOT")

with tab4:
    df_watch = load_watchlist()
    df_watch = df_watch[df_watch["list_name"] == st.session_state["current_list_name"]]
    render_table(df_watch, "Watchlist")

# Footer
st.markdown("---")
st.caption("✅ PRO 20.0 Autocontenuto - Francesco Serafini 2026")
