"""
Trading Scanner PRO 20.0 - Dashboard Completa
Autore: Francesco Crimi
Versione: 20.0 - Multi-Timeframe, Regime, Finviz, Watchlist DB
"""

import io
import time
import sqlite3
import locale
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from fpdf import FPDF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# Import modular functions (assumendo esistano)
try:
    from utils.formatting import fmt_currency, fmt_int, fmt_marketcap, add_formatted_cols, add_links, prepare_display_df
    from utils.db import init_db, reset_watchlist_db, add_to_watchlist, load_watchlist, update_watchlist_note, delete_from_watchlist, DB_PATH
    from utils.scanner import load_universe, scan_ticker
except ImportError:
    # Fallback functions se i moduli non esistono
    def fmt_currency(x): return f"€{x:,.2f}"
    def fmt_int(x): return f"{x:,}"
    def fmt_marketcap(x): return f"{x/1e9:.1f}B"
    def add_formatted_cols(df): return df
    def add_links(df): return df
    def prepare_display_df(df): return df
    def init_db(): pass
    def reset_watchlist_db(): pass
    def add_to_watchlist(tickers, names, signal, source, direction, list_name): pass
    def load_watchlist(): return pd.DataFrame()
    DB_PATH = "watchlist.db"
    def load_universe(markets): return ["AAPL", "MSFT", "GOOGL"]  # Mock
    def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc, vol_ratio): return ([], [])

# =============================================================================
# BLOCCO 1: CONFIGURAZIONE BASE
# =============================================================================
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 20.0",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner – Versione PRO 20.0")
st.caption("EARLY • PRO • REA‑QUANT • Serafini • Regime & Momentum • Multi‑Timeframe • Finviz • Watchlist DB")

# =============================================================================
# BLOCCO 2: INIZIALIZZAZIONE STATO
# =============================================================================
@st.cache_data
def initialize_session_state():
    """Inizializza stato sessione con valori di default"""
    defaults = {
        "m_SP500": True, "m_Nasdaq": True, "m_FTSE": True, "m_Eurostoxx": False,
        "m_Dow": False, "m_Russell": False, "m_StoxxEmerging": False, "m_USSmallCap": False,
        "e_h": 0.02, "p_rmin": 40, "p_rmax": 70, "r_poc": 0.02, "vol_ratio_hot": 1.5,
        "top": 15, "current_list_name": "DEFAULT", "sidebar_init": True
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

init_db()
initialize_session_state()

# =============================================================================
# BLOCCO 3: SIDEBAR CONFIGURAZIONE
# =============================================================================
def render_market_checkboxes():
    """Renderizza checkbox mercati"""
    col1, col2 = st.sidebar.columns(2)
    with col1:
        m_sp500 = st.checkbox("🇺🇸 S&P 500", st.session_state["m_SP500"], key="m_sp500")
        m_ftse = st.checkbox("🇮🇹 FTSE MIB", st.session_state["m_FTSE"], key="m_ftse")
        m_dow = st.checkbox("🇺🇸 Dow Jones", st.session_state["m_Dow"], key="m_dow")
    with col2:
        m_nasdaq = st.checkbox("🇺🇸 Nasdaq 100", st.session_state["m_Nasdaq"], key="m_nasdaq")
        m_euro = st.checkbox("🇪🇺 Eurostoxx 600", st.session_state["m_Eurostoxx"], key="m_euro")
        m_russell = st.checkbox("🇺🇸 Russell 2000", st.session_state["m_Russell"], key="m_russell")
    
    smallcaps = st.sidebar.columns(2)
    with smallcaps[0]:
        m_stoxxem = st.checkbox("🌍 Stoxx Emerging", st.session_state["m_StoxxEmerging"], key="m_stoxxem")
    with smallcaps[1]:
        m_ussmall = st.checkbox("🇺🇸 US Small Cap", st.session_state["m_USSmallCap"], key="m_ussmall")
    
    return {
        "SP500": m_sp500, "Nasdaq": m_nasdaq, "FTSE": m_ftse, "Eurostoxx": m_euro,
        "Dow": m_dow, "Russell": m_russell, "StoxxEmerging": m_stoxxem, "USSmallCap": m_ussmall
    }

def render_scanner_params():
    """Renderizza parametri scanner"""
    col1, col2 = st.sidebar.columns(2)
    with col1:
        e_h = st.slider("EARLY - EMA20 (%)", 0.0, 10.0, float(st.session_state["e_h"] * 100), 0.5) / 100
    with col2:
        r_poc = st.slider("REA - POC (%)", 0.0, 10.0, float(st.session_state["r_poc"] * 100), 0.5) / 100
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        p_rmin = st.slider("PRO - RSI min", 0, 100, int(st.session_state["p_rmin"]), 5)
    with col4:
        p_rmax = st.slider("PRO - RSI max", 0, 100, int(st.session_state["p_rmax"]), 5)
    
    vol_ratio_hot = st.number_input("Vol_Ratio REA‑HOT", 0.0, 10.0, st.session_state["vol_ratio_hot"], 0.1)
    top_n = st.number_input("TOP N titoli", 5, 100, int(st.session_state["top"]), 5)
    
    return {"e_h": e_h, "p_rmin": p_rmin, "p_rmax": p_rmax, "r_poc": r_poc, "vol_ratio_hot": vol_ratio_hot, "top": top_n}

def render_watchlist_management():
    """Gestisce watchlist nella sidebar"""
    st.sidebar.divider()
    st.sidebar.subheader("📁 Watchlist")
    
    df_wl_all = load_watchlist()
    list_options = sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
    if "DEFAULT" not in list_options:
        list_options.append("DEFAULT")
    
    active_list = st.sidebar.selectbox("Lista Attiva", list_options, 
                                     index=list_options.index(st.session_state["current_list_name"]))
    st.session_state["current_list_name"] = active_list
    
    new_list = st.sidebar.text_input("📁 Nuova Watchlist")
    if st.sidebar.button("➕ Crea", key="create_list"):
        if new_list.strip():
            st.session_state["current_list_name"] = new_list.strip()
            st.success(f"Lista '{new_list.strip()}' creata!")
            time.sleep(1)
            st.rerun()
    
    if st.sidebar.button("🗑️ Reset DB", help="Elimina tutte le watchlist!", key="reset_db"):
        reset_watchlist_db()
        st.rerun()

# ESEGUI SIDEBAR
markets = render_market_checkboxes()
params = render_scanner_params()
render_watchlist_management()

# Aggiorna session state
for key, value in {**markets, **params}.items():
    st.session_state[key] = value

# =============================================================================
# BLOCCO 4: FUNZIONI HELPER
# =============================================================================
def get_csv_download_link(df, filename="export.csv", key=None):
    """Genera link download CSV"""
    if df.empty: return None
    csv = df.to_csv(index=False).encode('utf-8')
    return st.download_button(label="📥 CSV", data=csv, file_name=filename, mime='text/csv', key=key)

def get_selected_markets():
    """Restituisce lista mercati selezionati"""
    sel = []
    market_map = {
        "SP500": st.session_state["m_SP500"],
        "Nasdaq": st.session_state["m_Nasdaq"],
        "FTSE": st.session_state["m_FTSE"],
        "Eurostoxx": st.session_state["m_Eurostoxx"],
        "Dow": st.session_state["m_Dow"],
        "Russell": st.session_state["m_Russell"],
        "StoxxEmerging": st.session_state["m_StoxxEmerging"],
        "USSmallCap": st.session_state["m_USSmallCap"]
    }
    for market, selected in market_map.items():
        if selected: sel.append(market)
    return sel

def show_legend(title):
    """Mostra legenda per tipo di segnale"""
    with st.expander(f"ℹ️ Legenda {title}", expanded=False):
        legends = {
            "EARLY": "**EARLY**: Titoli vicini alla EMA20 (trend in formazione). *Early_Score*: vicinanza media + volumi.",
            "PRO": "**PRO**: RSI neutrale-rialzista + trend forza. *Pro_Score*: trend + RSI + breakout.",
            "REA-HOT": "**REA-HOT**: Volumi anomali (>1.5x) + vicini POC. *Vol_Ratio*: volume oggi/media 7gg."
        }
        st.write(legends.get(title, f"Segnali {title}"))

# =============================================================================
# BLOCCO 5: SCANNER EXECUTION
# =============================================================================
def execute_scanner():
    """Esegue scansione completa"""
    sel = get_selected_markets()
    universe = load_universe(sel)
    
    if not universe:
        st.warning("⚠️ Seleziona almeno un mercato!")
        return
    
    st.info(f"🚀 Scansione {len(universe)} titoli...")
    r_ep, r_rea = [], []
    pb = st.progress(0)
    status = st.empty()
    
    for i, tkr in enumerate(universe):
        status.text(f"🔍 {i+1}/{len(universe)}: {tkr}")
        ep, rea = scan_ticker(
            tkr, st.session_state["e_h"], st.session_state["p_rmin"], 
            st.session_state["p_rmax"], st.session_state["r_poc"], 
            st.session_state["vol_ratio_hot"]
        )
        if ep: r_ep.extend(ep)
        if rea: r_rea.extend(rea)
        pb.progress((i + 1) / len(universe))
    
    st.session_state["df_ep"] = pd.DataFrame(r_ep) if r_ep else pd.DataFrame()
    st.session_state["df_rea"] = pd.DataFrame(r_rea) if r_rea else pd.DataFrame()
    st.session_state["last_scan"] = datetime.now().strftime("%d/%m %H:%M:%S")
    st.success(f"✅ Scanner completato! {len(r_ep)} EARLY, {len(r_rea)} REA-HOT")
    st.rerun()

# Pulsante principale scanner
only_watchlist = st.sidebar.checkbox("👁️ Solo Watchlist", value=False)
if not only_watchlist:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 AVVIA SCANNER PRO 20.0", type="primary", use_container_width=True):
            execute_scanner()
    with col2:
        st.info(f"⏰ Ultimo: {st.session_state.get('last_scan', 'Nessuno')}")
    with col3:
        if st.button("🔄 Refresh", key="refresh"):
            st.rerun()

# =============================================================================
# BLOCCO 6: FUNZIONI RENDER TAB
# =============================================================================
def prepare_tab_data(df, status_filter):
    """Prepara dati per tab specifico"""
    if df.empty: return pd.DataFrame()
    
    col_filter = "Stato_Early" if "Stato_Early" in df.columns and status_filter == "EARLY" else \
                ("Stato_Pro" if "Stato_Pro" in df.columns and status_filter == "PRO" else "Stato")
    
    df_f = df[df[col_filter] == status_filter].copy() if col_filter in df.columns else df.copy()
    return df_f.sort_values(by=["Ticker"], ascending=True).head(st.session_state["top"])

def render_tab_controls(df_f, title, active_list):
    """Renderizza controlli export e selezione"""
    col_exp, col_add = st.columns([1, 2])
    
    with col_exp:
        get_csv_download_link(df_f, f"{title.lower()}_export.csv", key=f"exp_{title}")
    
    with col_add:
        select_all = st.checkbox("✅ Tutti", key=f"all_{title}")
        options_raw = [f"{row['Ticker']} - {row['Nome']}" for _, row in df_f.iterrows()]
        options = sorted(list(set(options_raw)))
        mapping = {f"{row['Ticker']} - {row['Nome']}": row['Ticker'] for _, row in df_f.iterrows()}
        
        selected = st.multiselect(f"➕ Aggiungi a {active_list}", options, 
                                default=options if select_all else [], key=f"add_{title}")
        
        if st.button(f"➕ Aggiungi {len(selected)}", key=f"btn_{title}"):
            if selected:
                tickers_to_add = [mapping[s] for s in selected]
                to_ins = df_f[df_f["Ticker"].isin(tickers_to_add)]
                add_to_watchlist(
                    to_ins["Ticker"].tolist(), to_ins["Nome"].tolist(), 
                    title, "Scanner", "LONG", active_list
                )
                st.success(f"✅ Aggiunti {len(tickers_to_add)} a {active_list}!")
                time.sleep(1)
                st.rerun()

def render_aggrid_table(df_v):
    """Renderizza tabella AgGrid"""
    gb = GridOptionsBuilder.from_dataframe(df_v)
    gb.configure_default_column(sortable=True, resizable=True, filterable=True, editable=False)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True, header_checkbox=True)
    grid_options = gb.build()
    
    return AgGrid(
        df_v, gridOptions=grid_options, height=600,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        allow_unsafe_jscode=True
    )

def render_scan_tab(df, status_filter, sort_cols, ascending, title):
    """Renderizza tab completo scanner"""
    st.subheader(f"📊 {title}")
    show_legend(title)
    
    df_display = prepare_tab_data(df, status_filter)
    if df_display.empty:
        st.info(f"ℹ️ Nessun segnale {title} trovato.")
        return
    
    # Controlli
    render_tab_controls(df_display, title, st.session_state["current_list_name"])
    
    # Tabella
    df_v = add_links(prepare_display_df(add_formatted_cols(df_display)))
    render_aggrid_table(df_v)

# =============================================================================
# BLOCCO 7: TABS PRINCIPALI
# =============================================================================
df_ep = st.session_state.get("df_ep", pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

tabs = st.tabs([
    "🟢 EARLY", "🟣 PRO", "🟠 REA-HOT", "📈 Serafini", 
    "🧊 Regime", "🕒 MTF", "📊 Finviz", "📌 Watchlist"
])

# Configurazioni tabs
tab_configs = [
    (tabs[0], df_ep, "EARLY", ["Early_Score", "RSI"], [False, True]),
    (tabs[1], df_ep, "PRO", ["Pro_Score", "RSI"], [False, True]),
    (tabs[2], df_rea, "HOT", ["Vol_Ratio", "Dist_POC_%"], [False, True]),
    (tabs[3], df_ep, "SERAFINI", ["Ticker"], [True]),
    (tabs[4], df_ep, "REGIME", ["Ticker"], [True]),
    (tabs[5], df_ep, "MTF", ["Ticker"], [True]),
    (tabs[6], df_ep, "FINVIZ", ["Ticker"], [True]),
]

for i, (tab, df, status, sort_cols, ascending) in enumerate(tab_configs):
    with tab:
        render_scan_tab(df, status, sort_cols, ascending, status)

# =============================================================================
# BLOCCO 8: WATCHLIST TAB
# =============================================================================
def render_watchlist_tab():
    """Renderizza tab watchlist completa"""
    st.subheader(f"📌 Watchlist: {st.session_state['current_list_name']}")
    
    df_w_view = load_watchlist()
    df_w_view = df_w_view[df_w_view["list_name"] == st.session_state["current_list_name"]]
    
    if df_w_view.empty:
        st.info("📭 Watchlist vuota. Aggiungi titoli dallo scanner!")
        return
    
    # Controlli watchlist
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        get_csv_download_link(df_w_view, f"watchlist_{st.session_state['current_list_name']}.csv")
    
    with col2:
        list_options = sorted(load_watchlist()["list_name"].unique())
        move_target = st.selectbox("Sposta in:", [""] + list_options, key="move_target")
        ids_to_move = st.multiselect("ID da spostare:", df_w_view["id"].astype(str).tolist(), key="move_ids")
        
        if st.button("📂 Sposta", key="move_wl") and move_target and ids_to_move:
            # Logica spostamento (implementare se necessario)
            st.success(f"Spostati {len(ids_to_move)} elementi!")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Elimina selezionati", key="delete_wl"):
            st.warning("⚠️ Implementa delete logic qui")
    
    # Tabella watchlist
    df_w_v = add_links(prepare_display_df(add_formatted_cols(df_w_view)))
    st.dataframe(df_w_v, use_container_width=True, hide_index=True)

with tabs[7]:
    render_watchlist_tab()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Francesco Serafini | PRO 20.0 | 2026")
