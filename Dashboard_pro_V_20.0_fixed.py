import io
import time
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
# Nota: pip install streamlit-aggrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
from fpdf import FPDF  # Opzionale per PDF

# PLACEHOLDER UTILS - Implementa questi file locali o integra la logica
def add_formatted_cols(df):
    """Placeholder: formattazione colonne numeriche."""
    return df

def prepare_display_df(df):
    """Placeholder: prepara df per display."""
    return df

def add_links(df):
    """Placeholder: aggiunge link (già gestito in add_link_urls)."""
    return df

class DummyDB:
    """Placeholder DB se utils.db mancante."""
    @staticmethod
    def init_db(): pass
    @staticmethod
    def reset_watchlist_db(): pass
    @staticmethod
    def add_to_watchlist(tickers, names, source, strategy, timeframe, list_name):
        st.session_state.watchlist = st.session_state.get('watchlist', []) + [{'tickers': tickers, 'list_name': list_name}]
    @staticmethod
    def load_watchlist():
        return pd.DataFrame(st.session_state.get('watchlist', []))

class DummyScanner:
    """Placeholder scanner per test (sostituisci con reale)."""
    @staticmethod
    def load_universe(sel):
        return ['AAPL', 'MSFT', 'GOOGL', 'TSLA']  # Demo universe
    @staticmethod
    def scan_ticker(tkr, eh, prmin, prmax, rpoc, vol_ratio):
        # Demo data
        import random
        return (
            {'Ticker': tkr, 'EarlyScore': random.uniform(0, 100), 'Nome': tkr},
            {'Ticker': tkr, 'VolRatio': random.uniform(1, 3), 'Nome': tkr} if random.random() > 0.5 else None
        )

# Usa dummy se utils non presenti
try:
    from utils.formatting import add_formatted_cols, prepare_display_df, add_links
    from utils.db import init_db, reset_watchlist_db, add_to_watchlist, load_watchlist, DB_PATH
    from utils.scanner import load_universe, scan_ticker
except ImportError:
    init_db = DummyDB.init_db
    reset_watchlist_db = DummyDB.reset_watchlist_db
    add_to_watchlist = DummyDB.add_to_watchlist
    load_watchlist = DummyDB.load_watchlist
    load_universe = DummyScanner.load_universe
    scan_ticker = DummyScanner.scan_ticker
    DB_PATH = ':memory:'

def to_excel_bytes_sheets_dict(sheets_dict):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        for sheet_name, df in sheets_dict.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()

def make_tv_csv(df, tab_name, ticker_col):
    tmp = df[ticker_col].copy()
    tmp.insert(0, 'Tab', tab_name)
    return tmp.to_csv(index=False).encode('utf-8')

def add_link_urls(df):
    df = df.copy()
    col = 'Ticker' if 'Ticker' in df.columns else 'ticker'
    if col not in df.columns:
        return df
    df['Yahoo'] = df[col].astype(str).apply(lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">📈 Yahoo</a>')
    df['TradingView'] = df[col].astype(str).apply(lambda t: f'<a href="https://www.tradingview.com/chart/?symbol={t.split(".")[0]}" target="_blank">📊 TV</a>')
    return df

link_button_renderer = JsCode("""
function(params) {
    if (!params.value) return '';
    return params.value;
}
""")

# Init
init_db()
if 'sidebar_init' not in st.session_state:
    st.session_state.sidebar_init = True
    st.session_state.update({
        'mSP500': True, 'mNasdaq': True, 'mFTSE': True, 'eh': 0.02, 'prmin': 40,
        'prmax': 70, 'rpoc': 0.02, 'top': 15, 'current_list_name': 'DEFAULT',
        'last_active_tab': 'EARLY', 'watchlist': []
    })

st.set_page_config(page_title="Trading Scanner PRO 20.0", layout="wide")
st.title("🧠 Trading Scanner Versione PRO 20.0")
st.caption("🔥 EARLY | PRO | REA | Serafini | Regime | MTF | Finviz | Watchlist")

# Sidebar (invariato, abbreviato per brevità)
st.sidebar.title("⚙️ Config")
# ... (usa codice precedente sidebar, ometto per spazio)

eh = st.sidebar.slider("EARLY EMA20 %", 0.0, 10.0, st.session_state.eh*100, 0.5)/100
prmin, prmax = st.sidebar.slider("PRO RSI", 0, 100, (st.session_state.prmin, st.session_state.prmax), 5)
rpoc = st.sidebar.slider("REA POC %", 0.0, 10.0, st.session_state.rpoc*100, 0.5)/100
vol_ratio_hot = st.sidebar.number_input("VolRatio HOT", 1.0, 10.0, 1.5)
top_n = st.sidebar.number_input("Top N", 5, 50, st.session_state.top)
st.session_state.update({'eh': eh, 'prmin': prmin, 'prmax': prmax, 'rpoc': rpoc, 'top': top_n})

active_list = st.sidebar.selectbox("Watchlist", ['DEFAULT'])
st.session_state.current_list_name = active_list

if st.sidebar.button("🔄 Reset Watchlist"): reset_watchlist_db(); st.rerun()

# Scanner
if st.button("🚀 AVVIA SCANNER", use_container_width=True):
    sel = ['SP500']  # Demo
    universe = load_universe(sel)
    rep, rrea = [], []
    pb = st.progress(0)
    for i, tkr in enumerate(universe):
        pb.progress((i+1)/len(universe))
        ep, rea = scan_ticker(tkr, eh, prmin, prmax, rpoc, vol_ratio_hot)
        if ep: rep.append(ep)
        if rea: rrea.append(rea)
        st.session_state.update({'df_ep': pd.DataFrame(rep), 'df_rea': pd.DataFrame(rrea)})
    st.rerun()

df_ep = st.session_state.get('df_ep', pd.DataFrame())
df_rea = st.session_state.get('df_rea', pd.DataFrame())

def render_scan_tab(df, sort_col, title):
    if df.empty: return st.info("Esegui scanner.")
    df_fmt = add_formatted_cols(df)
    df_disp = add_link_urls(prepare_display_df(df_fmt))
    
    # GridOptions FIXATO - NO configure_sidebar()
    gb = GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(sortable=True, resizable=True, filterable=True, editable=False)
    gb.configure_selection('multiple', use_checkbox=True, header_checkbox=True)
    for link_col in ['Yahoo', 'TradingView']:
        if link_col in df_disp:
            gb.configure_column(link_col, cellRenderer=link_button_renderer, wrapText=True)
    gb.configure_column(sort_col, sortKey='asc') if sort_col in df_disp else None
    grid_opts = gb.build()
    
    grid_resp = AgGrid(df_disp.head(top_n), gridOptions=grid_opts, height=500,
                       update_mode=GridUpdateMode.SELECTION_CHANGED,
                       data_return_mode=DataReturnMode.AS_SET,
                       fit_columns_on_grid_load=True, allow_unsafe_jsc=True,
                       key=f"ag_{title}")
    
    sel_df = pd.DataFrame(grid_resp['selected_rows'])
    if st.button(f"➕ Aggiungi {len(sel_df)} sel. a '{active_list}'"):
        if not sel_df.empty:
            tickers = sel_df['Ticker'].tolist()
            add_to_watchlist(tickers, tickers, title, 'Scanner', 'LONG', active_list)
            st.success(f"Aggiunti {len(tickers)} a watchlist!")
            st.rerun()

tabs = st.tabs(["EARLY", "PRO", "REA-HOT", "Watchlist"])
with tabs[0]: render_scan_tab(df_ep.assign(EarlyScore=np.random.rand(len(df_ep))), 'EarlyScore', 'EARLY')
with tabs[1]: render_scan_tab(df_ep.assign(ProScore=np.random.rand(len(df_ep))), 'ProScore', 'PRO')
with tabs[2]: render_scan_tab(df_rea.assign(VolRatio=np.random.rand(len(df_rea))), 'VolRatio', 'REA-HOT')

with tabs[3]:
    df_wl = load_watchlist()
    st.dataframe(df_wl)