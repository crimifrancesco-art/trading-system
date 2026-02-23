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

# Import modular functions
from utils.formatting import fmt_currency, fmt_int, fmt_marketcap, add_formatted_cols, add_links
from utils.db import init_db, reset_watchlist_db, add_to_watchlist, load_watchlist, update_watchlist_note, delete_from_watchlist, DB_PATH
from utils.scanner import load_universe, scan_ticker

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 9.6",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner – Versione PRO 9.6")
st.caption(
    "EARLY • PRO • REA‑QUANT • Serafini • Regime & Momentum • "
    "Multi‑Timeframe • Finviz • Watchlist DB"
)

# =============================================================================
# INIZIALIZZAZIONE STATO
# =============================================================================
init_db()
if "sidebar_init" not in st.session_state:
    st.session_state["sidebar_init"] = True
    st.session_state.setdefault("m_SP500", True)
    st.session_state.setdefault("m_Nasdaq", True)
    st.session_state.setdefault("m_FTSE", True)
    st.session_state.setdefault("m_Eurostoxx", False)
    st.session_state.setdefault("m_Dow", False)
    st.session_state.setdefault("m_Russell", False)
    st.session_state.setdefault("m_StoxxEmerging", False)
    st.session_state.setdefault("m_USSmallCap", False)
    st.session_state.setdefault("e_h", 0.02)
    st.session_state.setdefault("p_rmin", 40)
    st.session_state.setdefault("p_rmax", 70)
    st.session_state.setdefault("r_poc", 0.02)
    st.session_state.setdefault("top", 15)
    st.session_state.setdefault("current_list_name", "DEFAULT")

# =============================================================================
# SIDEBAR – MERCATI E PARAMETRI
# =============================================================================
st.sidebar.title("⚙️ Configurazione")
with st.sidebar.expander("📈 Selezione Mercati", expanded=True):
    m_sp500 = st.checkbox("🇺🇸 S&P 500", st.session_state["m_SP500"])
    m_nasdaq = st.checkbox("🇺🇸 Nasdaq 100", st.session_state["m_Nasdaq"])
    m_ftse = st.checkbox("🇮🇹 FTSE MIB", st.session_state["m_FTSE"])
    m_euro = st.checkbox("🇪🇺 Eurostoxx 600", st.session_state["m_Eurostoxx"])
    m_dow = st.checkbox("🇺🇸 Dow Jones", st.session_state["m_Dow"])
    m_russell = st.checkbox("🇺🇸 Russell 2000", st.session_state["m_Russell"])
    m_stoxxem = st.checkbox("🌍 Stoxx Emerging 50", st.session_state["m_StoxxEmerging"])
    m_ussmall = st.checkbox("🇺🇸 US Small Cap 2000", st.session_state["m_USSmallCap"])
    
    sel = []
    if m_sp500: sel.append("SP500")
    if m_nasdaq: sel.append("Nasdaq")
    if m_ftse: sel.append("FTSE")
    if m_euro: sel.append("Eurostoxx")
    if m_dow: sel.append("Dow")
    if m_russell: sel.append("Russell")
    if m_stoxxem: sel.append("StoxxEmerging")
    if m_ussmall: sel.append("USSmallCap")
    
    st.session_state["m_SP500"] = m_sp500
    st.session_state["m_Nasdaq"] = m_nasdaq
    st.session_state["m_FTSE"] = m_ftse
    st.session_state["m_Eurostoxx"] = m_euro
    st.session_state["m_Dow"] = m_dow
    st.session_state["m_Russell"] = m_russell
    st.session_state["m_StoxxEmerging"] = m_stoxxem
    st.session_state["m_USSmallCap"] = m_ussmall

with st.sidebar.expander("🎛️ Parametri Scanner", expanded=False):
    e_h = st.slider("EARLY - Distanza EMA20 (%)", 0.0, 10.0, float(st.session_state["e_h"] * 100), 0.5) / 100
    p_rmin = st.slider("PRO - RSI minimo", 0, 100, int(st.session_state["p_rmin"]), 5)
    p_rmax = st.slider("PRO - RSI massimo", 0, 100, int(st.session_state["p_rmax"]), 5)
    r_poc = st.slider("REA - Distanza POC (%)", 0.0, 10.0, float(st.session_state["r_poc"] * 100), 0.5) / 100
    vol_ratio_hot = st.number_input("Vol_Ratio minimo REA‑HOT", 0.0, 10.0, 1.5, 0.1)
    top = st.number_input("TOP N titoli per tab", 5, 100, int(st.session_state["top"]), 5)
    
    st.session_state["e_h"] = e_h
    st.session_state["p_rmin"] = p_rmin
    st.session_state["p_rmax"] = p_rmax
    st.session_state["r_poc"] = r_poc
    st.session_state["top"] = top

st.sidebar.divider()
st.sidebar.subheader("📁 Gestione Watchlist")
df_wl_all = load_watchlist()
list_options = sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
if "DEFAULT" not in list_options: list_options.append("DEFAULT")
active_list = st.sidebar.selectbox("Lista Attiva", list_options, index=list_options.index(st.session_state["current_list_name"]))
st.session_state["current_list_name"] = active_list

new_list = st.sidebar.text_input("Nuova Lista (opzionale)")
if new_list.strip():
    active_list = new_list.strip()
    st.session_state["current_list_name"] = active_list

if st.sidebar.button("🗑️ Reset DB Completo", help="Elimina tutte le watchlist!"):
    reset_watchlist_db()
    st.rerun()

# =============================================================================
# LOGICA EXPORT (Helper)
# =============================================================================
def get_csv_download_link(df, filename="export.csv", key=None):
    csv = df.to_csv(index=False).encode('utf-8')
    return st.download_button(label="📥 Export CSV", data=csv, file_name=filename, mime='text/csv', key=key)

# =============================================================================
# SCANNER EXECUTION
# =============================================================================
only_watchlist = st.sidebar.checkbox("Mostra solo Watchlist", value=False)
if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 9.6", type="primary", use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            r_ep, r_rea = [], []
            pb = st.progress(0)
            status = st.empty()
            for i, tkr in enumerate(universe):
                status.text(f"Analisi {i+1}/{len(universe)}: {tkr}")
                ep, rea = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
                if ep: r_ep.append(ep)
                if rea: r_rea.append(rea)
                pb.progress((i + 1) / len(universe))
            
            st.session_state["df_ep"] = pd.DataFrame(r_ep)
            st.session_state["df_rea"] = pd.DataFrame(r_rea)
            st.session_state["last_scan"] = datetime.now().strftime("%H:%M:%S")
            st.rerun()

df_ep = st.session_state.get("df_ep", pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

# =============================================================================
# TABS PRINCIPALI
# =============================================================================
tabs = st.tabs([
    "🟢 EARLY", "🟣 PRO", "🟠 REA-QUANT", 
    "📈 Serafini Systems", "🧊 Regime & Momentum", "🕒 Multi-Timeframe", 
    "📊 Finviz", "📌 Watchlist & Note"
])
tab_e, tab_p, tab_r, tab_serafini, tab_regime, tab_mtf, tab_finviz, tab_w = tabs

def render_scan_tab(df, status_filter, sort_cols, ascending, title):
    st.subheader(f"Tab {title}")
    
    if df.empty:
        st.info(f"Nessun dato {title}. Esegui lo scanner.")
        return
    
    col_f = "Stato_Early" if status_filter == "EARLY" else ("Stato_Pro" if status_filter == "PRO" else "Stato")
    df_f = df[df[col_f] == status_filter].copy() if col_f in df.columns else df.copy()
    
    if df_f.empty:
        st.write(f"Nessun segnale {title} trovato.")
        return
    
    df_f = df_f.sort_values(sort_cols, ascending=ascending).head(st.session_state["top"])
    df_v = add_links(add_formatted_cols(df_f))
    
    col_exp, col_add = st.columns([1, 1])
    with col_exp:
        get_csv_download_link(df_f, f"{title.lower()}_export.csv", key=f"exp_{title}")
    
    with col_add:
        options = [f"{row['Ticker']} - {row['Nome']}" for _, row in df_f.iterrows()]
        mapping = {f"{row['Ticker']} - {row['Nome']}": row['Ticker'] for _, row in df_f.iterrows()}
        
        c1, c2 = st.columns([3, 1])
        with c2:
            select_all = st.checkbox("Seleziona tutti", key=f"all_{title}")
        
        with c1:
            default_sel = options if select_all else []
            selected_display = st.multiselect(f"Aggiungi a {active_list}", options, default=default_sel, key=f"add_{title}")
        
        if st.button(f"Aggiungi selezionati", key=f"btn_{title}"):
            tickers_to_add = [mapping[s] for s in selected_display]
            to_ins = df_f[df_f["Ticker"].isin(tickers_to_add)]
            add_to_watchlist(to_ins["Ticker"].tolist(), to_ins["Nome"].tolist(), title, "Scanner", "LONG", active_list)
            st.success(f"Aggiunti {len(tickers_to_add)} titoli!")
            time.sleep(1)
            st.rerun()
    
    st.dataframe(df_v, use_container_width=True)

with tab_e: render_scan_tab(df_ep, "EARLY", ["Early_Score", "RSI"], [False, True], "EARLY")
with tab_p: render_scan_tab(df_ep, "PRO", ["Pro_Score", "RSI"], [False, True], "PRO")
with tab_r: render_scan_tab(df_rea, "HOT", ["Vol_Ratio", "Dist_POC_%"], [False, True], "REA-HOT")
with tab_serafini: st.info("Serafini Systems: Tab in fase di implementazione...")
with tab_regime: st.info("Regime & Momentum: Tab in fase di implementazione...")
with tab_mtf: st.info("Multi-Timeframe: Tab in fase di implementazione...")
with tab_finviz: st.info("Finviz: Tab in fase di implementazione...")

with tab_w:
    st.subheader(f"Watchlist: {active_list}")
    df_w_view = load_watchlist()
    df_w_view = df_w_view[df_w_view["list_name"] == active_list]
    
    if df_w_view.empty:
        st.info("Watchlist vuota.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            get_csv_download_link(df_w_view, f"watchlist_{active_list}.csv", key="exp_wl")
        with c2:
            move_target = st.selectbox("Sposta in:", list_options, key="move_target")
            ids_to_move = st.multiselect("Seleziona ID:", df_w_view["id"].tolist())
            if st.button("Sposta"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                for i in ids_to_move:
                    c.execute("UPDATE watchlist SET list_name = ? WHERE id = ?", (move_target, i))
                conn.commit(); conn.close()
                st.rerun()
        with c3:
            if st.button("🗑️ Elimina selezionati"):
                # Simplified delete logic
                st.warning("Usa la checkbox della tabella se disponibile o specifica ID")
        
        st.dataframe(df_w_view, use_container_width=True)
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
