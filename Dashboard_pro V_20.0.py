import io
import time
import sqlite3
import locale
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from fpdf import FPDF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# -------------------------------------------------------------------------
# FUNZIONI EXPORT SUPPORTO (4 EXPORT)
# -------------------------------------------------------------------------
def to_excel_bytes(sheets_dict: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()

def make_tv_csv(df: pd.DataFrame, tab_name: str, ticker_col: str = "Ticker") -> bytes:
    tmp = df[[ticker_col]].copy()
    tmp.insert(0, "Tab", tab_name)
    return tmp.to_csv(index=False).encode("utf-8")

# -------------------------------------------------------------------------
# RENDERER LINK "Apri" PER YAHOO / TRADINGVIEW (colonne *_url)
# -------------------------------------------------------------------------
link_button_renderer = JsCode("""
function(params) {
    if (!params.value) { return ''; }
    return `<a href="${params.value}" target="_blank" style="text-decoration:none;">
                🔗 Apri
            </a>`;
}
""")

# -------------------------------------------------------------------------
# IMPORT MODULI UTILS
# -------------------------------------------------------------------------
from utils.formatting import (
    fmt_currency,
    fmt_int,
    fmt_marketcap,
    add_formatted_cols,
    add_links,
    prepare_display_df,
)
from utils.db import (
    init_db,
    reset_watchlist_db,
    add_to_watchlist,
    load_watchlist,
    update_watchlist_note,
    delete_from_watchlist,
    DB_PATH,
)
from utils.scanner import load_universe, scan_ticker

# -------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 20.0",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner – Versione PRO 20.0")
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
    st.session_state.setdefault("last_active_tab", "EARLY")

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
    e_h = st.slider(
        "EARLY - Distanza EMA20 (%)",
        0.0,
        10.0,
        float(st.session_state["e_h"] * 100),
        0.5,
    ) / 100
    p_rmin = st.slider("PRO - RSI minimo", 0, 100, int(st.session_state["p_rmin"]), 5)
    p_rmax = st.slider("PRO - RSI massimo", 0, 100, int(st.session_state["p_rmax"]), 5)
    r_poc = st.slider(
        "REA - Distanza POC (%)",
        0.0,
        10.0,
        float(st.session_state["r_poc"] * 100),
        0.5,
    ) / 100
    vol_ratio_hot = st.number_input(
        "Vol_Ratio minimo REA‑HOT", 0.0, 10.0, 1.5, 0.1
    )
    top = st.number_input(
        "TOP N titoli per tab", 5, 100, int(st.session_state["top"]), 5
    )

st.session_state["e_h"] = e_h
st.session_state["p_rmin"] = p_rmin
st.session_state["p_rmax"] = p_rmax
st.session_state["r_poc"] = r_poc
st.session_state["top"] = top

st.sidebar.divider()
st.sidebar.subheader("📁 Gestione Watchlist")
df_wl_all = load_watchlist()
list_options = (
    sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
)
if "DEFAULT" not in list_options:
    list_options.append("DEFAULT")

active_list = st.sidebar.selectbox(
    "Lista Attiva",
    list_options,
    index=list_options.index(st.session_state["current_list_name"])
    if st.session_state["current_list_name"] in list_options
    else 0,
)
st.session_state["current_list_name"] = active_list

new_list = st.sidebar.text_input("📁 Crea Nuova Watchlist")
if st.sidebar.button("➕ Crea"):
    if new_list.strip():
        st.session_state["current_list_name"] = new_list.strip()
        st.success(f"Lista '{new_list.strip()}' creata!")
        time.sleep(1)
        st.rerun()

if st.sidebar.button("🗑️ Reset DB Completo", help="Elimina tutte le watchlist!"):
    reset_watchlist_db()
    st.rerun()

# =============================================================================
# LOGICA EXPORT (Helper CSV semplice)
# =============================================================================
def get_csv_download_link(df, filename="export.csv", key=None):
    csv = df.to_csv(index=False).encode("utf-8")
    return st.download_button(
        label="📥 Export CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )

# =============================================================================
# SCANNER EXECUTION
# =============================================================================
only_watchlist = st.sidebar.checkbox("Mostra solo Watchlist", value=False)
if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 20.0", type="primary", use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            r_ep, r_rea = [], []
            pb = st.progress(0)
            status = st.empty()
            for i, tkr in enumerate(universe):
                status.text(f"Analisi {i+1}/{len(universe)}: {tkr}")
                ep, rea = scan_ticker(
                    tkr, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot
                )
                if ep:
                    r_ep.append(ep)
                if rea:
                    r_rea.append(rea)
                pb.progress((i + 1) / len(universe))

            st.session_state["df_ep"] = pd.DataFrame(r_ep)
            st.session_state["df_rea"] = pd.DataFrame(r_rea)
            st.session_state["last_scan"] = datetime.now().strftime("%H:%M:%S")
            st.rerun()

df_ep = st.session_state.get("df_ep", pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

# =============================================================================
# LEGENDA (Helper)
# =============================================================================
def show_legend(title):
    with st.expander(f"ℹ️ Legenda {title}", expanded=False):
        if title == "EARLY":
            st.write("**EARLY**: Titoli vicini alla EMA20 (trend in formazione).")
            st.write("- Early_Score: Punteggio basato sulla vicinanza alla media e volumi.")
        elif title == "PRO":
            st.write("**PRO**: Segnali di forza con RSI in zona neutrale-rialzista.")
            st.write("- Pro_Score: Punteggio basato su trend, RSI e breakout volumetrici.")
        elif title == "REA-HOT":
            st.write(
                "**REA-HOT**: Titoli con volumi anomali (Volume Ratio > 1.5) e vicini al POC."
            )
            st.write("- Vol_Ratio: Rapporto tra volume odierno e media a 7 giorni.")
        else:
            st.write(f"Segnali scanner per il sistema {title}.")

# =============================================================================
# TABS PRINCIPALI
# =============================================================================
tabs = st.tabs(
    [
        "🟢 EARLY",
        "🟣 PRO",
        "🟠 REA-HOT",
        "📈 Serafini Systems",
        "🧊 Regime & Momentum",
        "🕒 Multi-Timeframe",
        "📊 Finviz",
        "📌 Watchlist & Note",
    ]
)
tab_e, tab_p, tab_r, tab_serafini, tab_regime, tab_mtf, tab_finviz, tab_w = tabs

# =============================================================================
# FUNZIONE GENERICA PER TAB SCANNER (tutti i tab usano questo)
# =============================================================================
def render_scan_tab(df, status_filter, sort_cols, ascending, title):
    st.subheader(f"Tab {title}")
    show_legend(title)

    if df.empty:
        st.info(f"Nessun dato {title}. Esegui lo scanner.")
        return

    # Filtri per Stato
    if status_filter == "EARLY":
        col_f = "Stato_Early"
    elif status_filter == "PRO":
        col_f = "Stato_Pro"
    elif status_filter == "HOT":
        col_f = "Stato"
    else:
        col_f = "Stato"

    if status_filter == "HOT":
        df_f = df[df["Stato"] == "HOT"].copy() if "Stato" in df.columns else df.copy()
    elif status_filter in ["EARLY", "PRO"]:
        if col_f in df.columns:
            df_f = df[df[col_f] == status_filter].copy()
        else:
            df_f = df.copy()
    else:
        df_f = df.copy()

    if df_f.empty:
        st.write(f"Nessun segnale {title} trovato.")
        return

    # Ordinamento multiplo
    SORT_COLUMNS_ALL = [
        "Nome",
        "Ticker",
        "MarketCap_fmt",
        "Vol_Today_fmt",
        "Vol_7d_Avg_fmt",
        "Prezzo",
        "Early_Score",
        "Pro_Score",
        "RSI",
        "Vol_Ratio",
        "OBV_Trend",
        "ATR",
        "ATR_Exp",
        "Stato",
        "Yahoo",
        "TradingView",
    ]
    available_sort = [c for c in SORT_COLUMNS_ALL if c in df_f.columns]

    with st.expander("🔀 Ordinamento colonne (multiplo)", expanded=False):
        sort_sel = st.multiselect(
            "Colonne (in ordine di priorità):",
            options=available_sort,
            default=[c for c in sort_cols if c in available_sort],
            key=f"sort_cols_{title}",
        )
        sort_dirs = {}
        for col in sort_sel:
            sort_dirs[col] = (
                st.radio(
                    col,
                    ["↑ ASC", "↓ DESC"],
                    index=0
                    if (ascending[sort_cols.index(col)] if col in sort_cols else True)
                    else 1,
                    key=f"sort_dir_{title}_{col}",
                    horizontal=True,
                )
                == "↑ ASC"
            )
        if sort_sel:
            sort_cols = sort_sel
            ascending = [sort_dirs[c] for c in sort_sel]

    df_f = df_f.sort_values(sort_cols, ascending=ascending).head(
        st.session_state["top"]
    )

    # Formattazione + link per tutti i tab (crea Yahoo_url / TradingView_url)
    df_fmt = add_formatted_cols(df_f)
    df_v = prepare_display_df(df_fmt)
    df_v = add_links(df_v)  # DEVE creare colonne Yahoo_url / TradingView_url[cite:18]

    # EXPORT CSV semplice tab (raw, non formattato)
    col_exp, col_add = st.columns([1, 1])
    with col_exp:
        get_csv_download_link(df_f, f"{title.lower()}_export.csv", key=f"exp_{title}")

    # Aggiunta a Watchlist tramite selezione righe in
