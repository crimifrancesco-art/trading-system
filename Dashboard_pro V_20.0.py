import io
import time
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

from utils.formatting import add_formatted_cols, add_links, prepare_display_df
from utils.db import init_db, reset_watchlist_db, add_to_watchlist, load_watchlist, DB_PATH
from utils.scanner import load_universe, scan_ticker

# -------------------------------------------------------------------------
# EXPORT HELPERS
# -------------------------------------------------------------------------
def to_excel_bytes_sheets_dict(sheets_dict: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()

def make_tv_csv(df: pd.DataFrame, tab_name: str, ticker_col: str) -> bytes:
    tmp = df[[ticker_col]].copy()
    tmp.insert(0, "Tab", tab_name)
    return tmp.to_csv(index=False).encode("utf-8")

def get_csv_download_button(df: pd.DataFrame, filename: str, key: str):
    csv = df.to_csv(index=False).encode("utf-8")
    return st.download_button(
        label="📥 Export CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )

# -------------------------------------------------------------------------
# RENDERER JS: doppio click su Nome -> TradingView
# -------------------------------------------------------------------------
name_dblclick_renderer = JsCode("""
class NameDoubleClickRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        this.eGui.innerText = params.value || '';
        const ticker = params.data.Ticker || params.data.ticker;
        if (!ticker) return;

        this.eGui.style.cursor = 'pointer';

        this.eGui.ondblclick = function() {
            const symbol = String(ticker).split(".")[0];
            const url = "https://www.tradingview.com/chart/?symbol=" + symbol;
            window.open(url, "_blank");
        }
    }
    getGui() {
        return this.eGui;
    }
}
""")

# -------------------------------------------------------------------------
# INIT
# -------------------------------------------------------------------------
st.set_page_config(page_title="Trading Scanner PRO 20.0", layout="wide", page_icon="📈")
st.title("🧠 Trading Scanner Versione PRO 20.0")
st.caption("EARLY • PRO • REA-HOT • Serafini • Regime • MultiTF • Finviz • Watchlist DB")

init_db()
if "init_done" not in st.session_state:
    st.session_state.init_done = True
    st.session_state.setdefault("mSP500", True)
    st.session_state.setdefault("mNasdaq", True)
    st.session_state.setdefault("mFTSE", True)
    st.session_state.setdefault("mEurostoxx", False)
    st.session_state.setdefault("mDow", False)
    st.session_state.setdefault("mRussell", False)
    st.session_state.setdefault("mStoxxEmerging", False)
    st.session_state.setdefault("mUSSmallCap", False)
    st.session_state.setdefault("eh", 0.02)
    st.session_state.setdefault("prmin", 40)
    st.session_state.setdefault("prmax", 70)
    st.session_state.setdefault("rpoc", 0.02)
    st.session_state.setdefault("top", 15)
    st.session_state.setdefault("current_list_name", "DEFAULT")
    st.session_state.setdefault("last_active_tab", "EARLY")

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurazione")

with st.sidebar.expander("🌍 Selezione Mercati", expanded=True):
    msp500 = st.checkbox("S&P 500", st.session_state.mSP500)
    mnasdaq = st.checkbox("Nasdaq 100", st.session_state.mNasdaq)
    mftse = st.checkbox("FTSE MIB", st.session_state.mFTSE)
    meuro = st.checkbox("Eurostoxx 600", st.session_state.mEurostoxx)
    mdow = st.checkbox("Dow Jones", st.session_state.mDow)
    mrussell = st.checkbox("Russell 2000", st.session_state.mRussell)
    mstoxxem = st.checkbox("Stoxx Emerging 50", st.session_state.mStoxxEmerging)
    mussmall = st.checkbox("US Small Cap 2000", st.session_state.mUSSmallCap)

sel = []
if msp500: sel.append("SP500")
if mnasdaq: sel.append("Nasdaq")
if mftse: sel.append("FTSE")
if meuro: sel.append("Eurostoxx")
if mdow: sel.append("Dow")
if mrussell: sel.append("Russell")
if mstoxxem: sel.append("StoxxEmerging")
if mussmall: sel.append("USSmallCap")

st.session_state.mSP500 = msp500
st.session_state.mNasdaq = mnasdaq
st.session_state.mFTSE = mftse
st.session_state.mEurostoxx = meuro
st.session_state.mDow = mdow
st.session_state.mRussell = mrussell
st.session_state.mStoxxEmerging = mstoxxem
st.session_state.mUSSmallCap = mussmall

with st.sidebar.expander("🎛️ Parametri Scanner", expanded=False):
    eh = st.slider("EARLY - Distanza EMA20 %", 0.0, 10.0, float(st.session_state.eh * 100), 0.5) / 100
    prmin = st.slider("PRO - RSI minimo", 0, 100, int(st.session_state.prmin), 5)
    prmax = st.slider("PRO - RSI massimo", 0, 100, int(st.session_state.prmax), 5)
    rpoc = st.slider("REA - Distanza POC %", 0.0, 10.0, float(st.session_state.rpoc * 100), 0.5) / 100
    vol_ratio_hot = st.number_input("VolRatio minimo REA-HOT", 0.0, 10.0, 1.5, 0.1)
    top = st.number_input("TOP N titoli per tab", 5, 100, int(st.session_state.top), 5)

st.session_state.eh = eh
st.session_state.prmin = prmin
st.session_state.prmax = prmax
st.session_state.rpoc = rpoc
st.session_state.top = top

st.sidebar.divider()
st.sidebar.subheader("📋 Gestione Watchlist")

df_wl_all = load_watchlist()
list_options = sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
if "DEFAULT" not in list_options:
    list_options.append("DEFAULT")

active_list = st.sidebar.selectbox(
    "Lista Attiva",
    list_options,
    index=list_options.index(st.session_state.current_list_name)
    if st.session_state.current_list_name in list_options
    else 0,
    key="active_list",
)
st.session_state.current_list_name = active_list

new_list = st.sidebar.text_input("Crea Nuova Watchlist")
if st.sidebar.button("Crea") and new_list.strip():
    st.session_state.current_list_name = new_list.strip()
    st.success(f"Lista '{new_list.strip()}' creata!")
    time.sleep(1)
    st.rerun()

if st.sidebar.button("Reset DB Completo", help="Elimina tutte le watchlist!"):
    reset_watchlist_db()
    st.rerun()

only_watchlist = st.sidebar.checkbox("Mostra solo Watchlist", value=False)

# -------------------------------------------------------------------------
# SCANNER
# -------------------------------------------------------------------------
if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 20.0", type="primary", use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            rep, rrea = [], []
            pb = st.progress(0)
            status = st.empty()
            tot = len(universe)
            for i, tkr in enumerate(universe, 1):
                status.text(f"Analisi {i}/{tot}: {tkr}")
                ep, rea = scan_ticker(tkr, eh, prmin, prmax, rpoc, vol_ratio_hot)
                if ep:
                    rep.append(ep)
                if rea:
                    rrea.append(rea)
                pb.progress(i / tot)
            st.session_state.df_ep = pd.DataFrame(rep)
            st.session_state.df_rea = pd.DataFrame(rrea)
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
            st.rerun()

df_ep = st.session_state.get("df_ep", pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

# -------------------------------------------------------------------------
# LEGENDA
# -------------------------------------------------------------------------
def show_legend(title: str):
    with st.expander(f"📖 Legenda {title}", expanded=False):
        if title == "EARLY":
            st.write("EARLY: titoli vicini alla EMA20 (trend in formazione).")
            st.write("- Early_Score: punteggio basato su vicinanza alla media e volumi.")
        elif title == "PRO":
            st.write("PRO: segnali di forza con RSI in zona neutrale-rialzista.")
            st.write("- Pro_Score: punteggio basato su trend, RSI e volumi.")
        elif title == "REA-HOT":
            st.write("REA-HOT: volumi anomali (Vol_Ratio >= soglia) e vicini al POC.")
            st.write("- Vol_Ratio: volume odierno / media 7gg.")
        else:
            st.write(f"Segnali scanner per il sistema {title}.")

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tabs = st.tabs(
    ["EARLY", "PRO", "REA-HOT", "Serafini Systems", "Regime Momentum", "Multi-Timeframe", "Finviz", "Watchlist"]
)
tab_e, tab_p, tab_r, tab_serafini, tab_regime, tab_mtf, tab_finviz, tab_w = tabs

# -------------------------------------------------------------------------
# FUNZIONE GENERICA TAB SCANNER
# -------------------------------------------------------------------------
def render_scan_tab(df: pd.DataFrame, status_filter: str, sort_cols, ascending, title: str):
    st.subheader(f"📊 Tab {title}")
    show_legend(title)

    if df.empty:
        st.info(f"Nessun dato {title}. Esegui lo scanner.")
        return

    # --- FILTRO STATO COMUNE (stile 9.x) ---
    col_f = None
    if status_filter == "EARLY":
        col_f = "Stato_Early"
    elif status_filter == "PRO":
        col_f = "Stato_Pro"
    elif status_filter in ("HOT", "SERAFINI", "REGIME", "MTF", "FINVIZ"):
        col_f = "Stato"

    if col_f and col_f in df.columns:
        if status_filter == "HOT":
            df_f = df[df[col_f] == "HOT"].copy()
        elif status_filter in ("SERAFINI", "REGIME", "MTF", "FINVIZ"):
            # Sistemi avanzati: usiamo come base solo i PRO
            df_f = df[df.get("Stato_Pro", "-") == "PRO"].copy()
        else:
            df_f = df[df[col_f] == status_filter].copy()
    else:
        df_f = df.copy()

    if df_f.empty:
        st.info(f"Nessun segnale {title} trovato.")
        return

    # --- LOGICHE SPECIFICHE PER TAB AVANZATI ---
    if status_filter == "REGIME":
        # Momentum = Pro_Score*10 + RSI
        if "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
            df_f["Momentum"] = df_f["Pro_Score"] * 10 + df_f["RSI"]
            sort_cols = ["Momentum"]
            ascending = [False]
    elif status_filter == "SERAFINI":
        # Serafini: privilegia i trend forti -> Pro_Score decrescente
        if "Pro_Score" in df_f.columns:
            sort_cols = ["Pro_Score"]
            ascending = [False]

    # --- ORDINAMENTO ---
    try:
        df_f = df_f.sort_values(sort_cols, ascending=ascending).head(st.session_state.top)
    except Exception:
        pass

    # --- FORMATTAZIONE E RIMOZIONE COLONNE LINK ---
    df_fmt = add_formatted_cols(df_f)
    df_disp = prepare_display_df(df_fmt)

    for c in ["Yahoo", "TradingView"]:
        if c in df_disp.columns:
            df_disp = df_disp.drop(columns=[c])

    cols = list(df_disp.columns)
    base_cols = ["Ticker"]
    if "Nome" in df_disp.columns:
        base_cols.append("Nome")
    for c in base_cols:
        if c in cols:
            cols.remove(c)
    ordered = base_cols + cols
    df_disp = df_disp[[c for c in ordered if c in df_disp.columns]]

    # --- EXPORT E ISTRUZIONI ---
    c1, c2 = st.columns([1, 1])
    with c1:
        get_csv_download_button(df_f, f"{title.lower()}_export.csv", key=f"exp_{title}")
    with c2:
        st.markdown(
            f"Seleziona righe nella tabella e clicca **Aggiungi selezionati** a '{st.session_state.current_list_name}'."
        )

    # --- AGGRID: doppio click su Nome -> TradingView ---
    gb = GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(sortable=True, resizable=True, filterable=True, editable=False)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)

    if "Nome" in df_disp.columns:
        gb.configure_column(
            "Nome",
            headerName="Nome",
            cellRenderer=name_dblclick_renderer,
        )

    grid_options = gb.build()

    grid_response = AgGrid(
        df_disp,
        gridOptions=grid_options,
        height=600,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        allow_unsafe_jscode=True,
        key=f"grid_{title}",
    )

    selected_rows = grid_response["selected_rows"]
    selected_df = pd.DataFrame(selected_rows)

    if st.button(f"Aggiungi selezionati a '{st.session_state.current_list_name}'", key=f"btn_{title}"):
        if not selected_df.empty and "Ticker" in selected_df.columns:
            tickers = selected_df["Ticker"].tolist()
            names = selected_df["Nome"].tolist() if "Nome" in selected_df.columns else tickers
            add_to_watchlist(
                tickers,
                names,
                title,
                "Scanner",
                "LONG",
                st.session_state.current_list_name,
            )
            st.success(f"Aggiunti {len(tickers)} titoli alla watchlist!")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Nessuna riga selezionata.")

# -------------------------------------------------------------------------
# RENDER TABS SCANNER
# -------------------------------------------------------------------------
with tab_e:
    st.session_state.last_active_tab = "EARLY"
    render_scan_tab(df_ep, "EARLY", ["Early_Score", "RSI"], [False, True], "EARLY")

with tab_p:
    st.session_state.last_active_tab = "PRO"
    render_scan_tab(df_ep, "PRO", ["Pro_Score", "RSI"], [False, True], "PRO")

with tab_r:
    st.session_state.last_active_tab = "REA-HOT"
    render_scan_tab(df_rea, "HOT", ["Vol_Ratio", "Dist_POC_%"], [False, True], "REA-HOT")

with tab_serafini:
    render_scan_tab(df_ep, "SERAFINI", ["Pro_Score"], [False], "Serafini Systems")

with tab_regime:
    render_scan_tab(df_ep, "REGIME", ["Pro_Score"], [False], "Regime Momentum")

with tab_mtf:
    render_scan_tab(df_ep, "MTF", ["Ticker"], [True], "Multi-Timeframe")

with tab_finviz:
    render_scan_tab(df_ep, "FINVIZ", ["Ticker"], [True], "Finviz")

# -------------------------------------------------------------------------
# TAB WATCHLIST
# -------------------------------------------------------------------------
with tab_w:
    st.subheader(f"📋 Watchlist '{st.session_state.current_list_name}'")
    df_w_view = load_watchlist()
    df_w_view = df_w_view[df_w_view["list_name"] == st.session_state.current_list_name]

    if df_w_view.empty:
        st.info("Watchlist vuota.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            get_csv_download_button(
                df_w_view,
                f"watchlist_{st.session_state.current_list_name}.csv",
                key="exp_wl",
            )
        with c2:
            move_target = st.selectbox("Sposta in", list_options, key="move_target")
            ids_to_move = st.multiselect("Seleziona ID", df_w_view["id"].tolist())
            if st.button("Sposta"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                for i in ids_to_move:
                    c.execute("UPDATE watchlist SET list_name=? WHERE id=?", (move_target, i))
                conn.commit()
                conn.close()
                st.rerun()
        with c3:
            if st.button("Elimina selezionati"):
                st.warning("Da implementare: delete by ID sul DB.")

        df_wv = add_links(prepare_display_df(add_formatted_cols(df_w_view)))
        st.write(df_wv.to_html(escape=False, index=False), unsafe_allow_html=True)

    if st.button("🔄 Refresh Data"):
        st.rerun()

# -------------------------------------------------------------------------
# EXPORT GLOBALI (4 EXPORT)
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("💾 Export Globali")

all_tabs_raw = {
    "EARLY": df_ep,
    "PRO": df_ep,
    "REA-HOT": df_rea,
    "Watchlist": df_w_view if "df_w_view" in locals() else pd.DataFrame(),
}

# 1) XLSX tutti i tab
xlsx_all = to_excel_bytes_sheets_dict(all_tabs_raw)
st.download_button(
    label="📊 Export XLSX Tutti i tab",
    data=xlsx_all,
    file_name="TradingScanner_Tutti_i_tab.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="xlsx_all_tabs",
)

# 2) CSV TradingView tutti i tab
tv_rows = []
for name, df_tab in all_tabs_raw.items():
    if isinstance(df_tab, pd.DataFrame) and not df_tab.empty and "Ticker" in df_tab.columns:
        tmp = df_tab[["Ticker"]].copy()
        tmp.insert(0, "Tab", name)
        tv_rows.append(tmp)
if tv_rows:
    df_tv_all = pd.concat(tv_rows, ignore_index=True)
    csv_tv_all = df_tv_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📈 Export CSV TradingView Tutti i tab",
        data=csv_tv_all,
        file_name="TradingScanner_Tutti_i_tab_TradingView.csv",
        mime="text/csv",
        key="csv_tv_all_tabs",
    )

# 3) XLSX tab corrente
current_tab = st.session_state.get("last_active_tab", "EARLY")
df_current = all_tabs_raw.get(current_tab, pd.DataFrame())
xlsx_current = to_excel_bytes_sheets_dict({current_tab: df_current})
st.download_button(
    label=f"📊 Export XLSX Tab corrente ({current_tab})",
    data=xlsx_current,
    file_name=f"TradingScanner_{current_tab}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="xlsx_current_tab",
)

# 4) CSV TradingView tab corrente
if isinstance(df_current, pd.DataFrame) and not df_current.empty and "Ticker" in df_current.columns:
    csv_tv_current = make_tv_csv(df_current, current_tab, "Ticker")
    st.download_button(
        label=f"📈 Export CSV TradingView Tab corrente ({current_tab})",
        data=csv_tv_current,
        file_name=f"TradingScanner_{current_tab}_TradingView.csv",
        mime="text/csv",
        key="csv_tv_current_tab",
    )

