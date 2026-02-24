import io
import time
import sqlite3
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from fpdf import FPDF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# Import moduli locali
from utils.formatting import fmt_currency, fmt_int, fmt_marketcap, add_formatted_cols, add_links, prepare_display_df
from utils.db import init_db, reset_watchlist_db, add_to_watchlist, load_watchlist, DB_PATH
from utils.scanner import load_universe, scan_ticker

st.set_page_config(page_title="Trading Scanner PRO 20.0", layout="wide", page_icon="📊")

# --- JS CODE PER LINK CLICCABILI ---
js_link_code = JsCode("""
function(params) {
    if (params.value.includes('href')) {
        return params.value;
    }
    return params.value;
}
""")

# --- INIZIALIZZAZIONE ---
init_db()
for key, val in {"m_SP500": True, "m_Nasdaq": True, "m_FTSE": True, "e_h": 0.02, 
                "p_rmin": 40, "p_rmax": 70, "r_poc": 0.02, "top": 15, 
                "current_list_name": "DEFAULT"}.items():
    st.session_state.setdefault(key, val)

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configurazione")
with st.sidebar.expander("📈 Mercati", expanded=True):
    sel = [m for m in ["SP500", "Nasdaq", "FTSE", "Eurostoxx", "Dow", "Russell"] if st.sidebar.checkbox(m, True)]

# --- FUNZIONI EXPORT ---
def export_buttons(df, title):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.download_button("CSV", df.to_csv(index=False), f"{title}.csv", "text/csv")
    with c2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button("Excel", output.getvalue(), f"{title}.xlsx")
    with c3: st.download_button("JSON", df.to_json(orient='records'), f"{title}.json", "application/json")
    with c4: st.download_button("HTML", df.to_html(index=False), f"{title}.html", "text/html")
    with c5:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Export {title}", ln=1, align='C')
        for i, row in df.head(20).iterrows():
            pdf.cell(200, 10, txt=str(row.tolist())[:100], ln=1)
        st.download_button("PDF (Preview)", pdf.output(dest='S').encode('latin-1'), f"{title}.pdf")

# --- LOGICA SCANNER ---
if st.button("🚀 AVVIA SCANNER PRO 20.0", type="primary", use_container_width=True):
    universe = load_universe(sel)
    results_ep, results_rea = [], []
    pb = st.progress(0)
    for i, tkr in enumerate(universe):
        ep, rea = scan_ticker(tkr, st.session_state.e_h, st.session_state.p_rmin, st.session_state.p_rmax, st.session_state.r_poc)
        if ep: results_ep.append(ep)
        if rea: results_rea.append(rea)
        pb.progress((i + 1) / len(universe))
    st.session_state["df_ep"] = pd.DataFrame(results_ep)
    st.session_state["df_rea"] = pd.DataFrame(results_rea)
    st.rerun()

df_ep = st.session_state.get("df_ep", pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

# --- RENDER TAB ---
def render_tab(df, filter_val, title):
    st.subheader(f"Segnali {title}")
    if df.empty:
        st.info("Nessun dato disponibile.")
        return
    
    # Filtro logico per tab
    df_f = df.copy()
    if filter_val in ["EARLY", "PRO", "HOT"]:
        col = "Stato_Early" if filter_val == "EARLY" else ("Stato_Pro" if filter_val == "PRO" else "Stato")
        df_f = df[df[col] == filter_val]
    
    df_disp = add_links(prepare_display_df(add_formatted_cols(df_f)))
    
    gb = GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    gb.configure_column("Yahoo", cellRenderer=js_link_code)
    gb.configure_column("TradingView", cellRenderer=js_link_code)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    
    grid_res = AgGrid(df_disp, gridOptions=gb.build(), height=400, 
                      fit_columns_on_grid_load=True, 
                      allow_unsafe_jscode=True, 
                      update_mode=GridUpdateMode.SELECTION_CHANGED)
    
    selected = grid_res['selected_rows']
    if st.button(f"Aggiungi selezionati ({len(selected) if selected is not None else 0})", key=f"btn_{title}"):
        if selected is not None and len(selected) > 0:
            sel_df = pd.DataFrame(selected)
            add_to_watchlist(sel_df["Ticker"].tolist(), sel_df["Nome"].tolist(), title, "Scanner", "LONG", st.session_state.current_list_name)
            st.success("Aggiunti!")
    
    export_buttons(df_f, title)

tabs = st.tabs(["🟢 EARLY", "🟣 PRO", "🟠 REA-HOT", "📈 Altri", "📌 Watchlist"])
with tabs[0]: render_tab(df_ep, "EARLY", "EARLY")
with tabs[1]: render_tab(df_ep, "PRO", "PRO")
with tabs[2]: render_tab(df_rea, "HOT", "REA-HOT")
with tabs[3]: render_tab(df_ep, "ALL", "SCANNER-ALL")
with tabs[4]:
    wl = load_watchlist()
    st.dataframe(wl)
