import io
import sqlite3
import pandas as pd
import streamlit as st
from fpdf import FPDF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# Import moduli locali
from utils.formatting import add_formatted_cols, add_links, prepare_display_df
from utils.db import init_db, add_to_watchlist, load_watchlist
from utils.scanner import load_universe, scan_ticker

st.set_page_config(page_title="Trading Scanner PRO 20.0", layout="wide", page_icon="📊")

# JS per rendere i link HTML cliccabili dentro AgGrid
js_link_code = JsCode("""
function(params) {
    return params.value;
}
""")

init_db()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configurazione")
markets = ["SP500", "Nasdaq", "FTSE", "Eurostoxx", "Dow", "Russell"]
selected_markets = [m for m in markets if st.sidebar.checkbox(m, value=True)]

# --- FUNZIONE EXPORT MULTIPLO ---
def render_export_buttons(df, filename):
    if df.empty: return
    st.write("📥 **Export dati:**")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.download_button("CSV", df.to_csv(index=False), f"{filename}.csv", "text/csv")
    with c2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button("Excel", output.getvalue(), f"{filename}.xlsx")
    with c3:
        st.download_button("JSON", df.to_json(orient='records'), f"{filename}.json", "application/json")
    with c4:
        st.download_button("HTML", df.to_html(index=False), f"{filename}.html", "text/html")
    with c5:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=8)
        pdf.cell(200, 10, txt=f"Export: {filename}", ln=1, align='C')
        for i, row in df.head(50).iterrows(): # Limite righe per PDF veloce
            pdf.cell(200, 7, txt=str(row.tolist())[:120], ln=1)
        st.download_button("PDF (Anteprima)", pdf.output(dest='S').encode('latin-1'), f"{filename}.pdf")

# --- LOGICA SCANNER ---
if st.button("🚀 AVVIA SCANNER PRO 20.0", type="primary", use_container_width=True):
    universe = load_universe(selected_markets)
    results_ep, results_rea = [], []
    pb = st.progress(0)
    for i, tkr in enumerate(universe):
        ep, rea = scan_ticker(tkr, 0.02, 40, 70, 0.02)
        if ep: results_ep.append(ep)
        if rea: results_rea.append(rea)
        pb.progress((i + 1) / len(universe))
    st.session_state["df_ep"] = pd.DataFrame(results_ep)
    st.session_state["df_rea"] = pd.DataFrame(results_rea)

df_ep = st.session_state.get("df_ep", pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

# --- RENDER TABELLE ---
def render_tab_data(df, tab_title, filter_col=None, filter_val=None):
    if df.empty:
        st.info(f"Nessun dato per {tab_title}")
        return

    dff = df.copy()
    if filter_col and filter_val:
        dff = dff[dff[filter_col] == filter_val]

    # Preparazione display con link
    display_df = prepare_display_df(add_formatted_cols(dff))
    display_df = add_links(display_df)

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    gb.configure_column("Yahoo", cellRenderer=js_link_code)
    gb.configure_column("TradingView", cellRenderer=js_link_code)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    
    # Auto-resize attivo: fit_columns_on_grid_load=True
    grid_res = AgGrid(
        display_df, 
        gridOptions=gb.build(), 
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="streamlit"
    )

    selected = grid_res['selected_rows']
    if st.button(f"Aggiungi selezionati alla Watchlist ({tab_title})"):
        if selected is not None and len(selected) > 0:
            sel_df = pd.DataFrame(selected)
            add_to_watchlist(sel_df["Ticker"].tolist(), sel_df["Nome"].tolist(), tab_title, "Scan")
            st.success(f"{len(selected)} titoli aggiunti!")

    render_export_buttons(dff, tab_title)

# --- TABS ---
t1, t2, t3, t4 = st.tabs(["🟢 EARLY", "🟣 PRO", "🟠 REA-HOT", "📌 WATCHLIST"])
with t1: render_tab_data(df_ep, "EARLY", "Stato_Early", "EARLY")
with t2: render_tab_data(df_ep, "PRO", "Stato_Pro", "PRO")
with t3: render_tab_data(df_rea, "REA-HOT", "Stato", "HOT")
with t4:
    wl = load_watchlist()
    st.dataframe(wl, use_container_width=True)
