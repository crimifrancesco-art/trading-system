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

# JS per rendere i link cliccabili
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

# --- FUNZIONE EXPORT (I 4 EXPORT RICHIESTI) ---
def render_export_buttons(df, filename):
    if df.empty: return
    st.write("📥 **Export:**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.download_button("CSV", df.to_csv(index=False), f"{filename}.csv", "text/csv")
    with c2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button("EXCEL", output.getvalue(), f"{filename}.xlsx")
    with c3: st.download_button("JSON", df.to_json(orient='records'), f"{filename}.json", "application/json")
    with c4: st.download_button("HTML", df.to_html(index=False), f"{filename}.html", "text/html")
    with c5:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=8)
        pdf.cell(200, 10, txt=f"Export: {filename}", ln=1, align='C')
        for i, row in df.head(30).iterrows():
            pdf.cell(200, 7, txt=str(row.tolist())[:100], ln=1)
        # Correzione AttributeError: gestisce diverse versioni di fpdf
        pdf_out = pdf.output(dest='S')
        if isinstance(pdf_out, str): pdf_out = pdf_out.encode('latin-1')
        st.download_button("PDF", pdf_out, f"{filename}.pdf")

# --- FUNZIONE RENDER TAB (CORRETTA) ---
def render_scan_tab(df, tab_title, filter_col=None, filter_val=None):
    if df.empty:
        st.info(f"Nessun dato per {tab_title}")
        return
    
    dff = df.copy()
    if filter_col and filter_val:
        dff = dff[dff[filter_col] == filter_val] if not isinstance(filter_val, list) else dff[dff[filter_col].isin(filter_val)]

    if dff.empty:
        st.warning("Nessun match per i filtri.")
        return

    # Visualizzazione
    display_df = add_formatted_cols(dff)
    display_df = prepare_display_df(display_df)
    display_df = add_links(display_df)

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    gb.configure_column("Yahoo", cellRenderer=js_link_code)
    gb.configure_column("Tradingview", cellRenderer=js_link_code)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    
    grid_res = AgGrid(
        display_df, 
        gridOptions=gb.build(), 
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True, # RESIZE AUTOMATICO
        allow_unsafe_jscode=True,
        theme="streamlit"
    )

    # Watchlist logic (senza checkbox esterno)
    selected = grid_res['selected_rows']
    if st.button(f"Aggiungi selezionati ({tab_title})", key=f"btn_{tab_title}"):
        if selected is not None and len(selected) > 0:
            sel_df = pd.DataFrame(selected)
            add_to_watchlist(sel_df["Ticker"].tolist(), sel_df["Nome"].tolist(), tab_title, "Scan")
            st.success("Aggiunti!")

    render_export_buttons(dff, tab_title)

# --- AVVIO ---
if st.button("🚀 AVVIA SCANNER", type="primary"):
    universe = load_universe(selected_markets)
    res_ep, res_rea = [], []
    pb = st.progress(0)
    for i, tkr in enumerate(universe):
        ep, rea = scan_ticker(tkr, 0.02, 40, 70, 0.02)
        if ep: res_ep.append(ep)
        if rea: res_rea.append(rea)
        pb.progress((i+1)/len(universe))
    st.session_state["df_ep"] = pd.DataFrame(res_ep)
    st.session_state["df_rea"] = pd.DataFrame(res_rea)

df_ep = st.session_state.get("df_ep", pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

# --- TAB ORIGINALI ---
t_early, t_pro, t_rea, t_serafini, t_regime, t_mtf, t_finviz, t_w = st.tabs([
    "🟢 EARLY", "🟣 PRO", "🟠 REA-HOT", "🏆 SERAFINI", "📊 REGIME", "⏱️ MTF", "🔍 FINVIZ", "📌 WATCHLIST"
])

with t_early: render_scan_tab(df_ep, "EARLY", "Stato_Early", "EARLY")
with t_pro: render_scan_tab(df_ep, "PRO", "Stato_Pro", "PRO")
with t_rea: render_scan_tab(df_rea, "REA-HOT", "Stato", "HOT")
with t_serafini: render_scan_tab(df_ep, "SERAFINI", "Stato", "HOT") # Filtro corretto
with t_regime: render_scan_tab(df_ep, "REGIME", "Stato_Pro", "PRO")
with t_mtf: render_scan_tab(df_ep, "MTF")
with t_finviz: render_scan_tab(df_ep, "FINVIZ")
with t_w:
    st.dataframe(load_watchlist(), use_container_width=True)
