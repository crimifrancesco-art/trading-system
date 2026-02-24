import io
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from fpdf import FPDF
from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
)

# utils
from utils.formatting import (
    add_formatted_cols,
    add_links,
    prepare_display_df,
)
from utils.db import (
    init_db,
    add_to_watchlist,
    load_watchlist,
    reset_watchlist_db,
)
from utils.scanner import load_universe, scan_ticker

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 20.0",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner – Versione PRO 20.0")

init_db()

# -----------------------------------------------------------------------------
# SESSION DEFAULT
# -----------------------------------------------------------------------------
defaults = {
    "top": 15,
    "current_list_name": "DEFAULT",
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurazione")

markets = {
    "S&P500": "SP500",
    "Nasdaq": "Nasdaq",
    "FTSE": "FTSE",
}

selected_markets = [
    code for name, code in markets.items()
    if st.sidebar.checkbox(name, True)
]

# WATCHLIST
st.sidebar.subheader("📁 Watchlist")

df_wl_all = load_watchlist()

lists = (
    sorted(df_wl_all["list_name"].unique())
    if not df_wl_all.empty else ["DEFAULT"]
)

active_list = st.sidebar.selectbox("Lista Attiva", lists)
st.session_state["current_list_name"] = active_list

if st.sidebar.button("🗑 Reset DB"):
    reset_watchlist_db()
    st.rerun()

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------
def export_buttons(df, name):

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.download_button("CSV",
                           df.to_csv(index=False),
                           f"{name}.csv")

    with c2:
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button("Excel",
                           buffer.getvalue(),
                           f"{name}.xlsx")

    with c3:
        st.download_button("JSON",
                           df.to_json(orient="records"),
                           f"{name}.json")

    with c4:
        st.download_button("HTML",
                           df.to_html(index=False),
                           f"{name}.html")

    with c5:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=8)

        for _, r in df.head(40).iterrows():
            pdf.cell(0, 5, str(r.values), ln=1)

        st.download_button(
            "PDF",
            pdf.output(dest="S").encode("latin-1"),
            f"{name}.pdf",
        )

# -----------------------------------------------------------------------------
# SCANNER
# -----------------------------------------------------------------------------
if st.button("🚀 AVVIA SCANNER", use_container_width=True):

    universe = load_universe(selected_markets)

    results = []
    pb = st.progress(0)

    for i, tkr in enumerate(universe):
        ep, _ = scan_ticker(tkr, 0.02, 40, 70, 0.02, 1.5)
        if ep:
            results.append(ep)

        pb.progress((i + 1) / len(universe))

    st.session_state["df_scan"] = pd.DataFrame(results)
    st.session_state["last_scan"] = datetime.now().strftime("%H:%M:%S")

    st.rerun()

df_scan = st.session_state.get("df_scan", pd.DataFrame())

# -----------------------------------------------------------------------------
# AGGRID LINK RENDERER
# -----------------------------------------------------------------------------
link_renderer = JsCode("""
function(params){
    if(!params.value){return ''}
    return `<a href="${params.value}" target="_blank">Apri</a>`
}
""")

# -----------------------------------------------------------------------------
# TAB RENDER
# -----------------------------------------------------------------------------
def render_tab(df, title):

    st.subheader(title)

    if df.empty:
        st.info("Nessun dato disponibile")
        return

    df_v = add_links(
        prepare_display_df(
            add_formatted_cols(df)
        )
    )

    export_buttons(df, title)

    gb = GridOptionsBuilder.from_dataframe(df_v)

    gb.configure_default_column(
        sortable=True,
        resizable=True,
        filter=True,
    )

    gb.configure_column("Yahoo", cellRenderer=link_renderer)
    gb.configure_column("TradingView", cellRenderer=link_renderer)

    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
    )

    grid_options = gb.build()

    grid = AgGrid(
        df_v,
        gridOptions=grid_options,
        height=600,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
    )

    selected = grid["selected_rows"]

    if st.button(f"➕ Aggiungi selezionati ({title})"):
        if selected:
            sel_df = pd.DataFrame(selected)

            add_to_watchlist(
                sel_df["Ticker"].tolist(),
                sel_df["Nome"].tolist(),
                title,
                "Scanner",
                "LONG",
                active_list,
            )

            st.success("Titoli aggiunti alla watchlist")
            time.sleep(1)
            st.rerun()

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🟢 EARLY",
    "🟣 PRO",
    "🟠 REA-HOT",
    "📈 Serafini",
    "🧊 Regime",
    "🕒 MTF",
    "📊 Finviz",
    "📌 Watchlist",
])

for t in tabs[:-1]:
    with t:
        render_tab(df_scan, t.label)

# -----------------------------------------------------------------------------
# WATCHLIST TAB
# -----------------------------------------------------------------------------
with tabs[-1]:

    st.subheader(f"Watchlist — {active_list}")

    df_w = load_watchlist()
    df_w = df_w[df_w["list_name"] == active_list]

    if df_w.empty:
        st.info("Watchlist vuota")
    else:
        st.dataframe(df_w, use_container_width=True)

    if st.button("🔄 Refresh"):
        st.rerun()
