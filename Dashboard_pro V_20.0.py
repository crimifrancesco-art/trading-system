import io
import time
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

from utils.formatting import add_formatted_cols, prepare_display_df, add_links
from utils.db import init_db, add_to_watchlist, load_watchlist
from utils.scanner import load_universe, scan_ticker

st.set_page_config(layout="wide", page_title="Trading Scanner PRO 20")

init_db()

st.title("📊 Trading Scanner PRO 20")

# -----------------------------------------------------------------------------
# SCAN
# -----------------------------------------------------------------------------
if st.button("🚀 AVVIA SCANNER"):

    universe = load_universe(["SP500","Nasdaq","FTSE"])

    res = []

    for t in universe:
        ep,_ = scan_ticker(t,0,0,0,0,0)
        if ep:
            res.append(ep)

    st.session_state["df"] = pd.DataFrame(res)
    st.rerun()

df = st.session_state.get("df", pd.DataFrame())

# -----------------------------------------------------------------------------
# GRID RENDER
# -----------------------------------------------------------------------------
def render_grid(df):

    if df.empty:
        st.info("Nessun dato")
        return None

    df_v = add_links(
        prepare_display_df(
            add_formatted_cols(df)
        )
    )

    link_renderer = JsCode("""
    function(params){
        if(!params.value){return ''}
        return `<a href="${params.value}" target="_blank">Apri</a>`
    }
    """)

    gb = GridOptionsBuilder.from_dataframe(df_v)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True
    )

    gb.configure_column("Yahoo", cellRenderer=link_renderer)
    gb.configure_column("TradingView", cellRenderer=link_renderer)

    gb.configure_selection("multiple", use_checkbox=True)

    grid = AgGrid(
        df_v,
        gridOptions=gb.build(),
        height=600,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        allow_unsafe_jscode=True,
    )

    return grid["selected_rows"]


selected = render_grid(df)

if st.button("➕ Aggiungi selezionati"):
    if selected:
        sel = pd.DataFrame(selected)

        add_to_watchlist(
            sel["Ticker"].tolist(),
            sel["Nome"].tolist()
        )

        st.success("Aggiunti!")
        time.sleep(1)
        st.rerun()

# -----------------------------------------------------------------------------
# WATCHLIST
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📌 Watchlist")

st.dataframe(load_watchlist(), use_container_width=True)
