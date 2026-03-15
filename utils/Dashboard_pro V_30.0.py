# Dashboard_pro_V_32.py
# Trading Scanner PRO Dashboard (clean stable version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

from scanner import scan_universe
from db import init_db, add_ticker, remove_ticker, get_watchlist

st.set_page_config(
    page_title="Trading Scanner PRO",
    layout="wide"
)

init_db()

# ----------------------------------------------------
# CARICAMENTO UNIVERSO TITOLI
# ----------------------------------------------------

@st.cache_data
def load_universe():

    sp500 = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]["Symbol"].tolist()

    return sp500


# ----------------------------------------------------
# SIDEBAR FILTRI
# ----------------------------------------------------

st.sidebar.title("Scanner Settings")

workers = st.sidebar.slider(
    "Scanner Threads",
    2,
    20,
    8
)

min_score = st.sidebar.slider(
    "Minimum Score",
    0,
    100,
    30
)

rsi_min = st.sidebar.slider(
    "RSI Min",
    0,
    100,
    40
)

volume_ratio = st.sidebar.slider(
    "Volume Ratio Min",
    0.5,
    5.0,
    1.2
)

# ----------------------------------------------------
# SCAN BUTTON
# ----------------------------------------------------

if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()

if st.sidebar.button("Run Scanner"):

    universe = load_universe()

    with st.spinner("Scanning market..."):

        df = scan_universe(
            universe,
            workers=workers
        )

    st.session_state.scan_results = df


df = st.session_state.scan_results

# ----------------------------------------------------
# FILTRI CORRETTI
# ----------------------------------------------------

if not df.empty:

    df_filtered = df[
        (df["Score"] >= min_score)
        & (df["RSI"] >= rsi_min)
        & (df["Vol_Ratio"] >= volume_ratio)
    ]

else:
    df_filtered = df

# ----------------------------------------------------
# KPI BAR
# ----------------------------------------------------

if not df_filtered.empty:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Signals",
        len(df_filtered)
    )

    col2.metric(
        "Avg Score",
        round(df_filtered["Score"].mean(), 1)
    )

    col3.metric(
        "High RSI",
        len(df_filtered[df_filtered["RSI"] > 70])
    )

    col4.metric(
        "High Volume",
        len(df_filtered[df_filtered["Vol_Ratio"] > 2])
    )

# ----------------------------------------------------
# TABELLA RISULTATI
# ----------------------------------------------------

st.subheader("Scanner Results")

st.dataframe(
    df_filtered,
    use_container_width=True
)

# ----------------------------------------------------
# WATCHLIST
# ----------------------------------------------------

st.subheader("Watchlist")

watchlist = get_watchlist()

ticker_add = st.text_input("Add Ticker")

if st.button("Add to Watchlist"):

    if ticker_add:
        add_ticker(ticker_add.upper())

if not watchlist.empty:

    st.dataframe(watchlist)

# ----------------------------------------------------
# CHART
# ----------------------------------------------------

st.subheader("Chart")

ticker_chart = st.text_input("Ticker")

if ticker_chart:

    data = yf.download(
        ticker_chart,
        period="6mo",
        interval="1d"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
