# ==========================================================
# DASHBOARD V11 — PROFESSIONAL SCANNER
# ==========================================================

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from run_scan import run_scan

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="Trading Scanner V11",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner — V11 Professional")

# ----------------------------------------------------------
# MARKET UNIVERSES
# ----------------------------------------------------------

MARKETS = {

    "FTSE": [
        "ENI.MI","ISP.MI","UCG.MI","STM.MI"
    ],

    "SP500": [
        "AAPL","MSFT","NVDA","AMZN","META"
    ],

    "Nasdaq": [
        "TSLA","AMD","AVGO","INTC"
    ],

    "ETF": [
        "SPY","QQQ","IWM"
    ]
}

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("⚙️ Scanner Settings")

selected_markets = []

for market in MARKETS:
    if st.sidebar.checkbox(market, value=True):
        selected_markets.append(market)

run_scan_btn = st.sidebar.button("▶ RUN SCAN")

# ----------------------------------------------------------
# BUILD UNIVERSE (⭐ FIX V11)
# ----------------------------------------------------------

def build_universe(markets):

    tickers = []

    for m in markets:
        tickers.extend(MARKETS[m])

    return sorted(set(tickers))


active_tickers = build_universe(selected_markets)

st.sidebar.write(f"Ticker attivi: {len(active_tickers)}")

# ----------------------------------------------------------
# SAVE RUNTIME UNIVERSE
# ----------------------------------------------------------

runtime_path = Path("data/runtime_universe.json")
runtime_path.parent.mkdir(exist_ok=True)

def save_runtime(tickers):

    with open(runtime_path, "w") as f:
        json.dump({"tickers": tickers}, f, indent=2)

# ----------------------------------------------------------
# RUN SCAN
# ----------------------------------------------------------

if run_scan_btn:

    if not active_tickers:
        st.warning("Seleziona almeno un mercato.")
        st.stop()

    save_runtime(active_tickers)

    with st.spinner("Scanning markets..."):
        try:
            run_scan()
            st.success("Scan completato ✅")
        except Exception as e:
            st.error(f"Errore durante lo scan: {e}")


# ----------------------------------------------------------
# LOAD RESULTS
# ----------------------------------------------------------

result_path = Path("data/scan_results.json")

if result_path.exists():

    results = json.loads(result_path.read_text())
    df = pd.DataFrame(results)

    if not df.empty:

        st.subheader("📊 Risultati")

        def color_signal(val):
            if val == "STRONG BUY":
                return "background-color:#0f5132;color:white"
            if val == "BUY":
                return "background-color:#664d03;color:white"
            return ""

        st.dataframe(
            df.style.applymap(color_signal, subset=["signal"]),
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("STRONG BUY", (df.signal=="STRONG BUY").sum())
        c2.metric("BUY", (df.signal=="BUY").sum())
        c3.metric("Assets", len(df))

else:
    st.info("Esegui uno scan.")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.caption(
    f"Aggiornato: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
