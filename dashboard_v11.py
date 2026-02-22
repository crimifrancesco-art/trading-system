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
    "FTSE": ["ENI.MI", "ISP.MI", "UCG.MI", "STM.MI"],
    "SP500": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
    "Nasdaq": ["TSLA", "AMD", "AVGO", "INTC"],
    "ETF": ["SPY", "QQQ", "IWM"]
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
# BUILD UNIVERSE
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
runtime_path.parent.mkdir(parents=True, exist_ok=True)


def save_runtime(tickers):
    runtime_path.write_text(json.dumps({"tickers": tickers}, indent=2))

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
    try:
        results = json.loads(result_path.read_text())
        df = pd.DataFrame(results)

        if not df.empty:
            st.subheader("📊 Risultati Scan")

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
            c1.metric("STRONG BUY", int((df["signal"] == "STRONG BUY").sum()))
            c2.metric("BUY", int((df["signal"] == "BUY").sum()))
            c3.metric("Assets scansionati", len(df))

        else:
            st.info("Scan eseguito ma nessun segnale trovato. Premi RUN SCAN di nuovo.")

    except Exception as e:
        st.error(f"Errore lettura risultati: {e}")

else:
    st.info("Premi RUN SCAN per avviare lo scanner.")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.caption(
    f"Aggiornato: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
