# ==========================================================
# DASHBOARD V11 — PROFESSIONAL SCANNER (MODULAR)
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
    page_title="Trading Scanner V11 Professional",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner — V11 Professional")

# ----------------------------------------------------------
# MARKET UNIVERSES (Sync with V9 and run_scan)
# ----------------------------------------------------------

MARKETS = {
    "FTSE": ["ENI.MI", "ISP.MI", "UCG.MI", "STM.MI", "ENEL.MI", "LDO.MI", "PRY.MI", "TEN.MI", "A2A.MI", "AMP.MI"],
    "SP500": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "V"],
    "Nasdaq": ["TSLA", "AMD", "AVGO", "INTC", "NFLX", "ADBE", "COST", "PEP", "CSCO", "AMD"],
    "ETF": ["SPY", "QQQ", "IWM", "GLD", "TLT", "VTI", "EEM"],
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
    "Eurostoxx": ["ASML.AS", "MC.PA", "OR.PA", "TTE.PA", "AIR.PA", "SAN.PA"]
}

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("⚙️ Configurazione")

st.sidebar.subheader("📈 Selezione Mercati")
selected_markets = []
for market in MARKETS:
    if st.sidebar.checkbox(market, value=True if market in ["FTSE", "SP500", "Nasdaq"] else False):
        selected_markets.append(market)

st.sidebar.divider()

st.sidebar.subheader("📤 Azioni")
run_scan_btn = st.sidebar.button("🚀 AVVIA SCANNER V11", type="primary", use_container_width=True)

# ----------------------------------------------------------
# BUILD UNIVERSE
# ----------------------------------------------------------

def build_universe(markets):
    tickers = []
    for m in markets:
        tickers.extend(MARKETS[m])
    return sorted(set(tickers))


active_tickers = build_universe(selected_markets)
st.sidebar.write(f"Ticker attivi: **{len(active_tickers)}**")

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

    with st.spinner("Scansione mercati in corso..."):
        try:
            run_scan()
            st.success("Scan completato ✅")
        except Exception as e:
            st.error(f"Errore durante lo scan: {e}")

# ----------------------------------------------------------
# LOAD RESULTS & TABS
# ----------------------------------------------------------

result_path = Path("data/scan_results.json")

tab_results, tab_legend = st.tabs(["📊 Risultati Scan", "📘 Legenda Filtri"])

with tab_results:
    if result_path.exists():
        try:
            results = json.loads(result_path.read_text())
            df = pd.DataFrame(results)

            if not df.empty:
                # Metrics
                c1, c2, c3 = st.columns(3)
                n_strong = (df["signal"] == "STRONG BUY").sum()
                n_buy = (df["signal"] == "BUY").sum()
                
                c1.metric("STRONG BUY", int(n_strong))
                c2.metric("BUY", int(n_buy))
                c3.metric("Assets scansionati", len(df))

                # Table formatting
                def color_signal(val):
                    if val == "STRONG BUY":
                        return "background-color:#0f5132;color:white"
                    if val == "BUY":
                        return "background-color:#664d03;color:white"
                    return ""

                st.subheader("Tabella Segnali")
                st.dataframe(
                    df.style.applymap(color_signal, subset=["signal"]),
                    use_container_width=True,
                )
            else:
                st.info("Nessun segnale trovato. Prova a cambiare mercati o avviare un nuovo scan.")
        except Exception as e:
            st.error(f"Errore lettura risultati: {e}")
    else:
        st.info("Premi 'AVVIA SCANNER' nella sidebar per vedere i risultati.")

with tab_legend:
    st.subheader("Spiegazione Filtri V11")
    st.markdown(\"\"\"
    Lo scanner V11 utilizza un sistema a **punteggio (score)** basato su 5 criteri tecnici:
    
    1.  **Trend EMA50**: Punteggio se la media mobile a 50 giorni è sopra il prezzo di 5 giorni fa (Trend Rialzista).
    2.  **RSI Momentum**: Punteggio se l'RSI(14) era in ipervenduto (<35) e sta iniziando a risalire.
    3.  **MACD Cross**: Punteggio se la linea MACD è sopra la Signal Line.
    4.  **Volume Confirm**: Punteggio se il volume odierno è superiore alla media degli ultimi 20 giorni.
    5.  **Volatility (ATR)**: Punteggio se la volatilità (ATR/Prezzo) è contenuta (tra 0.5% e 10%), evitando titoli troppo piatti o troppo nervosi.

    **Livelli di Segnale:**
    *   🔴 **NONE** (Score < 3): Nessuna configurazione interessante.
    *   🟡 **BUY** (Score 3-4): Configurazione rialzista in formazione.
    *   🟢 **STRONG BUY** (Score 5): Tutti i criteri tecnici sono allineati al rialzo.
    \"\"\")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.divider()
st.caption(
    f"Ultimo aggiornamento interfaccia: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
