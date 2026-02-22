# ==========================================================
# DASHBOARD V11 — PROFESSIONAL SCANNER (MODULAR)
# ==========================================================
import streamlit as st
import pandas as pd
import json
import io
import numpy as np
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
# UTILITIES & FORMATTING
# ----------------------------------------------------------
def fmt_currency(value, currency="USD"):
    if value is None or np.isnan(value): return ""
    symbol = "$" if currency == "USD" else "€"
    # Format: $1.234,56 or €1.234,56 (European style)
    return f"{symbol}{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_int(value):
    if value is None or np.isnan(value): return ""
    return f"{int(value):,}".replace(",", ".")

def fmt_marketcap(value, currency="USD"):
    if value is None or value == 0 or np.isnan(value): return ""
    symbol = "$" if currency == "USD" else "€"
    if value >= 1_000_000_000_000:
        return f"{symbol}{value / 1_000_000_000_000:,.2f}T".replace(",", "X").replace(".", ",").replace("X", ".")
    if value >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:,.2f}B".replace(",", "X").replace(".", ",").replace("X", ".")
    if value >= 1_000_000:
        return f"{symbol}{value / 1_000_000:,.2f}M".replace(",", "X").replace(".", ",").replace("X", ".")
    return fmt_currency(value, currency)

# ----------------------------------------------------------
# MARKET UNIVERSES
# ----------------------------------------------------------
MARKETS = {
    "FTSE": ["ENI.MI", "ISP.MI", "UCG.MI", "STM.MI", "ENEL.MI", "LDO.MI", "PRY.MI", "TEN.MI", "A2A.MI", "AMP.MI", "BAMI.MI", "BMED.MI", "FBK.MI", "MONC.MI", "PST.MI"],
    "SP500": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "V", "TSLA", "WMT", "JPM", "UNH", "MA", "ORCL", "COST", "HD", "PG", "CVX"],
    "Nasdaq": ["TSLA", "AMD", "AVGO", "INTC", "NFLX", "ADBE", "COST", "PEP", "CSCO", "AZN", "QCOM", "AMGN", "TMUS", "TXN", "AMAT", "SBUX", "ISRG", "MDLZ", "LRCX", "ADI"],
    "ETF": ["SPY", "QQQ", "IWM", "GLD", "TLT", "VTI", "EEM", "VXX", "SOXX", "XLE"],
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD"],
    "Eurostoxx": ["ASML.AS", "MC.PA", "OR.PA", "TTE.PA", "AIR.PA", "SAN.PA", "SAP.DE", "SIE.DE", "IBE.MC", "NESN.SW"]
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
# LOAD RESULTS & DISPLAY
# ----------------------------------------------------------
result_path = Path("data/scan_results.json")
tab_results, tab_legend = st.tabs(["📊 Risultati Scan", "📘 Legenda Filtri"])

with tab_results:
    if result_path.exists():
        try:
            results = json.loads(result_path.read_text())
            df = pd.DataFrame(results)
            
            if not df.empty:
                # Add links
                df['Yahoo'] = df['ticker'].apply(lambda x: f"https://finance.yahoo.com/quote/{x}")
                df['TradingView'] = df['ticker'].apply(lambda x: f"https://www.tradingview.com/chart/?symbol={x.split('.')[0]}")
                
                # Apply Formatting for Display
                df_display = df.copy()
                df_display['Prezzo'] = df.apply(lambda r: fmt_currency(r['price'], r.get('currency', 'USD')), axis=1)
                df_display['Market Cap'] = df.apply(lambda r: fmt_marketcap(r.get('market_cap', 0), r.get('currency', 'USD')), axis=1)
                df_display['Vol giorno'] = df['vol_today'].apply(fmt_int)
                df_display['Vol medio 7g'] = df['vol_7d_avg'].apply(fmt_int)
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                n_strong = (df["signal"] == "STRONG BUY").sum()
                n_buy = (df["signal"] == "BUY").sum()
                
                c1.metric("STRONG BUY", int(n_strong))
                c2.metric("BUY", int(n_buy))
                c3.metric("Assets scansionati", len(df))

                # Table styling
                def color_signal(val):
                    if val == "STRONG BUY": return "background-color:#0f5132;color:white"
                    if val == "BUY": return "background-color:#664d03;color:white"
                    return ""

                st.subheader("Tabella Segnali")
                
                # Define columns to show (sync with image)
                cols_to_show = [
                    "name", "ticker", "Prezzo", "Market Cap", "Vol giorno", "Vol medio 7g", 
                    "score", "rsi", "vol_ratio", "obv_trend", "atr", "atr_exp", "signal", 
                    "Yahoo", "TradingView"
                ]
                
                # Rename for UI
                df_final = df_display[cols_to_show].rename(columns={
                    "name": "Nome", "ticker": "Ticker", "score": "Early_Score", 
                    "rsi": "RSI", "vol_ratio": "Vol_Ratio", "obv_trend": "OBV_Trend",
                    "atr": "ATR", "atr_exp": "ATR_Exp", "signal": "Stato"
                })

                st.dataframe(
                    df_final.style.applymap(color_signal, subset=["Stato"]),
                    use_container_width=True,
                    column_config={
                        "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                        "TradingView": st.column_config.LinkColumn("TradingView", display_text="Apri"),
                        "ATR_Exp": st.column_config.CheckboxColumn("ATR_Exp")
                    }
                )
                
                # Export
                st.divider()
                st.subheader("📥 Esporta Risultati")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Scarica Risultati CSV", data=csv, file_name=f"scan_v11_{datetime.now().strftime('%Y%m%d')}.csv", use_container_width=True)

            else:
                st.info("Nessun segnale trovato. Prova a cambiare mercati o avviare un nuovo scan.")
        except Exception as e:
            st.error(f"Errore lettura risultati: {e}")
    else:
        st.info("Premi 'AVVIA SCANNER' nella sidebar per vedere i risultati.")

with tab_legend:
    st.subheader("Spiegazione Filtri V11")
    st.markdown("""
    Lo scanner V11 utilizza un sistema a **punteggio (score)** basato su 5 criteri tecnici:
    1. **Trend EMA50**: Rialzo EMA50 vs prezzo 5 giorni fa.
    2. **RSI Momentum**: RSI in salita e < 70.
    3. **MACD Cross**: MACD sopra la Signal Line.
    4. **Volume Confirm**: Volume odierno > media 20 giorni.
    5. **Volatility (ATR)**: Volatilità percentuale tra 0.5% e 10%.
    """)

st.divider()
st.caption(f"Ultimo aggiornamento interfaccia: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
