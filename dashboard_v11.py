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

# Custom CSS
st.markdown("""
    <style>
    .sidebar .sidebar-content { padding-top: 1rem; }
    .st-emotion-cache-16idsys p { font-weight: bold; }
    /* Aligning specific columns to the right if needed via dataframe styling, 
       but for general text we can use markdown in some cases. 
       However, st.dataframe handles alignment better with column_config. */
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# UTILITIES & DATA HANDLING
# ----------------------------------------------------------
WATCHLIST_FILE = Path("data/watchlists.json")
WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_watchlists():
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text())
        except:
            return {"DEFAULT": []}
    return {"DEFAULT": []}

def save_watchlists(data):
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))

def fmt_currency(value, currency="USD"):
    if value is None: return ""
    try:
        v = float(value)
        if np.isnan(v): return ""
    except: return str(value)
    symbol = "$" if currency == "USD" else "€"
    # Format with 2 decimals and thousands separator
    return f"{symbol} {v:,.2f}"

def fmt_int(value):
    if value is None: return ""
    try:
        v = float(value)
        if np.isnan(v): return ""
        return f"{int(v):,}"
    except: return str(value)

def fmt_market_cap(value):
    if not value: return "N/A"
    try:
        v = float(value)
        if np.isnan(v) or v == 0: return "N/A"
        if v >= 1e12: return f"{v/1e12:.2f}T"
        if v >= 1e9: return f"{v/1e9:.2f}B"
        if v >= 1e6: return f"{v/1e6:.2f}M"
        return f"{int(v):,}"
    except: return "N/A"

# ----------------------------------------------------------
# SIDEBAR - CONFIGURAZIONE
# ----------------------------------------------------------
st.sidebar.title("⚙️ Configurazione")

st.sidebar.markdown("### 📈 Selezione Mercati")
MARKETS_DICT = {
    "EU Eurostoxx 600": ["ASML", "MC.PA", "SAP", "OR.PA", "TTE.PA", "SIE.DE", "NESN.SW", "NOVN.SW", "ROG.SW", "LVMH.PA"],
    "IT FTSE MIB": ["ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI", "STLAM.MI", "G.MI", "FER.MI", "PST.MI", "A2A.MI", "PRY.MI"],
    "US S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK-B", "JPM", "V", "UNH", "PG"],
    "US Nasdaq 100": ["NVDA", "TSLA", "AVGO", "COST", "ADBE", "NFLX", "AMD", "PEP", "AZN", "LIN"],
    "US Dow Jones": ["DJI", "GS", "HD", "MCD", "BA", "CRM", "TRV", "HON", "AXP", "WMT"],
    "US Russell 2000": ["IWM"],
    "🛢️ Materie Prime": ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F"],
    "📦 ETF": ["SPY", "QQQ", "EEM", "VTI", "AGG"],
    "₿ Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "🌍 Emergenti": ["BABA", "TCEHY", "JD", "VALE", "PBR"]
}

selected_markets = []
for label in MARKETS_DICT.keys():
    default_val = label in ["IT FTSE MIB", "US Nasdaq 100"]
    if st.sidebar.checkbox(label, value=default_val):
        selected_markets.append(label)

st.sidebar.markdown("---")
st.sidebar.title("🕹️ Output")
top_n = st.sidebar.number_input("TOP N titoli per tab", min_value=1, max_value=100, value=15, step=1)

st.sidebar.markdown("---")
st.sidebar.title("📁 Watchlist")
watchlists = load_watchlists()
active_list = st.sidebar.selectbox("Lista attiva", list(watchlists.keys()), index=0)
new_list_name = st.sidebar.text_input("Crea nuova lista", placeholder="Nome...")
if st.sidebar.button("Crea Lista") and new_list_name:
    if new_list_name not in watchlists:
        watchlists[new_list_name] = []
        save_watchlists(watchlists)
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🚀 AVVIA SCANNER", use_container_width=True):
    all_tickers = []
    for m in selected_markets:
        all_tickers.extend(MARKETS_DICT[m])
    
    if not all_tickers:
        st.warning("Seleziona almeno un mercato.")
    else:
        runtime_path = Path("data/runtime_universe.json")
        runtime_path.write_text(json.dumps({"tickers": list(set(all_tickers))}))
        with st.spinner("Scansione in corso..."):
            try:
                run_scan()
                st.success("Scan completato ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")

# ----------------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------------
st.title("📊 Trading Scanner — V11 Professional")

tab_results, tab_watchlist, tab_legend = st.tabs(["🗓️ Risultati Scan", "⭐ Watchlist", "ℹ️ Legenda"])

results_file = Path("data/scan_results.json")

def get_links(ticker):
    # Yahoo Finance Link
    yahoo_url = f"https://finance.yahoo.com/quote/{ticker}"
    # TradingView Link (Assuming US markets or common ones)
    tv_ticker = ticker.replace("-", "").replace(".", "")
    tv_url = f"https://www.tradingview.com/symbols/{tv_ticker}/"
    return f"[Yahoo]({yahoo_url}) | [TV]({tv_url})"

with tab_results:
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text())
            df = pd.DataFrame(data)
            if not df.empty:
                c1, c2 = st.columns(2)
                with c1: min_score = st.slider("Min Score", 0, 5, 3)
                with c2: signal_filter = st.multiselect("Segnale", ["STRONG BUY", "BUY", "NONE"], default=["STRONG BUY", "BUY"])
                
                filtered_df = df[(df['score'] >= min_score) & (df['signal'].isin(signal_filter))].head(top_n).copy()
                
                if not filtered_df.empty:
                    # Formatting
                    filtered_df["Links"] = filtered_df["ticker"].apply(get_links)
                    
                    # Display using st.dataframe with column_config for better control
                    st.dataframe(
                        filtered_df,
                        column_order=("name", "ticker", "price", "market_cap", "vol_today", "vol_7d_avg", "rsi", "score", "signal", "Links"),
                        column_config={
                            "name": "Nome",
                            "ticker": "Ticker",
                            "price": st.column_config.NumberColumn("Prezzo", format="$ %.2f"),
                            "market_cap": st.column_config.TextColumn("Market Cap"), # We keep our string format
                            "vol_today": st.column_config.NumberColumn("Vol Giorno", format="%d"),
                            "vol_7d_avg": st.column_config.NumberColumn("Vol Medio 7g", format="%d"),
                            "rsi": st.column_config.NumberColumn("RSI", format="%.2f"),
                            "score": "Score",
                            "signal": "Segnale",
                            "Links": st.column_config.LinkColumn("Links")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("### ➕ Aggiungi a Watchlist")
                    to_add = st.selectbox("Seleziona Ticker", filtered_df["ticker"].tolist())
                    if st.button(f"Aggiungi {to_add} a {active_list}"):
                        if to_add not in watchlists[active_list]:
                            watchlists[active_list].append(to_add)
                            save_watchlists(watchlists)
                            st.success(f"{to_add} aggiunto!")
                else:
                    st.info("Nessun risultato con i filtri attuali.")
        except Exception as e:
            st.error(f"Errore lettura: {e}")
    else:
        st.info("Avvia una scansione per caricare i dati.")

with tab_watchlist:
    st.subheader(f"Watchlist: {active_list}")
    current_tickers = watchlists.get(active_list, [])
    if not current_tickers:
        st.info("La watchlist è vuota.")
    else:
        if results_file.exists():
            full_data = pd.DataFrame(json.loads(results_file.read_text()))
            w_df = full_data[full_data["ticker"].isin(current_tickers)]
            if not w_df.empty:
                st.dataframe(w_df[["ticker", "price", "signal", "score"]], use_container_width=True, hide_index=True)
            else:
                st.write(f"Titoli: {', '.join(current_tickers)}")
        
        if st.button("Svuota Watchlist"):
            watchlists[active_list] = []
            save_watchlists(watchlists)
            st.rerun()

with tab_legend:
    st.markdown("""
    ### ℹ️ Legenda del Sistema
    **Score (0-5):**
    - 🟢 **STRONG BUY (5/5)**: Tutti i criteri soddisfatti.
    - 🟡 **BUY (3-4/5)**: Trend e momentum positivi.
    - ⚪ **NONE (<3)**: Nessun segnale chiaro.

    **Indicatori:**
    - **Trend**: EMA50 > Prezzo 5gg fa.
    - **RSI**: Momentum crescente (RSI > RSI prec) e non ipercomprato (< 70).
    - **MACD**: Linea MACD > Linea Segnale.
    - **Volume**: Volume odierno > Media mobile volumi a 20gg.
    - **ATR**: Volatilità (ATR/Prezzo) tra 0.5% e 10%.
    
    **Links:**
    - I link puntano direttamente a Yahoo Finance e TradingView per analisi approfondita.
    """)
