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

# Custom CSS for Sidebar Icons and Layout
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-weight: bold;
    }
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
    return f"{symbol}{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_int(value):
    if value is None: return ""
    try:
        v = float(value)
        if np.isnan(v): return ""
        return f"{int(v):,}".replace(",", ".")
    except: return str(value)

def fmt_market_cap(value):
    if not value: return "N/A"
    try:
        v = float(value)
        if np.isnan(v) or v == 0: return "N/A"
        if v >= 1e12: return f"{v/1e12:.2f}T".replace(".", ",")
        if v >= 1e9: return f"{v/1e9:.2f}B".replace(".", ",")
        if v >= 1e6: return f"{v/1e6:.2f}M".replace(".", ",")
        return str(int(v))
    except: return "N/A"

# ----------------------------------------------------------
# SIDEBAR - CONFIGURAZIONE
# ----------------------------------------------------------
st.sidebar.title("⚙️ Configurazione")

# --- Selezione Mercati ---
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
    default_val = label in ["IT FTSE MIB", "US Nasdaq 100"] # Pre-selected as per image
    if st.sidebar.checkbox(label, value=default_val):
        selected_markets.append(label)

# ----------------------------------------------------------
# SIDEBAR - OUTPUT
# ----------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("🕹️ Output")
top_n = st.sidebar.number_input("TOP N titoli per tab", min_value=1, max_value=100, value=15, step=1)

# ----------------------------------------------------------
# SIDEBAR - WATCHLIST
# ----------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("📁 Lista Watchlist attiva")

watchlists = load_watchlists()
list_names = list(watchlists.keys())

# Lista esistente
active_list = st.sidebar.selectbox("Lista esistente", list_names, index=0, help="Seleziona la watchlist da visualizzare o modificare")

# Crea nuova lista
new_list_name = st.sidebar.text_input("Crea nuova lista", placeholder="Es. Swing, LT, Crypto...", help="Inserisci il nome e premi Invio")
if new_list_name and new_list_name not in watchlists:
    watchlists[new_list_name] = []
    save_watchlists(watchlists)
    st.rerun()

# Rinomina lista
rename_to = st.sidebar.text_input("Rinomina lista", placeholder="Nuovo nome...", help="Rinomina la lista selezionata sopra")
if rename_to and active_list in watchlists and rename_to not in watchlists:
    watchlists[rename_to] = watchlists.pop(active_list)
    save_watchlists(watchlists)
    st.rerun()

# --- Avvia Scanner Button ---
st.sidebar.markdown("---")
if st.sidebar.button("🚀 AVVIA SCANNER", use_container_width=True):
    all_tickers = []
    for m in selected_markets:
        all_tickers.extend(MARKETS_DICT[m])
    
    if not all_tickers:
        st.warning("Seleziona almeno un mercato.")
    else:
        # Update runtime universe
        runtime_path = Path("data/runtime_universe.json")
        runtime_path.write_text(json.dumps({"tickers": list(set(all_tickers))}))
        
        with st.spinner("Scansione mercati in corso..."):
            try:
                run_scan()
                st.success("Scan completato ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Errore durante lo scan: {e}")

# ----------------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------------
st.title("📊 Trading Scanner — V11 Professional")

# Tabs
tab_results, tab_watchlist, tab_legend = st.tabs(["🗓️ Risultati Scan", "⭐ Watchlist", "ℹ️ Legenda"])

results_file = Path("data/scan_results.json")

with tab_results:
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text())
            df = pd.DataFrame(data)
            
            if not df.empty:
                # Compatibility fix
                for col in ["name", "vol_today", "vol_7d_avg", "market_cap"]:
                    if col not in df.columns: df[col] = None
                
                # Filters
                c1, c2 = st.columns(2)
                with c1: min_score = st.slider("Min Score", 0, 5, 3)
                with c2: signal_filter = st.multiselect("Segnale", ["STRONG BUY", "BUY", "NONE"], default=["STRONG BUY", "BUY"])
                
                filtered_df = df[(df['score'] >= min_score) & (df['signal'].isin(signal_filter))].head(top_n).copy()
                
                # Display Table
                df_display = filtered_df.copy()
                df_display["Prezzo"] = filtered_df.apply(lambda r: fmt_currency(r["price"], r.get("currency", "USD")), axis=1)
                df_display["Market Cap"] = filtered_df["market_cap"].apply(fmt_market_cap)
                df_display["Vol Giorno"] = filtered_df["vol_today"].apply(fmt_int)
                df_display["Vol Medio 7g"] = filtered_df["vol_7d_avg"].apply(fmt_int)
                
                COLS = ["name", "ticker", "Prezzo", "Market Cap", "Vol Giorno", "Vol Medio 7g", "rsi", "score", "signal"]
                rename_map = {"name": "Nome", "ticker": "Ticker", "rsi": "RSI", "score": "Score", "signal": "Segnale"}
                
                def color_signal(val):
                    if val == "STRONG BUY": return "background-color: #0f5132; color: white;"
                    if val == "BUY": return "background-color: #664d03; color: white;"
                    return ""

                st.dataframe(
                    df_display[COLS].rename(columns=rename_map).style.applymap(color_signal, subset=["Segnale"]),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add to Watchlist Logic
                st.markdown("### ➕ Aggiungi a Watchlist")
                to_add = st.selectbox("Seleziona Ticker", filtered_df["ticker"].tolist())
                if st.button(f"Aggiungi {to_add} a {active_list}"):
                    if to_add not in watchlists[active_list]:
                        watchlists[active_list].append(to_add)
                        save_watchlists(watchlists)
                        st.success(f"{to_add} aggiunto!")
            else:
                st.info("Nessun risultato trovato.")
        except Exception as e:
            st.error(f"Errore: {e}")
    else:
        st.info("Avvia una scansione per vedere i risultati.")

with tab_watchlist:
    st.subheader(f"Watchlist: {active_list}")
    current_tickers = watchlists.get(active_list, [])
    
    if not current_tickers:
        st.info("La watchlist è vuota.")
    else:
        # Show mini table for watchlist items if results exist
        if results_file.exists():
            full_data = pd.DataFrame(json.loads(results_file.read_text()))
            w_df = full_data[full_data["ticker"].isin(current_tickers)]
            if not w_df.empty:
                st.dataframe(w_df[["ticker", "price", "signal", "score"]], use_container_width=True)
            else:
                st.write(f"Titoli in lista: {', '.join(current_tickers)}")
        
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
    - **RSI**: Momentum crescente sotto 70.
    - **MACD**: Cross rialzista confermato.
    - **Volume**: Oggi > Media 20gg.
    - **ATR**: Volatilità in range ottimale.
    """)
