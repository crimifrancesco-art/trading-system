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

def fmt_compact(value):
    if not value: return "N/A"
    try:
        v = float(value)
        if np.isnan(v) or v == 0: return "N/A"
        if v >= 1e12: return f"{v/1e12:.2f}T"
        if v >= 1e9: return f"{v/1e9:.2f}B"
        if v >= 1e6: return f"{v/1e6:.2f}M"
        return f"{int(v):,}"
    except: return "N/A"

def color_signal(val):
    if val == "STRONG BUY":
        return "background-color: #0f5132; color: white;"
    if val == "BUY":
        return "background-color: #664d03; color: white;"
    return ""

# ----------------------------------------------------------
# SIDEBAR - CONFIGURAZIONE
# ----------------------------------------------------------
st.sidebar.title("⚙️ Configurazione")

# --- AVVIA SCANNER BUTTON (TOP) ---
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

# we need to define selected_markets here or before the button
st.sidebar.markdown("### 🚀 Azioni")

# We create the checkbox list first to know what to scan
st.sidebar.markdown("### 📈 Selezione Mercati")
selected_markets = []
for label in MARKETS_DICT.keys():
    default_val = label in ["IT FTSE MIB", "US Nasdaq 100"]
    if st.sidebar.checkbox(label, value=default_val, key=f"mkt_{label}"):
        selected_markets.append(label)

if st.sidebar.button("🚀 AVVIA SCANNER", use_container_width=True, type="primary"):
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

st.sidebar.markdown("---")

# --- MODALITÀ ---
st.sidebar.markdown("### 🧠 Modalità")
show_only_watchlist = st.sidebar.checkbox("Mostra solo Watchlist (salta scanner)", value=False)

if not show_only_watchlist:
    st.sidebar.markdown("---")
    st.sidebar.title("🕹️ Output")
    top_n = st.sidebar.number_input("TOP N titoli per tab", min_value=1, max_value=100, value=15, step=1)

# --- WATCHLIST MANAGEMENT ---
st.sidebar.markdown("---")
st.sidebar.title("📁 Lista Watchlist")
watchlists = load_watchlists()
list_names = list(watchlists.keys())
active_list = st.sidebar.selectbox("Lista esistente", list_names, index=0)

new_list_name = st.sidebar.text_input("Crea nuova lista", placeholder="Es. Swing, LT, Crypto...")
if st.sidebar.button("Crea Lista") and new_list_name:
    if new_list_name not in watchlists:
        watchlists[new_list_name] = []
        save_watchlists(watchlists)
        st.rerun()

rename_target = st.sidebar.selectbox("Rinomina lista", list_names, index=list_names.index(active_list))
new_rename_name = st.sidebar.text_input("Nuovo nome per la lista selezionata", placeholder="Nuovo nome...")
if st.sidebar.button("Applica rinomina") and new_rename_name and rename_target:
    if new_rename_name not in watchlists:
        watchlists[new_rename_name] = watchlists.pop(rename_target)
        save_watchlists(watchlists)
        st.rerun()

if st.sidebar.button("Elimina Lista Selezionata", type="secondary"):
    if active_list != "DEFAULT":
        watchlists.pop(active_list)
        save_watchlists(watchlists)
        st.rerun()
    else:
        st.sidebar.warning("Non puoi eliminare la lista DEFAULT.")

st.sidebar.markdown(f"**Lista attiva attuale: {active_list}**")

# ----------------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------------
st.title("📊 Trading Scanner — V11 Professional")

if show_only_watchlist:
    st.subheader(f"⭐ Visualizzazione Watchlist: {active_list}")
    results_file = Path("data/scan_results.json")
    current_tickers = watchlists.get(active_list, [])
    if not current_tickers:
        st.info("La watchlist è vuota.")
    else:
        if results_file.exists():
            data = pd.DataFrame(json.loads(results_file.read_text()))
            w_df = data[data["ticker"].isin(current_tickers)]
            if not w_df.empty:
                st.dataframe(w_df, use_container_width=True, hide_index=True)
            else:
                st.write(f"Titoli monitorati: {', '.join(current_tickers)}")
        else:
            st.write(f"Titoli in lista: {', '.join(current_tickers)}")
    st.stop()

tab_results, tab_watchlist, tab_legend = st.tabs(["🗓️ Risultati Scan", "⭐ Watchlist", "ℹ️ Legenda"])

results_file = Path("data/scan_results.json")

with tab_results:
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text())
            df = pd.DataFrame(data)
            if not df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    min_score = st.slider("Min Score", 0, 5, 3)
                with c2:
                    signal_filter = st.multiselect(
                        "Segnale",
                        ["STRONG BUY", "BUY", "NONE"],
                        default=["STRONG BUY", "BUY"]
                    )

                filtered_df = df[
                    (df['score'] >= min_score) &
                    (df['signal'].isin(signal_filter))
                ].head(top_n).copy()

                if not filtered_df.empty:
                    disp = pd.DataFrame()
                    disp["Nome"] = filtered_df["name"]
                    disp["Ticker"] = filtered_df["ticker"]
                    disp["Prezzo"] = filtered_df["price"]
                    disp["Market Cap"] = filtered_df["market_cap"].apply(fmt_compact)
                    disp["Vol Giorno"] = filtered_df["vol_today"].apply(fmt_compact)
                    disp["Vol Medio 7g"] = filtered_df["vol_7d_avg"].apply(fmt_compact)
                    disp["RSI"] = filtered_df["rsi"]
                    disp["Score"] = filtered_df["score"]
                    disp["Segnale"] = filtered_df["signal"]
                    disp["Yahoo"] = filtered_df["ticker"].apply(lambda t: f"https://finance.yahoo.com/quote/{t}")
                    disp["TV"] = filtered_df["ticker"].apply(lambda t: f"https://www.tradingview.com/symbols/{t.replace('-','').replace('.','')}/")

                    st.dataframe(
                        disp.style.applymap(color_signal, subset=["Segnale"]),
                        column_config={
                            "Prezzo": st.column_config.NumberColumn("Prezzo", format="$ %.2f"),
                            "RSI": st.column_config.NumberColumn("RSI", format="%.2f"),
                            "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                            "TV": st.column_config.LinkColumn("TradingView", display_text="Apri"),
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("### 📥 Export Risultati")
                    col_ex1, col_ex2, col_ex3 = st.columns(3)
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    col_ex1.download_button("Export CSV", csv, f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='ScanResults')
                    col_ex2.download_button("Export XLSX", buffer.getvalue(), f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", use_container_width=True)
                    tv_list = ",".join(filtered_df["ticker"].tolist())
                    col_ex3.download_button("Export TV", tv_list, f"tv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", use_container_width=True)

                    st.markdown("---")
                    st.markdown("### ➕ Aggiungi a Watchlist")
                    options_df = filtered_df[["name", "ticker"]].sort_values("name")
                    ticker_options = [f"{r['name']} ({r['ticker']})" for _, r in options_df.iterrows()]
                    selected_display_names = st.multiselect("Seleziona uno o più Ticker", ticker_options)
                    tickers_to_save = [name.split(" (")[-1].replace(")", "") for name in selected_display_names]
                    
                    if st.button(f"Salva {len(tickers_to_save)} titoli in {active_list}", use_container_width=True):
                        if not tickers_to_save:
                            st.warning("Seleziona almeno un ticker.")
                        else:
                            added = 0
                            for t in tickers_to_save:
                                if t not in watchlists[active_list]:
                                    watchlists[active_list].append(t)
                                    added += 1
                            save_watchlists(watchlists)
                            st.success(f"Aggiunti {added} titoli a {active_list}!")

                else:
                    st.info("Nessun risultato trovato.")
        except Exception as e:
            st.error(f"Errore: {e}")
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
    """)
