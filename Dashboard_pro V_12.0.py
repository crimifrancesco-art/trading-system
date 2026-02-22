# =============================================================================
# DASHBOARD PRO V_12.0 - INTEGRAZIONE V11 & V9
# =============================================================================
import io
import time
import sqlite3
import locale
import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
from pathlib import Path
from datetime import datetime
from run_scan import run_scan

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 12.0",
    layout="wide",
    page_icon="📊",
)

st.markdown("""
    <style>
    .stDownloadButton > button { width: 100% !important; }
    .st-emotion-cache-16idsys p { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Trading Scanner – Versione PRO 12.0")
st.caption(
    "EARLY • PRO • REA‑QUANT • Serafini • Regime & Momentum • "
    "Multi‑Timeframe • Finviz • RISULTATI SCAN (V11) • Watchlist DB"
)

# -----------------------------------------------------------------------------
# FORMATTAZIONE E UTILITY
# -----------------------------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass

def fmt_currency(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return (f"{symbol}{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

def fmt_marketcap(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == 0:
        return "N/A"
    v = float(value)
    if v >= 1_000_000_000_000:
        val, suff = v / 1_000_000_000_000, "T"
    elif v >= 1_000_000_000:
        val, suff = v / 1_000_000_000, "B"
    elif v >= 1_000_000:
        val, suff = v / 1_000_000, "M"
    elif v >= 1_000:
        val, suff = v / 1_000, "K"
    else:
        return fmt_currency(v, symbol)
    return (f"{symbol}{val:,.2f}{suff}".replace(",", "X").replace(".", ",").replace("X", "."))

def color_signal(val):
    if val == "STRONG BUY":
        return "background-color: #0f5132; color: white;"
    if val == "BUY":
        return "background-color: #664d03; color: white;"
    return ""

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if "Currency" not in df.columns: df["Currency"] = "USD"
    for col, new_col in [("MarketCap", "MarketCap_fmt"), ("market_cap", "MarketCap_fmt"), 
                         ("Vol_Today", "Vol_Today_fmt"), ("vol_today", "Vol_Today_fmt"),
                         ("Vol_7d_Avg", "Vol_7d_Avg_fmt"), ("vol_7d_avg", "Vol_7d_Avg_fmt")]:
        if col in df.columns:
            df[new_col] = df.apply(lambda r: fmt_marketcap(r[col], "€" if r.get("Currency", r.get("currency")) == "EUR" else "$"), axis=1)
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df.apply(lambda r: fmt_currency(r["Prezzo"], "€" if r["Currency"] == "EUR" else "$"), axis=1)
    elif "price" in df.columns:
        df["Prezzo_fmt"] = df.apply(lambda r: fmt_currency(r["price"], "€" if r.get("currency") == "EUR" else "$"), axis=1)
    return df

def add_links(df: pd.DataFrame) -> pd.DataFrame:
    col = "Ticker" if "Ticker" in df.columns else "ticker"
    if col not in df.columns: return df
    df["Yahoo"] = df[col].astype(str).apply(lambda t: f"https://finance.yahoo.com/quote/{t}")
    df["TV"] = df[col].astype(str).apply(lambda t: f"https://www.tradingview.com/chart/?symbol={t.split('.')[0]}")
    return df

# -----------------------------------------------------------------------------
# DB WATCHLIST (SQLite)
# -----------------------------------------------------------------------------
DB_PATH = Path("watchlist.db")
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, name TEXT, 
        trend TEXT, origine TEXT, note TEXT, list_name TEXT, created_at TEXT)""")
    conn.commit()
    conn.close()

def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute("INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at) VALUES (?,?,?,?,?,?,?)",
                  (t, n, trend, origine, note, list_name, now))
    conn.commit()
    conn.close()

def load_watchlist():
    if not DB_PATH.exists(): return pd.DataFrame(columns=["id","ticker","name","trend","origine","note","list_name","created_at"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    return df

init_db()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurazione")
st.sidebar.subheader("🚀 Azioni")

MARKETS_DICT = {
    "Eurostoxx 600": ["ASML", "MC.PA", "SAP", "OR.PA", "TTE.PA", "SIE.DE", "NESN.SW", "NOVN.SW", "ROG.SW", "LVMH.PA"],
    "FTSE MIB": ["UCG.MI", "ISP.MI", "ENEL.MI", "ENI.MI", "LDO.MI", "PRY.MI", "STM.MI", "TEN.MI", "A2A.MI", "AMP.MI"],
    "S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK-B", "JPM", "V", "UNH", "PG"],
    "Nasdaq 100": ["NVDA", "TSLA", "AVGO", "COST", "ADBE", "NFLX", "AMD", "PEP", "AZN", "LIN"],
    "Dow Jones": ["GS", "HD", "MCD", "BA", "CRM", "WMT", "DIS", "KO", "CAT", "JNJ"],
    "Russell 2000": ["IWM", "VTWO"],
    "Materie Prime": ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F"],
    "ETF": ["SPY", "QQQ", "IWM", "GLD", "TLT", "VTI", "EEM"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "Emerging": ["BABA", "TCEHY", "JD", "VALE", "PBR"]
}

selected_markets = []
st.sidebar.subheader("📈 Selezione Mercati")
for mkt in MARKETS_DICT.keys():
    if st.sidebar.checkbox(mkt, value=(mkt in ["FTSE MIB", "Nasdaq 100"])):
        selected_markets.append(mkt)

if st.sidebar.button("🚀 AVVIA SCANNER PRO 12.0", type="primary", use_container_width=True):
    universe = []
    for m in selected_markets: universe.extend(MARKETS_DICT[m])
    universe = list(set(universe))
    if universe:
        Path("data").mkdir(exist_ok=True)
        Path("data/runtime_universe.json").write_text(json.dumps({"tickers": universe}))
        with st.spinner("Scansione V11 in corso..."):
            run_scan()
        st.success("Scansione V11 Completa!")
        st.rerun()

st.sidebar.divider()
show_only_watchlist = st.sidebar.checkbox("Mostra solo Watchlist", False)
top_n = st.sidebar.number_input("TOP N titoli", 5, 50, 15)

df_wl_all = load_watchlist()
list_options = sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
active_list = st.sidebar.selectbox("Lista attiva", list_options, index=0)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_v11, tab_watch = st.tabs(["🗓️ Risultati Scan", "📌 Watchlist & Note"])

with tab_v11:
    results_path = Path("data/scan_results.json")
    if results_path.exists():
        df_v11 = pd.DataFrame(json.loads(results_path.read_text()))
        if not df_v11.empty:
            df_filtered = df_v11.head(top_n).copy()
            df_filtered = add_formatted_cols(df_filtered)
            df_filtered = add_links(df_filtered)
            view = df_filtered[["name", "ticker", "Prezzo_fmt", "MarketCap_fmt", "Vol_Today_fmt", "rsi", "score", "signal", "Yahoo", "TV"]].rename(
                columns={"name":"Nome", "ticker":"Ticker", "rsi":"RSI", "score":"Score", "signal":"Segnale"}
            )
            st.dataframe(view.style.applymap(color_signal, subset=["Segnale"]), use_container_width=True, hide_index=True, column_config={
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "TV": st.column_config.LinkColumn("TradingView", display_text="Apri"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.2f")
            })
            opts = sorted([f"{r['Nome']} ({r['Ticker']})" for _, r in view.iterrows()])
            sel = st.multiselect("Aggiungi a Watchlist", opts)
            if st.button("📌 Salva in Watchlist"):
                tkrs = [s.split(" (")[-1][:-1] for s in sel]
                nms = [s.split(" (")[0] for s in sel]
                add_to_watchlist(tkrs, nms, "SCAN_V12", "", list_name=active_list)
                st.success("Salvati!")

with tab_watch:
    df_wl_filt = df_wl_all[df_wl_all["list_name"] == active_list]
    if not df_wl_filt.empty:
        st.dataframe(df_wl_filt[["name", "ticker", "trend", "origine", "note", "created_at"]], use_container_width=True, hide_index=True)
