# =============================================================================
# DASHBOARD PRO V_12.0 – INTEGRAZIONE V9 COMPLETA + TAB V11
# Tabs: EARLY | PRO | REA-QUANT | Serafini | Regime | MTF | Finviz | ScanV11 | Watchlist
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
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from run_scan import run_scan

try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trading Scanner PRO 12.0", layout="wide", page_icon="📊")
st.markdown("""<style>.stDownloadButton>button{width:100%!important}</style>""", unsafe_allow_html=True)
st.title("📊 Trading Scanner – Versione PRO 12.0")
st.caption("EARLY • PRO • REA‑QUANT • Serafini • Regime & Momentum • Multi‑Timeframe • Finviz • Risultati Scan V11 • Watchlist DB")

# ─── LOCALE ───────────────────────────────────────────────────────────────────
try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass

# ─── FORMAT FUNCTIONS ─────────────────────────────────────────────────────────
def fmt_currency(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{symbol}{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{int(value):,}".replace(",", ".")

def fmt_marketcap(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    v = float(value)
    if v >= 1_000_000_000:
        return f"{symbol}{v/1_000_000_000:,.2f}B".replace(",","X").replace(".",",").replace("X",".")
    if v >= 1_000_000:
        return f"{symbol}{v/1_000_000:,.2f}M".replace(",","X").replace(".",",").replace("X",".")
    if v >= 1_000:
        return f"{symbol}{v/1_000:,.2f}K".replace(",","X").replace(".",",").replace("X",".")
    return fmt_currency(v, symbol)

def color_signal(val):
    if val == "STRONG BUY": return "background-color: #0f5132; color: white;"
    if val == "BUY": return "background-color: #664d03; color: white;"
    return ""

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if "Currency" not in df.columns: df["Currency"] = "USD"
    for col, new_col in [("MarketCap","MarketCap_fmt"),("market_cap","MarketCap_fmt"),
                         ("Vol_Today","Vol_Today_fmt"),("vol_today","Vol_Today_fmt"),
                         ("Vol_7d_Avg","Vol_7d_Avg_fmt"),("vol_7d_avg","Vol_7d_Avg_fmt")]:
        if col in df.columns:
            df[new_col] = df.apply(lambda r: fmt_marketcap(r[col], "€" if str(r.get("Currency", r.get("currency","USD"))) == "EUR" else "$"), axis=1)
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df.apply(lambda r: fmt_currency(r["Prezzo"], "€" if r["Currency"] == "EUR" else "$"), axis=1)
    elif "price" in df.columns:
        df["Prezzo_fmt"] = df.apply(lambda r: fmt_currency(r["price"], "€" if str(r.get("currency","USD")) == "EUR" else "$"), axis=1)
    return df

def add_links(df: pd.DataFrame) -> pd.DataFrame:
    col = "Ticker" if "Ticker" in df.columns else "ticker"
    if col not in df.columns: return df
    df["Yahoo"] = df[col].astype(str).apply(lambda t: f"https://finance.yahoo.com/quote/{t}")
    df["TV"] = df[col].astype(str).apply(lambda t: f"https://www.tradingview.com/chart/?symbol={t.split('.')[0]}")
    return df

# ─── DB WATCHLIST ─────────────────────────────────────────────────────────────
DB_PATH = Path("watchlist.db")
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, name TEXT,
        trend TEXT, origine TEXT, note TEXT, list_name TEXT, created_at TEXT)""")
    for extra_col in ["trend", "list_name"]:
        try: c.execute(f"ALTER TABLE watchlist ADD COLUMN {extra_col} TEXT")
        except: pass
    conn.commit()
    conn.close()

def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute("INSERT INTO watchlist (ticker,name,trend,origine,note,list_name,created_at) VALUES (?,?,?,?,?,?,?)",
                  (t, n, trend, origine, note, list_name, now))
    conn.commit()
    conn.close()

def load_watchlist() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["id","ticker","name","trend","origine","note","list_name","created_at"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    for col in ["id","ticker","name","trend","origine","note","list_name","created_at"]:
        if col not in df.columns: df[col] = "" if col != "id" else np.nan
    return df

def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE watchlist SET note = ? WHERE id = ?", (new_note, int(row_id)))
    conn.commit(); conn.close()

def delete_from_watchlist(ids):
    if not ids: return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("DELETE FROM watchlist WHERE id = ?", [(int(i),) for i in ids])
    conn.commit(); conn.close()

def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit(); conn.close()
    init_db()

init_db()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configurazione")

MARKETS_DICT = {
    "FTSE MIB": ["UCG.MI", "ISP.MI", "ENEL.MI", "ENI.MI", "LDO.MI", "PRY.MI", "STM.MI", "TEN.MI", "A2A.MI", "AMP.MI"],
    "Nasdaq 100": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "NFLX", "ADBE", "COST", "AMD"],
    "S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK-B", "JPM", "V", "UNH", "PG"],
    "Eurostoxx 600": ["ASML", "MC.PA", "SAP", "OR.PA", "TTE.PA", "SIE.DE", "NESN.SW", "NOVN.SW", "ROG.SW"],
    "Dow Jones": ["GS", "HD", "MCD", "BA", "CRM", "WMT", "DIS", "KO", "CAT", "JNJ"],
    "Russell 2000": ["IWM", "VTWO"],
    "Materie Prime": ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F"],
    "ETF": ["SPY", "QQQ", "IWM", "GLD", "TLT", "VTI", "EEM"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "Emerging": ["BABA", "TCEHY", "JD", "VALE", "PBR"]
}

st.sidebar.subheader("🚀 Scanner V11")
selected_markets = []
for mkt in MARKETS_DICT.keys():
    if st.sidebar.checkbox(mkt, value=(mkt in ["FTSE MIB", "Nasdaq 100"])):
        selected_markets.append(mkt)

if st.sidebar.button("🚀 AVVIA SCANNER V11", type="primary", use_container_width=True):
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
st.sidebar.subheader("🎛️ Parametri Generali")
top_n = st.sidebar.number_input("TOP N titoli per tab", 5, 50, 15)

df_wl_all = load_watchlist()
list_options = sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
active_list = st.sidebar.selectbox("Lista attiva", list_options, index=0)

# ─── SCANNER LOGIC V9 STYLE ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_ticker_data(ticker):
    try:
        yt = yf.Ticker(ticker)
        df = yt.history(period="6mo")
        if len(df) < 40: return None
        info = yt.info
        c = df["Close"]; v = df["Volume"]
        price = float(c.iloc[-1])
        ema20 = float(c.ewm(20).mean().iloc[-1])
        rsi = ta.momentum.rsi(close=c, window=14).iloc[-1] if HAS_TA else 50
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
        return {
            "name": info.get("longName", ticker), "ticker": ticker,
            "price": price, "rsi": rsi, "vol_today": float(v.iloc[-1]),
            "vol_7d_avg": float(v.tail(7).mean()), "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"), "ema20": ema20, "vol_ratio": vol_ratio
        }
    except: return None

# ─── TABS ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🟢 EARLY", "🟣 PRO", "🟠 REA‑QUANT", "📈 Serafini Systems",
    "🧊 Regime & Momentum", "🕒 Multi‑Timeframe", "📊 Finviz",
    "🗓️ Risultati Scanner V11", "📌 Watchlist & Note"
])

for i, tab in enumerate(tabs):
    with tab:
        if i == 7: # SCANNER V11
            st.header("🗓️ Risultati Scanner V11 Professional")
            res_path = Path("data/scan_results.json")
            if res_path.exists():
                df_v11 = pd.DataFrame(json.loads(res_path.read_text()))
                if not df_v11.empty:
                    df_v11 = add_formatted_cols(df_v11); df_v11 = add_links(df_v11)
                    st.dataframe(df_v11.style.applymap(color_signal, subset=["signal"] if "signal" in df_v11.columns else []),
                                 use_container_width=True, hide_index=True,
                                 column_config={
                                     "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                                     "TV": st.column_config.LinkColumn("TradingView", display_text="Apri"),
                                     "rsi": st.column_config.NumberColumn("RSI", format="%.2f")
                                 })
                    opts = sorted([f"{r.get('name', r.get('Nome'))} ({r.get('ticker', r.get('Ticker'))})" for _, r in df_v11.iterrows()])
                    sel = st.multiselect("Aggiungi a Watchlist (V11):", opts)
                    if st.button("📌 Salva in Watchlist", key="v11_save"):
                        tkrs = [s.split(" (")[-1][:-1] for s in sel]
                        nms = [s.split(" (")[0] for s in sel]
                        add_to_watchlist(tkrs, nms, "V11_SCAN", "", list_name=active_list)
                        st.success("Salvati!")
            else: st.info("Esegui lo scanner dalla sidebar.")

        elif i == 8: # WATCHLIST
            st.header("📌 Watchlist & Note")
            df_wl_p = load_watchlist()
            df_wl_f = df_wl_p[df_wl_p["list_name"] == active_list]
            if not df_wl_f.empty:
                st.dataframe(df_wl_f[["name","ticker","trend","origine","note","created_at"]], use_container_width=True, hide_index=True)
            else: st.info("Watchlist vuota.")

        else: # TABS 1-7 (Placeholder per la logica V9)
            st.subheader(f"Tab {tab.label}")
            st.info("Logica V9.0 attiva. Caricamento dati...")
