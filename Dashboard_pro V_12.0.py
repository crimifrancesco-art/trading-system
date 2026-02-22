import io
import json
import sqlite3
import locale
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pathlib import Path
from datetime import datetime
from run_scan import run_scan

# --- CONFIG ---
st.set_page_config(page_title="Trading Scanner PRO 12.0", layout="wide", page_icon="📊")
st.markdown("<style>.stDownloadButton>button{width:100%!important}</style>", unsafe_allow_html=True)
st.title("📊 Trading Scanner – Versione PRO 12.0")
st.caption("🟢 EARLY • 🟣 PRO • 🟠 REA‑QUANT • 📈 Serafini • 🧊 Regime • 🕒 MTF • 📊 Finviz | 🗓️ Scan V11 | 📌 Watchlist")

# --- FORMATTERS ---
def fmt_currency(v, s="€"):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    return f"{s}{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
def fmt_marketcap(v, s="€"):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    if v >= 1e9: return f"{s}{v/1e9:,.2f}B".replace(",","X").replace(".",",").replace("X",".")
    if v >= 1e6: return f"{s}{v/1e6:,.2f}M".replace(",","X").replace(".",",").replace("X",".")
    return fmt_currency(v, s)
def color_signal(v):
    if v == "STRONG BUY": return "background-color: #0f5132; color: white;"
    if v == "BUY": return "background-color: #664d03; color: white;"
    return ""
def add_formatted_cols(df):
    if df.empty: return df
    df = df.copy()
    if "Currency" not in df.columns: df["Currency"] = "USD"
    for col, new in [("MarketCap","MarketCap_fmt"),("market_cap","MarketCap_fmt"),("Vol_Today","Vol_Today_fmt"),("vol_today","Vol_Today_fmt")]:
        if col in df.columns: df[new] = df.apply(lambda r: fmt_marketcap(r[col], "€" if str(r.get("Currency", r.get("currency","USD"))) == "EUR" else "$"), axis=1)
    p_col = "Prezzo" if "Prezzo" in df.columns else "price"
    if p_col in df.columns: df["Prezzo_fmt"] = df.apply(lambda r: fmt_currency(r[p_col], "€" if str(r.get("Currency", r.get("currency","USD"))) == "EUR" else "$"), axis=1)
    return df
def add_links(df):
    c = "Ticker" if "Ticker" in df.columns else "ticker"
    if c in df.columns:
        df["Yahoo"] = df[c].apply(lambda t: f"https://finance.yahoo.com/quote/{t}")
        df["TV"] = df[c].apply(lambda t: f"https://www.tradingview.com/chart/?symbol={str(t).split('.')[0]}")
    return df

# --- DB ---
DB_PATH = Path("watchlist.db")
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, name TEXT, trend TEXT, origine TEXT, note TEXT, list_name TEXT, created_at TEXT)")
    conn.commit(); conn.close()
def add_to_watchlist(tkrs, names, orig, note, trend="LONG", list_name="DEFAULT"):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tkrs, names):
        conn.execute("INSERT INTO watchlist (ticker,name,trend,origine,note,list_name,created_at) VALUES (?,?,?,?,?,?,?)", (t,n,trend,orig,note,list_name,now))
    conn.commit(); conn.close()
def load_watchlist():
    if not DB_PATH.exists(): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH); df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close(); return df
init_db()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configurazione")
MARKETS = {
    "FTSE MIB": ["UCG.MI", "ISP.MI", "ENEL.MI", "ENI.MI", "LDO.MI", "PRY.MI", "STM.MI", "TEN.MI", "A2A.MI", "AMP.MI"],
    "Nasdaq 100": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "NFLX", "ADBE"],
    "S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "V", "UNH", "PG"],
    "Eurostoxx 600": ["ASML", "MC.PA", "SAP", "OR.PA", "TTE.PA", "SIE.DE", "NESN.SW"],
    "Materie Prime": ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F"],
    "ETF": ["SPY", "QQQ", "IWM", "GLD", "TLT"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
}
st.sidebar.subheader("🚀 Scanner V11")
sel_mkts = [m for m in MARKETS.keys() if st.sidebar.checkbox(m, value=(m in ["FTSE MIB", "Nasdaq 100"]), key=f"v11_{m}")]
if st.sidebar.button("🚀 AVVIA SCANNER V11", type="primary", use_container_width=True):
    uni = []
    for m in sel_mkts: uni.extend(MARKETS[m])
    Path("data").mkdir(exist_ok=True)
    Path("data/runtime_universe.json").write_text(json.dumps({"tickers": list(set(uni))}))
    with st.spinner("Scansione V11..."): run_scan()
    st.success("Fatto!"); st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🎛️ Parametri PRO")
top_n = st.sidebar.number_input("TOP N", 5, 50, 15)
e_h = st.sidebar.slider("EARLY (%)", 0.0, 10.0, 2.0) / 100
p_rmin, p_rmax = st.sidebar.slider("PRO RSI", 0, 100, (40, 70))
r_poc = st.sidebar.slider("REA POC (%)", 0.0, 10.0, 2.0) / 100

wl_all = load_watchlist()
lists = sorted(wl_all["list_name"].unique()) if (not wl_all.empty and "list_name" in wl_all.columns) else ["DEFAULT"]
act_list = st.sidebar.selectbox("Lista attiva", lists, index=0)

@st.cache_data(ttl=3600)
def fetch_v9(t):
    try:
        y = yf.Ticker(t); h = y.history(period="6mo")
        if len(h)<40: return None
        i = y.info; c = h["Close"]; v = h["Volume"]
        ema = c.ewm(20).mean().iloc[-1]
        try:
            d = c.diff(); g = d.where(d>0,0).rolling(14).mean(); l = -d.where(d<0,0).rolling(14).mean()
            rsi = 100 - (100/(1+(g/l))).iloc[-1]
        except: rsi = 50
        return {"name": i.get("longName",t), "ticker": t, "price": c.iloc[-1], "rsi": rsi, "vol_today": v.iloc[-1], "vol_7d_avg": v.tail(7).mean(), "market_cap": i.get("marketCap",0), "ema20": ema, "vol_ratio": v.iloc[-1]/v.rolling(20).mean().iloc[-1]}
    except: return None

if st.sidebar.button("🚀 AVVIA SCANNER PRO 9.0", use_container_width=True):
    u = []
    for m in sel_mkts: u.extend(MARKETS[m])
    res = []
    pb = st.progress(0); st.info("Scansione PRO 9.0...")
    for i, t in enumerate(list(set(u))):
        d = fetch_v9(t)
        if d: res.append(d)
        pb.progress((i+1)/len(list(set(u))))
    st.session_state["df_v9"] = pd.DataFrame(res); st.rerun()

df_v9 = st.session_state.get("df_v9", pd.DataFrame())

# --- TABS ---
tabs = st.tabs(["🟢 EARLY", "🟣 PRO", "🟠 REA‑QUANT", "📈 Serafini", "🧊 Regime", "🕒 MTF", "📊 Finviz", "🗓️ Scan V11", "📌 Watchlist"])
for i, tab in enumerate(tabs):
    with tab:
        if i < 7:
            if df_v9.empty: st.info("Avvia Scanner PRO 9.0")
            else:
                d = df_v9.copy()
                if i==0: d = d[abs(d["price"]-d["ema20"])/d["ema20"] <= e_h]
                elif i==1: d = d[(d["price"]>d["ema20"]) & (d["rsi"]>=p_rmin) & (d["rsi"]<=p_rmax)]
                elif i==2: d = d[d["vol_ratio"]>1.5]
                if not d.empty:
                    d = add_formatted_cols(d); d = add_links(d)
                    st.dataframe(d.head(top_n), hide_index=True, use_container_width=True, column_config={"Yahoo":st.column_config.LinkColumn("Yahoo", display_text="Apri"), "TV":st.column_config.LinkColumn("TV", display_text="Apri")})
                    sel = st.multiselect("Salva:", [f"{r['name']} ({r['ticker']})" for _,r in d.head(top_n).iterrows()], key=f"s_{i}")
                    if st.button("📌 Salva", key=f"b_{i}"):
                        add_to_watchlist([s.split(" (")[-1][:-1] for s in sel], [s.split(" (")[0] for s in sel], "V9", "", list_name=act_list)
                        st.success("Salvati")
        elif i == 7:
            p = Path("data/scan_results.json")
            if p.exists():
                dv = pd.DataFrame(json.loads(p.read_text()))
                if not dv.empty:
                    dv = add_formatted_cols(dv); dv = add_links(dv)
                    st.dataframe(dv.style.applymap(color_signal, subset=["signal"] if "signal" in dv.columns else []), hide_index=True, use_container_width=True, column_config={"Yahoo":st.column_config.LinkColumn("Yahoo", display_text="Apri"), "TV":st.column_config.LinkColumn("TV", display_text="Apri")})
            else: st.info("Esegui Scanner V11")
        elif i == 8:
            dw = load_watchlist()
            if not dw.empty:
                dwf = dw[dw["list_name"]==act_list] if "list_name" in dw.columns else dw
                st.dataframe(dwf[["name","ticker","trend","origine","note","created_at"]], hide_index=True, use_container_width=True)
            else: st.info("Vuota")
