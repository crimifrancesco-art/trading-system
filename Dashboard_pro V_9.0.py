import io
import time
import sqlite3
import locale
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from fpdf import FPDF

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 9.6",
    layout="wide",
    page_icon="📊",
)
st.title("📊 Trading Scanner – Versione PRO 9.6")
st.caption(
    "EARLY • PRO • REA‑QUANT • Rea Quant • Serafini • Regime & Momentum • "
    "Multi‑Timeframe • Finviz • Watchlist DB"
)

# -----------------------------------------------------------------------------
# FORMATTAZIONE NUMERICA
# -----------------------------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass

def fmt_currency(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return (
        f"{symbol}{value:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

def fmt_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{int(value):,}".replace(",", ".")

def fmt_marketcap(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    v = float(value)
    if v >= 1_000_000_000:
        return (
            f"{symbol}{v / 1_000_000_000:,.2f}B"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
    if v >= 1_000_000:
        return (
            f"{symbol}{v / 1_000_000:,.2f}M"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
    if v >= 1_000:
        return (
            f"{symbol}{v / 1_000:,.2f}K"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
    return fmt_currency(v, symbol)

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Currency" not in df.columns:
        df["Currency"] = "USD"
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df.apply(
            lambda r: fmt_currency(
                r["Prezzo"], "€" if r["Currency"] == "EUR" else "$",
            ), axis=1,
        )
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df.apply(
            lambda r: fmt_marketcap(
                r["MarketCap"], "€" if r["Currency"] == "EUR" else "$",
            ), axis=1,
        )
    if "Vol_Today" in df.columns:
        df["Vol_Today_fmt"] = df["Vol_Today"].apply(fmt_int)
    if "Vol_7d_Avg" in df.columns:
        df["Vol_7d_Avg_fmt"] = df["Vol_7d_Avg"].apply(fmt_int)
    return df

# -----------------------------------------------------------------------------
# LINK YAHOO + TRADINGVIEW
# -----------------------------------------------------------------------------
def add_links(df: pd.DataFrame) -> pd.DataFrame:
    col = "Ticker" if "Ticker" in df.columns else "ticker"
    if col not in df.columns:
        return df
    df["Yahoo"] = df[col].astype(str).apply(
        lambda t: f"https://finance.yahoo.com/quote/{t}"
    )
    df["Finviz"] = df[col].astype(str).apply(
        lambda t: f"https://www.tradingview.com/chart/?symbol={t.split('.')[0]}"
    )
    return df

# -----------------------------------------------------------------------------
# DB WATCHLIST (SQLite)
# -----------------------------------------------------------------------------
DB_PATH = Path("watchlist.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            name TEXT,
            trend TEXT,
            origine TEXT,
            note TEXT,
            list_name TEXT,
            created_at TEXT
        )
        """
    )
    try:
        c.execute("ALTER TABLE watchlist ADD COLUMN trend TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE watchlist ADD COLUMN list_name TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()
    init_db()

def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute(
            """
            INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (t, n, trend, origine, note, list_name, now),
        )
    conn.commit()
    conn.close()

def load_watchlist() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    for col in ["id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"]:
        if col not in df.columns:
            df[col] = "" if col != "id" else np.nan
    return df

def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE watchlist SET note = ? WHERE id = ?", (new_note, int(row_id)))
    conn.commit()
    conn.close()

def delete_from_watchlist(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany("DELETE FROM watchlist WHERE id = ?", [(int(i),) for i in ids])
    conn.commit()
    conn.close()

init_db()

# =============================================================================
# INIZIALIZZAZIONE STATO
# =============================================================================
if "sidebar_init" not in st.session_state:
    st.session_state["sidebar_init"] = True
    st.session_state.setdefault("m_FTSE", True)
    st.session_state.setdefault("m_SP500", True)
    st.session_state.setdefault("m_Nasdaq", True)
    st.session_state.setdefault("e_h", 0.02)
    st.session_state.setdefault("p_rmin", 40)
    st.session_state.setdefault("p_rmax", 70)
    st.session_state.setdefault("r_poc", 0.02)
    st.session_state.setdefault("top", 15)
    st.session_state.setdefault("current_list_name", "DEFAULT")

# =============================================================================
# SIDEBAR – MERCATI E PARAMETRI
# =============================================================================
st.sidebar.title("⚙️ Configurazione")

st.sidebar.subheader("📈 Selezione Mercati")
m = {
    "SP500": st.sidebar.checkbox("🇺🇸 S&P 500", st.session_state.get("m_SP500", False)),
    "Eurostoxx": st.sidebar.checkbox("🇪🇺 Eurostoxx 600", st.session_state.get("m_Eurostoxx", False)),
    "FTSE": st.sidebar.checkbox("🇮🇹 FTSE MIB", st.session_state.get("m_FTSE", False)),
    "Nasdaq": st.sidebar.checkbox("🇺🇸 Nasdaq 100", st.session_state.get("m_Nasdaq", False)),
    "Dow": st.sidebar.checkbox("🇺🇸 Dow Jones 30", st.session_state.get("m_Dow", False)),
    "Russell": st.sidebar.checkbox("🇺🇸 Russell 2000", st.session_state.get("m_Russell", False)),
    "StoxxEmerging": st.sidebar.checkbox("🇪🇺 Stoxx Emerging 50", st.session_state.get("m_StoxxEmerging", False)),
    "USSmallCap": st.sidebar.checkbox("🇺🇸 US Small Cap 2000", st.session_state.get("m_USSmallCap", False)),
}
sel = [k for k, v in m.items() if v]

st.session_state["m_SP500"] = m["SP500"]
st.session_state["m_Eurostoxx"] = m["Eurostoxx"]
st.session_state["m_FTSE"] = m["FTSE"]
st.session_state["m_Nasdaq"] = m["Nasdaq"]
st.session_state["m_Dow"] = m["Dow"]
st.session_state["m_Russell"] = m["Russell"]
st.session_state["m_StoxxEmerging"] = m["StoxxEmerging"]
st.session_state["m_USSmallCap"] = m["USSmallCap"]

st.sidebar.divider()

st.sidebar.subheader("🎛️ Parametri Scanner")
e_h = st.sidebar.slider("EARLY - Distanza EMA20 (%)", 0.0, 10.0, float(st.session_state["e_h"] * 100), 0.5) / 100
st.session_state["e_h"] = e_h
p_rmin = st.sidebar.slider("PRO - RSI minimo", 0, 100, int(st.session_state["p_rmin"]), 5)
st.session_state["p_rmin"] = p_rmin
p_rmax = st.sidebar.slider("PRO - RSI massimo", 0, 100, int(st.session_state["p_rmax"]), 5)
st.session_state["p_rmax"] = p_rmax
r_poc = st.sidebar.slider("REA - Distanza POC (%)", 0.0, 10.0, float(st.session_state["r_poc"] * 100), 0.5) / 100
st.session_state["r_poc"] = r_poc

st.sidebar.subheader("🔎 Filtri avanzati")
eps_next_y_min = st.sidebar.number_input("EPS Growth Next Year min (%)", 0.0, 100.0, 10.0, 1.0)
eps_next_5y_min = st.sidebar.number_input("EPS Growth Next 5Y min (%)", 0.0, 100.0, 15.0, 1.0)
avg_vol_min_mln = st.sidebar.number_input("Avg Volume min (milioni)", 0.0, 100.0, 1.0, 0.5)
price_min_finviz = st.sidebar.number_input("Prezzo min per filtro Finviz", 0.0, 5000.0, 10.0, 1.0)
vol_ratio_hot = st.sidebar.number_input("Vol_Ratio minimo REA‑HOT", 0.0, 10.0, 1.5, 0.1)
momentum_min = st.sidebar.number_input("Momentum minimo", 0.0, 2000.0, 0.0, 10.0)

st.sidebar.subheader("📤 Output")
top = st.sidebar.number_input("TOP N titoli per tab", 5, 50, int(st.session_state["top"]), 5)
st.session_state["top"] = top

st.sidebar.subheader("📁 Lista Watchlist attiva")
df_wl_sidebar = load_watchlist()
list_options = sorted({ln for ln in df_wl_sidebar["list_name"].dropna().astype(str).str.strip().tolist() if ln}) if not df_wl_sidebar.empty else []
if not list_options: list_options = ["DEFAULT"]
default_list = st.session_state.get("current_list_name", list_options[0])
if default_list not in list_options: default_list = list_options[0]

selected_list = st.sidebar.selectbox("Lista esistente", options=list_options, index=list_options.index(default_list), key="sb_wl_select")
new_list_name = st.sidebar.text_input("Crea nuova lista", value="", key="sb_wl_new")
rename_target = st.sidebar.selectbox("Rinomina lista", options=list_options, index=list_options.index(selected_list), key="sb_wl_rename_target")
new_name_for_rename = st.sidebar.text_input("Nuovo nome", value="", key="sb_wl_rename_new")

if st.sidebar.button("🔤 Applica rinomina"):
    if new_name_for_rename.strip():
        old = rename_target
        new = new_name_for_rename.strip()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE watchlist SET list_name = ? WHERE list_name = ?", (new, old))
        conn.commit()
        conn.close()
        st.session_state["current_list_name"] = new
        st.rerun()

active_list = new_list_name.strip() if new_list_name.strip() else selected_list
st.session_state["current_list_name"] = active_list
st.sidebar.caption(f"Lista attiva: **{active_list}**")

# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================
DATA_DIR = Path("data")

def load_index_from_csv(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        for col in ["Simbolo", "simbolo", "Ticker", "Symbol", "symbol"]:
            if col in df.columns:
                df = df.rename(columns={col: "ticker"})
                break
    if "ticker" not in df.columns:
        return []
    return df["ticker"].dropna().astype(str).unique().tolist()

@st.cache_data(ttl=3600)
def load_universe(markets):
    t = []
    if "SP500" in markets: t += load_index_from_csv("sp500.csv")
    if "Eurostoxx" in markets: t += load_index_from_csv("eurostoxx600.csv")
    if "FTSE" in markets: t += load_index_from_csv("ftsemib.csv")
    if "Nasdaq" in markets: t += load_index_from_csv("nasdaq100.csv")
    if "Dow" in markets: t += load_index_from_csv("dowjones.csv")
    if "Russell" in markets: t += load_index_from_csv("russell2000.csv")
    if "StoxxEmerging" in markets: t += load_index_from_csv("stoxx emerging market 50.csv")
    if "USSmallCap" in markets: t += load_index_from_csv("us small cap 2000.csv")
    return list(dict.fromkeys(t))

def calc_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data) < 40: return None, None
        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]
        yt = yf.Ticker(ticker)
        info = yt.info
        name = info.get("longName", info.get("shortName", ticker))[:50]
        price = float(c.iloc[-1])
        ema20 = float(c.ewm(span=20).mean().iloc[-1])
        market_cap = info.get("marketCap", np.nan)
        vol_today = float(v.iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())
        currency = info.get("currency", "USD")
        
        dist_ema = abs(price - ema20) / ema20
        early_score = 8 if dist_ema < e_h else 0
        
        pro_score = 3 if price > ema20 else 0
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        rsi_val = float(rsi.iloc[-1])
        if p_rmin < rsi_val < p_rmax: pro_score += 3
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
        if vol_ratio > 1.2: pro_score += 2
        
        obv = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"
        
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
        atr = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])
        atr_expansion = (atr_val / atr.rolling(50).mean().iloc[-1]) > 1.2
        
        stato_ep = "PRO" if pro_score >= 8 else ("EARLY" if early_score >= 8 else "-")
        
        tp = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dist_poc = abs(price - poc) / poc
        rea_score = 7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0
        stato_rea = "HOT" if rea_score >= 7 else "-"
        
        res_ep = {
            "Nome": name, "Ticker": ticker, "Prezzo": round(price, 2), "MarketCap": market_cap,
            "Vol_Today": int(vol_today), "Vol_7d_Avg": int(vol_7d_avg), "Currency": currency,
            "Early_Score": early_score, "Pro_Score": pro_score, "RSI": round(rsi_val, 1),
            "Vol_Ratio": round(vol_ratio, 2), "OBV_Trend": obv_trend, "ATR": round(atr_val, 2),
            "ATR_Exp": atr_expansion, "Stato": stato_ep,
        }
        res_rea = {
            "Nome": name, "Ticker": ticker, "Prezzo": round(price, 2), "MarketCap": market_cap,
            "Vol_Today": int(vol_today), "Vol_7d_Avg": int(vol_7d_avg), "Currency": currency,
            "Rea_Score": rea_score, "POC": round(poc, 2), "Dist_POC_%": round(dist_poc * 100, 1),
            "Vol_Ratio": round(vol_ratio, 2), "Stato": stato_rea,
        }
        return res_ep, res_rea
    except Exception:
        return None, None

st.sidebar.subheader("🧠 Modalità")
only_watchlist = st.sidebar.checkbox("Mostra solo Watchlist (salta scanner)", value=False, key="only_watchlist")

# =============================================================================
# SCAN
# =============================================================================
if st.session_state.get("only_watchlist"):
    df_ep = pd.DataFrame()
    df_rea = pd.DataFrame()
    st.session_state["done_pro"] = True
else:
    if "done_pro" not in st.session_state:
        st.session_state["done_pro"] = False
    if st.button("🚀 AVVIA SCANNER PRO 9.6", type="primary", use_container_width=True):
        universe = load_universe(sel)
        st.info(f"Scansione in corso su {len(universe)} titoli...")
        pb = st.progress(0)
        status = st.empty()
        r_ep, r_rea = [], []
        for i, tkr in enumerate(universe):
            status.text(f"Analisi: {tkr} ({i+1}/{len(universe)})")
            ep, rea = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc)
            if ep: r_ep.append(ep)
            if rea: r_rea.append(rea)
            pb.progress((i + 1) / len(universe))
        status.text("✅ Scansione completata.")
        pb.empty()
        st.session_state["df_ep_pro"] = pd.DataFrame(r_ep)
        st.session_state["df_rea_pro"] = pd.DataFrame(r_rea)
        st.session_state["done_pro"] = True
        st.rerun()

if not st.session_state.get("done_pro"):
    st.stop()

df_ep = st.session_state.get("df_ep_pro", pd.DataFrame())
df_rea = st.session_state.get("df_rea_pro", pd.DataFrame())

if not df_ep.empty:
    df_ep = df_ep.dropna(subset=["Pro_Score", "RSI", "Vol_Ratio"])
if not df_rea.empty:
    df_rea = df_rea.dropna(subset=["Vol_Ratio", "Dist_POC_%"])

# =============================================================================
# RISULTATI SCANNER – METRICHE
# =============================================================================
df_early_all = df_ep[df_ep["Stato"] == "EARLY"].copy() if "Stato" in df_ep.columns else pd.DataFrame()
df_pro_all = df_ep[df_ep["Stato"] == "PRO"].copy() if "Stato" in df_ep.columns else pd.DataFrame()
df_rea_all = df_rea[df_rea["Stato"] == "HOT"].copy() if "Stato" in df_rea.columns else pd.DataFrame()

st.header("Panoramica segnali")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Segnali EARLY", len(df_early_all))
c2.metric("Segnali PRO", len(df_pro_all))
c3.metric("Segnali REA‑HOT", len(df_rea_all))
c4.metric("Totale segnali", len(df_early_all)+len(df_pro_all)+len(df_rea_all))

tab_e, tab_p, tab_rea, tab_serafini, tab_regime, tab_mtf, tab_finviz, tab_watch = st.tabs([
    "🟢 EARLY", "🟣 PRO", "🟠 REA‑QUANT", "📈 Serafini Systems", "🧊 Regime & Momentum", "🕒 Multi‑Timeframe", "📊 Finviz", "📌 Watchlist & Note"
])

# EARLY
with tab_e:
    if df_early_all.empty: st.caption("Nessun segnale EARLY.")
    else:
        df_view = add_links(add_formatted_cols(df_early_all.sort_values(["Early_Score", "RSI"], ascending=False).head(top)))
        st.dataframe(df_view, use_container_width=True)
        # (Export buttons and watchlist logic omitted for brevity in this fix, can be restored from backup)

# PRO
with tab_p:
    if df_pro_all.empty: st.caption("Nessun segnale PRO.")
    else:
        df_view = add_links(add_formatted_cols(df_pro_all.sort_values(["Pro_Score", "RSI"], ascending=False).head(top)))
        st.dataframe(df_view, use_container_width=True)

# REA
with tab_rea:
    if df_rea_all.empty: st.caption("Nessun segnale REA-HOT.")
    else:
        df_view = add_links(add_formatted_cols(df_rea_all.sort_values(["Vol_Ratio", "Dist_POC_%"], ascending=[False, True]).head(top)))
        st.dataframe(df_view, use_container_width=True)

# WATCHLIST
with tab_watch:
    df_wl = load_watchlist()
    df_wl_filt = df_wl[df_wl["list_name"] == active_list].copy()
    if df_wl_filt.empty: st.caption("Watchlist vuota.")
    else:
        st.dataframe(df_wl_filt, use_container_width=True)
