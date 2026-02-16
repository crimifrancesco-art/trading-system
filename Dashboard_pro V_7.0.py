import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import io
import sqlite3
from pathlib import Path
from fpdf import FPDF  # pip install fpdf2

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 7.0",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="auto",
)

st.title("📊 Trading Scanner – Versione PRO 7.0")
st.caption(
    "EARLY • PRO • REA‑QUANT • Rea Quant • Serafini • Regime & Momentum • "
    "Multi‑Timeframe • Finviz‑style • Fondamentali • Export TradingView • Watchlist DB"
)

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
            origine TEXT,
            note TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def add_to_watchlist(tickers, names, origine, note):
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute(
            "INSERT INTO watchlist (ticker, name, origine, note, created_at) VALUES (?,?,?,?,?)",
            (t, n, origine, note, now),
        )
    conn.commit()
    conn.close()

def load_watchlist():
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["id", "ticker", "name", "origine", "note", "created_at"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    for col in ["ticker", "name", "origine", "note", "created_at"]:
        if col not in df.columns:
            df[col] = ""
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

def reset_watchlist_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()

init_db()

# =============================================================================
# SIDEBAR – MERCATI E PARAMETRI
# =============================================================================
st.sidebar.title("⚙️ Configurazione")

st.sidebar.subheader("📈 Selezione Mercati")
m = {
    "Eurostoxx":   st.sidebar.checkbox("🇪🇺 Eurostoxx 600", False),
    "FTSE":        st.sidebar.checkbox("🇮🇹 FTSE MIB", False),
    "SP500":       st.sidebar.checkbox("🇺🇸 S&P 500", True),
    "Nasdaq":      st.sidebar.checkbox("🇺🇸 Nasdaq 100", False),
    "Dow":         st.sidebar.checkbox("🇺🇸 Dow Jones", False),
    "Russell":     st.sidebar.checkbox("🇺🇸 Russell 2000", False),
    "Commodities": st.sidebar.checkbox("🛢️ Materie Prime", False),
    "ETF":         st.sidebar.checkbox("📦 ETF", False),
    "Crypto":      st.sidebar.checkbox("₿ Crypto", False),
    "Emerging":    st.sidebar.checkbox("🌍 Emergenti", False),
}
sel = [k for k, v in m.items() if v]

st.sidebar.divider()
st.sidebar.subheader("🎛️ Parametri Scanner")

e_h    = st.sidebar.slider("EARLY - Distanza EMA20 (%)", 0.0, 10.0, 2.0, 0.5) / 100
p_rmin = st.sidebar.slider("PRO - RSI minimo", 0, 100, 40, 5)
p_rmax = st.sidebar.slider("PRO - RSI massimo", 0, 100, 70, 5)
r_poc  = st.sidebar.slider("REA - Distanza POC (%)", 0.0, 10.0, 2.0, 0.5) / 100

top = st.sidebar.number_input("TOP N titoli per tab", 5, 50, 15, 5)

if not sel:
    st.warning("⚠️ Seleziona almeno un mercato nel menu.")
    st.stop()

st.info(f"Mercati selezionati: **{', '.join(sel)}**")

# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================
@st.cache_data(ttl=3600)
def load_universe(markets):
    t = []
    if "SP500" in markets:
        sp = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        )["Symbol"].tolist()
        t += sp
    if "Nasdaq" in markets:
        t += [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
            "NFLX", "ADBE", "COST", "PEP", "CSCO", "INTC", "AMD"
        ]
    if "Dow" in markets:
        t += [
            "AAPL", "MSFT", "JPM", "V", "UNH", "JNJ", "WMT", "PG", "HD",
            "DIS", "KO", "MCD", "BA", "CAT", "GS"
        ]
    if "Russell" in markets:
        t += ["IWM", "VTWO"]
    if "FTSE" in markets:
        t += [
            "UCG.MI", "ISP.MI", "ENEL.MI", "ENI.MI", "LDO.MI",
            "PRY.MI", "STM.MI", "TEN.MI", "A2A.MI", "AMP.MI"
        ]
    if "Eurostoxx" in markets:
        t += [
            "ASML.AS", "NESN.SW", "SAN.PA", "TTE.PA",
            "AIR.PA", "MC.PA", "OR.PA", "SU.PA"
        ]
    if "Commodities" in markets:
        t += ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F"]
    if "ETF" in markets:
        t += ["SPY", "QQQ", "IWM", "GLD", "TLT", "VTI", "EEM"]
    if "Crypto" in markets:
        t += ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD"]
    if "Emerging" in markets:
        t += ["EEM", "EWZ", "INDA", "FXI"]
    return list(dict.fromkeys(t))

def calc_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data) < 40:
            return None, None, None

        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]

        info = yf.Ticker(ticker).info
        name = info.get("longName", info.get("shortName", ticker))[:50]

        price = float(c.iloc[-1])
        ema20 = float(c.ewm(20).mean().iloc[-1])
        sma20 = ema20
        sma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else ema20
        sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else ema20

        # EARLY
        dist_ema = abs(price - ema20) / ema20
        early_score = max(0, 8 - (dist_ema / e_h) * 8) if dist_ema <= 3 * e_h else 0

        # PRO
        pro_score = 0
        if price > ema20:
            pro_score += 3
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        rsi_val = float(rsi.iloc[-1])

        if p_rmin < rsi_val < p_rmax:
            pro_score += 3
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
        if vol_ratio > 1.2:
            pro_score += 2

        obv = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
        atr = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])

        atr_ratio = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])
        atr_expansion = atr_ratio > 1.2

        stato_ep = "PRO" if pro_score >= 8 else ("EARLY" if early_score >= 8 else "-")

        # REA‑QUANT
        tp = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dist_poc = abs(price - poc) / poc

        rea_score = 0
        if dist_poc < r_poc:
            rea_score += max(0, 7 - (dist_poc / r_poc) * 7)
        if vol_ratio > 1.5:
            rea_score += 2
        stato_rea = "HOT" if rea_score >= 7 else "-"

        avg_volume = float(v.rolling(20).mean().iloc[-1])
        rel_volume = vol_ratio
        eps_next_y = info.get("earningsGrowth", None)
        eps_next_5y = info.get("earningsQuarterlyGrowth", None)
        options_short = bool(info.get("sharesShort", 0) > 0)

        res_ep = {
            "Nome": name,
            "Ticker": ticker,
            "Prezzo": round(price, 2),
            "Early_Score": round(early_score, 2),
            "Pro_Score": round(pro_score, 2),
            "RSI": round(rsi_val, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "OBV_Trend": obv_trend,
            "ATR": round(atr_val, 2),
            "ATR_Exp": atr_expansion,
            "Stato": stato_ep,
        }

        res_rea = {
            "Nome": name,
            "Ticker": ticker,
            "Prezzo": round(price, 2),
            "Rea_Score": round(rea_score, 2),
            "POC": round(poc, 2),
            "Dist_POC_%": round(dist_poc * 100, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "Stato": stato_rea,
        }

        finviz_features = {
            "Ticker": ticker,
            "Nome": name,
            "Prezzo": price,
            "EPS_NextY": eps_next_y,
            "EPS_Next5Y": eps_next_5y,
            "Avg_Volume": avg_volume / 1000.0,
            "Rel_Volume": rel_volume,
            "Has_OptionsShort": options_short,
            "Above_SMA20": price > sma20,
            "Above_SMA50": price > sma50,
            "Above_SMA200": price > sma200,
        }

        return res_ep, res_rea, finviz_features

    except Exception:
        return None, None, None
            
# =============================================================================
# FUNZIONI FONDAMENTALI YFINANCE
# =============================================================================
@st.cache_data(ttl=3600)
def get_fundamentals_single(ticker: str) -> pd.DataFrame:
    """
    Restituisce un DataFrame 1 riga con i principali fondamentali di un singolo ticker.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return pd.DataFrame()

    row = {
        "Ticker": ticker,
        "Nome": info.get("longName", info.get("shortName", ticker)),
        "MarketCap": info.get("marketCap"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Country": info.get("country"),

        "PE": info.get("trailingPE"),
        "Forward_PE": info.get("forwardPE"),
        "PEG_5Y": info.get("pegRatio"),
        "Price_to_Book": info.get("priceToBook"),

        "Profit_Margin": info.get("profitMargins"),
        "Operating_Margin": info.get("operatingMargins"),
        "Gross_Margin": info.get("grossMargins"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),

        "Debt_to_Equity": info.get("debtToEquity"),
        "Current_Ratio": info.get("currentRatio"),
        "Quick_Ratio": info.get("quickRatio"),

        "Div_Yield": info.get("dividendYield"),
        "Payout_Ratio": info.get("payoutRatio"),
        "Dividend_Rate": info.get("dividendRate"),

        "EPS_Growth_NextY": info.get("earningsGrowth"),
        "EPS_Growth_5Y": info.get("earningsQuarterlyGrowth"),
        "Revenue_Growth": info.get("revenueGrowth"),
    }
    df = pd.DataFrame([row])

    # converto alcune colonne in percentuale
    pct_cols = [
        "Profit_Margin", "Operating_Margin", "Gross_Margin",
        "ROE", "ROA", "Div_Yield", "Payout_Ratio",
        "EPS_Growth_NextY", "EPS_Growth_5Y", "Revenue_Growth",
    ]
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c] * 100

    return df


@st.cache_data(ttl=3600)
def fetch_fundamentals_bulk(tickers: list[str]) -> pd.DataFrame:
    """
    Scarica i fondamentali per una lista di ticker, usando get_fundamentals_single.
    """
    records = []
    for tkr in tickers:
        df_one = get_fundamentals_single(tkr)
        if not df_one.empty:
            records.append(df_one.iloc[0].to_dict())
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
            
     
     
            

        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]

        info = yf.Ticker(ticker).info
        name = info.get("longName", info.get("shortName", ticker))[:50]

        price = float(c.iloc[-1])
        ema20 = float(c.ewm(20).mean().iloc[-1])
        sma20 = ema20
        sma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else ema20
        sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else ema20

        # EARLY
        dist_ema = abs(price - ema20) / ema20
        early_score = max(0, 8 - (dist_ema / e_h) * 8) if dist_ema <= 3 * e_h else 0

        # PRO
        pro_score = 0
        if price > ema20:
            pro_score += 3
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        rsi_val = float(rsi.iloc[-1])
        if p_rmin < rsi_val < p_rmax:
            pro_score += 3
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
        if vol_ratio > 1.2:
            pro_score += 2

        obv = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
        atr = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])
        atr_ratio = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])
        atr_expansion = atr_ratio > 1.2

        stato_ep = "PRO" if pro_score >= 8 else ("EARLY" if early_score >= 8 else "-")

        # REA‑QUANT
        tp = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dist_poc = abs(price - poc) / poc

        rea_score = 0
        if dist_poc < r_poc:
            rea_score += max(0, 7 - (dist_poc / r_poc) * 7)
        if vol_ratio > 1.5:
            rea_score += 2
        stato_rea = "HOT" if rea_score >= 7 else "-"

        avg_volume = float(v.rolling(20).mean().iloc[-1])
        rel_volume = vol_ratio
        eps_next_y = info.get("earningsGrowth", None)
        eps_next_5y = info.get("earningsQuarterlyGrowth", None)
        options_short = bool(info.get("sharesShort", 0) > 0)

        res_ep = {
            "Nome": name,
            "Ticker": ticker,
            "Prezzo": round(price, 2),
            "Early_Score": round(early_score, 2),
            "Pro_Score": round(pro_score, 2),
            "RSI": round(rsi_val, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "OBV_Trend": obv_trend,
            "ATR": round(atr_val, 2),
            "ATR_Exp": atr_expansion,
            "Stato": stato_ep,
        }

        res_rea = {
            "Nome": name,
            "Ticker": ticker,
            "Prezzo": round(price, 2),
            "Rea_Score": round(rea_score, 2),
            "POC": round(poc, 2),
            "Dist_POC_%": round(dist_poc * 100, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "Stato": stato_rea,
        }

        finviz_features = {
            "Ticker": ticker,
            "Nome": name,
            "Prezzo": price,
            "EPS_NextY": eps_next_y,
            "EPS_Next5Y": eps_next_5y,
            "Avg_Volume": avg_volume / 1000.0,
            "Rel_Volume": rel_volume,
            "Has_OptionsShort": options_short,
            "Above_SMA20": price > sma20,
            "Above_SMA50": price > sma50,
            "Above_SMA200": price > sma200,
        }

        return res_ep, res_rea, finviz_features

    except Exception:
        return None, None, None

# =============================================================================
# FUNZIONE FONDAMENTALI SINGOLO TICKER
# =============================================================================
@st.cache_data(ttl=3600)
def get_fundamentals_single(ticker: str) -> pd.DataFrame:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return pd.DataFrame()

    row = {
        "Ticker": ticker,
        "Nome": info.get("longName", info.get("shortName", ticker)),
        "MarketCap": info.get("marketCap"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Country": info.get("country"),
        "PE": info.get("trailingPE"),
        "Forward_PE": info.get("forwardPE"),
        "PEG_5Y": info.get("pegRatio"),
        "Price_to_Book": info.get("priceToBook"),
        "Profit_Margin": info.get("profitMargins"),
        "Operating_Margin": info.get("operatingMargins"),
        "Gross_Margin": info.get("grossMargins"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Debt_to_Equity": info.get("debtToEquity"),
        "Current_Ratio": info.get("currentRatio"),
        "Quick_Ratio": info.get("quickRatio"),
        "Div_Yield": info.get("dividendYield"),
        "Payout_Ratio": info.get("payoutRatio"),
        "Dividend_Rate": info.get("dividendRate"),
        "EPS_Growth_NextY": info.get("earningsGrowth"),
        "EPS_Growth_5Y": info.get("earningsQuarterlyGrowth"),
        "Revenue_Growth": info.get("revenueGrowth"),
    }
    df = pd.DataFrame([row])

    pct_cols = [
        "Profit_Margin", "Operating_Margin", "Gross_Margin",
        "ROE", "ROA", "Div_Yield", "Payout_Ratio",
        "EPS_Growth_NextY", "EPS_Growth_5Y", "Revenue_Growth",
    ]
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c] * 100
    return df

# =============================================================================
# SCAN
# =============================================================================
if "done_pro" not in st.session_state:
    st.session_state["done_pro"] = False

if st.button("🚀 AVVIA SCANNER PRO 7.0", type="primary", use_container_width=True):
    universe = load_universe(sel)
    st.info(f"Scansione in corso su {len(universe)} titoli...")

    pb = st.progress(0)
    status = st.empty()

    r_ep, r_rea, r_finviz = [], [], []

    for i, tkr in enumerate(universe):
        status.text(f"Analisi: {tkr} ({i+1}/{len(universe)})")
        ep, rea, fv = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc)
        if ep:
            r_ep.append(ep)
        if rea:
            r_rea.append(rea)
        if fv:
            r_finviz.append(fv)
        pb.progress((i + 1) / len(universe))
        if (i + 1) % 10 == 0:
            time.sleep(0.1)

    status.text("✅ Scansione completata.")
    pb.empty()

    st.session_state["df_ep_pro"] = pd.DataFrame(r_ep)
    st.session_state["df_rea_pro"] = pd.DataFrame(r_rea)
    st.session_state["df_finviz"] = pd.DataFrame(r_finviz)
    st.session_state["done_pro"] = True

    st.rerun()

if not st.session_state.get("done_pro"):
    st.stop()

df_ep = st.session_state.get("df_ep_pro", pd.DataFrame())
df_rea = st.session_state.get("df_rea_pro", pd.DataFrame())
df_finviz = st.session_state.get("df_finviz", pd.DataFrame())

# =============================================================================
# RISULTATI SCANNER – METRICHE
# =============================================================================
if "Stato" in df_ep.columns:
    df_early_all = df_ep[df_ep["Stato"] == "EARLY"].copy()
    df_pro_all   = df_ep[df_ep["Stato"] == "PRO"].copy()
else:
    df_early_all = pd.DataFrame()
    df_pro_all   = pd.DataFrame()

if "Stato" in df_rea.columns:
    df_rea_all = df_rea[df_rea["Stato"] == "HOT"].copy()
else:
    df_rea_all = pd.DataFrame()

n_early = len(df_early_all)
n_pro   = len(df_pro_all)
n_rea   = len(df_rea_all)
n_tot   = n_early + n_pro + n_rea

st.header("Panoramica segnali")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Segnali EARLY", n_early)
c2.metric("Segnali PRO", n_pro)
c3.metric("Segnali REA‑QUANT", n_rea)
c4.metric("Totale segnali scanner", n_tot)

# =============================================================================
# TABS
# =============================================================================
tab_e, tab_p, tab_r, tab_rea_q, tab_serafini, tab_regime, tab_mtf, tab_finviz_tab, tab_funda, tab_watch = st.tabs(
    [
        "🟢 EARLY",
        "🟣 PRO",
        "🟠 REA‑QUANT",
        "🧮 Rea Quant",
        "📈 Serafini Systems",
        "🧊 Regime & Momentum",
        "🕒 Multi‑Timeframe",
        "📊 Finviz",
        "📑 Fondamentali",
        "📌 Watchlist & Note",
    ]
)

# =============================================================================
# EARLY
# =============================================================================
with tab_e:
    st.subheader("🟢 Segnali EARLY")
    st.markdown(
        f"Filtro EARLY: titoli con **Stato = EARLY** (distanza prezzo–EMA20 < {e_h*100:.1f}%), "
        "punteggio Early_Score elevato."
    )

    if df_early_all.empty:
        st.caption("Nessun segnale EARLY.")
    else:
        df_early = df_early_all.copy()
        df_early_view = df_early.sort_values("Early_Score", ascending=False).head(top)
        st.dataframe(df_early_view, use_container_width=True)

        # Watchlist
        options_early = [
            f"{row['Ticker']} – {row['Nome']}" for _, row in df_early_view.iterrows()
        ]
        selection_early = st.multiselect(
            "Aggiungi alla Watchlist (EARLY):",
            options=options_early,
            key="wl_early",
        )
        note_early = st.text_input("Note comuni per questi ticker EARLY", key="note_wl_early")
        if st.button("📌 Salva in Watchlist (EARLY)"):
            tickers = [s.split(" – ")[0] for s in selection_early]
            names   = [s.split(" – ")[1] for s in selection_early]
            add_to_watchlist(tickers, names, "EARLY", note_early)
            st.success("EARLY salvati in watchlist.")
            st.experimental_rerun()

        # Fondamentali popup
        st.markdown("### 🔍 Analisi fondamentale yfinance (EARLY)")
        tickers_list = df_early_view["Ticker"].tolist()
        if tickers_list:
            sel_tkr = st.selectbox(
                "Seleziona il titolo:",
                options=tickers_list,
                key="funda_early_select",
            )
            if st.button("🔍 Mostra fondamentali (EARLY)", key="btn_funda_early"):
                df_funda_single = get_fundamentals_single(sel_tkr)
                if df_funda_single.empty:
                    st.warning("Fondamentali non disponibili per questo ticker.")
                else:
                    with st.expander(f"Fondamentali yfinance – {sel_tkr}", expanded=True):
                        st.dataframe(df_funda_single.T, use_container_width=True)

# =============================================================================
# PRO
# =============================================================================
with tab_p:
    st.subheader("🟣 Segnali PRO")

    if df_pro_all.empty:
        st.caption("Nessun segnale PRO.")
    else:
        df_pro = df_pro_all.copy()
        df_pro_view = df_pro.sort_values("Pro_Score", ascending=False).head(top)

        df_pro_view["OBV_Trend"] = df_pro_view["OBV_Trend"].replace(
            {"UP": "UP (flusso in ingresso)", "DOWN": "DOWN (flusso in uscita)"}
        )

        st.dataframe(df_pro_view, use_container_width=True)

        # Watchlist
        options_pro = [
            f"{row['Ticker']} – {row['Nome']}" for _, row in df_pro_view.iterrows()
        ]
        selection_pro = st.multiselect(
            "Aggiungi alla Watchlist (PRO):",
            options=options_pro,
            key="wl_pro",
        )
        note_pro = st.text_input("Note comuni per questi ticker PRO", key="note_wl_pro")
        if st.button("📌 Salva in Watchlist (PRO)"):
            tickers = [s.split(" – ")[0] for s in selection_pro]
            names   = [s.split(" – ")[1] for s in selection_pro]
            add_to_watchlist(tickers, names, "PRO", note_pro)
            st.success("PRO salvati in watchlist.")
            st.experimental_rerun()

        # Fondamentali popup
        st.markdown("### 🔍 Analisi fondamentale yfinance (PRO)")
        tickers_list = df_pro_view["Ticker"].tolist()
        sel_tkr = st.selectbox(
            "Seleziona il titolo:",
            options=tickers_list,
            key="funda_pro_select",
        )
        if st.button("🔍 Mostra fondamentali (PRO)", key="btn_funda_pro"):
            df_funda_single = get_fundamentals_single(sel_tkr)
            if df_funda_single.empty:
                st.warning("Fondamentali non disponibili per questo ticker.")
            else:
                with st.expander(f"Fondamentali yfinance – {sel_tkr}", expanded=True):
                    st.dataframe(df_funda_single.T, use_container_width=True)

# =============================================================================
# REA‑QUANT (segnali)
# =============================================================================
with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT")

    if df_rea_all.empty:
        st.caption("Nessun segnale REA‑QUANT.")
    else:
        df_rea_view = df_rea_all.sort_values("Rea_Score", ascending=False).head(top)
        st.dataframe(df_rea_view, use_container_width=True)

        # Watchlist
        options_rea = [
            f"{row['Ticker']} – {row['Nome']}" for _, row in df_rea_view.iterrows()
        ]
        selection_rea = st.multiselect(
            "Aggiungi alla Watchlist (REA‑QUANT HOT):",
            options=options_rea,
            key="wl_rea",
        )
        note_rea = st.text_input("Note comuni per questi ticker REA‑QUANT", key="note_wl_rea")
        if st.button("📌 Salva in Watchlist (REA‑QUANT)"):
            tickers = [s.split(" – ")[0] for s in selection_rea]
            names   = [s.split(" – ")[1] for s in selection_rea]
            add_to_watchlist(tickers, names, "REA_HOT", note_rea)
            st.success("REA‑QUANT salvati in watchlist.")
            st.experimental_rerun()

        # Fondamentali popup
        st.markdown("### 🔍 Analisi fondamentale yfinance (REA‑QUANT)")
        tickers_list = df_rea_view["Ticker"].tolist()
        sel_tkr = st.selectbox(
            "Seleziona il titolo:",
            options=tickers_list,
            key="funda_rea_select",
        )
        if st.button("🔍 Mostra fondamentali (REA‑QUANT)", key="btn_funda_rea"):
            df_funda_single = get_fundamentals_single(sel_tkr)
            if df_funda_single.empty:
                st.warning("Fondamentali non disponibili per questo ticker.")
            else:
                with st.expander(f"Fondamentali yfinance – {sel_tkr}", expanded=True):
                    st.dataframe(df_funda_single.T, use_container_width=True)

# =============================================================================
# (qui lascio invariati Rea Quant, Serafini, Regime, MTF – puoi copiare lo stesso pattern di popup se vuoi)
# =============================================================================

# ... [per brevità, mantieni i blocchi Rea Quant / Serafini / Regime / MTF e Finviz
# della versione 7.0 che hai già, e in Finviz aggiungi la stessa sezione popup come fatto sopra] ...

# =============================================================================
# 📑 TAB FONDAMENTALI GENERALE
# =============================================================================
with tab_funda:
    st.subheader("📑 Analisi fondamentali (yfinance) – universo scansionato")

    if df_ep.empty:
        st.caption("Nessun dato di base disponibile (esegui lo scanner).")
    else:
        tickers_funda = df_ep["Ticker"].unique().tolist()

        @st.cache_data(ttl=3600)
        def fetch_fundamentals_bulk(tickers):
            rec = []
            for tkr in tickers:
                df = get_fundamentals_single(tkr)
                if not df.empty:
                    rec.append(df.iloc[0].to_dict())
            return pd.DataFrame(rec)

        with st.spinner("Download fondamentali per tutti i ticker..."):
            df_funda = fetch_fundamentals_bulk(tickers_funda)

        if df_funda.empty:
            st.caption("Impossibile recuperare fondamentali (limiti o assenza dati).")
        else:
            df_funda_view = df_funda.sort_values("MarketCap", ascending=False).head(100)
            st.dataframe(
                df_funda_view[[
                    "Nome", "Ticker", "Sector", "Industry",
                    "MarketCap", "PE", "Forward_PE", "PEG_5Y",
                    "Price_to_Book", "Profit_Margin", "Operating_Margin",
                    "ROE", "ROA", "Debt_to_Equity",
                    "Div_Yield", "Payout_Ratio",
                    "EPS_Growth_NextY", "Revenue_Growth"
                ]],
                use_container_width=True,
            )

# =============================================================================
# WATCHLIST & resto (esportazioni, ecc.) – puoi riusare esattamente la parte
# già funzionante dell'ultima versione 7.0 che ti ho dato.
# =============================================================================

