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
from fpdf import FPDF  # pip install fpdf2

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
# usa locale di sistema; se serve forza "it_IT.UTF-8"
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
    # se manca Currency, default USD
    if "Currency" not in df.columns:
        df["Currency"] = "USD"

    # Prezzo
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df.apply(
            lambda r: fmt_currency(
                r["Prezzo"],
                "€" if r["Currency"] == "EUR" else "$",
            ),
            axis=1,
        )

    # MarketCap
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df.apply(
            lambda r: fmt_marketcap(
                r["MarketCap"],
                "€" if r["Currency"] == "EUR" else "$",
            ),
            axis=1,
        )

    # Volumi
    if "Vol_Today" in df.columns:
        df["Vol_Today_fmt"] = df["Vol_Today"].apply(fmt_int)

    if "Vol_7d_Avg" in df.columns:
        df["Vol_7d_Avg_fmt"] = df["Vol_7d_Avg"].apply(fmt_int)

    return df


# -----------------------------------------------------------------------------
# LINK YAHOO + TRADINGVIEW
# -----------------------------------------------------------------------------
def add_links(df: pd.DataFrame) -> pd.DataFrame:
    # usa 'Ticker' se c'è, altrimenti 'ticker'
    col = "Ticker" if "Ticker" in df.columns else "ticker"
    if col not in df.columns:
        return df

    df["Yahoo"] = df[col].astype(str).apply(
        lambda t: f"https://finance.yahoo.com/quote/{t}"
    )
    # link TradingView nella colonna Finviz (mantengo il nome per compatibilità)
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
    # Migrazione: aggiungi colonna trend se DB vecchio
    try:
        c.execute("ALTER TABLE watchlist ADD COLUMN trend TEXT")
    except sqlite3.OperationalError:
        pass

    # Migrazione: aggiungi colonna list_name se DB vecchio
    try:
        c.execute("ALTER TABLE watchlist ADD COLUMN list_name TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


def reset_watchlist_db():
    """Elimina completamente la tabella watchlist e la ricrea vuota."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()
    init_db()


def add_to_watchlist(
    tickers, names, origine, note, trend="LONG", list_name="DEFAULT"
):
    """Inserisce una lista di ticker in watchlist sulla lista indicata."""
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute(
            """
            INSERT INTO watchlist
            (ticker, name, trend, origine, note, list_name, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (t, n, trend, origine, note, list_name, now),
        )
    conn.commit()
    conn.close()


def load_watchlist() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=[
                "id",
                "ticker",
                "name",
                "trend",
                "origine",
                "note",
                "list_name",
                "created_at",
            ]
        )
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM watchlist ORDER BY created_at DESC", conn
    )
    conn.close()
    # garantisco la presenza delle colonne chiave
    for col in [
        "id",
        "ticker",
        "name",
        "trend",
        "origine",
        "note",
        "list_name",
        "created_at",
    ]:
        if col not in df.columns:
            df[col] = "" if col != "id" else np.nan
    return df


def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE watchlist SET note = ? WHERE id = ?", (new_note, int(row_id))
    )
    conn.commit()
    conn.close()


def delete_from_watchlist(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany(
        "DELETE FROM watchlist WHERE id = ?", [(int(i),) for i in ids]
    )
    conn.commit()
    conn.close()


# inizializza DB
init_db()

# =============================================================================
# INIZIALIZZAZIONE STATO
# =============================================================================
if "sidebar_init" not in st.session_state:
    st.session_state["sidebar_init"] = True

    # mercati
    st.session_state.setdefault("m_FTSE", True)
    st.session_state.setdefault("m_SP500", True)
    st.session_state.setdefault("m_Nasdaq", True)

    # parametri principali
    st.session_state.setdefault("e_h", 0.02)
    st.session_state.setdefault("p_rmin", 40)
    st.session_state.setdefault("p_rmax", 70)
    st.session_state.setdefault("r_poc", 0.02)
    st.session_state.setdefault("top", 15)

# lista attiva di default
if "current_list_name" not in st.session_state:
    st.session_state["current_list_name"] = "DEFAULT"

# =============================================================================
# SIDEBAR – MERCATI E PARAMETRI
# =============================================================================
st.sidebar.title("⚙️ Configurazione")

# ---------------- Selezione Mercati (persistente) ----------------
st.sidebar.subheader("📈 Selezione Mercati")
m = {
    "Eurostoxx": st.sidebar.checkbox("🇪🇺 Eurostoxx 600", False),
    "FTSE": st.sidebar.checkbox("🇮🇹 FTSE MIB", st.session_state["m_FTSE"]),
    "SP500": st.sidebar.checkbox("🇺🇸 S&P 500", st.session_state["m_SP500"]),
    "Nasdaq": st.sidebar.checkbox("🇺🇸 Nasdaq 100", st.session_state["m_Nasdaq"]),
    "Dow": st.sidebar.checkbox("🇺🇸 Dow Jones", False),
    "Russell": st.sidebar.checkbox("🇺🇸 Russell 2000", False),
    "Commodities": st.sidebar.checkbox("🛢️ Materie Prime", False),
    "ETF": st.sidebar.checkbox("📦 ETF", False),
    "Crypto": st.sidebar.checkbox("₿ Crypto", False),
    "Emerging": st.sidebar.checkbox("🌍 Emergenti", False),
}
sel = [k for k, v in m.items() if v]

# aggiorno stato mercati
st.session_state["m_FTSE"] = m["FTSE"]
st.session_state["m_SP500"] = m["SP500"]
st.session_state["m_Nasdaq"] = m["Nasdaq"]

st.sidebar.divider()

# ---------------- Parametri Scanner (persistenti) ----------------
st.sidebar.subheader("🎛️ Parametri Scanner")

e_h = (
    st.sidebar.slider(
        "EARLY - Distanza EMA20 (%)",
        0.0,
        10.0,
        float(st.session_state["e_h"] * 100),
        0.5,
    )
    / 100
)
st.session_state["e_h"] = e_h

p_rmin = st.sidebar.slider(
    "PRO - RSI minimo", 0, 100, int(st.session_state["p_rmin"]), 5
)
st.session_state["p_rmin"] = p_rmin

p_rmax = st.sidebar.slider(
    "PRO - RSI massimo", 0, 100, int(st.session_state["p_rmax"]), 5
)
st.session_state["p_rmax"] = p_rmax

r_poc = (
    st.sidebar.slider(
        "REA - Distanza POC (%)",
        0.0,
        10.0,
        float(st.session_state["r_poc"] * 100),
        0.5,
    )
    / 100
)
st.session_state["r_poc"] = r_poc

# ---------------- Filtri avanzati ----------------
st.sidebar.subheader("🔎 Filtri avanzati")

eps_next_y_min = st.sidebar.number_input(
    "EPS Growth Next Year min (%)", 0.0, 100.0, 10.0, 1.0
)
eps_next_5y_min = st.sidebar.number_input(
    "EPS Growth Next 5Y min (%)", 0.0, 100.0, 15.0, 1.0
)
avg_vol_min_mln = st.sidebar.number_input(
    "Avg Volume min (milioni)", 0.0, 100.0, 1.0, 0.5
)
price_min_finviz = st.sidebar.number_input(
    "Prezzo min per filtro Finviz", 0.0, 5000.0, 10.0, 1.0
)
vol_ratio_hot = st.sidebar.number_input(
    "Vol_Ratio minimo REA‑HOT", 0.0, 10.0, 1.5, 0.1
)
momentum_min = st.sidebar.number_input(
    "Momentum minimo (Pro_Score×10 + RSI)", 0.0, 2000.0, 0.0, 10.0
)

# ---------------- Output (persistente) ----------------
st.sidebar.subheader("📤 Output")
top = st.sidebar.number_input(
    "TOP N titoli per tab", 5, 50, int(st.session_state["top"]), 5
)
st.session_state["top"] = top

# ---------------- Lista Watchlist attiva ----------------
st.sidebar.subheader("📁 Lista Watchlist attiva")

df_wl_sidebar = load_watchlist()

# elenco liste esistenti dal DB
if not df_wl_sidebar.empty and "list_name" in df_wl_sidebar.columns:
    list_options = (
        df_wl_sidebar["list_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    list_options = sorted({ln for ln in list_options if ln})
else:
    list_options = []

if not list_options:
    list_options = ["DEFAULT"]

# lista corrente salvata in sessione
default_list = (
    st.session_state.get("current_list_name", list_options[0])
    if st.session_state.get("current_list_name", list_options[0]) in list_options
    else list_options[0]
)

# 1) scegli la lista attiva
selected_list = st.sidebar.selectbox(
    "Lista esistente",
    options=list_options,
    index=list_options.index(default_list),
    key="sb_wl_select",
    help="Seleziona una watchlist già presente.",
)

# 2) crea una nuova lista (vuota)
new_list_name = st.sidebar.text_input(
    "Crea nuova lista",
    value="",
    key="sb_wl_new",
    placeholder="Es. Swing, LT, Crypto...",
    help="Scrivi un nome e premi Invio per iniziare una nuova lista vuota.",
)

# 3) rinomina la lista selezionata
rename_target = st.sidebar.selectbox(
    "Rinomina lista",
    options=list_options,
    index=list_options.index(selected_list),
    key="sb_wl_rename_target",
    help="Scegli quale lista vuoi rinominare.",
)

new_name_for_rename = st.sidebar.text_input(
    "Nuovo nome per la lista selezionata",
    value="",
    key="sb_wl_rename_new",
    placeholder="Nuovo nome...",
)

if st.sidebar.button("🔤 Applica rinomina"):
    if new_name_for_rename.strip():
        old = rename_target
        new = new_name_for_rename.strip()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE watchlist SET list_name = ? WHERE list_name = ?",
            (new, old),
        )
        conn.commit()
        conn.close()
        st.sidebar.success(f"Lista '{old}' rinominata in '{new}'.")
        # aggiorno stato e forzo refresh nomi
        st.session_state["current_list_name"] = new
        st.rerun()

    else:
        st.sidebar.warning("Inserisci un nuovo nome per rinominare.")

# LOGICA LISTA ATTIVA
active_list = selected_list

# se compilato "Crea nuova lista", quella diventa la lista attiva (vuota finché non aggiungi titoli)
if new_list_name.strip():
    active_list = new_list_name.strip()

st.session_state["current_list_name"] = active_list
st.sidebar.caption(f"Lista attiva: **{active_list}**")

# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"  # [web:16]
EUROSTOXX600_URL = "https://stoxx.com/document/Indices/Download/History/weights/sxxp_historical_components.csv"  # [web:7][web:22]

def load_sp500_tickers():
    df = pd.read_csv(SP500_URL)  # contiene colonna Symbol con i ticker S&P 500 [web:16][web:18]
    return df["Symbol"].dropna().unique().tolist()

def load_eurostoxx600_tickers():
    df = pd.read_csv(EUROSTOXX600_URL)  # file storico componenti STOXX Europe 600 [web:22][web:24]
    date_col_candidates = ["Date", "HISTORY_DATE", "HistoryDate"]
    comp_col_candidates = ["Ticker", "TICKER_SYMBOL", "Instrument", "ISIN"]

    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is not None:
        latest_date = df[date_col].max()
        df = df[df[date_col] == latest_date]

    comp_col = None
    for c in comp_col_candidates:
        if c in df.columns:
            comp_col = c
            break

    if comp_col is None:
        return []

    return df[comp_col].dropna().unique().tolist()

@st.cache_data(ttl=3600)
def load_universe(markets):
    t = []

    # ========================
    # INDICI PRINCIPALI
    # ========================
    if "SP500" in markets:
        sp = load_sp500_tickers()
        t += sp

    if "Eurostoxx" in markets:
        eu = load_eurostoxx600_tickers()
        if not eu:
            eu = [
                "ASML.AS",
                "NESN.SW",
                "SAN.PA",
                "TTE.PA",
                "AIR.PA",
                "MC.PA",
                "OR.PA",
                "SU.PA",
            ]
        t += eu

# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================

import pandas as pd
import requests
from io import StringIO
import streamlit as st

# ------------------------
# URL INDICI PRINCIPALI
# ------------------------
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"  # [web:16]
EUROSTOXX600_URL = "https://stoxx.com/document/Indices/Download/History/weights/sxxp_historical_components.csv"  # [web:22][web:24]

# Nasdaq 100 da CSV pubblico (lista Nasdaq 100) [web:52][web:41]
NASDAQ100_CSV_URL = "https://raw.githubusercontent.com/azymnis/python-stocks/master/nasdaq100.csv"

# Dow Jones 30 da Slickcharts (tabella HTML) [web:39]
DOWJONES_URL = "https://www.slickcharts.com/dowjones"

# FTSE MIB da MarketScreener (tabella componenti) [web:34]
FTSEMIB_URL = "https://www.marketscreener.com/quote/index/FTSE-MIB-121229431/components/"

# Russell 2000: usiamo i componenti dell’ETF IWM da file CSV pubblico (proxy pratico dell’indice) [web:51]
RUSSELL2000_CSV_URL = "https://datahub.io/core/russell-2000-companies/r/russell-2000-companies.csv"


# ------------------------
# FUNZIONI DI CARICAMENTO
# ------------------------

def load_sp500_tickers():
    df = pd.read_csv(SP500_URL)
    return df["Symbol"].dropna().unique().tolist()


def load_eurostoxx600_tickers():
    df = pd.read_csv(EUROSTOXX600_URL)
    date_col_candidates = ["Date", "HISTORY_DATE", "HistoryDate"]
    comp_col_candidates = ["Ticker", "TICKER_SYMBOL", "Instrument", "ISIN"]

    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is not None:
        latest_date = df[date_col].max()
        df = df[df[date_col] == latest_date]

    comp_col = None
    for c in comp_col_candidates:
        if c in df.columns:
            comp_col = c
            break

    if comp_col is None:
        return []

    return df[comp_col].dropna().unique().tolist()


def load_nasdaq100_tickers():
    try:
        df = pd.read_csv(NASDAQ100_CSV_URL)
    except Exception:
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
            "NFLX", "ADBE", "COST", "PEP", "CSCO", "INTC", "AMD"
        ]

    for col in ["Ticker", "Symbol", "Symbols"]:
        if col in df.columns:
            return df[col].dropna().unique().tolist()

    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        "NFLX", "ADBE", "COST", "PEP", "CSCO", "INTC", "AMD"
    ]


def load_dow30_tickers():
    try:
        tables = pd.read_html(DOWJONES_URL)
    except Exception:
        return [
            "AAPL", "MSFT", "JPM", "V", "UNH", "JNJ", "WMT", "PG",
            "HD", "DIS", "KO", "MCD", "BA", "CAT", "GS"
        ]

    df = tables[0]
    symbol_col = None
    for col in ["Symbol", "Ticker"]:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        return [
            "AAPL", "MSFT", "JPM", "V", "UNH", "JNJ", "WMT", "PG",
            "HD", "DIS", "KO", "MCD", "BA", "CAT", "GS"
        ]

    return df[symbol_col].dropna().unique().tolist()


def load_ftsemib_tickers():
    try:
        tables = pd.read_html(FTSEMIB_URL)
    except Exception:
        return [
            "UCG.MI", "ISP.MI", "ENEL.MI", "ENI.MI", "LDO.MI",
            "PRY.MI", "STM.MI", "TEN.MI", "A2A.MI", "AMP.MI",
        ]

    df = tables[0]
    symbol_col = None
    for col in ["Ticker", "Symbol", "Codice", "Code"]:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        return [
            "UCG.MI", "ISP.MI", "ENEL.MI", "ENI.MI", "LDO.MI",
            "PRY.MI", "STM.MI", "TEN.MI", "A2A.MI", "AMP.MI",
        ]

    tickers = df[symbol_col].dropna().astype(str).unique().tolist()
    norm = []
    for tk in tickers:
        tk = tk.strip()
        if not tk.endswith(".MI"):
            tk = tk + ".MI"
        norm.append(tk)

    return norm


def load_russell2000_tickers():
    """Universo Russell 2000 tramite lista componenti (proxy IWM)."""
    try:
        df = pd.read_csv(RUSSELL2000_CSV_URL)
    except Exception:
        # fallback ETF proxy se il file non è disponibile
        return ["IWM", "VTWO"]

    # proviamo varie colonne tipiche per il ticker
    for col in ["Symbol", "Ticker", "Ticker Symbol"]:
        if col in df.columns:
            return df[col].dropna().unique().tolist()

    return ["IWM", "VTWO"]


# ------------------------
# UNIVERSO MERCATI
# ------------------------

@st.cache_data(ttl=3600)
def load_universe(markets):
    t = []

    # INDICI PRINCIPALI
    if "SP500" in markets:
        t += load_sp500_tickers()

    if "Eurostoxx" in markets:
        eu = load_eurostoxx600_tickers()
        if not eu:
            eu = [
                "ASML.AS", "NESN.SW", "SAN.PA", "TTE.PA",
                "AIR.PA", "MC.PA", "OR.PA", "SU.PA",
            ]
        t += eu

    if "Nasdaq" in markets:
        t += load_nasdaq100_tickers()

    if "Dow" in markets:
        t += load_dow30_tickers()

    if "FTSE" in markets:
        t += load_ftsemib_tickers()

    # ALTRI MERCATI
    if "Russell" in markets:
        t += load_russell2000_tickers()

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
            return None, None

        c = data["Close"]
        h = data["High"]
        l = data["Low"]
        v = data["Volume"]

        yt = yf.Ticker(ticker)
        info = yt.info
        name = info.get("longName", info.get("shortName", ticker))[:50]

        price = float(c.iloc[-1])
        ema20 = float(c.ewm(20).mean().iloc[-1])

        # Capitalizzazione, volumi e valuta
        market_cap = info.get("marketCap", np.nan)
        vol_today = float(v.iloc[-1])
        vol_7d_avg = float(v.tail(7).mean())
        currency = info.get("currency", "USD")

        # EARLY
        dist_ema = abs(price - ema20) / ema20
        early_score = 8 if dist_ema < e_h else 0

        # PRO
        pro_score = 3 if price > ema20 else 0

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

        tr = np.maximum(
            h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift()))
        )
        atr = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])

        atr_ratio = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])
        atr_expansion = atr_ratio > 1.2

        stato_ep = (
            "PRO"
            if pro_score >= 8
            else ("EARLY" if early_score >= 8 else "-")
        )

        # REA‑QUANT
        tp = (h + l + c) / 3
        bins = np.linspace(float(l.min()), float(h.max()), 50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = (
            pd.DataFrame({"P": price_bins, "V": v})
            .groupby("P")["V"]
            .sum()
        )
        poc = float(vp.idxmax())
        dist_poc = abs(price - poc) / poc

        rea_score = (
            7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0
        )
        stato_rea = "HOT" if rea_score >= 7 else "-"

        res_ep = {
            "Nome": name,
            "Ticker": ticker,
            "Prezzo": round(price, 2),
            "MarketCap": market_cap,
            "Vol_Today": int(vol_today),
            "Vol_7d_Avg": int(vol_7d_avg),
            "Currency": currency,
            "Early_Score": early_score,
            "Pro_Score": pro_score,
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
            "MarketCap": market_cap,
            "Vol_Today": int(vol_today),
            "Vol_7d_Avg": int(vol_7d_avg),
            "Currency": currency,
            "Rea_Score": rea_score,
            "POC": round(poc, 2),
            "Dist_POC_%": round(dist_poc * 100, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "Stato": stato_rea,
        }

        return res_ep, res_rea

    except Exception:
        return None, None

st.sidebar.subheader("🧠 Modalità")
only_watchlist = st.sidebar.checkbox(
    "Mostra solo Watchlist (salta scanner)",
    value=False,
    key="only_watchlist",
)


# =============================================================================
# SCAN
# =============================================================================
if st.session_state.get("only_watchlist"):
    # non eseguo scanner, ma mi serve comunque df_ep/df_rea vuoti
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
            if ep:
                r_ep.append(ep)
            if rea:
                r_rea.append(rea)
            pb.progress((i + 1) / len(universe))
            if (i + 1) % 10 == 0:
                time.sleep(0.1)

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

# pulizia righe con NaN sulle colonne chiave
if not df_ep.empty:
    cols_ep = [c for c in ["Pro_Score", "RSI", "Vol_Ratio"] if c in df_ep.columns]
    if cols_ep:
        df_ep = df_ep.dropna(subset=cols_ep)

if not df_rea.empty:
    cols_rea = [c for c in ["Vol_Ratio", "Distanza_POC"] if c in df_rea.columns]
    if cols_rea:
        df_rea = df_rea.dropna(subset=cols_rea)


# =============================================================================
# RISULTATI SCANNER – METRICHE
# =============================================================================
if "Stato" in df_ep.columns:
    df_early_all = df_ep[df_ep["Stato"] == "EARLY"].copy()
    df_pro_all = df_ep[df_ep["Stato"] == "PRO"].copy()
else:
    df_early_all = pd.DataFrame()
    df_pro_all = pd.DataFrame()

if "Stato" in df_rea.columns:
    df_rea_all = df_rea[df_rea["Stato"] == "HOT"].copy()
else:
    df_rea_all = pd.DataFrame()

n_early = len(df_early_all)
n_pro = len(df_pro_all)
n_rea = len(df_rea_all)
n_tot = n_early + n_pro + n_rea

st.header("Panoramica segnali")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Segnali EARLY", n_early)
c2.metric("Segnali PRO", n_pro)
c3.metric("Segnali REA‑QUANT", n_rea)
c4.metric("Totale segnali scanner", n_tot)

st.caption(
    "Legenda generale: EARLY = vicinanza alla EMA20; PRO = trend consolidato con RSI e Vol_Ratio favorevoli; "
    "REA‑QUANT = pressione volumetrica vicino al POC."
)

# =============================================================================
# TABS
# =============================================================================
tab_e, tab_p, tab_rea, tab_serafini, tab_regime, tab_mtf, tab_finviz, tab_watch = st.tabs(
    [
        "🟢 EARLY",
        "🟣 PRO",
        "🟠 REA‑QUANT",
        "📈 Serafini Systems",
        "🧊 Regime & Momentum",
        "🕒 Multi‑Timeframe",
        "📊 Finviz",
        "📌 Watchlist & Note",
    ]
)

# =============================================================================
# EARLY – Top N per Early_Score / RSI / Vol_Ratio
# =============================================================================
with tab_e:
    st.subheader("🟢 Segnali EARLY")
    st.markdown(
        f"Filtro EARLY: titoli con **Stato = EARLY** "
        f"(distanza prezzo–EMA20 ≤ {e_h*100:.1f}%, punteggio Early_Score ≥ 8)."
    )

    with st.expander("📘 Legenda EARLY"):
        st.markdown(
            "- **Early_Score**: 8 se il prezzo è entro la soglia percentuale dalla EMA20, 0 altrimenti.\n"
            "- **RSI**: RSI a 14 periodi.\n"
            "- **Vol_Ratio**: volume odierno / media 20 giorni.\n"
            "- **Market Cap**: capitalizzazione abbreviata (K/M/B) con valuta.\n"
            "- **Vol_Today / Vol_7d_Avg**: volume odierno e media ultimi 7 giorni.\n"
            "- **Stato = EARLY**: setup in formazione vicino alla media.\n"
            "- Colonne **Yahoo** e **TradingView**: pulsanti link per ogni ticker."
        )

    if df_early_all.empty:
        st.caption("Nessun segnale EARLY.")
    else:
        df_early = df_early_all.copy()
        df_early = add_formatted_cols(df_early)
        df_early = add_links(df_early)

        base_cols = [
            "Nome",
            "Ticker",
            "Prezzo",
            "Prezzo_fmt",
            "MarketCap",
            "MarketCap_fmt",
            "Vol_Today",
            "Vol_Today_fmt",
            "Vol_7d_Avg",
            "Vol_7d_Avg_fmt",
            "Early_Score",
            "Pro_Score",
            "RSI",
            "Vol_Ratio",
            "OBV_Trend",
            "ATR",
            "ATR_Exp",
            "Stato",
            "Yahoo",
            "Finviz",
        ]
        df_early = df_early[[c for c in base_cols if c in df_early.columns]]

        sort_cols = [c for c in ["Early_Score", "RSI", "Vol_Ratio"] if c in df_early.columns]
        if sort_cols:
            df_early_view = df_early.sort_values(
                by=sort_cols,
                ascending=[False] * len(sort_cols),
            ).head(top)
        else:
            df_early_view = df_early.head(top)

        view_cols = [
            "Nome",
            "Ticker",
            "Prezzo_fmt",
            "MarketCap_fmt",
            "Vol_Today_fmt",
            "Vol_7d_Avg_fmt",
            "Early_Score",
            "Pro_Score",
            "RSI",
            "Vol_Ratio",
            "OBV_Trend",
            "ATR",
            "ATR_Exp",
            "Stato",
            "Yahoo",
            "Finviz",
        ]
        df_early_show = df_early_view[[c for c in view_cols if c in df_early_view.columns]]

        st.dataframe(
            df_early_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
            },
        )

        # -------- Export su una riga --------
        csv_data = df_early_view.to_csv(index=False).encode("utf-8")
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df_early_view.to_excel(writer, index=False, sheet_name="EARLY")
        data_xlsx = out.getvalue()
        tv_data = df_early_view["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

        col_csv, col_xlsx, col_tv = st.columns(3)
        with col_csv:
            st.download_button(
                "⬇️ Export EARLY CSV",
                data=csv_data,
                file_name=f"EARLY_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_early_csv",
            )
        with col_xlsx:
            st.download_button(
                "⬇️ Export EARLY XLSX",
                data=data_xlsx,
                file_name=f"EARLY_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_early_xlsx",
            )
        with col_tv:
            st.download_button(
                "⬇️ Export EARLY TradingView (solo ticker)",
                data=csv_tv,
                file_name=f"TV_EARLY_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_tv_early",
            )

        # -------- Watchlist --------
        options_early = sorted(
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_early_view.iterrows()
        )

        col_sel_all_early, _ = st.columns([1, 3])
        with col_sel_all_early:
            if st.button("✅ Seleziona tutti (Top N EARLY)", key="btn_sel_all_early"):
                st.session_state["wl_early"] = options_early

        selection_early = st.multiselect(
            "Aggiungi alla Watchlist (EARLY):",
            options=options_early,
            key="wl_early",
        )

        note_early = st.text_input(
            "Note comuni per questi ticker EARLY", key="note_wl_early"
        )

        if st.button("📌 Salva in Watchlist (EARLY)"):
            tickers = [s.split(" – ")[1] for s in selection_early]
            names = [s.split(" – ")[0] for s in selection_early]
            add_to_watchlist(
                tickers,
                names,
                "EARLY",
                note_early,
                trend="LONG",
                list_name=st.session_state.get("current_list_name", "DEFAULT"),
            )
            st.success("EARLY salvati in watchlist.")
            st.rerun()


# =============================================================================
# PRO – Top N per Pro_Score / RSI / Vol_Ratio
# =============================================================================
with tab_p:
    st.subheader("🟣 Segnali PRO")
    st.markdown(
        f"Filtro PRO: titoli con **Stato = PRO** "
        f"(prezzo sopra EMA20, RSI tra {p_rmin} e {p_rmax}, Vol_Ratio > 1.2, Pro_Score elevato)."
    )

    with st.expander("📘 Legenda PRO"):
        st.markdown(
            "- **Pro_Score**: punteggio composito (prezzo sopra EMA20, RSI nel range, volume sopra media).\n"
            "- **Market Cap**: capitalizzazione abbreviata (K/M/B) con valuta.\n"
            "- **Vol_Today / Vol_7d_Avg**: volume odierno e media 7 giorni.\n"
            "- **OBV_Trend**: UP/DOWN in base alla pendenza media OBV 5 periodi.\n"
            "- **Stato = PRO**: trend avanzato con conferme.\n"
            "- Colonne **Yahoo** e **TradingView**: pulsanti link per ogni ticker."
        )

    if df_pro_all.empty:
        st.caption("Nessun segnale PRO.")
    else:
        df_pro = df_pro_all.copy()
        df_pro = add_formatted_cols(df_pro)
        df_pro = add_links(df_pro)

        base_cols = [
            "Nome",
            "Ticker",
            "Prezzo",
            "Prezzo_fmt",
            "MarketCap",
            "MarketCap_fmt",
            "Vol_Today",
            "Vol_Today_fmt",
            "Vol_7d_Avg",
            "Vol_7d_Avg_fmt",
            "Early_Score",
            "Pro_Score",
            "RSI",
            "Vol_Ratio",
            "OBV_Trend",
            "ATR",
            "ATR_Exp",
            "Stato",
            "Yahoo",
            "Finviz",
        ]
        df_pro = df_pro[[c for c in base_cols if c in df_pro.columns]]

        sort_cols = [c for c in ["Pro_Score", "RSI", "Vol_Ratio"] if c in df_pro.columns]
        if sort_cols:
            df_pro_view = df_pro.sort_values(
                by=sort_cols,
                ascending=[False] * len(sort_cols),
            ).head(top)
        else:
            df_pro_view = df_pro.head(top)

        view_cols = [
            "Nome",
            "Ticker",
            "Prezzo_fmt",
            "MarketCap_fmt",
            "Vol_Today_fmt",
            "Vol_7d_Avg_fmt",
            "Early_Score",
            "Pro_Score",
            "RSI",
            "Vol_Ratio",
            "OBV_Trend",
            "ATR",
            "ATR_Exp",
            "Stato",
            "Yahoo",
            "Finviz",
        ]
        df_pro_show = df_pro_view[[c for c in view_cols if c in df_pro_view.columns]]

        st.dataframe(
            df_pro_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
            },
        )

        # Export su una riga
        csv_data = df_pro_view.to_csv(index=False).encode("utf-8")
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df_pro_view.to_excel(writer, index=False, sheet_name="PRO")
        data_xlsx = out.getvalue()
        tv_data = df_pro_view["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

        col_csv, col_xlsx, col_tv = st.columns(3)
        with col_csv:
            st.download_button(
                "⬇️ Export PRO CSV",
                data=csv_data,
                file_name=f"PRO_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_pro_csv",
            )
        with col_xlsx:
            st.download_button(
                "⬇️ Export PRO XLSX",
                data=data_xlsx,
                file_name=f"PRO_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_pro_xlsx",
            )
        with col_tv:
            st.download_button(
                "⬇️ Export PRO TradingView (solo ticker)",
                data=csv_tv,
                file_name=f"TV_PRO_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_tv_pro",
            )

        # Watchlist PRO
        options_pro = sorted(
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_pro_view.iterrows()
        )

        col_sel_all_pro, _ = st.columns([1, 3])
        with col_sel_all_pro:
            if st.button("✅ Seleziona tutti (Top N PRO)", key="btn_sel_all_pro"):
                st.session_state["wl_pro"] = options_pro

        selection_pro = st.multiselect(
            "Aggiungi alla Watchlist (PRO):",
            options=options_pro,
            key="wl_pro",
        )

        note_pro = st.text_input(
            "Note comuni per questi ticker PRO", key="note_wl_pro"
        )

        if st.button("📌 Salva in Watchlist (PRO)"):
            tickers = [s.split(" – ")[1] for s in selection_pro]
            names = [s.split(" – ")[0] for s in selection_pro]
            add_to_watchlist(
                tickers,
                names,
                "PRO",
                note_pro,
                trend="LONG",
                list_name=st.session_state.get("current_list_name", "DEFAULT"),
            )
            st.success("PRO salvati in watchlist.")
            st.rerun()


# =============================================================================
# REA‑QUANT / Rea Quant – Ranking unico HOT volumetrico
# =============================================================================
with tab_rea:
    st.subheader("🟠 REA‑QUANT / Rea Quant")
    st.markdown(
        f"Ranking unico in stile **Massimo Rea**: titoli con forte pressione volumetrica "
        f"(Vol_Ratio elevato) e prezzo **vicino al POC** (≤ {r_poc*100:.1f}%), "
        f"ordinati per combinazione di volume, distanza POC e qualità tecnica (RSI / Pro_Score)."
    )

    with st.expander("📘 Legenda REA‑QUANT / Rea Quant"):
        st.markdown(
            "- **Vol_Ratio**: volume odierno / media 20 giorni.\n"
            "- **Dist_POC_%**: distanza percentuale prezzo–POC (più è bassa, meglio è).\n"
            "- **Rea_Score**: ranking combinato che privilegia Vol_Ratio alto, "
            "distanza POC bassa e buoni valori di Pro_Score / RSI.\n"
            "- **HOT**: titoli che soddisfano i requisiti volumetrici vicino al POC.\n"
            "- Colonne **Yahoo** e **TradingView**: link rapidi per l’analisi grafica."
        )

    if df_rea_all.empty:
        st.caption("Nessun segnale REA‑QUANT disponibile.")
    else:
        df_rea = df_rea_all.copy()
        df_rea = add_formatted_cols(df_rea)
        df_rea = add_links(df_rea)

                # Assicuro la presenza della distanza POC in formato %/decimale
        if "Distanza_POC" not in df_rea.columns and "Dist_POC_%" in df_rea.columns:
            df_rea["Distanza_POC"] = df_rea["Dist_POC_%"]


        # Filtra solo i veri HOT in stile Rea:
        # vicino al POC e con Vol_Ratio sopra la soglia scelta in sidebar
        df_rea = df_rea[
            (df_rea["Vol_Ratio"] >= vol_ratio_hot)
            & (df_rea["Distanza_POC"] <= r_poc * 100)
        ].copy()

        if df_rea.empty:
            st.caption("Nessun titolo soddisfa i criteri HOT (Vol_Ratio e distanza POC).")
        else:
            # ---------------------------
            # Rea_Score: ranking combinato
            # ---------------------------
            # 1) Volumi relativi (più alto è meglio)
            rank_vol = df_rea["Vol_Ratio"].rank(ascending=False)

            # 2) distanza POC (più vicino è meglio)
            rank_poc = df_rea["Distanza_POC"].rank(ascending=True)

            # 3) componente tecnica: Pro_Score e RSI (se presenti)
            if "Pro_Score" in df_rea.columns:
                rank_pro = df_rea["Pro_Score"].rank(ascending=False)
            else:
                rank_pro = 0

            if "RSI" in df_rea.columns:
                # privilegio RSI medi (evito estremi overbought/oversold)
                rsi_mid = (df_rea["RSI"] - 50).abs()
                rank_rsi = rsi_mid.rank(ascending=True)
            else:
                rank_rsi = 0

            # combinazione pesata (puoi regolare i pesi se vuoi)
            df_rea["Rea_Score"] = (
                rank_vol * 0.5
                + rank_poc * 0.3
                + rank_pro * 0.1
                + rank_rsi * 0.1
            )

            # Top N in stile Rea: punteggi più bassi = combinazione migliore
            df_rea_view = df_rea.sort_values("Rea_Score", ascending=True).head(top)

            # View tabellare – Nome e Ticker davanti
            cols_show = [
                "Nome",
                "Ticker",
                "Prezzo_fmt",
                "MarketCap_fmt",
                "Vol_Today_fmt",
                "Vol_7d_Avg_fmt",
                "Vol_Ratio",
                "Distanza_POC",
                "Pro_Score",
                "RSI",
                "Rea_Score",
                "OBV_Trend",
                "ATR",
                "ATR_Exp",
                "Stato",
                "Yahoo",
                "Finviz",
            ]
            df_rea_show = df_rea_view[[c for c in cols_show if c in df_rea_view.columns]]

            st.dataframe(
                df_rea_show,
                use_container_width=True,
                column_config={
                    "Prezzo_fmt": "Prezzo",
                    "MarketCap_fmt": "Market Cap",
                    "Vol_Today_fmt": "Vol giorno",
                    "Vol_7d_Avg_fmt": "Vol medio 7g",
                    "Distanza_POC": "Dist POC (%)",
                    "Rea_Score": "Rea_Rank",
                    "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                    "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
                },
            )

            # ======================
            # Export CSV/XLSX + TV
            # ======================
            csv_data = df_rea_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export REA‑QUANT / Rea Quant CSV",
                data=csv_data,
                file_name=f"REA_COMBINED_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_rea_combined_csv",
            )

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_rea_view.to_excel(writer, index=False, sheet_name="REA_COMBINED")
            data_xlsx = output.getvalue()

            st.download_button(
                "⬇️ Export REA‑QUANT / Rea Quant XLSX",
                data=data_xlsx,
                file_name=f"REA_COMBINED_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_rea_combined_xlsx",
            )

            tv_data = df_rea_view["Ticker"].drop_duplicates().to_frame(name="symbol")
            csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")
            st.download_button(
                "⬇️ Export REA‑QUANT / Rea Quant TradingView (solo ticker)",
                data=csv_tv,
                file_name=f"TV_REA_COMBINED_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_rea_combined_tv",
            )

            # ======================
            # Watchlist in stile Rea
            # ======================
            options_rea = sorted(
                f"{row['Nome']} – {row['Ticker']}" for _, row in df_rea_view.iterrows()
            )

            col_sel_all_rea, _ = st.columns([1, 3])
            with col_sel_all_rea:
                if st.button("✅ Seleziona tutti (Top N REA‑QUANT / Rea Quant)", key="btn_sel_all_rea_combined"):
                    st.session_state["wl_rea_combined"] = options_rea

            selection_rea = st.multiselect(
                "Aggiungi alla Watchlist (REA‑QUANT / Rea Quant):",
                options=options_rea,
                key="wl_rea_combined",
            )

            note_rea = st.text_input(
                "Note comuni per questi ticker REA‑QUANT / Rea Quant", key="note_wl_rea_combined"
            )

            if st.button("📌 Salva in Watchlist (REA‑QUANT / Rea Quant)"):
                tickers = [s.split(" – ")[1] for s in selection_rea]
                names = [s.split(" – ")[0] for s in selection_rea]
                add_to_watchlist(
                    tickers,
                    names,
                    "REA_COMBINED",
                    note_rea,
                    trend="LONG",
                    list_name=st.session_state.get("current_list_name", "DEFAULT"),
                )
                st.success("REA‑QUANT / Rea Quant salvati in watchlist.")
                st.rerun()

# =============================================================================
# SERAFINI – Top N per Pro_Score su breakout UP
# =============================================================================
with tab_serafini:
    st.subheader("📈 Approccio Trend‑Following stile Stefano Serafini")
    st.markdown(
        "Sistema Donchian‑style su 20 giorni: breakout su massimi/minimi 20 giorni "
        "calcolato su tutti i ticker scansionati."
    )

    with st.expander("📘 Legenda Serafini Systems"):
        st.markdown(
            "- **Hi20 / Lo20**: massimo/minimo a 20 giorni.\n"
            "- **Breakout_Up/Down**: rottura massimi/minimi.\n"
            "- **Market Cap / Volumi**: info di contesto.\n"
            "- Ordinamento per Pro_Score per privilegiare i breakout in trend forti.\n"
            "- Colonne **Yahoo** e **TradingView**: pulsanti link per ogni ticker."
        )

    if df_ep.empty:
        st.caption("Nessun dato scanner disponibile.")
    else:
        universe = df_ep["Ticker"].unique().tolist()
        records = []

        for tkr in universe:
            try:
                data = yf.Ticker(tkr).history(period="3mo")
                if len(data) < 20:
                    continue

                close = data["Close"]
                high20 = close.rolling(20).max()
                low20 = close.rolling(20).min()
                last = close.iloc[-1]

                breakout_up = last >= high20.iloc[-2]
                breakout_down = last <= low20.iloc[-2]

                records.append(
                    {
                        "Ticker": tkr,
                        "Prezzo": round(last, 2),
                        "Hi20": round(high20.iloc[-2], 2),
                        "Lo20": round(low20.iloc[-2], 2),
                        "Breakout_Up": breakout_up,
                        "Breakout_Down": breakout_down,
                    }
                )
            except Exception:
                continue

        df_break = pd.DataFrame(records)

        if df_break.empty:
            st.caption("Nessun breakout rilevato.")
        else:
            df_seraf = df_break.merge(
                df_ep[
                    [
                        "Ticker",
                        "Nome",
                        "MarketCap",
                        "Vol_Today",
                        "Vol_7d_Avg",
                        "Pro_Score",
                        "RSI",
                        "Vol_Ratio",
                        "OBV_Trend",
                    ]
                ],
                on="Ticker",
                how="left",
            )

            df_seraf = add_formatted_cols(df_seraf)
            df_seraf = add_links(df_seraf)

            df_seraf = df_seraf[df_seraf["Breakout_Up"]].copy()
            df_seraf_view = df_seraf.sort_values("Pro_Score", ascending=False).head(top)

            if df_seraf_view.empty:
                st.caption("Nessun breakout LONG con criteri Serafini.")
            else:
                df_seraf_show = df_seraf_view[
                    [
                        "Nome",
                        "Ticker",
                        "Prezzo_fmt",
                        "MarketCap_fmt",
                        "Vol_Today_fmt",
                        "Vol_7d_Avg_fmt",
                        "Hi20",
                        "Lo20",
                        "Pro_Score",
                        "RSI",
                        "Vol_Ratio",
                        "OBV_Trend",
                        "Yahoo",
                        "Finviz",
                    ]
                ]

                st.dataframe(
                    df_seraf_show,
                    use_container_width=True,
                    column_config={
                        "Prezzo_fmt": "Prezzo",
                        "MarketCap_fmt": "Market Cap",
                        "Vol_Today_fmt": "Vol giorno",
                        "Vol_7d_Avg_fmt": "Vol medio 7g",
                        "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                        "Finviz": st.column_config.LinkColumn(
                            "TradingView", display_text="Apri"
                        ),
                    },
                )

                # EXPORT SERAFINI
                csv_data = df_seraf_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Export Serafini CSV",
                    data=csv_data,
                    file_name=f"SERAFINI_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_seraf_csv",
                )

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df_seraf_view.to_excel(writer, index=False, sheet_name="SERAFINI")
                data_xlsx = output.getvalue()

                st.download_button(
                    "⬇️ Export Serafini XLSX",
                    data=data_xlsx,
                    file_name=f"SERAFINI_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                    key="dl_seraf_xlsx",
                )

                options_seraf = sorted(
                    f"{row['Nome']} – {row['Ticker']}"
                    for _, row in df_seraf_view.iterrows()
                )

                col_sel_all_seraf, _ = st.columns([1, 3])
                with col_sel_all_seraf:
                    if st.button(
                        "✅ Seleziona tutti (Top Serafini)", key="btn_sel_all_seraf"
                    ):
                        st.session_state["wl_seraf"] = options_seraf

                selection_seraf = st.multiselect(
                    "Aggiungi alla Watchlist (Serafini):",
                    options=options_seraf,
                    key="wl_seraf",
                )

                note_seraf = st.text_input(
                    "Note comuni per questi ticker Serafini", key="note_wl_seraf"
                )

                if st.button("📌 Salva in Watchlist (Serafini)"):
                    tickers = [s.split(" – ")[1] for s in selection_seraf]
                    names = [s.split(" – ")[0] for s in selection_seraf]
                    add_to_watchlist(
                        tickers,
                        names,
                        "SERAFINI",
                        note_seraf,
                        trend="LONG",
                        list_name=st.session_state.get("current_list_name", "DEFAULT"),
                    )
                    st.success("Serafini salvati in watchlist.")
                    st.rerun()

# =============================================================================
# REGIME & MOMENTUM
# =============================================================================
with tab_regime:
    st.subheader("🧊 Regime & Momentum")

    if df_ep.empty:
        st.caption("Nessun dato scanner disponibile per il Regime & Momentum.")
    else:
        df_mom = df_ep.copy()
        df_mom = add_formatted_cols(df_mom)
        df_mom = add_links(df_mom)

        # Momentum combinato (slider momentum_min in sidebar)
        df_mom["Momentum"] = df_mom["Pro_Score"] * 10 + df_mom["RSI"]
        df_mom = df_mom[df_mom["Momentum"] >= momentum_min]

        if df_mom.empty:
            st.caption(
                "Nessun titolo con Momentum sufficiente per i criteri impostati."
            )
        else:
            df_mom = df_mom.sort_values("Momentum", ascending=False)
            df_mom_view = df_mom.head(top)

            cols_show = [
                "Nome",
                "Ticker",
                "Prezzo_fmt",
                "MarketCap_fmt",
                "Vol_Today_fmt",
                "Vol_7d_Avg_fmt",
                "Early_Score",
                "Pro_Score",
                "RSI",
                "Vol_Ratio",
                "Momentum",
                "OBV_Trend",
                "ATR",
                "ATR_Exp",
                "Stato",
                "Yahoo",
                "Finviz",
            ]
            cols_show = [c for c in cols_show if c in df_mom_view.columns]

            st.dataframe(
                df_mom_view[cols_show],
                use_container_width=True,
                column_config={
                    "Prezzo_fmt": "Prezzo",
                    "MarketCap_fmt": "Market Cap",
                    "Vol_Today_fmt": "Vol giorno",
                    "Vol_7d_Avg_fmt": "Vol medio 7g",
                    "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                    "Finviz": st.column_config.LinkColumn(
                        "TradingView", display_text="Apri"
                    ),
                },
            )

            # ---------- Export in una riga ----------
            csv_data = df_mom_view.to_csv(index=False).encode("utf-8")

            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                df_mom_view.to_excel(writer, index=False, sheet_name="MOMENTUM")
            data_xlsx = out.getvalue()

            tv_data = (
                df_mom_view["Ticker"]
                .drop_duplicates()
                .to_frame(name="symbol")
            )
            csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

            col_csv, col_xlsx, col_tv = st.columns(3)
            with col_csv:
                st.download_button(
                    "⬇️ Export Momentum CSV",
                    data=csv_data,
                    file_name=f"MOMENTUM_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_mom_csv",
                )
            with col_xlsx:
                st.download_button(
                    "⬇️ Export Momentum XLSX",
                    data=data_xlsx,
                    file_name=f"MOMENTUM_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                    key="dl_mom_xlsx",
                )
            with col_tv:
                st.download_button(
                    "⬇️ CSV Top Momentum (solo ticker)",
                    data=csv_tv,
                    file_name=f"TV_MOMENTUM_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_mom_tv",
                )

            # ---------- Blocchi Watchlist (come in EARLY/PRO) ----------
            options_mom = sorted(
                f"{row['Nome']} – {row['Ticker']}" for _, row in df_mom_view.iterrows()
            )

            col_sel_all_mom, _ = st.columns([1, 3])
            with col_sel_all_mom:
                if st.button("✅ Seleziona tutti (Top N Momentum)", key="btn_sel_all_mom"):
                    st.session_state["wl_mom"] = options_mom

            selection_mom = st.multiselect(
                "Aggiungi alla Watchlist (Momentum):",
                options=options_mom,
                key="wl_mom",
            )

            note_mom = st.text_input(
                "Note comuni per questi ticker Momentum", key="note_wl_mom"
            )

            if st.button("📌 Salva in Watchlist (Momentum)"):
                tickers = [s.split(" – ")[1] for s in selection_mom]
                names = [s.split(" – ")[0] for s in selection_mom]
                add_to_watchlist(
                    tickers,
                    names,
                    "MOMENTUM",
                    note_mom,
                    trend="LONG",
                    list_name=st.session_state.get("current_list_name", "DEFAULT"),
                )
                st.success("Ticker Momentum salvati in watchlist.")
                st.rerun()

            # ---------- Sintesi per mercato in expander ----------
            if "Market" in df_mom.columns:
                df_mom_summary = (
                    df_mom.groupby("Market")
                    .agg(
                        Momentum_mean=("Momentum", "mean"),
                        N=("Ticker", "nunique"),
                        MktCap_mean=("MarketCap", "mean"),
                        Vol_mean=("Vol_Today", "mean"),
                    )
                    .reset_index()
                )

                df_mom_summary["MktCap_mean_fmt"] = df_mom_summary["MktCap_mean"].apply(
                    fmt_marketcap
                )
                df_mom_summary["Vol_mean_fmt"] = df_mom_summary["Vol_mean"].apply(
                    fmt_int
                )

                with st.expander(
                    "📊 Sintesi Regime & Momentum per mercato (tabella)", expanded=False
                ):
                    st.dataframe(
                        df_mom_summary[
                            [
                                "Market",
                                "Momentum_mean",
                                "N",
                                "MktCap_mean_fmt",
                                "Vol_mean_fmt",
                            ]
                        ],
                        use_container_width=True,
                        column_config={
                            "Market": "Mercato",
                            "Momentum_mean": "Momentum medio",
                            "N": "N titoli",
                            "MktCap_mean_fmt": "Market Cap med",
                            "Vol_mean_fmt": "Vol medio giorno",
                        },
                    )


# =============================================================================
# MULTI‑TIMEFRAME – Top N per UP_count / Momentum_W / Momentum_M
# =============================================================================
with tab_mtf:
    st.subheader("⏱️ Multi‑Timeframe (D / W / M)")
    st.markdown(
        "Vista congiunta **daily / weekly / monthly** sugli stessi titoli, "
        "per verificare allineamento di trend e momentum."
    )

    with st.expander("📘 Legenda Multi‑Timeframe"):
        st.markdown(
            "- Timeframe: **D** (giornaliero), **W** (settimanale), **M** (mensile).\n"
            "- **Trend_TF**: direzione trend per timeframe (UP / DOWN / SIDE).\n"
            "- **RSI_TF**: RSI calcolato sul timeframe.\n"
            "- **Momentum_TF**: punteggio sintetico (tipo Pro_Score×10 + RSI) per timeframe.\n"
            "- Filtriamo solo i titoli presenti nello scanner principale."
        )

    if df_ep.empty:
        st.caption("Nessun dato scanner disponibile.")
    else:
        universe = df_ep["Ticker"].unique().tolist()

        @st.cache_data(show_spinner=False)
        def fetch_mt_data(tickers):
            records = []
            for tkr in tickers:
                try:
                    df_d = yf.Ticker(tkr).history(period="6mo", interval="1d")
                    df_w = yf.Ticker(tkr).history(period="2y", interval="1wk")
                    df_m = yf.Ticker(tkr).history(period="5y", interval="1mo")

                    if len(df_d) < 30 or len(df_w) < 30 or len(df_m) < 30:
                        continue

                    def mk_row(df, tf_label):
                        close = df["Close"]
                        rsi = ta.momentum.rsi(close, window=14).iloc[-1]

                        ma_fast = close.rolling(20).mean().iloc[-1]
                        ma_slow = close.rolling(50).mean().iloc[-1]
                        if ma_fast > ma_slow:
                            trend = "UP"
                        elif ma_fast < ma_slow:
                            trend = "DOWN"
                        else:
                            trend = "SIDE"

                        last = close.iloc[-1]
                        momentum_tf = (
                            (1 if trend == "UP" else -1 if trend == "DOWN" else 0) * 10
                            + rsi
                        )

                        return {
                            "Ticker": tkr,
                            "TF": tf_label,
                            f"Close_{tf_label}": round(last, 2),
                            f"RSI_{tf_label}": round(rsi, 1),
                            f"Trend_{tf_label}": trend,
                            f"Momentum_{tf_label}": round(momentum_tf, 1),
                        }

                    row_d = mk_row(df_d, "D")
                    row_w = mk_row(df_w, "W")
                    row_m = mk_row(df_m, "M")

                    merged = {"Ticker": tkr}
                    merged.update(row_d)
                    merged.update(row_w)
                    merged.update(row_m)
                    records.append(merged)
                except Exception:
                    continue

            return pd.DataFrame(records)

        with st.spinner("Calcolo multi‑timeframe..."):
            df_mt = fetch_mt_data(universe)

        if df_mt.empty:
            st.caption("Nessun dato multi‑timeframe disponibile.")
        else:
            base_cols = [
                "Ticker",
                "Nome",
                "Prezzo",
                "MarketCap",
                "Pro_Score",
                "RSI",
                "Vol_Today",
                "Vol_7d_Avg",
                "Stato",
            ]
            df_base = df_ep[base_cols].drop_duplicates(subset=["Ticker"])
            df_mt_full = df_mt.merge(df_base, on="Ticker", how="left")

            df_mt_full["UP_count"] = (
                (df_mt_full["Trend_D"] == "UP").astype(int)
                + (df_mt_full["Trend_W"] == "UP").astype(int)
                + (df_mt_full["Trend_M"] == "UP").astype(int)
            )
            tf_min_up = st.slider(
                "Minimo n° timeframe in UP",
                min_value=0,
                max_value=3,
                value=2,
                step=1,
            )
            df_mt_full = df_mt_full[df_mt_full["UP_count"] >= tf_min_up]

            if df_mt_full.empty:
                st.caption("Nessun titolo con i criteri selezionati.")
            else:
                df_mt_full = add_formatted_cols(df_mt_full)
                df_mt_full = add_links(df_mt_full)

                cols_show = [
                    "Nome",
                    "Ticker",
                    "Prezzo_fmt",
                    "MarketCap_fmt",
                    "Pro_Score",
                    "RSI",
                    "Stato",
                    "Close_D",
                    "RSI_D",
                    "Trend_D",
                    "Momentum_D",
                    "Close_W",
                    "RSI_W",
                    "Trend_W",
                    "Momentum_W",
                    "Close_M",
                    "RSI_M",
                    "Trend_M",
                    "Momentum_M",
                    "Yahoo",
                    "Finviz",
                ]
                cols_show = [c for c in cols_show if c in df_mt_full.columns]

                df_mt_view = df_mt_full.sort_values(
                    ["UP_count", "Momentum_W", "Momentum_M"],
                    ascending=[False, False, False],
                ).head(top)

                st.dataframe(
                    df_mt_view[cols_show],
                    use_container_width=True,
                    column_config={
                        "Prezzo_fmt": "Prezzo",
                        "MarketCap_fmt": "Market Cap",
                        "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                        "Finviz": st.column_config.LinkColumn(
                            "TradingView", display_text="Apri"
                        ),
                    },
                )

                csv_data = df_mt_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Export Multi‑Timeframe CSV",
                    data=csv_data,
                    file_name=f"MULTITF_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_mt_csv",
                )

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df_mt_view.to_excel(writer, index=False, sheet_name="MULTI_TF")
                data_xlsx = output.getvalue()
                st.download_button(
                    "⬇️ Export Multi‑Timeframe XLSX",
                    data=data_xlsx,
                    file_name=f"MULTITF_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                    key="dl_mt_xlsx",
                )

                df_mt_tv = df_mt_view[["Ticker"]].rename(columns={"Ticker": "symbol"})
                csv_mt_tv = df_mt_tv.to_csv(index=False, header=False).encode("utf-8")
                st.download_button(
                    "⬇️ CSV Multi‑TF (solo ticker)",
                    data=csv_mt_tv,
                    file_name=(
                        f"signals_multitf_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_tv_mt",
                )

                options_mt = sorted(
                    f"{row['Nome']} – {row['Ticker']}"
                    for _, row in df_mt_view.iterrows()
                )

                col_sel_all_mt, _ = st.columns([1, 3])
                with col_sel_all_mt:
                    if st.button(
                        "✅ Seleziona tutti (Top Multi‑TF)", key="btn_sel_all_mt"
                    ):
                        st.session_state["wl_multitf"] = options_mt

                selection_mt = st.multiselect(
                    "Aggiungi alla Watchlist (Multi‑TF):",
                    options=options_mt,
                    key="wl_multitf",
                )
                note_mt = st.text_input(
                    "Note comuni per questi ticker Multi‑TF", key="note_wl_multitf"
                )
                if st.button("📌 Salva in Watchlist (Multi‑TF)"):
                    tickers = [s.split(" – ")[1] for s in selection_mt]
                    names = [s.split(" – ")[0] for s in selection_mt]
                    add_to_watchlist(
                        tickers,
                        names,
                        "MULTI_TF",
                        note_mt,
                        trend="LONG",
                        list_name=st.session_state.get("current_list_name", "DEFAULT"),
                    )
                    st.success("Multi‑TF salvati in watchlist.")
                    st.rerun()

# =============================================================================
# FINVIZ-LIKE
# =============================================================================
with tab_finviz:
    st.subheader("📊 Finviz‑like")

    if df_ep.empty:
        st.caption("Nessun dato scanner disponibile per il filtro Finviz‑like.")
    else:
        dffinviz = df_ep.copy()

        if all(
            col in dffinviz.columns
            for col in ["EPSnextY", "EPS5Y", "AvgVolmln", "Prezzo"]
        ):
            # rimuovo righe con NaN sulle colonne fondamentali
            dffinviz = dffinviz.dropna(
                subset=["EPSnextY", "EPS5Y", "AvgVolmln", "Prezzo"]
            )

            # filtri stile Finviz
            dffinviz = dffinviz[
                (dffinviz["EPSnextY"] >= eps_next_y_min)
                & (dffinviz["EPS5Y"] >= eps_next_5y_min)
                & (dffinviz["AvgVolmln"] >= avg_vol_min_mln)
                & (dffinviz["Prezzo"] >= price_min_finviz)
            ]

            if dffinviz.empty:
                st.caption("Nessun titolo soddisfa i filtri fondamentali Finviz‑like.")
            else:
                dffinviz = add_formatted_cols(dffinviz)
                dffinviz = add_links(dffinviz)

                cols_order = [
                    "Nome",
                    "Ticker",
                    "Prezzo",
                    "Prezzo_fmt",
                    "MarketCap",
                    "MarketCap_fmt",
                    "Vol_Today",
                    "Vol_Today_fmt",
                    "Vol_7d_Avg",
                    "Vol_7d_Avg_fmt",
                    "EPSnextY",
                    "EPS5Y",
                    "AvgVolmln",
                    "Early_Score",
                    "Pro_Score",
                    "RSI",
                    "Vol_Ratio",
                    "Stato",
                    "Yahoo",
                    "Finviz",
                ]
                dffinviz = dffinviz[[c for c in cols_order if c in dffinviz.columns]]

                # ordino per MarketCap decrescente
                if "MarketCap" in dffinviz.columns:
                    dffinviz_view = dffinviz.sort_values(
                        "MarketCap", ascending=False
                    ).head(top)
                else:
                    dffinviz_view = dffinviz.head(top)

                dffinviz_show = dffinviz_view[
                    [
                        "Nome",
                        "Ticker",
                        "Prezzo_fmt",
                        "MarketCap_fmt",
                        "Vol_Today_fmt",
                        "Vol_7d_Avg_fmt",
                        "EPSnextY",
                        "EPS5Y",
                        "AvgVolmln",
                        "Early_Score",
                        "Pro_Score",
                        "RSI",
                        "Vol_Ratio",
                        "Stato",
                        "Yahoo",
                        "Finviz",
                    ]
                ]

                st.dataframe(
                    dffinviz_show,
                    use_container_width=True,
                    column_config={
                        "Prezzo_fmt": "Prezzo",
                        "MarketCap_fmt": "Market Cap",
                        "Vol_Today_fmt": "Vol giorno",
                        "Vol_7d_Avg_fmt": "Vol medio 7g",
                        "Yahoo": st.column_config.LinkColumn(
                            "Yahoo", display_text="Apri"
                        ),
                        "Finviz": st.column_config.LinkColumn(
                            "TradingView", display_text="Apri"
                        ),
                    },
                )

                # ---------- Export in una riga ----------
                csv_data = dffinviz_view.to_csv(index=False).encode("utf-8")

                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                    dffinviz_view.to_excel(writer, index=False, sheet_name="FINVIZ")
                data_xlsx = out.getvalue()

                tv_data = (
                    dffinviz_view["Ticker"]
                    .drop_duplicates()
                    .to_frame(name="symbol")
                )
                csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

                col_csv, col_xlsx, col_tv = st.columns(3)
                with col_csv:
                    st.download_button(
                        "⬇️ Export Finviz‑like CSV",
                        data=csv_data,
                        file_name=f"FINVIZ_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_finviz_csv",
                    )
                with col_xlsx:
                    st.download_button(
                        "⬇️ Export Finviz‑like XLSX",
                        data=data_xlsx,
                        file_name=f"FINVIZ_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        use_container_width=True,
                        key="dl_finviz_xlsx",
                    )
                with col_tv:
                    st.download_button(
                        "⬇️ Export Finviz‑like TradingView (solo ticker)",
                        data=csv_tv,
                        file_name=f"TV_FINVIZ_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_finviz_tv",
                    )

        else:
            st.caption(
                "Mancano alcune colonne fondamentali (EPSnextY, EPS5Y, AvgVolmln, Prezzo) per il filtro Finviz‑like."
            )


# =============================================================================
# WATCHLIST & NOTE
# =============================================================================
with tab_watch:
    st.subheader("📌 Watchlist & Note")

    df_wl = load_watchlist()

    # filtro per lista attiva
    df_wl_filt = df_wl[df_wl["list_name"] == active_list].copy()

    if df_wl_filt.empty:
        st.caption("La watchlist attiva è vuota.")
    else:
        # -------- merge con df_ep per aggiungere dati tecnici correnti --------
        if not df_ep.empty:
            cols_merge = [
                "Ticker",
                "Prezzo",
                "MarketCap",
                "Vol_Today",
                "Vol_7d_Avg",
                "Stato",
            ]
            df_merge = df_ep[cols_merge].drop_duplicates(subset=["Ticker"])
            df_merge.rename(columns={"Ticker": "ticker", "Stato": "Stato_scan"}, inplace=True)
            df_wl_filt = df_wl_filt.merge(df_merge, on="ticker", how="left")
        else:
            df_wl_filt["Prezzo"] = np.nan
            df_wl_filt["MarketCap"] = np.nan
            df_wl_filt["Vol_Today"] = np.nan
            df_wl_filt["Vol_7d_Avg"] = np.nan
            df_wl_filt["Stato_scan"] = ""

        # ordina e crea etichetta: Nome – Ticker – Origine
        df_wl_filt = df_wl_filt.sort_values(["name", "ticker"], na_position="last")

        df_wl_filt = add_formatted_cols(df_wl_filt)

        df_wl_filt["label"] = (
            df_wl_filt["name"].fillna("")
            + " – "
            + df_wl_filt["ticker"].fillna("")
            + " – "
            + df_wl_filt["origine"].fillna("")
        )

        # tabella principale
        df_show = df_wl_filt[
            [
                "id",
                "name",
                "ticker",
                "trend",
                "origine",
                "Prezzo_fmt",
                "MarketCap_fmt",
                "Vol_Today_fmt",
                "Vol_7d_Avg_fmt",
                "Stato_scan",
                "note",
                "list_name",
                "created_at",
            ]
        ].rename(
            columns={
                "name": "Nome",
                "ticker": "Ticker",
                "trend": "Trend",
                "origine": "Origine",
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Stato_scan": "Stato",
                "note": "Note",
                "list_name": "Lista",
                "created_at": "Creato il",
            }
        )

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # ==============================
        # Export watchlist corrente
        # ==============================
        csv_wl = df_wl_filt.to_csv(index=False).encode("utf-8")
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df_wl_filt.to_excel(writer, index=False, sheet_name=str(active_list)[:31])
        xlsx_wl = out.getvalue()
        wl_tv = (
            df_wl_filt[["ticker"]]
            .drop_duplicates()
            .rename(columns={"ticker": "symbol"})
        )
        csv_wl_tv = wl_tv.to_csv(index=False, header=False).encode("utf-8")

        col_csv, col_xlsx, col_tv = st.columns(3)
        with col_csv:
            st.download_button(
                "⬇️ Export Watchlist corrente CSV",
                data=csv_wl,
                file_name=f"WATCHLIST_{active_list}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_wl_csv",
            )
        with col_xlsx:
            st.download_button(
                "⬇️ Export Watchlist corrente XLSX",
                data=xlsx_wl,
                file_name=f"WATCHLIST_{active_list}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                key="dl_wl_xlsx",
            )
        with col_tv:
            st.download_button(
                "⬇️ CSV Watchlist corrente (solo ticker)",
                data=csv_wl_tv,
                file_name=f"WATCHLIST_{active_list}_TICKER_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_wl_tv",
            )

        st.markdown("---")

        # ==============================
        # Modifica nota per una riga
        # ==============================
        st.markdown("### 📝 Modifica nota per una riga")

        row_options = ["–"] + df_wl_filt["label"].tolist()
        sel_row = st.selectbox("Seleziona riga", options=row_options)

        if sel_row != "–":
            row = df_wl_filt.loc[df_wl_filt["label"] == sel_row].iloc[0]
            new_note = st.text_area(
                "Nuova nota",
                value=row["note"] or "",
                key=f"note_edit_{row['id']}",
            )
            if st.button("💾 Salva nota", key=f"btn_save_note_{row['id']}"):
                update_watchlist_note(row["id"], new_note)
                st.success("Nota aggiornata.")
                st.rerun()

        st.markdown("---")

        # ==============================
        # Sposta righe tra watchlist
        # ==============================
        st.markdown("### 📁 Sposta righe tra watchlist")

        rows_move = st.multiselect(
            "Seleziona righe da spostare",
            options=df_wl_filt["label"].tolist(),
            key="rows_move",
        )

        dest_list = st.selectbox(
            "Lista di destinazione",
            options=list_options,
            index=list_options.index(active_list)
            if active_list in list_options
            else 0,
            key="dest_list_move",
        )

        if st.button("📦 Sposta righe"):
            if not rows_move:
                st.warning("Seleziona almeno una riga da spostare.")
            else:
                ids_move = df_wl_filt.loc[
                    df_wl_filt["label"].isin(rows_move), "id"
                ].tolist()
                if ids_move:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.executemany(
                        "UPDATE watchlist SET list_name = ? WHERE id = ?",
                        [(dest_list, int(i)) for i in ids_move],
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"{len(ids_move)} righe spostate nella lista '{dest_list}'.")
                    st.rerun()

        st.markdown("---")

        # ==============================
        # Cancella righe dalla lista corrente
        # ==============================
        st.markdown("### 🗑️ Cancella righe dalla lista corrente")

        rows_del = st.multiselect(
            "Seleziona righe da cancellare",
            options=df_wl_filt["label"].tolist(),
            key="rows_del",
        )

        if st.button("❌ Cancella righe selezionate"):
            if not rows_del:
                st.warning("Seleziona almeno una riga da cancellare.")
            else:
                ids_del = df_wl_filt.loc[
                    df_wl_filt["label"].isin(rows_del), "id"
                ].tolist()
                delete_from_watchlist(ids_del)
                st.success(f"{len(ids_del)} righe cancellate dalla watchlist.")
                st.rerun()
        st.markdown("---")

        # ==============================
        # Gestione DB Watchlist
        # ==============================
        st.markdown("### 🧹 Gestione DB Watchlist")

        col_ref, col_reset_list, col_reset_all = st.columns(3)

        # 1) Refresh DB (riload da disco)
        with col_ref:
            if st.button("🔄 Refresh DB"):
                st.experimental_rerun()

        # 2) Reset di UNA watchlist (solo list_name scelto)
        with col_reset_list:
            wl_to_reset = st.selectbox(
                "Watchlist da resettare",
                options=list_options,
                index=list_options.index(active_list)
                if active_list in list_options
                else 0,
                key="wl_to_reset",
            )
            if st.button("⚠️ Reset watchlist selezionata"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute(
                    "DELETE FROM watchlist WHERE list_name = ?",
                    (wl_to_reset,),
                )
                conn.commit()
                conn.close()
                st.success(f"Watchlist '{wl_to_reset}' azzerata.")
                st.rerun()

        # 3) Reset DB completo (tutte le watchlist)
        with col_reset_all:
            if st.button("🔥 Reset DB completo"):
                reset_watchlist_db()
                st.success("DB watchlist completamente resettato.")
                st.rerun()



