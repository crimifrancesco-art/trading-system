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
    page_title="Trading Scanner – Versione PRO 9.0",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner – Versione PRO 9.0")

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
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AVGO",
            "NFLX",
            "ADBE",
            "COST",
            "PEP",
            "CSCO",
            "INTC",
            "AMD",
        ]

    if "Dow" in markets:
        t += [
            "AAPL",
            "MSFT",
            "JPM",
            "V",
            "UNH",
            "JNJ",
            "WMT",
            "PG",
            "HD",
            "DIS",
            "KO",
            "MCD",
            "BA",
            "CAT",
            "GS",
        ]

    if "Russell" in markets:
        t += ["IWM", "VTWO"]

    if "FTSE" in markets:
        t += [
            "UCG.MI",
            "ISP.MI",
            "ENEL.MI",
            "ENI.MI",
            "LDO.MI",
            "PRY.MI",
            "STM.MI",
            "TEN.MI",
            "A2A.MI",
            "AMP.MI",
        ]

    if "Eurostoxx" in markets:
        t += [
            "ASML.AS",
            "NESN.SW",
            "SAN.PA",
            "TTE.PA",
            "AIR.PA",
            "MC.PA",
            "OR.PA",
            "SU.PA",
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

    if st.button("🚀 AVVIA SCANNER PRO 9.0", type="primary", use_container_width=True):
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
    df_ep = df_ep.dropna(subset=["Pro_Score", "RSI", "Vol_Ratio"])
if not df_rea.empty:
    df_rea = df_rea.dropna(subset=["Vol_Ratio", "Distanza_POC"])

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
tab_e, tab_p, tab_r, tab_rea_q, tab_serafini, tab_regime, tab_mtf, tab_finviz, tab_watch = st.tabs(
    [
        "🟢 EARLY",
        "🟣 PRO",
        "🟠 REA‑QUANT",
        "🧮 Rea Quant",
        "📈 Serafini Systems",
        "🧊 Regime & Momentum",
        "🕒 Multi‑Timeframe",
        "📊 Finviz",
        "📌 Watchlist & Note",
    ]
)

# =============================================================================
# EARLY – Top N per Early_Score
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
        df_early = df_early[[c for c in cols_order if c in df_early.columns]]

        # Top N per Early_Score
        df_early_view = df_early.sort_values("Early_Score", ascending=False).head(top)

        df_early_show = df_early_view[
            [
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
        ]

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

        csv_data = df_early_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export EARLY CSV",
            data=csv_data,
            file_name=f"EARLY_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_early_csv",
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_early_view.to_excel(writer, index=False, sheet_name="EARLY")
        data_xlsx = output.getvalue()

        st.download_button(
            "⬇️ Export EARLY XLSX",
            data=data_xlsx,
            file_name=f"EARLY_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_early_xlsx",
        )

        tv_data = df_early_view["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")
        st.download_button(
            "⬇️ Export EARLY TradingView (solo ticker)",
            data=csv_tv,
            file_name=f"TV_EARLY_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_tv_early",
        )

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
# PRO – Top N per Pro_Score
# =============================================================================
with tab_p:
    st.subheader("🟣 Segnali PRO")
    st.markdown(
        f"Filtro PRO: titoli con **Stato = PRO** "
        f"(prezzo sopra EMA20, RSI tra {p_rmin} e {p_rmax}, "
        "Vol_Ratio > 1.2, Pro_Score elevato)."
    )

    with st.expander("📘 Legenda PRO"):
        st.markdown(
            "- **Pro_Score**: punteggio composito (prezzo sopra EMA20, RSI nel range, volume sopra media.\n"
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
        df_pro = df_pro[[c for c in cols_order if c in df_pro.columns]]

        # Top N per Pro_Score
        df_pro_view = df_pro.sort_values("Pro_Score", ascending=False).head(top)

        df_pro_show = df_pro_view[
            [
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
        ]

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

        # EXPORT PRO
        csv_data = df_pro_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export PRO CSV",
            data=csv_data,
            file_name=f"PRO_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_pro_csv",
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_pro_view.to_excel(writer, index=False, sheet_name="PRO")
        data_xlsx = output.getvalue()

        st.download_button(
            "⬇️ Export PRO XLSX",
            data=data_xlsx,
            file_name=f"PRO_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_pro_xlsx",
        )

        # EXPORT TradingView (solo ticker) PRO
        tv_data = df_pro_view["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

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
# REA‑QUANT (HOT) – Top N Vol_Ratio / distanza POC
# =============================================================================
with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT (HOT)")
    st.markdown(
        f"Filtro REA‑QUANT HOT: titoli con **Stato = HOT** e Vol_Ratio ≥ {vol_ratio_hot:.1f}, "
        f"vicini al POC (distanza ≤ {r_poc*100:.1f}%)."
    )

    with st.expander("📘 Legenda REA‑QUANT"):
        st.markdown(
            "- **REA‑QUANT HOT**: pressione volumetrica significativa vicino al POC.\n"
            "- **Vol_Ratio**: volume odierno / media 20 giorni.\n"
            "- **Distanza POC**: distanza percentuale prezzo–POC.\n"
            "- **Market Cap / Volumi**: per contestualizzare la forza dello swing.\n"
            "- Colonne **Yahoo** e **TradingView**: pulsanti link per ogni ticker."
        )

    if df_rea_all.empty:
        st.caption("Nessun segnale REA‑QUANT HOT.")
    else:
        df_rea = df_rea_all.copy()
        df_rea = add_formatted_cols(df_rea)
        df_rea = add_links(df_rea)

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
            "Pro_Score",
            "RSI",
            "Vol_Ratio",
            "Distanza_POC",
            "OBV_Trend",
            "ATR",
            "ATR_Exp",
            "Stato",
            "Yahoo",
            "Finviz",
        ]
        df_rea = df_rea[[c for c in cols_order if c in df_rea.columns]]

        sort_cols = [c for c in ["Vol_Ratio", "Distanza_POC"] if c in df_rea.columns]
        if sort_cols:
            df_rea_view = df_rea.sort_values(
                by=sort_cols, ascending=[False, True][: len(sort_cols)]
            ).head(top)
        else:
            df_rea_view = df_rea.head(top)

        df_rea_show = df_rea_view[
            [
                c
                for c in [
                    "Nome",
                    "Ticker",
                    "Prezzo_fmt",
                    "MarketCap_fmt",
                    "Vol_Today_fmt",
                    "Vol_7d_Avg_fmt",
                    "Pro_Score",
                    "RSI",
                    "Vol_Ratio",
                    "Distanza_POC",
                    "OBV_Trend",
                    "ATR",
                    "ATR_Exp",
                    "Stato",
                    "Yahoo",
                    "Finviz",
                ]
                if c in df_rea_view.columns
            ]
        ]

        st.dataframe(
            df_rea_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Distanza_POC": "Dist POC (%)",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
            },
        )

        # EXPORT REA‑QUANT
        csv_data = df_rea_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export REA‑QUANT CSV",
            data=csv_data,
            file_name=f"REA_HOT_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_rea_csv",
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_rea_view.to_excel(writer, index=False, sheet_name="REA_HOT")
        data_xlsx = output.getvalue()

        st.download_button(
            "⬇️ Export REA‑QUANT XLSX",
            data=data_xlsx,
            file_name=f"REA_HOT_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_rea_xlsx",
        )

        tv_data = df_rea_view["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

        st.download_button(
            "⬇️ Export REA‑QUANT TradingView (solo ticker)",
            data=csv_tv,
            file_name=f"TV_REA_HOT_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_tv_rea",
        )

        options_rea = sorted(
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_rea_view.iterrows()
        )

        col_sel_all_rea, _ = st.columns([1, 3])
        with col_sel_all_rea:
            if st.button("✅ Seleziona tutti (Top N REA‑QUANT)", key="btn_sel_all_rea"):
                st.session_state["wl_rea"] = options_rea

        selection_rea = st.multiselect(
            "Aggiungi alla Watchlist (REA‑QUANT):",
            options=options_rea,
            key="wl_rea",
        )

        note_rea = st.text_input(
            "Note comuni per questi ticker REA‑QUANT", key="note_wl_rea"
        )

        if st.button("📌 Salva in Watchlist (REA‑QUANT)"):
            tickers = [s.split(" – ")[1] for s in selection_rea]
            names = [s.split(" – ")[0] for s in selection_rea]
            add_to_watchlist(
                tickers,
                names,
                "REA_HOT",
                note_rea,
                trend="LONG",
                list_name=st.session_state.get("current_list_name", "DEFAULT"),
            )
            st.success("REA‑QUANT salvati in watchlist.")
            st.rerun()

# =============================================================================
# Rea Quant – Top N combinazione Vol_Ratio / Distanza_POC / Pro_Score
# =============================================================================
with tab_rea_q:
    st.subheader("🧮 Massimo Rea – Analisi Quantitativa")
    st.markdown(
        "Analisi volumetrica e quantitativa sui titoli scansionati, con focus su "
        "pressione volumetrica (Vol_Ratio), vicinanza al POC e forza di trend."
    )

    with st.expander("📘 Legenda Rea Quant"):
        st.markdown(
            "- **Vol_Ratio**: volume odierno / media 20 giorni.\n"
            "- **Distanza_POC**: distanza percentuale prezzo–POC (più è bassa, più il prezzo è sul volume point of control).\n"
            "- **Pro_Score**: forza trend.\n"
            "- **RSI**: momentum di breve.\n"
            "- Colonne **Yahoo** e **TradingView**: pulsanti link per ogni ticker."
        )

    if df_rea_all.empty:
        df_rea_q = pd.DataFrame()
    else:
        df_rea_q = df_rea_all.copy()

    if df_rea_q.empty:
        st.caption("Nessun dato disponibile per l’analisi Rea Quant.")
    else:
        df_rq = df_rea_q.copy()
        df_rq = add_formatted_cols(df_rq)
        df_rq = add_links(df_rq)

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
            "Pro_Score",
            "RSI",
            "Vol_Ratio",
            "Distanza_POC",
            "OBV_Trend",
            "ATR",
            "ATR_Exp",
            "Stato",
            "Yahoo",
            "Finviz",
        ]
        df_rq = df_rq[[c for c in cols_order if c in df_rq.columns]]

        sort_cols = [c for c in ["Vol_Ratio", "Distanza_POC", "Pro_Score"] if c in df_rq.columns]
        ascending = [False, True, False][: len(sort_cols)]
        if sort_cols:
            df_rq_view = df_rq.sort_values(by=sort_cols, ascending=ascending).head(top)
        else:
            df_rq_view = df_rq.head(top)

        df_rq_show = df_rq_view[
            [
                c
                for c in [
                    "Nome",
                    "Ticker",
                    "Prezzo_fmt",
                    "MarketCap_fmt",
                    "Vol_Today_fmt",
                    "Vol_7d_Avg_fmt",
                    "Pro_Score",
                    "RSI",
                    "Vol_Ratio",
                    "Distanza_POC",
                    "OBV_Trend",
                    "ATR",
                    "ATR_Exp",
                    "Stato",
                    "Yahoo",
                    "Finviz",
                ]
                if c in df_rq_view.columns
            ]
        ]

        st.dataframe(
            df_rq_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Distanza_POC": "Dist POC (%)",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
            },
        )

        # EXPORT Rea Quant
        csv_data = df_rq_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Rea Quant CSV",
            data=csv_data,
            file_name=f"REA_QUANT_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_rea_q_csv",
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_rq_view.to_excel(writer, index=False, sheet_name="REA_QUANT")
        data_xlsx = output.getvalue()

        st.download_button(
            "⬇️ Export Rea Quant XLSX",
            data=data_xlsx,
            file_name=f"REA_QUANT_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_rea_q_xlsx",
        )

        tv_data = df_rq_view["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

        st.download_button(
            "⬇️ Export Rea Quant TradingView (solo ticker)",
            data=csv_tv,
            file_name=f"TV_REA_QUANT_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_tv_rea_q",
        )

        options_rea_q = sorted(
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_rq_view.iterrows()
        )

        col_sel_all_rea_q, _ = st.columns([1, 3])
        with col_sel_all_rea_q:
            if st.button(
                "✅ Seleziona tutti (Top Rea Quant)", key="btn_sel_all_rea_q"
            ):
                st.session_state["wl_rea_q"] = options_rea_q

        selection_rea_q = st.multiselect(
            "Aggiungi alla Watchlist (Rea Quant):",
            options=options_rea_q,
            key="wl_rea_q",
        )

        note_rea_q = st.text_input(
            "Note comuni per questi ticker Rea Quant", key="note_wl_rea_q"
        )

        if st.button("📌 Salva in Watchlist (Rea Quant)"):
            tickers = [s.split(" – ")[1] for s in selection_rea_q]
            names = [s.split(" – ")[0] for s in selection_rea_q]
            add_to_watchlist(
                tickers,
                names,
                "REA_QUANT",
                note_rea_q,
                trend="LONG",
                list_name=st.session_state.get("current_list_name", "DEFAULT"),
            )
            st.success("Rea Quant salvati in watchlist.")
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
# REGIME & MOMENTUM – Top N per Momentum
# =============================================================================
with tab_regime:
    st.subheader("🧊 Regime & Momentum multi‑mercato")
    st.markdown(
        "Regime: % PRO vs EARLY sul totale segnali.\n"
        "Momentum: ranking per Pro_Score×10 + RSI su tutti i titoli scansionati."
    )

    with st.expander("📘 Legenda Regime & Momentum"):
        st.markdown(
            "- **Regime**: quota segnali PRO vs EARLY.\n"
            "- **Momentum**: Pro_Score×10 + RSI.\n"
            "- **MarketCap / Volumi**: per contestualizzare i top momentum.\n"
            "- Colonne **Yahoo** e **TradingView**: pulsanti link per ogni ticker."
        )

    if df_ep.empty or "Stato" not in df_ep.columns:
        st.caption("Nessun dato scanner disponibile.")
    else:
        df_all = df_ep.copy().dropna(subset=["Pro_Score", "RSI"])

        n_tot_signals = len(df_all)
        n_pro_tot = (df_all["Stato"] == "PRO").sum()
        n_early_tot = (df_all["Stato"] == "EARLY").sum()

        c1r, c2r, c3r = st.columns(3)
        c1r.metric("Totale segnali (EARLY+PRO)", int(n_tot_signals))
        c2r.metric(
            "% PRO",
            f"{(n_pro_tot / n_tot_signals * 100):.1f}%" if n_tot_signals else "0.0%",
        )
        c3r.metric(
            "% EARLY",
            f"{(n_early_tot / n_tot_signals * 100):.1f}%" if n_tot_signals else "0.0%",
        )

        st.markdown("**Top N momentum (Pro_Score×10 + RSI)**")

        df_all["Momentum"] = df_all["Pro_Score"] * 10 + df_all["RSI"]

        df_all = df_all[df_all["Momentum"] >= momentum_min]

        if df_all.empty:
            st.caption("Nessun titolo soddisfa il filtro Momentum minimo.")
        else:
            df_mom = df_all.sort_values("Momentum", ascending=False).head(top)

            cols_order = [
                "Nome",
                "Ticker",
                "Prezzo",
                "MarketCap",
                "Vol_Today",
                "Vol_7d_Avg",
                "Pro_Score",
                "RSI",
                "Vol_Ratio",
                "OBV_Trend",
                "ATR",
                "Stato",
                "Momentum",
            ]
            df_mom = df_mom[[c for c in cols_order if c in df_mom.columns]]
            df_mom = add_formatted_cols(df_mom)
            df_mom = add_links(df_mom)

            df_mom_show = df_mom[
                [
                    "Nome",
                    "Ticker",
                    "Prezzo_fmt",
                    "MarketCap_fmt",
                    "Vol_Today_fmt",
                    "Vol_7d_Avg_fmt",
                    "Pro_Score",
                    "RSI",
                    "Vol_Ratio",
                    "OBV_Trend",
                    "ATR",
                    "Stato",
                    "Momentum",
                    "Yahoo",
                    "Finviz",
                ]
            ]

            st.dataframe(
                df_mom_show,
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

            csv_data = df_mom.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Momentum CSV",
                data=csv_data,
                file_name=f"MOMENTUM_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_mom_csv",
            )

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_mom.to_excel(writer, index=False, sheet_name="MOMENTUM")
            data_xlsx = output.getvalue()

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

            df_mom_tv = df_mom[["Ticker"]].rename(columns={"Ticker": "symbol"})
            csv_mom_tv = df_mom_tv.to_csv(index=False, header=False).encode("utf-8")
            st.download_button(
                "⬇️ CSV Top Momentum (solo ticker)",
                data=csv_mom_tv,
                file_name=(
                    f"signals_momentum_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="dl_tv_mom",
            )

            def detect_market_simple(t):
                if t.endswith(".MI"):
                    return "FTSE"
                if t.endswith(".PA") or t.endswith(".AS") or t.endswith(".SW"):
                    return "Eurostoxx"
                if t in ["SPY", "QQQ", "IWM", "VTI", "EEM"]:
                    return "USA ETF"
                if t.endswith("-USD"):
                    return "Crypto"
                return "Altro"

            df_all["Mercato"] = df_all["Ticker"].apply(detect_market_simple)

            heat = df_all.groupby("Mercato").agg(
                Momentum_med=("Momentum", "mean"),
                N=("Ticker", "count"),
                MarketCap_med=("MarketCap", "mean"),
                Vol_Today_med=("Vol_Today", "mean"),
            ).reset_index()

            heat["MarketCap_med_fmt"] = heat["MarketCap_med"].apply(
                lambda v: fmt_marketcap(v, "€")
            )
            heat["Vol_Today_med_fmt"] = heat["Vol_Today_med"].apply(fmt_int)

            st.markdown("**Sintesi Regime & Momentum per mercato (tabella)**")
            if not heat.empty:
                st.dataframe(
                    heat[
                        [
                            "Mercato",
                            "Momentum_med",
                            "N",
                            "MarketCap_med_fmt",
                            "Vol_Today_med_fmt",
                        ]
                    ].sort_values("Momentum_med", ascending=False),
                    use_container_width=True,
                    column_config={
                        "Momentum_med": "Momentum medio",
                        "N": "N titoli",
                        "MarketCap_med_fmt": "Market Cap med",
                        "Vol_Today_med_fmt": "Vol medio giorno",
                    },
                )
            else:
                st.caption("Nessun dato sufficiente per la sintesi per mercato.")

            options_regime = sorted(
                f"{row['Nome']} – {row['Ticker']}" for _, row in df_mom.iterrows()
            )

            col_sel_all_regime, _ = st.columns([1, 3])
            with col_sel_all_regime:
                if st.button(
                    "✅ Seleziona tutti (Top Momentum)", key="btn_sel_all_regime"
                ):
                    st.session_state["wl_regime"] = options_regime

            selection_regime = st.multiselect(
                "Aggiungi alla Watchlist (Top Momentum):",
                options=options_regime,
                key="wl_regime",
            )
            note_regime = st.text_input(
                "Note comuni per questi ticker Momentum", key="note_wl_regime"
            )
            if st.button("📌 Salva in Watchlist (Regime/Momentum)"):
                tickers = [s.split(" – ")[1] for s in selection_regime]
                names = [s.split(" – ")[0] for s in selection_regime]
                add_to_watchlist(
                    tickers,
                    names,
                    "REGIME_MOMENTUM",
                    note_regime,
                    trend="LONG",
                    list_name=st.session_state.get("current_list_name", "DEFAULT"),
                )
                st.success("Regime/Momentum salvati in watchlist.")
                st.rerun()

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
# FINVIZ‑LIKE – Top N per MarketCap dopo filtri fondamentali
# =============================================================================
with tab_finviz:
    st.subheader("📊 Screener stile Finviz")
    st.markdown(
        "Filtraggio **fondamentale + quantitativo** (EPS Growth, volume medio, prezzo) "
        "per individuare titoli di qualità con liquidità adeguata."
    )

    with st.expander("📘 Legenda Finviz‑like"):
        st.markdown(
            "- **EPSnextY / EPS5Y**: crescita utili attesa a 1 anno e 5 anni.\n"
            "- **AvgVolmln**: volume medio giornaliero (milioni di pezzi).\n"
            "- **Prezzo minimo**: filtro per evitare penny stock.\n"
            "- Dati derivati dall’universo già scansionato (df_ep).\n"
            "- Colonne **Yahoo** e **TradingView**: pulsanti link per ogni ticker."
        )

    if df_ep.empty:
        st.caption("Nessun dato scanner disponibile per il filtro Finviz‑like.")
    else:
                dffinviz = df_ep.copy()

        if all(col in dffinviz.columns for col in ["EPSnextY", "EPS5Y", "AvgVolmln", "Prezzo"]):
            # 1) rimuovo righe con valori mancanti sulle colonne fondamentali
            dffinviz = dffinviz.dropna(subset=["EPSnextY", "EPS5Y", "AvgVolmln", "Prezzo"])

            # 2) applico i filtri numerici
            dffinviz = dffinviz[
                (dffinviz["EPSnextY"] >= eps_next_y_min)
                & (dffinviz["EPS5Y"] >= eps_next_5y_min)
                & (dffinviz["AvgVolmln"] >= avg_vol_min_mln)
                & (dffinviz["Prezzo"] >= price_min_finviz)
            ]


        if dffinviz.empty:
            st.caption("Nessun titolo soddisfa i filtri Finviz‑like.")
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
                "Stato",
                "Yahoo",
                "Finviz",
            ]
            dffinviz = dffinviz[[c for c in cols_order if c in dffinviz.columns]]

            dffinviz_view = dffinviz.sort_values("MarketCap", ascending=False).head(top)

            dffinviz_show = dffinviz_view[
                [
                    c
                    for c in [
                        "Nome",
                        "Ticker",
                        "Prezzo_fmt",
                        "MarketCap_fmt",
                        "Vol_Today_fmt",
                        "Vol_7d_Avg_fmt",
                        "EPSnextY",
                        "EPS5Y",
                        "AvgVolmln",
                        "Stato",
                        "Yahoo",
                        "Finviz",
                    ]
                    if c in dffinviz_view.columns
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
                    "EPSnextY": "EPS Next Y %",
                    "EPS5Y": "EPS Next 5Y %",
                    "AvgVolmln": "Avg Vol (mln)",
                    "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                    "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
                },
            )

            csv_data = dffinviz_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Finviz‑like CSV",
                data=csv_data,
                file_name=f"FINVIZ_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_finviz_csv",
            )

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                dffinviz_view.to_excel(writer, index=False, sheet_name="FINVIZ")
            data_xlsx = output.getvalue()

            st.download_button(
                "⬇️ Export Finviz‑like XLSX",
                data=data_xlsx,
                file_name=f"FINVIZ_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_finviz_xlsx",
            )

            options_finviz = sorted(
                f"{row['Nome']} – {row['Ticker']}"
                for _, row in dffinviz_view.iterrows()
            )

            col_sel_all_finviz, _ = st.columns([1, 3])
            with col_sel_all_finviz:
                if st.button(
                    "✅ Seleziona tutti (Top Finviz‑like)", key="btn_sel_all_finviz"
                ):
                    st.session_state["wl_finviz"] = options_finviz

            selection_finviz = st.multiselect(
                "Aggiungi alla Watchlist (Finviz‑like):",
                options=options_finviz,
                key="wl_finviz",
            )

            note_finviz = st.text_input(
                "Note comuni per questi ticker Finviz‑like", key="note_wl_finviz"
            )

            if st.button("📌 Salva in Watchlist (Finviz‑like)"):
                tickers = [s.split(" – ")[1] for s in selection_finviz]
                names = [s.split(" – ")[0] for s in selection_finviz]
                add_to_watchlist(
                    tickers,
                    names,
                    "FINVIZ",
                    note_finviz,
                    trend="LONG",
                    list_name=st.session_state.get("current_list_name", "DEFAULT"),
                )
                st.success("Finviz‑like salvati in watchlist.")
                st.rerun()

# =============================================================================
# WATCHLIST & NOTE – gestione completa
# =============================================================================
with tab_watch:
    st.subheader("📌 Watchlist & Note")

    # Carico l'intera tabella dal DB
    df_wl = load_watchlist()

    # =======================
    # Azioni globali DB Watchlist
    # =======================
    st.markdown("### ⚙️ Azioni rapide Watchlist")

    col_a1, col_a2, col_a3 = st.columns(3)

    # 1) Refresh semplice (rilegge il DB)
    with col_a1:
        if st.button("🔄 Refresh watchlist"):
            st.rerun()

    # 2) Reset COMPLETO DB (tutte le liste)
    with col_a2:
        if st.button("🧨 RESET COMPLETO DB", type="secondary"):
            reset_watchlist_db()
            st.success("DB watchlist azzerato (tutte le liste).")
            st.rerun()

    # 3) Reset SOLO lista corrente (verrà definita poco dopo)
    with col_a3:
        # uso il valore in sessione, se non c'è metto DEFAULT
        wl_to_reset = st.session_state.get("current_list_name", "DEFAULT")
        if st.button(f"🗑️ Svuota lista corrente ({wl_to_reset})"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "DELETE FROM watchlist WHERE list_name = ?",
                (wl_to_reset,),
            )
            conn.commit()
            conn.close()
            st.success(f"Watchlist '{wl_to_reset}' svuotata.")
            st.rerun()

    
    if df_wl.empty:
        st.caption("Watchlist vuota. Aggiungi titoli dai tab dello scanner.")
        st.stop()

    # Normalizzo colonne chiave
    for col in ["id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"]:
        if col not in df_wl.columns:
            df_wl[col] = "" if col != "id" else np.nan

    # =======================
    # Riepilogo per lista
    # =======================
    df_wl["list_name"] = (
        df_wl["list_name"]
        .fillna("DEFAULT")
        .astype(str)
        .str.strip()
        .replace("", "DEFAULT")
    )
    all_lists = sorted(df_wl["list_name"].unique().tolist())

    summary = (
        df_wl.groupby("list_name")["id"]
        .count()
        .reset_index(name="N_titoli")
        .sort_values("list_name")
    )

    st.markdown("**Riepilogo watchlist (numero titoli per lista):**")
    st.dataframe(
        summary,
        use_container_width=True,
        column_config={"list_name": "Lista", "N_titoli": "N titoli"},
        hide_index=True,
    )

    st.markdown("---")

    # =======================
    # Selezione lista da visualizzare
    # =======================
    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        default_list = st.session_state.get("current_list_name", all_lists[0])
        if default_list not in all_lists:
            default_list = all_lists[0]
        current_list = st.selectbox(
            "Lista da visualizzare",
            options=all_lists,
            index=all_lists.index(default_list),
            help="Seleziona quale watchlist vuoi visualizzare e modificare.",
        )
    with col_l2:
        st.markdown(
            f"Lista attiva sidebar: **{st.session_state.get('current_list_name', 'DEFAULT')}**"
        )

    df_wl_list = df_wl[df_wl["list_name"] == current_list].copy()

    if df_wl_list.empty:
        st.caption(f"Questa lista è vuota: '{current_list}'.")
        st.stop()

    # =======================
    # Dati di mercato da Yahoo
    # =======================
    tickers_wl = df_wl_list["ticker"].dropna().unique().tolist()
    records_mkt = []
    for tkr in tickers_wl:
        try:
            yt = yf.Ticker(tkr)
            info = yt.info
            hist = yt.history(period="7d")
            if hist.empty:
                continue

            close = hist["Close"]
            vol = hist["Volume"]

            price = float(close.iloc[-1])
            vol_today = float(vol.iloc[-1])
            vol_7d_avg = float(vol.mean())
            marketcap = info.get("marketCap", np.nan)
            currency = info.get("currency", "USD")

            records_mkt.append(
                {
                    "ticker": tkr,
                    "Prezzo": price,
                    "MarketCap": marketcap,
                    "Vol_Today": vol_today,
                    "Vol_7d_Avg": vol_7d_avg,
                    "Currency": currency,
                }
            )
        except Exception:
            continue

    df_mkt = pd.DataFrame(records_mkt)

    df_wl_filt = df_wl_list.copy()
    if not df_mkt.empty:
        df_wl_filt = df_wl_filt.merge(df_mkt, on="ticker", how="left")

    df_wl_filt = add_formatted_cols(df_wl_filt)
    df_wl_filt = add_links(df_wl_filt)

    # Label per selezioni
    df_wl_filt["label"] = df_wl_filt.apply(
        lambda r: f"{r['id']} – {r.get('name', '')} ({r['ticker']})",
        axis=1,
    )

    # (se vuoi i Top N ultimi inseriti, decommenta queste righe)
    # df_wl_filt = df_wl_filt.sort_values("created_at", ascending=False).head(top)

    # Ordine tabella (alfabetico, o cambia qui se vuoi un ranking diverso)
    df_wl_filt = df_wl_filt.sort_values(["name", "ticker"], na_position="last")

    # =======================
    # Tabella principale
    # =======================
    cols_show = ["label", "trend", "origine", "note", "created_at"]

    if "Prezzo_fmt" in df_wl_filt.columns:
        cols_show.insert(1, "Prezzo_fmt")
    if "MarketCap_fmt" in df_wl_filt.columns:
        cols_show.insert(2, "MarketCap_fmt")
    if "Vol_Today_fmt" in df_wl_filt.columns:
        cols_show.insert(3, "Vol_Today_fmt")
    if "Vol_7d_Avg_fmt" in df_wl_filt.columns:
        cols_show.insert(4, "Vol_7d_Avg_fmt")
    if "Yahoo" in df_wl_filt.columns:
        cols_show.append("Yahoo")
    if "Finviz" in df_wl_filt.columns:
        cols_show.append("Finviz")

    cols_show = [c for c in cols_show if c in df_wl_filt.columns]
    df_wl_view = df_wl_filt[cols_show]

    st.dataframe(
        df_wl_view,
        use_container_width=True,
        column_config={
            "label": "ID – Nome (Ticker)",
            "Prezzo_fmt": "Prezzo",
            "MarketCap_fmt": "Market Cap",
            "Vol_Today_fmt": "Vol giorno",
            "Vol_7d_Avg_fmt": "Vol medio 7g",
            "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
            "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
        },
    )

    st.markdown("---")

    # =======================
    # Export Watchlist corrente
    # =======================
    csv_wl = df_wl_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Watchlist corrente CSV",
        data=csv_wl,
        file_name=f"WATCHLIST_{current_list}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_wl_csv",
    )

    output_wl = io.BytesIO()
    with pd.ExcelWriter(output_wl, engine="xlsxwriter") as writer:
        df_wl_filt.to_excel(writer, index=False, sheet_name=str(current_list)[:31])
    xlsx_wl = output_wl.getvalue()

    st.download_button(
        "⬇️ Export Watchlist corrente XLSX",
        data=xlsx_wl,
        file_name=f"WATCHLIST_{current_list}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_wl_xlsx",
    )

    # Solo ticker, utile per TradingView
    wl_tv = df_wl_filt[["ticker"]].drop_duplicates().rename(columns={"ticker": "symbol"})
    csv_wl_tv = wl_tv.to_csv(index=False, header=False).encode("utf-8")
    st.download_button(
        "⬇️ CSV Watchlist corrente (solo ticker)",
        data=csv_wl_tv,
        file_name=f"WATCHLIST_{current_list}_TICKER_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_wl_tv",
    )

    
    # =======================
    # Modifica nota singola
    # =======================
    st.subheader("📝 Modifica nota per una riga")

    options_row = df_wl_filt["label"].tolist()
    row_sel = st.selectbox(
        "Seleziona riga",
        options=["—"] + options_row,
        index=0,
    )

    if row_sel != "—":
        row_id = int(row_sel.split(" – ")[0])
        old_note = df_wl_filt.loc[df_wl_filt["id"] == row_id, "note"].iloc[0]
        new_note = st.text_area(
            "Nota",
            value=old_note,
            key=f"note_edit_{row_id}",
        )
        if st.button("💾 Salva nota"):
            update_watchlist_note(row_id, new_note)
            st.success("Nota aggiornata.")
            st.rerun()

    st.markdown("---")

    # =======================
    # Spostamento N righe tra liste
    # =======================
    st.subheader("📂 Sposta righe tra watchlist")

    move_rows = st.multiselect(
        "Seleziona righe da spostare",
        options=options_row,
        key="move_rows",
        help="Puoi selezionare una o più righe dalla lista corrente.",
    )

    target_list = st.selectbox(
        "Lista di destinazione",
        options=all_lists + ["[NUOVA LISTA]"],
        index=0,
    )

    new_list_name_input = ""
    if target_list == "[NUOVA LISTA]":
        new_list_name_input = st.text_input(
            "Nome nuova lista",
            value="",
            key="new_list_name_move",
            help="Se lasci vuoto userò 'NUOVA'.",
        )

    if st.button("➡️ Sposta righe"):
        if not move_rows:
            st.warning("Seleziona almeno una riga da spostare.")
        else:
            if target_list == "[NUOVA LISTA]":
                dest = new_list_name_input.strip() or "NUOVA"
            else:
                dest = target_list

            ids_to_move = [int(s.split(" – ")[0]) for s in move_rows]

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.executemany(
                "UPDATE watchlist SET list_name = ? WHERE id = ?",
                [(dest, rid) for rid in ids_to_move],
            )
            conn.commit()
            conn.close()

            st.success(f"Spostate {len(ids_to_move)} righe nella lista '{dest}'.")
            st.rerun()

    st.markdown("---")

    # =======================
    # Cancellazione N righe
    # =======================
    st.subheader("🗑️ Cancella righe dalla lista corrente")

    del_rows = st.multiselect(
        "Seleziona righe da cancellare",
        options=options_row,
        key="del_rows",
    )

    if st.button("❌ Elimina righe selezionate"):
        if not del_rows:
            st.warning("Seleziona almeno una riga da eliminare.")
        else:
            ids_to_del = [int(s.split(" – ")[0]) for s in del_rows]
            delete_from_watchlist(ids_to_del)
            st.success(f"Eliminate {len(ids_to_del)} righe dalla lista '{current_list}'.")
            st.rerun()



