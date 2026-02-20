import io
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sqlite3
from pathlib import Path
from fpdf import FPDF  # pip install fpdf2

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 9.0",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Trading Scanner – Versione PRO 9.0")

st.caption(
    "EARLY • PRO • REA‑QUANT • Rea Quant • Serafini • Regime & Momentum • "
    "Multi‑Timeframe • Finviz • Watchlist DB"
)

# -----------------------------------------------------------------------------
# DB WATCHLIST (SQLite) + MIGRAZIONE TREND
# -----------------------------------------------------------------------------
import locale
locale.setlocale(locale.LC_ALL, "")

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
        return f"{symbol}{v/1_000_000_000:,.2f}B".replace(",", "X").replace(".", ",").replace("X", ".")
    if v >= 1_000_000:
        return f"{symbol}{v/1_000_000:,.2f}M".replace(",", "X").replace(".", ",").replace("X", ".")
    if v >= 1_000:
        return f"{symbol}{v/1_000:,.2f}K".replace(",", "X").replace(".", ",").replace("X", ".")
    return fmt_currency(v, symbol)

def add_formatted_cols(df):
    # se manca Currency, default USD
    if "Currency" not in df.columns:
        df["Currency"] = "USD"

    # Prezzo: crea Prezzo_fmt solo se la colonna esiste
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df.apply(
            lambda r: fmt_currency(
                r["Prezzo"],
                "€" if r["Currency"] == "EUR" else "$"
            ),
            axis=1,
        )

    # MarketCap
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df.apply(
            lambda r: fmt_marketcap(
                r["MarketCap"],
                "€" if r["Currency"] == "EUR" else "$"
            ),
            axis=1,
        )

    # Volumi
    if "Vol_Today" in df.columns:
        df["Vol_Today_fmt"] = df["Vol_Today"].apply(fmt_int)

    if "Vol_7d_Avg" in df.columns:
        df["Vol_7d_Avg_fmt"] = df["Vol_7d_Avg"].apply(fmt_int)

    return df


#########################################################
#######          VECCHIO CODICE FINVIZ        ###########
#########################################################
#def add_links(df):
#    df["Yahoo"] = df["Ticker"].apply(
#        lambda t: f"https://finance.yahoo.com/quote/{t}"
#    )
#    df["Finviz"] = df["Ticker"].apply(
#        lambda t: f"https://finviz.com/quote.ashx?t={t.split('.')[0]}"
#   )
#    return df
#########################################################
#######          NUOVO CODICE TRADINGVIEW     ###########
#########################################################
def add_links(df):
    # usa 'Ticker' se c'è, altrimenti 'ticker'
    if "Ticker" in df.columns:
        col = "Ticker"
    else:
        col = "ticker"

    df["Yahoo"] = df[col].apply(
        lambda t: f"https://finance.yahoo.com/quote/{t}"
    )
    # link TradingView nella colonna Finviz
    df["Finviz"] = df[col].apply(
        lambda t: f"https://www.tradingview.com/chart/?symbol={t.split('.')[0]}"
    )
    return df



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

def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute(
            "INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (t, n, trend, origine, note, list_name, now),
        )
    conn.commit()
    conn.close()


def load_watchlist():
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=["id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"]
        )
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    for col in ["ticker", "name", "trend", "origine", "note", "list_name", "created_at"]:
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

init_db()

# =============================================================================
# SIDEBAR – MERCATI E PARAMETRI
# =============================================================================
st.sidebar.title("⚙️ Configurazione")

# inizializzazione una sola volta
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

# ---------------- Selezione Mercati (persistente) ----------------
st.sidebar.subheader("📈 Selezione Mercati")
m = {
    "Eurostoxx":   st.sidebar.checkbox("🇪🇺 Eurostoxx 600", False),
    "FTSE":        st.sidebar.checkbox("🇮🇹 FTSE MIB", st.session_state["m_FTSE"]),
    "SP500":       st.sidebar.checkbox("🇺🇸 S&P 500", st.session_state["m_SP500"]),
    "Nasdaq":      st.sidebar.checkbox("🇺🇸 Nasdaq 100", st.session_state["m_Nasdaq"]),
    "Dow":         st.sidebar.checkbox("🇺🇸 Dow Jones", False),
    "Russell":     st.sidebar.checkbox("🇺🇸 Russell 2000", False),
    "Commodities": st.sidebar.checkbox("🛢️ Materie Prime", False),
    "ETF":         st.sidebar.checkbox("📦 ETF", False),
    "Crypto":      st.sidebar.checkbox("₿ Crypto", False),
    "Emerging":    st.sidebar.checkbox("🌍 Emergenti", False),
}
sel = [k for k, v in m.items() if v]

# aggiorno lo stato mercati
st.session_state["m_FTSE"] = m["FTSE"]
st.session_state["m_SP500"] = m["SP500"]
st.session_state["m_Nasdaq"] = m["Nasdaq"]

st.sidebar.divider()

# ---------------- Parametri Scanner (persistenti) ----------------
st.sidebar.subheader("🎛️ Parametri Scanner")

e_h = st.sidebar.slider(
    "EARLY - Distanza EMA20 (%)",
    0.0, 10.0,
    float(st.session_state["e_h"] * 100),
    0.5,
) / 100
st.session_state["e_h"] = e_h

p_rmin = st.sidebar.slider(
    "PRO - RSI minimo", 0, 100, int(st.session_state["p_rmin"]), 5
)
st.session_state["p_rmin"] = p_rmin

p_rmax = st.sidebar.slider(
    "PRO - RSI massimo", 0, 100, int(st.session_state["p_rmax"]), 5
)
st.session_state["p_rmax"] = p_rmax

r_poc = st.sidebar.slider(
    "REA - Distanza POC (%)",
    0.0, 10.0,
    float(st.session_state["r_poc"] * 100),
    0.5,
) / 100
st.session_state["r_poc"] = r_poc

# ---------------- Filtri avanzati (come prima) ----------------
st.sidebar.subheader("🔎 Filtri avanzati")

# Finviz-like
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

# Rea-Quant
vol_ratio_hot = st.sidebar.number_input(
    "Vol_Ratio minimo REA‑HOT", 0.0, 10.0, 1.5, 0.1
)

# Momentum
momentum_min = st.sidebar.number_input(
    "Momentum minimo (Pro_Score*10+RSI)", 0.0, 2000.0, 0.0, 10.0
)

# ---------------- Output (persistente) ----------------
st.sidebar.subheader("📤 Output")
top = st.sidebar.number_input(
    "TOP N titoli per tab", 5, 50, int(st.session_state["top"]), 5
)
st.session_state["top"] = top

st.sidebar.subheader("📁 Lista Watchlist attiva")

existing_lists = load_watchlist()
list_options = sorted(existing_lists["list_name"].dropna().unique().tolist())
list_options = [ln for ln in list_options if ln]
default_list = list_options[0] if list_options else "DEFAULT"

current_list = st.sidebar.text_input(
    "Nome lista (nuova o esistente)",
    value=st.session_state.get("current_list_name", default_list),
)
st.session_state["current_list_name"] = current_list


# ---------------- Controllo mercati selezionati ----------------
if not sel:
    st.warning("⚠️ Seleziona almeno un mercato dalla sidebar.")
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

        rea_score = 7 if (dist_poc < r_poc and vol_ratio > vol_ratio_hot) else 0

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

# =============================================================================
# SCAN
# =============================================================================
if "done_pro" not in st.session_state:
    st.session_state["done_pro"] = False

if st.button("🚀 AVVIA SCANNER PRO 8.0", type="primary", use_container_width=True):
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
# EARLY
# =============================================================================
with tab_e:
    st.subheader("🟢 Segnali EARLY")
    st.markdown(
        f"Filtro EARLY: titoli con **Stato = EARLY** (distanza prezzo–EMA20 < {e_h*100:.1f}%), "
        "punteggio Early_Score ≥ 8."
    )

    with st.expander("📘 Legenda EARLY"):
        st.markdown(
            "- **Early_Score**: 8 se il prezzo è entro la soglia percentuale dalla EMA20; 0 altrimenti.\n"
            "- **RSI**: RSI a 14 periodi.\n"
            "- **Vol_Ratio**: volume odierno / media 20 giorni.\n"
            "- **Market Cap**: capitalizzazione abbreviata (K/M/B) con valuta.\n"
            "- **Vol_Today / Vol_7d_Avg**: volume odierno e media degli ultimi 7 giorni.\n"
            "- **Stato = EARLY**: setup in formazione vicino alla media.\n"
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
        )

    if df_early_all.empty:
        st.caption("Nessun segnale EARLY.")
    else:
        df_early = df_early_all.copy()
        df_early = add_formatted_cols(df_early)
        df_early = add_links(df_early)

        # mantengo anche le colonne numeriche Prezzo/MarketCap/Vol per CSV
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

        df_early_view = df_early.sort_values("Early_Score", ascending=False).head(top)

        # Tabella: uso solo le colonne formattate + pulsanti link
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

        # ==========================
        # EXPORT EARLY
        # ==========================
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

        # EXPORT TradingView (solo ticker) EARLY
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

        # ==========================
        # Watchlist EARLY (con seleziona tutti)
        # ==========================
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
    tickers, names, "EARLY", note_early, trend="LONG",
    list_name=st.session_state.get("current_list_name", "DEFAULT"),
)
 st.rerun()

# =============================================================================
# PRO
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
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
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

        df_pro_view = df_pro.sort_values("Pro_Score", ascending=False).head(top)
        df_pro_view["OBV_Trend"] = df_pro_view["OBV_Trend"].replace(
            {"UP": "UP (flusso in ingresso)", "DOWN": "DOWN (flusso in uscita)"}
        )

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

        # ==========================
        # EXPORT PRO
        # ==========================
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

        # EXPORT PRO TradingView (solo ticker)
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

        # CSV PRO arricchito per TradingView
        df_pro_tv = df_pro_view.rename(
            columns={
                "Ticker": "symbol",
                "Prezzo": "price",
                "RSI": "rsi",
                "Vol_Ratio": "volume_ratio",
                "OBV_Trend": "obv_trend",
            }
        )[[ "symbol", "price", "rsi", "volume_ratio", "obv_trend" ]]
        csv_pro = df_pro_tv.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ CSV PRO per TradingView (dettagliato)",
            data=csv_pro,
            file_name=f"signals_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_pro_tv_full",
        )

        # ==========================
        # Watchlist PRO (con seleziona tutti) – key wl_pro
        # ==========================
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
    tickers, names, "PRO", note_early, trend="LONG",
    list_name=st.session_state.get("current_list_name", "DEFAULT"),
)

            st.rerun()

# =============================================================================
# REA‑QUANT (segnali)
# =============================================================================
with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT")
    st.markdown(
        f"Filtro REA‑QUANT: titoli con **Stato = HOT** "
        f"(distanza dal POC < {r_poc*100:.1f}%, Vol_Ratio > {vol_ratio_hot})."
    )

    with st.expander("📘 Legenda REA‑QUANT (segnali)"):
        st.markdown(
            "- **Rea_Score**: 7 quando prezzo vicino al POC e volume molto sopra la media.\n"
            "- **Market Cap**: capitalizzazione abbreviata (K/M/B) con valuta.\n"
            "- **Vol_Today / Vol_7d_Avg**: volume odierno e media 7 giorni.\n"
            "- **POC**: livello di prezzo con il massimo volume scambiato.\n"
            "- **Dist_POC_%**: distanza % tra prezzo e POC.\n"
            "- **Stato = HOT**: area di forte decisione.\n"
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
        )

    if df_rea_all.empty:
        st.caption("Nessun segnale REA‑QUANT.")
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
            "Rea_Score",
            "POC",
            "Dist_POC_%",
            "Vol_Ratio",
            "Stato",
            "Yahoo",
            "Finviz",
        ]
        df_rea = df_rea[[c for c in cols_order if c in df_rea.columns]]

        df_rea_view = df_rea.sort_values("Rea_Score", ascending=False).head(top)

        df_rea_show = df_rea_view[
            [
                "Nome",
                "Ticker",
                "Prezzo_fmt",
                "MarketCap_fmt",
                "Vol_Today_fmt",
                "Vol_7d_Avg_fmt",
                "Rea_Score",
                "POC",
                "Dist_POC_%",
                "Vol_Ratio",
                "Stato",
                "Yahoo",
                "Finviz",
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
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("TradingView", display_text="Apri"),
            },
        )

        # ==========================
        # EXPORT REA‑QUANT
        # ==========================
        csv_data = df_rea_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export REA‑QUANT CSV",
            data=csv_data,
            file_name=f"REA_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_rea_csv",
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_rea_view.to_excel(writer, index=False, sheet_name="REA")
        data_xlsx = output.getvalue()

        st.download_button(
            "⬇️ Export REA‑QUANT XLSX",
            data=data_xlsx,
            file_name=f"REA_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_rea_xlsx",
        )

        # EXPORT REA‑QUANT TradingView (solo ticker)
        tv_data = df_rea_view["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

        st.download_button(
            "⬇️ Export REA‑QUANT TradingView (solo ticker)",
            data=csv_tv,
            file_name=f"TV_REA_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_tv_rea",
        )

        # CSV REA‑QUANT per TradingView (dettagliato)
        df_rea_tv = df_rea_view.rename(
            columns={
                "Ticker": "symbol",
                "Prezzo": "price",
                "POC": "poc",
                "Dist_POC_%": "dist_poc_percent",
                "Vol_Ratio": "volume_ratio",
            }
        )[[ "symbol", "price", "poc", "dist_poc_percent", "volume_ratio" ]]
        csv_rea = df_rea_tv.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ CSV REA‑QUANT per TradingView (dettagliato)",
            data=csv_rea,
            file_name=f"signals_rea_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_rea_tv_full",
        )

        # ==========================
        # Watchlist REA‑QUANT (con seleziona tutti) – key wl_rea
        # ==========================
        options_rea = sorted(
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_rea_view.iterrows()
        )

        col_sel_all_rea, _ = st.columns([1, 3])
        with col_sel_all_rea:
            if st.button("✅ Seleziona tutti (Top N REA‑QUANT)", key="btn_sel_all_rea"):
                st.session_state["wl_rea"] = options_rea

        selection_rea = st.multiselect(
            "Aggiungi alla Watchlist (REA‑QUANT HOT):",
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
        tickers, names, "REA_HOT", note_early, trend="LONG",
        list_name=st.session_state.get("current_list_name", "DEFAULT"),
)

            st.rerun()
# =============================================================================
# MASSIMO REA – ANALISI QUANT
# =============================================================================
with tab_rea_q:
    st.subheader("🧮 Rea Quant – Top N e analisi avanzata")

    if df_rea_all.empty:
        st.caption("Nessun dato REA‑QUANT disponibile.")
        df_rea_q = pd.DataFrame()
    else:
        df_rea_q = df_rea_all.copy()

        # porto dentro il prezzo dal dataframe principale, se disponibile
        if "Prezzo" in df_ep.columns:
            df_rea_q = df_rea_q.merge(
                df_ep[["Ticker", "Prezzo", "Currency"]],
                on="Ticker",
                how="left",
            )

        # ==========================
        # 1) TOP N per pressione volumetrica (sempre visibile)
        # ==========================
        st.markdown("**Top N per pressione volumetrica (Vol_Ratio)**")

        df_rea_top = df_rea_q.sort_values("Vol_Ratio", ascending=False).head(top)
        df_rea_top = add_formatted_cols(df_rea_top)
        df_rea_top = add_links(df_rea_top)

        # costruisco la vista, usando Prezzo_fmt se esiste, altrimenti Prezzo
        if "Prezzo_fmt" in df_rea_top.columns:
            prezzo_col = "Prezzo_fmt"
        elif "Prezzo" in df_rea_top.columns:
            prezzo_col = "Prezzo"
        else:
            prezzo_col = None

        cols = ["Nome", "Ticker"]
        if prezzo_col is not None:
            cols.append(prezzo_col)
        cols += [
            "MarketCap_fmt",
            "Vol_Today_fmt",
            "Vol_7d_Avg_fmt",
            "POC",
            "Dist_POC_%",
            "Vol_Ratio",
            "Stato",
            "Yahoo",
            "Finviz",
        ]

        df_rea_top_show = df_rea_top[[c for c in cols if c in df_rea_top.columns]]

        if prezzo_col == "Prezzo_fmt":
            df_rea_top_show = df_rea_top_show.rename(columns={"Prezzo_fmt": "Prezzo"})
        elif prezzo_col == "Prezzo":
            df_rea_top_show = df_rea_top_show.rename(columns={"Prezzo": "Prezzo"})

        st.dataframe(
            df_rea_top_show,
            use_container_width=True,
            column_config={
                "Prezzo": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn(
                    "TradingView", display_text="Apri"
                ),
            },
        )

        # EXPORT REA TOP N
        csv_data = df_rea_top.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export REA TopN CSV",
            data=csv_data,
            file_name=f"REA_TOPN_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_rea_topn_csv",
        )

        tv_data = df_rea_top["Ticker"].drop_duplicates().to_frame(name="symbol")
        csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

        st.download_button(
            "⬇️ Export REA TopN TradingView (solo ticker)",
            data=csv_tv,
            file_name=f"TV_REA_TOPN_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_tv_rea_topn",
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_rea_top.to_excel(writer, index=False, sheet_name="REA_TOPN")
        data_xlsx = output.getvalue()

        st.download_button(
            "⬇️ Export REA TopN XLSX",
            data=data_xlsx,
            file_name=f"REA_TOPN_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
            key="dl_rea_topn_xlsx",
        )

        # ==========================
        # Watchlist Rea Quant TopN (con seleziona tutti) – key wl_rea_q
        # ==========================
        options_rea_q = sorted(
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_rea_top.iterrows()
        )

        col_sel_all_rea_q, _ = st.columns([1, 3])
        with col_sel_all_rea_q:
            if st.button("✅ Seleziona tutti (Rea Quant TopN)", key="btn_sel_all_rea_q"):
                st.session_state["wl_rea_q"] = options_rea_q

        selection_rea_q = st.multiselect(
            "Aggiungi alla Watchlist (Rea Quant TopN):",
            options=options_rea_q,
            key="wl_rea_q",
        )

        note_rea_q = st.text_input(
            "Note comuni per questi ticker (Rea Quant)", key="note_wl_rea_q"
        )

        if st.button("📌 Salva in Watchlist (Rea Quant)"):
            tickers = [s.split(" – ")[1] for s in selection_rea_q]
            names = [s.split(" – ")[0] for s in selection_rea_q]
            add_to_watchlist(
        tickers, names, "REA_QUANT", note_early, trend="LONG",
        list_name=st.session_state.get("current_list_name", "DEFAULT"),
)

            st.rerun()

        # ==========================
        # 2) Analisi per mercato (AVANZATA, nascosta)
        # ==========================
        with st.expander("📊 Analisi Rea Quant per mercato (avanzata)", expanded=False):

            def detect_market_rea(t):
                if t.endswith(".MI"):
                    return "FTSE"
                if t.endswith(".PA") or t.endswith(".AS") or t.endswith(".SW"):
                    return "Eurostoxx"
                if t in ["SPY", "QQQ", "IWM", "VTI"]:
                    return "USA ETF"
                if t.endswith("-USD"):
                    return "Crypto"
                return "Altro"

            df_rea_q["Mercato"] = df_rea_q["Ticker"].apply(detect_market_rea)

            agg = df_rea_q.groupby("Mercato").agg(
                N=("Ticker", "count"),
                Vol_Ratio_med=("Vol_Ratio", "mean"),
                Rea_Score_med=("Rea_Score", "mean"),
                MarketCap_med=("MarketCap", "mean"),
                Vol_Today_med=("Vol_Today", "mean"),
            ).reset_index()

            st.dataframe(agg, use_container_width=True)


# =============================================================================
# STEFANO SERAFINI – SYSTEMS
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
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
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
            # unisco con df_ep per avere Nome, MarketCap, Volumi, Pro_Score, ecc.
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

            # preferisco i breakout LONG, ordinati per Pro_Score
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

                # ==========================
                # EXPORT SERAFINI
                # ==========================
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

                # ==========================
                # Watchlist Serafini (con seleziona tutti) – key wl_seraf
                # ==========================
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
                        add_to_watchlist(
            tickers, names, "SERAFINI", note_early, trend="LONG",
            list_name=st.session_state.get("current_list_name", "DEFAULT"),
)

                    st.rerun()
# =============================================================================
# REGIME & MOMENTUM
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
        # uso solo righe con Pro_Score e RSI validi
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

        # calcolo Momentum
        df_all["Momentum"] = df_all["Pro_Score"] * 10 + df_all["RSI"]

        # applico filtro minimo da sidebar (momentum_min)
        df_all = df_all[df_all["Momentum"] >= momentum_min]

        if df_all.empty:
            st.caption("Nessun titolo soddisfa il filtro Momentum minimo.")
        else:
            # Top N dal df_all filtrato
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

            # ==========================
            # EXPORT MOMENTUM
            # ==========================
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

            # EXPORT Momentum TradingView (solo ticker)
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

            # ==========================
            # Sintesi Regime & Momentum per mercato
            # ==========================
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

            # formatto MarketCap_med e Vol_Today_med
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

            # ==========================
            # Watchlist Top Momentum (con seleziona tutti) – key wl_regime
            # ==========================
            options_regime = sorted(
                f"{row['Nome']} – {row['Ticker']}" for _, row in df_mom.iterrows()
            )

            col_sel_all_regime, _ = st.columns([1, 3])
            with col_sel_all_regime:
                if st.button("✅ Seleziona tutti (Top Momentum)", key="btn_sel_all_regime"):
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
            tickers, names, "REGIME_MOMENTUM", note_early, trend="LONG",
            list_name=st.session_state.get("current_list_name", "DEFAULT"),
)

                st.rerun()

# =============================================================================
# MULTI‑TIMEFRAME
# =============================================================================
with tab_mtf:
    st.subheader("🕒 Analisi Multi‑Timeframe")
    st.markdown(
        "Vista sintetica dei segnali su più timeframe, usando i risultati PRO/EARLY "
        "come base (es. daily con supporto di segnali su timeframe maggiori/minori)."
    )

    with st.expander("📘 Legenda Multi‑Timeframe"):
        st.markdown(
            "- Usa i segnali PRO/EARLY come base.\n"
            "- Mostra metriche di trend/momentum utili su più orizzonti.\n"
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
        )

    if df_ep.empty:
        st.caption("Nessun dato scanner disponibile per la vista Multi‑Timeframe.")
    else:
        # per semplicità usiamo df_ep così com'è; puoi arricchirlo con altre logiche MTF
        df_mtf = df_ep.copy()
        df_mtf = add_formatted_cols(df_mtf)
        df_mtf = add_links(df_mtf)

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
        df_mtf = df_mtf[[c for c in cols_order if c in df_mtf.columns]]

        # ordino per Pro_Score + Early_Score come proxy di forza multi‑timeframe
        df_mtf["MTF_Score"] = df_mtf.get("Pro_Score", 0) + df_mtf.get("Early_Score", 0)
        df_mtf_view = df_mtf.sort_values("MTF_Score", ascending=False).head(top)

        df_mtf_show = df_mtf_view[
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
                "MTF_Score",
                "Yahoo",
                "Finviz",
            ]
        ]

        st.dataframe(
            df_mtf_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "MTF_Score": "MTF Score",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn(
                    "TradingView", display_text="Apri"
                ),
            },
        )

        # ==========================
        # EXPORT MTF
        # ==========================
        csv_data = df_mtf_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export MTF CSV",
            data=csv_data,
            file_name=f"MTF_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_mtf_csv",
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_mtf_view.to_excel(writer, index=False, sheet_name="MTF")
        data_xlsx = output.getvalue()

        st.download_button(
            "⬇️ Export MTF XLSX",
            data=data_xlsx,
            file_name=f"MTF_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_mtf_xlsx",
        )

        # ==========================
        # Watchlist Multi‑Timeframe (con seleziona tutti) – key wl_mtf
        # ==========================
        options_mtf = sorted(
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_mtf_view.iterrows()
        )

        col_sel_all_mtf, _ = st.columns([1, 3])
        with col_sel_all_mtf:
            if st.button("✅ Seleziona tutti (Top MTF)", key="btn_sel_all_mtf"):
                st.session_state["wl_mtf"] = options_mtf

        selection_mtf = st.multiselect(
            "Aggiungi alla Watchlist (Multi‑Timeframe):",
            options=options_mtf,
            key="wl_mtf",
        )

        note_mtf = st.text_input(
            "Note comuni per questi ticker Multi‑Timeframe", key="note_wl_mtf"
        )

        if st.button("📌 Salva in Watchlist (Multi‑Timeframe)"):
            tickers = [s.split(" – ")[1] for s in selection_mtf]
            names = [s.split(" – ")[0] for s in selection_mtf]
            add_to_watchlist(
            tickers, names, "MTF", note_early, trend="LONG",
            list_name=st.session_state.get("current_list_name", "DEFAULT"),
)

            st.rerun()

# =============================================================================
# FINVIZ‑LIKE SCREENER
# =============================================================================
with tab_finviz:
    st.subheader("📊 Screener stile Finviz")
    st.markdown(
        "Filtraggio fondamentale/quantitativo (EPS Growth, volume medio, prezzo) "
        "per individuare titoli di qualità e liquidità adeguata."
    )

    with st.expander("📘 Legenda Finviz"):
        st.markdown(
            "- Filtri EPS Next Year / Next 5Y, volume medio e prezzo minimo.\n"
            "- I dati sono derivati dall'universo scansionato.\n"
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
        )

    if df_ep.empty:
        st.caption("Nessun dato scanner disponibile per il filtro Finviz.")
    else:
        # qui assumo che tu abbia già creato df_finviz a partire da df_ep
        # se non c'è, puoi partire da df_ep e aggiungere colonne fondamentali a modo tuo
        df_finviz = df_ep.copy()

        # ESEMPIO: se hai già colonne EPS_nextY, EPS_5Y, Avg_Vol_mln, Price
        # applica i filtri della sidebar
        if all(col in df_finviz.columns for col in ["EPS_nextY", "EPS_5Y", "Avg_Vol_mln", "Prezzo"]):
            df_finviz = df_finviz[
                (df_finviz["EPS_nextY"] >= eps_next_y_min)
                & (df_finviz["EPS_5Y"] >= eps_next_5y_min)
                & (df_finviz["Avg_Vol_mln"] >= avg_vol_min_mln)
                & (df_finviz["Prezzo"] >= price_min_finviz)
            ]

        if df_finviz.empty:
            st.caption("Nessun titolo soddisfa i filtri Finviz.")
        else:
            df_finviz = add_formatted_cols(df_finviz)
            df_finviz = add_links(df_finviz)

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
                "EPS_nextY",
                "EPS_5Y",
                "Avg_Vol_mln",
                "Stato",
                "Yahoo",
                "Finviz",
            ]
            df_finviz = df_finviz[[c for c in cols_order if c in df_finviz.columns]]

            df_finviz_view = df_finviz.sort_values("MarketCap", ascending=False).head(top)

            df_finviz_show = df_finviz_view[
                [
                    c
                    for c in [
                        "Nome",
                        "Ticker",
                        "Prezzo_fmt",
                        "MarketCap_fmt",
                        "Vol_Today_fmt",
                        "Vol_7d_Avg_fmt",
                        "EPS_nextY",
                        "EPS_5Y",
                        "Avg_Vol_mln",
                        "Stato",
                        "Yahoo",
                        "Finviz",
                    ]
                    if c in df_finviz_view.columns
                ]
            ]

            st.dataframe(
                df_finviz_show,
                use_container_width=True,
                column_config={
                    "Prezzo_fmt": "Prezzo",
                    "MarketCap_fmt": "Market Cap",
                    "Vol_Today_fmt": "Vol giorno",
                    "Vol_7d_Avg_fmt": "Vol medio 7g",
                    "EPS_nextY": "EPS Next Y (%)",
                    "EPS_5Y": "EPS Next 5Y (%)",
                    "Avg_Vol_mln": "Avg Vol (mln)",
                    "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                    "Finviz": st.column_config.LinkColumn(
                        "TradingView", display_text="Apri"
                    ),
                },
            )

            # ==========================
            # EXPORT FINVIZ
            # ==========================
            csv_data = df_finviz_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Finviz CSV",
                data=csv_data,
                file_name=f"FINVIZ_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_finviz_csv",
            )

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_finviz_view.to_excel(writer, index=False, sheet_name="FINVIZ")
            data_xlsx = output.getvalue()

            st.download_button(
                "⬇️ Export Finviz XLSX",
                data=data_xlsx,
                file_name=f"FINVIZ_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                key="dl_finviz_xlsx",
            )

            # ==========================
            # Watchlist Finviz (con seleziona tutti) – key wl_finviz
            # ==========================
            options_finviz = sorted(
                f"{row['Nome']} – {row['Ticker']}"
                for _, row in df_finviz_view.iterrows()
            )

            col_sel_all_finviz, _ = st.columns([1, 3])
            with col_sel_all_finviz:
                if st.button("✅ Seleziona tutti (Top Finviz)", key="btn_sel_all_finviz"):
                    st.session_state["wl_finviz"] = options_finviz

            selection_finviz = st.multiselect(
                "Aggiungi alla Watchlist (Finviz):",
                options=options_finviz,
                key="wl_finviz",
            )

            note_finviz = st.text_input(
                "Note comuni per questi ticker Finviz", key="note_wl_finviz"
            )

            if st.button("📌 Salva in Watchlist (Finviz)"):
                tickers = [s.split(" – ")[1] for s in selection_finviz]
                names = [s.split(" – ")[0] for s in selection_finviz]
                add_to_watchlist(
                tickers, names, "FINVIZ", note_early, trend="LONG",
                list_name=st.session_state.get("current_list_name", "DEFAULT"),
)

                st.rerun()

# =============================================================================
# 📌 WATCHLIST & NOTE
# =============================================================================
with tab_watch:
    # titolo + pulsanti sulla stessa riga
    col_title, col_refresh, col_reset = st.columns([4, 1, 2])
    with col_title:
        st.subheader("📌 Watchlist & Note")
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_reset:
        current_list = st.session_state.get("current_list_name", "DEFAULT")
        if st.button(f"🧨 Reset lista '{current_list}'"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM watchlist WHERE list_name = ?", (current_list,))
            conn.commit()
            conn.close()
            st.success(f"Lista '{current_list}' azzerata.")
            st.rerun()
        if st.button("🧨 Reset DB (tutte le liste)"):
            reset_watchlist_db()
            st.success("Watchlist COMPLETAMENTE azzerata.")
            st.rerun()

    # carico il DB
    df_wl = load_watchlist()
    if df_wl.empty:
        st.caption("Watchlist vuota. Aggiungi titoli dai tab scanner.")
        st.stop()

    # ---------------- Filtro per lista multipla ----------------
    list_names = sorted(df_wl["list_name"].dropna().unique().tolist())
    list_names = [ln for ln in list_names if ln]
    if not list_names:
        list_names = ["DEFAULT"]

    list_filter = st.selectbox(
        "Lista watchlist:",
        options=list_names,
        index=list_names.index(
            st.session_state.get("current_list_name", list_names[0])
        ) if st.session_state.get("current_list_name") in list_names else 0,
        key="wl_filter_list",
    )

    df_wl = df_wl[df_wl["list_name"] == list_filter]
    if df_wl.empty:
        st.caption("Questa lista è vuota.")
        st.stop()

    # ---------------- Filtri rapidi origine / trend ----------------
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        origine_filter = st.selectbox(
            "Filtro origine:",
            options=["Tutte"] + sorted(df_wl["origine"].unique().tolist()),
            index=0,
            key="wl_filter_origine",
        )
    with col_f2:
        trend_filter = st.selectbox(
            "Filtro trend:",
            options=["Tutti"] + sorted(df_wl["trend"].unique().tolist()),
            index=0,
            key="wl_filter_trend",
        )

    df_wl_filt = df_wl.copy()
    if origine_filter != "Tutte":
        df_wl_filt = df_wl_filt[df_wl_filt["origine"] == origine_filter]
    if trend_filter != "Tutti":
        df_wl_filt = df_wl_filt[df_wl_filt["trend"] == trend_filter]

    if df_wl_filt.empty:
        st.caption("Nessuna riga corrisponde ai filtri selezionati.")
        st.stop()

    # ---------------- Dati di mercato da Yahoo per i ticker filtrati ----------------
    tickers_wl = df_wl_filt["ticker"].unique().tolist()
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
            market_cap = info.get("marketCap", np.nan)
            vol_today = float(vol.iloc[-1])
            vol_7d_avg = float(vol.mean())
            currency = info.get("currency", "USD")

            records_mkt.append(
                {
                    "ticker": tkr,
                    "Prezzo": price,
                    "MarketCap": market_cap,
                    "Vol_Today": vol_today,
                    "Vol_7d_Avg": vol_7d_avg,
                    "Currency": currency,
                }
            )
        except Exception:
            continue

    df_mkt = pd.DataFrame(records_mkt)

    if not df_mkt.empty:
        df_wl_filt = df_wl_filt.merge(df_mkt, on="ticker", how="left")

    # formatto prezzo / market cap / volumi e aggiungo link Yahoo + TradingView
    df_wl_filt = add_formatted_cols(df_wl_filt)
    df_wl_filt = add_links(df_wl_filt)

    # ---------------- Tabella principale watchlist ----------------
    df_wl_filt["label"] = (
        df_wl_filt["id"].astype(str)
        + " | "
        + df_wl_filt["name"].fillna("")
        + " | "
        + df_wl_filt["ticker"].fillna("")
    )

    df_wl_filt = df_wl_filt.sort_values(["name", "ticker"])

    cols_show = [
        "label",
        "trend",
        "origine",
        "note",
        "created_at",
    ]

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

    df_wl_view = df_wl_filt[cols_show]

    st.dataframe(
        df_wl_view,
        use_container_width=True,
        column_config={
            "label": "ID | Nome | Ticker",
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

    # ==========================
    # EXPORT WATCHLIST
    # ==========================
    csv_data = df_wl_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Watchlist CSV",
        data=csv_data,
        file_name=f"WATCHLIST_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_watch_csv",
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_wl_view.to_excel(writer, index=False, sheet_name="WATCHLIST")
    data_xlsx = output.getvalue()

    st.download_button(
        "⬇️ Export Watchlist XLSX",
        data=data_xlsx,
        file_name=f"WATCHLIST_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_watch_xlsx",
    )

    # EXPORT TradingView (solo ticker)
    tv_data = df_wl_filt["ticker"].drop_duplicates().to_frame(name="symbol")
    csv_tv = tv_data.to_csv(index=False, header=False).encode("utf-8")

    st.download_button(
        "⬇️ Export Watchlist TradingView (solo ticker)",
        data=csv_tv,
        file_name=f"TV_WATCHLIST_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_tv_watch",
    )

    st.markdown("---")

    # ---------------- Modifica note ----------------
    st.subheader("✏️ Modifica nota per una riga")

    id_options = df_wl_filt["id"].astype(str).tolist()
    labels = df_wl_filt["label"].tolist()
    id_map = dict(zip(labels, id_options))

    selected_row = st.selectbox(
        "Seleziona riga da modificare:",
        options=labels,
        key="wl_edit_row",
    )
    row_id = id_map[selected_row]

    current_note = df_wl_filt.loc[df_wl_filt["id"] == int(row_id), "note"].values[0]
    new_note = st.text_area("Nota", value=current_note, key="wl_edit_note")

    if st.button("💾 Salva nota"):
        update_watchlist_note(row_id, new_note)
        st.success("Nota aggiornata.")
        st.rerun()

    # ---------------- Eliminazione righe ----------------
    st.subheader("🗑️ Elimina righe dalla Watchlist")

    del_rows = st.multiselect(
        "Seleziona righe da eliminare:",
        options=labels,
        key="wl_delete_rows",
    )
    del_ids = [id_map[label] for label in del_rows]

    col_del1, col_del2 = st.columns(2)
    with col_del1:
        if st.button("❌ Elimina selezionate"):
            delete_from_watchlist(del_ids)
            st.success("Righe eliminate dalla watchlist.")
            st.rerun()
