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
    page_title="Trading Scanner – Versione PRO 8.0",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Trading Scanner – Versione PRO 8.0")
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
    if "Currency" not in df.columns:
        df["Currency"] = "USD"
    df["Prezzo_fmt"] = df.apply(
        lambda r: fmt_currency(r["Prezzo"], "€" if r["Currency"] == "EUR" else "$"),
        axis=1,
    )
    df["MarketCap_fmt"] = df.apply(
        lambda r: fmt_marketcap(r["MarketCap"], "€" if r["Currency"] == "EUR" else "$"),
        axis=1,
    )
    df["Vol_Today_fmt"] = df["Vol_Today"].apply(fmt_int)
    df["Vol_7d_Avg_fmt"] = df["Vol_7d_Avg"].apply(fmt_int)
    return df


def add_links(df):
    df["Yahoo"] = df["Ticker"].apply(
        lambda t: f"https://finance.yahoo.com/quote/{t}"
    )
    df["Finviz"] = df["Ticker"].apply(
        lambda t: f"https://finviz.com/quote.ashx?t={t.split('.')[0]}"
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
            created_at TEXT
        )
        """
    )
    # Migrazione: aggiungi colonna trend se DB vecchio
    try:
        c.execute("ALTER TABLE watchlist ADD COLUMN trend TEXT")
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

def add_to_watchlist(tickers, names, origine, note, trend="LONG"):
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute(
            "INSERT INTO watchlist (ticker, name, trend, origine, note, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (t, n, trend, origine, note, now),
        )
    conn.commit()
    conn.close()

def load_watchlist():
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["id", "ticker", "name", "trend", "origine", "note", "created_at"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    for col in ["ticker", "name", "trend", "origine", "note", "created_at"]:
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

        rea_score = 7 if (dist_poc < r_poc and vol_ratio > 1.5) else 0
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
            "Nome", "Ticker",
            "Prezzo", "Prezzo_fmt",
            "MarketCap", "MarketCap_fmt",
            "Vol_Today", "Vol_Today_fmt",
            "Vol_7d_Avg", "Vol_7d_Avg_fmt",
            "Early_Score", "Pro_Score",
            "RSI", "Vol_Ratio", "OBV_Trend", "ATR", "ATR_Exp", "Stato",
            "Yahoo", "Finviz",
        ]
        df_early = df_early[[c for c in cols_order if c in df_early.columns]]

        df_early_view = df_early.sort_values("Early_Score", ascending=False).head(top)

        # Tabella: uso solo le colonne formattate + pulsanti link
        df_early_show = df_early_view[[
            "Nome", "Ticker",
            "Prezzo_fmt", "MarketCap_fmt",
            "Vol_Today_fmt", "Vol_7d_Avg_fmt",
            "Early_Score", "Pro_Score",
            "RSI", "Vol_Ratio", "OBV_Trend", "ATR", "ATR_Exp", "Stato",
            "Yahoo", "Finviz",
        ]]

        st.dataframe(
            df_early_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
            },
        )

        # CSV per TradingView: uso la colonna Prezzo numerica originale
        df_early_tv = df_early_view.rename(
            columns={
                "Ticker": "symbol",
                "Prezzo": "price",
                "RSI": "rsi",
                "Vol_Ratio": "volume_ratio",
            }
        )[["symbol", "price", "rsi", "volume_ratio"]]
        csv_early = df_early_tv.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ CSV EARLY per TradingView",
            data=csv_early,
            file_name=f"signals_early_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        options_early = [
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_early_view.iterrows()
        ]
        selection_early = st.multiselect(
            "Aggiungi alla Watchlist (EARLY):",
            options=options_early,
            key="wl_early",
        )
        note_early = st.text_input("Note comuni per questi ticker EARLY", key="note_wl_early")
        if st.button("📌 Salva in Watchlist (EARLY)"):
            tickers = [s.split(" – ")[1] for s in selection_early]
            names   = [s.split(" – ")[0] for s in selection_early]
            add_to_watchlist(tickers, names, "EARLY", note_early, trend="LONG")
            st.success("EARLY salvati in watchlist.")
            st.rerun()

# =============================================================================
# PRO
# =============================================================================
with tab_p:
    st.subheader("🟣 Segnali PRO")
    st.markdown(
        f"Filtro PRO: titoli con **Stato = PRO** (prezzo sopra EMA20, RSI tra {p_rmin} e {p_rmax}, "
        "Vol_Ratio > 1.2, Pro_Score elevato)."
    )

    with st.expander("📘 Legenda PRO"):
        st.markdown(
            "- **Pro_Score**: punteggio composito (prezzo sopra EMA20, RSI nel range, volume sopra media).\n"
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
            "Nome", "Ticker",
            "Prezzo", "Prezzo_fmt",
            "MarketCap", "MarketCap_fmt",
            "Vol_Today", "Vol_Today_fmt",
            "Vol_7d_Avg", "Vol_7d_Avg_fmt",
            "Early_Score", "Pro_Score",
            "RSI", "Vol_Ratio", "OBV_Trend", "ATR", "ATR_Exp", "Stato",
            "Yahoo", "Finviz",
        ]
        df_pro = df_pro[[c for c in cols_order if c in df_pro.columns]]

        df_pro_view = df_pro.sort_values("Pro_Score", ascending=False).head(top)
        df_pro_view["OBV_Trend"] = df_pro_view["OBV_Trend"].replace(
            {"UP": "UP (flusso in ingresso)", "DOWN": "DOWN (flusso in uscita)"}
        )

        df_pro_show = df_pro_view[[
            "Nome", "Ticker",
            "Prezzo_fmt", "MarketCap_fmt",
            "Vol_Today_fmt", "Vol_7d_Avg_fmt",
            "Early_Score", "Pro_Score",
            "RSI", "Vol_Ratio", "OBV_Trend", "ATR", "ATR_Exp", "Stato",
            "Yahoo", "Finviz",
        ]]

        st.dataframe(
            df_pro_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
            },
        )

        df_pro_tv = df_pro_view.rename(
            columns={
                "Ticker": "symbol",
                "Prezzo": "price",
                "RSI": "rsi",
                "Vol_Ratio": "volume_ratio",
                "OBV_Trend": "obv_trend",
            }
        )[["symbol", "price", "rsi", "volume_ratio", "obv_trend"]]
        csv_pro = df_pro_tv.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ CSV PRO per TradingView",
            data=csv_pro,
            file_name=f"signals_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        options_pro = [
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_pro_view.iterrows()
        ]
        selection_pro = st.multiselect(
            "Aggiungi alla Watchlist (PRO):",
            options=options_pro,
            key="wl_pro",
        )
        note_pro = st.text_input("Note comuni per questi ticker PRO", key="note_wl_pro")
        if st.button("📌 Salva in Watchlist (PRO)"):
            tickers = [s.split(" – ")[1] for s in selection_pro]
            names   = [s.split(" – ")[0] for s in selection_pro]
            add_to_watchlist(tickers, names, "PRO", note_pro, trend="LONG")
            st.success("PRO salvati in watchlist.")
            st.rerun()

# =============================================================================
# REA‑QUANT (segnali)
# =============================================================================
with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT")
    st.markdown(
        f"Filtro REA‑QUANT: titoli con **Stato = HOT** "
        f"(distanza dal POC < {r_poc*100:.1f}%, Vol_Ratio > 1.5)."
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
            "Nome", "Ticker",
            "Prezzo", "Prezzo_fmt",
            "MarketCap", "MarketCap_fmt",
            "Vol_Today", "Vol_Today_fmt",
            "Vol_7d_Avg", "Vol_7d_Avg_fmt",
            "Rea_Score", "POC", "Dist_POC_%", "Vol_Ratio", "Stato",
            "Yahoo", "Finviz",
        ]
        df_rea = df_rea[[c for c in cols_order if c in df_rea.columns]]

        df_rea_view = df_rea.sort_values("Rea_Score", ascending=False).head(top)

        df_rea_show = df_rea_view[[
            "Nome", "Ticker",
            "Prezzo_fmt", "MarketCap_fmt",
            "Vol_Today_fmt", "Vol_7d_Avg_fmt",
            "Rea_Score", "POC", "Dist_POC_%", "Vol_Ratio", "Stato",
            "Yahoo", "Finviz",
        ]]

        st.dataframe(
            df_rea_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
            },
        )

        df_rea_tv = df_rea_view.rename(
            columns={
                "Ticker": "symbol",
                "Prezzo": "price",
                "POC": "poc",
                "Dist_POC_%": "dist_poc_percent",
                "Vol_Ratio": "volume_ratio",
            }
        )[["symbol", "price", "poc", "dist_poc_percent", "volume_ratio"]]
        csv_rea = df_rea_tv.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ CSV REA‑QUANT per TradingView",
            data=csv_rea,
            file_name=f"signals_rea_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        options_rea = [
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_rea_view.iterrows()
        ]
        selection_rea = st.multiselect(
            "Aggiungi alla Watchlist (REA‑QUANT HOT):",
            options=options_rea,
            key="wl_rea",
        )
        note_rea = st.text_input("Note comuni per questi ticker REA‑QUANT", key="note_wl_rea")
        if st.button("📌 Salva in Watchlist (REA‑QUANT)"):
            tickers = [s.split(" – ")[1] for s in selection_rea]
            names   = [s.split(" – ")[0] for s in selection_rea]
            add_to_watchlist(tickers, names, "REA_HOT", note_rea, trend="LONG")
            st.success("REA‑QUANT salvati in watchlist.")
            st.rerun()


# =============================================================================
# MASSIMO REA – ANALISI QUANT
# =============================================================================
with tab_rea_q:
    st.subheader("🧮 Analisi Quantitativa stile Massimo Rea")
    st.markdown(
        "Analisi per mercato sui soli titoli con **Stato = HOT**: "
        "conteggio segnali, Vol_Ratio medio, Rea_Score medio e top 10 per pressione volumetrica."
    )

    with st.expander("📘 Legenda Rea Quant (analisi)"):
        st.markdown(
            "- **N**: numero di titoli HOT per mercato.\n"
            "- **Vol_Ratio_med**: media Vol_Ratio.\n"
            "- **Rea_Score_med**: intensità media segnale.\n"
            "- **MarketCap / Volumi**: medie indicative per mercato.\n"
            "- Top 10: ordinati per Vol_Ratio con link Yahoo/Finviz."
        )

    if df_rea_all.empty:
        st.caption("Nessun dato REA‑QUANT disponibile.")
        df_rea_q = pd.DataFrame()
    else:
        df_rea_q = df_rea_all.copy()

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

        # Top 10 per pressione volumetrica
        st.markdown("**Top 10 per pressione volumetrica (Vol_Ratio)**")
        df_rea_top = df_rea_q.sort_values("Vol_Ratio", ascending=False).head(10)
        df_rea_top = add_formatted_cols(df_rea_top)
        df_rea_top = add_links(df_rea_top)

        df_rea_top_show = df_rea_top[[
            "Nome", "Ticker",
            "Prezzo_fmt", "MarketCap_fmt",
            "Vol_Today_fmt", "Vol_7d_Avg_fmt",
            "POC", "Dist_POC_%", "Vol_Ratio", "Stato",
            "Yahoo", "Finviz",
        ]]

        st.dataframe(
            df_rea_top_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
            },
        )

        options_rea_q = [
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_rea_top.iterrows()
        ]
        selection_rea_q = st.multiselect(
            "Aggiungi alla Watchlist (Rea Quant Top10):",
            options=options_rea_q,
            key="wl_rea_q",
        )
        note_rea_q = st.text_input("Note comuni per questi ticker (Rea Quant)", key="note_wl_rea_q")
        if st.button("📌 Salva in Watchlist (Rea Quant)"):
            tickers = [s.split(" – ")[1] for s in selection_rea_q]
            names   = [s.split(" – ")[0] for s in selection_rea_q]
            add_to_watchlist(tickers, names, "REA_QUANT", note_rea_q, trend="LONG")
            st.success("Rea Quant salvati in watchlist.")
            st.rerun()

# =============================================================================
# STEFANO SERAFINI – SYSTEMS
# =============================================================================
with tab_serafini:
    st.subheader("📈 Approccio Trend‑Following stile Stefano Serafini")
    st.markdown(
        "Sistema Donchian‑style su 20 giorni: breakout su massimi/minimi 20‑giorni "
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
        df_break_view = pd.DataFrame()
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

                records.append({
                    "Ticker": tkr,
                    "Prezzo": round(last, 2),
                    "Hi20": round(high20.iloc[-2], 2),
                    "Lo20": round(low20.iloc[-2], 2),
                    "Breakout_Up": breakout_up,
                    "Breakout_Down": breakout_down,
                })
            except Exception:
                continue

        df_break = pd.DataFrame(records)
        if df_break.empty:
            st.caption("Nessun breakout rilevato (20 giorni).")
            df_break_view = pd.DataFrame()
        else:
            df_break = df_break.merge(
                df_ep[[
                    "Ticker", "Nome", "Pro_Score", "RSI", "Vol_Ratio",
                    "MarketCap", "Vol_Today", "Vol_7d_Avg", "Currency"
                ]],
                on="Ticker",
                how="left"
            )

            df_break = add_formatted_cols(df_break)
            df_break = add_links(df_break)

            cols_order = [
                "Nome", "Ticker",
                "Prezzo_fmt", "MarketCap_fmt",
                "Vol_Today_fmt", "Vol_7d_Avg_fmt",
                "Hi20", "Lo20",
                "Breakout_Up", "Breakout_Down",
                "Pro_Score", "RSI", "Vol_Ratio",
                "Yahoo", "Finviz",
            ]
            df_break = df_break[[c for c in cols_order if c in df_break.columns]]

            st.markdown("**Breakout su massimi/minimi 20 giorni (Donchian style)**")
            df_break_view = df_break[
                (df_break["Breakout_Up"]) | (df_break["Breakout_Down"])
            ].sort_values("Pro_Score", ascending=False)

            st.dataframe(
                df_break_view,
                use_container_width=True,
                column_config={
                    "Prezzo_fmt": "Prezzo",
                    "MarketCap_fmt": "Market Cap",
                    "Vol_Today_fmt": "Vol giorno",
                    "Vol_7d_Avg_fmt": "Vol medio 7g",
                    "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                    "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
                },
            )

            options_seraf = [
                f"{row['Nome']} – {row['Ticker']}" for _, row in df_break_view.iterrows()
            ]
            selection_seraf = st.multiselect(
                "Aggiungi alla Watchlist (Serafini Systems):",
                options=options_seraf,
                key="wl_seraf",
            )
            note_seraf = st.text_input("Note comuni per questi ticker Serafini", key="note_wl_seraf")
            if st.button("📌 Salva in Watchlist (Serafini)"):
                tickers = [s.split(" – ")[1] for s in selection_seraf]
                names   = [s.split(" – ")[0] for s in selection_seraf]
                add_to_watchlist(tickers, names, "SERAFINI", note_seraf, trend="LONG")
                st.success("Serafini salvati in watchlist.")
                st.rerun()


# =============================================================================
# REGIME & MOMENTUM
# =============================================================================
with tab_regime:
    st.subheader("🧊 Regime & Momentum multi‑mercato")
    st.markdown(
        "Regime: % PRO vs EARLY sul totale segnali. "
        "Momentum: ranking per Pro_Score × 10 + RSI su tutti i titoli scansionati."
    )

    with st.expander("📘 Legenda Regime & Momentum"):
        st.markdown(
            "- **Regime**: quota segnali PRO vs EARLY.\n"
            "- **Momentum**: Pro_Score×10 + RSI.\n"
            "- **MarketCap / Volumi**: per contestualizzare i top momentum.\n"
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
        )

    if df_ep.empty or "Stato" not in df_ep.columns:
        st.caption("Nessun dato scanner disponibile.")
    else:
        df_all = df_ep.copy()
        n_tot_signals = len(df_all)
        n_pro_tot = (df_all["Stato"] == "PRO").sum()
        n_early_tot = (df_all["Stato"] == "EARLY").sum()

        c1r, c2r, c3r = st.columns(3)
        c1r.metric("Totale segnali (EARLY+PRO)", n_tot_signals)
        c2r.metric("% PRO", f"{(n_pro_tot / n_tot_signals * 100):.1f}%" if n_tot_signals else "0.0%")
        c3r.metric("% EARLY", f"{(n_early_tot / n_tot_signals * 100):.1f}%" if n_tot_signals else "0.0%")

        st.markdown("**Top 10 momentum (Pro_Score + RSI)**")
        df_all["Momentum"] = df_all["Pro_Score"] * 10 + df_all["RSI"]
        df_mom = df_all.sort_values("Momentum", ascending=False).head(10)

        cols_order = [
            "Nome", "Ticker", "Prezzo",
            "MarketCap", "Vol_Today", "Vol_7d_Avg",
            "Pro_Score", "RSI",
            "Vol_Ratio", "OBV_Trend", "ATR", "Stato", "Momentum"
        ]
        df_mom = df_mom[[c for c in cols_order if c in df_mom.columns]]
        df_mom = add_formatted_cols(df_mom)
        df_mom = add_links(df_mom)

        df_mom_show = df_mom[[
            "Nome", "Ticker",
            "Prezzo_fmt", "MarketCap_fmt",
            "Vol_Today_fmt", "Vol_7d_Avg_fmt",
            "Pro_Score", "RSI",
            "Vol_Ratio", "OBV_Trend", "ATR", "Stato", "Momentum",
            "Yahoo", "Finviz",
        ]]

        st.dataframe(
            df_mom_show,
            use_container_width=True,
            column_config={
                "Prezzo_fmt": "Prezzo",
                "MarketCap_fmt": "Market Cap",
                "Vol_Today_fmt": "Vol giorno",
                "Vol_7d_Avg_fmt": "Vol medio 7g",
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
            },
        )

        df_mom_tv = df_mom[["Ticker"]].rename(columns={"Ticker": "symbol"})
        csv_mom = df_mom_tv.to_csv(index=False, header=False).encode("utf-8")
        st.download_button(
            "⬇️ CSV Top Momentum (solo ticker)",
            data=csv_mom,
            file_name=f"signals_momentum_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
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

        st.markdown("**Sintesi Regime & Momentum per mercato (tabella)**")
        if not heat.empty:
            st.dataframe(heat.sort_values("Momentum_med", ascending=False), use_container_width=True)
        else:
            st.caption("Nessun dato sufficiente per la sintesi per mercato.")

        options_regime = [
            f"{row['Nome']} – {row['Ticker']}" for _, row in df_mom.iterrows()
        ]
        selection_regime = st.multiselect(
            "Aggiungi alla Watchlist (Top Momentum):",
            options=options_regime,
            key="wl_regime",
        )
        note_regime = st.text_input("Note comuni per questi ticker Momentum", key="note_wl_regime")
        if st.button("📌 Salva in Watchlist (Regime/Momentum)"):
            tickers = [s.split(" – ")[1] for s in selection_regime]
            names   = [s.split(" – ")[0] for s in selection_regime]
            add_to_watchlist(tickers, names, "REGIME_MOMENTUM", note_regime, trend="LONG")
            st.success("Regime/Momentum salvati in watchlist.")
            st.rerun()

# =============================================================================
# MULTI‑TIMEFRAME – RSI 1D/1W/1M
# =============================================================================
with tab_mtf:
    st.subheader("🕒 Analisi Multi‑Timeframe (RSI 1D / 1W / 1M)")
    st.markdown(
        "Analisi RSI su tre timeframe (daily, weekly, monthly) per tutti i titoli scansionati, "
        "con segnale sintetico ALIGN_LONG / ALIGN_SHORT / MIXED."
    )

    with st.expander("📘 Legenda Multi‑Timeframe"):
        st.markdown(
            "- **RSI_1D / RSI_1W / RSI_1M**: RSI(14) su TF giornaliero, settimanale e mensile.\n"
            "- **MTF_Score**: media dei tre RSI.\n"
            "- **MarketCap / Volumi**: dati di contesto per selezionare titoli più liquidi.\n"
            "- **Segnale_MTF**: ALIGN_LONG, ALIGN_SHORT o MIXED.\n"
            "- Colonne **Yahoo** e **Finviz**: pulsanti link per ogni ticker."
        )

    if df_ep.empty:
        st.caption("Nessun dato base disponibile per il Multi‑Timeframe.")
        df_mtf = pd.DataFrame()
    else:
        tickers_all = df_ep["Ticker"].unique().tolist()

        @st.cache_data(ttl=1800)
        def fetch_mtf_data(tickers):
            records = []
            for tkr in tickers:
                try:
                    yt = yf.Ticker(tkr)
                    d_daily = yt.history(period="6mo", interval="1d")
                    d_week  = yt.history(period="2y",  interval="1wk")
                    d_month = yt.history(period="5y",  interval="1mo")

                    def rsi_from_close(close, period=14):
                        if len(close) < period + 1:
                            return np.nan
                        delta = close.diff()
                        gain = delta.where(delta > 0, 0).rolling(period).mean()
                        loss = -delta.where(delta < 0, 0).rolling(period).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        return float(rsi.iloc[-1])

                    rsi_1d = rsi_from_close(d_daily["Close"]) if not d_daily.empty else np.nan
                    rsi_1w = rsi_from_close(d_week["Close"])  if not d_week.empty  else np.nan
                    rsi_1m = rsi_from_close(d_month["Close"]) if not d_month.empty else np.nan

                    if np.isnan(rsi_1d) and np.isnan(rsi_1w) and np.isnan(rsi_1m):
                        continue

                    mtf_score = np.nanmean([rsi_1d, rsi_1w, rsi_1m])

                    signal = "MIXED"
                    if (rsi_1d > 50) and (rsi_1w > 50) and (rsi_1m > 50):
                        signal = "ALIGN_LONG"
                    elif (rsi_1d < 50) and (rsi_1w < 50) and (rsi_1m < 50):
                        signal = "ALIGN_SHORT"

                    records.append({
                        "Ticker": tkr,
                        "RSI_1D": round(rsi_1d, 1) if not np.isnan(rsi_1d) else np.nan,
                        "RSI_1W": round(rsi_1w, 1) if not np.isnan(rsi_1w) else np.nan,
                        "RSI_1M": round(rsi_1m, 1) if not np.isnan(rsi_1m) else np.nan,
                        "MTF_Score": round(mtf_score, 1) if not np.isnan(mtf_score) else np.nan,
                        "Segnale_MTF": signal,
                    })
                except Exception:
                    continue

            return pd.DataFrame(records)

        with st.spinner("Calcolo RSI multi‑timeframe in corso..."):
            df_mtf = fetch_mtf_data(tickers_all)

        if df_mtf.empty:
            st.caption("Nessun dato Multi‑Timeframe disponibile.")
        else:
            df_mtf = df_mtf.merge(
                df_ep[[
                    "Ticker", "Nome", "Pro_Score", "Stato",
                    "MarketCap", "Vol_Today", "Vol_7d_Avg", "Currency"
                ]],
                on="Ticker",
                how="left"
            ).drop_duplicates(subset=["Ticker"])

            cols_order = [
                "Nome", "Ticker",
                "Prezzo",          # se non c'è, verrà ignorato
                "MarketCap", "Vol_Today", "Vol_7d_Avg",
                "RSI_1D", "RSI_1W", "RSI_1M",
                "MTF_Score", "Segnale_MTF", "Pro_Score", "Stato"
            ]
            df_mtf = df_mtf[[c for c in cols_order if c in df_mtf.columns]]
            df_mtf = add_formatted_cols(df_mtf)
            df_mtf = add_links(df_mtf)

            st.markdown("**Top 30 per MTF_Score (allineamento forza RSI multi‑TF)**")
            if "MTF_Score" in df_mtf.columns:
                df_mtf_view = df_mtf.sort_values("MTF_Score", ascending=False).head(30)
            else:
                df_mtf_view = df_mtf.head(30)

            df_mtf_show = df_mtf_view[[
                "Nome", "Ticker",
                "Prezzo_fmt" if "Prezzo_fmt" in df_mtf_view.columns else "Prezzo",
                "MarketCap_fmt",
                "Vol_Today_fmt", "Vol_7d_Avg_fmt",
                "RSI_1D", "RSI_1W", "RSI_1M",
                "MTF_Score", "Segnale_MTF", "Pro_Score", "Stato",
                "Yahoo", "Finviz",
            ]]

            # rinomino la colonna prezzo nel caso usi Prezzo_fmt
            df_mtf_show = df_mtf_show.rename(
                columns={"Prezzo_fmt": "Prezzo"}
            )

            st.dataframe(
                df_mtf_show,
                use_container_width=True,
                column_config={
                    "Prezzo": "Prezzo",
                    "MarketCap_fmt": "Market Cap",
                    "Vol_Today_fmt": "Vol giorno",
                    "Vol_7d_Avg_fmt": "Vol medio 7g",
                    "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                    "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
                },
            )

# =============================================================================
# TAB FINVIZ – FILTRI LIKE FINVIZ
# =============================================================================
with tab_finviz:
    st.subheader("📊 Segnali stile Finviz")

    with st.expander("📘 Legenda Filtri Finviz‑like"):
        st.markdown(
            "- **EPS Growth Next Year > 10%** (proxy: earningsGrowth di Yahoo Finance > 0.10).\n"
            "- **EPS Growth Next 5 Years > 15%** (proxy: earningsQuarterlyGrowth > 0.15).\n"
            "- **Average Volume > 1.000K**.\n"
            "- **Options – Short available** (optionable = True).\n"
            "- **Price > 10$**.\n"
            "- **Relative Volume > 1** (vol corrente / avg vol > 1).\n"
            "- **Price above SMA20 / SMA50 / SMA200** calcolate sui close giornalieri.\n"
            "- Colonne **Prezzo / Market Cap / Volumi** formattate e link **Yahoo/Finviz** per ogni ticker."
        )

    if df_ep.empty:
        st.caption("Nessun dato base disponibile per il filtro Finviz.")
    else:
        tickers_all = df_ep["Ticker"].unique().tolist()

        @st.cache_data(ttl=3600)
        def fetch_finviz_like_info(tickers):
            recs = []
            for tkr in tickers:
                try:
                    yt = yf.Ticker(tkr)
                    info = yt.info

                    price = info.get("currentPrice") or info.get("regularMarketPrice")
                    if price is None:
                        price = np.nan

                    eps_next_y = info.get("earningsGrowth")
                    eps_next_5y = info.get("earningsQuarterlyGrowth")
                    avg_vol = info.get("averageVolume")
                    rel_vol = info.get("regularMarketVolume") / avg_vol if avg_vol and avg_vol > 0 else np.nan
                    optionable = info.get("optionable", False)

                    hist = yt.history(period="260d")
                    if hist.empty:
                        sma20 = sma50 = sma200 = np.nan
                    else:
                        close = hist["Close"]
                        sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
                        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
                        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

                    recs.append({
                        "Ticker": tkr,
                        "Price": price,
                        "EPS_NextY": eps_next_y,
                        "EPS_Next5Y": eps_next_5y,
                        "AvgVolume": avg_vol,
                        "RelVolume": rel_vol,
                        "Optionable": optionable,
                        "SMA20": sma20,
                        "SMA50": sma50,
                        "SMA200": sma200,
                    })
                except Exception:
                    continue
            return pd.DataFrame(recs)

        with st.spinner("Calcolo filtri Finviz‑like su universo scansionato..."):
            df_fund = fetch_finviz_like_info(tickers_all)

        if df_fund.empty:
            st.caption("Impossibile calcolare i filtri Finviz‑like (dati insufficienti).")
        else:
            df_finviz = df_fund.merge(
                df_ep[[
                    "Ticker", "Nome", "Prezzo", "Pro_Score", "RSI", "Vol_Ratio", "Stato",
                    "MarketCap", "Vol_Today", "Vol_7d_Avg", "Currency"
                ]],
                on="Ticker",
                how="left",
            )

            # Filtri Finviz-like (puoi parametrizzarli da sidebar se vuoi)
            cond_price = df_finviz["Price"] > 10
            cond_eps_y = df_finviz["EPS_NextY"] > 0.10
            cond_eps_5y = df_finviz["EPS_Next5Y"] > 0.15
            cond_avg_vol = df_finviz["AvgVolume"] > 1_000_000
            cond_option = df_finviz["Optionable"] == True
            cond_rel_vol = df_finviz["RelVolume"] > 1
            cond_sma20 = df_finviz["Price"] > df_finviz["SMA20"]
            cond_sma50 = df_finviz["Price"] > df_finviz["SMA50"]
            cond_sma200 = df_finviz["Price"] > df_finviz["SMA200"]

            df_finviz_sel = df_finviz[
                cond_price &
                cond_eps_y &
                cond_eps_5y &
                cond_avg_vol &
                cond_option &
                cond_rel_vol &
                cond_sma20 &
                cond_sma50 &
                cond_sma200
            ].copy()

            if df_finviz_sel.empty:
                st.caption("Nessun titolo soddisfa tutti i filtri Finviz‑like.")
            else:
                st.markdown("**Titoli che soddisfano tutti i filtri Finviz‑like**")
                cols_order = [
                    "Nome", "Ticker",
                    "Price", "Prezzo",
                    "MarketCap", "Vol_Today", "Vol_7d_Avg",
                    "Pro_Score", "RSI", "Vol_Ratio",
                    "EPS_NextY", "EPS_Next5Y",
                    "AvgVolume", "RelVolume",
                    "SMA20", "SMA50", "SMA200", "Stato"
                ]
                df_finviz_sel = df_finviz_sel[[c for c in cols_order if c in df_finviz_sel.columns]]
                df_finviz_sel = df_finviz_sel.sort_values("Pro_Score", ascending=False)

                # aggiungo formattazione e link
                df_finviz_sel = add_formatted_cols(df_finviz_sel)
                df_finviz_sel = add_links(df_finviz_sel)

                df_finviz_show = df_finviz_sel.head(top)[[
                    "Nome", "Ticker",
                    "Prezzo_fmt", "MarketCap_fmt",
                    "Vol_Today_fmt", "Vol_7d_Avg_fmt",
                    "Pro_Score", "RSI", "Vol_Ratio",
                    "EPS_NextY", "EPS_Next5Y",
                    "AvgVolume", "RelVolume",
                    "SMA20", "SMA50", "SMA200", "Stato",
                    "Yahoo", "Finviz",
                ]]

                st.dataframe(
                    df_finviz_show,
                    use_container_width=True,
                    column_config={
                        "Prezzo_fmt": "Prezzo",
                        "MarketCap_fmt": "Market Cap",
                        "Vol_Today_fmt": "Vol giorno",
                        "Vol_7d_Avg_fmt": "Vol medio 7g",
                        "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Apri"),
                        "Finviz": st.column_config.LinkColumn("Finviz", display_text="Apri"),
                    },
                )

                # CSV Finviz (symbol, price numerico)
                df_finviz_tv = df_finviz_sel.rename(
                    columns={
                        "Ticker": "symbol",
                        "Price": "price",
                    }
                )[["symbol", "price"]]
                csv_finviz = df_finviz_tv.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ CSV Finviz‑like (symbol, price)",
                    data=csv_finviz,
                    file_name=f"signals_finviz_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                options_finviz = [
                    f"{row['Nome']} – {row['Ticker']}" for _, row in df_finviz_sel.head(top).iterrows()
                ]
                note_finviz = st.text_input(
                    "Note comuni per questi ticker Finviz‑like",
                    value="Preset Finviz EPS/Vol/MA",
                    key="note_wl_finviz"
                )
                selection_finviz = st.multiselect(
                    "Aggiungi alla Watchlist (Finviz‑like):",
                    options=options_finviz,
                    key="wl_finviz",
                )
                if st.button("📌 Salva in Watchlist (Finviz‑like)"):
                    tickers = [s.split(" – ")[1] for s in selection_finviz]
                    names   = [s.split(" – ")[0] for s in selection_finviz]
                    add_to_watchlist(tickers, names, "FINVIZ_LIKE", note_finviz, trend="LONG")
                    st.success("Titoli Finviz‑like salvati in watchlist.")
                    st.rerun()

# =============================================================================
# 📌 WATCHLIST & NOTE
# =============================================================================
with tab_watch:
    st.subheader("📌 Watchlist & Note")
    st.markdown(
        "Gestisci la watchlist centralizzata: Nome, Ticker, Trend (LONG/SHORT), "
        "più origine, note e data."
    )

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        if st.button("🔄 Refresh Watchlist"):
            st.rerun()
    with col_w2:
        if st.button("🧨 Elimina intero DB Watchlist"):
            reset_watchlist_db()
            st.success("Watchlist azzerata. DB ricreato da zero.")
            st.rerun()

    df_watch = load_watchlist()

    if df_watch.empty:
        st.caption("Watchlist vuota. Aggiungi ticker dalle varie tab dello scanner.")
    else:
        st.dataframe(
            df_watch[["id", "name", "ticker", "trend", "origine", "note", "created_at"]],
            use_container_width=True,
        )

        ids_to_delete = st.multiselect(
            "Seleziona ID da eliminare dalla watchlist:",
            options=df_watch["id"].tolist(),
            format_func=lambda x: (
                f"ID {x} – {df_watch[df_watch['id'] == x]['name'].iloc[0]} "
                f"({df_watch[df_watch['id'] == x]['ticker'].iloc[0]})"
            ),
            key="wl_delete_ids",
        )
        if st.button("🗑️ Elimina selezionati"):
            delete_from_watchlist(ids_to_delete)
            st.success("Righe eliminate dalla watchlist.")
            st.rerun()

        st.markdown("**Modifica nota per singolo elemento**")
        id_edit = st.selectbox(
            "Seleziona ID per modificare la nota:",
            options=[None] + df_watch["id"].tolist(),
            format_func=lambda x: (
                "—"
                if x is None
                else f"ID {x} – {df_watch[df_watch['id'] == x]['name'].iloc[0]} "
                     f"({df_watch[df_watch['id'] == x]['ticker'].iloc[0]})"
            ),
            key="wl_edit_id",
        )
        if id_edit is not None:
            old_note = df_watch.loc[df_watch["id"] == id_edit, "note"].iloc[0]
            new_note = st.text_area("Nuova nota:", value=old_note, key="wl_edit_note")
            if st.button("💾 Salva nota aggiornata"):
                update_watchlist_note(id_edit, new_note)
                st.success("Nota aggiornata.")
                st.rerun()

        csv_watch = df_watch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Esporta Watchlist in CSV",
            data=csv_watch,
            file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        def build_watchlist_pdf(df: pd.DataFrame) -> bytes:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Watchlist – Trading Scanner PRO 8.0", ln=True)

            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 8, f"Generato il: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(4)

            pdf.set_font("Arial", "B", 9)
            pdf.cell(20, 8, "Ticker", 1)
            pdf.cell(50, 8, "Nome", 1)
            pdf.cell(20, 8, "Trend", 1)
            pdf.cell(25, 8, "Origine", 1)
            pdf.cell(75, 8, "Note", 1)
            pdf.ln(8)

            pdf.set_font("Arial", "", 8)
            for _, row in df.iterrows():
                pdf.cell(20, 6, str(row["ticker"])[:12], 1)
                pdf.cell(50, 6, str(row["name"])[:28], 1)
                pdf.cell(20, 6, str(row["trend"])[:10], 1)
                pdf.cell(25, 6, str(row["origine"])[:12], 1)
                note_txt = (str(row["note"])[:70]) if row["note"] else ""
                pdf.cell(75, 6, note_txt, 1)
                pdf.ln(6)

            pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
            return pdf_bytes

        if st.button("📄 Genera PDF Watchlist"):
            pdf_bytes = build_watchlist_pdf(df_watch)
            st.download_button(
                "⬇️ Download PDF Watchlist",
                data=pdf_bytes,
                file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
