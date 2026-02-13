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
    page_title="Trading Scanner – Versione PRO 6.0",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Trading Scanner – Versione PRO 6.0")
st.caption(
    "EARLY • PRO • REA‑QUANT • Rea Quant • Serafini • Regime & Momentum • "
    "Multi‑Timeframe • Risk & Portfolio • Export TradingView • Watchlist DB"
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

        info = yf.Ticker(ticker).info
        name = info.get("longName", info.get("shortName", ticker))[:50]

        price = float(c.iloc[-1])
        ema20 = float(c.ewm(20).mean().iloc[-1])

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

if st.button("🚀 AVVIA SCANNER PRO 6.0", type="primary", use_container_width=True):
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

    st.experimental_rerun()

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
tab_e, tab_p, tab_r, tab_rea_q, tab_serafini, tab_regime, tab_mtf, tab_risk, tab_watch = st.tabs(
    [
        "🟢 EARLY",
        "🟣 PRO",
        "🟠 REA‑QUANT",
        "🧮 Rea Quant",
        "📈 Serafini Systems",
        "🧊 Regime & Momentum",
        "🕒 Multi‑Timeframe",
        "💼 Risk & Portfolio",
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
            "- **Vol_Ratio**: volume odierno / media 20 giorni; >1 = volume sopra media.\n"
            "- **Stato = EARLY**: setup in formazione vicino alla media."
        )

    if df_early_all.empty:
        st.caption("Nessun segnale EARLY.")
    else:
        df_early = df_early_all.copy()
        df_early_view = df_early.sort_values("Early_Score", ascending=False).head(top)
        st.dataframe(df_early_view, use_container_width=True)

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

        # Aggiunta alla watchlist
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
            "- **RSI**: 14 periodi, nel range definito da slider.\n"
            "- **Vol_Ratio**: volume relativo, >1.2 = forte interesse.\n"
            "- **OBV_Trend**: UP/DOWN in base alla pendenza media OBV 5 periodi.\n"
            "- **Stato = PRO**: trend avanzato con conferme."
        )

    if df_pro_all.empty:
        st.caption("Nessun segnale PRO.")
    else:
        df_pro = df_pro_all.copy()
        df_pro_view = df_pro.sort_values("Pro_Score", ascending=False).head(top)

        df_pro_view["OBV_Trend"] = df_pro_view["OBV_Trend"].replace(
            {"UP": "UP (flusso in ingresso)", "DOWN": "DOWN (flusso in uscita)"}
        )

        st.dataframe(df_pro_view, use_container_width=True)

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
            "- **POC**: livello di prezzo con il massimo volume scambiato.\n"
            "- **Dist_POC_%**: distanza % tra prezzo e POC.\n"
            "- **Vol_Ratio**: proxy di pressione volumetrica.\n"
            "- **Stato = HOT**: area di forte decisione."
        )

    if df_rea_all.empty:
        st.caption("Nessun segnale REA‑QUANT.")
    else:
        df_rea_view = df_rea_all.sort_values("Rea_Score", ascending=False).head(top)
        st.dataframe(df_rea_view, use_container_width=True)

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
            "- Top 10: ordinati per Vol_Ratio."
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
        ).reset_index()

        st.markdown("**Distribuzione segnali per mercato**")
        st.dataframe(agg, use_container_width=True)

        st.markdown("**Top 10 per pressione volumetrica (Vol_Ratio)**")
        df_rea_top = df_rea_q.sort_values("Vol_Ratio", ascending=False).head(10)
        st.dataframe(
            df_rea_top[["Nome", "Ticker", "Prezzo", "POC",
                        "Dist_POC_%", "Vol_Ratio", "Stato"]],
            use_container_width=True,
        )

        options_rea_q = [
            f"{row['Ticker']} – {row['Nome']}" for _, row in df_rea_top.iterrows()
        ]
        selection_rea_q = st.multiselect(
            "Aggiungi alla Watchlist (Rea Quant Top10):",
            options=options_rea_q,
            key="wl_rea_q",
        )
        note_rea_q = st.text_input("Note comuni per questi ticker (Rea Quant)", key="note_wl_rea_q")
        if st.button("📌 Salva in Watchlist (Rea Quant)"):
            tickers = [s.split(" – ")[0] for s in selection_rea_q]
            names   = [s.split(" – ")[1] for s in selection_rea_q]
            add_to_watchlist(tickers, names, "REA_QUANT", note_rea_q)
            st.success("Rea Quant salvati in watchlist.")
            st.experimental_rerun()

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
            "- **Breakout_Up**: True se l'ultimo close rompe i massimi.\n"
            "- **Breakout_Down**: True se l'ultimo close rompe i minimi.\n"
            "- Ordinamento per Pro_Score per privilegiare i breakout in trend forti."
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
                df_ep[["Ticker", "Nome", "Pro_Score", "RSI", "Vol_Ratio"]],
                on="Ticker",
                how="left"
            )

            st.markdown("**Breakout su massimi/minimi 20 giorni (Donchian style)**")
            df_break_view = df_break[
                (df_break["Breakout_Up"]) | (df_break["Breakout_Down"])
            ].sort_values("Pro_Score", ascending=False)

            st.dataframe(df_break_view, use_container_width=True)

            options_seraf = [
                f"{row['Ticker']} – {row['Nome']}" for _, row in df_break_view.iterrows()
            ]
            selection_seraf = st.multiselect(
                "Aggiungi alla Watchlist (Serafini Systems):",
                options=options_seraf,
                key="wl_seraf",
            )
            note_seraf = st.text_input("Note comuni per questi ticker Serafini", key="note_wl_seraf")
            if st.button("📌 Salva in Watchlist (Serafini)"):
                tickers = [s.split(" – ")[0] for s in selection_seraf]
                names   = [s.split(" – ")[1] for s in selection_seraf]
                add_to_watchlist(tickers, names, "SERAFINI", note_seraf)
                st.success("Serafini salvati in watchlist.")
                st.experimental_rerun()

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
            "- Tabella per capire se il mercato è in costruzione o in trend."
        )

    if df_ep.empty or "Stato" not in df_ep.columns:
        st.caption("Nessun dato scanner disponibile.")
        sheet_regime = pd.DataFrame()
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
        st.dataframe(
            df_mom[["Nome", "Ticker", "Prezzo", "Pro_Score", "RSI",
                    "Vol_Ratio", "OBV_Trend", "ATR", "Stato", "Momentum"]],
            use_container_width=True,
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
            N=("Ticker", "count")
        ).reset_index()

        st.markdown("**Sintesi Regime & Momentum per mercato (tabella)**")
        if not heat.empty:
            st.dataframe(heat.sort_values("Momentum_med", ascending=False), use_container_width=True)
        else:
            st.caption("Nessun dato sufficiente per la sintesi per mercato.")

        sheet_regime = df_all.sort_values("Momentum", ascending=False)

        options_regime = [
            f"{row['Ticker']} – {row['Nome']}" for _, row in df_mom.iterrows()
        ]
        selection_regime = st.multiselect(
            "Aggiungi alla Watchlist (Top Momentum):",
            options=options_regime,
            key="wl_regime",
        )
        note_regime = st.text_input("Note comuni per questi ticker Momentum", key="note_wl_regime")
        if st.button("📌 Salva in Watchlist (Regime/Momentum)"):
            tickers = [s.split(" – ")[0] for s in selection_regime]
            names   = [s.split(" – ")[1] for s in selection_regime]
            add_to_watchlist(tickers, names, "REGIME_MOMENTUM", note_regime)
            st.success("Regime/Momentum salvati in watchlist.")
            st.experimental_rerun()

# =============================================================================
# MULTI‑TIMEFRAME – RSI Daily / Weekly / Monthly (con Segnale_MTF)
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
            "- **Segnale_MTF**:\n"
            "  - ALIGN_LONG: tutti e tre gli RSI > 50 (bias long su tutte le scale).\n"
            "  - ALIGN_SHORT: tutti e tre < 50 (bias short allineato).\n"
            "  - MIXED: situazione non allineata."
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
                    d_daily = yf.Ticker(tkr).history(period="6mo", interval="1d")
                    d_week  = yf.Ticker(tkr).history(period="2y",  interval="1wk")
                    d_month = yf.Ticker(tkr).history(period="5y",  interval="1mo")

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
                df_ep[["Ticker", "Nome", "Pro_Score", "Stato"]],
                on="Ticker",
                how="left"
            ).drop_duplicates(subset=["Ticker"])

            st.markdown("**Top 30 per MTF_Score (allineamento forza RSI multi‑TF)**")
            df_mtf_view = df_mtf.sort_values("MTF_Score", ascending=False).head(30)
            st.dataframe(
                df_mtf_view[["Nome", "Ticker", "RSI_1D", "RSI_1W", "RSI_1M",
                             "MTF_Score", "Segnale_MTF", "Pro_Score", "Stato"]],
                use_container_width=True,
            )

            df_mtf_tv = df_mtf_view[["Ticker"]].rename(columns={"Ticker": "symbol"})
            csv_mtf = df_mtf_tv.to_csv(index=False, header=False).encode("utf-8")

            st.download_button(
                "⬇️ CSV Multi‑Timeframe (solo ticker, top MTF_Score)",
                data=csv_mtf,
                file_name=f"signals_multitimeframe_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            mtf_long = df_mtf[df_mtf["Segnale_MTF"] == "ALIGN_LONG"].sort_values("MTF_Score", ascending=False)
            mtf_short = df_mtf[df_mtf["Segnale_MTF"] == "ALIGN_SHORT"].sort_values("MTF_Score", ascending=False)

            if not mtf_long.empty:
                csv_mtf_long = mtf_long[["Ticker"]].rename(columns={"Ticker": "symbol"}).to_csv(
                    index=False, header=False
                ).encode("utf-8")
                st.download_button(
                    "⬇️ CSV MTF – ALIGN_LONG (solo ticker)",
                    data=csv_mtf_long,
                    file_name=f"signals_mtf_align_long_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            if not mtf_short.empty:
                csv_mtf_short = mtf_short[["Ticker"]].rename(columns={"Ticker": "symbol"}).to_csv(
                    index=False, header=False
                ).encode("utf-8")
                st.download_button(
                    "⬇️ CSV MTF – ALIGN_SHORT (solo ticker)",
                    data=csv_mtf_short,
                    file_name=f"signals_mtf_align_short_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            options_mtf = [
                f"{row['Ticker']} – {row['Nome']}" for _, row in mtf_long.iterrows()
            ]
            selection_mtf = st.multiselect(
                "Aggiungi alla Watchlist (MTF ALIGN_LONG):",
                options=options_mtf,
                key="wl_mtf",
            )
            note_mtf = st.text_input("Note comuni per questi ticker MTF", key="note_wl_mtf")
            if st.button("📌 Salva in Watchlist (MTF ALIGN_LONG)"):
                tickers = [s.split(" – ")[0] for s in selection_mtf]
                names   = [s.split(" – ")[1] for s in selection_mtf]
                add_to_watchlist(tickers, names, "MTF_ALIGN_LONG", note_mtf)
                st.success("MTF ALIGN_LONG salvati in watchlist.")
                st.experimental_rerun()

# =============================================================================
# 💼 RISK & PORTFOLIO
# =============================================================================
with tab_risk:
    st.subheader("💼 Risk & Portfolio")
    st.markdown(
        "Carica un CSV del portafoglio (colonne: **Ticker,Qty,Prezzo_medio**) per analizzare pesi, volatilità e contributo a rischio."
    )

    with st.expander("📘 Legenda Risk & Portfolio"):
        st.markdown(
            "- **Valore_posizione**: Qty × Prezzo corrente.\n"
            "- **Peso_%**: Valore_posizione / Valore_totale.\n"
            "- **Vol_20d_%**: volatilità storica a 20 giorni (deviazione standard dei rendimenti giornalieri × √252).\n"
            "- **Risk_Contribution**: Peso_% × Vol_20d_% (proxy di quanto ogni posizione contribuisce al rischio totale).\n"
            "- Obiettivo: identificare le posizioni sovradimensionate per rischio rispetto al portafoglio."
        )

    uploaded_port = st.file_uploader(
        "Carica portafoglio (CSV con colonne: Ticker, Qty, Prezzo_medio)", type=["csv"]
    )

    if uploaded_port is not None:
        try:
            port_df = pd.read_csv(uploaded_port)
            port_df = port_df.dropna(subset=["Ticker", "Qty"])
        except Exception:
            st.error("Errore nella lettura del CSV. Controlla che le colonne siano: Ticker, Qty, Prezzo_medio.")
            port_df = pd.DataFrame()
    else:
        port_df = pd.DataFrame()

    if port_df.empty:
        st.caption("Nessun CSV caricato o dati non validi.")
    else:
        tickers_port = port_df["Ticker"].unique().tolist()

        @st.cache_data(ttl=1800)
        def fetch_portfolio_market_data(tickers):
            prices = {}
            vols = {}
            for tkr in tickers:
                try:
                    data = yf.Ticker(tkr).history(period="3mo")
                    if data.empty:
                        continue
                    last_price = float(data["Close"].iloc[-1])
                    ret = data["Close"].pct_change().dropna()
                    if len(ret) > 1:
                        vol_20d = ret.rolling(20).std().iloc[-1] * np.sqrt(252)
                        vol_20d = float(vol_20d) * 100
                    else:
                        vol_20d = np.nan
                    prices[tkr] = last_price
                    vols[tkr] = vol_20d
                except Exception:
                    continue
            return prices, vols

        with st.spinner("Calcolo metriche di rischio sul portafoglio..."):
            last_prices, vol_20d_dict = fetch_portfolio_market_data(tickers_port)

        port_df["Prezzo_corrente"] = port_df["Ticker"].map(last_prices)
        port_df["Valore_posizione"] = port_df["Qty"] * port_df["Prezzo_corrente"]
        tot_val = port_df["Valore_posizione"].sum()

        if tot_val <= 0:
            st.caption("Valore totale portafoglio non valido (verifica Qty e prezzi).")
        else:
            port_df["Peso_%"] = port_df["Valore_posizione"] / tot_val * 100
            port_df["Vol_20d_%"] = port_df["Ticker"].map(vol_20d_dict)
            port_df["Risk_Contribution"] = port_df["Peso_%"] * port_df["Vol_20d_%"] / 100

            st.markdown("**Portafoglio con metriche di rischio**")
            st.dataframe(
                port_df[["Ticker", "Qty", "Prezzo_medio", "Prezzo_corrente",
                         "Valore_posizione", "Peso_%", "Vol_20d_%", "Risk_Contribution"]],
                use_container_width=True,
            )

            top_risk = port_df.sort_values("Risk_Contribution", ascending=False).head(3)
            vol_port = np.nanmean(port_df["Vol_20d_%"].values)
            col1, col2, col3 = st.columns(3)
            col1.metric("Valore totale portafoglio", f"{tot_val:,.0f}")
            col2.metric("Volatilità media 20d (approx)", f"{vol_port:,.1f}%")
            if not top_risk.empty:
                col3.metric(
                    "Posizione più rischiosa",
                    f"{top_risk.iloc[0]['Ticker']} ({top_risk.iloc[0]['Risk_Contribution']:.1f}%)"
                )

            csv_port = port_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ CSV Portafoglio con metriche di rischio",
                data=csv_port,
                file_name=f"portfolio_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# =============================================================================
# EXPORT XLSX COMPLETO (TUTTI I TAB)
# =============================================================================
st.subheader("⬇️ Esportazioni")

sheet_early      = df_early_all.copy()
sheet_pro        = df_pro_all.copy()
sheet_rea_sig    = df_rea_all.copy()
sheet_rea_quant  = df_rea_q if 'df_rea_q' in locals() else pd.DataFrame()
sheet_serafini   = df_break_view if 'df_break_view' in locals() else pd.DataFrame()
if 'sheet_regime' not in locals():
    sheet_regime = pd.DataFrame()
sheet_mtf_full   = df_mtf if 'df_mtf' in locals() else pd.DataFrame()
sheet_portfolio  = port_df if 'port_df' in locals() and not port_df.empty else pd.DataFrame()

output_all = io.BytesIO()
with pd.ExcelWriter(output_all, engine="xlsxwriter") as writer:
    if not sheet_early.empty:
        sheet_early.to_excel(writer, index=False, sheet_name="EARLY")
    if not sheet_pro.empty:
        sheet_pro.to_excel(writer, index=False, sheet_name="PRO")
    if not sheet_rea_sig.empty:
        sheet_rea_sig.to_excel(writer, index=False, sheet_name="REA_SIGNALS")
    if not sheet_rea_quant.empty:
        sheet_rea_quant.to_excel(writer, index=False, sheet_name="REA_QUANT")
    if not sheet_serafini.empty:
        sheet_serafini.to_excel(writer, index=False, sheet_name="SERAFINI")
    if not sheet_regime.empty:
        sheet_regime.to_excel(writer, index=False, sheet_name="REGIME_MOMENTUM")
    if not sheet_mtf_full.empty:
        sheet_mtf_full.to_excel(writer, index=False, sheet_name="MULTI_TIMEFRAME")
    if not sheet_portfolio.empty:
        sheet_portfolio.to_excel(writer, index=False, sheet_name="RISK_PORTFOLIO")

xlsx_all_tabs = output_all.getvalue()

st.download_button(
    "⬇️ XLSX COMPLETO (EARLY • PRO • REA • Rea Quant • Serafini • Regime • MTF • Risk)",
    data=xlsx_all_tabs,
    file_name=f"scanner_full_pro6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

# =============================================================================
# EXPORT UNICO PER TRADINGVIEW (SOLO TICKER TOP 10 PER TAB)
# =============================================================================
st.subheader("⬇️ Export unico TradingView (solo ticker, top 10 per tab)")

def unique_list(seq):
    seen = set()
    res = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res

top10_early   = unique_list(df_early_all.sort_values("Early_Score", ascending=False)["Ticker"].head(10).tolist()) if not df_early_all.empty else []
top10_pro     = unique_list(df_pro_all.sort_values("Pro_Score",   ascending=False)["Ticker"].head(10).tolist())  if not df_pro_all.empty   else []
top10_rea     = unique_list(df_rea_all.sort_values("Rea_Score",   ascending=False)["Ticker"].head(10).tolist())  if not df_rea_all.empty   else []
top10_seraf   = unique_list(df_break_view.sort_values("Pro_Score",ascending=False)["Ticker"].head(10).tolist())  if 'df_break_view' in locals() and not df_break_view.empty else []
top10_regime  = unique_list(sheet_regime.sort_values("Momentum",  ascending=False)["Ticker"].head(10).tolist())  if not sheet_regime.empty else []
top10_mtf     = unique_list(sheet_mtf_full.sort_values("MTF_Score",ascending=False)["Ticker"].head(10).tolist()) if not sheet_mtf_full.empty else []

lines = []

if top10_early:
    lines.append("# EARLY")
    lines.extend(top10_early)
    lines.append("# PRO")

if top10_pro:
    if not top10_early and "# PRO" not in lines:
        lines.append("# PRO")
    lines.extend(top10_pro)
    lines.append("# REA_QUANT")

if top10_rea:
    if not top10_pro and "# REA_QUANT" not in lines:
        lines.append("# REA_QUANT")
    lines.extend(top10_rea)
    lines.append("# SERAFINI")

if top10_seraf:
    if not top10_rea and "# SERAFINI" not in lines:
        lines.append("# SERAFINI")
    lines.extend(top10_seraf)
    lines.append("# REGIME_MOMENTUM")

if top10_regime:
    if not top10_seraf and "# REGIME_MOMENTUM" not in lines:
        lines.append("# REGIME_MOMENTUM")
    lines.extend(top10_regime)
    lines.append("# MULTI_TIMEFRAME")

if top10_mtf:
    if not top10_regime and "# MULTI_TIMEFRAME" not in lines:
        lines.append("# MULTI_TIMEFRAME")
    lines.extend(top10_mtf)

if lines:
    tv_text = "\n".join(lines)
    st.download_button(
        "⬇️ CSV unico TradingView (top 10 per tab, con separatori)",
        data=tv_text,
        file_name=f"tradingview_all_tabs_pro6_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.caption("Nessun ticker disponibile per l'export TradingView.")

# =============================================================================
# 📌 WATCHLIST & NOTE (tab dedicato)
# =============================================================================
with tab_watch:
    st.subheader("📌 Watchlist & Note (DB persistente)")
    st.markdown(
        "Gestisci la watchlist unificata: aggiunte dai tab, note, cancellazioni, export PDF/XLSX/CSV."
    )

    wl_df = load_watchlist()

    if wl_df.empty:
        st.caption("La watchlist è vuota. Aggiungi ticker dagli altri tab.")
    else:
        st.markdown("### Watchlist corrente")
        st.dataframe(wl_df, use_container_width=True)

        st.markdown("### Modifica nota di una riga")
        labels = [f"{r['ticker']} – {r['name']} ({r['origine']}) - {r['created_at']}" for _, r in wl_df.iterrows()]
        ids = wl_df["id"].astype(str).tolist()
        mapping = dict(zip(labels, ids))

        sel_label = st.selectbox("Seleziona riga", options=labels)
        sel_id = int(mapping[sel_label])
        current_note = wl_df.loc[wl_df["id"] == sel_id, "note"].values[0] or ""
        new_note = st.text_input("Nuova nota", value=current_note, key="wl_edit_note")

        col_upd, col_del = st.columns(2)
        if col_upd.button("💾 Aggiorna nota"):
            update_watchlist_note(sel_id, new_note)
            st.success("Nota aggiornata.")
            st.experimental_rerun()

        st.markdown("### Rimuovi più elementi")
        ids_to_delete = st.multiselect(
            "Seleziona elementi da rimuovere",
            options=wl_df["id"].astype(str).tolist(),
            format_func=lambda x: f"{wl_df.loc[wl_df['id']==int(x),'ticker'].values[0]} – "
                                  f"{wl_df.loc[wl_df['id']==int(x),'name'].values[0]} "
                                  f"({wl_df.loc[wl_df['id']==int(x),'origine'].values[0]})",
        )
        if col_del.button("🗑️ Rimuovi selezionati"):
            delete_from_watchlist(ids_to_delete)
            st.success("Elementi rimossi dalla watchlist.")
            st.experimental_rerun()

        # Export XLSX
        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            wl_df.to_excel(writer, index=False, sheet_name="WATCHLIST")
        xlsx_bytes = out_xlsx.getvalue()

        # CSV solo ticker
        csv_tickers = wl_df[["ticker"]].drop_duplicates().to_csv(
            index=False, header=False
        ).encode("utf-8")

        # Export PDF (tabella semplice)
        pdf_buffer = io.BytesIO()
        try:
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 12)
                    self.cell(0, 10, "Watchlist & Note", 0, 1, "C")
                    self.ln(2)

            pdf = PDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=8)

            pdf.set_font("Arial", "B", 8)
            pdf.cell(30, 6, "Ticker", 1)
            pdf.cell(50, 6, "Nome", 1)
            pdf.cell(25, 6, "Origine", 1)
            pdf.cell(35, 6, "Data", 1)
            pdf.cell(50, 6, "Note", 1)
            pdf.ln()

            pdf.set_font("Arial", size=8)
            for _, row in wl_df.iterrows():
                pdf.cell(30, 6, str(row["ticker"])[:12], 1)
                pdf.cell(50, 6, str(row["name"])[:22], 1)
                pdf.cell(25, 6, str(row["origine"])[:10], 1)
                pdf.cell(35, 6, str(row["created_at"])[:16], 1)
                note_txt = (row["note"] or "")[:30]
                pdf.cell(50, 6, note_txt, 1)
                pdf.ln()

            pdf.output(pdf_buffer)
            pdf_bytes = pdf_buffer.getvalue()
            pdf_ok = True
        except Exception:
            pdf_ok = False
            pdf_bytes = b""

        c1, c2, c3 = st.columns(3)
        if pdf_ok:
            c1.download_button(
                "⬇️ PDF Watchlist",
                data=pdf_bytes,
                file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            c1.caption("PDF non disponibile (errore nella generazione).")

        c2.download_button(
            "⬇️ XLSX Watchlist",
            data=xlsx_bytes,
            file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        c3.download_button(
            "⬇️ CSV Watchlist (solo ticker)",
            data=csv_tickers,
            file_name=f"watchlist_tickers_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
