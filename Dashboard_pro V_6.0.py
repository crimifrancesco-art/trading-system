import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import io
import sqlite3
from pathlib import Path
from fpdf import FPDF  # pip install fpdf2

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 6.0 (Light)",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Trading Scanner – Versione PRO 6.0 (Light)")
st.caption(
    "EARLY • PRO • REA‑QUANT • Rea Quant • Serafini • Regime & Momentum • "
    "Multi‑Timeframe • Risk & Portfolio • Alert Designer • Cruscotto Giornaliero • Watchlist DB"
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
            origine TEXT,
            note TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def add_to_watchlist(tickers, origine, note):
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t in tickers:
        c.execute(
            "INSERT INTO watchlist (ticker, origine, note, created_at) VALUES (?,?,?,?)",
            (t, origine, note, now),
        )
    conn.commit()
    conn.close()

def load_watchlist():
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["id", "ticker", "origine", "note", "created_at"])
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
# FUNZIONI DI SUPPORTO SCANNER
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
        name = info.get("longName", info.get("shortName", ticker))[:25]

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

if st.button("🚀 AVVIA SCANNER PRO 6.0 (Light)", type="primary", use_container_width=True):
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

# =============================================================================
# TABS
# =============================================================================
(
    tab_e,
    tab_p,
    tab_r,
    tab_rea_q,
    tab_serafini,
    tab_regime,
    tab_mtf,
    tab_risk,
    tab_alert,
    tab_daily,
    tab_watch,
) = st.tabs(
    [
        "🟢 EARLY",
        "🟣 PRO",
        "🟠 REA‑QUANT",
        "🧮 Rea Quant",
        "📈 Serafini Systems",
        "🧊 Regime & Momentum",
        "🕒 Multi‑Timeframe",
        "💼 Risk & Portfolio",
        "🔔 Alert Designer",
        "📆 Cruscotto Giornaliero",
        "📌 Watchlist & Note",
    ]
)

# =============================================================================
# TAB EARLY
# =============================================================================
with tab_e:
    st.subheader("🟢 Segnali EARLY")
    st.markdown(
        f"Filtro EARLY: titoli con **Stato = EARLY** (distanza prezzo–EMA20 < {e_h*100:.1f}%), "
        "punteggio Early_Score ≥ 8."
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

        tick_early_sel = st.multiselect(
            "Seleziona ticker EARLY da aggiungere a Watchlist",
            options=df_early_view["Ticker"].tolist(),
            key="sel_early",
        )
        note_early = st.text_input("Note comuni per questi ticker EARLY", key="note_early")
        if st.button("📌 Aggiungi alla Watchlist (EARLY)", key="btn_watch_early"):
            add_to_watchlist(tick_early_sel, "EARLY", note_early)
            st.success("Ticker EARLY aggiunti alla watchlist.")

# =============================================================================
# TAB PRO
# =============================================================================
with tab_p:
    st.subheader("🟣 Segnali PRO")
    st.markdown(
        f"Filtro PRO: titoli con **Stato = PRO** (prezzo sopra EMA20, RSI tra {p_rmin} e {p_rmax}, "
        "Vol_Ratio > 1.2, Pro_Score elevato)."
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

        tick_pro_sel = st.multiselect(
            "Seleziona ticker PRO da aggiungere a Watchlist",
            options=df_pro_view["Ticker"].tolist(),
            key="sel_pro",
        )
        note_pro = st.text_input("Note comuni per questi ticker PRO", key="note_pro")
        if st.button("📌 Aggiungi alla Watchlist (PRO)", key="btn_watch_pro"):
            add_to_watchlist(tick_pro_sel, "PRO", note_pro)
            st.success("Ticker PRO aggiunti alla watchlist.")

# =============================================================================
# TAB REA‑QUANT (segnali)
# =============================================================================
with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT")
    st.markdown(
        f"Filtro REA‑QUANT: titoli con **Stato = HOT** "
        f"(distanza dal POC < {r_poc*100:.1f}%, Vol_Ratio > 1.5)."
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

        tick_rea_sel = st.multiselect(
            "Seleziona ticker REA‑QUANT da aggiungere a Watchlist",
            options=df_rea_view["Ticker"].tolist(),
            key="sel_rea",
        )
        note_rea = st.text_input("Note comuni per questi ticker REA‑QUANT", key="note_rea")
        if st.button("📌 Aggiungi alla Watchlist (REA_QUANT)", key="btn_watch_rea"):
            add_to_watchlist(tick_rea_sel, "REA_QUANT", note_rea)
            st.success("Ticker REA‑QUANT aggiunti alla watchlist.")

# =============================================================================
# TAB REA QUANT (analisi)
# =============================================================================
with tab_rea_q:
    st.subheader("🧮 Analisi Quantitativa stile Massimo Rea")

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

        st.dataframe(agg, use_container_width=True)

        st.markdown("**Top 10 per pressione volumetrica (Vol_Ratio)**")
        st.dataframe(
            df_rea_q.sort_values("Vol_Ratio", ascending=False)
                    .head(10)[["Nome", "Ticker", "Prezzo", "POC",
                               "Dist_POC_%", "Vol_Ratio", "Stato"]],
            use_container_width=True,
        )

# =============================================================================
# TAB SERAFINI
# =============================================================================
with tab_serafini:
    st.subheader("📈 Approccio Trend‑Following stile Stefano Serafini")

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
            df_break_view = df_break[
                (df_break["Breakout_Up"]) | (df_break["Breakout_Down"])
            ].sort_values("Pro_Score", ascending=False)

            st.dataframe(df_break_view, use_container_width=True)

            tick_seraf_sel = st.multiselect(
                "Seleziona ticker SERAFINI da aggiungere a Watchlist",
                options=df_break_view["Ticker"].tolist(),
                key="sel_seraf",
            )
            note_seraf = st.text_input("Note comuni per questi ticker SERAFINI", key="note_seraf")
            if st.button("📌 Aggiungi alla Watchlist (SERAFINI)", key="btn_watch_seraf"):
                add_to_watchlist(tick_seraf_sel, "SERAFINI", note_seraf)
                st.success("Ticker SERAFINI aggiunti alla watchlist.")

# =============================================================================
# TAB REGIME & MOMENTUM
# =============================================================================
with tab_regime:
    st.subheader("🧊 Regime & Momentum multi‑mercato")

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

        df_all["Momentum"] = df_all["Pro_Score"] * 10 + df_all["RSI"]
        df_mom = df_all.sort_values("Momentum", ascending=False).head(10)
        st.dataframe(
            df_mom[["Nome", "Ticker", "Prezzo", "Pro_Score", "RSI",
                    "Vol_Ratio", "OBV_Trend", "ATR", "Stato", "Momentum"]],
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

        st.dataframe(heat.sort_values("Momentum_med", ascending=False), use_container_width=True)

        sheet_regime = df_all.sort_values("Momentum", ascending=False)

        tick_regime_sel = st.multiselect(
            "Seleziona ticker REGIME (Top Momentum) da aggiungere a Watchlist",
            options=df_mom["Ticker"].tolist(),
            key="sel_regime",
        )
        note_regime = st.text_input("Note comuni per questi ticker REGIME", key="note_regime")
        if st.button("📌 Aggiungi alla Watchlist (REGIME_MOMENTUM)", key="btn_watch_regime"):
            add_to_watchlist(tick_regime_sel, "REGIME_MOMENTUM", note_regime)
            st.success("Ticker REGIME_MOMENTUM aggiunti alla watchlist.")

# =============================================================================
# TAB MULTI‑TIMEFRAME
# =============================================================================
with tab_mtf:
    st.subheader("🕒 Analisi Multi‑Timeframe (RSI 1D / 1W / 1M)")

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

            df_mtf_view = df_mtf.sort_values("MTF_Score", ascending=False).head(30)
            st.dataframe(
                df_mtf_view[["Nome", "Ticker", "RSI_1D", "RSI_1W", "RSI_1M",
                             "MTF_Score", "Segnale_MTF", "Pro_Score", "Stato"]],
                use_container_width=True,
            )

            mtf_long = df_mtf[df_mtf["Segnale_MTF"] == "ALIGN_LONG"].sort_values("MTF_Score", ascending=False)

            tick_mtf_sel = st.multiselect(
                "Seleziona ticker MTF (ALIGN_LONG) da aggiungere a Watchlist",
                options=mtf_long["Ticker"].tolist() if not mtf_long.empty else [],
                key="sel_mtf",
            )
            note_mtf = st.text_input("Note comuni per questi ticker MTF", key="note_mtf")
            if st.button("📌 Aggiungi alla Watchlist (MTF_ALIGN_LONG)", key="btn_watch_mtf"):
                add_to_watchlist(tick_mtf_sel, "MTF_ALIGN_LONG", note_mtf)
                st.success("Ticker MTF_ALIGN_LONG aggiunti alla watchlist.")

# =============================================================================
# TAB RISK & PORTFOLIO
# =============================================================================
with tab_risk:
    st.subheader("💼 Risk & Portfolio")

    uploaded_port = st.file_uploader(
        "Carica portafoglio (CSV con colonne: Ticker, Qty, Prezzo_medio)", type=["csv"]
    )

    if uploaded_port is not None:
        try:
            port_df = pd.read_csv(uploaded_port)
            port_df = port_df.dropna(subset=["Ticker", "Qty"])
        except Exception:
            st.error("Errore nella lettura del CSV. Colonne richieste: Ticker, Qty, Prezzo_medio.")
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
                        vol_20d = ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
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

            st.dataframe(
                port_df[["Ticker", "Qty", "Prezzo_medio", "Prezzo_corrente",
                         "Valore_posizione", "Peso_%", "Vol_20d_%", "Risk_Contribution"]],
                use_container_width=True,
            )

# =============================================================================
# TAB ALERT DESIGNER (robusto)
# =============================================================================
with tab_alert:
    st.subheader("🔔 Alert Designer per TradingView")
    st.markdown(
        "Costruisci una lista di ticker + commento standard per usarli come alert in TradingView."
    )

    sources = {}

    if not df_early_all.empty and "Early_Score" in df_early_all.columns:
        sources["EARLY (top)"] = df_early_all.sort_values(
            "Early_Score", ascending=False
        ).head(top)

    if not df_pro_all.empty and "Pro_Score" in df_pro_all.columns:
        sources["PRO (top)"] = df_pro_all.sort_values(
            "Pro_Score", ascending=False
        ).head(top)

    if not df_rea_all.empty and "Rea_Score" in df_rea_all.columns:
        sources["REA_HOT (top)"] = df_rea_all.sort_values(
            "Rea_Score", ascending=False
        ).head(top)

    if "df_mtf" in locals() and isinstance(df_mtf, pd.DataFrame) and not df_mtf.empty:
        if "Segnale_MTF" in df_mtf.columns and "MTF_Score" in df_mtf.columns:
            mtf_long_src = df_mtf[df_mtf["Segnale_MTF"] == "ALIGN_LONG"]
            if not mtf_long_src.empty:
                sources["MTF ALIGN_LONG"] = mtf_long_src.sort_values(
                    "MTF_Score", ascending=False
                ).head(top)

    if not sources:
        st.caption(
            "Nessun insieme di segnali disponibile al momento per costruire alert."
        )
    else:
        sel_sources = st.multiselect(
            "Seleziona insiemi da includere negli alert",
            options=list(sources.keys()),
            default=[list(sources.keys())[0]],
        )
        alert_comment = st.text_input(
            "Commento/Tag standard per alert (es: PRO6_MTF_LONG)",
            "PRO6_SIGNAL",
        )
        prefix = st.text_input("Prefisso script TV (opzionale)", "")

        all_rows = []
        for src in sel_sources:
            df_src = sources.get(src, pd.DataFrame())
            if df_src.empty or "Ticker" not in df_src.columns:
                continue
            for t in df_src["Ticker"].tolist():
                all_rows.append({"symbol": t, "comment": alert_comment, "source": src})

        if all_rows:
            df_alert = pd.DataFrame(all_rows).drop_duplicates(subset=["symbol", "comment"])
            if prefix:
                df_alert["symbol"] = prefix + df_alert["symbol"]

            st.dataframe(df_alert, use_container_width=True)

            csv_alert = (
                df_alert[["symbol", "comment"]]
                .to_csv(index=False, header=False)
                .encode("utf-8")
            )
            st.download_button(
                "⬇️ CSV Alert (symbol, comment)",
                data=csv_alert,
                file_name=f"alerts_tradingview_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Nessun simbolo utile selezionato per gli alert.")

# =============================================================================
# TAB CRUSCOTTO GIORNALIERO
# =============================================================================
with tab_daily:
    st.subheader("📆 Cruscotto Giornaliero")
    st.markdown(
        "Watchlist operativa del giorno: fusione dei migliori segnali PRO, REA‑QUANT HOT e MTF ALIGN_LONG."
    )

    watch_rows = []

    if not df_pro_all.empty:
        pro_top = df_pro_all.sort_values("Pro_Score", ascending=False).head(10)
        for _, r in pro_top.iterrows():
            watch_rows.append({
                "Ticker": r["Ticker"],
                "Nome": r["Nome"],
                "Tipo_Segnale": "PRO",
                "Score": r["Pro_Score"],
                "RSI": r["RSI"],
                "Vol_Ratio": r["Vol_Ratio"],
            })

    if not df_rea_all.empty:
        rea_top = df_rea_all.sort_values("Rea_Score", ascending=False).head(10)
        for _, r in rea_top.iterrows():
            watch_rows.append({
                "Ticker": r["Ticker"],
                "Nome": r["Nome"],
                "Tipo_Segnale": "REA_HOT",
                "Score": r["Rea_Score"],
                "RSI": np.nan,
                "Vol_Ratio": r["Vol_Ratio"],
            })

    if "df_mtf" in locals() and isinstance(df_mtf, pd.DataFrame) and not df_mtf.empty:
        mtf_long_top = df_mtf[df_mtf["Segnale_MTF"] == "ALIGN_LONG"].sort_values("MTF_Score", ascending=False).head(10)
        for _, r in mtf_long_top.iterrows():
            watch_rows.append({
                "Ticker": r["Ticker"],
                "Nome": r["Nome"],
                "Tipo_Segnale": "MTF_ALIGN_LONG",
                "Score": r["MTF_Score"],
                "RSI": r["RSI_1D"],
                "Vol_Ratio": np.nan,
            })

    if not watch_rows:
        st.caption("Nessun segnale disponibile per la watchlist giornaliera.")
    else:
        df_daily = pd.DataFrame(watch_rows).drop_duplicates(subset=["Ticker", "Tipo_Segnale"])
        st.dataframe(df_daily, use_container_width=True)

        txt_lines = [f"{r['Ticker']} - {r['Tipo_Segnale']}" for _, r in df_daily.iterrows()]
        txt_daily = "\n".join(txt_lines).encode("utf-8")
        st.download_button(
            "⬇️ TXT Watchlist Giornaliera (Ticker - Tipo)",
            data=txt_daily,
            file_name=f"daily_watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# =============================================================================
# 📌 WATCHLIST & NOTE (persistente)
# =============================================================================
with tab_watch:
    st.subheader("📌 Watchlist & Note (DB persistente)")
    st.markdown(
        "Ticker salvati dai vari tab con origine e note. Puoi aggiornare le note ed esportare la watchlist."
    )

    wl_df = load_watchlist()

    if wl_df.empty:
        st.caption("La watchlist è vuota. Aggiungi ticker dagli altri tab.")
    else:
        st.markdown("### Modifica note per un singolo elemento")
        row_labels = [f"{r['ticker']} ({r['origine']}) - {r['created_at']}" for _, r in wl_df.iterrows()]
        row_ids = wl_df["id"].astype(str).tolist()
        row_map = dict(zip(row_labels, row_ids))

        sel_row_label = st.selectbox("Seleziona riga da modificare", options=row_labels)
        current_id = int(row_map[sel_row_label])
        current_note = wl_df.loc[wl_df["id"] == current_id, "note"].values[0] or ""
        new_note_val = st.text_input("Nuova nota", value=current_note, key="note_edit")

        if st.button("💾 Aggiorna nota"):
            update_watchlist_note(current_id, new_note_val)
            st.success("Nota aggiornata. Ricarica la pagina per vedere l'aggiornamento.")

        st.markdown("### Watchlist completa")
        st.dataframe(wl_df, use_container_width=True)

        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            wl_df.to_excel(writer, index=False, sheet_name="WATCHLIST")
        xlsx_bytes = out_xlsx.getvalue()

        csv_tickers = wl_df[["ticker"]].drop_duplicates().to_csv(index=False, header=False).encode("utf-8")

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
        pdf.cell(30, 6, "Origine", 1)
        pdf.cell(30, 6, "Data", 1)
        pdf.cell(100, 6, "Note", 1)
        pdf.ln()

        pdf.set_font("Arial", size=8)
        for _, row in wl_df.iterrows():
            pdf.cell(30, 6, str(row["ticker"])[:12], 1)
            pdf.cell(30, 6, str(row["origine"])[:12], 1)
            pdf.cell(30, 6, str(row["created_at"])[:16], 1)
            note_txt = (row["note"] or "")[:60]
            pdf.cell(100, 6, note_txt, 1)
            pdf.ln()

        pdf_raw = pdf.output(dest="S")
        if isinstance(pdf_raw, str):
            pdf_bytes = pdf_raw.encode("latin1")
        else:
            pdf_bytes = pdf_raw

        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "⬇️ PDF Watchlist",
            data=pdf_bytes,
            file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
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
