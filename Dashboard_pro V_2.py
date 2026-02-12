import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from io import BytesIO

# -----------------------------------------------------------------------------#
# RESET SESSIONE SE DF VECCHI (senza colonna 'Stato')
# -----------------------------------------------------------------------------#
if "df_ep_pro" in st.session_state:
    df_tmp = st.session_state["df_ep_pro"]
    if isinstance(df_tmp, pd.DataFrame) and "Stato" not in df_tmp.columns:
        st.session_state.clear()

# -----------------------------------------------------------------------------#
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------#
st.set_page_config(
    page_title="Trading Dashboard PRO",
    layout="wide",
    page_icon="📊"
)

# -----------------------------------------------------------------------------#
# STILE EXIFA-LIKE (CSS)
# -----------------------------------------------------------------------------#
st.markdown("""
<style>
.main {
    background-color: #020617;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
h1 {
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
h2, h3 {
    font-weight: 600;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #111827;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.2rem 0.9rem;
    background-color: transparent;
    color: #9CA3AF;
    font-size: 0.85rem;
    border: 1px solid transparent;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: radial-gradient(circle at 0 0, #1D4ED8 0, #020617 60%);
    color: #F9FAFB;
    border-color: #1D4ED8;
}
[data-testid="stMetric"] {
    background-color: #020617;
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    border: 1px solid #111827;
}
.stButton > button {
    border-radius: 999px;
    background: linear-gradient(135deg, #2563EB, #4F46E5);
    color: #F9FAFB;
    border: none;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
}
.stButton > button:hover {
    filter: brightness(1.05);
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("#### Quantitative Trading Dashboard")
st.title("SCAN • FILTER • EXECUTE")
st.caption("Versione PRO – EARLY • PRO • REA‑QUANT")

# =============================================================================#
# SIDEBAR – MERCATI E PARAMETRI
# =============================================================================#
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

# =============================================================================#
# FUNZIONI DI SUPPORTO
# =============================================================================#
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
    return (direction * volume).cumsum()  # [web:29]

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

        dist_ema = abs(price - ema20) / ema20
        early_score = 8 if dist_ema < e_h else 0

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
        atr_expansion = atr_ratio > 1.2  # [web:21]

        stato_ep = "PRO" if pro_score >= 8 else ("EARLY" if early_score >= 8 else "-")

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

def all_tabs_to_xlsx(df_early, df_pro, df_rea,
                     cols_early, cols_pro, cols_rea) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if not df_early.empty:
            df_early[cols_early].to_excel(writer, index=False, sheet_name="EARLY")
        if not df_pro.empty:
            df_pro[cols_pro].to_excel(writer, index=False, sheet_name="PRO")
        if not df_rea.empty:
            df_rea[cols_rea].to_excel(writer, index=False, sheet_name="REA_QUANT")
    return output.getvalue()

# =============================================================================#
# SCAN
# =============================================================================#
if st.button("🚀 AVVIA SCANNER PRO", type="primary", use_container_width=True):
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

    st.rerun()

# lettura sicura da session_state (anche se vuoto)
df_ep = st.session_state.get("df_ep_pro", pd.DataFrame())
df_rea = st.session_state.get("df_rea_pro", pd.DataFrame())

# =============================================================================#
# METRICHE
# =============================================================================#
st.header("Risultati Scanner")

if "Stato" in df_ep.columns:
    n_early = (df_ep["Stato"] == "EARLY").sum()
    n_pro   = (df_ep["Stato"] == "PRO").sum()
else:
    n_early = 0
    n_pro   = 0

if "Stato" in df_rea.columns:
    n_rea = (df_rea["Stato"] == "HOT").sum()
else:
    n_rea = 0

col1, col2, col3 = st.columns(3)
col1.metric("Segnali EARLY", n_early)
col2.metric("Segnali PRO", n_pro)
col3.metric("Segnali REA‑QUANT", n_rea)

# =============================================================================#
# TABS PRINCIPALI
# =============================================================================#
tab_e, tab_p, tab_r, tab_rea_quant, tab_serafini = st.tabs(
    ["🟢 EARLY", "🟣 PRO", "🟠 REA‑QUANT", "🧮 Rea Quant", "📈 Serafini Systems"]
)

cols_early = [
    "Nome", "Ticker", "Prezzo",
    "Early_Score", "Pro_Score",
    "RSI", "Vol_Ratio",
    "OBV_Trend", "ATR", "ATR_Exp", "Stato"
]
cols_pro = cols_early
cols_rea = ["Nome", "Ticker", "Prezzo",
            "Rea_Score", "POC", "Dist_POC_%", "Vol_Ratio", "Stato"]

# EARLY
with tab_e:
    st.subheader("🟢 Segnali EARLY")
    if "Stato" not in df_ep.columns:
        st.caption("Nessun dato EARLY disponibile (ancora nessuno scan o scan vuoto).")
    else:
        df_early = df_ep[df_ep["Stato"] == "EARLY"].copy()
        if df_early.empty:
            st.caption("Nessun segnale EARLY.")
        else:
            df_early_view = df_early.sort_values("Early_Score", ascending=False).head(top)
            st.dataframe(df_early_view[cols_early], use_container_width=True)

            df_early_tv = df_early_view[["Ticker"]].rename(columns={"Ticker": "symbol"})
            csv_early = df_early_tv.to_csv(index=False, header=False).encode("utf-8")

            st.download_button(
                "⬇️ CSV EARLY (watchlist TradingView)",
                data=csv_early,
                file_name=f"signals_early_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# PRO
with tab_p:
    st.subheader("🟣 Segnali PRO")
    if "Stato" not in df_ep.columns:
        st.caption("Nessun dato PRO disponibile (ancora nessuno scan o scan vuoto).")
    else:
        df_pro = df_ep[df_ep["Stato"] == "PRO"].copy()
        if df_pro.empty:
            st.caption("Nessun segnale PRO.")
        else:
            df_pro_view = df_pro.sort_values("Pro_Score", ascending=False).head(top)
            st.dataframe(df_pro_view[cols_pro], use_container_width=True)

            df_pro_tv = df_pro_view[["Ticker"]].rename(columns={"Ticker": "symbol"})
            csv_pro = df_pro_tv.to_csv(index=False, header=False).encode("utf-8")

            st.download_button(
                "⬇️ CSV PRO (watchlist TradingView)",
                data=csv_pro,
                file_name=f"signals_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# REA‑QUANT
with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT")
    if df_rea.empty:
        st.caption("Nessun segnale REA‑QUANT.")
    else:
        st.dataframe(df_rea[cols_rea].sort_values("Rea_Score", ascending=False).head(top),
                     use_container_width=True)

        df_rea_tv = df_rea.sort_values("Rea_Score", ascending=False).head(top)[["Ticker"]]
        df_rea_tv = df_rea_tv.rename(columns={"Ticker": "symbol"})
        csv_rea = df_rea_tv.to_csv(index=False, header=False).encode("utf-8")

        st.download_button(
            "⬇️ CSV REA‑QUANT (watchlist TradingView)",
            data=csv_rea,
            file_name=f"signals_rea_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# MASSIMO REA – ANALISI QUANT
with tab_rea_quant:
    st.subheader("🧮 Analisi Quantitativa stile Massimo Rea")

    if df_rea.empty:
        st.caption("Nessun dato REA‑QUANT disponibile.")
    else:
        df_rea_q = df_rea.copy()

        def detect_market(t):
            if t.endswith(".MI"):
                return "FTSE"
            if t.endswith(".PA") or t.endswith(".AS") or t.endswith(".SW"):
                return "Eurostoxx"
            if t in ["SPY", "QQQ", "IWM", "VTI"]:
                return "USA ETF"
            if t.endswith("-USD"):
                return "Crypto"
            return "Altro"

        df_rea_q["Mercato"] = df_rea_q["Ticker"].apply(detect_market)

        agg = df_rea_q.groupby("Mercato").agg(
            N=("Ticker", "count"),
            Vol_Ratio_med=("Vol_Ratio", "mean"),
            Rea_Score_med=("Rea_Score", "mean"),
        ).reset_index()

        st.markdown("**Heat‑map mercati (numero segnali e volume medio)**")
        st.dataframe(agg, use_container_width=True)

        st.markdown("**Top 10 per pressione volumetrica (Vol_Ratio)**")
        st.dataframe(
            df_rea_q.sort_values("Vol_Ratio", ascending=False)
                    .head(10)[["Nome", "Ticker", "Prezzo", "POC",
                               "Dist_POC_%", "Vol_Ratio", "Stato"]],
            use_container_width=True,
        )

# STEFANO SERAFINI – SYSTEMS
with tab_serafini:
    st.subheader("📈 Approccio Trend‑Following stile Stefano Serafini")

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

# XLSX COMPLETO (3 FOGLI)
df_early_all = df_ep[df_ep.get("Stato", "") == "EARLY"].copy() if "Stato" in df_ep.columns else pd.DataFrame()
df_pro_all   = df_ep[df_ep.get("Stato", "") == "PRO"].copy() if "Stato" in df_ep.columns else pd.DataFrame()
df_rea_all   = df_rea.copy()

xlsx_all = all_tabs_to_xlsx(df_early_all, df_pro_all, df_rea_all,
                            cols_early, cols_pro, cols_rea)

st.download_button(
    "⬇️ XLSX completo (EARLY • PRO • REA‑QUANT)",
    data=xlsx_all,
    file_name=f"scanner_all_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
