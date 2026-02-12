import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from io import BytesIO
from yfinance.utils import YFRateLimitError

st.set_page_config(
    page_title="Trading Dashboard PRO (Lite)",
    layout="wide",
    page_icon="📊"
)

st.markdown("#### Quantitative Trading Dashboard")
st.title("SCAN • FILTER • EXECUTE – LITE")
st.caption("Versione PRO ottimizzata per Streamlit Cloud (no rate limit)")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurazione")

st.sidebar.subheader("📈 Selezione Mercati")
m = {
    "SP500":  st.sidebar.checkbox("🇺🇸 S&P 500 (sample 40)", True),
    "Nasdaq": st.sidebar.checkbox("🇺🇸 Nasdaq 100 (big names)", False),
}
sel = [k for k, v in m.items() if v]

st.sidebar.divider()
st.sidebar.subheader("🎛️ Parametri Scanner")

e_h    = st.sidebar.slider("EARLY - Distanza EMA20 (%)", 0.0, 20.0, 5.0, 0.5) / 100
p_rmin = st.sidebar.slider("PRO - RSI minimo", 0, 100, 30, 5)
p_rmax = st.sidebar.slider("PRO - RSI massimo", 0, 100, 70, 5)
r_poc  = st.sidebar.slider("REA - Distanza POC (%)", 0.0, 20.0, 5.0, 0.5) / 100

top = st.sidebar.number_input("TOP N titoli per tab", 5, 40, 15, 5)

if not sel:
    st.warning("Seleziona almeno un mercato.")
    st.stop()

# -----------------------------------------------------------------------------
# SUPPORTO
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_universe(markets):
    t = []
    if "SP500" in markets:
        sp = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        )["Symbol"].tolist()
        t += sp[:40]  # solo primi 40 per evitare rate limit
    if "Nasdaq" in markets:
        t += ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ADBE"]
    return list(dict.fromkeys(t))

def calc_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
    except YFRateLimitError:
        # salta il ticker se Yahoo blocca la richiesta
        return None, None
    except Exception:
        return None, None

    if len(data) < 40:
        return None, None

    c = data["Close"]
    h = data["High"]
    l = data["Low"]
    v = data["Volume"]

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
    atr_expansion = atr_ratio > 1.2

    stato_ep = "PRO" if pro_score >= 6 else ("EARLY" if early_score >= 8 else "-")

    tp = (h + l + c) / 3
    bins = np.linspace(float(l.min()), float(h.max()), 40)
    price_bins = pd.cut(tp, bins, labels=bins[:-1])
    vp = pd.DataFrame({"P": price_bins, "V": v}).groupby("P")["V"].sum()
    poc = float(vp.idxmax())
    dist_poc = abs(price - poc) / poc

    rea_score = 7 if (dist_poc < r_poc and vol_ratio > 1.2) else 0
    stato_rea = "HOT" if rea_score >= 7 else "-"

    res_ep = {
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
        "Ticker": ticker,
        "Prezzo": round(price, 2),
        "Rea_Score": rea_score,
        "POC": round(poc, 2),
        "Dist_POC_%": round(dist_poc * 100, 1),
        "Vol_Ratio": round(vol_ratio, 2),
        "Stato": stato_rea,
    }

    return res_ep, res_rea

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

# -----------------------------------------------------------------------------
# SCAN
# -----------------------------------------------------------------------------
if st.button("🚀 AVVIA SCANNER PRO", type="primary", use_container_width=True):
    universe = load_universe(sel)
    st.info(f"Scansione in corso su {len(universe)} titoli…")

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
        time.sleep(0.05)  # piccolo delay per non saturare Yahoo

    status.text("✅ Scansione completata.")
    pb.empty()

    st.session_state["df_ep_pro"] = pd.DataFrame(r_ep)
    st.session_state["df_rea_pro"] = pd.DataFrame(r_rea)

    st.rerun()

df_ep = st.session_state.get("df_ep_pro", pd.DataFrame())
df_rea = st.session_state.get("df_rea_pro", pd.DataFrame())

# -----------------------------------------------------------------------------
# METRICHE
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_e, tab_p, tab_r = st.tabs(["🟢 EARLY", "🟣 PRO", "🟠 REA‑QUANT"])

cols_early = [
    "Ticker", "Prezzo",
    "Early_Score", "Pro_Score",
    "RSI", "Vol_Ratio",
    "OBV_Trend", "ATR", "ATR_Exp", "Stato"
]
cols_pro = cols_early
cols_rea = ["Ticker", "Prezzo", "Rea_Score", "POC", "Dist_POC_%", "Vol_Ratio", "Stato"]

with tab_e:
    st.subheader("🟢 Segnali EARLY")
    if "Stato" not in df_ep.columns:
        st.caption("Nessun dato EARLY (ancora nessuno scan valido).")
    else:
        df_early = df_ep[df_ep["Stato"] == "EARLY"].copy()
        if df_early.empty:
            st.caption("Nessun segnale EARLY.")
        else:
            df_early_view = df_early.sort_values("Early_Score", ascending=False).head(top)
            st.dataframe(df_early_view[cols_early], use_container_width=True)

with tab_p:
    st.subheader("🟣 Segnali PRO")
    if "Stato" not in df_ep.columns:
        st.caption("Nessun dato PRO (ancora nessuno scan valido).")
    else:
        df_pro = df_ep[df_ep["Stato"] == "PRO"].copy()
        if df_pro.empty:
            st.caption("Nessun segnale PRO.")
        else:
            df_pro_view = df_pro.sort_values("Pro_Score", ascending=False).head(top)
            st.dataframe(df_pro_view[cols_pro], use_container_width=True)

with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT")
    if df_rea.empty:
        st.caption("Nessun segnale REA‑QUANT.")
    else:
        st.dataframe(df_rea[cols_rea].sort_values("Rea_Score", ascending=False).head(top),
                     use_container_width=True)

# -----------------------------------------------------------------------------
# XLSX COMPLETO
# -----------------------------------------------------------------------------
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
