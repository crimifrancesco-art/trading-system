import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Dashboard PRO",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Trading System – VERSIONE PRO")
st.caption("Layer: Risk ▸ Market Context ▸ Technical Signals ▸ Advanced")

# =============================================================================
# SIDEBAR – PARAMETRI GLOBALI
# =============================================================================
st.sidebar.title("⚙️ Configurazione")

# LAYER 1: Risk Management
st.sidebar.subheader("💰 Risk Management")
account_equity = st.sidebar.number_input(
    "Capitale (EUR)", min_value=1000.0, value=10000.0, step=500.0
)
risk_pct = st.sidebar.slider(
    "Rischio per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
)
atr_mult_stop = st.sidebar.slider(
    "ATR Stop Loss (x)", min_value=1.0, max_value=5.0, value=2.0, step=0.5
)
min_rr = st.sidebar.slider(
    "Risk/Reward minimo", min_value=1.0, max_value=5.0, value=2.0, step=0.5
)

st.sidebar.divider()

# LAYER 2+3: Mercati
st.sidebar.subheader("📈 Selezione Mercati")
m = {
    "SP500":       st.sidebar.checkbox("🇺🇸 S&P 500", True),
    "Nasdaq":      st.sidebar.checkbox("🇺🇸 Nasdaq 100", False),
    "Eurostoxx":   st.sidebar.checkbox("🇪🇺 Eurostoxx 600", False),
    "FTSE":        st.sidebar.checkbox("🇮🇹 FTSE MIB", False),
    "Commodities": st.sidebar.checkbox("🛢️ Commodities", False),
    "ETF":         st.sidebar.checkbox("📦 ETF", False),
}
sel = [k for k, v in m.items() if v]

st.sidebar.divider()
st.sidebar.subheader("🎛️ Parametri Scanner (Layer 3)")

e_h    = st.sidebar.slider("EARLY - Distanza EMA20 (%)", 0.0, 10.0, 2.0, 0.5) / 100
p_rmin = st.sidebar.slider("PRO - RSI minimo", 0, 100, 40, 5)
p_rmax = st.sidebar.slider("PRO - RSI massimo", 0, 100, 70, 5)
r_poc  = st.sidebar.slider("REA - Distanza POC (%)", 0.0, 10.0, 2.0, 0.5) / 100

top = st.sidebar.number_input("TOP N titoli per tabelle Sintesi", 5, 50, 15, 5)

if not sel:
    st.warning("⚠️ Seleziona almeno un mercato dalla sidebar.")
    st.stop()

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
        t += ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
              "NFLX", "ADBE", "COST", "PEP", "CSCO", "INTC", "AMD"]
    if "Eurostoxx" in markets:
        t += ["ASML.AS", "NESN.SW", "SAN.PA", "TTE.PA", "AIR.PA", "MC.PA", "OR.PA", "SU.PA"]
    if "FTSE" in markets:
        t += ["UCG.MI", "ISP.MI", "ENEL.MI", "ENI.MI", "LDO.MI", "PRY.MI",
              "STM.MI", "TEN.MI", "A2A.MI", "AMP.MI"]
    if "Commodities" in markets:
        t += ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F"]
    if "ETF" in markets:
        t += ["SPY", "QQQ", "IWM", "GLD", "TLT", "VTI", "EEM"]
    return list(dict.fromkeys(t))

def calc_obv(close, volume):
    """On-Balance Volume cumulato semplice."""  # [web:29]
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def scan_ticker(ticker, e_h, p_rmin, p_rmax, r_poc):
    """Scanner EARLY + PRO + REA + OBV + ATR expansion (Layer 3)."""
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

        # PRO: trend + RSI + volume
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

        # OBV + ATR expansion
        obv = calc_obv(c, v)
        obv_slope = obv.diff().rolling(5).mean().iloc[-1]
        obv_trend = "UP" if obv_slope > 0 else "DOWN"

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
        atr = tr.rolling(14).mean()
        atr_val = float(atr.iloc[-1])
        atr_ratio = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])
        atr_expansion = atr_ratio > 1.2  # [web:21]

        stato_ep = "PRO" if pro_score >= 8 else ("EARLY" if early_score >= 8 else "-")

        # REA-QUANT: Volume Profile semplificato su POC
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

def calc_position_size(price, atr, account_equity, risk_pct, atr_mult_stop):
    """Position sizing ATR-based (Layer 1)."""  # [web:21]
    risk_money = account_equity * (risk_pct / 100.0)
    stop_distance = atr * atr_mult_stop
    if stop_distance <= 0:
        return 0
    size = risk_money / stop_distance
    return max(int(size), 0)

# =============================================================================
# SCAN
# =============================================================================
if "done_pro" not in st.session_state:
    st.session_state["done_pro"] = False

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
        pb.progress((i+1) / len(universe))
        if (i+1) % 10 == 0:
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
# TABS GLOBALI (4 LAYER)
# =============================================================================
tab_risk, tab_context, tab_signals, tab_adv = st.tabs(
    ["💰 Layer 1 – Risk", "🌍 Layer 2 – Market Context",
     "📊 Layer 3 – Signals", "🧪 Layer 4 – Advanced"]
)

# -----------------------------------------------------------------------------
# LAYER 1 – RISK MANAGEMENT
# -----------------------------------------------------------------------------
with tab_risk:
    st.subheader("💰 Position Sizing & Rischio")

    col1, col2, col3 = st.columns(3)
    col1.metric("Capitale", f"{account_equity:,.0f} €")
    col2.metric("Rischio per trade", f"{risk_pct:.1f} %")
    col3.metric("RR minimo", f"{min_rr:.1f}:1")

    st.markdown("---")
    st.markdown("### Calcolatore Position Size (segnali PRO)")

    if df_ep.empty:
        st.info("Esegui lo scanner per avere segnali.")
    else:
        df_pro = df_ep[df_ep["Stato"] == "PRO"].copy()
        if df_pro.empty:
            st.caption("Nessun segnale PRO disponibile.")
        else:
            df_pro = df_pro.sort_values("Pro_Score", ascending=False).head(top)
            df_pro["Size"] = df_pro.apply(
                lambda r: calc_position_size(
                    price=r["Prezzo"],
                    atr=r["ATR"],
                    account_equity=account_equity,
                    risk_pct=risk_pct,
                    atr_mult_stop=atr_mult_stop
                ),
                axis=1,
            )
            df_pro["Risk_€"] = (df_pro["Size"] * df_pro["ATR"] * atr_mult_stop).round(2)

            st.dataframe(
                df_pro[["Nome", "Ticker", "Prezzo", "ATR", "Size", "Risk_€"]],
                use_container_width=True,
            )

            csv_risk = df_pro[["Ticker", "Prezzo", "ATR", "Size", "Risk_€"]].rename(
                columns={"Ticker": "symbol", "Prezzo": "price"}
            ).to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Esporta Position Size (CSV)",
                data=csv_risk,
                file_name=f"risk_positions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# -----------------------------------------------------------------------------
# LAYER 2 – MARKET CONTEXT
# -----------------------------------------------------------------------------
with tab_context:
    st.subheader("🌍 Market Breadth & Rotazione Settoriale")

    if df_ep.empty:
        st.info("Servono dati scanner per stimare contesto di mercato.")
    else:
        total = len(df_ep)
        n_early = (df_ep["Stato"] == "EARLY").sum()
        n_pro = (df_ep["Stato"] == "PRO").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Titoli scansionati", total)
        col2.metric("EARLY", n_early)
        col3.metric("PRO", n_pro)

        st.markdown("---")
        st.markdown(
            "Placeholder: puoi aggiungere A/D line, % sopra MA, nuovi high/low su indici esterni."
        )

# -----------------------------------------------------------------------------
# LAYER 3 – TECHNICAL SIGNALS (DOPPIO SET TAB + GRAFICI STREAMLIT)
# -----------------------------------------------------------------------------
with tab_signals:
    st.subheader("📊 Scanner EARLY + PRO + REA-QUANT")

    if df_ep.empty:
        st.info("Nessun dato: esegui lo scanner.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Segnali EARLY", (df_ep["Stato"] == "EARLY").sum())
        col2.metric("Segnali PRO", (df_ep["Stato"] == "PRO").sum())
        col3.metric(
            "Segnali REA-QUANT",
            (df_rea["Stato"] == "HOT").sum() if not df_rea.empty else 0
        )

        st.markdown("---")

        # =========================
        # GRUPPO 1: VISTA SINTESI
        # =========================
        st.markdown("### 🔎 Vista Sintesi")
        tab_e1, tab_p1, tab_r1 = st.tabs(["🟢 EARLY", "🟣 PRO", "🟠 REA-QUANT"])

        # 🟢 EARLY – Sintesi + grafico
        with tab_e1:
            df_early = df_ep[df_ep["Stato"] == "EARLY"].copy()
            if df_early.empty:
                st.caption("Nessun segnale EARLY.")
            else:
                df_early = df_early.sort_values("Early_Score", ascending=False).head(top)
                st.dataframe(df_early, use_container_width=True)

                chart_early = df_early.set_index("RSI")[["Prezzo"]]
                st.line_chart(chart_early, use_container_width=True)  # [web:43]

                df_early_tv = df_early.rename(
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

        # 🟣 PRO – Sintesi + grafico
        with tab_p1:
            df_pro = df_ep[df_ep["Stato"] == "PRO"].copy()
            if df_pro.empty:
                st.caption("Nessun segnale PRO.")
            else:
                df_pro = df_pro.sort_values("Pro_Score", ascending=False).head(top)
                st.dataframe(df_pro, use_container_width=True)

                chart_pro = df_pro.set_index("RSI")[["Prezzo"]]
                st.line_chart(chart_pro, use_container_width=True)

                df_pro_tv = df_pro.rename(
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

        # 🟠 REA-QUANT – Sintesi + grafico
        with tab_r1:
            if df_rea.empty:
                st.caption("Nessun segnale REA-QUANT.")
            else:
                df_rea_view = df_rea.sort_values("Rea_Score", ascending=False).head(top)
                st.dataframe(df_rea_view, use_container_width=True)

                chart_rea = df_rea_view.set_index("Dist_POC_%")[["Prezzo"]]
                st.line_chart(chart_rea, use_container_width=True)

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
                    "⬇️ CSV REA-QUANT per TradingView",
                    data=csv_rea,
                    file_name=f"signals_rea_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown("---")

        # =========================
        # GRUPPO 2: VISTA DETTAGLIO
        # =========================
        st.markdown("### 📋 Vista Dettaglio")
        tab_e2, tab_p2, tab_r2 = st.tabs(["🟢 EARLY", "🟣 PRO", "🟠 REA-QUANT"])

        with tab_e2:
            st.markdown("#### 🟢 EARLY – Dettaglio completo")
            df_early_full = df_ep[df_ep["Stato"] == "EARLY"].copy()
            if df_early_full.empty:
                st.caption("Nessun segnale EARLY.")
            else:
                st.dataframe(
                    df_early_full.sort_values("Early_Score", ascending=False),
                    use_container_width=True,
                )

        with tab_p2:
            st.markdown("#### 🟣 PRO – Dettaglio completo")
            df_pro_full = df_ep[df_ep["Stato"] == "PRO"].copy()
            if df_pro_full.empty:
                st.caption("Nessun segnale PRO.")
            else:
                st.dataframe(
                    df_pro_full.sort_values("Pro_Score", ascending=False),
                    use_container_width=True,
                )

        with tab_r2:
            st.markdown("#### 🟠 REA-QUANT – Dettaglio completo")
            if df_rea.empty:
                st.caption("Nessun segnale REA-QUANT.")
            else:
                st.dataframe(
                    df_rea.sort_values("Rea_Score", ascending=False),
                    use_container_width=True,
                )

# -----------------------------------------------------------------------------
# LAYER 4 – ADVANCED
# -----------------------------------------------------------------------------
with tab_adv:
    st.subheader("🧪 Advanced Tools (Idea Board)")
    st.markdown(
        "- Volume Profile semplificato (POC + VAH/VAL) su simboli selezionati.\n"
        "- Supertrend come trailing stop visivo.\n"
        "- Sistema di alert multi-condizione integrabile con broker/TradingView."
    )
    st.info("Sezione pronta per future estensioni PRO.")
