import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import io
import plotly.express as px  # Heatmap grafica

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 3.1",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Trading Scanner – Versione PRO 3.1")
st.caption("Scanner EARLY • PRO • REA‑QUANT + Rea Quant • Serafini • Regime & Momentum – Heatmap & Export TradingView")

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

if st.button("🚀 AVVIA SCANNER PRO 3.1", type="primary", use_container_width=True):
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
tab_e, tab_p, tab_r, tab_rea_q, tab_serafini, tab_regime = st.tabs(
    ["🟢 EARLY", "🟣 PRO", "🟠 REA‑QUANT", "🧮 Rea Quant", "📈 Serafini Systems", "🧊 Regime & Momentum"]
)

# EARLY
with tab_e:
    st.subheader("🟢 Segnali EARLY")
    st.markdown(
        f"Filtro EARLY: titoli con **Stato = EARLY** (distanza prezzo–EMA20 < {e_h*100:.1f}%), "
        "punteggio Early_Score ≥ 8."
    )

    with st.expander("📘 Legenda EARLY"):
        st.markdown(
            "- **Early_Score**: 8 se il prezzo è entro la soglia percentuale dalla EMA20; 0 altrimenti.\n"
            "- **RSI**: RSI a 14 periodi, utile per filtrare ipercomprato/ipervenduto.\n"
            "- **Vol_Ratio**: volume odierno / media 20 giorni; >1 indica volume superiore alla media.\n"
            "- **Stato = EARLY**: setup in formazione, prezzo ravvicinato alla media, possibile inizio di trend."
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

# PRO
with tab_p:
    st.subheader("🟣 Segnali PRO")
    st.markdown(
        f"Filtro PRO: titoli con **Stato = PRO** (prezzo sopra EMA20, RSI tra {p_rmin} e {p_rmax}, "
        "Vol_Ratio > 1.2, Pro_Score elevato)."
    )

    with st.expander("📘 Legenda PRO"):
        st.markdown(
            "- **Pro_Score**: punteggio composito (prezzo sopra EMA20, RSI nel range, volume sopra media).\n"
            "- **RSI**: finestra 14 periodi; il range configurato filtra le zone estreme.\n"
            "- **Vol_Ratio**: rapporto volume, >1.2 indica forte interesse sul titolo.\n"
            "- **OBV_Trend**: UP/DOWN in base alla pendenza media dell'OBV sugli ultimi 5 periodi.\n"
            "- **Stato = PRO**: trend già attivo, con conferme su momentum e volume."
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

# REA‑QUANT (segnali)
with tab_r:
    st.subheader("🟠 Segnali REA‑QUANT")
    st.markdown(
        f"Filtro REA‑QUANT: titoli con **Stato = HOT** "
        f"(distanza dal POC < {r_poc*100:.1f}%, Vol_Ratio > 1.5)."
    )

    with st.expander("📘 Legenda REA‑QUANT (segnali)"):
        st.markdown(
            "- **Rea_Score**: 7 quando prezzo vicino al POC e volume molto sopra la media.\n"
            "- **POC**: livello di prezzo con il massimo volume scambiato (volume profile semplificato).\n"
            "- **Dist_POC_%**: distanza percentuale tra prezzo attuale e POC.\n"
            "- **Vol_Ratio**: come sopra, ma qui usato come proxy di \"pressione\" volumetrica.\n"
            "- **Stato = HOT**: zona di forte attività, potenziale area di decisione istituzionale."
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

# MASSIMO REA – ANALISI QUANT
with tab_rea_q:
    st.subheader("🧮 Analisi Quantitativa stile Massimo Rea")
    st.markdown(
        "Analisi per mercato sui soli titoli con **Stato = HOT**: "
        "conteggio segnali, Vol_Ratio medio, Rea_Score medio e top 10 per pressione volumetrica."
    )

    with st.expander("📘 Legenda Rea Quant (analisi)"):
        st.markdown(
            "- **N**: numero di titoli HOT per mercato.\n"
            "- **Vol_Ratio_med**: media Vol_Ratio dei titoli HOT.\n"
            "- **Rea_Score_med**: intensità media del segnale REA per mercato.\n"
            "- Top 10: titoli ordinati per Vol_Ratio, i primi sono i più \"spinti\" dai volumi."
        )

    if df_rea_all.empty:
        st.caption("Nessun dato REA‑QUANT disponibile.")
        df_rea_q = pd.DataFrame()
    else:
        df_rea_q = df_rea_all.copy()

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

        st.markdown("**Distribuzione segnali per mercato**")
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
    st.markdown(
        "Sistema Donchian‑style su 20 giorni: breakout su massimi/minimi 20‑giorni "
        "calcolato su tutti i ticker scansionati."
    )

    with st.expander("📘 Legenda Serafini Systems"):
        st.markdown(
            "- **Hi20 / Lo20**: massimo/minimo a 20 giorni sul close.\n"
            "- **Breakout_Up**: True se l'ultimo close rompe i massimi a 20 giorni.\n"
            "- **Breakout_Down**: True se l'ultimo close rompe i minimi a 20 giorni.\n"
            "- L'ordinamento per Pro_Score privilegia i breakout in trend già forti."
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

# REGIME & MOMENTUM
with tab_regime:
    st.subheader("🧊 Regime & Momentum multi‑mercato")
    st.markdown(
        "Regime: % PRO vs EARLY sul totale segnali. "
        "Momentum: ranking per Pro_Score × 10 + RSI su tutti i titoli scansionati."
    )

    with st.expander("📘 Legenda Regime & Momentum"):
        st.markdown(
            "- **Regime**: quota di segnali PRO rispetto agli EARLY.\n"
            "- **Momentum**: metrica sintetica Pro_Score×10 + RSI; più è alta, più il titolo è forte.\n"
            "- Heatmap: media del Momentum per mercato, verde = forte, rosso = debole."
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

        # -------------------------------------------------------------
        # HEATMAP GRAFICA PLOTLY PER REGIME/MOMENTUM
        # -------------------------------------------------------------
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

        st.markdown("**Heatmap Regime & Momentum per mercato**")
        if not heat.empty:
            fig = px.imshow(
                heat[["Momentum_med"]].T,
                labels=dict(x="Mercato", y="Metrica", color="Momentum medio"),
                x=heat["Mercato"],
                y=["Momentum_med"],
                color_continuous_scale="RdYlGn"
            )
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="Momentum",
                    ticks="outside",
                    tickformat=".0f"
                ),
                xaxis_title="Mercato",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Nessun dato sufficiente per la heatmap.")

        sheet_regime = df_all.sort_values("Momentum", ascending=False)

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

xlsx_all_tabs = output_all.getvalue()

st.download_button(
    "⬇️ XLSX COMPLETO (EARLY • PRO • REA • Rea Quant • Serafini • Regime)",
    data=xlsx_all_tabs,
    file_name=f"scanner_full_pro3_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

# =============================================================================
# EXPORT UNICO PER TRADINGVIEW (SOLO TICKER PER TAB)
# =============================================================================
st.subheader("⬇️ Export unico TradingView (solo ticker)")

def unique_list(seq):
    seen = set()
    res = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res

# raccogli ticker per ogni sezione (usiamo df completi, non solo top N)
tick_early   = unique_list(sheet_early["Ticker"].tolist()) if not sheet_early.empty else []
tick_pro     = unique_list(sheet_pro["Ticker"].tolist()) if not sheet_pro.empty else []
tick_rea     = unique_list(sheet_rea_sig["Ticker"].tolist()) if not sheet_rea_sig.empty else []
tick_seraf   = unique_list(sheet_serafini["Ticker"].tolist()) if not sheet_serafini.empty else []
tick_regime  = unique_list(sheet_regime["Ticker"].tolist()) if not sheet_regime.empty else []

lines = []

if tick_early:
    lines.append("# EARLY")
    lines.extend(tick_early)
    lines.append("")

if tick_pro:
    lines.append("# PRO")
    lines.extend(tick_pro)
    lines.append("")

if tick_rea:
    lines.append("# REA_QUANT")
    lines.extend(tick_rea)
    lines.append("")

if tick_seraf:
    lines.append("# SERAFINI")
    lines.extend(tick_seraf)
    lines.append("")

if tick_regime:
    lines.append("# REGIME_MOMENTUM")
    lines.extend(tick_regime)
    lines.append("")

if lines:
    tv_text = "\n".join(lines)
    st.download_button(
        "⬇️ CSV unico TradingView (ticker per tab)",
        data=tv_text,
        file_name=f"tradingview_all_tabs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.caption("Nessun ticker disponibile per l'export TradingView.")
