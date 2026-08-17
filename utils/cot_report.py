"""COT Report Evoluto per Trading Scanner PRO V45.02.

Modulo indipendente: dati CFTC disaggregated, posizioni nette,
delta settimanale, percentile storico e rendering Streamlit.
"""
from __future__ import annotations

import io
from datetime import date
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

COT_CONTRACTS = {
    "S&P 500 E-mini": {"cftc_code": "13874", "symbol": "ES", "asset": "Equity"},
    "Nasdaq 100 E-mini": {"cftc_code": "209742", "symbol": "NQ", "asset": "Equity"},
    "Gold": {"cftc_code": "088691", "symbol": "GC", "asset": "Metal"},
    "Silver": {"cftc_code": "084691", "symbol": "SI", "asset": "Metal"},
    "Crude Oil WTI": {"cftc_code": "067651", "symbol": "CL", "asset": "Energy"},
    "Natural Gas": {"cftc_code": "023651", "symbol": "NG", "asset": "Energy"},
    "Copper": {"cftc_code": "085692", "symbol": "HG", "asset": "Metal"},
    "Euro FX": {"cftc_code": "099741", "symbol": "6E", "asset": "FX"},
    "Japanese Yen": {"cftc_code": "097741", "symbol": "6J", "asset": "FX"},
    "10Y Treasury Note": {"cftc_code": "043602", "symbol": "ZN", "asset": "Rates"},
    "Bitcoin": {"cftc_code": "133741", "symbol": "BTC", "asset": "Crypto"},
    "Wheat": {"cftc_code": "001602", "symbol": "ZW", "asset": "Agriculture"},
}


def _find_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _download_cftc(url: str) -> bytes:
    import requests
    response = requests.get(url, timeout=30, headers={"User-Agent": "TradingScannerPRO/45.02"})
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_cot_history(cftc_code: str, years: int = 5) -> pd.DataFrame:
    """Scarica il report Futures-Only disaggregated COT storico CFTC."""
    frames = []
    current_year = date.today().year
    for year in range(current_year - years + 1, current_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        try:
            payload = _download_cftc(url)
            with io.BytesIO(payload) as buffer:
                with __import__("zipfile").ZipFile(buffer) as archive:
                    csv_name = next((n for n in archive.namelist() if n.lower().endswith((".txt", ".csv"))), None)
                    if not csv_name:
                        continue
                    with archive.open(csv_name) as raw:
                        frame = pd.read_csv(raw, low_memory=False)
            frames.append(frame)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    market_col = _find_column(raw, "Market_and_Exchange_Names", "Market and Exchange Names")
    code_col = _find_column(raw, "CFTC_Contract_Market_Code", "CFTC Contract Market Code")
    if code_col:
        raw = raw[raw[code_col].astype(str).str.strip() == str(cftc_code).strip()]
    elif market_col:
        raw = raw[raw[market_col].astype(str).str.contains(str(cftc_code), na=False)]
    if raw.empty:
        return raw

    date_col = _find_column(raw, "Report_Date_as_YYYY-MM-DD", "Report Date as YYYY-MM-DD")
    if date_col:
        raw["Report_Date"] = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        raw["Report_Date"] = pd.NaT

    groups = {
        "Commercial": ("Commercial_Positions_Long_All", "Commercial_Positions_Short_All"),
        "Non-Commercial": ("Noncommercial_Positions_Long_All", "Noncommercial_Positions_Short_All"),
        "Non-Reportable": ("Nonreportable_Positions_Long_All", "Nonreportable_Positions_Short_All"),
    }
    result = pd.DataFrame({"Report_Date": raw["Report_Date"]})
    for label, (long_name, short_name) in groups.items():
        long_col = _find_column(raw, long_name)
        short_col = _find_column(raw, short_name)
        if long_col and short_col:
            result[f"{label}_Long"] = _to_number(raw[long_col])
            result[f"{label}_Short"] = _to_number(raw[short_col])
            result[f"{label}_Net"] = result[f"{label}_Long"] - result[f"{label}_Short"]

    result = result.dropna(subset=["Report_Date"]).sort_values("Report_Date").drop_duplicates("Report_Date")
    for col in [c for c in result.columns if c != "Report_Date"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.reset_index(drop=True)


def enrich_cot(df: pd.DataFrame, lookback: int = 156) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().sort_values("Report_Date").reset_index(drop=True)
    for group in ("Commercial", "Non-Commercial", "Non-Reportable"):
        net_col = f"{group}_Net"
        if net_col not in out:
            continue
        out[f"{group}_Delta"] = out[net_col].diff()
        out[f"{group}_Percentile"] = out[net_col].rolling(lookback, min_periods=20).rank(pct=True) * 100
    nc = out.get("Non-Commercial_Net")
    if nc is not None:
        out["COT_Score"] = ((nc.rolling(lookback, min_periods=20).rank(pct=True) - 0.5) * 200).round(1)
        out["COT_Signal"] = out["COT_Score"].apply(
            lambda x: "BULLISH" if x <= -70 else "BEARISH" if x >= 70 else "NEUTRAL"
        )
    return out


def cot_export(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_cot_report(key_prefix: str = "cot_v45_02") -> None:
    st.markdown("### 📊 COT Report Evoluto V45.02")
    st.caption("Posizionamento CFTC dei grandi operatori · dati settimanali · aggiornamento automatico ogni 6 ore")
    labels = list(COT_CONTRACTS)
    selected = st.selectbox("Contratto futures", labels, key=f"{key_prefix}_contract")
    years = st.slider("Storico", 1, 10, 5, key=f"{key_prefix}_years")
    cfg = COT_CONTRACTS[selected]

    if st.button("🔄 Carica COT", key=f"{key_prefix}_load", type="primary"):
        st.session_state[f"{key_prefix}_loaded"] = True
    if not st.session_state.get(f"{key_prefix}_loaded"):
        st.info("Seleziona un contratto e premi 'Carica COT'.")
        return

    with st.spinner("Scarico dati CFTC..."):
        raw = fetch_cot_history(cfg["cftc_code"], years)
    data = enrich_cot(raw)
    if data.empty:
        st.warning("Nessun dato CFTC disponibile per il contratto selezionato.")
        return

    latest = data.iloc[-1]
    net = latest.get("Non-Commercial_Net", float("nan"))
    delta = latest.get("Non-Commercial_Delta", float("nan"))
    signal = latest.get("COT_Signal", "NEUTRAL")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Report date", str(latest["Report_Date"].date()))
    c2.metric("NC net position", f"{net:,.0f}" if pd.notna(net) else "—")
    c3.metric("Delta settimanale", f"{delta:+,.0f}" if pd.notna(delta) else "—")
    c4.metric("Segnale COT", signal)

    net_cols = [c for c in ("Commercial_Net", "Non-Commercial_Net", "Non-Reportable_Net") if c in data]
    fig = go.Figure()
    colors = {"Commercial_Net": "#26a69a", "Non-Commercial_Net": "#60a5fa", "Non-Reportable_Net": "#f59e0b"}
    for col in net_cols:
        fig.add_trace(go.Scatter(x=data["Report_Date"], y=data[col], mode="lines", name=col.replace("_Net", ""), line=dict(color=colors[col], width=2)))
    fig.add_hline(y=0, line=dict(color="#6b7280", dash="dot"))
    fig.update_layout(paper_bgcolor="#131722", plot_bgcolor="#1e222d", font=dict(color="#b2b5be"), height=360, margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

    display_cols = [c for c in ("Report_Date", "Commercial_Net", "Commercial_Delta", "Non-Commercial_Net", "Non-Commercial_Delta", "Non-Commercial_Percentile", "COT_Score", "COT_Signal") if c in data]
    st.dataframe(data[display_cols].tail(52).sort_values("Report_Date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("📥 Export COT CSV", cot_export(data), f"COT_{cfg['symbol']}_V45_02.csv", "text/csv", key=f"{key_prefix}_export")
    st.caption("Il COT è un indicatore di posizionamento con ritardo temporale: usarlo insieme a prezzo, regime e gestione del rischio.")
