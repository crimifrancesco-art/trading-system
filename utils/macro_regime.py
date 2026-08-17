from __future__ import annotations

from typing import Dict, Optional
import pandas as pd
import streamlit as st

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Fed Funds": "FEDFUNDS",
    "US 2Y": "DGS2",
    "US 10Y": "DGS10",
    "10Y Breakeven": "T10YIE",
    "Unemployment": "UNRATE",
    "Industrial Production": "INDPRO",
}


def _fred_key() -> str:
    try:
        return str(st.secrets.get("FRED_API_KEY", ""))
    except Exception:
        return ""


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fred(series_id: str, api_key: str, years: int = 10) -> pd.Series:
    if not api_key:
        return pd.Series(dtype=float, name=series_id)

    import requests
    from datetime import date

    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": f"{date.today().year - years}-01-01",
        },
        timeout=30,
    )
    response.raise_for_status()

    rows = response.json().get("observations", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.Series(dtype=float, name=series_id)

    values = pd.to_numeric(
        frame["value"].replace(".", pd.NA),
        errors="coerce",
    )

    series = pd.Series(
        values.to_numpy(),
        index=pd.to_datetime(frame["date"]),
        name=series_id,
    ).dropna()

    return series.sort_index()


def load_macro_data(years: int = 10) -> Dict[str, pd.Series]:
    key = _fred_key()
    return {
        name: fetch_fred(series_id, key, years)
        for name, series_id in FRED_SERIES.items()
    }


def _latest(data: Dict[str, pd.Series], name: str) -> Optional[float]:
    series = data.get(name, pd.Series(dtype=float))
    return float(series.iloc[-1]) if not series.empty else None


def compute_macro_regime(data: Dict[str, pd.Series]) -> dict:
    cpi = _latest(data, "CPI")
    core = _latest(data, "Core CPI")
    fed = _latest(data, "Fed Funds")
    y2 = _latest(data, "US 2Y")
    y10 = _latest(data, "US 10Y")
    unrate = _latest(data, "Unemployment")

    spread = y10 - y2 if y10 is not None and y2 is not None else None

    score = 50.0

    if spread is not None:
        score += 12 if spread > 0.5 else 4 if spread > 0 else -10

    if cpi is not None:
        score += 8 if cpi < 2.5 else -8 if cpi > 4 else 0

    if unrate is not None:
        score += 6 if unrate < 4.5 else -8 if unrate > 6 else 0

    score = max(0, min(100, round(score, 1)))
    regime = (
        "Risk-On" if score >= 65
        else "Caution" if score >= 45
        else "Risk-Off" if score >= 25
        else "Crisis"
    )

    return {
        "score": score,
        "regime": regime,
        "cpi": cpi,
        "core_cpi": core,
        "fed_funds": fed,
        "us_2y": y2,
        "us_10y": y10,
        "yield_spread": spread,
        "unemployment": unrate,
    }


def render_macro_regime(key_prefix: str = "macro_v45_03") -> None:
    st.markdown("### 🌡️ Macro Regime Engine V45.03")
    st.caption("Tassi, inflazione, curva dei rendimenti e indicatori macro USA.")

    if not _fred_key():
        st.warning("Configura FRED_API_KEY nei Secrets Streamlit.")
        return

    years = st.slider(
        "Storico macro",
        3,
        15,
        10,
        key=f"{key_prefix}_years",
    )

    if st.button(
        "🔄 Carica dati macro",
        key=f"{key_prefix}_load",
        type="primary",
    ):
        st.session_state[f"{key_prefix}_loaded"] = True

    if not st.session_state.get(f"{key_prefix}_loaded"):
        st.info("Premi 'Carica dati macro' per iniziare.")
        return

    with st.spinner("Carico dati FRED..."):
        data = load_macro_data(years)

    result = compute_macro_regime(data)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Regime", result["regime"])
    c2.metric("Macro score", result["score"])
    c3.metric(
        "CPI",
        f"{result['cpi']:.2f}" if result["cpi"] is not None else "—",
    )
    c4.metric(
        "Fed Funds",
        f"{result['fed_funds']:.2f}%"
        if result["fed_funds"] is not None else "—",
    )
    c5.metric(
        "10Y–2Y",
        f"{result['yield_spread']:+.2f}"
        if result["yield_spread"] is not None else "—",
    )

    rows = []
    for name, series in data.items():
        if not series.empty:
            rows.append(
                pd.DataFrame({
                    "Date": series.index,
                    "Indicator": name,
                    "Value": series.values,
                })
            )

    if rows:
        frame = pd.concat(rows, ignore_index=True)
        st.dataframe(
            frame.tail(100),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "📥 Export Macro CSV",
            frame.to_csv(index=False).encode(),
            "macro_regime_v45_03.csv",
            "text/csv",
            key=f"{key_prefix}_export",
        )
