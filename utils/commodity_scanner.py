from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

COMMODITIES = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "WTI Crude": "CL=F",
    "Brent Crude": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
}


def _ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if hasattr(frame.columns, "levels"):
        frame.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in frame.columns
        ]

    return frame.dropna(how="all")


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def analyze_commodity(frame: pd.DataFrame) -> dict:
    if frame.empty or len(frame) < 60:
        return {}

    close = frame["Close"].astype(float)
    atr = _atr(frame)
    atr_pct = atr / close * 100
    percentile = float(atr_pct.rank(pct=True).iloc[-1] * 100)

    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi = float(
        (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]
    )

    high20 = close.rolling(20).max().iloc[-2]
    low20 = close.rolling(20).min().iloc[-2]
    price = float(close.iloc[-1])

    trend = (
        "BULLISH" if price > ema20 > ema50
        else "BEARISH" if price < ema20 < ema50
        else "RANGE"
    )

    setup = (
        "BREAKOUT_UP" if price > high20
        else "BREAKDOWN" if price < low20
        else "PULLBACK" if trend == "BULLISH" and price > ema50
        else "NONE"
    )

    return {
        "Price": price,
        "ATR": float(atr.iloc[-1]),
        "ATR_%": float(atr_pct.iloc[-1]),
        "ATR_Percentile": percentile,
        "RSI": rsi,
        "EMA20": float(ema20),
        "EMA50": float(ema50),
        "EMA200": float(ema200),
        "Trend": trend,
        "Setup": setup,
        "Gap_Risk": (
            "HIGH" if percentile >= 85
            else "MEDIUM" if percentile >= 60
            else "LOW"
        ),
    }


@st.cache_data(ttl=900, show_spinner=False)
def scan_commodities(
    tickers: tuple,
    period: str = "2y",
) -> pd.DataFrame:
    rows = []

    for name, ticker in COMMODITIES.items():
        if ticker not in tickers:
            continue

        try:
            result = analyze_commodity(_ohlcv(ticker, period))
            if result:
                rows.append({
                    "Commodity": name,
                    "Ticker": ticker,
                    **result,
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


def render_commodity_scanner(
    key_prefix: str = "commodity_v45_04",
) -> None:
    st.markdown("### 🛢️ Commodity Scanner V45.04")
    st.caption(
        "Oro, petrolio, gas, argento e rame · "
        "ATR percentile · setup multiday."
    )

    selected = st.multiselect(
        "Commodity",
        list(COMMODITIES),
        default=["Gold", "WTI Crude", "Natural Gas"],
        key=f"{key_prefix}_sel",
    )

    period = st.selectbox(
        "Storico",
        ["6mo", "1y", "2y", "5y"],
        index=2,
        key=f"{key_prefix}_period",
    )

    if st.button(
        "🔍 Scansiona commodity",
        key=f"{key_prefix}_run",
        type="primary",
    ):
        tickers = tuple(COMMODITIES[name] for name in selected)
        data = scan_commodities(tickers, period)
        st.session_state[f"{key_prefix}_data"] = data

    data = st.session_state.get(
        f"{key_prefix}_data",
        pd.DataFrame(),
    )

    if data.empty:
        st.info("Seleziona le commodity e avvia la scansione.")
        return

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "📥 Export Commodity CSV",
        data.to_csv(index=False).encode(),
        "commodity_scanner_v45_04.csv",
        "text/csv",
        key=f"{key_prefix}_export",
    )
