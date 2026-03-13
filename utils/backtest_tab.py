"""
backtest_tab.py — v32.1
========================
Tab "📈 Backtest" + Strategy Chart riusabile.

Dipende da:
- utils.db: load_signals, signal_summary_stats, update_signal_performance
- st_aggrid (opzionale), plotly
"""

import urllib.request
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np

# ── Palette TV (stile Blue Chip Dip) ───────────────────────────────────────
_TV_BG = "#131722"; _TV_PANEL = "#1e222d"; _TV_BORDER = "#2a2e39"
_TV_GREEN = "#26a69a"; _TV_RED = "#ef5350"; _TV_GOLD = "#ffd700"
_TV_BLUE = "#2962ff"; _TV_CYAN = "#50c4e0"; _TV_GRAY = "#787b86"
_TV_TEXT = "#d1d4dc"; _TV_ORANGE = "#ff9800"; _TV_PURPLE = "#9c27b0"

# ── Colori per tipo segnale ────────────────────────────────────────────────
SIGNAL_COLORS = {
    "EARLY": "#60a5fa",
    "PRO": "#00ff88",
    "HOT": "#f97316",
    "CONFLUENCE": "#a78bfa",
    "SERAFINI": "#f59e0b",
    "FINVIZ": "#38bdf8",
}

PLOTLY_DARK = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Courier New"),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
)

# ── Fetch OHLCV per strategy chart ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _bt_fetch_ohlcv(symbol: str, range_: str = "1y") -> pd.DataFrame:
    try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1d&range={range_}"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        q = result["indicators"]["quote"][0]
        df = (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(ts, unit="s"),
                    "open": q.get("open", []),
                    "high": q.get("high", []),
                    "low": q.get("low", []),
                    "close": q.get("close", []),
                    "volume": q.get("volume", []),
                }
            )
            .dropna(subset=["close"])
            .reset_index(drop=True)
        )
        df["date"] = df["date"].dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# ── Indicatori locali ──────────────────────────────────────────────────────
def _bt_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _bt_rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _bt_macd(s: pd.Series):
    ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sl = ml.ewm(span=9).mean()
    return ml, sl, ml - sl


def _bt_vwap(c: pd.Series, v: pd.Series, win: int = 20) -> pd.Series:
    return (c * v).rolling(win).sum() / v.rolling(win).sum().replace(0, np.nan)


def _bt_adx(h: pd.Series, lo: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX robusto (periodo standard 14) con gestione sicura degli indici.
    Restituisce una Series allineata a c, con NaN iniziali.
    """
    h_ = h.values.astype(float)
    l_ = lo.values.astype(float)
    c_ = c.values.astype(float)
    n = len(c_)
    out = np.full(n, np.nan)

    if n < period + 2:
        return pd.Series(out, index=c.index)

    tr = np.zeros(n - 1)
    dp = np.zeros(n - 1)
    dn = np.zeros(n - 1)
    for i in range(1, n):
        tr[i - 1] = max(
            h_[i] - l_[i],
            abs(h_[i] - c_[i - 1]),
            abs(l_[i] - c_[i - 1]),
        )
        up = h_[i] - h_[i - 1]
        down = l_[i - 1] - l_[i]
        dp[i - 1] = up if (up > down and up > 0) else 0
        dn[i - 1] = down if (down > up and down > 0) else 0

    atr = np.zeros_like(tr)
    pdm = np.zeros_like(tr)
    ndm = np.zeros_like(tr)

    atr[0] = tr[:period].mean()
    pdm[0] = dp[:period].mean()
    ndm[0] = dn[:period].mean()

    for i in range(1, len(tr)):
        atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
        pdm[i] = pdm[i - 1] - pdm[i - 1] / period + dp[i]
        ndm[i] = ndm[i - 1] - ndm[i - 1] / period + dn[i]

    dip = np.where(atr > 0, 100 * pdm / atr, 0)
    din = np.where(atr > 0, 100 * ndm / atr, 0)

    denom = dip + din
    dx = np.where(denom > 0, 100 * np.abs(dip - din) / denom, 0)

    if len(dx) < period:
        return pd.Series(out, index=c.index)

    adx = np.zeros_like(dx)
    adx[period - 1] = dx[:period].mean()
    for i in range(period, len(dx)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    start = period * 2
    valid_len = min(len(adx) - start, n - start)
    if valid_len > 0:
        out[start : start + valid_len] = adx[start : start + valid_len]

    return pd.Series(out, index=c.index)

# ── Strategy chart (riusabile) ─────────────────────────────────────────────
_SC_RULES = {
    "RSI+VWAP": (
        "📊",
        "#e91e63",
        "RSI incrocia sopra 30 + Prezzo > VWAP",
        "RSI incrocia sotto 70 o Prezzo < VWAP",
    ),
    "ADX+EMA": (
        "📈",
        "#ff9800",
        "EMA20 incrocia sopra EMA50 + ADX > 25",
        "EMA20 < EMA50 o ADX < 25",
    ),
    "MACD": (
        "⚡",
        "#2962ff",
        "MACD hist incrocia sopra 0",
        "MACD hist incrocia sotto 0",
    ),
    "EMA Cross": (
        "🔀",
        "#26a69a",
        "EMA20 incrocia sopra EMA50",
        "EMA20 incrocia sotto EMA50",
    ),
}


def _bt_detect_signals(df: pd.DataFrame, strategy: str):
    c = df["close"].reset_index(drop=True)
    h = df["high"].reset_index(drop=True)
    lo = df["low"].reset_index(drop=True)
    v = df["volume"].fillna(0).reset_index(drop=True)
    dt = df["date"].reset_index(drop=True)

    ema20 = _bt_ema(c, 20).values
    ema50 = _bt_ema(c, 50).values
    rsi_s = _bt_rsi(c).values
    ml, sl, mh = _bt_macd(c)
    mh_ = mh.values
    vwap_ = _bt_vwap(c, v, 20).values
    adx_ = _bt_adx(h, lo, c, 14).values

    e_d, e_p, x_d, x_p = [], [], [], []
    in_t = False
    for i in range(30, len(c) - 1):
        ri = rsi_s[i] if not np.isnan(rsi_s[i]) else 50
        rp = rsi_s[i - 1] if not np.isnan(rsi_s[i - 1]) else 50
        vi = vwap_[i] if not np.isnan(vwap_[i]) else float(c.iloc[i])
        ai = adx_[i] if not np.isnan(adx_[i]) else 0

        ent, ex_ = False, False
        if strategy == "RSI+VWAP":
            rsicross30 = (rp <= 30) and (ri > 30)
            ent = rsicross30 and (float(c.iloc[i]) > vi)
            ex_ = ((rp >= 70) and (ri < 70)) or (in_t and float(c.iloc[i]) < vi)
        elif strategy == "ADX+EMA":
            ent = (ema20[i - 1] <= ema50[i - 1]) and (ema20[i] > ema50[i]) and (ai > 25)
            ex_ = (ema20[i] < ema50[i]) or (ai < 25)
        elif strategy == "MACD":
            ent = (mh_[i - 1] <= 0) and (mh_[i] > 0)
            ex_ = (mh_[i - 1] >= 0) and (mh_[i] < 0)
        else:  # EMA Cross
            ent = (ema20[i - 1] <= ema50[i - 1]) and (ema20[i] > ema50[i])
            ex_ = (ema20[i - 1] >= ema50[i - 1]) and (ema20[i] < ema50[i])

        if not in_t and ent:
            e_d.append(str(dt.iloc[i].date()))
            e_p.append(float(c.iloc[i]))
            in_t = True
        elif in_t and ex_:
            x_d.append(str(dt.iloc[i].date()))
            x_p.append(float(c.iloc[i]))
            in_t = False

    return e_d, e_p, x_d, x_p


def _bt_render_strategy_chart(ticker: str, strategy: str, range_: str = "1y"):
    with st.spinner(f"⏳ Caricamento dati {ticker} ({range_})..."):
        df = _bt_fetch_ohlcv(ticker, range_)

    if df.empty:
        st.warning(f"⚠️ Dati non disponibili per {ticker}")
        return

    c = df["close"]; h = df["high"]; lo = df["low"]; v = df["volume"].fillna(0)
    dt = [str(d)[:10] for d in df["date"]]
    ema20 = _bt_ema(c, 20); ema50 = _bt_ema(c, 50); ema200 = _bt_ema(c, 200)
    rsi_s = _bt_rsi(c)
    ml, sl, mh = _bt_macd(c); hist_colors = [_TV_GREEN if x >= 0 else _TV_RED for x in mh]
    vwap_ = _bt_vwap(c, v, 20)
    adx_ = _bt_adx(h, lo, c, 14)

    use_adx = (strategy == "ADX+EMA")
    row4_title = {
        "RSI+VWAP": "RSI (14) · Zone 30/70 — segnali RSI+VWAP",
        "ADX+EMA": "ADX (14) · Soglia 25",
        "MACD": "MACD (12,26,9)",
        "EMA Cross": "MACD (12,26,9)",
    }.get(strategy, "MACD (12,26,9)")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.12, 0.18, 0.18],
        vertical_spacing=0.02,
        subplot_titles=["", "", "RSI (14)", row4_title],
    )

    # Row 1 — Candele + EMA + VWAP/EMA200
    fig.add_trace(
        go.Candlestick(
            x=dt,
            open=df["open"].values,
            high=h.values,
            low=lo.values,
            close=c.values,
            name="Price",
            increasing=dict(
                fillcolor=_TV_GREEN,
                line=dict(color=_TV_GREEN, width=1),
            ),
            decreasing=dict(
                fillcolor=_TV_RED,
                line=dict(color=_TV_RED, width=1),
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dt, y=ema20, mode="lines", name="EMA20",
            line=dict(color="#26c6da", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dt, y=ema50, mode="lines", name="EMA50",
            line=dict(color=_TV_GOLD, width=1.2),
        ),
        row=1,
        col=1,
    )

    if strategy == "RSI+VWAP":
        fig.add_trace(
            go.Scatter(
                x=dt, y=vwap_, mode="lines", name="VWAP",
                line=dict(color=_TV_ORANGE, width=2),
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=dt, y=ema200, mode="lines", name="EMA200",
                line=dict(color="#7e57c2", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )

    # Entry/Exit markers
    e_d, e_p, x_d, x_p = _bt_detect_signals(df, strategy)
    if e_d:
        fig.add_trace(
            go.Scatter(
                x=e_d,
                y=e_p,
                mode="markers",
                name="▲ Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=_TV_GREEN,
                    line=dict(color="#ffffff", width=1.5),
                ),
                hovertemplate="▲ ENTRY<br>%{x}<br>%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if x_d:
        fig.add_trace(
            go.Scatter(
                x=x_d,
                y=x_p,
                mode="markers",
                name="▼ Exit",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=_TV_RED,
                    line=dict(color="#ffffff", width=1.5),
                ),
                hovertemplate="▼ EXIT<br>%{x}<br>%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Row 2 — Volume
    vcol = [_TV_GREEN if cl >= op else _TV_RED for cl, op in zip(c, df["open"])]
    fig.add_trace(
        go.Bar(
            x=dt,
            y=v,
            marker_color=vcol,
            marker_line_width=0,
            name="Vol",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Row 3 — RSI
    fig.add_trace(
        go.Scatter(
            x=dt,
            y=rsi_s,
            mode="lines",
            line=dict(color=_TV_PURPLE, width=1.8),
            name="RSI",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hrect(
        y0=70,
        y1=100,
        row=3,
        col=1,
        fillcolor="rgba(239,83,80,0.08)",
        line_width=0,
    )
    fig.add_hrect(
        y0=0,
        y1=30,
        row=3,
        col=1,
        fillcolor="rgba(38,166,154,0.08)",
        line_width=0,
    )
    for yv, clr in [(70, _TV_RED), (50, _TV_GRAY), (30, _TV_GREEN)]:
        fig.add_hline(
            y=yv,
            row=3,
            col=1,
            line=dict(color=clr, width=0.7, dash="dot"),
        )

    # Row 4 — indicatore strategia
    if use_adx:
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=adx_,
                mode="lines",
                line=dict(color=_TV_RED, width=2),
                name="ADX",
                showlegend=False,
                fill="tozeroy",
                fillcolor="rgba(239,83,80,0.06)",
            ),
            row=4,
            col=1,
        )
        fig.add_hline(
            y=25,
            row=4,
            col=1,
            line=dict(color=_TV_GOLD, dash="dot", width=1.5),
        )
        fig.add_hrect(
            y0=25,
            y1=80,
            row=4,
            col=1,
            fillcolor="rgba(255,152,0,0.07)",
            line_width=0,
        )
    elif strategy == "RSI+VWAP":
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=rsi_s,
                mode="lines",
                line=dict(color=_TV_PURPLE, width=1.8),
                showlegend=False,
                fill="tozeroy",
                fillcolor="rgba(156,39,176,0.05)",
            ),
            row=4,
            col=1,
        )
        for yv, clr in [(70, _TV_RED), (30, _TV_GREEN)]:
            fig.add_hline(
                y=yv,
                row=4,
                col=1,
                line=dict(color=clr, width=1.2, dash="dot"),
            )
        fig.add_hrect(
            y0=70,
            y1=100,
            row=4,
            col=1,
            fillcolor="rgba(239,83,80,0.12)",
            line_width=0,
        )
        fig.add_hrect(
            y0=0,
            y1=30,
            row=4,
            col=1,
            fillcolor="rgba(38,166,154,0.12)",
            line_width=0,
        )
    else:
        fig.add_trace(
            go.Bar(
                x=dt,
                y=mh,
                marker_color=hist_colors,
                marker_line_width=0,
                opacity=0.8,
                name="Hist",
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=ml,
                mode="lines",
                line=dict(color=_TV_BLUE, width=1.3),
                name="MACD",
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=sl,
                mode="lines",
                line=dict(color=_TV_ORANGE, width=1.3),
                name="Signal",
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_hline(
            y=0,
            row=4,
            col=1,
            line=dict(color=_TV_BORDER, width=1),
        )

    last_p = float(c.iloc[-1])
    first_p = float(c.dropna().iloc[0])
    chg = (last_p / first_p - 1) * 100
    n_e, n_x = len(e_d), len(x_d)

    fig.update_layout(
        title=dict(
            text=(
                f"{ticker} · {strategy} "
                f"{'▲' if chg >= 0 else '▼'}{abs(chg):.1f}% "
                f" · ▲ {n_e} entry · ▼ {n_x} exit"
            ),
            font=dict(size=13, color=_TV_TEXT),
            x=0.01,
        ),
        height=640,
        paper_bgcolor=_TV_BG,
        plot_bgcolor=_TV_PANEL,
        legend=dict(
            bgcolor=_TV_PANEL,
            bordercolor=_TV_BORDER,
            font=dict(size=9, color=_TV_TEXT),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=60, b=8),
        font=dict(color=_TV_TEXT, size=10),
        hovermode="x unified",
    )

    for row in [1, 2, 3, 4]:
        n_ = "" if row == 1 else str(row)
        fig.update_layout(
            **{
                f"xaxis{n_}": dict(
                    showgrid=True,
                    gridcolor=_TV_BORDER,
                    zeroline=False,
                    showticklabels=(row == 4),
                ),
                f"yaxis{n_}": dict(
                    showgrid=True,
                    gridcolor=_TV_BORDER,
                    zeroline=False,
                    tickfont=dict(size=9),
                ),
            }
        )
    fig.update_layout(yaxis3=dict(range=[0, 100], tickvals=[30, 50, 70]))
    if use_adx:
        fig.update_layout(yaxis4=dict(range=[0, 80], tickvals=[0, 25, 50]))

    k = strategy.replace("+", "_").replace(" ", "_")
    st.plotly_chart(fig, use_container_width=True, key=f"bt_sc_{ticker}_{k}")


def strategy_chart_widget(
    tickers: list,
    key_suffix: str = "bt",
    default_ticker: str = "",
    ticker_labels: dict = None,
) -> None:
    """
    Widget Strategy Chart riusabile in qualsiasi tab.

    tickers: lista di ticker.
    default_ticker: ticker pre-selezionato.
    ticker_labels: {ticker: "Nome (TICKER)"}; se presente, la select è
    ordinata alfabeticamente per Nome.
    """
    ks = key_suffix.replace(" ", "_").replace("-", "_")

    st.markdown(
        f"""
        <div style="background:{_TV_PANEL};border-left:3px solid {_TV_CYAN};
             padding:8px 14px;border-radius:0 4px 4px 0;margin:12px 0 10px;">
          <span style="color:{_TV_CYAN};font-weight:700;">STRATEGY CHART</span>
          <span style="color:{_TV_GRAY};font-size:0.78rem;margin-left:10px;">
            Candele + indicatori dedicati + Entry/Exit automatici
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_tkr, c_str, c_per, c_btn = st.columns((3, 2, 2, 1))

    with c_tkr:
        if tickers:
            base = [str(t).upper() for t in tickers if str(t).strip()]
            uniq = []
            seen = set()
            for t in base:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)

            if ticker_labels:
                options = []
                for t in uniq:
                    lbl = ticker_labels.get(t, t)
                    options.append((t, str(lbl)))
                options = sorted(options, key=lambda x: x[1].lower())
                display_labels = [lbl for (_, lbl) in options]
                raw_tickers = [t for (t, _) in options]

                if default_ticker and default_ticker in raw_tickers:
                    idx = raw_tickers.index(default_ticker)
                else:
                    idx = 0

                sel_label = st.selectbox(
                    "Ticker",
                    display_labels,
                    index=idx if display_labels else 0,
                    key=f"sc_tkr_{ks}",
                    help=f"{len(raw_tickers)} titoli disponibili",
                )
                label_to_tkr = {lbl: t for t, lbl in options}
                scticker = label_to_tkr.get(
                    sel_label, raw_tickers[0] if raw_tickers else ""
                )
            else:
                uniq_sorted = sorted(uniq)
                if default_ticker and default_ticker in uniq_sorted:
                    idx = uniq_sorted.index(default_ticker)
                else:
                    idx = 0
                scticker = st.selectbox(
                    "Ticker",
                    uniq_sorted,
                    index=idx if uniq_sorted else 0,
                    key=f"sc_tkr_{ks}",
                    help=f"{len(uniq_sorted)} ticker disponibili",
                )
        else:
            scticker = (
                st.text_input(
                    "Ticker",
                    value=default_ticker or "AAPL",
                    key=f"sc_tkr_{ks}",
                    placeholder="es. AAPL, ENEL.MI, ...",
                )
                .strip()
                .upper()
            )

    with c_str:
        scstrategy = st.selectbox(
            "Strategia",
            list(_SC_RULES.keys()),
            key=f"sc_str_{ks}",
        )

    with c_per:
        scrange = st.selectbox(
            "Periodo",
            ["3mo", "6mo", "1y", "2y"],
            index=2,
            key=f"sc_per_{ks}",
        )

    with c_btn:
        scrun = st.button(
            "Mostra",
            key=f"sc_run_{ks}",
            use_container_width=True,
            type="primary",
        )

    icon, sccolor, etxt, xtxt = _SC_RULES[scstrategy]
    bl, br = st.columns(2)
    with bl:
        st.markdown(
            f"""
            <div style="background:{_TV_PANEL};border:1px solid {_TV_BORDER};
                 border-left:4px solid {sccolor};border-radius:6px;
                 padding:6px 12px;margin-bottom:8px;">
              <div style="color:{_TV_GRAY};font-size:0.65rem;">ENTRY {icon}</div>
              <div style="color:{_TV_TEXT};font-size:0.8rem;font-weight:600;">{etxt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with br:
        st.markdown(
            f"""
            <div style="background:{_TV_PANEL};border:1px solid {_TV_BORDER};
                 border-left:4px solid {_TV_RED};border-radius:6px;
                 padding:6px 12px;margin-bottom:8px;">
              <div style="color:{_TV_GRAY};font-size:0.65rem;">EXIT</div>
              <div style="color:{_TV_TEXT};font-size:0.8rem;font-weight:600;">{xtxt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if scrun and scticker:
        try:
            _bt_render_strategy_chart(scticker, scstrategy, scrange)
        except Exception as e:
            st.error(f"Errore nel caricamento grafico per {scticker}: {e}")
    else:
        st.caption("Seleziona ticker e strategia, poi clicca **Mostra**.")
