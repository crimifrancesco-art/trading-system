"""
backtest_tab.py — v32.0
========================
Tab "📈 Backtest" per il dashboard PRO.

Dipende da:
- utils.db: load_signals, signal_summary_stats, update_signal_performance
- st_aggrid (opzionale), plotly

Struttura:
• 📊 Riepilogo — tabella aggregata avanzata (Win, Avg, Med, Std, P25/P75, Max, Min, Sharpe)
• 📈 Equity curve — curva cumulata con Sharpe/MaxDD aggregati
• 🔍 Dettaglio segnali — griglia filtrabile con tutti i segnali registrati
• 🔄 Aggiorna performance — pulsante per aggiornare prezzi forward manualmente
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

# ── Colori per tipo segnale (coerenti col dashboard) ──────────────────────
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

# ── Fetch OHLCV per strategy chart (riuso da v31.1) ───────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _bt_fetch_ohlcv(symbol: str, range_: str = "1y") -> pd.DataFrame:
    try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1d&range={range_}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        q = result["indicators"]["quote"][0]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(ts, unit="s"),
                "open": q.get("open", []),
                "high": q.get("high", []),
                "low": q.get("low", []),
                "close": q.get("close", []),
                "volume": q.get("volume", []),
            }
        ).dropna(subset=["close"]).reset_index(drop=True)
        df["date"] = df["date"].dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# ── Indicatori locali (RSI / EMA / MACD / VWAP / ADX) ─────────────────────
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
    h_ = h.values.astype(float)
    l_ = lo.values.astype(float)
    c_ = c.values.astype(float)
    n = len(c_)
    tr_, dp_, dn_ = [], [], []
    for i in range(1, n):
        tr_.append(
            max(
                h_[i] - l_[i],
                abs(h_[i] - c_[i - 1]),
                abs(l_[i] - c_[i - 1]),
            )
        )
        up, dn = h_[i] - h_[i - 1], l_[i - 1] - l_[i]
        dp_.append(up if up > dn and up > 0 else 0)
        dn_.append(dn if dn > up and dn > 0 else 0)
    out = np.full(n, np.nan)
    if len(tr_) < period:
        return pd.Series(out, index=c.index)
    atr = np.mean(tr_[:period])
    dp = np.mean(dp_[:period])
    dn = np.mean(dn_[:period])
    dx = []
    for i in range(period, len(tr_)):
        atr = atr - atr / period + tr_[i]
        dp = dp - dp / period + dp_[i]
        dn = dn - dn / period + dn_[i]
        dip = 100 * dp / atr if atr > 0 else 0
        din = 100 * dn / atr if atr > 0 else 0
        dx.append(100 * abs(dip - din) / (dip + din) if (dip + din) > 0 else 0)
    if len(dx) < period:
        return pd.Series(out, index=c.index)
    av = np.mean(dx[:period])
    st2 = period + period
    out_idx = 1 + st2
    out[out_idx] = av
    for k in range(1, len(dx) - period + 1):
        av = (av * (period - 1) + dx[period - 1 + k]) / period
        out[out_idx + k] = av
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

    # Layout
    last_p = float(c.iloc[-1])
    first_p = float(c.dropna().iloc[0])
    chg = (last_p / first_p - 1) * 100
    chg_c = _TV_GREEN if chg >= 0 else _TV_RED
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
    """Widget Strategy Chart riusabile in qualsiasi tab."""
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
            seen = set()
            ordered = []
            for t in tickers:
                if t and t not in seen:
                    seen.add(t)
                    ordered.append(t)
            if ticker_labels:
                opts_display = [ticker_labels.get(t, t) for t in ordered]
                opts_raw = ordered
                idx = opts_raw.index(default_ticker) if default_ticker in opts_raw else 0
                sel_label = st.selectbox(
                    "Azienda / Ticker",
                    opts_display,
                    index=idx,
                    key=f"sc_tkr_{ks}",
                    help=f"{len(ordered)} titoli disponibili",
                )
                scticker = opts_raw[opts_display.index(sel_label)]
            else:
                idx = ordered.index(default_ticker) if default_ticker in ordered else 0
                scticker = st.selectbox(
                    "Ticker",
                    ordered,
                    index=idx,
                    key=f"sc_tkr_{ks}",
                    help=f"{len(ordered)} ticker disponibili",
                )
        else:
            scticker = st.text_input(
                "Ticker",
                value=default_ticker or "AAPL",
                key=f"sc_tkr_{ks}",
                placeholder="es. AAPL, ENEL.MI, ...",
            ).strip().upper()

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
        scrun = st.button("Mostra", key=f"sc_run_{ks}", use_container_width=True, type="primary")

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
        _bt_render_strategy_chart(scticker, scstrategy, scrange)
    else:
        st.caption("Seleziona ticker e strategia, poi clicca **Mostra**.")

# ── Import funzioni DB ------------------------------------------------------
try:
    from utils.db import (
        load_signals,
        signal_summary_stats,
        update_signal_performance,
        cache_stats,
    )

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# ── Funzione principale tab Backtest ----------------------------------------
def render_backtest_tab():
    st.markdown('<div class="section-pill">📈 BACKTEST SEGNALI</div>', unsafe_allow_html=True)

    if not DB_AVAILABLE:
        st.error("utils.db non disponibile: assicurati che db.py v32 sia installato.")
        return

    # Controlli
    colctrl1, colctrl2, colctrl3 = st.columns((2, 2, 2))
    with colctrl1:
        days_back = st.selectbox(
            "Periodo analisi",
            [7, 14, 30, 60, 90, 180, 365],
            index=2,
            key="bt_days",
        )
    with colctrl2:
        signal_filter = st.selectbox(
            "Tipo segnale",
            ["Tutti", "EARLY", "PRO", "HOT", "CONFLUENCE", "SERAFINI", "FINVIZ"],
            key="bt_sigtype",
        )
        sigtype_arg = None if signal_filter == "Tutti" else signal_filter
    with colctrl3:
        if st.button("Aggiorna performance", key="bt_update", use_container_width=True):
            with st.spinner("Aggiorno prezzi forward (yfinance)..."):
                n = update_signal_performance(max_signals=300)
            st.success(f"Aggiornati {n} segnali.")
            st.experimental_rerun()

    # Pulsante reset elenco segnali
    bc1, bc2 = st.columns((1, 5))
    with bc1:
        if st.button(
            "Reset Elenco Segnali",
            key="bt_resetsigs",
            type="secondary",
            help="Cancella tutti i segnali registrati dal DB. I dati scanner rimangono intatti.",
        ):
            try:
                from utils.db import _get_db_path  # opzionale, se esposto
                import sqlite3 as sq

                dbp = _get_db_path()
                c = sq.connect(str(dbp))
                c.execute("DELETE FROM signals")
                c.commit()
                c.close()
                st.success("Elenco segnali cancellato!")
                st.experimental_rerun()
            except Exception as re:
                st.error(f"Errore reset: {re}")
    with bc2:
        pass

    # Carica dati
    dfsigs = load_signals(signal_type=sigtype_arg, days_back=days_back, with_perf=True)
    dfsumm = signal_summary_stats(days_back=days_back)

    strategy_chart_widget(
        dfsigs["ticker"].dropna().unique().tolist() if not dfsigs.empty else [],
        key_suffix="bt",
    )

    st.markdown("---")

    if dfsigs.empty:
        st.info(
            "Nessun segnale registrato ancora.\n\n"
            "Come iniziare:\n"
            "1. Esegui lo scanner almeno una volta con db.py v32 attivo.\n"
            "2. Il giorno dopo clicca **Aggiorna performance** per popolare ret_1d/5d/10d/20d."
        )
        return

    st.caption(
        f"{len(dfsigs)} segnali negli ultimi {days_back} giorni · "
        f"{dfsigs['ticker'].nunique()} ticker unici · "
        f"{dfsigs['ret_20d'].notna().sum()} con performance completa (20g)"
    )

    # ── 1) Riepilogo per tipo segnale ──────────────────────────────────────
    st.markdown("### 📊 Riepilogo per tipo segnale")

    if not dfsumm.empty:
        view_all = st.checkbox(
            "Mostra tutti gli orizzonti (1g/5g/10g/20g)",
            value=False,
            key="bt_show_all_horizons",
        )

        dfs_view = dfsumm.copy()
        if not view_all:
            dfs_view = dfs_view[dfs_view["Periodo"] == "20g"]

        colsshow = [
            "Signal",
            "Periodo",
            "N",
            "N_tot",
            "Win",
            "Avg",
            "Med",
            "Std",
            "P25",
            "P75",
            "Max",
            "Min",
            "Sharpe",
        ]
        colsshow = [c for c in colsshow if c in dfs_view.columns]
        dfshow = dfs_view[colsshow].copy()

        def color_ret(v):
            if pd.isna(v):
                return "color: #374151"
            return "color: #00ff88; font-weight: bold" if v > 0 else "color: #ef4444; font-weight: bold"

        def color_win(v):
            if pd.isna(v):
                return "color: #374151"
            if v >= 60:
                return "color: #00ff88; font-weight: bold"
            if v >= 50:
                return "color: #f59e0b"
            return "color: #ef4444"

        def color_sharpe(v):
            if pd.isna(v):
                return "color: #374151"
            if v >= 1.0:
                return "color: #00ff88; font-weight: bold"
            if v >= 0.5:
                return "color: #f59e0b"
            return "color: #ef4444"

        retcols = [c for c in dfshow.columns if c in ("Avg", "Med", "Max", "Min")]
        wcols = ["Win"] if "Win" in dfshow.columns else []
        scol = ["Sharpe"] if "Sharpe" in dfshow.columns else []

        styled = (
            dfshow.style.applymap(color_ret, subset=retcols)
            .applymap(color_win, subset=wcols)
            .applymap(color_sharpe, subset=scol)
            .format(
                {
                    "Win": "{:.1f}",
                    "Avg": "{:.2f}",
                    "Med": "{:.2f}",
                    "Std": "{:.2f}",
                    "P25": "{:.2f}",
                    "P75": "{:.2f}",
                    "Max": "{:.2f}",
                    "Min": "{:.2f}",
                    "Sharpe": "{:.2f}",
                },
                na_rep="",
            )
        )

        st.dataframe(styled, use_container_width=True, height=260)
    else:
        st.info("Nessuna statistica disponibile per il periodo selezionato.")

    # ── 2) Equity curve cumulata ───────────────────────────────────────────
    st.markdown("### 📈 Equity curve cumulata")

    horizon_map = {
        "ret_1d": "1 giorno",
        "ret_5d": "5 giorni",
        "ret_10d": "10 giorni",
        "ret_20d": "20 giorni",
    }
    horizon = st.radio(
        "Orizzonte temporale",
        list(horizon_map.keys()),
        format_func=lambda x: horizon_map[x],
        horizontal=True,
        key="bt_horizon",
    )

    eqcol = horizon
    if eqcol not in dfsigs.columns:
        st.info("Colonne performance non ancora calcolate per questo orizzonte.")
    else:
        dfvalid = dfsigs.dropna(subset=[eqcol, "scanned_at"]).copy()
        if dfvalid.empty:
            st.info(
                "Nessun segnale con performance disponibile per questo orizzonte. "
                "Clicca **Aggiorna performance**."
            )
        else:
            dfvalid["scanned_at"] = pd.to_datetime(dfvalid["scanned_at"])
            dfvalid = dfvalid.sort_values("scanned_at")

            figeq = go.Figure()
            typestoplot = (
                dfvalid["signal_type"].unique().tolist()
                if signal_filter == "Tutti"
                else [signal_filter]
            )

            all_rets = []
            for stype in typestoplot:
                sub = dfvalid[dfvalid["signal_type"] == stype].copy()
                if sub.empty:
                    continue

                daily = (
                    sub.groupby(sub["scanned_at"].dt.date)[eqcol]
                    .mean()
                    .reset_index()
                )
                daily.columns = ["date", "avgret"]
                daily["equity"] = (1 + daily["avgret"] / 100.0).cumprod() * 100.0
                all_rets.append(daily["avgret"])

                color = SIGNAL_COLORS.get(stype, "#c9d1d9")
                figeq.add_trace(
                    go.Scatter(
                        x=daily["date"].astype(str),
                        y=daily["equity"].round(2),
                        mode="lines+markers",
                        name=stype,
                        line=dict(color=color, width=2),
                        marker=dict(size=5),
                        hovertemplate=(
                            f"<b>{stype}</b><br>%{{x}}<br>Equity %{y:.1f}<extra></extra>"
                        ),
                    )
                )

            if all_rets:
                concat_rets = pd.concat(all_rets, ignore_index=True).dropna()
                if not concat_rets.empty:
                    avg = concat_rets.mean()
                    std = concat_rets.std()
                    sharpe = avg / std if std and std > 0 else 0.0

                    eq_all = (1 + concat_rets / 100.0).cumprod() * 100.0
                    peak = eq_all.cummax()
                    dd = (eq_all - peak) / peak * 100.0
                    maxdd = dd.min()

                    kc1, kc2, kc3 = st.columns(3)
                    kc1.metric("Sharpe (aggregato)", f"{sharpe:.2f}")
                    kc2.metric("Max Drawdown agg.", f"{maxdd:.1f}%")
                    kc3.metric("N segnali", f"{len(concat_rets)}")

            figeq.add_hline(
                y=100,
                line=dict(color="#374151", width=1, dash="dash"),
            )
            figeq.update_layout(
                PLOTLY_DARK,
                title=dict(
                    text=f"Rendimento cumulato {horizon_map[eqcol]}",
                    font=dict(color="#00ff88", size=14),
                ),
                height=380,
                yaxis=dict(title="Equity (base 100)", ticksuffix=""),
                xaxis=dict(title="Data segnale"),
                legend=dict(
                    orientation="h",
                    y=1.05,
                    x=0,
                    bgcolor="rgba(0,0,0,0)",
                ),
                hovermode="x unified",
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(figeq, use_container_width=True)

    # ── 3) Dettaglio segnali registrati ────────────────────────────────────
    st.markdown("### 🔍 Dettaglio segnali registrati")

    dispcols = [
        "scanned_at",
        "ticker",
        "nome",
        "signal_type",
        "prezzo",
        "rsi",
        "quality_score",
        "ser_score",
        "fv_score",
        "squeeze",
        "weekly_bull",
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
    ]
    dispcols = [c for c in dispcols if c in dfsigs.columns]
    dfdisp = dfsigs[dispcols].copy()

    rename_map = {
        "scanned_at": "Data",
        "ticker": "Ticker",
        "nome": "Nome",
        "signal_type": "Tipo",
        "prezzo": "Prezzo",
        "rsi": "RSI",
        "quality_score": "Quality",
        "ser_score": "Ser",
        "fv_score": "FV",
        "squeeze": "SQ",
        "weekly_bull": "W",
        "ret_1d": "1d",
        "ret_5d": "5d",
        "ret_10d": "10d",
        "ret_20d": "20d",
    }
    dfdisp = dfdisp.rename(columns=rename_map)

    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

        gb = GridOptionsBuilder.from_dataframe(dfdisp)
        gb.configure_default_column(sortable=True, resizable=True, filter=True)
        gb.configure_column("Data", width=130)
        gb.configure_column("Ticker", width=75, pinned="left")
        gb.configure_column("Nome", width=160)
        gb.configure_column("Tipo", width=100)
        gb.configure_column("Prezzo", width=80)
        for rc in ["1d", "5d", "10d", "20d"]:
            if rc in dfdisp.columns:
                gb.configure_column(
                    rc,
                    width=80,
                    cellStyle="""
                        function(params) {
                            if (params.value == null) return {color: '#374151'};
                            if (params.value >= 0) return {color: '#00ff88', fontWeight: 'bold'};
                            return {color: '#ef4444', fontWeight: 'bold'};
                        }
                    """,
                )
        gobt = gb.build()
        AgGrid(
            dfdisp,
            gridOptions=gobt,
            height=440,
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="streamlit",
            key="bt_detail_grid",
        )
    except Exception:
        st.dataframe(dfdisp, use_container_width=True, height=440)
