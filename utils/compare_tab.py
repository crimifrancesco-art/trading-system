# -*- coding: utf-8 -*-
"""
compare_tab.py  —  📊 Comparatore Multi-Ticker  v30.0
══════════════════════════════════════════════════════
Confronto visuale fino a 4 ticker side-by-side.

Ogni chart contiene:
  • Candele OHLC
  • EMA 20 / 50 / 200
  • Bollinger Bands
  • Volume bars (colore verde/rosso)
  • RSI(14) panel sotto
  • Linea Close normalizzata % (overlay panel centrale)

Periodo: 1m / 3m / 6m / 1y / 2y
Fonte dati: Yahoo Finance (stesso stack del progetto)
══════════════════════════════════════════════════════
"""

import urllib.request, json, time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Palette TV ────────────────────────────────────
TV_BG     = "#131722"
TV_PANEL  = "#1e222d"
TV_BORDER = "#2a2e39"
TV_BLUE   = "#2962ff"
TV_GREEN  = "#26a69a"
TV_RED    = "#ef5350"
TV_GOLD   = "#ffd700"
TV_CYAN   = "#50c4e0"
TV_GRAY   = "#787b86"
TV_TEXT   = "#d1d4dc"
TV_PURPLE = "#9c27b0"

PERIOD_MAP = {
    "1 mese":   ("1mo",  "1d"),
    "3 mesi":   ("3mo",  "1d"),
    "6 mesi":   ("6mo",  "1d"),
    "1 anno":   ("1y",   "1d"),
    "2 anni":   ("2y",   "1wk"),
}

COLORS = [TV_BLUE, TV_GOLD, TV_GREEN, TV_PURPLE]

# ── Fetch OHLCV ───────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval={interval}&range={period}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts   = result["timestamp"]
        q    = result["indicators"]["quote"][0]
        meta = result["meta"]
        name = meta.get("longName") or meta.get("shortName") or symbol
        df = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s"),
            "open":   q.get("open",  []),
            "high":   q.get("high",  []),
            "low":    q.get("low",   []),
            "close":  q.get("close", []),
            "volume": q.get("volume",[]),
        }).dropna(subset=["close", "open", "high", "low"])
        df["name"] = name
        return df
    except Exception as e:
        return pd.DataFrame()


# ── Indicatori ────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _bollinger(s: pd.Series, n=20, std=2):
    ma  = s.rolling(n).mean()
    std_ = s.rolling(n).std()
    return ma + std*std_, ma, ma - std*std_

def _rsi(s: pd.Series, n=14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _norm(s: pd.Series) -> pd.Series:
    """Normalizza a 0 = primo valore valido."""
    base = s.dropna().iloc[0] if not s.dropna().empty else 1
    return (s / base - 1) * 100


# ── Costruisce un singolo chart OHLC+indicatori ───

def _build_chart(df: pd.DataFrame, symbol: str, color: str,
                 show_norm: bool = False) -> go.Figure:
    """
    Ritorna una figura Plotly con:
      Row 1 (60%): candele + EMA20/50/200 + BB + linea normalizzata (se show_norm)
      Row 2 (15%): volume
      Row 3 (25%): RSI
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"⚠️ Dati non disponibili per {symbol}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=TV_RED, size=14))
        fig.update_layout(height=520, paper_bgcolor=TV_BG,
                          plot_bgcolor=TV_PANEL)
        return fig

    c  = df["close"]
    o  = df["open"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]
    dt = df["date"]
    name = df["name"].iloc[0] if "name" in df.columns else symbol

    # Indicatori
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema200= _ema(c, 200)
    bb_up, bb_mid, bb_dn = _bollinger(c)
    rsi   = _rsi(c)
    norm  = _norm(c)

    # Colori volume
    v_colors = [TV_GREEN if cl >= op else TV_RED
                for cl, op in zip(c, o)]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.15, 0.25],
        vertical_spacing=0.02,
    )

    # ── Row 1: Candele ────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dt, open=o, high=h, low=l, close=c,
        name="OHLC",
        increasing_line_color=TV_GREEN,
        decreasing_line_color=TV_RED,
        increasing_fillcolor=TV_GREEN,
        decreasing_fillcolor=TV_RED,
        line_width=1,
        showlegend=False,
    ), row=1, col=1)

    # Bollinger Bands (fill tra up e dn)
    fig.add_trace(go.Scatter(
        x=dt, y=bb_up, mode="lines",
        line=dict(color=TV_CYAN, width=0.8, dash="dot"),
        name="BB Up", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dt, y=bb_dn, mode="lines",
        line=dict(color=TV_CYAN, width=0.8, dash="dot"),
        fill="tonexty",
        fillcolor=f"rgba(80,196,224,0.05)",
        name="BB Dn", showlegend=False,
    ), row=1, col=1)

    # EMA 20 / 50 / 200
    fig.add_trace(go.Scatter(x=dt, y=ema20,  mode="lines",
        line=dict(color="#f48fb1", width=1.2),
        name="EMA20", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=ema50,  mode="lines",
        line=dict(color=TV_GOLD,   width=1.2),
        name="EMA50", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=ema200, mode="lines",
        line=dict(color="#7e57c2",  width=1.5),
        name="EMA200", showlegend=True), row=1, col=1)

    # ── Row 2: Volume ─────────────────────────────
    fig.add_trace(go.Bar(
        x=dt, y=v,
        marker_color=v_colors,
        marker_line_width=0,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── Row 3: RSI ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dt, y=rsi, mode="lines",
        line=dict(color=color, width=1.5),
        name="RSI", showlegend=False,
    ), row=3, col=1)
    # Zone RSI
    fig.add_hrect(y0=70, y1=100, row=3, col=1,
        fillcolor=f"rgba(239,83,80,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  row=3, col=1,
        fillcolor=f"rgba(38,166,154,0.08)", line_width=0)
    fig.add_hline(y=70, row=3, col=1,
        line=dict(color=TV_RED,   width=0.6, dash="dot"))
    fig.add_hline(y=30, row=3, col=1,
        line=dict(color=TV_GREEN, width=0.6, dash="dot"))
    fig.add_hline(y=50, row=3, col=1,
        line=dict(color=TV_GRAY,  width=0.4, dash="dot"))

    # ── Prezzo ultimo + variazione ────────────────
    last_price = float(c.iloc[-1])
    first_price= float(c.dropna().iloc[0])
    chg_pct    = (last_price / first_price - 1) * 100
    chg_color  = TV_GREEN if chg_pct >= 0 else TV_RED
    arrow      = "▲" if chg_pct >= 0 else "▼"
    last_rsi   = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 0

    title_text = (
        f"<b style='color:{color}'>{symbol}</b>"
        f"  <span style='color:{TV_GRAY};font-size:0.85em'>{name[:28]}</span><br>"
        f"<span style='color:{TV_TEXT}'>${last_price:.2f}</span>"
        f"  <span style='color:{chg_color}'>{arrow}{abs(chg_pct):.1f}%</span>"
        f"  <span style='color:{TV_GRAY};font-size:0.8em'>RSI {last_rsi:.1f}</span>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13), x=0.01, xanchor="left"),
        height=520,
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_PANEL,
        margin=dict(l=6, r=6, t=50, b=6),
        legend=dict(
            orientation="h", x=0.01, y=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=TV_GRAY),
        ),
        xaxis_rangeslider_visible=False,
        font=dict(color=TV_TEXT, size=10),
    )

    for row in [1, 2, 3]:
        n = "" if row == 1 else str(row)
        fig.update_layout(**{
            f"xaxis{n}": dict(
                showgrid=True, gridcolor=TV_BORDER,
                zeroline=False, linecolor=TV_BORDER,
                showticklabels=(row == 3),
            ),
            f"yaxis{n}": dict(
                showgrid=True, gridcolor=TV_BORDER,
                zeroline=False, linecolor=TV_BORDER,
                tickfont=dict(size=9),
            ),
        })

    # RSI axis range fisso
    fig.update_layout(yaxis3=dict(range=[0, 100], tickvals=[30, 50, 70]))

    return fig


# ── Chart normalizzato overlay (tutti i ticker) ──

def _build_normalized_chart(dfs: dict) -> go.Figure:
    """
    Un singolo chart con le linee Close normalizzate (base 0%)
    di tutti i ticker sovrapposti — per confronto diretto delle performance.
    """
    fig = go.Figure()
    for (sym, color), df in zip(
            [(s, COLORS[i]) for i, s in enumerate(dfs.keys())],
            dfs.values()):
        if df.empty:
            continue
        norm = _norm(df["close"])
        last = float(norm.dropna().iloc[-1]) if not norm.dropna().empty else 0
        arrow = "▲" if last >= 0 else "▼"
        clr   = TV_GREEN if last >= 0 else TV_RED
        fig.add_trace(go.Scatter(
            x=df["date"], y=norm,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{sym}  <span style='color:{clr}'>{arrow}{abs(last):.1f}%</span>",
        ))

    fig.add_hline(y=0, line=dict(color=TV_GRAY, width=0.8, dash="dot"))

    fig.update_layout(
        title=dict(
            text="📈 <b>Performance relativa</b>  (base 0% = primo giorno periodo)",
            font=dict(size=13, color=TV_TEXT), x=0.01
        ),
        height=280,
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_PANEL,
        margin=dict(l=6, r=6, t=44, b=6),
        legend=dict(
            orientation="h", x=0.01, y=1.15,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TV_TEXT),
        ),
        xaxis=dict(showgrid=True, gridcolor=TV_BORDER, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=TV_BORDER, zeroline=False,
                   ticksuffix="%"),
        font=dict(color=TV_TEXT, size=10),
        hovermode="x unified",
    )
    return fig


# ── Tabella riepilogo ─────────────────────────────

def _summary_table(dfs: dict) -> None:
    rows = []
    for sym, df in dfs.items():
        if df.empty:
            rows.append({"Ticker": sym, "Prezzo": "—", "Var%": "—",
                         "RSI": "—", "EMA20": "—", "EMA50": "—",
                         "EMA200": "—", "Vol medio": "—"})
            continue
        c = df["close"]
        v = df["volume"]
        ema20  = round(float(_ema(c, 20).iloc[-1]), 2)
        ema50  = round(float(_ema(c, 50).iloc[-1]), 2)
        ema200 = round(float(_ema(c, 200).iloc[-1]), 2)
        rsi_v  = round(float(_rsi(c).dropna().iloc[-1]), 1)
        price  = round(float(c.iloc[-1]), 2)
        first  = float(c.dropna().iloc[0])
        chg    = round((price / first - 1) * 100, 2)
        avgvol = int(v.tail(20).mean()) if len(v) >= 5 else 0
        trend  = ("▲" if price > ema20 else "▼")
        rows.append({
            "Ticker":   sym,
            "Nome":     df["name"].iloc[0][:22] if "name" in df.columns else sym,
            "Prezzo $": f"${price:,.2f}",
            "Var %":    f"{'▲' if chg>=0 else '▼'}{abs(chg):.1f}%",
            "RSI":      rsi_v,
            "EMA20":    f"${ema20:,.2f}",
            "EMA50":    f"${ema50:,.2f}",
            "EMA200":   f"${ema200:,.2f}",
            "Trend":    trend,
            "Vol20 avg":f"{avgvol:,}",
        })

    df_s = pd.DataFrame(rows)

    def _style(val):
        if isinstance(val, str):
            if val.startswith("▲"): return "color:#26a69a;font-weight:600"
            if val.startswith("▼"): return "color:#ef5350;font-weight:600"
        if isinstance(val, float):
            if val >= 70: return "color:#ef5350"
            if val <= 30: return "color:#26a69a"
        return "color:#d1d4dc"

    st.dataframe(
        df_s,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker":    st.column_config.TextColumn("Ticker",    width=80),
            "Nome":      st.column_config.TextColumn("Nome",      width=160),
            "Prezzo $":  st.column_config.TextColumn("Prezzo $",  width=90),
            "Var %":     st.column_config.TextColumn("Var %",     width=80),
            "RSI":       st.column_config.NumberColumn("RSI",     width=60, format="%.1f"),
            "EMA20":     st.column_config.TextColumn("EMA20",     width=90),
            "EMA50":     st.column_config.TextColumn("EMA50",     width=90),
            "EMA200":    st.column_config.TextColumn("EMA200",    width=90),
            "Trend":     st.column_config.TextColumn("Trend",     width=55),
            "Vol20 avg": st.column_config.TextColumn("Vol20 avg", width=100),
        }
    )


# ── Entry point ───────────────────────────────────

def render_compare(df_scanner=None):
    """Renderizza il tab Comparatore Multi-Ticker."""

    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_BLUE};'
        f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">'
        f'<span style="color:{TV_BLUE};font-weight:700;font-size:1rem">'
        f'📊 COMPARATORE MULTI-TICKER</span>'
        f'<span style="color:{TV_GRAY};font-size:0.8rem;margin-left:12px">'
        f'Confronto tecnico fino a 4 ticker · v30.0</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Controlli ─────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([4, 2, 1])

    with ctrl2:
        period_label = st.selectbox(
            "📅 Periodo",
            list(PERIOD_MAP.keys()),
            index=3,   # default: 1 anno
            key="compare_period",
        )
    with ctrl3:
        st.write("")
        if st.button("🔄", key="compare_refresh", help="Svuota cache e ricarica"):
            st.cache_data.clear()
            st.rerun()

    period, interval = PERIOD_MAP[period_label]

    with ctrl1:
        # Suggerisci ticker dallo scanner se disponibili
        _suggestions = []
        if df_scanner is not None and not df_scanner.empty and "Ticker" in df_scanner.columns:
            _suggestions = df_scanner["Ticker"].dropna().tolist()[:30]

        st.markdown(
            f'<span style="color:{TV_GRAY};font-size:0.8rem">'
            f'Inserisci fino a 4 ticker (es. AAPL, MSFT, GOOGL, AMZN) '
            f'— usa simboli Yahoo Finance (es. ENI.MI per Milano)</span>',
            unsafe_allow_html=True
        )

    # ── Input ticker ──────────────────────────────
    t_cols = st.columns(4)
    defaults = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    tickers_input = []
    for i, col in enumerate(t_cols):
        with col:
            val = st.text_input(
                f"Ticker {i+1}",
                value=st.session_state.get(f"compare_t{i}", defaults[i]),
                key=f"compare_ticker_{i}",
                placeholder=f"es. {defaults[i]}",
                label_visibility="collapsed",
            ).strip().upper()
            st.session_state[f"compare_t{i}"] = val
            if val:
                tickers_input.append(val)

    tickers = list(dict.fromkeys(t for t in tickers_input if t))[:4]

    if not tickers:
        st.info("Inserisci almeno un ticker per avviare il confronto.")
        return

    # ── Fetch dati ────────────────────────────────
    dfs = {}
    with st.spinner("📡 Caricamento dati..."):
        for sym in tickers:
            dfs[sym] = _fetch(sym, period, interval)

    # Filtra ticker con dati validi
    valid   = {s: d for s, d in dfs.items() if not d.empty}
    invalid = [s for s, d in dfs.items() if d.empty]

    if invalid:
        st.warning(f"⚠️ Ticker non trovati / dati non disponibili: {', '.join(invalid)}")

    if not valid:
        st.error("Nessun dato disponibile. Controlla i simboli inseriti.")
        return

    # ── Chart normalizzato overlay ─────────────────
    if len(valid) > 1:
        fig_norm = _build_normalized_chart(valid)
        st.plotly_chart(fig_norm, use_container_width=True,
                        key="compare_norm_chart")
        st.markdown("<hr style='border-color:#2a2e39;margin:8px 0'>",
                    unsafe_allow_html=True)

    # ── Chart individuali side-by-side ────────────
    n = len(valid)
    if n == 1:
        syms = list(valid.keys())
        fig = _build_chart(valid[syms[0]], syms[0], COLORS[0])
        st.plotly_chart(fig, use_container_width=True, key=f"compare_c0")

    elif n == 2:
        cols = st.columns(2)
        for i, (sym, df) in enumerate(valid.items()):
            with cols[i]:
                fig = _build_chart(df, sym, COLORS[i])
                st.plotly_chart(fig, use_container_width=True,
                                key=f"compare_c{i}")

    elif n == 3:
        cols = st.columns(3)
        for i, (sym, df) in enumerate(valid.items()):
            with cols[i]:
                fig = _build_chart(df, sym, COLORS[i])
                st.plotly_chart(fig, use_container_width=True,
                                key=f"compare_c{i}")

    else:  # 4 ticker — 2x2
        row1 = st.columns(2)
        row2 = st.columns(2)
        grid = [row1[0], row1[1], row2[0], row2[1]]
        for i, (sym, df) in enumerate(valid.items()):
            with grid[i]:
                fig = _build_chart(df, sym, COLORS[i])
                st.plotly_chart(fig, use_container_width=True,
                                key=f"compare_c{i}")

    # ── Tabella riepilogo ─────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{TV_PANEL};border-left:3px solid {TV_GOLD};'
        f'padding:6px 14px;border-radius:0 4px 4px 0;margin-bottom:8px">'
        f'<span style="color:{TV_GOLD};font-weight:700">📋 RIEPILOGO TECNICO</span>'
        f'</div>', unsafe_allow_html=True
    )
    _summary_table(valid)

    # ── Footer ────────────────────────────────────
    st.markdown(
        f'<div style="color:{TV_GRAY};font-size:0.72rem;text-align:center;'
        f'margin-top:16px;padding-top:8px;border-top:1px solid {TV_BORDER}">'
        f'Dati: Yahoo Finance · Cache 10 min · '
        f'Aggiornato: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
        f'</div>',
        unsafe_allow_html=True
    )
