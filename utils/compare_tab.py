# -*- coding: utf-8 -*-
"""
compare_tab.py  —  📊 Comparatore Multi-Ticker  v31.1
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

def _macd(s: pd.Series, fast=12, slow=26, sig=9):
    """Ritorna (macd_line, signal_line, histogram)."""
    ml = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
    sl = ml.ewm(span=sig).mean()
    return ml, sl, ml - sl

def _vwap(close: pd.Series, volume: pd.Series, win=20) -> pd.Series:
    """VWAP rolling su finestra win giorni."""
    cv = (close * volume).rolling(win).sum()
    v  = volume.rolling(win).sum()
    return cv / v.replace(0, np.nan)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    """ADX semplificato, ritorna pd.Series allineata all'indice di close."""
    h = high.values.astype(float)
    l = low.values.astype(float)
    c = close.values.astype(float)
    n = len(c)
    tr_arr, dm_p, dm_n = [], [], []
    for i in range(1, n):
        tr_arr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
        up, dn = h[i]-h[i-1], l[i-1]-l[i]
        dm_p.append(up if up > dn and up > 0 else 0)
        dm_n.append(dn if dn > up and dn > 0 else 0)
    adx_out = np.full(n, np.nan)
    if len(tr_arr) < period:
        return pd.Series(adx_out, index=close.index)
    atr = np.mean(tr_arr[:period])
    dp  = np.mean(dm_p[:period])
    dn_ = np.mean(dm_n[:period])
    dx_vals = []
    for i in range(period, len(tr_arr)):
        atr = atr - atr/period + tr_arr[i]
        dp  = dp  - dp /period + dm_p[i]
        dn_ = dn_ - dn_/period + dm_n[i]
        dip = 100*dp/atr  if atr>0 else 0
        din = 100*dn_/atr if atr>0 else 0
        dx_vals.append(100*abs(dip-din)/(dip+din) if (dip+din)>0 else 0)
    if len(dx_vals) >= period:
        adx_v = np.mean(dx_vals[:period])
        start  = period + period  # offset in original array
        if start < n:
            adx_out[start] = adx_v
            for k in range(1, len(dx_vals)-period+1):
                adx_v = (adx_v*(period-1) + dx_vals[period-1+k]) / period
                if start+k < n:
                    adx_out[start+k] = adx_v
    return pd.Series(adx_out, index=close.index)

def _detect_signals(df: pd.DataFrame, strategy: str) -> tuple:
    """
    Detecta segnali Entry/Exit per la strategia indicata.
    Ritorna (entry_dates, entry_prices, exit_dates, exit_prices).
    """
    c  = df["close"].reset_index(drop=True)
    h  = df["high"].reset_index(drop=True)
    lo = df["low"].reset_index(drop=True)
    v  = df["volume"].fillna(0).reset_index(drop=True)
    dt = df["date"].reset_index(drop=True)

    ema20 = _ema(c, 20).values
    ema50 = _ema(c, 50).values
    rsi_s = _rsi(c).values
    _, _, macd_hist_s = _macd(c)
    macd_hist_arr = macd_hist_s.values
    vwap_s = _vwap(c, v, win=20).values
    adx_s  = _adx(h, lo, c, period=14).values

    entry_d, entry_p, exit_d, exit_p = [], [], [], []
    in_trade = False

    for i in range(30, len(c)-1):
        rsi_i    = rsi_s[i]    if not np.isnan(rsi_s[i])    else 50
        rsi_prev = rsi_s[i-1]  if not np.isnan(rsi_s[i-1]) else 50
        vwap_i   = vwap_s[i]   if not np.isnan(vwap_s[i])  else c.iloc[i]
        adx_i    = adx_s[i]    if not np.isnan(adx_s[i])   else 0

        if strategy == "RSI+VWAP":
            entry = (rsi_prev < 30) and (rsi_i >= 30) and (c.iloc[i] > vwap_i)
            exit_ = ((rsi_prev > 70) and (rsi_i <= 70)) or \
                    (in_trade and c.iloc[i] < vwap_i and rsi_i > 65)
        elif strategy == "ADX+EMA":
            entry = (ema20[i-1] <= ema50[i-1]) and (ema20[i] > ema50[i]) and (adx_i > 25)
            exit_ = (ema20[i] < ema50[i]) or (in_trade and adx_i < 25)
        elif strategy == "MACD":
            entry = (macd_hist_arr[i-1] <= 0) and (macd_hist_arr[i] > 0)
            exit_ = (macd_hist_arr[i-1] >= 0) and (macd_hist_arr[i] < 0)
        else:  # EMA Cross
            entry = (ema20[i-1] <= ema50[i-1]) and (ema20[i] > ema50[i])
            exit_ = (ema20[i-1] >= ema50[i-1]) and (ema20[i] < ema50[i])

        if not in_trade and entry:
            entry_d.append(str(dt.iloc[i])[:10])
            entry_p.append(float(c.iloc[i]))
            in_trade = True
        elif in_trade and exit_:
            exit_d.append(str(dt.iloc[i])[:10])
            exit_p.append(float(c.iloc[i]))
            in_trade = False

    return entry_d, entry_p, exit_d, exit_p


# ── Costruisce un singolo chart OHLC+indicatori ───

def _build_chart(df: pd.DataFrame, symbol: str, color: str,
                 show_norm: bool = False,
                 strategy: str = "Nessuna") -> go.Figure:
    """
    Ritorna una figura Plotly con:
      Row 1 (55%): candele + EMA20/50/200 + BB + marker Entry/Exit strategia
      Row 2 (12%): volume
      Row 3 (18%): RSI(14) con zone 30/70
      Row 4 (15%): MACD(12,26,9) istogramma + linee — oppure ADX/VWAP label
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"⚠️ Dati non disponibili per {symbol}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=TV_RED, size=14))
        fig.update_layout(height=600, paper_bgcolor=TV_BG,
                          plot_bgcolor=TV_PANEL)
        return fig

    c  = df["close"]
    o  = df["open"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]
    dt = df["date"]
    name = df["name"].iloc[0] if "name" in df.columns else symbol

    # ── Indicatori ────────────────────────────────
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema200= _ema(c, 200)
    bb_up, bb_mid, bb_dn = _bollinger(c)
    rsi_s = _rsi(c)
    macd_line, macd_sig, macd_hist = _macd(c)
    vwap_s = _vwap(c, v, win=20)
    adx_s  = _adx(h, l, c, period=14)

    # Colori volume e MACD
    v_colors    = [TV_GREEN if cl >= op else TV_RED for cl, op in zip(c, o)]
    hist_colors = [TV_GREEN if val >= 0 else TV_RED for val in macd_hist]

    # ── Subplot 4 righe ───────────────────────────
    # Row4 label dipende dalla strategia
    if strategy in ("RSI+VWAP",):
        row4_title = "RSI (14)  ·  Zone 30/70 — strategia RSI+VWAP"
        show_adx_row = False
    elif strategy == "ADX+EMA":
        row4_title = "ADX (14)  ·  Soglia 25"
        show_adx_row = True
    else:
        row4_title = "MACD (12,26,9)"
        show_adx_row = False

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.12, 0.18, 0.18],
        vertical_spacing=0.02,
        subplot_titles=["", "", "RSI (14)", row4_title],
    )

    # ── Row 1: Candele + EMA + BB ─────────────────
    fig.add_trace(go.Candlestick(
        x=dt, open=o, high=h, low=l, close=c,
        name="OHLC",
        increasing=dict(fillcolor=TV_GREEN, line=dict(color=TV_GREEN, width=1)),
        decreasing=dict(fillcolor=TV_RED,   line=dict(color=TV_RED,   width=1)),
        showlegend=False,
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=dt, y=bb_up, mode="lines",
        line=dict(color=TV_CYAN, width=0.8, dash="dot"),
        name="BB", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=bb_dn, mode="lines",
        line=dict(color=TV_CYAN, width=0.8, dash="dot"),
        fill="tonexty", fillcolor="rgba(80,196,224,0.05)",
        name="BB Dn", showlegend=False), row=1, col=1)

    # EMA 20 / 50 / 200
    fig.add_trace(go.Scatter(x=dt, y=ema20, mode="lines",
        line=dict(color="#26c6da", width=1.2), name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=ema50, mode="lines",
        line=dict(color=TV_GOLD, width=1.2),  name="EMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=ema200, mode="lines",
        line=dict(color="#7e57c2", width=1.5), name="EMA200"), row=1, col=1)

    # VWAP (sempre visibile nel pannello prezzo)
    fig.add_trace(go.Scatter(x=dt, y=vwap_s, mode="lines",
        line=dict(color="#ff9800", width=1.5, dash="dot"),
        name="VWAP", showlegend=True), row=1, col=1)

    # ── Segnali Entry / Exit ──────────────────────
    if strategy != "Nessuna":
        e_d, e_p, x_d, x_p = _detect_signals(df, strategy)
        if e_d:
            fig.add_trace(go.Scatter(
                x=e_d, y=e_p, mode="markers",
                name="▲ Entry",
                marker=dict(symbol="triangle-up", size=12, color=TV_GREEN,
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate="<b>▲ ENTRY</b><br>%{x}<br>%{y:.2f}<extra></extra>",
            ), row=1, col=1)
        if x_d:
            fig.add_trace(go.Scatter(
                x=x_d, y=x_p, mode="markers",
                name="▼ Exit",
                marker=dict(symbol="triangle-down", size=12, color=TV_RED,
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate="<b>▼ EXIT</b><br>%{x}<br>%{y:.2f}<extra></extra>",
            ), row=1, col=1)

    # ── Row 2: Volume ─────────────────────────────
    fig.add_trace(go.Bar(x=dt, y=v,
        marker_color=v_colors, marker_line_width=0,
        name="Volume", showlegend=False), row=2, col=1)

    # ── Row 3: RSI ────────────────────────────────
    fig.add_trace(go.Scatter(x=dt, y=rsi_s, mode="lines",
        line=dict(color=TV_PURPLE, width=1.5),
        name="RSI", showlegend=False), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, row=3, col=1,
        fillcolor="rgba(239,83,80,0.08)", line_width=0)
    fig.add_hrect(y0=0, y1=30, row=3, col=1,
        fillcolor="rgba(38,166,154,0.08)", line_width=0)
    fig.add_hline(y=70, row=3, col=1,
        line=dict(color=TV_RED,   width=0.7, dash="dot"))
    fig.add_hline(y=50, row=3, col=1,
        line=dict(color=TV_GRAY,  width=0.5, dash="dot"))
    fig.add_hline(y=30, row=3, col=1,
        line=dict(color=TV_GREEN, width=0.7, dash="dot"))

    # ── Row 4: MACD o ADX ─────────────────────────
    if show_adx_row:
        # ADX per strategia ADX+EMA
        fig.add_trace(go.Scatter(x=dt, y=adx_s, mode="lines",
            line=dict(color=TV_RED, width=2),
            name="ADX(14)", showlegend=False,
            fill="tozeroy", fillcolor="rgba(239,83,80,0.06)"), row=4, col=1)
        fig.add_hline(y=25, row=4, col=1,
            line=dict(color=TV_GOLD, dash="dot", width=1.5))
        fig.add_hrect(y0=25, y1=80, row=4, col=1,
            fillcolor="rgba(255,152,0,0.07)", line_width=0)
    else:
        # MACD standard
        fig.add_trace(go.Bar(x=dt, y=macd_hist,
            marker_color=hist_colors, marker_line_width=0,
            name="Hist", showlegend=False, opacity=0.8), row=4, col=1)
        fig.add_trace(go.Scatter(x=dt, y=macd_line, mode="lines",
            line=dict(color=TV_BLUE, width=1.3),
            name="MACD", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=dt, y=macd_sig, mode="lines",
            line=dict(color="#ff9800", width=1.3),
            name="Signal", showlegend=False), row=4, col=1)
        fig.add_hline(y=0, row=4, col=1,
            line=dict(color=TV_BORDER, width=1))

    # ── Prezzo ultimo + variazione ────────────────
    last_price  = float(c.iloc[-1])
    first_price = float(c.dropna().iloc[0])
    chg_pct     = (last_price / first_price - 1) * 100
    chg_color   = TV_GREEN if chg_pct >= 0 else TV_RED
    arrow       = "▲" if chg_pct >= 0 else "▼"
    last_rsi    = float(rsi_s.dropna().iloc[-1]) if not rsi_s.dropna().empty else 0
    last_adx    = float(adx_s.dropna().iloc[-1]) if not adx_s.dropna().empty else 0
    strategy_badge = (
        f"  <span style='color:#ff9800;font-size:0.75em'>ADX {last_adx:.0f}</span>"
        if strategy == "ADX+EMA" else ""
    )

    title_text = (
        f"<b style='color:{color}'>{symbol}</b>"
        f"  <span style='color:{TV_GRAY};font-size:0.85em'>{name[:28]}</span><br>"
        f"<span style='color:{TV_TEXT}'>${last_price:.2f}</span>"
        f"  <span style='color:{chg_color}'>{arrow}{abs(chg_pct):.1f}%</span>"
        f"  <span style='color:{TV_GRAY};font-size:0.8em'>RSI {last_rsi:.1f}</span>"
        f"{strategy_badge}"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13), x=0.01, xanchor="left"),
        height=620,
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_PANEL,
        margin=dict(l=6, r=6, t=55, b=6),
        legend=dict(
            orientation="h", x=0.01, y=1.01,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9, color=TV_GRAY),
        ),
        xaxis_rangeslider_visible=False,
        font=dict(color=TV_TEXT, size=10),
        hovermode="x unified",
    )

    for row in [1, 2, 3, 4]:
        n_ = "" if row == 1 else str(row)
        fig.update_layout(**{
            f"xaxis{n_}": dict(
                showgrid=True, gridcolor=TV_BORDER,
                zeroline=False, linecolor=TV_BORDER,
                showticklabels=(row == 4),
            ),
            f"yaxis{n_}": dict(
                showgrid=True, gridcolor=TV_BORDER,
                zeroline=False, linecolor=TV_BORDER,
                tickfont=dict(size=9),
            ),
        })

    fig.update_layout(
        yaxis3=dict(range=[0, 100], tickvals=[30, 50, 70]),
    )
    if show_adx_row:
        fig.update_layout(yaxis4=dict(range=[0, 80], tickvals=[0, 25, 50]))

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
        f'Confronto tecnico fino a 4 ticker · v31.1</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Controlli ─────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 2, 2, 1])

    with ctrl2:
        period_label = st.selectbox(
            "📅 Periodo",
            list(PERIOD_MAP.keys()),
            index=3,
            key="compare_period",
        )
    with ctrl3:
        strategy_sel = st.selectbox(
            "📊 Segnali strategia",
            ["Nessuna", "RSI+VWAP", "ADX+EMA", "MACD", "EMA Cross"],
            key="compare_strategy",
            help="Aggiunge ▲ Entry / ▼ Exit sul grafico candele"
        )
    with ctrl4:
        st.write("")
        if st.button("🔄", key="compare_refresh", help="Svuota cache e ricarica"):
            st.cache_data.clear()
            st.rerun()

    period, interval = PERIOD_MAP[period_label]

    # ── Banner strategia selezionata ──────────────
    if strategy_sel != "Nessuna":
        strategy_rules = {
            "RSI+VWAP": ("📊", "#e91e63",
                          "▲ Entry: RSI incrocia sopra 30 + Prezzo > VWAP",
                          "▼ Exit: RSI incrocia sotto 70 o Prezzo < VWAP"),
            "ADX+EMA":  ("📈", "#ff9800",
                          "▲ Entry: EMA20 incrocia sopra EMA50 + ADX > 25",
                          "▼ Exit: EMA20 < EMA50 o ADX < 25"),
            "MACD":     ("⚡", "#2962ff",
                          "▲ Entry: MACD histogram incrocia sopra 0",
                          "▼ Exit: MACD histogram incrocia sotto 0"),
            "EMA Cross":("🔀", "#26a69a",
                          "▲ Entry: EMA20 incrocia sopra EMA50",
                          "▼ Exit: EMA20 incrocia sotto EMA50"),
        }
        icon, sc, entry_txt, exit_txt = strategy_rules[strategy_sel]
        c_l, c_r = st.columns(2)
        with c_l:
            st.markdown(
                f'<div style="background:#1e222d;border:1px solid #2a2e39;'
                f'border-left:4px solid {sc};border-radius:6px;'
                f'padding:7px 12px;margin-bottom:10px">'
                f'<span style="color:#787b86;font-size:0.65rem">ENTRY {icon}</span><br>'
                f'<span style="color:#d1d4dc;font-size:0.8rem;font-weight:600">{entry_txt}</span>'
                f'</div>', unsafe_allow_html=True)
        with c_r:
            st.markdown(
                f'<div style="background:#1e222d;border:1px solid #2a2e39;'
                f'border-left:4px solid #ef5350;border-radius:6px;'
                f'padding:7px 12px;margin-bottom:10px">'
                f'<span style="color:#787b86;font-size:0.65rem">EXIT</span><br>'
                f'<span style="color:#d1d4dc;font-size:0.8rem;font-weight:600">{exit_txt}</span>'
                f'</div>', unsafe_allow_html=True)

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
        fig = _build_chart(valid[syms[0]], syms[0], COLORS[0], strategy=strategy_sel)
        st.plotly_chart(fig, use_container_width=True, key=f"compare_c0")

    elif n == 2:
        cols = st.columns(2)
        for i, (sym, df) in enumerate(valid.items()):
            with cols[i]:
                fig = _build_chart(df, sym, COLORS[i], strategy=strategy_sel)
                st.plotly_chart(fig, use_container_width=True,
                                key=f"compare_c{i}")

    elif n == 3:
        cols = st.columns(3)
        for i, (sym, df) in enumerate(valid.items()):
            with cols[i]:
                fig = _build_chart(df, sym, COLORS[i], strategy=strategy_sel)
                st.plotly_chart(fig, use_container_width=True,
                                key=f"compare_c{i}")

    else:  # 4 ticker — 2x2
        row1 = st.columns(2)
        row2 = st.columns(2)
        grid = [row1[0], row1[1], row2[0], row2[1]]
        for i, (sym, df) in enumerate(valid.items()):
            with grid[i]:
                fig = _build_chart(df, sym, COLORS[i], strategy=strategy_sel)
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
