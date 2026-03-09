"""
analysis_chart.py  —  Grafico Analitico Avanzato  v29.0
══════════════════════════════════════════════════════════
Componenti:
  • Candlestick TV-style (sfondo #1e222d)
  • Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)
  • Supporti & Resistenze automatici (pivot fractals + swing highs/lows)
  • Trend Ribbon (EMA 8/21/55/200 colorato per direzione)
  • Volume profile (VP orizzontale) con POC/VAH/VAL
  • Squeeze Momentum (BB vs KC)
  • Segnali visivi: frecce buy/sell, etichette S/R

Uso:
    from analysis_chart import render_analysis_chart
    render_analysis_chart(row_full, key_suffix="")
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Colori TradingView ─────────────────────────────────────────────────────
TV = dict(
    bg_paper   = "#131722",
    bg_plot    = "#1e222d",
    grid       = "#2a2e39",
    grid_line  = "#363a45",
    text       = "#b2b5be",
    tick       = "#787b86",
    bull       = "#26a69a",   # teal verde TV
    bear       = "#ef5350",   # rosso TV
    bull_fill  = "rgba(38,166,154,0.85)",
    bear_fill  = "rgba(239,83,80,0.85)",
    ema8       = "#2962ff",   # blu
    ema21      = "#ff9800",   # arancio
    ema55      = "#9c27b0",   # viola
    ema200     = "#ffffff",   # bianco
    ichi_tenkan = "#2196f3",  # blu chiaro
    ichi_kijun  = "#ff6d00",  # arancio scuro
    ichi_cloud_bull = "rgba(38,166,154,0.18)",
    ichi_cloud_bear = "rgba(239,83,80,0.18)",
    ichi_chikou = "#ba68c8",  # lilla
    sr_support  = "#26a69a",  # teal
    sr_resist   = "#ef5350",  # rosso
    poc         = "#ffd700",  # oro
    va_fill     = "rgba(41,98,255,0.62)",
    va_out      = "rgba(120,123,134,0.40)",
    squeeze_on  = "#ef5350",
    squeeze_off = "#26a69a",
    momentum_pos = "rgba(38,166,154,0.75)",
    momentum_neg = "rgba(239,83,80,0.75)",
)

LAYOUT_BASE = dict(
    paper_bgcolor=TV["bg_paper"],
    plot_bgcolor =TV["bg_plot"],
    font=dict(color=TV["text"], family="Trebuchet MS, sans-serif", size=12),
    legend=dict(orientation="h", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
    margin=dict(l=0,r=0,t=60,b=0),
)


# ── Helper calcoli ─────────────────────────────────────────────────────────

def _ema(series, period):
    """EMA semplice su lista."""
    n = len(series)
    if n < period: return [None]*n
    result = [None]*n
    k = 2.0/(period+1)
    # Seed con SMA
    seed = sum(x for x in series[:period] if x is not None)
    valid = sum(1 for x in series[:period] if x is not None)
    if valid < period: return [None]*n
    prev = seed/period
    result[period-1] = prev
    for i in range(period, n):
        if series[i] is None: result[i]=None
        else:
            prev = series[i]*k + prev*(1-k)
            result[i] = prev
    return result


def _smma(series, period, shift=0):
    """Smoothed Moving Average (Wilder) con offset per Ichimoku."""
    n = len(series)
    if n < period+shift: return [None]*n
    result = [None]*n
    # Usa i valori shiftati
    data = series[:n-shift] if shift else series
    ema_v = _ema(data, period)
    if shift:
        result = [None]*shift + ema_v
    else:
        result = ema_v
    return result


def _ichimoku(highs, lows, closes):
    """
    Calcola Ichimoku Cloud.
    Ritorna dict con: tenkan, kijun, senkou_a, senkou_b, chikou
    """
    n = len(highs)
    def midpoint(hi, lo, period):
        res = [None]*n
        for i in range(period-1, n):
            h_max = max(hi[max(0,i-period+1):i+1])
            l_min = min(lo[max(0,i-period+1):i+1])
            res[i] = (h_max+l_min)/2
        return res

    tenkan  = midpoint(highs, lows, 9)   # Conversion Line
    kijun   = midpoint(highs, lows, 26)  # Base Line
    # Senkou A = (tenkan+kijun)/2 proiettato 26 avanti
    senkou_a_raw = [
        (tenkan[i]+kijun[i])/2 if tenkan[i] and kijun[i] else None
        for i in range(n)
    ]
    # Senkou B = midpoint(52) proiettato 26 avanti
    senkou_b_raw = midpoint(highs, lows, 52)
    # Proiezione +26 candle
    senkou_a = [None]*26 + senkou_a_raw[:n-26]
    senkou_b = [None]*26 + senkou_b_raw[:n-26]
    # Chikou = close spostato -26
    chikou   = closes[26:] + [None]*26

    return dict(tenkan=tenkan, kijun=kijun,
                senkou_a=senkou_a, senkou_b=senkou_b,
                chikou=chikou)


def _find_sr_levels(highs, lows, closes, dates, n_levels=6, lookback=5):
    """
    Supporti e resistenze via Fractal Pivots (Williams).
    Un pivot high: high[i] > high[i-k] per k=1..lookback
    Un pivot low:  low[i]  < low[i-k]  per k=1..lookback
    Restituisce liste di (price, date, tipo) più recenti.
    """
    n = len(highs)
    supports = []
    resistances = []
    lb = lookback
    for i in range(lb, n-lb):
        # Pivot High
        if all(highs[i]>highs[i-k] for k in range(1,lb+1)) and            all(highs[i]>highs[i+k] for k in range(1,lb+1)):
            resistances.append((highs[i], dates[i]))
        # Pivot Low
        if all(lows[i]<lows[i-k] for k in range(1,lb+1)) and            all(lows[i]<lows[i+k] for k in range(1,lb+1)):
            supports.append((lows[i], dates[i]))

    # Raggruppa livelli vicini (entro 0.5% dello stesso prezzo)
    def cluster(levels):
        if not levels: return []
        levels_sorted = sorted(levels, key=lambda x: x[0])
        clustered = []
        prev = levels_sorted[0]
        for price, date in levels_sorted[1:]:
            if abs(price - prev[0])/prev[0] < 0.005:
                # Prendi il più recente
                prev = (price, date) if date > prev[1] else prev
            else:
                clustered.append(prev)
                prev = (price, date)
        clustered.append(prev)
        return clustered

    sup_cl   = cluster(supports)
    res_cl   = cluster(resistances)
    cur_price = closes[-1] if closes else 0
    # Prendi i più vicini al prezzo corrente
    sup_cl   = sorted(sup_cl,   key=lambda x: abs(x[0]-cur_price))[:n_levels]
    res_cl   = sorted(res_cl,   key=lambda x: abs(x[0]-cur_price))[:n_levels]
    return sup_cl, res_cl


def _squeeze_momentum(highs, lows, closes, bb_len=20, kc_len=20, mom_len=12):
    """
    Squeeze Momentum (LazyBear):
    Squeeze ON quando BB dentro KC.
    Momentum = differenza linreg del delta prezzo.
    """
    n = len(closes)
    if n < bb_len+2: return [None]*n, [None]*n, [None]*n

    c = np.array(closes, dtype=float)
    h = np.array(highs,  dtype=float)
    l = np.array(lows,   dtype=float)

    # Bollinger Bands
    def sma_arr(arr, p):
        res = np.full(n, np.nan)
        for i in range(p-1, n): res[i]=arr[i-p+1:i+1].mean()
        return res
    def std_arr(arr, p):
        res = np.full(n, np.nan)
        for i in range(p-1, n): res[i]=arr[i-p+1:i+1].std()
        return res

    sma_bb = sma_arr(c, bb_len)
    std_bb = std_arr(c, bb_len)
    bb_up  = sma_bb + 2*std_bb
    bb_dn  = sma_bb - 2*std_bb

    # Keltner Channels
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr_arr = np.full(n, np.nan)
    for i in range(kc_len, n):
        atr_arr[i] = tr[i-kc_len:i].mean()
    kc_up = sma_arr(c, kc_len) + 1.5*atr_arr
    kc_dn = sma_arr(c, kc_len) - 1.5*atr_arr

    # Squeeze ON/OFF
    sqz_on  = (bb_dn>kc_dn) & (bb_up<kc_up)
    sqz_off = (bb_dn<kc_dn) | (bb_up>kc_up)

    # Momentum (linreg delta)
    def linreg(arr, p):
        res = np.full(n, np.nan)
        x = np.arange(p, dtype=float)
        for i in range(p-1, n):
            y = arr[i-p+1:i+1]
            if not np.isnan(y).any():
                m = np.polyfit(x, y, 1)[0]
                res[i] = m
        return res

    highest_h = np.full(n, np.nan)
    lowest_l  = np.full(n, np.nan)
    for i in range(mom_len-1, n):
        highest_h[i] = h[i-mom_len+1:i+1].max()
        lowest_l[i]  = l[i-mom_len+1:i+1].min()

    delta = c - (highest_h + lowest_l)/2
    delta -= sma_arr(delta, mom_len)
    momentum = linreg(delta, mom_len)

    return list(momentum), list(sqz_on.astype(float)), list(sqz_off.astype(float))


def _volume_profile(highs, lows, closes, vols, n_bins=40):
    """Volume Profile: ritorna (centers, vols, poc, vah, val)."""
    try:
        h=np.array(highs,float); l=np.array(lows,float); v=np.array(vols,float)
        pmin,pmax=l.min(),h.max()
        if pmax<=pmin or len(h)<5: return [],[],None,None,None
        bins=np.linspace(pmin,pmax,n_bins+1); centers=(bins[:-1]+bins[1:])/2
        vpvol=np.zeros(n_bins)
        for i in range(len(h)):
            if v[i]<=0 or h[i]<=l[i]: continue
            b0=int(np.searchsorted(bins,l[i],"left")); b1=int(np.searchsorted(bins,h[i],"right"))
            b0=max(0,min(b0,n_bins-1)); b1=max(0,min(b1,n_bins)); span=h[i]-l[i]
            for b in range(b0,b1):
                lo=max(bins[b],l[i]); hi=min(bins[b+1] if b+1<len(bins) else pmax,h[i])
                vpvol[b]+=v[i]*max(0,hi-lo)/span
        poc_i=int(np.argmax(vpvol)); poc=float(centers[poc_i])
        tot=vpvol.sum(); tgt=tot*0.70; acc=vpvol[poc_i]; lo_i=hi_i=poc_i
        while acc<tgt and (lo_i>0 or hi_i<n_bins-1):
            alo=vpvol[lo_i-1] if lo_i>0 else 0; ahi=vpvol[hi_i+1] if hi_i<n_bins-1 else 0
            if ahi>=alo and hi_i<n_bins-1: hi_i+=1; acc+=ahi
            elif lo_i>0:                   lo_i-=1; acc+=alo
            else:                           hi_i+=1; acc+=ahi
        return list(centers),list(vpvol),poc,float(centers[hi_i]),float(centers[lo_i])
    except Exception: return [],[],None,None,None


# ── Builder grafico ────────────────────────────────────────────────────────

def build_analysis_chart(row: pd.Series,
                         show_ichimoku=True,
                         show_sr=True,
                         show_ema_ribbon=True,
                         show_vp=True,
                         show_squeeze=True) -> go.Figure:
    """
    Costruisce il grafico analitico avanzato.
    Layout: 3 righe
      Row 1 (70%): Candlestick + Ichimoku + S/R + EMA Ribbon + VP
      Row 2 (18%): Volume a barre
      Row 3 (12%): Squeeze Momentum
    """
    cd = row.get("_chart_data")
    if not cd or not isinstance(cd, dict): return None
    dates  = cd.get("dates", [])
    opens  = cd.get("open",  [])
    highs  = cd.get("high",  [])
    lows   = cd.get("low",   [])
    closes = cd.get("close", [])
    vols   = cd.get("volume",[])
    if not dates or not closes or len(closes)<26: return None

    n = len(closes)
    n_rows = 3 if show_squeeze else 2
    heights = [0.68, 0.18, 0.14][:n_rows]

    # Layout 2 colonne se VP attivo (85% + 15%)
    if show_vp and vols:
        specs = [[{"secondary_y":False},{"secondary_y":False}]]*n_rows
        fig = make_subplots(rows=n_rows, cols=2,
                            shared_xaxes=False, shared_yaxes=False,
                            row_heights=heights, vertical_spacing=0.025,
                            column_widths=[0.84,0.16],
                            specs=specs, horizontal_spacing=0.004)
        vp_col = 2
    else:
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                            row_heights=heights, vertical_spacing=0.025)
        vp_col = None

    # ── 1. Candlestick ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color=TV["bull"], increasing_fillcolor=TV["bull_fill"],
        decreasing_line_color=TV["bear"], decreasing_fillcolor=TV["bear_fill"],
        name="Prezzo", showlegend=False
    ), row=1, col=1)

    # ── 2. EMA Ribbon ───────────────────────────────────────────────────
    if show_ema_ribbon:
        for period, color, name in [
            (8,   TV["ema8"],   "EMA8"),
            (21,  TV["ema21"],  "EMA21"),
            (55,  TV["ema55"],  "EMA55"),
            (200, TV["ema200"], "EMA200"),
        ]:
            ev = _ema(closes, period)
            if any(v is not None for v in ev):
                width = 2.0 if period==200 else 1.5
                dash  = "dot" if period==200 else "solid"
                fig.add_trace(go.Scatter(
                    x=dates, y=ev,
                    line=dict(color=color, width=width, dash=dash),
                    name=name
                ), row=1, col=1)
        # Riempimento tra EMA8 e EMA21 per trend direction
        e8  = _ema(closes, 8)
        e21 = _ema(closes, 21)
        valid = [(i, e8[i], e21[i]) for i in range(n)
                 if e8[i] is not None and e21[i] is not None]
        if valid:
            idxs = [v[0] for v in valid]
            v8   = [v[1] for v in valid]
            v21  = [v[2] for v in valid]
            fill_dates = [dates[i] for i in idxs]
            # Riempimento verde se EMA8>EMA21, rosso altrimenti
            last_cross = 0
            segments = []
            for i in range(1, len(v8)):
                if (v8[i]>=v21[i]) != (v8[i-1]>=v21[i-1]):
                    segments.append((last_cross, i, v8[i-1]>=v21[i-1]))
                    last_cross = i
            segments.append((last_cross, len(v8), v8[-1]>=v21[-1]))
            for s0, s1, bull in segments:
                fill_c = "rgba(38,166,154,0.12)" if bull else "rgba(239,83,80,0.10)"
                fig.add_trace(go.Scatter(
                    x=fill_dates[s0:s1]+fill_dates[s0:s1][::-1],
                    y=v8[s0:s1]+v21[s0:s1][::-1],
                    fill="toself", fillcolor=fill_c,
                    line=dict(width=0), showlegend=False,
                    hoverinfo="skip"
                ), row=1, col=1)

    # ── 3. Ichimoku Cloud ───────────────────────────────────────────────
    if show_ichimoku:
        ichi = _ichimoku(highs, lows, closes)
        fig.add_trace(go.Scatter(x=dates, y=ichi["tenkan"],
            line=dict(color=TV["ichi_tenkan"], width=1.5),
            name="Tenkan (9)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ichi["kijun"],
            line=dict(color=TV["ichi_kijun"], width=1.5, dash="dash"),
            name="Kijun (26)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ichi["chikou"],
            line=dict(color=TV["ichi_chikou"], width=1, dash="dot"),
            name="Chikou"), row=1, col=1)
        # Cloud: senkou A e B con fill
        sa = ichi["senkou_a"]; sb = ichi["senkou_b"]
        fig.add_trace(go.Scatter(
            x=dates+dates[::-1],
            y=sa+sb[::-1],
            fill="toself",
            fillcolor=TV["ichi_cloud_bull"],
            line=dict(width=0), showlegend=False, name="Cloud Bull",
            hoverinfo="skip"
        ), row=1, col=1)
        # Cloud bear (dove sb > sa)
        bear_sa = [s if (sa[i] is not None and sb[i] is not None and sb[i]>sa[i]) else None
                   for i,s in enumerate(sa)]
        bear_sb = [s if (sa[i] is not None and sb[i] is not None and sb[i]>sa[i]) else None
                   for i,s in enumerate(sb)]
        fig.add_trace(go.Scatter(
            x=dates+dates[::-1],
            y=bear_sa+bear_sb[::-1],
            fill="toself",
            fillcolor=TV["ichi_cloud_bear"],
            line=dict(width=0), showlegend=False, name="Cloud Bear",
            hoverinfo="skip"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=sa,
            line=dict(color=TV["bull"], width=1),
            name="Senkou A"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=sb,
            line=dict(color=TV["bear"], width=1),
            name="Senkou B"), row=1, col=1)

    # ── 4. Supporti & Resistenze ────────────────────────────────────────
    if show_sr and len(closes) > 20:
        supports, resistances = _find_sr_levels(highs, lows, closes, dates)
        cur_price = closes[-1]
        for price, date in supports:
            strength = 1 + min(2, abs(price-cur_price)/cur_price * 20)
            fig.add_hline(y=price,
                line=dict(color=TV["sr_support"], width=strength, dash="dash"),
                annotation_text=f" S {price:.2f}",
                annotation_font_color=TV["sr_support"],
                annotation_font_size=9,
                row=1, col=1)
        for price, date in resistances:
            strength = 1 + min(2, abs(price-cur_price)/cur_price * 20)
            fig.add_hline(y=price,
                line=dict(color=TV["sr_resist"], width=strength, dash="dash"),
                annotation_text=f" R {price:.2f}",
                annotation_font_color=TV["sr_resist"],
                annotation_font_size=9,
                row=1, col=1)

    # ── 5. Volume bars (row 2) ──────────────────────────────────────────
    if vols:
        vcols = [TV["bull_fill"] if c>=o else TV["bear_fill"]
                 for c,o in zip(closes,opens)]
        fig.add_trace(go.Bar(
            x=dates, y=vols,
            marker_color=vcols, name="Volume", showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Vol", tickfont=dict(size=8,color=TV["tick"]),
                         row=2, col=1)

    # ── 6. Volume Profile (col 2 row 1) ─────────────────────────────────
    if vp_col and vols:
        vc,vv,poc,vah,val = _volume_profile(highs,lows,closes,vols)
        if vc:
            mx=max(vv); norm=[x/mx for x in vv]
            binw=(vc[1]-vc[0]) if len(vc)>1 else 0
            clrs=[]
            for i,p in enumerate(vc):
                if poc and binw and abs(p-poc)<binw: clrs.append(TV["poc"])
                elif val and vah and val<=p<=vah:     clrs.append(TV["va_fill"])
                else:                                  clrs.append(TV["va_out"])
            fig.add_trace(go.Bar(x=norm, y=vc, orientation="h",
                marker=dict(color=clrs,line=dict(width=0)),
                name="VP", showlegend=False,
                hovertemplate="P:%{y:.2f}<br>Vol:%{customdata:,.0f}<extra>VP</extra>",
                customdata=vv,
            ), row=1, col=vp_col)
            for lvl,col,lbl in [(poc,TV["poc"],"POC"),(vah,"#2962ff","VAH"),(val,"#2962ff","VAL")]:
                if lvl:
                    fig.add_hline(y=lvl,
                        line=dict(color=col,width=1.5,dash="dot" if lbl!="POC" else "solid"),
                        annotation_text=f" {lbl}",annotation_font_color=col,
                        annotation_font_size=8, row=1, col=vp_col)
            fig.update_xaxes(showticklabels=False,showgrid=False,zeroline=False,col=vp_col)
            for rv in range(1, n_rows+1):
                fig.update_yaxes(showticklabels=False,showgrid=False,col=vp_col,row=rv)

    # ── 7. Squeeze Momentum (row 3) ─────────────────────────────────────
    if show_squeeze and n_rows==3:
        mom, sqz_on, sqz_off = _squeeze_momentum(highs,lows,closes)
        if any(m is not None for m in mom):
            # Istogramma momentum colorato per direzione e accelerazione
            mom_colors = []
            for i,m in enumerate(mom):
                if m is None: mom_colors.append(TV["tick"]); continue
                prev = next((mom[j] for j in range(i-1,-1,-1) if mom[j] is not None), 0)
                if m>=0:
                    mom_colors.append("rgba(38,166,154,0.85)" if m>=prev else "rgba(38,166,154,0.45)")
                else:
                    mom_colors.append("rgba(239,83,80,0.85)" if m<=prev else "rgba(239,83,80,0.45)")
            fig.add_trace(go.Bar(x=dates, y=mom,
                marker_color=mom_colors,
                name="Momentum", showlegend=False
            ), row=3, col=1)
            # Punti squeeze ON/OFF
            sqz_y  = [closes[i]*0 if sqz_on[i] else None for i in range(n)]
            sqz_y2 = [closes[i]*0 if sqz_off[i] else None for i in range(n)]
            fig.add_trace(go.Scatter(x=dates,y=sqz_y,
                mode="markers",
                marker=dict(color=TV["squeeze_on"],size=4,symbol="circle"),
                name="Sqz ON", showlegend=False
            ), row=3, col=1)
            fig.add_trace(go.Scatter(x=dates,y=sqz_y2,
                mode="markers",
                marker=dict(color=TV["squeeze_off"],size=4,symbol="cross"),
                name="Sqz OFF", showlegend=False
            ), row=3, col=1)
            fig.update_yaxes(title_text="Sqz Mom",tickfont=dict(size=8,color=TV["tick"]),
                             row=3, col=1)

    # ── Layout finale ────────────────────────────────────────────────────
    tkr   = row.get("Ticker","")
    nome  = row.get("Nome","")
    price = row.get("Prezzo","")
    rsi   = row.get("RSI","")
    sq    = "  🔥" if row.get("Squeeze") else ""

    fig.update_layout(**LAYOUT_BASE,
        title=dict(
            text=f"<b>{tkr}</b>  {nome}  |  {price}  |  RSI {rsi}{sq}"
                 f"  <span style=\'color:#787b86;font-size:11px\'>— Analisi Avanzata</span>",
            font=dict(color="#50c4e0",size=14), x=0.01, xanchor="left"
        ),
        height=720,
    )
    for r in range(1, n_rows+1):
        fig.update_xaxes(gridcolor=TV["grid"],gridwidth=1,
                         showline=True,linecolor=TV["grid_line"],
                         tickfont=dict(color=TV["tick"],size=10),
                         row=r,col=1)
        fig.update_yaxes(gridcolor=TV["grid"],gridwidth=1,
                         showline=True,linecolor=TV["grid_line"],
                         tickfont=dict(color=TV["tick"],size=10),
                         row=r,col=1)
    return fig


# ── Entry point Streamlit ──────────────────────────────────────────────────

def render_analysis_chart(row: pd.Series, key_suffix: str = ""):
    """
    Renderizza il grafico analitico in Streamlit.
    Chiamare dentro un tab o expander.
    """
    tkr = row.get("Ticker","")

    st.markdown('''<div style="background:#1e222d;border-left:3px solid #2962ff;
        padding:8px 16px;border-radius:0 6px 6px 0;margin-bottom:12px">
        <span style="color:#50c4e0;font-size:0.85rem;letter-spacing:1px">
        📐 ANALISI AVANZATA — Ichimoku · Trend · S/R · Volume Profile · Squeeze
        </span></div>''', unsafe_allow_html=True)

    # Controlli
    ac1,ac2,ac3,ac4,ac5,ac6 = st.columns(6)
    with ac1: show_ichi  = st.checkbox("☁️ Ichimoku",    value=True,  key=f"an_ichi_{tkr}_{key_suffix}")
    with ac2: show_sr    = st.checkbox("📌 S/R Levels",  value=True,  key=f"an_sr_{tkr}_{key_suffix}")
    with ac3: show_rib   = st.checkbox("🎀 EMA Ribbon",  value=True,  key=f"an_rib_{tkr}_{key_suffix}")
    with ac4: show_vp    = st.checkbox("📊 Vol Profile", value=True,  key=f"an_vp_{tkr}_{key_suffix}")
    with ac5: show_sqz   = st.checkbox("⚡ Squeeze",     value=True,  key=f"an_sqz_{tkr}_{key_suffix}")
    with ac6:
        st.write("")
        if st.button("🔄",key=f"an_ref_{tkr}_{key_suffix}",help="Aggiorna"): st.rerun()

    fig = build_analysis_chart(row,
        show_ichimoku  = show_ichi,
        show_sr        = show_sr,
        show_ema_ribbon= show_rib,
        show_vp        = show_vp,
        show_squeeze   = show_sqz,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True,
                        key=f"analysis_{tkr}_{key_suffix}")
    else:
        st.info("Dati non sufficienti per il grafico analitico (min 52 candele).")
