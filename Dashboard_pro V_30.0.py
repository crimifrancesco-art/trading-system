# -*- coding: utf-8 -*-
import io
import time
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# ── Import robusti: fallback gracile se un modulo non è aggiornato ──────────
try:
    from utils.db import (
        init_db, reset_watchlist_db, add_to_watchlist, load_watchlist,
        DB_PATH, save_scan_history, load_scan_history, load_scan_snapshot,
        delete_from_watchlist, move_watchlist_rows, rename_watchlist,
        update_watchlist_note, save_grid_layout, load_grid_layout,
    )
except ImportError as _e:
    st.error(f"❌ Errore import utils.db: {_e}"); st.stop()

# ── GitHub Sync (watchlist persistente tra deploy) ──────────────────────────
try:
    from utils.github_sync import (
        pull_watchlist        as _gh_pull,
        push_watchlist        as _gh_push,
        sync_status           as _gh_status,
        gh_add_to_watchlist,
        gh_delete_from_watchlist,
        gh_rename_watchlist,
        gh_move_watchlist_rows,
        gh_update_watchlist_note,
        gh_reset_watchlist_by_name,
    )
    _GH_SYNC = True
except ImportError:
    _GH_SYNC = False
    gh_add_to_watchlist        = add_to_watchlist
    gh_delete_from_watchlist   = delete_from_watchlist
    gh_rename_watchlist        = rename_watchlist
    gh_move_watchlist_rows     = move_watchlist_rows
    gh_update_watchlist_note   = update_watchlist_note
    from utils.db import reset_watchlist_by_name
    gh_reset_watchlist_by_name = reset_watchlist_by_name

# Funzioni v28 opzionali (non presenti nel db vecchio → stub silenziosi)
try:
    from utils.db import save_signals
except ImportError:
    def save_signals(*a, **k): pass

try:
    from utils.db import cache_stats
except ImportError:
    def cache_stats(): return {"fresh":0,"stale":0,"size_mb":0,"total_entries":0}

try:
    from utils.db import cache_clear
except ImportError:
    def cache_clear(*a, **k): pass

# Scanner: prova scan_universe (v28), fallback a scan_ticker (v27)
try:
    from utils.scanner import load_universe, scan_universe, scan_ticker
    _HAS_SCAN_UNIVERSE = True
except ImportError:
    from utils.scanner import load_universe, scan_ticker
    _HAS_SCAN_UNIVERSE = False

    def scan_universe(universe, e_h, p_rmin, p_rmax, r_poc,
                      vol_ratio_hot=1.5, cache_enabled=True, finviz_enabled=False,
                      n_workers=8, progress_callback=None):
        import concurrent.futures, threading, time
        rep, rrea = [], []
        lock = threading.Lock(); counter = [0]; t0 = time.time()
        def _one(tkr):
            ep, rea = scan_ticker(tkr, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
            with lock:
                counter[0] += 1
                if progress_callback: progress_callback(counter[0], len(universe), tkr)
            return ep, rea
        nw = min(max(n_workers,1), 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as ex:
            for fut in concurrent.futures.as_completed({ex.submit(_one,t):t for t in universe}):
                try:
                    ep, rea = fut.result()
                    if ep:  rep.append(ep)
                    if rea: rrea.append(rea)
                except Exception: pass
        df_ep  = pd.DataFrame(rep)  if rep  else pd.DataFrame()
        df_rea = pd.DataFrame(rrea) if rrea else pd.DataFrame()
        stats  = {"elapsed_s": round(time.time()-t0,1), "cache_hits": 0,
                  "downloaded": len(universe), "workers": nw, "total": len(universe),
                  "ep_found": len(rep), "rea_found": len(rrea), "finviz": False}
        return df_ep, df_rea, stats

# Backtest tab opzionale — wrappato per gestire errori db v27
try:
    from utils.orderflow_tab import render_orderflow_tab as _of_render
except Exception:
    _of_render = None
try:
    from utils.backtest_tab import render_backtest_tab as _bt_orig
    def render_backtest_tab():
        try:
            _bt_orig()
        except Exception as _e:
            st.error(f"❌ Errore Backtest: {_e}")
            import traceback as _tbc; st.code(_tbc.format_exc())
    _HAS_BACKTEST = True
except ImportError as _bt_ie:
    _HAS_BACKTEST = False
    def render_backtest_tab():
        st.warning(f"⚠️ backtest_tab.py non trovato: {_bt_ie}")
        st.info("Carica utils/backtest_tab.py nel repo e fai redeploy.")
# =========================================================================
# ENRICH: normalizza e arricchisce DataFrame dallo scanner
# Compatibile con scanner v22 (repo) e v28 (aggiornato)
# =========================================================================
def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge/ricalcola colonne che il vecchio scanner.py non produce:
    - Stato_Pro  con soglia >= 6 (il vecchio usa >= 8, troppo restrittivo)
    - Stato_Early assicurato
    - Ser_OK / Ser_Score  (metodo Serafini — 6 criteri tecnici)
    - FV_OK  / FV_Score   (filtri Finviz base)
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    df = df.copy()
        # ── Normalizza nomi colonne camelCase → underscore (compatibilità scanner v28) ─
    _col_map = {
        "ProScore": "Pro_Score", "EarlyScore": "Early_Score",
        "QualityScore": "Quality_Score", "StatoEarly": "Stato_Early",
        "StatoPro": "Stato_Pro", "OBVTrend": "OBV_Trend",
        "VolRatio": "Vol_Ratio", "WeeklyBull": "Weekly_Bull",
        "VolToday": "Vol_Today", "Vol7dAvg": "Vol_7d_Avg",
        "AvgVol20": "Avg_Vol_20", "RelVol": "Rel_Vol",
        "ATRExp": "ATR_Exp", "RSIDiv": "RSI_Div",
        "SerOK": "Ser_OK", "SerScore": "Ser_Score",
        "FVOK": "FV_OK", "FVScore": "FV_Score",
        "MarketCap": "MarketCap",  # già corretto
        "chartdata": "_chart_data", "qualitycomponents": "_quality_components",
    }
    df = df.rename(columns={k: v for k, v in _col_map.items() if k in df.columns})

    # ── Stato_Pro con soglia 6 ───────────────────────────────────────────
    if "Pro_Score" in df.columns:
        df["Stato_Pro"] = df["Pro_Score"].apply(
            lambda x: "PRO" if pd.notna(x) and float(x) >= 4 else "-")

    # ── Stato_Early assicurato ───────────────────────────────────────────
    if "Stato_Early" not in df.columns:
        if "Early_Score" in df.columns:
            df["Stato_Early"] = df["Early_Score"].apply(
                lambda x: "EARLY" if pd.notna(x) and float(x) > 0 else "-")
        else:
            df["Stato_Early"] = "-"

    # ── Ser_OK / Ser_Score ───────────────────────────────────────────────
    if "RSI" in df.columns and "OBV_Trend" in df.columns and "Vol_Ratio" in df.columns:
        pr  = df["Prezzo"]   if "Prezzo"   in df.columns else pd.Series(0.0, index=df.index)
        e20 = df["EMA20"]    if "EMA20"    in df.columns else pd.Series(dtype=float)
        e50 = df["EMA50"]    if "EMA50"    in df.columns else pd.Series(dtype=float)

        c1 = df["RSI"] > 50
        c2 = (pr > e20)    if "EMA20" in df.columns else (df["Quality_Score"] >= 4)
        c3 = (e20 > e50)   if ("EMA20" in df.columns and "EMA50" in df.columns)                            else (df["Quality_Score"] >= 6)
        c4 = df["OBV_Trend"] == "UP"
        c5 = df["Vol_Ratio"] > 1.0
        c6_raw = df.get("Earnings_Soon", pd.Series(False, index=df.index))
        c6 = ~c6_raw.astype(bool)

        df["Ser_OK"]    = c1 & c2 & c3 & c4 & c5 & c6
        df["Ser_Score"] = (c1.astype(int) + c2.astype(int) + c3.astype(int) +
                           c4.astype(int) + c5.astype(int) + c6.astype(int))

    # ── FV_OK / FV_Score ─────────────────────────────────────────────────
    if "Prezzo" in df.columns and "Vol_Ratio" in df.columns:
        pr    = df["Prezzo"]
        f1    = pr > 10
        vol7  = df.get("Vol_7d_Avg", pd.Series(0, index=df.index))
        f2    = vol7.fillna(0) > 500_000
        f3    = df["Vol_Ratio"] > 1.0
        e20   = df["EMA20"] if "EMA20" in df.columns else None
        e50   = df["EMA50"] if "EMA50" in df.columns else None
        if e20 is not None:
            f4 = pr > e20
            f5 = pr > e50
        else:
            qs = df.get("Quality_Score", pd.Series(0, index=df.index))
            f4 = qs >= 4
            f5 = qs >= 6

        df["FV_OK"]    = f1 & f2 & f3 & f4 & f5
        df["FV_Score"] = (f1.astype(int) + f2.astype(int) + f3.astype(int) +
                          f4.astype(int) + f5.astype(int))
    return df


# =========================================================================
# CSS
# =========================================================================
DARK_CSS = """
<style>
/* ── TradingView-style skin ─────────────────────────────────── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="block-container"]{
    background-color:#131722 !important; color:#d1d4dc !important;
    font-family:'Trebuchet MS','Segoe UI',sans-serif !important;}
[data-testid="stSidebar"]{background-color:#1e222d !important;border-right:1px solid #2a2e39 !important;}
[data-testid="stSidebar"] *{color:#d1d4dc !important;}
h1{color:#2962ff !important;font-family:'Trebuchet MS',sans-serif !important;
   letter-spacing:1px;text-shadow:0 0 16px #2962ff44;}
h2,h3{color:#50c4e0 !important;font-family:'Trebuchet MS',sans-serif !important;}
.stCaption,small{color:#6b7280 !important;}
[data-testid="stTabs"] button{background:#131722 !important;color:#787b86 !important;
    border-bottom:2px solid transparent !important;
    font-family:'Trebuchet MS',sans-serif !important;font-size:0.83rem !important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#2962ff !important;border-bottom:2px solid #2962ff !important;
    background:#1e222d !important;}
[data-testid="stMetric"]{background:#1e222d !important;border:1px solid #2a2e39 !important;
    border-radius:6px !important;padding:12px 16px !important;}
[data-testid="stMetricLabel"]{color:#787b86 !important;font-size:0.75rem !important;}
[data-testid="stMetricValue"]{color:#26a69a !important;font-size:1.6rem !important;
    font-family:'Trebuchet MS',sans-serif !important;font-weight:700 !important;}
[data-testid="stButton"]>button{background:#1e222d !important;
    color:#d1d4dc !important;border:1px solid #363a45 !important;
    border-radius:4px !important;font-family:'Trebuchet MS',sans-serif !important;transition:all 0.15s;}
[data-testid="stButton"]>button:hover{background:#2a2e39 !important;border-color:#50c4e0 !important;color:#ffffff !important;}
[data-testid="stButton"]>button[kind="primary"]{background:#2962ff !important;
    border-color:#2962ff !important;color:#ffffff !important;font-size:1rem !important;}
[data-testid="stButton"]>button[kind="secondary"]{background:#1e222d !important;
    color:#ef5350 !important;border:1px solid #ef535055 !important;}
[data-testid="stDownloadButton"]>button{background:#0d1117 !important;color:#58a6ff !important;
    border:1px solid #1f3a5f !important;border-radius:6px !important;}
[data-testid="stExpander"]{background:#0d1117 !important;border:1px solid #1f2937 !important;border-radius:8px !important;}
[data-testid="stExpander"] summary{color:#58a6ff !important;}
hr{border-color:#1f2937 !important;}
.ag-root-wrapper{background:#1e222d !important;border:1px solid #2a2e39 !important;border-radius:4px !important;}
.ag-header{background:#131722 !important;border-bottom:1px solid #363a45 !important;}
.ag-header-cell-label{color:#50c4e0 !important;font-family:'Trebuchet MS',sans-serif !important;
    font-size:0.79rem !important;letter-spacing:0.5px;text-transform:uppercase;}
.ag-header-cell-resize{background:#363a45 !important;}
.ag-row{background:#1e222d !important;border-bottom:1px solid #2a2e39 !important;}
.ag-row:hover{background:#2a2e39 !important;}
.ag-row-selected{background:rgba(41,98,255,0.18) !important;border-left:3px solid #2962ff !important;}
.ag-cell{color:#d1d4dc !important;font-family:'Trebuchet MS',sans-serif !important;font-size:0.83rem !important;}
.ag-paging-panel{background:#131722 !important;color:#787b86 !important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0a0e1a;}
::-webkit-scrollbar-thumb{background:#1f2937;border-radius:3px;}
.section-pill{display:inline-block;background:#1e222d;
    border-left:3px solid #2962ff;border-radius:0 4px 4px 0;padding:5px 16px;
    font-family:'Trebuchet MS',sans-serif;font-size:0.82rem;color:#50c4e0;
    letter-spacing:1px;margin-bottom:14px;}
.wl-card{background:linear-gradient(135deg,#0d1117 0%,#111827 100%);
    border:1px solid #1f2937;border-radius:12px;padding:14px 18px;margin-bottom:8px;transition:border-color 0.2s;}
.wl-card:hover{border-color:#374151;}
.wl-card-ticker{font-family:'Courier New',monospace;font-size:1.05rem;font-weight:bold;color:#00ff88;letter-spacing:1px;}
.wl-card-name{color:#8b949e;font-size:0.82rem;margin-top:2px;}
.wl-card-badge{display:inline-block;border-radius:10px;padding:2px 8px;font-size:0.72rem;font-weight:bold;margin-right:4px;}
.badge-green{background:rgba(0,255,136,0.15);color:#00ff88;border:1px solid #00ff8844;}
.badge-orange{background:rgba(245,158,11,0.15);color:#f59e0b;border:1px solid #f59e0b44;}
.badge-red{background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid #ef444444;}
.badge-blue{background:rgba(88,166,255,0.15);color:#58a6ff;border:1px solid #58a6ff44;}
.badge-gray{background:rgba(107,114,128,0.15);color:#6b7280;border:1px solid #6b728044;}
.badge-purple{background:rgba(167,139,250,0.15);color:#a78bfa;border:1px solid #a78bfa44;}
.legend-table{width:100%;border-collapse:collapse;font-family:'Courier New',monospace;font-size:0.82rem;}
.legend-table th{color:#58a6ff;border-bottom:1px solid #1f2937;padding:6px 10px;text-align:left;}
.legend-table td{color:#c9d1d9;border-bottom:1px solid #1a2233;padding:5px 10px;}
.legend-table tr:hover td{background:#131d2e;}
.legend-col-name{color:#00ff88;font-weight:bold;}
.legend-col-range{color:#f59e0b;}
.crit-ok{color:#00ff88;font-weight:bold;}
.crit-no{color:#ef4444;}
</style>
"""

PLOTLY_DARK = dict(
    paper_bgcolor="#131722",
    plot_bgcolor="#1e222d",
    font=dict(color="#b2b5be", family="Trebuchet MS, sans-serif", size=12),
    xaxis=dict(gridcolor="#2a2e39", zerolinecolor="#363a45",
               linecolor="#363a45", tickfont=dict(color="#787b86",size=10)),
    yaxis=dict(gridcolor="#2a2e39", zerolinecolor="#363a45",
               linecolor="#363a45", tickfont=dict(color="#787b86",size=10)),
)
# =========================================================================
# FORMATTING HELPERS  (inline — non richiedono utils.formatting)
# =========================================================================
def _fmt_large(v):
    """Abbrevia numeri grandi: 1234567 → '1.2M', 12345678901 → '12.3B'"""
    try:
        v = float(v)
        if v != v or v <= 0: return "—"   # NaN o zero
        if v >= 1e12: return f"{v/1e12:.1f}T"
        if v >= 1e9:  return f"{v/1e9:.1f}B"
        if v >= 1e6:  return f"{v/1e6:.1f}M"
        if v >= 1e3:  return f"{v/1e3:.0f}K"
        return "—"  # valori irrisori non ha senso mostrarli
    except Exception:
        return "—"

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonne _fmt usate dal display."""
    df = df.copy()
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df["Prezzo"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df["MarketCap"].apply(
            lambda x: _fmt_large(x) if (pd.notna(x) and not (isinstance(x,float) and (x!=x))
                      and float(x) > 1_000_000) else "—")
    if "EMA200" in df.columns:
        df["EMA200_fmt"] = df["EMA200"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) and not (isinstance(x,float) and (x!=x)) else "—")
    return df

def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara DataFrame per visualizzazione AgGrid:
    - Rimuove colonne interne (prefisso _)
    - Converte bool numpy in bool Python
    - Resetta indice
    """
    df = df.copy()
    drop = [c for c in df.columns if c.startswith("_")]
    df   = df.drop(columns=drop, errors="ignore")
    for col in df.columns:
        try:
            df[col] = df[col].apply(
                lambda x: bool(x)  if isinstance(x, np.bool_)   else
                          float(x) if isinstance(x, np.floating) else
                          int(x)   if isinstance(x, np.integer)  else
                          None     if isinstance(x, float) and (np.isnan(x) or np.isinf(x))
                          else x
            )
        except Exception:
            pass
    return df.reset_index(drop=True)



# =========================================================================
# INDICATORI TECNICI (per grafici)
# =========================================================================
def _sma(arr, n):   return pd.Series(arr).rolling(n).mean().tolist()
def _rsi_calc(arr, n=14):
    s=pd.Series(arr); d=s.diff()
    up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(com=n-1,adjust=False).mean()/dn.ewm(com=n-1,adjust=False).mean()
    return (100-100/(1+rs)).tolist()
def _macd_calc(arr,fast=12,slow=26,sig=9):
    s=pd.Series(arr)
    m=s.ewm(span=fast,adjust=False).mean()-s.ewm(span=slow,adjust=False).mean()
    sg=m.ewm(span=sig,adjust=False).mean()
    return m.tolist(),sg.tolist(),(m-sg).tolist()
def _parabolic_sar(highs,lows,af_start=0.02,af_max=0.2):
    h=list(highs);l=list(lows);n=len(h)
    if n<2: return [None]*n,[0]*n
    sar=[0.0]*n;bull=[True]*n;ep=h[0];af=af_start;sar[0]=l[0]
    for i in range(1,n):
        pb=bull[i-1];ps=sar[i-1]
        if pb:
            ns=min(ps+af*(ep-ps),l[i-1],l[i-2] if i>=2 else l[i-1])
            if l[i]<ns: bull[i]=False;sar[i]=ep;ep=l[i];af=af_start
            else:
                bull[i]=True;sar[i]=ns
                if h[i]>ep: ep=h[i];af=min(af+af_start,af_max)
        else:
            ns=max(ps+af*(ep-ps),h[i-1],h[i-2] if i>=2 else h[i-1])
            if h[i]>ns: bull[i]=True;sar[i]=ep;ep=h[i];af=af_start
            else:
                bull[i]=False;sar[i]=ns
                if l[i]<ep: ep=l[i];af=min(af+af_start,af_max)
    return sar,[1 if b else -1 for b in bull]

# =========================================================================
# CHART BUILDER
# =========================================================================

def _calc_volume_profile(highs, lows, closes, vols, n_bins=36):
    """
    Volume Profile: distribuzione volume per livello di prezzo.
    Restituisce (bin_centers, vol_per_bin, poc, vah, val)
    POC = Point of Control  |  VAH/VAL = Value Area (70%)
    """
    try:
        import numpy as _np
        h=_np.array(highs,dtype=float); l=_np.array(lows,dtype=float)
        v=_np.array(vols,dtype=float)
        pmin,pmax = l.min(), h.max()
        if pmax<=pmin or len(h)<5: return [],[],None,None,None
        bins   = _np.linspace(pmin, pmax, n_bins+1)
        centers= (bins[:-1]+bins[1:])/2
        vpvol  = _np.zeros(n_bins)
        for i in range(len(h)):
            if v[i]<=0 or h[i]<=l[i]: continue
            b0=int(_np.searchsorted(bins,l[i],'left'))
            b1=int(_np.searchsorted(bins,h[i],'right'))
            b0=max(0,min(b0,n_bins-1)); b1=max(0,min(b1,n_bins))
            span=h[i]-l[i]
            for b in range(b0,b1):
                lo=max(bins[b],l[i]); hi=min(bins[b+1] if b+1<len(bins) else pmax,h[i])
                vpvol[b]+=v[i]*max(0,hi-lo)/span
        poc_i=int(_np.argmax(vpvol))
        poc=float(centers[poc_i])
        # Value Area 70%
        tot=vpvol.sum(); tgt=tot*0.70
        acc=vpvol[poc_i]; lo_i=hi_i=poc_i
        while acc<tgt and (lo_i>0 or hi_i<n_bins-1):
            add_lo=vpvol[lo_i-1] if lo_i>0 else 0
            add_hi=vpvol[hi_i+1] if hi_i<n_bins-1 else 0
            if add_hi>=add_lo and hi_i<n_bins-1: hi_i+=1; acc+=add_hi
            elif lo_i>0:                          lo_i-=1; acc+=add_lo
            else:                                  hi_i+=1; acc+=add_hi
        vah=float(centers[hi_i]); val=float(centers[lo_i])
        return list(centers),list(vpvol),poc,vah,val
    except Exception: return [],[],None,None,None


def build_full_chart(row: pd.Series, indicators: list) -> go.Figure:
    cd=row.get("_chart_data")
    if not cd or not isinstance(cd,dict): return None
    dates=cd.get("dates",[]); opens=cd.get("open",[])
    highs=cd.get("high",[]); lows=cd.get("low",[])
    closes=cd.get("close",[]); vols=cd.get("volume",[])
    ema20=cd.get("ema20",[]); ema50=cd.get("ema50",[])
    ema200=cd.get("ema200",[])
    bb_up=cd.get("bb_up",[]); bb_dn=cd.get("bb_dn",[])
    if not dates or not closes: return None

    show_sma=("SMA 9 & 21 + RSI" in indicators)
    show_macd=("MACD" in indicators)
    show_sar=("Parabolic SAR" in indicators)
    show_alligator=("Alligator + Vortex" in indicators)

    cur=2; row_rsi=None; row_macd=None; row_vortex=None
    if show_sma:        row_rsi=cur;    cur+=1
    if show_macd:       row_macd=cur;   cur+=1
    if show_alligator:  row_vortex=cur; cur+=1
    row_vol=cur; n_rows=cur

    ht={2:[0.65,0.15],3:[0.52,0.18,0.13],4:[0.44,0.17,0.15,0.12],5:[0.38,0.15,0.15,0.12,0.10]}
    heights=ht.get(n_rows,[0.38,0.15,0.15,0.12,0.10])[:n_rows]
    s=sum(heights); heights=[h/s for h in heights]

    show_vp = ("Volume Profile" in indicators)
    if show_vp and vols:
        # 2 colonne: 84% candlestick | 16% Volume Profile
        _specs = [[{"secondary_y":False},{"secondary_y":False}]]*n_rows
        fig=make_subplots(rows=n_rows,cols=2,shared_xaxes=False,
                          shared_yaxes=False,
                          row_heights=heights,vertical_spacing=0.025,
                          column_widths=[0.84,0.16],
                          specs=_specs,horizontal_spacing=0.004)
        _vp_col=2
    else:
        show_vp=False
        fig=make_subplots(rows=n_rows,cols=1,shared_xaxes=True,
                          row_heights=heights,vertical_spacing=0.025)
        _vp_col=None
    fig.add_trace(go.Candlestick(x=dates,open=opens,high=highs,low=lows,close=closes,
        increasing_line_color="#26a69a",increasing_fillcolor="rgba(38,166,154,0.85)",
        decreasing_line_color="#ef5350",decreasing_fillcolor="rgba(239,83,80,0.85)",
        name="Prezzo",showlegend=False),row=1,col=1)
    if bb_up and bb_dn:
        fig.add_trace(go.Scatter(x=dates+dates[::-1],y=bb_up+bb_dn[::-1],fill="toself",
            fillcolor="rgba(88,166,255,0.06)",line=dict(color="rgba(0,0,0,0)"),
            showlegend=False),row=1,col=1)
        for b,n in [(bb_up,"BB↑"),(bb_dn,"BB↓")]:
            fig.add_trace(go.Scatter(x=dates,y=b,
                line=dict(color="#58a6ff",width=1,dash="dot"),showlegend=False,name=n),row=1,col=1)
    if ema20: fig.add_trace(go.Scatter(x=dates,y=ema20,line=dict(color="#f59e0b",width=1.5),name="EMA20"),row=1,col=1)
    if ema50: fig.add_trace(go.Scatter(x=dates,y=ema50,line=dict(color="#a78bfa",width=1.5),name="EMA50"),row=1,col=1)
    # EMA200 gialla — già letta nell'header da chart_data
    if ema200:
        fig.add_trace(go.Scatter(x=dates,y=ema200,
            line=dict(color="#ffffff",width=2.0,dash="dot"),name="EMA200"),row=1,col=1)

    if show_sma:
        sma9=_sma(closes,9); sma21=_sma(closes,21)
        fig.add_trace(go.Scatter(x=dates,y=sma9,line=dict(color="#c084fc",width=1.5,dash="dash"),name="SMA9"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=sma21,line=dict(color="#fb923c",width=1.5,dash="dash"),name="SMA21"),row=1,col=1)
        for i in range(1,len(closes)):
            if any(v is None for v in [sma9[i],sma21[i],sma9[i-1],sma21[i-1]]): continue
            if sma9[i-1]<=sma21[i-1] and sma9[i]>sma21[i]:
                fig.add_annotation(x=dates[i],y=lows[i]*0.995,text="▲ ENTRY",
                    font=dict(color="#00ff88",size=10),showarrow=True,
                    arrowhead=2,arrowcolor="#00ff88",ay=30,ax=0,row=1,col=1)
            elif sma9[i-1]>=sma21[i-1] and sma9[i]<sma21[i]:
                fig.add_annotation(x=dates[i],y=highs[i]*1.005,text="▼ EXIT",
                    font=dict(color="#ef4444",size=10),showarrow=True,
                    arrowhead=2,arrowcolor="#ef4444",ay=-30,ax=0,row=1,col=1)

    if show_sar:
        sv,sd=_parabolic_sar(highs,lows)
        fig.add_trace(go.Scatter(x=dates,y=[sv[i] if sd[i]==1 else None for i in range(len(sv))],
            mode="markers",marker=dict(color="#00ff88",size=4),name="SAR ↑"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=[sv[i] if sd[i]==-1 else None for i in range(len(sv))],
            mode="markers",marker=dict(color="#ef4444",size=4),name="SAR ↓"),row=1,col=1)

    if show_sma and row_rsi:
        rv=_rsi_calc(closes)
        fig.add_hrect(y0=70,y1=100,fillcolor="rgba(239,68,68,0.08)",line_width=0,row=row_rsi,col=1)
        fig.add_hrect(y0=0,y1=30,fillcolor="rgba(0,255,136,0.08)",line_width=0,row=row_rsi,col=1)
        fig.add_trace(go.Scatter(x=dates,y=rv,line=dict(color="#60a5fa",width=1.5),name="RSI"),row=row_rsi,col=1)
        for lvl,col in [(70,"#ef4444"),(50,"#6b7280"),(30,"#00ff88")]:
            fig.add_hline(y=lvl,line=dict(color=col,width=1,dash="dot"),row=row_rsi,col=1)
        fig.update_yaxes(title_text="RSI",range=[0,100],tickfont=dict(size=9),row=row_rsi,col=1)

    if show_macd and row_macd:
        ml,ms,mh=_macd_calc(closes)
        fig.add_trace(go.Bar(x=dates,y=mh,
            marker_color=["rgba(0,255,136,0.7)" if v>=0 else "rgba(239,68,68,0.7)" for v in mh],
            name="MACD Hist",showlegend=False),row=row_macd,col=1)
        fig.add_trace(go.Scatter(x=dates,y=ml,line=dict(color="#60a5fa",width=1.5),name="MACD"),row=row_macd,col=1)
        fig.add_trace(go.Scatter(x=dates,y=ms,line=dict(color="#f97316",width=1.5),name="Signal"),row=row_macd,col=1)
        fig.add_hline(y=0,line=dict(color="#6b7280",width=1,dash="dot"),row=row_macd,col=1)
        fig.update_yaxes(title_text="MACD",tickfont=dict(size=9),row=row_macd,col=1)

    # ── Alligator (Jaw/Teeth/Lips) + Vortex (+VI/-VI) ─────────────────────
    if show_alligator and row_vortex:
        # Alligator: Jaw=SMA13, Teeth=SMA8, Lips=SMA5 (Williams)
        _jaw   = _sma(closes, 13)
        _teeth = _sma(closes, 8)
        _lips  = _sma(closes, 5)
        fig.add_trace(go.Scatter(x=dates,y=_jaw,
            line=dict(color="#3b82f6",width=1.5),name="Jaw(13)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=_teeth,
            line=dict(color="#ef4444",width=1.5),name="Teeth(8)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=_lips,
            line=dict(color="#22c55e",width=1.5),name="Lips(5)"),row=1,col=1)
        # Vortex Indicator (+VI/-VI) su pannello separato
        import numpy as _np2
        def _vortex(highs_l, lows_l, closes_l, period=14):
            n = len(highs_l)
            if n < period+1: return [None]*n, [None]*n
            h=_np2.array(highs_l,dtype=float); l=_np2.array(lows_l,dtype=float)
            c=_np2.array(closes_l,dtype=float)
            tr  = _np2.maximum(h[1:]-l[1:], _np2.maximum(_np2.abs(h[1:]-c[:-1]),_np2.abs(l[1:]-c[:-1])))
            vm_pos = _np2.abs(h[1:]-l[:-1])
            vm_neg = _np2.abs(l[1:]-h[:-1])
            vi_pos=[None]*period; vi_neg=[None]*period
            for i in range(period, n):
                s=i-period
                vi_pos.append(vm_pos[s:i].sum()/tr[s:i].sum() if tr[s:i].sum()>0 else 1.0)
                vi_neg.append(vm_neg[s:i].sum()/tr[s:i].sum() if tr[s:i].sum()>0 else 1.0)
            return vi_pos, vi_neg
        _vp, _vn = _vortex(highs, lows, closes)
        fig.add_trace(go.Scatter(x=dates,y=_vp,
            line=dict(color="#3b82f6",width=1.5),name="+VI"),row=row_vortex,col=1)
        fig.add_trace(go.Scatter(x=dates,y=_vn,
            line=dict(color="#ef4444",width=1.5),name="-VI"),row=row_vortex,col=1)
        fig.add_hline(y=1.0,line=dict(color="#6b7280",width=1,dash="dot"),row=row_vortex,col=1)
        fig.update_yaxes(title_text="Vortex",tickfont=dict(size=8),row=row_vortex,col=1)

    if vols:
        fig.add_trace(go.Bar(x=dates,y=vols,
            marker_color=["rgba(38,166,154,0.55)" if c>=o else "rgba(239,83,80,0.55)" for c,o in zip(closes,opens)],
            name="Volume",showlegend=False),row=row_vol,col=1)
        fig.update_yaxes(title_text="Vol",tickfont=dict(size=8),row=row_vol,col=1)

    # ── Volume Profile ──────────────────────────────────────────────────
    if show_vp and _vp_col:
        _vp_c,_vp_v,_poc,_vah,_val=_calc_volume_profile(highs,lows,closes,vols)
        if _vp_c:
            _mx=max(_vp_v) if _vp_v else 1
            _norm=[x/_mx for x in _vp_v]
            # Colori: dentro VA=blu TV, POC=oro, fuori=grigio
            _binw=(_vp_c[1]-_vp_c[0]) if len(_vp_c)>1 else 0
            _colors=[]
            for _i,_p in enumerate(_vp_c):
                if _poc and _binw and abs(_p-_poc)<_binw:
                    _colors.append("rgba(255,215,0,0.92)")    # POC oro
                elif _val and _vah and _val<=_p<=_vah:
                    _colors.append("rgba(41,98,255,0.70)")    # VA blu TV
                else:
                    _colors.append("rgba(120,123,134,0.42)")  # fuori VA grigio
            fig.add_trace(go.Bar(
                x=_norm, y=_vp_c, orientation="h",
                marker=dict(color=_colors,line=dict(width=0)),
                name="Vol Profile", showlegend=False,
                hovertemplate="P: %{y:.2f}<br>Vol: %{customdata:,.0f}<extra>VP</extra>",
                customdata=_vp_v,
            ),row=1,col=_vp_col)
            # Linee POC/VAH/VAL su asse Y condiviso con il prezzo
            if _poc:
                fig.add_hline(y=_poc,line=dict(color="#ffd700",width=1.5,dash="dot"),
                    annotation_text=" POC",annotation_font_color="#ffd700",
                    annotation_font_size=9,row=1,col=_vp_col)
            if _vah:
                fig.add_hline(y=_vah,line=dict(color="#2962ff",width=1,dash="dash"),
                    annotation_text=" VAH",annotation_font_color="#2962ff",
                    annotation_font_size=8,row=1,col=_vp_col)
            if _val:
                fig.add_hline(y=_val,line=dict(color="#2962ff",width=1,dash="dash"),
                    annotation_text=" VAL",annotation_font_color="#2962ff",
                    annotation_font_size=8,row=1,col=_vp_col)
            # Nascondi assi VP
            fig.update_xaxes(showticklabels=False,showgrid=False,zeroline=False,
                             col=_vp_col)
            for _rv in range(1,n_rows+1):
                fig.update_yaxes(showticklabels=False,showgrid=False,
                                 col=_vp_col,row=_rv)

    tkr=row.get("Ticker",""); sq="  🔥" if row.get("Squeeze") else ""
    fig.update_layout(**PLOTLY_DARK,
        title=dict(text=f"<b>{tkr}</b> — {row.get('Nome','')}  |  {row.get('Prezzo','')}  |  RSI {row.get('RSI','')}{sq}",
            font=dict(color="#50c4e0",size=13),x=0.01,xanchor="left"),
        height=160+180*n_rows,xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.01,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
        margin=dict(l=0,r=0,t=55,b=0),hovermode="x unified")
    for r in range(1,n_rows+1):
        fig.update_xaxes(gridcolor="#2a2e39",gridwidth=1,showline=True,linecolor="#363a45",row=r,col=1)
        fig.update_yaxes(gridcolor="#2a2e39",gridwidth=1,showline=True,linecolor="#363a45",row=r,col=1)
    return fig

def build_radar(row: pd.Series) -> go.Figure:
    qc=row.get("_quality_components")
    if not qc or not isinstance(qc,dict): return None
    keys=list(qc.keys()); vals=list(qc.values())
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=keys+[keys[0]],fill="toself",
        fillcolor="rgba(0,255,136,0.15)",line=dict(color="#00ff88",width=2)))
    fig.update_layout(**PLOTLY_DARK,
        polar=dict(bgcolor="#0d1117",
            radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=9,color="#6b7280"),
                gridcolor="#1f2937",linecolor="#1f2937"),
            angularaxis=dict(tickfont=dict(size=11,color="#c9d1d9"),
                gridcolor="#1f2937",linecolor="#1f2937")),
        title=dict(text=f"<b>{row.get('Ticker','')}</b>  Q: <b>{row.get('Quality_Score',0)}/12</b>",
            font=dict(color="#58a6ff",size=13)),
        height=340,margin=dict(l=40,r=40,t=55,b=20),showlegend=False)
    return fig

def show_charts(row_full: pd.Series, key_suffix: str=""):
    tkr=row_full.get("Ticker","")
    st.markdown("---")
    ind_opts=["SMA 9 & 21 + RSI","MACD","Parabolic SAR","Alligator + Vortex","Volume Profile"]
    c1,c2=st.columns([4,1])
    with c1:
        indicators=st.multiselect("🔧 Indicatori",options=ind_opts,
            default=st.session_state.get("active_indicators",ind_opts),
            key=f"ind_{tkr}_{key_suffix}")
        st.session_state["active_indicators"]=indicators
    with c2:
        st.write("")
        if st.button("🔄 Aggiorna",key=f"ref_{tkr}_{key_suffix}"): st.rerun()
    fig=build_full_chart(row_full,indicators)
    if fig: st.plotly_chart(fig,use_container_width=True,key=f"full_{tkr}_{key_suffix}")
    else:   st.info("Dati grafici non disponibili. Riesegui lo scanner.")
    fig_r=build_radar(row_full)
    if fig_r:
        _,c2,_=st.columns([1,1,1])
        with c2: st.plotly_chart(fig_r,use_container_width=True,key=f"radar_{tkr}_{key_suffix}")
    # ── Grafico Analitico Avanzato ──────────────────────────────────────
    try:
        from analysis_chart import render_analysis_chart as _adv_chart
        with st.expander(f"📐 Analisi Avanzata — Ichimoku · S/R · Trend · Squeeze  [{tkr}]",
                         expanded=False):
            _adv_chart(row_full, key_suffix=key_suffix)
    except ImportError:
        pass  # analysis_chart.py non presente

# =========================================================================
# JS RENDERERS
# =========================================================================
name_dblclick_renderer=JsCode("""class N{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value||'';const t=p.data.Ticker||p.data.ticker;if(!t)return;
this.eGui.style.cursor='pointer';this.eGui.title='Doppio click → TradingView';
this.eGui.ondblclick=()=>window.open("https://it.tradingview.com/chart/?symbol="+String(t).split(".")[0],"_blank");}
getGui(){return this.eGui;}}""")

rsi_renderer=JsCode("""class R{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'-':v.toFixed(1);
this.eGui.style.fontWeight='bold';this.eGui.style.fontFamily='Courier New';
if(v<30)this.eGui.style.color='#60a5fa';
else if(v<40)this.eGui.style.color='#93c5fd';
else if(v<=65)this.eGui.style.color='#00ff88';
else if(v<=70)this.eGui.style.color='#f59e0b';
else this.eGui.style.color='#ef4444';}getGui(){return this.eGui;}}""")

# Renderer stringa già formattata (MarketCap_fmt = "1.2B", "—", etc.)
mcap_str_renderer=JsCode("""class MS{init(p){this.eGui=document.createElement('span');
const s=String(p.value||'\u2014').trim();
let color='#6b7280';
if(s.endsWith('T'))color='#00ff88';
else if(s.endsWith('B'))color='#58a6ff';
else if(s.endsWith('M'))color='#f59e0b';
this.eGui.innerText=s;this.eGui.style.color=color;
this.eGui.style.fontFamily='Courier New';this.eGui.style.fontWeight='bold';}
getGui(){return this.eGui;}refresh(){return false;}}""")

vol_ratio_renderer=JsCode("""class V{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'-':v.toFixed(2)+'x';
this.eGui.style.fontFamily='Courier New';this.eGui.style.fontWeight='bold';
if(v<1)this.eGui.style.color='#6b7280';
else if(v<2)this.eGui.style.color='#00ff88';
else if(v<3)this.eGui.style.color='#f59e0b';
else{this.eGui.style.color='#ef4444';this.eGui.style.textShadow='0 0 6px #ef4444';}
}getGui(){return this.eGui;}}""")

# Renderer per volumi abbreviati (es. 1.2M, 45.6K, 2.3B)
vol_abbrev_renderer=JsCode("""class VA{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
let txt='-';
if(!isNaN(v)){
  if(v>=1e9)txt=(v/1e9).toFixed(1)+'B';
  else if(v>=1e6)txt=(v/1e6).toFixed(1)+'M';
  else if(v>=1e3)txt=(v/1e3).toFixed(0)+'K';
  else txt=v.toFixed(0);
}
this.eGui.innerText=txt;
this.eGui.style.fontFamily='Courier New';this.eGui.style.color='#c9d1d9';
}getGui(){return this.eGui;}}""")

# Renderer MarketCap abbreviato
mcap_renderer=JsCode("""class MC{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
let txt='—';let color='#6b7280';
if(!isNaN(v) && v>1000000){
  if(v>=1e12){txt=(v/1e12).toFixed(2)+'T';color='#00ff88';}
  else if(v>=1e9){txt=(v/1e9).toFixed(1)+'B';color='#58a6ff';}
  else if(v>=1e6){txt=(v/1e6).toFixed(0)+'M';color='#f59e0b';}
  else{txt=(v/1e3).toFixed(0)+'K';color='#6b7280';}
}
this.eGui.innerText=txt;
this.eGui.style.fontFamily='Courier New';this.eGui.style.color=color;this.eGui.style.fontWeight='bold';
}getGui(){return this.eGui;}}""")

quality_renderer=JsCode("""class Q{init(p){this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:6px';
const v=parseInt(p.value||0);const pct=Math.round((v/12)*100);
const c=v>=9?'#00ff88':v>=6?'#f59e0b':'#6b7280';
this.eGui.innerHTML=`<span style="font-family:Courier New;font-weight:bold;color:${c};min-width:20px">${v}</span>
<div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
<div style="width:${pct}%;background:${c};height:6px;border-radius:3px"></div></div>`;}
getGui(){return this.eGui;}}""")

ser_score_renderer=JsCode("""class S{init(p){this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:6px';
const v=parseInt(p.value||0);const pct=Math.round((v/6)*100);
const c=v>=6?'#00ff88':v>=4?'#f59e0b':'#ef4444';
this.eGui.innerHTML=`<span style="font-family:Courier New;font-weight:bold;color:${c};min-width:20px">${v}/6</span>
<div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
<div style="width:${pct}%;background:${c};height:6px;border-radius:3px"></div></div>`;}
getGui(){return this.eGui;}}""")

fv_score_renderer=JsCode("""class F{init(p){this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:6px';
const v=parseInt(p.value||0);const pct=Math.round((v/8)*100);
const c=v>=7?'#00ff88':v>=5?'#f59e0b':'#6b7280';
this.eGui.innerHTML=`<span style="font-family:Courier New;font-weight:bold;color:${c};min-width:20px">${v}/8</span>
<div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
<div style="width:${pct}%;background:${c};height:6px;border-radius:3px"></div></div>`;}
getGui(){return this.eGui;}}""")

bool_renderer=JsCode("""class B{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v===true||v==='True'||v==='true'||v===1){this.eGui.innerText='✅';this.eGui.style.color='#00ff88';}
else if(v===false||v==='False'||v==='false'||v===0){this.eGui.innerText='❌';this.eGui.style.color='#ef4444';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

squeeze_renderer=JsCode("""class Sq{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v===true||v==='True'||v==='true'){this.eGui.innerText='🔥 SQ';this.eGui.style.color='#f97316';this.eGui.style.fontWeight='bold';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

weekly_renderer=JsCode("""class W{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v===true||v==='True'||v==='true'){this.eGui.innerText='📈 W+';this.eGui.style.color='#00ff88';}
else if(v===false||v==='False'||v==='false'){this.eGui.innerText='📉 W—';this.eGui.style.color='#ef4444';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

rsi_div_renderer=JsCode("""class RD{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v==='BEARISH'){this.eGui.innerText='⚠️ BEAR';this.eGui.style.color='#ef4444';}
else if(v==='BULLISH'){this.eGui.innerText='✅ BULL';this.eGui.style.color='#00ff88';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

price_renderer=JsCode("""class P{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value??'-';this.eGui.style.fontFamily='Courier New';
this.eGui.style.color='#e2e8f0';this.eGui.style.fontWeight='bold';}
getGui(){return this.eGui;}}""")

trend_renderer=JsCode("""class T{init(p){this.eGui=document.createElement('span');
const v=(p.value||'').toUpperCase();
const map={LONG:{c:'#00ff88',e:'🟢 LONG'},SHORT:{c:'#ef4444',e:'🔴 SHORT'},WATCH:{c:'#f59e0b',e:'👁 WATCH'}};
const m=map[v]||{c:'#6b7280',e:v||'—'};
this.eGui.innerText=m.e;this.eGui.style.color=m.c;this.eGui.style.fontWeight='bold';}
getGui(){return this.eGui;}}""")

pct_renderer=JsCode("""class Pct{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
if(isNaN(v)){this.eGui.innerText='—';this.eGui.style.color='#6b7280';}
else{this.eGui.innerText=(v*100).toFixed(1)+'%';
this.eGui.style.color=v>0?'#00ff88':v<0?'#ef4444':'#6b7280';
this.eGui.style.fontWeight='bold';this.eGui.style.fontFamily='Courier New';}
}getGui(){return this.eGui;}}""")

# =========================================================================
# EXPORT
# =========================================================================
def to_excel_bytes(d):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="xlsxwriter") as w:
        for nm,df in d.items():
            if isinstance(df,pd.DataFrame) and not df.empty:
                df.to_excel(w,sheet_name=nm[:31],index=False)
    return buf.getvalue()

def make_tv_csv(df,tab):
    t=df[["Ticker"]].copy(); t.insert(0,"Tab",tab)
    return t.to_csv(index=False).encode()

def csv_btn(df,fname,key):
    st.download_button("📥 CSV",df.to_csv(index=False).encode(),fname,"text/csv",key=key)

# =========================================================================
# PRESETS
# =========================================================================
PRESETS={
    "⚡ Aggressivo":   dict(eh=0.01,prmin=45,prmax=65,rpoc=0.01,vol_ratio_hot=1.2,top=20,min_early_score=2.0,min_quality=3,min_pro_score=2.0),
    "⚖️ Bilanciato":   dict(eh=0.02,prmin=40,prmax=70,rpoc=0.02,vol_ratio_hot=1.5,top=15,min_early_score=4.0,min_quality=5,min_pro_score=4.0),
    "🛡️ Conservativo": dict(eh=0.04,prmin=35,prmax=75,rpoc=0.04,vol_ratio_hot=2.0,top=10,min_early_score=6.0,min_quality=7,min_pro_score=6.0),
    "🔓 Nessun Filtro":dict(eh=0.05,prmin=10,prmax=90,rpoc=0.05,vol_ratio_hot=0.3,top=100,min_early_score=0.0,min_quality=0,min_pro_score=0.0),
}

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(page_title="Trading Scanner PRO 31.1",layout="wide",page_icon="🧠")
st.markdown(DARK_CSS,unsafe_allow_html=True)
st.markdown("# 🧠 Trading Scanner PRO 31.1")
st.markdown('<div class="section-pill">CACHE · BACKTEST · FINVIZ · MULTI-WATCHLIST · BLUE CHIP DIP · v31.1</div>',unsafe_allow_html=True)
init_db()

# ── GitHub pull al boot (ripristina watchlist dopo ogni deploy) ─────────────
if _GH_SYNC and not st.session_state.get("_gh_pulled"):
    with st.spinner("☁️ Ripristino watchlist da GitHub..."):
        _ok, _n, _gh_src = _gh_pull(DB_PATH)
    st.session_state["_gh_pulled"] = True
    if _ok and _n > 0:
        st.toast(f"☁️ Watchlist ripristinata: {_n} ticker", icon="✅")
    elif not _ok and _gh_src == "github_error":
        st.toast("⚠️ GitHub sync: errore connessione — uso dati locali", icon="⚠️")

# =========================================================================
# SESSION STATE
# =========================================================================
defaults=dict(
    mSP500=True,mNasdaq=True,mFTSE=True,mEurostoxx=False,
    mDow=False,mRussell=False,mStoxxEmerging=False,mUSSmallCap=False,
    eh=0.02,prmin=40,prmax=70,rpoc=0.02,vol_ratio_hot=1.5,top=15,
    min_early_score=2.0,min_quality=3,min_pro_score=2.0,
    current_list_name="DEFAULT",last_active_tab="EARLY",
    active_indicators=["SMA 9 & 21 + RSI","MACD","Parabolic SAR","Alligator + Vortex"],
    wl_view_mode="cards",
)
for k,v in defaults.items():
    st.session_state.setdefault(k,v)

# =========================================================================
# KPI BAR
# =========================================================================
def render_kpi_bar(df_ep,df_rea):
    hist=load_scan_history(2); p_e=p_p=p_h=p_c=0
    if len(hist)>=2:
        pr=hist.iloc[1];p_e=int(pr.get("n_early",0));p_p=int(pr.get("n_pro",0))
        p_h=int(pr.get("n_rea",0));p_c=int(pr.get("n_confluence",0))
    n_e=int((df_ep.get("Stato_Early",pd.Series())=="EARLY").sum()) if not df_ep.empty else 0
    n_p=int((df_ep.get("Stato_Pro",pd.Series())=="PRO").sum()) if not df_ep.empty else 0
    n_h=len(df_rea) if not df_rea.empty else 0
    n_c=0
    if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
        n_c=int(((df_ep["Stato_Early"]=="EARLY")&(df_ep["Stato_Pro"]=="PRO")).sum())
    k1,k2,k3,k4=st.columns(4)
    k1.metric("📡 EARLY",n_e,delta=n_e-p_e if p_e else None)
    k2.metric("💪 PRO",n_p,delta=n_p-p_p if p_p else None)
    k3.metric("🔥 REA-HOT",n_h,delta=n_h-p_h if p_h else None)
    k4.metric("⭐ CONFLUENCE",n_c,delta=n_c-p_c if p_c else None)

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.title("⚙️ Configurazione")

with st.sidebar.expander("🎯 Preset Rapidi",expanded=False):
    for pname,pvals in PRESETS.items():
        if st.button(pname,use_container_width=True,key=f"preset_{pname}"):
            for k,v in pvals.items(): st.session_state[k]=v
            st.rerun()

with st.sidebar.expander("🌍 Mercati",expanded=True):
    msp500   =st.checkbox("S&P 500",         st.session_state.mSP500)
    mnasdaq  =st.checkbox("Nasdaq 100",       st.session_state.mNasdaq)
    mftse    =st.checkbox("FTSE MIB",         st.session_state.mFTSE)
    meuro    =st.checkbox("Eurostoxx 600",    st.session_state.mEurostoxx)
    mdow     =st.checkbox("Dow Jones",        st.session_state.mDow)
    mrussell =st.checkbox("Russell 2000",     st.session_state.mRussell)
    mstoxxem =st.checkbox("Stoxx Emerging 50",st.session_state.mStoxxEmerging)
    mussmall =st.checkbox("US Small Cap 2000",st.session_state.mUSSmallCap)

sel=[mkt for flag,mkt in [
    (msp500,"SP500"),(mnasdaq,"Nasdaq"),(mftse,"FTSE"),(meuro,"Eurostoxx"),
    (mdow,"Dow"),(mrussell,"Russell"),(mstoxxem,"StoxxEmerging"),(mussmall,"USSmallCap"),
] if flag]
(st.session_state.mSP500,st.session_state.mNasdaq,st.session_state.mFTSE,
 st.session_state.mEurostoxx,st.session_state.mDow,st.session_state.mRussell,
 st.session_state.mStoxxEmerging,st.session_state.mUSSmallCap)=(
    msp500,mnasdaq,mftse,meuro,mdow,mrussell,mstoxxem,mussmall)

with st.sidebar.expander("🎛️ Parametri Scanner",expanded=False):
    eh           =st.slider("EARLY EMA20 %",0.0,10.0,float(st.session_state.eh*100),0.5)/100
    prmin        =st.slider("PRO RSI min",0,100,int(st.session_state.prmin),5)
    prmax        =st.slider("PRO RSI max",0,100,int(st.session_state.prmax),5)
    rpoc         =st.slider("REA POC %",0.0,10.0,float(st.session_state.rpoc*100),0.5)/100
    vol_ratio_hot=st.number_input("VolRatio HOT",0.0,10.0,float(st.session_state.vol_ratio_hot),0.1)
    top          =st.number_input("TOP N",5,200,int(st.session_state.top),5)
(st.session_state.eh,st.session_state.prmin,st.session_state.prmax,
 st.session_state.rpoc,st.session_state.vol_ratio_hot,st.session_state.top)=(
    eh,prmin,prmax,rpoc,vol_ratio_hot,top)

with st.sidebar.expander("🔬 Soglie Filtri (live)",expanded=True):
    st.caption("⬇️ Abbassa per vedere più segnali  |  0 = nessun filtro")
    min_early_score=st.slider("Early Score ≥",0.0,10.0,float(st.session_state.min_early_score),0.5)
    min_quality    =st.slider("Quality ≥",0,12,int(st.session_state.min_quality),1)
    min_pro_score  =st.slider("Pro Score ≥",0.0,10.0,float(st.session_state.min_pro_score),0.5)
    st.session_state.min_early_score=min_early_score
    st.session_state.min_quality    =min_quality
    st.session_state.min_pro_score  =min_pro_score

with st.sidebar.expander("📊 Indicatori Grafici",expanded=False):
    ind_opts_all=["SMA 9 & 21 + RSI","MACD","Parabolic SAR","Alligator + Vortex"]
    ai=st.multiselect("Attivi",options=ind_opts_all,
        default=[x for x in st.session_state.active_indicators if x in ind_opts_all],
        key="global_indicators")
    st.session_state.active_indicators=ai

st.sidebar.divider()
st.sidebar.subheader("📋 Watchlist")

df_wl_all=load_watchlist()
list_options=sorted(df_wl_all["list_name"].unique().tolist()) if not df_wl_all.empty else []
if "DEFAULT" not in list_options: list_options.append("DEFAULT")
list_options=sorted(list_options)

active_list=st.sidebar.selectbox("Lista Attiva",list_options,
    index=list_options.index(st.session_state.current_list_name)
    if st.session_state.current_list_name in list_options else 0,
    key="active_list")
st.session_state.current_list_name=active_list

# ── Crea nuova lista ─────────────────────────────────────────────────────
with st.sidebar.expander("➕ Nuova Lista",expanded=False):
    new_list_name=st.text_input("Nome lista",key="new_list_input",placeholder="es. Watchlist Tech")
    if st.button("✅ Crea e Attiva",key="create_list_btn",use_container_width=True):
        if new_list_name.strip():
            nm=new_list_name.strip()
            # Crea la lista inserendo un placeholder temporaneo e cancellandolo subito
            # (la lista esiste nel DB solo se ha almeno un record)
            # → salviamo il nome in session_state e sarà visibile quando si aggiunge un ticker
            st.session_state.current_list_name=nm
            st.session_state["pending_new_list"]=nm
            st.sidebar.success(f"Lista '{nm}' creata. Aggiungici ticker dallo scanner.")
            st.rerun()
        else:
            st.sidebar.warning("Inserisci un nome.")

if st.sidebar.button("⚠️ Reset Watchlist DB",key="rst_wl"):
    reset_watchlist_db(); st.rerun()

st.sidebar.divider()
st.sidebar.subheader("⚡ Scanner v29")
with st.sidebar.expander("🔧 Opzioni avanzate",expanded=False):
    use_cache  = st.checkbox("⚡ Cache SQLite (più veloce)",True,key="use_cache",
                              help="Riusa dati yfinance già scaricati oggi (TTL 4h). "
                                   "Secondo scanner della giornata → ~30 sec totali.")
    use_finviz = st.checkbox("📊 Finviz scraping (EPS reali)",False,key="use_finviz",
                              help="Scarica EPS growth, short float, PEG da Finviz. "
                                   "Più lento (+20-40% tempo). Richiede finvizfinance installato.")
    n_workers  = st.slider("🔄 Worker paralleli",2,16,8,2,key="n_workers",
                            help="Thread simultanei. 8 = ottimale. Aumenta con cautela "
                                 "(troppi → rate limit yfinance).")
    if st.button("🗑️ Svuota cache",key="clear_cache_btn",use_container_width=True):
        try:
            cache_clear()
            st.success("✅ Cache svuotata.")
        except Exception as e:
            st.error(f"Errore: {e}")
    if st.button("📊 Info cache",key="cache_info_btn",use_container_width=True):
        try:
            cs = cache_stats()
            st.info(f"🟢 {cs['fresh']} fresche  ⏰ {cs['stale']} scadute  💾 {cs['size_mb']} MB")
        except Exception as e:
            st.info("Cache non disponibile.")

# Scan stats ultima scansione
if "scan_stats" in st.session_state:
    ss = st.session_state.scan_stats
    st.sidebar.caption(
        f"⏱️ Ultima: **{ss['elapsed_s']}s**  "
        f"⚡ {ss['cache_hits']} cache  "
        f"☁️ {ss['downloaded']} scaricati"
    )

st.sidebar.divider()
if st.sidebar.button("🗑️ Reset Storico",key="reset_hist_sidebar"):
    try:
        conn=sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM scan_history");conn.commit();conn.close()
        st.sidebar.success("Storico cancellato.");st.rerun()
    except Exception as e: st.sidebar.error(f"Errore: {e}")

only_watchlist=st.sidebar.checkbox("Solo Watchlist",False)

st.sidebar.divider()
st.sidebar.markdown("**🔧 Layout Griglie**")
st.sidebar.caption("Le larghezze/ordinamenti colonne vengono salvati nel browser (localStorage).")
if st.sidebar.button("↺ Reset layout griglie",key="reset_grid_layout",use_container_width=True):
    # Inietta JS per cancellare tutte le chiavi grid_state_* dal localStorage
    st.markdown("""<script>
(function(){
  Object.keys(localStorage).filter(k=>k.startsWith('grid_state_')).forEach(k=>localStorage.removeItem(k));
  console.log('Grid states cleared');
})();
</script>""",unsafe_allow_html=True)
    st.sidebar.success("Layout resettato — ricarica la pagina.")

# =========================================================================
# SCANNER
# =========================================================================
if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 31.1",type="primary",use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            tot        = len(universe)
            use_cache  = st.session_state.get("use_cache", True)
            use_finviz = st.session_state.get("use_finviz", False)
            n_wk       = st.session_state.get("n_workers", 8)

            # ── Test connessione Yahoo Finance ────────────────────────────
            import requests as _req
            _conn_ok  = False
            _test_tkr = next((t for t in universe if len(t) <= 5), universe[0])
            _conn_box = st.empty()
            try:
                _s = _req.Session()
                _s.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Referer": "https://finance.yahoo.com/"
                })
                _r = _s.get(
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{_test_tkr}",
                    params={"interval": "1d", "range": "5d"}, timeout=20
                )
                if _r.status_code == 200:
                    _res = _r.json().get("chart", {}).get("result", [])
                    if _res and _res[0].get("timestamp"):
                        _conn_box.success(f"✅ Connessione Yahoo OK — ticker test: `{_test_tkr}`")
                        _conn_ok = True
                    else:
                        _conn_box.error(f"❌ Yahoo Finance risposta vuota per `{_test_tkr}`")
                else:
                    _conn_box.error(f"❌ Yahoo Finance HTTP {_r.status_code}")
            except Exception as _ce:
                _conn_box.error(f"❌ Connessione fallita: {_ce}")

            if not _conn_ok:
                st.warning("⚠️ Test connessione fallito. Lo scanner proverà comunque — "
                           "potrebbe restituire 0 risultati se Yahoo Finance non è raggiungibile.")

            # ── Barra progressiva SEQUENZIALE (aggiornamento in tempo reale) ──
            st.markdown(f"### 🔍 Scansione: **{tot}** ticker")
            pb     = st.progress(0.0)
            status = st.empty()
            errors_box = st.empty()
            found_box  = st.empty()

            rep_live  = [0]   # contatore segnali trovati in tempo reale
            rea_live  = [0]

            def _progress(done, total, tkr):
                pct = done / total
                pb.progress(pct)
                n_ep  = rep_live[0]
                n_rea = rea_live[0]
                status.info(
                    f"🔍 **{done} / {total}** "
                    f"({pct*100:.0f}%) — `{tkr}`  "
                    f"| 📡 EARLY/PRO: **{n_ep}** | 🔥 HOT: **{n_rea}**"
                )

            # Patch scan_universe per aggiornare contatori live
            import utils.scanner as _sc_mod
            _orig_scan = _sc_mod.scan_ticker
            def _patched_scan(tkr, *a, **k):
                ep, rea = _orig_scan(tkr, *a, **k)
                if ep:  rep_live[0] += 1
                if rea: rea_live[0] += 1
                return ep, rea
            _sc_mod.scan_ticker = _patched_scan

            try:
                df_ep_new, df_rea_new, scan_stats = scan_universe(
                    universe, eh, prmin, prmax, rpoc, vol_ratio_hot,
                    cache_enabled=use_cache, finviz_enabled=use_finviz,
                    n_workers=n_wk, progress_callback=_progress
                )
            finally:
                _sc_mod.scan_ticker = _orig_scan  # ripristina

            # ── Normalizza colonne ────────────────────────────────────────
            df_ep_new  = _enrich_df(df_ep_new)
            df_rea_new = _enrich_df(df_rea_new)
            pb.progress(1.0)

            elapsed = scan_stats.get("elapsed_s", 0)
            n_err   = scan_stats.get("n_errors", 0)
            errs    = scan_stats.get("errors", [])

            status.success(
                f"✅ **{tot} ticker** in **{elapsed:.0f}s** — "
                f"📡 **{len(df_ep_new)}** segnali EP | "
                f"🔥 **{len(df_rea_new)}** HOT | "
                f"⚠️ {n_err} errori"
            )

            if n_err > 0:
                with st.expander(f"⚠️ {n_err} errori (espandi per dettagli)",
                                  expanded=(len(df_ep_new) == 0)):
                    for _e in errs[:20]:
                        st.code(_e)

            if df_ep_new.empty and df_rea_new.empty:
                st.error(
                    "🔴 **0 segnali trovati.** Cause possibili:\n"
                    "1. Yahoo Finance irraggiungibile (prova tra 5 min)\n"
                    "2. Parametri troppo restrittivi → usa Preset **'🔓 Nessun Filtro'**\n"
                    f"3. {n_err} ticker con errori (vedi sopra)"
                )

            st.session_state.df_ep     = df_ep_new
            st.session_state.df_rea    = df_rea_new
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
            st.session_state.scan_stats = scan_stats

            try:
                scan_id = save_scan_history(sel, df_ep_new, df_rea_new,
                                             elapsed_s=elapsed, cache_hits=0)
            except TypeError:
                scan_id = save_scan_history(sel, df_ep_new, df_rea_new)
            save_signals(scan_id, df_ep_new, df_rea_new, sel)

            n_h = len(df_rea_new)
            n_c = 0
            if not df_ep_new.empty and "Stato_Early" in df_ep_new.columns:
                n_c = int(((df_ep_new["Stato_Early"]=="EARLY")&
                            (df_ep_new["Stato_Pro"]=="PRO")).sum())
            if n_h >= 5: st.toast(f"🔥 {n_h} HOT!", icon="🔥")
            if n_c >= 3: st.toast(f"⭐ {n_c} CONFLUENCE!", icon="⭐")
            st.rerun()

# ── Auto-load: se session_state è vuoto (refresh/reboot), ricarica l'ultima
#    scansione salvata nel DB così i tab non sono mai completamente vuoti ─────
if "df_ep" not in st.session_state:
    try:
        _hist = load_scan_history(1)
        if not _hist.empty:
            _last_id = int(_hist.iloc[0]["id"])
            _df_ep_load, _df_rea_load = load_scan_snapshot(_last_id)
            if not _df_ep_load.empty or not _df_rea_load.empty:
                # Arricchisce con campi calcolati (Ser_OK, FV_OK, Stato_Pro>=6)
                _df_ep_load  = _enrich_df(_df_ep_load)
                _df_rea_load = _enrich_df(_df_rea_load)
                st.session_state.df_ep     = _df_ep_load
                st.session_state.df_rea    = _df_rea_load
                st.session_state.last_scan = str(_hist.iloc[0].get("scanned_at",""))[:16]
                st.session_state["_autoloaded"] = True
    except Exception:
        pass

df_ep =st.session_state.get("df_ep", pd.DataFrame())
df_rea=st.session_state.get("df_rea",pd.DataFrame())

if st.session_state.get("_autoloaded"):
    st.caption(f"📂 Dati dall'ultima scansione: {st.session_state.get('last_scan','')} _(ricaricati dal DB)_")
elif "last_scan" in st.session_state:
    st.caption(f"⏱️ Ultima scansione: {st.session_state.last_scan}")
render_kpi_bar(df_ep,df_rea)

# ── Pannello diagnostico (visibile solo se df non vuoto o si clicca) ─────────
with st.expander("🔎 Diagnostica dati scanner",expanded=False):
    c1,c2,c3=st.columns(3)
    c1.metric("Righe df_ep",  len(df_ep)  if not df_ep.empty  else 0)
    c2.metric("Righe df_rea", len(df_rea) if not df_rea.empty else 0)
    c3.metric("Autoloaded",   "Sì" if st.session_state.get("_autoloaded") else "No")
    if not df_ep.empty:
        _col_check = {
            "Stato_Early":  df_ep.get("Stato_Early","").eq("EARLY").sum() if "Stato_Early" in df_ep.columns else "colonna assente",
            "Stato_Pro":    df_ep.get("Stato_Pro","").eq("PRO").sum()     if "Stato_Pro"   in df_ep.columns else "colonna assente",
            "Ser_OK=True":  df_ep.get("Ser_OK","").isin([True,"True","true"]).sum() if "Ser_OK" in df_ep.columns else "colonna assente",
            "FV_OK=True":   df_ep.get("FV_OK","").isin([True,"True","true"]).sum()  if "FV_OK"  in df_ep.columns else "colonna assente",
            "Weekly_Bull":  df_ep.get("Weekly_Bull","").isin([True,"True","true",1]).sum() if "Weekly_Bull" in df_ep.columns else "colonna assente",
        }
        st.write("**Conteggi segnali:**", _col_check)
        st.write("**Colonne disponibili:**", list(df_ep.columns))

    else:
        st.write("df_ep è vuoto.")
        _hist_diag = load_scan_history(3)
        if not _hist_diag.empty:
            st.write("**Ultime scansioni nel DB:**")
            st.dataframe(_hist_diag[["id","scanned_at","n_early","n_pro","n_rea"]],
                         use_container_width=True)
        else:
            st.write("Nessuna scansione trovata nel DB.")

st.markdown("---")

# =========================================================================
# AGGRID BUILDER  — resize + sort + filter
# =========================================================================
def build_aggrid(df_disp, grid_key, height=480, editable_cols=None):
    gb=GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(sortable=True,resizable=True,filterable=True,
                                 editable=False,wrapText=False,suppressSizeToFit=False,
                                 minWidth=95)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple",use_checkbox=True)

    if editable_cols:
        for ec in editable_cols:
            if ec in df_disp.columns:
                gb.configure_column(ec,editable=True)

    col_w={"Ticker":100,"Nome":230,"Prezzo":95,"Prezzo_fmt":105,"MarketCap":130,"MarketCap_fmt":130,
           "Early_Score":105,"Pro_Score":95,"Quality_Score":145,"Ser_Score":100,"FV_Score":100,
           "RSI":80,"Vol_Ratio":100,"Squeeze":85,"RSI_Div":95,
           "Weekly_Bull":95,"Stato_Early":100,"Stato_Pro":95,
           "Vol_Today":110,"Vol_7d_Avg":110,"Avg_Vol_20":110,
           "trend":115,"note":230,"origine":105,"created_at":115,
           "EPS_NY_Gr":100,"EPS_5Y_Gr":100,"PE":80,"Fwd_PE":85,
           "Earnings_Soon":105,"Optionable":95,"OBV_Trend":95,
           "EMA20":95,"EMA50":95,"EMA200":100,"EMA200_fmt":105,"ATR":85,"Rel_Vol":90,
           "Dist_POC_%":105,"POC":95,"Currency":85}
    for c,w in col_w.items():
        if c in df_disp.columns: gb.configure_column(c,width=w)
    hide_cols=["id","_chart_data","_quality_components","_ser_criteri","_fv_criteri",
               "Ser_OK","FV_OK","ATR_Exp","Stato",
               "Prezzo","MarketCap","EMA200","Currency"]
    for c in hide_cols:
        if c in df_disp.columns: gb.configure_column(c,hide=True)

    rmap={"Nome":name_dblclick_renderer,"RSI":rsi_renderer,
          "Vol_Ratio":vol_ratio_renderer,"Quality_Score":quality_renderer,
          "Ser_Score":ser_score_renderer,"FV_Score":fv_score_renderer,
          "Squeeze":squeeze_renderer,"RSI_Div":rsi_div_renderer,
          "Weekly_Bull":weekly_renderer,"Prezzo_fmt":price_renderer,"Prezzo":price_renderer,
          "trend":trend_renderer,
          "Vol_Today":vol_abbrev_renderer,"Vol_7d_Avg":vol_abbrev_renderer,"Avg_Vol_20":vol_abbrev_renderer,
          "MarketCap":mcap_renderer,"MarketCap_fmt":mcap_str_renderer,
          "EMA200_fmt":price_renderer,
          "EPS_NY_Gr":pct_renderer,"EPS_5Y_Gr":pct_renderer,
          "ROE":pct_renderer,"Gross_Mgn":pct_renderer,"Op_Mgn":pct_renderer,
          "Earnings_Soon":bool_renderer,"Optionable":bool_renderer,
          "Ser_OK":bool_renderer,"FV_OK":bool_renderer,
          "Dist_POC_%":JsCode("""class DP{init(p){this.eGui=document.createElement('span');const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'\u2014':v.toFixed(2)+'%';this.eGui.style.fontFamily='Courier New';}getGui(){return this.eGui;}}""")}
    for c,r in rmap.items():
        if c in df_disp.columns: gb.configure_column(c,cellRenderer=r)

    if "Ticker" in df_disp.columns: gb.configure_column("Ticker",pinned="left")
    if "Nome"   in df_disp.columns: gb.configure_column("Nome",  pinned="left")

    go_opts=gb.build()
    sk = "grid_state_" + grid_key

    # Carica layout salvato nel DB (persiste tra riavvii)
    saved_layout = load_grid_layout(grid_key)
    if saved_layout:
        _sl = repr(saved_layout)
        go_opts["onFirstDataRendered"]=JsCode("""
function(p){
  try{
    var db=""" + _sl + """;
    if(db.colState) p.columnApi.applyColumnState({state:db.colState,applyOrder:true});
    if(db.sortState) p.api.setSortModel(db.sortState);
    sessionStorage.setItem('""" + sk + """',JSON.stringify(db));
  }catch(e){p.api.sizeColumnsToFit();}
}""")
    else:
        go_opts["onFirstDataRendered"]=JsCode("""
function(p){
  try{
    var saved=sessionStorage.getItem('""" + sk + """');
    if(saved){
      var st=JSON.parse(saved);
      if(st.colState) p.columnApi.applyColumnState({state:st.colState,applyOrder:true});
      if(st.sortState) p.api.setSortModel(st.sortState);
    } else { p.api.sizeColumnsToFit(); }
  }catch(e){p.api.sizeColumnsToFit();}
}""")

    go_opts["onColumnResized"]=JsCode("""
function(p){
  if(!p.finished)return;
  try{
    var cur=JSON.parse(sessionStorage.getItem('""" + sk + """')||'{}');
    cur.colState=p.columnApi.getColumnState();
    sessionStorage.setItem('""" + sk + """',JSON.stringify(cur));
  }catch(e){}
}""")
    go_opts["onSortChanged"]=JsCode("""
function(p){
  try{
    var cur=JSON.parse(sessionStorage.getItem('""" + sk + """')||'{}');
    cur.sortState=p.api.getSortModel();
    sessionStorage.setItem('""" + sk + """',JSON.stringify(cur));
  }catch(e){}
}""")
    go_opts["onColumnMoved"]=JsCode("""
function(p){
  try{
    var cur=JSON.parse(sessionStorage.getItem('""" + sk + """')||'{}');
    cur.colState=p.columnApi.getColumnState();
    sessionStorage.setItem('""" + sk + """',JSON.stringify(cur));
  }catch(e){}
}""")

    update=GridUpdateMode.VALUE_CHANGED if editable_cols else GridUpdateMode.SELECTION_CHANGED
    resp = AgGrid(df_disp,gridOptions=go_opts,height=height,
                  update_mode=update,
                  data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                  fit_columns_on_grid_load=False,theme="streamlit",
                  allow_unsafe_jscode=True,key=grid_key)

    # ── Pulsante salva/reset layout ──────────────────────────────
    _lc1,_lc2,_lc3=st.columns([1,1,8])
    with _lc1:
        if st.button("💾 Layout",key="save_lay_"+grid_key,
                     help="Salva larghezza e ordinamento colonne nel DB (persiste dopo riavvio)"):
            try:
                # Leggiamo il colState dal DB resp (quello visible da AgGrid)
                _cols_data = resp.get("column_state", None)
                if _cols_data:
                    save_grid_layout(grid_key, {"colState": _cols_data})
                    st.success("✅ Layout salvato nel DB!")
                else:
                    # Fallback: salva le larghezze da col_w come baseline
                    save_grid_layout(grid_key, {"colState": [], "note": "baseline"})
                    st.info("Layout baseline salvato. Ridimensiona poi salva di nuovo.")
            except Exception as _le:
                st.error(f"Errore: {_le}")
    with _lc2:
        if st.button("↩️ Reset",key="reset_lay_"+grid_key,
                     help="Ripristina le larghezze predefinite delle colonne"):
            try:
                save_grid_layout(grid_key, None)
                st.success("↩️ Layout resettato!")
                st.rerun()
            except Exception as _le:
                st.error(f"Errore reset: {_le}")
    return resp

# =========================================================================
# LEGENDE
# =========================================================================
# ═══════════════════════════════════════════════════════════════════
# CRISIS MONITOR — asset difensivi per guerra, inflazione, crisi
# ═══════════════════════════════════════════════════════════════════
CRISIS_ASSETS = {
    "🥇 Metalli Preziosi": {
        "desc": "Riserva di valore in ogni crisi. Oro e argento salgono in guerra, inflazione, panic sell.",
        "assets": [
            ("GLD",  "SPDR Gold ETF",          "ETF oro fisico — il più liquido"),
            ("IAU",  "iShares Gold Trust",      "ETF oro fisico — costi ridotti"),
            ("SLV",  "iShares Silver Trust",    "ETF argento fisico — più volatile dell'oro"),
            ("GDX",  "VanEck Gold Miners ETF",  "Minatori oro — leva sull'oro"),
            ("GDXJ", "VanEck Junior Gold Miners","Minatori junior — leva maggiore"),
            ("NEM",  "Newmont Corp",            "Principale miner oro mondiale"),
            ("GOLD", "Barrick Gold",            "Secondo miner oro mondiale"),
            ("WPM",  "Wheaton Precious Metals", "Royalty streaming su oro/argento"),
        ]
    },
    "⚫ Energia & Petrolio": {
        "desc": "Conflitti in Medio Oriente o Russia fanno esplodere l'energia. Hedging naturale.",
        "assets": [
            ("USO",  "United States Oil Fund",  "ETF futures petrolio WTI"),
            ("BNO",  "United States Brent Oil", "ETF futures Brent (europeo)"),
            ("XOM",  "ExxonMobil",              "Prima Big Oil USA"),
            ("CVX",  "Chevron",                 "Big Oil USA, dividendo stabile"),
            ("XLE",  "Energy Select SPDR",      "ETF settore energia S&P500"),
            ("OXY",  "Occidental Petroleum",    "Preferita di Buffett"),
            ("VLO",  "Valero Energy",           "Raffinerie — beneficia da spread"),
            ("UNG",  "US Natural Gas Fund",     "ETF futures gas naturale"),
            ("LNG",  "Cheniere Energy",         "Esportatore LNG — guerra gas"),
        ]
    },
    "🔫 Difesa & Aerospazio": {
        "desc": "In caso di conflitto militare, i budget della difesa esplodono. Outperformer storici.",
        "assets": [
            ("LMT",  "Lockheed Martin",         "F-35, missili, sistemi difesa"),
            ("RTX",  "RTX Corp (Raytheon)",     "Missili Patriot, difesa aerea"),
            ("NOC",  "Northrop Grumman",        "B-21, sistemi spaziali, cyber"),
            ("GD",   "General Dynamics",        "Carri armati Abrams, navi"),
            ("BA",   "Boeing Defense",          "Aerei militari, elicotteri"),
            ("HII",  "Huntington Ingalls",      "Portaerei, sottomarini nucleari"),
            ("KTOS", "Kratos Defense",          "Droni, ipersonici, cyber"),
            ("CACI", "CACI International",      "Intelligence, cybersecurity gov"),
            ("ITA",  "iShares US Aerospace ETF","ETF settore difesa/aerospazio"),
            ("XAR",  "SPDR S&P Aerospace ETF",  "ETF difesa — più diversificato"),
        ]
    },
    "💊 Healthcare & Pharma": {
        "desc": "Settore difensivo per eccellenza. Domanda inelastica, dividendi stabili.",
        "assets": [
            ("JNJ",  "Johnson & Johnson",       "Healthcare diversificato, dividendo 60+ anni"),
            ("PFE",  "Pfizer",                  "Pharma globale, vaccini"),
            ("ABBV", "AbbVie",                  "Farmaceutico, alta cedola"),
            ("XLV",  "Health Care Select SPDR", "ETF healthcare S&P500"),
            ("IBB",  "iShares Biotech ETF",     "ETF biotech — più rischio/rendimento"),
        ]
    },
    "⚡ Utilities": {
        "desc": "Monopoli regolamentati, dividendi alti. Salgono quando i tassi scendono.",
        "assets": [
            ("XLU",  "Utilities Select SPDR",   "ETF utilities S&P500"),
            ("NEE",  "NextEra Energy",          "Prima utility USA, rinnovabili"),
            ("SO",   "Southern Company",        "Utility elettrica sud USA"),
            ("DUK",  "Duke Energy",             "Utility elettrica grande"),
            ("AWK",  "American Water Works",    "Acqua — utility anti-crisi"),
            ("VPU",  "Vanguard Utilities ETF",  "ETF utilities — costi bassi"),
        ]
    },
    "🏦 Treasuries & Obbligazioni": {
        "desc": "Flight-to-safety: in crisi il mercato compra T-Bond USA. Duration lunga = massimo beneficio.",
        "assets": [
            ("TLT",  "iShares 20+ Year Treasury","ETF treasury long duration — +forte"),
            ("IEF",  "iShares 7-10 Year Treasury","ETF treasury medium duration"),
            ("SHY",  "iShares 1-3 Year Treasury","ETF treasury short — cash-like"),
            ("TIPS", "iShares TIPS Bond ETF",   "ETF inflation-protected (TIPS)"),
            ("TIP",  "iShares TIPS ETF",        "TIPS — inflazione"),
            ("BIL",  "SPDR 1-3 Month T-Bill",   "Quasi-cash, rendimento risk-free"),
        ]
    },
    "🍞 Commodities & Agri": {
        "desc": "Guerra blocca export grano (Ucraina), mais, soia. Siccità + crisi = spike prezzi.",
        "assets": [
            ("DBA",  "Invesco DB Agriculture",  "ETF basket agri: grano, mais, soia"),
            ("WEAT", "Teucrium Wheat Fund",     "ETF puro grano — massima esposizione"),
            ("CORN", "Teucrium Corn Fund",      "ETF puro mais"),
            ("SOYB", "Teucrium Soybean Fund",   "ETF puro soia"),
            ("MOO",  "VanEck Agribusiness ETF", "Aziende agri: Deere, Mosaic"),
            ("MOS",  "The Mosaic Company",      "Fertilizzanti — crisi ucraina"),
            ("NTR",  "Nutrien",                 "Fertilizzanti — leader mondiale"),
        ]
    },
    "💵 Valute Rifugio": {
        "desc": "CHF e JPY si apprezzano in crisi. USD Index sale. Copre rischio valutario.",
        "assets": [
            ("FXF",  "Invesco CurrencyShares CHF","ETF franco svizzero vs USD"),
            ("FXY",  "Invesco CurrencyShares JPY","ETF yen giapponese vs USD"),
            ("UUP",  "Invesco DB USD Index Bull", "ETF dollaro USA (DXY long)"),
            ("UDN",  "Invesco DB USD Index Bear", "ETF short USD — hedge"),
        ]
    },
    "🪙 Crypto Rifugio": {
        "desc": "Bitcoin: 'oro digitale' per alcuni. Correlazione variabile con crisi tradizionali.",
        "assets": [
            ("IBIT", "iShares Bitcoin Trust",   "ETF Bitcoin spot BlackRock — più liquido"),
            ("FBTC", "Fidelity Bitcoin ETF",    "ETF Bitcoin spot Fidelity"),
            ("GBTC", "Grayscale Bitcoin Trust", "Il più vecchio veicolo Bitcoin"),
        ]
    },
    "🌍 Mercati Neutri / Commodity States": {
        "desc": "Paesi esportatori netti di commodities. Beneficiano da inflazione/guerra.",
        "assets": [
            ("EWZ",  "iShares Brazil ETF",      "Brasile: ferro, soia, petrolio"),
            ("EWC",  "iShares Canada ETF",      "Canada: petrolio, gas, oro"),
            ("EWA",  "iShares Australia ETF",   "Australia: ferro, carbone, LNG"),
            ("GXG",  "iShares Colombia ETF",    "Colombia: petrolio, carbone"),
            ("RSX",  "VanEck Russia ETF",       "Russia (attenzione: illiquido post-2022)"),
        ]
    },
}

CRISIS_LEGEND = {
    "🥇 Metalli Preziosi": "Rifugio universale. In ogni crisi guerra/inflazione l'oro sale. GLD/IAU = ETF più semplici. GDX/GDXJ = leva indiretta sui miner.",
    "⚫ Energia & Petrolio": "Conflitti in regioni produttrici → spike immediato del petrolio. XOM/CVX per dividendo stabile. USO/BNO per trading puro.",
    "🔫 Difesa & Aerospazio": "Budget difesa sale sempre in caso di conflitto. LMT, RTX, NOC = Big 3. ITA/XAR per esposizione ETF diversificata.",
    "💊 Healthcare & Pharma": "Domanda anelastica in ogni scenario. JNJ = qualità assoluta. XLV = ETF diversificato. ABBV per cedola elevata.",
    "⚡ Utilities": "Monopoli regolamentati con dividendi stabili. Sottoperformano in rialzo tassi, sovraperformano in panic/recessione. NEE = leader.",
    "🏦 Treasuries & Obbligazioni": "Flight-to-safety in crisi acute. TLT (20Y+) ha la massima duration = massimo guadagno se tassi scendono. TIPS contro inflazione.",
    "🍞 Commodities & Agri": "Ucraina e Russia = 30% export grano mondiale. Conflitto → spike immediato WEAT/CORN. DBA per basket diversificato.",
    "💵 Valute Rifugio": "CHF: mai in guerra dal 1815. JPY: carry trade → apprezzamento in crisi. UUP: dollaro sale in ogni stress globale.",
    "🪙 Crypto Rifugio": "Bitcoin come hedge è dibattuto: in crisi 2022 è sceso, in crisi bancaria 2023 è salito. IBIT (BlackRock) = più regolamentato.",
    "🌍 Mercati Neutri": "Paesi commodity-esportatori beneficiano da inflazione materie prime. Attenzione alla governance (EWZ) e sanzioni (RSX).",
}

LEGENDS={
    "EARLY":{"desc":"Titoli dove il prezzo è **vicino alla EMA20** — zona rimbalzo/continuazione. Ideale per ingressi anticipati.",
      "cols":[("Early_Score","0–10","Prossimità EMA20. ≥8 ottimo, 5-7 buono"),("RSI","0–100","Momentum. Blu<30, Verde 40-65, Rosso>70"),("Squeeze","🔥","Bollinger dentro Keltner: esplosione imminente")],
      "filters":"Stato_Early='EARLY' AND Early_Score ≥ soglia","sort":"Early_Score DESC"},
    "PRO":{"desc":"Trend confermato: prezzo>EMA20>EMA50, RSI neutro-rialzista, volume sopra media.",
      "cols":[("Pro_Score","0–8","+3 trend, +3 RSI, +2 volume. ≥8=PRO"),("Quality_Score","0–12","Composito 6 fattori. ≥9 alta qualità"),("RSI","40–70","Range ideale momentum")],
      "filters":"Stato_Pro='PRO' AND Pro_Score≥soglia_P AND Quality≥soglia_Q","sort":"Quality DESC"},
    "REA-HOT":{"desc":"Volumi anomali vicini al POC (Point of Control). Interesse istituzionale.",
      "cols":[("Vol_Ratio","x","Oggi/media20gg. >hot_soglia=trigger"),("Dist_POC_%","%","Distanza dal POC — minore=meglio"),("POC","$","Livello max volume storico")],
      "filters":"dist_poc<rpoc AND Vol_Ratio>vol_ratio_hot","sort":"Vol_Ratio DESC"},
    "⭐ CONFLUENCE":{"desc":"EARLY + PRO contemporaneamente. Setup ad altissima probabilità.",
      "cols":[("Early_Score","0–10","Timing"),("Pro_Score","0–8","Forza"),("Quality_Score","0–12","Qualità")],
      "filters":"Stato_Early='EARLY' AND Stato_Pro='PRO'","sort":"Quality DESC, Early DESC"},
    "Regime Momentum":{"desc":"PRO ordinati per Momentum = Pro×10+RSI. Maggiore forza relativa.",
      "cols":[("Momentum","calc","Pro_Score×10+RSI")],
      "filters":"Stato_Pro='PRO' AND Pro≥soglia","sort":"Momentum DESC"},
    "Multi-Timeframe":{"desc":"PRO con trend rialzista anche settimanale (EMA20 weekly).",
      "cols":[("Weekly_Bull","📈","Prezzo>EMA20 weekly"),("Quality_Score","0–12","Qualità daily")],
      "filters":"PRO AND Weekly_Bull=True","sort":"Quality DESC"},
    "Finviz":{"desc":"PRO con MarketCap≥mediana e Vol_Ratio>1.2. Focus liquido/istituzionale.",
      "cols":[("MarketCap","$","Cap≥mediana campione"),("Vol_Ratio","x",">1.2x partecipazione")],
      "filters":"PRO AND MarketCap≥median AND Vol_Ratio>1.2","sort":"Quality DESC"},
    "🎯 Serafini":{"desc":"**Metodo Stefano Serafini** — 6 criteri tecnici tutti soddisfatti: trend allineato, momentum, volume, no earnings imminenti.",
      "cols":[("Ser_Score","0–6","Criteri soddisfatti su 6"),("RSI>50","bool","Momentum positivo"),("EMA20>EMA50","bool","Trend allineato"),("OBV_UP","bool","Volume crescente"),("No_Earnings","bool","No earnings entro 14gg")],
      "filters":"Ser_OK=True (tutti e 6 i criteri)","sort":"Ser_Score DESC, Quality DESC"},
    "🔎 Finviz Pro":{"desc":"**Replica filtri Finviz** da immagine: Price>$10, AvgVol>1M, RelVol>1, Price above SMA20/50/200, EPS Next Year>10%, EPS 5Y>15%.",
      "cols":[("FV_Score","0–8","Filtri Finviz soddisfatti"),("EPS_NY_Gr","%","EPS Growth Next Year (>10%)"),("EPS_5Y_Gr","%","EPS Growth 5Y proxy (>15%)"),("EMA200","$","200-Day SMA"),("Avg_Vol_20","#","Average Volume 20gg"),("Rel_Vol","x","Relative Volume")],
      "filters":"Price > 10 AND AvgVol > 1M AND RelVol > 1 AND P > SMA20/50/200 AND EPS_NY > 10% AND EPS_5Y > 15%","sort":"FV_Score DESC, Quality DESC"},
}

def show_legend(key):
    info=LEGENDS.get(key)
    if not info: return
    with st.expander(f"📖 Come funziona: {key}",expanded=False):
        st.markdown(info["desc"])
        rows="".join(f'<tr><td class="legend-col-name">{c}</td><td class="legend-col-range">{r}</td><td>{d}</td></tr>'
                     for c,r,d in info["cols"])
        st.markdown(f"""<table class="legend-table"><tr><th>Colonna</th><th>Range</th><th>Significato</th></tr>
{rows}</table><br><span style="color:#6b7280;font-size:0.78rem">
🔬 <b>Filtro:</b> <code>{info["filters"]}</code> &nbsp;|&nbsp; 📊 <b>Sort:</b> <code>{info["sort"]}</code>
</span>""",unsafe_allow_html=True)

# =========================================================================
# RENDER SCAN TAB
# =========================================================================
def render_scan_tab(df,status_filter,sort_cols,ascending,title):
    if df is None or (hasattr(df,"empty") and df.empty):
        c1,c2=st.columns([3,1])
        c1.info(f"📭 Nessun dato in **{title}**. Avvia lo scanner dalla sidebar.")
        with c2:
            if st.button("🔄 Ricarica dal DB",key=f"reload_{title}"):
                try:
                    _h=load_scan_history(1)
                    if not _h.empty:
                        _id=int(_h.iloc[0]["id"])
                        ep,rea=load_scan_snapshot(_id)
                        st.session_state.df_ep=ep
                        st.session_state.df_rea=rea
                        st.session_state.last_scan=str(_h.iloc[0].get("scanned_at",""))[:16]
                        st.session_state.pop("_autoloaded",None)
                        st.rerun()
                except Exception as _e:
                    st.error(f"Errore ricarica: {_e}")
        return

    s_e=float(st.session_state.min_early_score)
    s_q=int(st.session_state.min_quality)
    s_p=float(st.session_state.min_pro_score)
    st.caption(f"🔬 Filtri: Early≥**{s_e}** | Quality≥**{s_q}** | Pro≥**{s_p}** _(sidebar → 🔬)_")

    if status_filter=="EARLY":
        if "Stato_Early" not in df.columns: st.warning("Colonna Stato_Early mancante."); return
        df_f=df[df["Stato_Early"]=="EARLY"].copy()
        if "Early_Score" in df_f.columns and s_e>0: df_f=df_f[df_f["Early_Score"]>=s_e]

    elif status_filter=="PRO":
        if "Stato_Pro" not in df.columns: st.warning("Colonna Stato_Pro mancante."); return
        df_f=df[df["Stato_Pro"]=="PRO"].copy()
        if "Pro_Score"     in df_f.columns and s_p>0: df_f=df_f[df_f["Pro_Score"]    >=s_p]
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="HOT":
        # REA-HOT: df_rea contiene già solo i HOT ma filtriamo per sicurezza
        if df is None or (hasattr(df,"empty") and df.empty):
            st.info("📭 Nessun segnale HOT trovato. Il segnale REA-HOT richiede"
                    " Vol_Ratio > soglia E distanza dal POC < soglia.\n\n"
                    " Abbassa `vol_ratio_hot` o `rpoc` nella sidebar → ⚙️ Avanzate.")
            return
        if "Stato" in df.columns:
            df_f=df[df["Stato"]=="HOT"].copy()
        else:
            df_f=df.copy()  # df_rea è già pre-filtrata

    elif status_filter=="CONFLUENCE":
        if "Stato_Early" not in df.columns or "Stato_Pro" not in df.columns:
            st.warning("Colonne Stato mancanti."); return
        df_f=df[(df["Stato_Early"]=="EARLY")&(df["Stato_Pro"]=="PRO")].copy()
        if "Early_Score"   in df_f.columns and s_e>0: df_f=df_f[df_f["Early_Score"]  >=s_e]
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="REGIME":
        df_f=df[df["Stato_Pro"]=="PRO"].copy() if "Stato_Pro" in df.columns else df.copy()
        if "Pro_Score" in df_f.columns and s_p>0: df_f=df_f[df_f["Pro_Score"]>=s_p]
        if "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
            df_f["Momentum"]=df_f["Pro_Score"]*10+df_f["RSI"]
            sort_cols=["Momentum"]; ascending=[False]

    elif status_filter=="MTF":
        df_f=df[df["Stato_Pro"]=="PRO"].copy() if "Stato_Pro" in df.columns else df.copy()
        if "Pro_Score"   in df_f.columns and s_p>0: df_f=df_f[df_f["Pro_Score"]>=s_p]
        if "Weekly_Bull" in df_f.columns:
            df_f=df_f[df_f["Weekly_Bull"].isin([True,"True","true",1])]

    elif status_filter=="SERAFINI":
        if "Ser_OK" not in df.columns:
            st.warning("Colonna Ser_OK non trovata. Riesegui scanner v29.0."); return
        df_f=df[df["Ser_OK"].isin([True,"True","true"])].copy()
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="FINVIZ_PRO":
        if "FV_Score" not in df.columns:
            st.warning("Colonna FV_Score non trovata. Riesegui scanner v29.0."); return
        df_f=df[df["FV_OK"].isin([True,"True","true"])].copy()
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    else:
        df_f=df.copy()

    if df_f.empty:
        # Conta quanti aveva prima dei filtri soglia (per diagnostica)
        _n_pre = len(df)
        _tipo_check = {
            "EARLY":     df.get("Stato_Early","").eq("EARLY").sum() if "Stato_Early" in df.columns else 0,
            "PRO":       df.get("Stato_Pro","").eq("PRO").sum()     if "Stato_Pro"   in df.columns else 0,
            "HOT":       df.get("Stato","").eq("HOT").sum()         if "Stato"       in df.columns else 0,
            "CONFLUENCE":((df.get("Stato_Early","").eq("EARLY"))&(df.get("Stato_Pro","").eq("PRO"))).sum()
                          if ("Stato_Early" in df.columns and "Stato_Pro" in df.columns) else 0,
            "SERAFINI":  df.get("Ser_OK","").isin([True,"True","true"]).sum() if "Ser_OK" in df.columns else 0,
            "FINVIZ_PRO":df.get("FV_OK","").isin([True,"True","true"]).sum()  if "FV_OK"  in df.columns else 0,
            "MTF":       df.get("Weekly_Bull","").isin([True,"True","true",1]).sum() if "Weekly_Bull" in df.columns else 0,
        }.get(status_filter, 0)
        if _tipo_check > 0:
            st.warning(
                f"⚠️ **{title}**: {_tipo_check} segnali trovati, ma tutti filtrati via soglie "
                f"(Early≥{s_e} | Quality≥{s_q} | Pro≥{s_p}).\n\n"
                f"👉 Abbassa le soglie nella sidebar → 🔬 Soglie oppure portale a 0."
            )
        else:
            st.info(
                f"📭 **{title}**: nessun segnale in questo scan "
                f"({_n_pre} titoli analizzati totali).\n\n"
                f"💡 Riprova con mercati diversi o parametri scanner più permissivi."
            )
        return

    valid_sort=[c for c in sort_cols if c in df_f.columns]
    if valid_sort: df_f=df_f.sort_values(valid_sort,ascending=ascending[:len(valid_sort)])
    df_f=df_f.head(int(st.session_state.top))

    m1,m2,m3,m4=st.columns(4)
    m1.metric("Titoli",len(df_f))
    if "Squeeze" in df_f.columns:
        m2.metric("🔥 Squeeze",int(df_f["Squeeze"].apply(lambda x:x is True or str(x).lower()=="true").sum()))
    if "Weekly_Bull" in df_f.columns:
        m3.metric("📈 Weekly+",int(df_f["Weekly_Bull"].apply(lambda x:x is True or str(x).lower()=="true").sum()))
    if "RSI_Div" in df_f.columns:
        m4.metric("⚠️ Div RSI",int((df_f["RSI_Div"]!="-").sum()))

    df_fmt =add_formatted_cols(df_f)
    df_disp=prepare_display_df(df_fmt)
    # Rimuovi colonne interne (prefisso _ e criteri grezzi)
    drop_cols=[c for c in df_disp.columns if c.startswith("_")]
    df_disp=df_disp.drop(columns=drop_cols, errors="ignore")
    # Ordine: Ticker, Nome, Prezzo_fmt, MarketCap_fmt, poi segnali, poi resto
    cols=list(df_disp.columns)
    priority=["Ticker","Nome","Prezzo_fmt","MarketCap_fmt","Early_Score","Pro_Score",
               "RSI","Vol_Ratio","Quality_Score","Stato_Early","Stato_Pro","EMA200_fmt"]
    base=[c for c in priority if c in cols]
    rest=[c for c in cols if c not in base]
    df_disp=df_disp[base+rest].reset_index(drop=True)

    ce1,ce2=st.columns([1,3])
    with ce1: csv_btn(df_f,f"{title.lower().replace(' ','_')}.csv",f"exp_{title}")
    with ce2: st.caption(f"Seleziona → **➕** per aggiungere a `{st.session_state.current_list_name}`. Doppio click Nome → TradingView.")

    grid_resp  =build_aggrid(df_disp,f"grid_{title}")
    selected_df=pd.DataFrame(grid_resp["selected_rows"])

    if st.button(f"➕ Aggiungi a '{st.session_state.current_list_name}'",key=f"btn_{title}"):
        if not selected_df.empty and "Ticker" in selected_df.columns:
            tickers=selected_df["Ticker"].tolist()
            names  =selected_df.get("Nome",selected_df["Ticker"]).tolist()
            gh_add_to_watchlist(tickers,names,title,"Scanner","LONG",st.session_state.current_list_name)
            st.success(f"✅ Aggiunti {len(tickers)} titoli a '{st.session_state.current_list_name}'.")
            time.sleep(0.8); st.rerun()
        else: st.warning("Seleziona almeno una riga.")

    if not selected_df.empty:
        ticker_sel=selected_df.iloc[0].get("Ticker","")
        match=df_f[df_f["Ticker"]==ticker_sel]
        if not match.empty: show_charts(match.iloc[0],key_suffix=title)

    # ── Strategy Chart widget ─────────────────────────────────────────────
    # Usa tutti i ticker del tab come opzioni selectbox
        # --- Strategy Chart GENERICO rimosso per evitare key duplicate ---
    # try:
    #     from utils.backtest_tab import strategy_chart_widget as _scw
    #     tkrs = dff["Ticker"].dropna().tolist() if "Ticker" in dff.columns else []
    #     default = selected_df.iloc[0].get("Ticker", "") if not selected_df.empty else ""
    #     st.markdown("---")
    #     _scw(tickers=tkrs, key_suffix=title, default_ticker=default)
    # except Exception:
    #     pass


# =========================================================================
# TABS
# =========================================================================
tabs=st.tabs(["🏠 Home",
              "📊 Comparatore",
              "💎 Blue Chip Dip",
              "📡 EARLY","💪 PRO","🔥 REA-HOT","⭐ CONFLUENCE",
              "🎯 Serafini","🔎 Finviz Pro",
              "🔬 Order Flow",
              "🛡️ Crisis Monitor",
              "📋 Watchlist","📈 Backtest"])
(tab_home,tab_mtf,tab_bcd,tab_e,tab_p,tab_r,tab_conf,
 tab_ser,tab_fvpro,tab_of,tab_crisis,tab_w,tab_bt)=tabs

with tab_home:
    try:
        from utils.home_tab import render_home
        render_home(df_ep, df_rea)
    except Exception as _he:
        import traceback
        st.error(f"Home tab error: {_he}")
        st.code(traceback.format_exc())

with tab_e:
    st.session_state.last_active_tab = "EARLY"
    show_legend("EARLY")

    # --- Scan table EARLY ---------------------------------------------------
    render_scan_tab(
        df_ep,
        "EARLY",
        ["Early_Score", "RSI"],
        [False, True],
        "EARLY",
    )

    # --- Strategy Chart EARLY ----------------------------------------------
    try:
        from utils.backtest_tab import strategy_chart_widget

        df_src = df_ep.copy() if df_ep is not None else pd.DataFrame()

        if not df_src.empty and "Ticker" in df_src.columns:
            if "Nome" not in df_src.columns and "name" in df_src.columns:
                df_src["Nome"] = df_src["name"]

            df_src = df_src.dropna(subset=["Ticker"])
            df_src["Ticker"] = df_src["Ticker"].astype(str).str.upper()
            df_src["Nome"] = df_src.get("Nome", "").fillna("").astype(str)

            # Mapping Nome (TICKER) ordinato alfabeticamente
            base = (
                df_src[["Ticker", "Nome"]]
                .drop_duplicates(subset=["Ticker"])
                .sort_values("Nome")
            )
            tickers = base["Ticker"].tolist()
            ticker_labels = {
                t: f"{n} ({t})" if n else t
                for t, n in zip(base["Ticker"], base["Nome"])
            }
            default_ticker = tickers[0] if tickers else ""

            st.markdown("---")
            strategy_chart_widget(
                tickers=tickers,
                key_suffix="EARLY",
                default_ticker=default_ticker,
                ticker_labels=ticker_labels,
            )
        else:
            st.markdown("---")
            strategy_chart_widget(
                tickers=[],
                key_suffix="EARLY",
                default_ticker="",
                ticker_labels=None,
            )
    except Exception as _sce:
        st.error(f"Errore Strategy Chart EARLY: {_sce}")


with tab_p:
    st.session_state.last_active_tab = "PRO"
    show_legend("PRO")

    _pro_sort = st.radio(
        "Ordina per",
        ["Quality", "Momentum (Pro×RSI)"],
        horizontal=True,
        key="pro_sort_mode",
        label_visibility="collapsed",
    )

    if _pro_sort == "Momentum (Pro×RSI)":
        _df_pro = df_ep.copy() if df_ep is not None else pd.DataFrame()
        if (
            not _df_pro.empty
            and "Pro_Score" in _df_pro.columns
            and "RSI" in _df_pro.columns
        ):
            _df_pro["_Momentum"] = (
                _df_pro["Pro_Score"].fillna(0) * 10
                + _df_pro["RSI"].fillna(0)
            )
        else:
            _df_pro["_Momentum"] = 0
        render_scan_tab(
            _df_pro,
            "PRO",
            ["_Momentum", "Quality_Score"],
            [False, False],
            "PRO — Momentum",
        )
        df_src = _df_pro
        title_for_sc = "PRO — Momentum"
    else:
        render_scan_tab(
            df_ep,
            "PRO",
            ["Quality_Score", "Pro_Score", "RSI"],
            [False, False, True],
            "PRO",
        )
        df_src = df_ep
        title_for_sc = "PRO"

    # --- Strategy Chart PRO -----------------------------------------------
    try:
        from utils.backtest_tab import strategy_chart_widget

        df_sc = df_src.copy() if df_src is not None else pd.DataFrame()
        if not df_sc.empty and "Ticker" in df_sc.columns:
            if "Nome" not in df_sc.columns and "name" in df_sc.columns:
                df_sc["Nome"] = df_sc["name"]

            df_sc = df_sc.dropna(subset=["Ticker"])
            df_sc["Ticker"] = df_sc["Ticker"].astype(str).str.upper()
            df_sc["Nome"] = df_sc.get("Nome", "").fillna("").astype(str)

            base = (
                df_sc[["Ticker", "Nome"]]
                .drop_duplicates(subset=["Ticker"])
                .sort_values("Nome")
            )
            tickers = base["Ticker"].tolist()
            ticker_labels = {
                t: f"{n} ({t})" if n else t
                for t, n in zip(base["Ticker"], base["Nome"])
            }
            default_ticker = tickers[0] if tickers else ""

            st.markdown("---")
            strategy_chart_widget(
                tickers=tickers,
                key_suffix="PRO",
                default_ticker=default_ticker,
                ticker_labels=ticker_labels,
            )
        else:
            st.markdown("---")
            strategy_chart_widget(
                tickers=[],
                key_suffix="PRO",
                default_ticker="",
                ticker_labels=None,
            )
    except Exception as _scp:
        st.error(f"Errore Strategy Chart PRO: {_scp}")


with tab_r:
    st.session_state.last_active_tab = "REA-HOT"
    show_legend("REA-HOT")

    render_scan_tab(
        df_rea,
        "HOT",
        ["Vol_Ratio", "Dist_POC_%"],
        [False, True],
        "REA-HOT",
    )

    # --- Strategy Chart REA-HOT -------------------------------------------
    try:
        from utils.backtest_tab import strategy_chart_widget

        df_src = df_rea.copy() if df_rea is not None else pd.DataFrame()
        if not df_src.empty and "Ticker" in df_src.columns:
            if "Nome" not in df_src.columns and "name" in df_src.columns:
                df_src["Nome"] = df_src["name"]

            df_src = df_src.dropna(subset=["Ticker"])
            df_src["Ticker"] = df_src["Ticker"].astype(str).str.upper()
            df_src["Nome"] = df_src.get("Nome", "").fillna("").astype(str)

            base = (
                df_src[["Ticker", "Nome"]]
                .drop_duplicates(subset=["Ticker"])
                .sort_values("Nome")
            )
            tickers = base["Ticker"].tolist()
            ticker_labels = {
                t: f"{n} ({t})" if n else t
                for t, n in zip(base["Ticker"], base["Nome"])
            }
            default_ticker = tickers[0] if tickers else ""

            st.markdown("---")
            strategy_chart_widget(
                tickers=tickers,
                key_suffix="HOT",
                default_ticker=default_ticker,
                ticker_labels=ticker_labels,
            )
        else:
            st.markdown("---")
            strategy_chart_widget(
                tickers=[],
                key_suffix="HOT",
                default_ticker="",
                ticker_labels=None,
            )
    except Exception as _scr:
        st.error(f"Errore Strategy Chart REA-HOT: {_scr}")

with tab_conf:
    st.session_state.last_active_tab = "CONFLUENCE"
    show_legend("⭐ CONFLUENCE")

    render_scan_tab(
        df_ep,
        "CONFLUENCE",
        ["Quality_Score", "Early_Score", "Pro_Score"],
        [False, False, False],
        "CONFLUENCE",
    )

    # --- Strategy Chart CONFLUENCE ----------------------------------------
    try:
        from utils.backtest_tab import strategy_chart_widget

        df_src = df_ep.copy() if df_ep is not None else pd.DataFrame()
        if not df_src.empty and "Ticker" in df_src.columns:
            if "Nome" not in df_src.columns and "name" in df_src.columns:
                df_src["Nome"] = df_src["name"]

            df_src = df_src.dropna(subset=["Ticker"])
            df_src["Ticker"] = df_src["Ticker"].astype(str).str.upper()
            df_src["Nome"] = df_src.get("Nome", "").fillna("").astype(str)

            base = (
                df_src[["Ticker", "Nome"]]
                .drop_duplicates(subset=["Ticker"])
                .sort_values("Nome")
            )
            tickers = base["Ticker"].tolist()
            ticker_labels = {
                t: f"{n} ({t})" if n else t
                for t, n in zip(base["Ticker"], base["Nome"])
            }
            default_ticker = tickers[0] if tickers else ""

            st.markdown("---")
            strategy_chart_widget(
                tickers=tickers,
                key_suffix="CONF",
                default_ticker=default_ticker,
                ticker_labels=ticker_labels,
            )
        else:
            st.markdown("---")
            strategy_chart_widget(
                tickers=[],
                key_suffix="CONF",
                default_ticker="",
                ticker_labels=None,
            )
    except Exception as _scc:
        st.error(f"Errore Strategy Chart CONFLUENCE: {_scc}")


with tab_mtf:
    try:
        from utils.compare_tab import render_compare
        _df_scan_all = pd.concat(
            [df for df in [df_ep, df_rea] if df is not None and not df.empty],
            ignore_index=True
        ) if any(df is not None and not df.empty for df in [df_ep, df_rea]) else None
        render_compare(_df_scan_all)
    except ImportError:
        st.info("📊 compare_tab.py non trovato in utils/")
    except Exception as _ce:
        st.error(f"Comparatore error: {_ce}")

with tab_ser:
    show_legend("🎯 Serafini")
    # Mostra criteri dettaglio
    with st.expander("✅ Criteri Serafini nel dettaglio",expanded=False):
        st.markdown("""
| # | Criterio | Calcolo | Soglia |
|---|----------|---------|--------|
| 1 | **RSI > 50** | RSI(14) | >50 |
| 2 | **Prezzo > EMA20** | Close > EMA(20) | Sì |
| 3 | **EMA20 > EMA50** | EMA(20) > EMA(50) | Sì |
| 4 | **OBV crescente** | OBV slope 5gg > 0 | Sì |
| 5 | **Volume > media** | Vol_Ratio > 1.0 | Sì |
| 6 | **No earnings prossimi** | Earnings Date > 14gg | Sì |

Tutti e 6 devono essere **True** per `Ser_OK=True`.  
`Ser_Score` indica quanti criteri su 6 sono soddisfatti (utile per veder titoli quasi-qualificati).
""")
    render_scan_tab(df_ep,"SERAFINI",["Ser_Score","Quality_Score","RSI"],[False,False,True],"🎯 Serafini")

with tab_fvpro:
    show_legend("🔎 Finviz Pro")
    with st.expander("✅ Filtri Finviz replicati",expanded=False):
        st.markdown("""
| Filtro Finviz | Replica yfinance | Soglia |
|---|---|---|
| Price $ | `Close` | > $10 |
| Average Volume | `avg_vol_20` | > 1.000.000 |
| Relative Volume | `vol_today / avg_vol_20` | > 1.0 |
| 20-Day SMA | `Close > EMA(20)` | Sì |
| 50-Day SMA | `Close > EMA(50)` | Sì |
| 200-Day SMA | `Close > SMA(200)` | Sì |
| EPS Growth Next Year | `(forwardEPS-trailingEPS)/abs(trailingEPS)` | > 10% |
| EPS Growth Next 5Y | `revenueGrowth` _(proxy)_ | > 15% |
| Optionable | Exchange in [NMS,NYQ,ASE,...] _(proxy)_ | — (info) |

> ⚠️ I dati fondamentali EPS Growth dipendono dalla disponibilità in yfinance.  
> Per dati precisi si consiglia Finviz Elite API.
""")
    render_scan_tab(df_ep,"FINVIZ_PRO",["FV_Score","Quality_Score","EPS_NY_Gr"],[False,False,False],"🔎 Finviz Pro")

# =========================================================================
# CRISIS MONITOR TAB
# =========================================================================
with tab_crisis:
    st.markdown('<div class="section-pill">🛡️ CRISIS MONITOR — Asset Difensivi</div>',
                unsafe_allow_html=True)

    st.markdown("""
> **Come usare questo tab**: seleziona lo scenario di rischio che ti preoccupa.
> Per ogni asset trovi ticker, nome e descrizione tattica. Clicca sul ticker per aprire TradingView.
> Aggiungi alla watchlist per seguire l'analisi tecnica con lo scanner.
""")

    # ── Selezione scenario ────────────────────────────────────────────
    scenario_labels = {
        "🌍 Guerra / Conflitto Militare":  ["🥇 Metalli Preziosi","⚫ Energia & Petrolio","🔫 Difesa & Aerospazio","🏦 Treasuries & Obbligazioni","💵 Valute Rifugio"],
        "📈 Inflazione Alta":              ["🥇 Metalli Preziosi","⚫ Energia & Petrolio","🍞 Commodities & Agri","💵 Valute Rifugio","🌍 Mercati Neutri / Commodity States"],
        "🧱 Stagflazione":                 ["🥇 Metalli Preziosi","⚫ Energia & Petrolio","🍞 Commodities & Agri","⚡ Utilities","💊 Healthcare & Pharma","💵 Valute Rifugio"],
        "📉 Crash / Panic Sell":            ["🥇 Metalli Preziosi","🏦 Treasuries & Obbligazioni","⚡ Utilities","💊 Healthcare & Pharma","💵 Valute Rifugio"],
        "🦠 Pandemia / Crisi Sanitaria":   ["💊 Healthcare & Pharma","🥇 Metalli Preziosi","⚡ Utilities","🏦 Treasuries & Obbligazioni"],
        "💻 Crisi Energetica":             ["⚫ Energia & Petrolio","⚡ Utilities","🌍 Mercati Neutri / Commodity States"],
        "📊 Tutti gli asset difensivi":    list(CRISIS_ASSETS.keys()),
    }

    # Inizializza session_state per evitare crash al primo render
    if "crisis_scenario" not in st.session_state:
        st.session_state["crisis_scenario"] = list(scenario_labels.keys())[0]

    sc_col1, sc_col2 = st.columns([2, 3])
    with sc_col1:
        selected_scenario = st.selectbox(
            "🎯 Seleziona scenario di rischio",
            list(scenario_labels.keys()),
            key="crisis_scenario"
        )
    with sc_col2:
        _n_cats   = len(scenario_labels.get(selected_scenario, []))
        _n_assets = sum(len(CRISIS_ASSETS.get(c,{}).get("assets",[]))
                        for c in scenario_labels.get(selected_scenario, []))
        st.markdown(f"""
<div style="background:#1a2332;border:1px solid #2d3f55;border-radius:8px;padding:10px;margin-top:8px">
<b style="color:#60a5fa">Scenario selezionato:</b>
<span style="color:#e2e8f0"> {selected_scenario}</span><br>
<span style="color:#6b7280;font-size:0.82rem">{_n_cats} categorie — {_n_assets} asset totali</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    active_categories = scenario_labels[selected_scenario]
    all_crisis_tickers = []

    # ── Scanner fisso Crisis Monitor ───────────────────────────────────────
    # Scansiona TUTTI i ticker CRISIS_ASSETS (indipendente dalla sidebar)
    # Cache unica per tutti gli scenari — si aggiorna solo su richiesta esplicita

    def _slug_fast(s):
        import re as _re2
        return _re2.sub(r'[^\w]','',s)[:16]

    # Lista COMPLETA di tutti i ticker crisis (tutte le categorie)
    _ALL_CRISIS_TKS = list(dict.fromkeys(
        t for _cat in CRISIS_ASSETS.values()
        for t,_,_ in _cat.get("assets", [])
    ))
    _CRISIS_CACHE_KEY = "_crisis_scan_all"   # chiave unica per tutti gli scenari

    _crisis_df_cached = st.session_state.get(_CRISIS_CACHE_KEY)

    # Barra di stato + bottone aggiorna
    _hc1, _hc2, _hc3 = st.columns([3, 2, 3])
    with _hc1:
        if _crisis_df_cached is not None:
            _ts = st.session_state.get("_crisis_scan_time", "")
            st.markdown(
                f'<div style="background:#1a2e1a;border:1px solid #2a4a2a;'
                f'border-radius:6px;padding:8px 12px;font-size:0.82rem">'
                f'✅ <b style="color:#26a69a">{len(_crisis_df_cached)} ticker</b>'
                f' con dati live'
                f'{"  ·  🕐 " + _ts if _ts else ""}'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#2e1a1a;border:1px solid #4a2a2a;'
                'border-radius:6px;padding:8px 12px;font-size:0.82rem">'
                '⚠️ <b style="color:#ef5350">Dati non disponibili</b>'
                ' — premi Scansiona per popolare RSI, Quality, Vol×, etc.'
                '</div>', unsafe_allow_html=True
            )
    with _hc2:
        _run_crisis = st.button(
            f"🔍 Scansiona tutti ({len(_ALL_CRISIS_TKS)})",
            key="crisis_scan_btn",
            type="primary",
            use_container_width=True,
            help=f"Scarica dati tecnici live per tutti i {len(_ALL_CRISIS_TKS)} asset difensivi — indipendente dalla selezione mercati"
        )
    with _hc3:
        _col_rf, _col_rs = st.columns(2)
        with _col_rf:
            if _crisis_df_cached is not None:
                if st.button("🔄 Aggiorna", key="crisis_scan_refresh",
                             use_container_width=True,
                             help="Forza nuova scansione"):
                    st.session_state.pop(_CRISIS_CACHE_KEY, None)
                    st.session_state.pop("_crisis_scan_time", None)
                    st.rerun()
        with _col_rs:
            if _crisis_df_cached is not None:
                if st.button("🗑️ Reset", key="crisis_scan_reset",
                             use_container_width=True):
                    st.session_state.pop(_CRISIS_CACHE_KEY, None)
                    st.session_state.pop("_crisis_scan_time", None)
                    st.rerun()

    # ── Esegui scansione ──────────────────────────────────────────────────
    if _run_crisis:
        _crisis_rows = []
        _crisis_errors = []
        _prog = st.progress(0, text="🔍 Avvio scansione Crisis Monitor...")
        _n = len(_ALL_CRISIS_TKS)
        for _i, _tkr in enumerate(_ALL_CRISIS_TKS):
            _prog.progress((_i + 1) / _n,
                           text=f"🔍 {_tkr}  ({_i+1}/{_n})")
            try:
                _ep_row, _rea_row = scan_ticker(
                    _tkr,
                    e_h=0.03,
                    p_rmin=25,
                    p_rmax=85,
                    r_poc=0.03,
                    vol_ratio_hot=1.2,
                )
                _row = _ep_row if _ep_row is not None else _rea_row
                if _row is not None:
                    _crisis_rows.append(_row)
                else:
                    _crisis_errors.append(f"{_tkr}: scan_ticker → (None, None)")
            except Exception as _ex:
                _crisis_errors.append(f"{_tkr}: {type(_ex).__name__}: {_ex}")
        _prog.empty()
        if _crisis_rows:
            _crisis_df = pd.DataFrame(_crisis_rows)
            st.session_state[_CRISIS_CACHE_KEY] = _crisis_df
            st.session_state["_crisis_scan_time"] = datetime.now().strftime("%H:%M")
            if _crisis_errors:
                with st.expander(f"⚠️ {len(_crisis_errors)} ticker non caricati", expanded=False):
                    st.code("\n".join(_crisis_errors[:20]))
            st.success(f"✅ Scansione completata: {len(_crisis_df)}/{_n} ticker")
            st.rerun()
        else:
            st.error(f"⚠️ Nessun dato recuperato ({_n} ticker tentati). Errori:")
            st.code("\n".join(_crisis_errors[:30]) if _crisis_errors else "Nessun errore registrato — scan_ticker ha restituito None per tutti")

    # Dati live da usare nel merge per ogni categoria
    _crisis_live = st.session_state.get(_CRISIS_CACHE_KEY)

    # ── Per ogni categoria — lista ticker con filtro scenario attivo ───────
    _all_crisis_tks = []   # mantieni compatibilità variabile downstream

    def _slug(s, maxlen=12):
        """Rimuove emoji e spazi per creare una chiave Streamlit valida."""
        import re as _re
        clean = _re.sub(r'[^\w]', '', s)
        return clean[:maxlen] if clean else "cat"

# ── Per ogni categoria ─────────────────────────────────────────────
    for cat_name in active_categories:
        cat_data = CRISIS_ASSETS.get(cat_name, {})
        if not cat_data: continue
        assets = cat_data.get("assets", [])
        if not assets: continue

        st.markdown(f"### {cat_name}")
        st.markdown(f"*{CRISIS_LEGEND.get(cat_name, cat_data.get('desc',''))}*")

        rows = [{"Ticker": t, "Nome": n, "Descrizione Tattica": d} for t,n,d in assets]
        df_crisis_cat = pd.DataFrame(rows)
        all_crisis_tickers.extend([r[0] for r in assets])
        # Arricchisci con dati scanner: prima crisis scan dedicato, poi scanner principale
        _live_keep = ["Ticker","Prezzo","RSI","Vol_Ratio","OBV_Trend",
                      "Stato_Early","Quality_Score","Early_Score","Pro_Score",
                      "Squeeze","Weekly_Bull"]
        for _ldf in [_crisis_live, df_ep, df_rea]:
            if _ldf is None or _ldf.empty or "Ticker" not in _ldf.columns: continue
            _sub = _ldf[[c for c in _live_keep if c in _ldf.columns]].copy()
            df_crisis_cat = df_crisis_cat.merge(_sub, on="Ticker", how="left")
            break  # primo df disponibile basta

        gb_c = GridOptionsBuilder.from_dataframe(df_crisis_cat)
        gb_c.configure_default_column(sortable=True, resizable=True, filterable=False, minWidth=65)
        gb_c.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb_c.configure_column("Ticker", width=100, pinned="left",
            cellRenderer=JsCode("""class T{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value||'';const t=p.value;if(!t)return;
this.eGui.style.cursor='pointer';this.eGui.style.color='#50c4e0';
this.eGui.style.fontWeight='bold';this.eGui.style.fontFamily='Trebuchet MS';
this.eGui.title='Doppio click → TradingView';
this.eGui.ondblclick=()=>window.open('https://it.tradingview.com/chart/?symbol='+String(t).split('.')[0],'_blank');}
getGui(){return this.eGui;}refresh(){return false;}}"""))
        gb_c.configure_column("Nome", width=195)
        gb_c.configure_column("Descrizione Tattica", width=360, wrapText=True, autoHeight=True)
        # Colonne dati live (se disponibili dallo scanner)
        if "Prezzo" in df_crisis_cat.columns:
            gb_c.configure_column("Prezzo", width=88, headerName="Prezzo $",
                cellRenderer=JsCode("""class P{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'—':'$'+v.toFixed(2);
this.eGui.style.color='#d1d4dc';this.eGui.style.fontWeight='600';}
getGui(){return this.eGui;}refresh(){return false;}}"""))
        if "RSI" in df_crisis_cat.columns:
            gb_c.configure_column("RSI", width=68, cellRenderer=rsi_renderer)
        if "Vol_Ratio" in df_crisis_cat.columns:
            gb_c.configure_column("Vol_Ratio", width=82, headerName="Vol×",
                cellRenderer=vol_ratio_renderer)
        if "Quality_Score" in df_crisis_cat.columns:
            gb_c.configure_column("Quality_Score", width=82, headerName="Quality",
                cellRenderer=quality_renderer)
        if "OBV_Trend" in df_crisis_cat.columns:
            gb_c.configure_column("OBV_Trend", width=80, headerName="OBV Trend")
        if "Stato_Early" in df_crisis_cat.columns:
            gb_c.configure_column("Stato_Early", width=85, headerName="Stato")
        if "Early_Score" in df_crisis_cat.columns:
            gb_c.configure_column("Early_Score", width=72, headerName="E.Score")
        if "Pro_Score" in df_crisis_cat.columns:
            gb_c.configure_column("Pro_Score", width=72, headerName="P.Score")
        if "Squeeze" in df_crisis_cat.columns:
            gb_c.configure_column("Squeeze", width=72, cellRenderer=squeeze_renderer)
        if "Weekly_Bull" in df_crisis_cat.columns:
            gb_c.configure_column("Weekly_Bull", width=68, headerName="W+",
                cellRenderer=weekly_renderer)
        go_c = gb_c.build()

        try:
            resp_c = AgGrid(df_crisis_cat, gridOptions=go_c,
                            height=min(120 + len(assets)*35, 440),
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            fit_columns_on_grid_load=True, theme="streamlit",
                            allow_unsafe_jscode=True, key=f"cg_{_slug(cat_name)}")
            sel_crisis = pd.DataFrame(resp_c["selected_rows"])
        except Exception as _ag_err:
            # Fallback: dataframe semplice se AgGrid non disponibile
            st.dataframe(df_crisis_cat, use_container_width=True, hide_index=True)
            sel_crisis = pd.DataFrame()

        c_a1, c_a2, _ = st.columns([2, 2, 4])
        with c_a1:
            if st.button(f"➕ Aggiungi selezionati", key=f"cadd_{_slug(cat_name)}"):
                if not sel_crisis.empty and "Ticker" in sel_crisis.columns:
                    tks = sel_crisis["Ticker"].tolist()
                    nms = sel_crisis["Nome"].tolist()
                    gh_add_to_watchlist(tks, nms, f"Crisis:{cat_name[:18]}", "CrisisMonitor",
                                     "WATCH", st.session_state.current_list_name)
                    st.success(f"✅ Aggiunti {len(tks)} ticker."); time.sleep(0.5); st.rerun()
                else:
                    st.warning("Seleziona almeno un asset dalla griglia.")
        with c_a2:
            if st.button(f"➕ Tutti ({len(assets)})", key=f"call_{_slug(cat_name)}"):
                tks=[r[0] for r in assets]; nms=[r[1] for r in assets]
                gh_add_to_watchlist(tks, nms, f"Crisis:{cat_name[:18]}", "CrisisMonitor",
                                 "WATCH", st.session_state.current_list_name)
                st.success(f"✅ Aggiunti tutti i {len(tks)} ticker."); time.sleep(0.5); st.rerun()
        # ── Grafico ticker selezionato (come negli altri tab) ──────────
        # Controlla selezione esplicita (non solo riga pre-selezionata)
        _has_selection = (not sel_crisis.empty
                          and "Ticker" in sel_crisis.columns
                          and len(sel_crisis) > 0)
        if _has_selection:
            _ctkr = sel_crisis.iloc[0].get("Ticker","")
            _crow = None
            for _cdf in [df_ep, df_rea]:
                if _cdf is None or _cdf.empty or "Ticker" not in _cdf.columns: continue
                _cm = _cdf[_cdf["Ticker"]==_ctkr]
                if not _cm.empty and "_chart_data" in _cm.columns:
                    _cd = _cm.iloc[0].get("_chart_data")
                    if _cd and isinstance(_cd, dict) and _cd.get("dates"):
                        _crow = _cm.iloc[0]; break
            if _crow is not None:
                show_charts(_crow, key_suffix=f"cr_{_slug(cat_name)}")
            else:
                st.info(f"📭 Dati tecnici per **{_ctkr}** non disponibili. "
                        f"Esegui lo scanner su questo mercato.")
        st.markdown("")

    # ── Legenda e guida ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 Guida — Come usare il Crisis Monitor e performance storiche", expanded=False):
        st.markdown("""
## 🛡️ Crisis Monitor — Guida Operativa

### 📊 Come usare il tab
| Azione | Come fare |
|--------|-----------|
| **Aprire grafico** | Clicca sul ticker (link blu) → TradingView |
| **Aggiungere alla watchlist** | Seleziona riga → ➕ Aggiungi selezionati |
| **Analisi tecnica** | Dopo averli in watchlist, esegui lo scanner per segnali |
| **Cambiare scenario** | Usa il selettore in cima |

### 🎯 Criteri di selezione asset
- ✅ **Liquidità** > 1M$/giorno — trattabili senza slippage
- ✅ **Correlazione provata** con lo scenario (dati storici reali)
- ✅ **Strumenti regolamentati** NYSE/NASDAQ — niente prodotti esotici
- ✅ **Diversificazione**: ETF broad + singoli titoli per leva

### 📈 Performance storica in scenari di crisi
| Scenario | Asset vincente | Performance tipica |
|----------|---------------|-------------------|
| Guerra Ucraina Feb 2022 | LMT +36%, RTX +28%, XOM +40% | +30/50% in 3 mesi |
| COVID Crash Mar 2020 | TLT +20%, GLD +15%, XLV -5% | TLT unico rialzista |
| Inflazione 2021-2022 | XOM +80%, OXY +120%, WEAT +65% | Energia/agri dominano |
| Crisi bancaria Mar 2023 | GLD +8%, BTC +40%, TLT +6% | Oro e Bitcoin |
| 9/11 Settembre 2001 | GLD, LMT, RTX +15% in 6 mesi | Difesa e oro |

### ⚠️ Avvertenze
> I rendimenti passati non garantiscono quelli futuri. Questo è uno strumento informativo,
> non consulenza finanziaria. Alcuni ETF (RSX Russia) possono diventare illiquidi in caso di sanzioni.
""")

    # ── Export ────────────────────────────────────────────────────────
    st.markdown("---")
    _cx1, _cx2 = st.columns(2)
    _unique = list(dict.fromkeys(all_crisis_tickers))
    with _cx1:
        st.download_button("📺 Export TradingView CSV",
            data=chr(10).join(_unique),
            file_name=f"crisis_{selected_scenario[:25].replace(' ','_')}.csv",
            mime="text/plain", key="crisis_tv_exp",
            help="Un ticker per riga — importabile in TradingView Watchlist")
    with _cx2:
        _cdf = pd.DataFrame([
            {"Categoria":cat,"Ticker":t,"Nome":n,"Descrizione":d}
            for cat in active_categories
            for t,n,d in CRISIS_ASSETS.get(cat,{}).get("assets",[])
        ])
        if not _cdf.empty:
            st.download_button("📊 Export Excel",
                data=to_excel_bytes({"Crisis Monitor":_cdf}),
                file_name="crisis_monitor.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="crisis_xlsx_exp")

    # ── Strategy Chart ────────────────────────────────────────────────────
    try:
        from utils.backtest_tab import strategy_chart_widget as _scw
        _crisis_tkrs = [
            t for cat in active_categories
            for t,_n,_d in CRISIS_ASSETS.get(cat,{}).get("assets",[])
        ]
        st.markdown("---")
        _scw(tickers=_crisis_tkrs, key_suffix="CRISIS")
    except Exception:
        pass


# =========================================================================
# WATCHLIST — AgGrid + cards + multi-lista
# =========================================================================
with tab_w:
    st.markdown(f'<div class="section-pill">📋 WATCHLIST MANAGER — {st.session_state.current_list_name}</div>',
                unsafe_allow_html=True)

    # ── Sync status GitHub + Diagnostica DB ─────────────────────────────
    _wl_col1, _wl_col2 = st.columns([3, 2])
    with _wl_col1:
        if _GH_SYNC:
            _gs = _gh_status(DB_PATH)
            st.markdown(
                f'<div style="background:#1e222d;border-left:3px solid #26a69a;'
                f'padding:6px 12px;border-radius:0 4px 4px 0;font-size:0.82rem;">'
                f'☁️ <b style="color:#26a69a">GitHub Sync attivo</b> — '
                f'<code style="color:#b2b5be">{_gs.get("repo","")}/{_gs.get("path","")}</code>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#1e222d;border-left:3px solid #f59e0b;'
                'padding:6px 12px;border-radius:0 4px 4px 0;font-size:0.82rem;">'
                '⚠️ <b style="color:#f59e0b">GitHub Sync non configurato</b> — '
                'watchlist solo locale (si azzera ad ogni deploy)'
                '</div>',
                unsafe_allow_html=True
            )
    with _wl_col2:
        try:
            _wl_db_ok = DB_PATH.exists()
            _wl_db_sz = round(DB_PATH.stat().st_size/1024,1) if _wl_db_ok else 0
            st.caption(f"💾 `{DB_PATH.name}` — {_wl_db_sz} KB {'✅' if _wl_db_ok else '⚠️'}")
        except Exception as _e:
            st.caption(f"⚠️ DB: {_e}")
        if _GH_SYNC:
            if st.button("☁️ Sync ora", key="wl_sync_now",
                         help="Forza upload watchlist → GitHub"):
                _gh_push(DB_PATH)
                st.success("✅ Watchlist inviata a GitHub!")
    st.markdown("")

    df_wl_full=load_watchlist()

    # gestione lista "pending" (creata dalla sidebar ma non ancora nel DB)
    pending=st.session_state.pop("pending_new_list",None)
    all_lists=sorted(df_wl_full["list_name"].unique().tolist()) if not df_wl_full.empty else []
    if "DEFAULT" not in all_lists: all_lists.append("DEFAULT")
    if pending and pending not in all_lists: all_lists.append(pending); all_lists=sorted(all_lists)

    # ── Pannello gestione liste ──────────────────────────────────────────
    with st.expander("⚙️ Gestione Liste",expanded=True):
        gc1,gc2,gc3,gc4=st.columns(4)

        with gc1:
            st.markdown("**📂 Liste**")
            for ln in all_lists:
                cnt=len(df_wl_full[df_wl_full["list_name"]==ln]) if not df_wl_full.empty else 0
                active_m=" ✅" if ln==st.session_state.current_list_name else ""
                if st.button(f"{ln} ({cnt}){active_m}",key=f"sw_{ln}",use_container_width=True):
                    st.session_state.current_list_name=ln; st.rerun()

        with gc2:
            st.markdown("**✏️ Rinomina**")
            ren_src=st.selectbox("Da",all_lists,key="ren_src")
            ren_dst=st.text_input("Nuovo nome",key="ren_dst")
            if st.button("✏️ Rinomina",key="do_ren") and ren_dst.strip():
                gh_rename_watchlist(ren_src,ren_dst.strip())
                if st.session_state.current_list_name==ren_src:
                    st.session_state.current_list_name=ren_dst.strip()
                st.rerun()

        with gc3:
            st.markdown("**📋 Copia lista**")
            cp_src=st.selectbox("Da",all_lists,key="cp_src")
            cp_dst=st.text_input("A (nuova o esistente)",key="cp_dst")
            if st.button("📋 Copia",key="do_cp") and cp_dst.strip():
                df_src=df_wl_full[df_wl_full["list_name"]==cp_src]
                if not df_src.empty:
                    tc="Ticker" if "Ticker" in df_src.columns else "ticker"
                    nc="Nome"   if "Nome"   in df_src.columns else "name"
                    gh_add_to_watchlist(df_src[tc].tolist(),
                                     df_src[nc].tolist() if nc in df_src.columns else df_src[tc].tolist(),
                                     "Copia",f"da {cp_src}","LONG",cp_dst.strip())
                    st.success(f"✅ Copiati {len(df_src)} ticker."); st.rerun()

        with gc4:
            st.markdown("**🗑️ Elimina lista**")
            dl_sel=st.selectbox("Lista",all_lists,key="dl_sel")
            if st.button("🗑️ Elimina lista",key="do_dl",type="secondary"):
                conn=sqlite3.connect(str(DB_PATH))
                conn.execute("DELETE FROM watchlist WHERE list_name=?",(dl_sel,))
                conn.commit();conn.close()
                if st.session_state.current_list_name==dl_sel:
                    rem=[l for l in all_lists if l!=dl_sel]
                    st.session_state.current_list_name=rem[0] if rem else "DEFAULT"
                st.rerun()

    # ── Contenuto lista attiva ───────────────────────────────────────────
    df_wl=df_wl_full[df_wl_full["list_name"]==st.session_state.current_list_name].copy() \
          if not df_wl_full.empty else pd.DataFrame()

    st.markdown(f'<div class="section-pill">📌 {st.session_state.current_list_name} — {len(df_wl)} titoli</div>',
                unsafe_allow_html=True)

    if df_wl.empty:
        st.info("Lista vuota. Aggiungi ticker dagli altri tab oppure usa **Copia lista**.")
    else:
        tcol="Ticker" if "Ticker" in df_wl.columns else "ticker"
        ncol="Nome"   if "Nome"   in df_wl.columns else "name"

        # ── Vista: toggle cards / griglia ────────────────────────────────
        vmode_col1,vmode_col2,_=st.columns([1,1,4])
        with vmode_col1:
            if st.button("🃏 Cards",key="vm_cards",
                         type="primary" if st.session_state.wl_view_mode=="cards" else "secondary"):
                st.session_state.wl_view_mode="cards"; st.rerun()
        with vmode_col2:
            if st.button("📊 Griglia",key="vm_grid",
                         type="primary" if st.session_state.wl_view_mode=="grid" else "secondary"):
                st.session_state.wl_view_mode="grid"; st.rerun()

        # Merge colonne scanner
        extra_cols=["Prezzo","RSI","Vol_Ratio","Quality_Score","OBV_Trend","Weekly_Bull",
                    "Squeeze","Early_Score","Pro_Score","Ser_Score","Ser_OK","FV_Score","FV_OK"]
        df_wl_disp=df_wl.copy()
        for src_df in [df_ep,df_rea]:
            if not src_df.empty and "Ticker" in src_df.columns:
                for ec in extra_cols:
                    if ec in src_df.columns and ec not in df_wl_disp.columns:
                        mm=src_df[["Ticker",ec]].drop_duplicates("Ticker")
                        df_wl_disp=df_wl_disp.merge(mm,left_on=tcol,right_on="Ticker",
                                                      how="left",suffixes=("","_sc"))
                        if "Ticker_sc" in df_wl_disp.columns:
                            df_wl_disp.drop(columns=["Ticker_sc"],inplace=True)

        # ── Azioni massa ──────────────────────────────────────────────────
        wa1,wa2,wa3=st.columns(3)
        with wa1:
            csv_btn(df_wl_disp,f"watchlist_{st.session_state.current_list_name}.csv","exp_wl_dl")
        other_lists=[l for l in all_lists if l!=st.session_state.current_list_name] or ["DEFAULT"]
        with wa2:
            move_dest=st.selectbox("Sposta selezione →",other_lists,key="mass_mv")
        with wa3:
            copy_dest2=st.selectbox("Copia selezione →",other_lists,key="mass_cp")

        # ── VISTA GRIGLIA (AgGrid con note/trend editabili) ──────────────
        if st.session_state.wl_view_mode=="grid":
            # Prepara colonne per griglia watchlist
            wl_grid_cols=["id",tcol,ncol,"Prezzo","trend","note","origine","created_at",
                          "RSI","Vol_Ratio","Quality_Score","Ser_Score","FV_Score",
                          "Weekly_Bull","Squeeze","Early_Score","Pro_Score","OBV_Trend"]
            df_wg=df_wl_disp[[c for c in wl_grid_cols if c in df_wl_disp.columns]].copy()
            # Rinomina per display
            rename_map={}
            if tcol!="Ticker": rename_map[tcol]="Ticker"
            if ncol!="Nome":   rename_map[ncol]="Nome"
            if rename_map: df_wg=df_wg.rename(columns=rename_map)

            grid_resp_wl=build_aggrid(df_wg,"wl_grid",height=520,
                                       editable_cols=["trend","note"])
            sel_wl_rows=pd.DataFrame(grid_resp_wl["selected_rows"])
            updated_wl =pd.DataFrame(grid_resp_wl["data"])

            # Salva modifiche note/trend
            if not updated_wl.empty and "id" in updated_wl.columns:
                if st.button("💾 Salva Note/Trend",key="save_wl_edits"):
                    conn=sqlite3.connect(str(DB_PATH))
                    for _,r in updated_wl.iterrows():
                        rid=int(r.get("id",0))
                        if rid>0:
                            conn.execute("UPDATE watchlist SET note=?,trend=? WHERE id=?",
                                         (str(r.get("note","")),str(r.get("trend","")),rid))
                    conn.commit();conn.close()
                    st.success("✅ Salvato!"); st.rerun()

            selected_ids=[int(r.get("id",0)) for _,r in sel_wl_rows.iterrows() if r.get("id")]

            if selected_ids:
                ac1,ac2,ac3=st.columns(3)
                with ac1:
                    if st.button(f"➡️ Sposta in '{move_dest}'",key="do_mv_g"):
                        gh_move_watchlist_rows(selected_ids,move_dest); st.rerun()
                with ac2:
                    if st.button(f"📋 Copia in '{copy_dest2}'",key="do_cp_g"):
                        rows_s=df_wl_disp[df_wl_disp["id"].isin(selected_ids)]
                        gh_add_to_watchlist(rows_s[tcol].tolist(),
                            rows_s[ncol].tolist() if ncol in rows_s.columns else rows_s[tcol].tolist(),
                            "Copia","da selezione","LONG",copy_dest2)
                        st.success("✅ Copiati."); st.rerun()
                with ac3:
                    if st.button("🗑️ Elimina sel.",key="do_dl_g",type="secondary"):
                        gh_delete_from_watchlist(selected_ids); st.rerun()

        # ── VISTA CARDS ───────────────────────────────────────────────────
        else:
            selected_ids=[]
            for _,wrow in df_wl_disp.iterrows():
                rid    =wrow.get("id","")
                tkr    =wrow.get(tcol,"")
                nom    =wrow.get(ncol,"")
                rsi_v  =wrow.get("RSI",None)
                vr_v   =wrow.get("Vol_Ratio",None)
                qs_v   =wrow.get("Quality_Score",None)
                sq_v   =wrow.get("Squeeze",False)
                wb_v   =wrow.get("Weekly_Bull",None)
                ser_v  =wrow.get("Ser_Score",None)
                fv_v   =wrow.get("FV_Score",None)
                origine=wrow.get("origine","")
                created=wrow.get("created_at","")
                trend_v=wrow.get("trend","")

                def badge(val,cls,txt): return f'<span class="wl-card-badge {cls}">{txt}</span>' if val else ""

                # RSI badge
                if rsi_v is not None and not (isinstance(rsi_v,float) and np.isnan(rsi_v)):
                    rn=float(rsi_v); rc="badge-blue" if rn<40 else "badge-green" if rn<=65 else "badge-orange" if rn<=70 else "badge-red"
                    rsi_b=f'<span class="wl-card-badge {rc}">RSI {rn:.1f}</span>'
                else: rsi_b=""
                # Vol badge
                if vr_v is not None and not (isinstance(vr_v,float) and np.isnan(vr_v)):
                    vn=float(vr_v); vc="badge-gray" if vn<1 else "badge-green" if vn<2 else "badge-orange" if vn<3 else "badge-red"
                    vr_b=f'<span class="wl-card-badge {vc}">Vol {vn:.1f}x</span>'
                else: vr_b=""
                # Quality badge
                if qs_v is not None and not (isinstance(qs_v,float) and np.isnan(qs_v)):
                    qn=int(float(qs_v)); qc="badge-green" if qn>=9 else "badge-orange" if qn>=6 else "badge-gray"
                    qs_b=f'<span class="wl-card-badge {qc}">Q {qn}/12</span>'
                else: qs_b=""
                # Serafini badge
                if ser_v is not None and not (isinstance(ser_v,float) and np.isnan(ser_v)):
                    sn=int(float(ser_v)); sc="badge-green" if sn==6 else "badge-orange" if sn>=4 else "badge-gray"
                    ser_b=f'<span class="wl-card-badge {sc}">🎯 S{sn}/6</span>'
                else: ser_b=""
                # Finviz badge
                if fv_v is not None and not (isinstance(fv_v,float) and np.isnan(fv_v)):
                    fn=int(float(fv_v)); fc="badge-green" if fn>=7 else "badge-orange" if fn>=5 else "badge-gray"
                    fv_b=f'<span class="wl-card-badge {fc}">📊 FV{fn}/8</span>'
                else: fv_b=""

                sq_b=badge(sq_v is True or str(sq_v).lower()=="true","badge-orange","🔥 SQ")
                wb_b=('<span class="wl-card-badge badge-green">📈 W+</span>' if wb_v is True or str(wb_v).lower()=="true" else
                      '<span class="wl-card-badge badge-red">📉 W—</span>'   if wb_v is False or str(wb_v).lower()=="false" else "")
                trend_cls={"LONG":"badge-green","SHORT":"badge-red","WATCH":"badge-orange"}.get(str(trend_v).upper(),"badge-gray")
                trend_b=f'<span class="wl-card-badge {trend_cls}">{trend_v}</span>' if trend_v and str(trend_v).upper() not in ("","NAN","NONE") else ""

                row_c=st.columns([0.3,3,1])
                with row_c[0]:
                    if st.checkbox("",key=f"chk_{rid}",label_visibility="collapsed"): selected_ids.append(rid)
                with row_c[1]:
                    st.markdown(f"""<div class="wl-card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div><span class="wl-card-ticker">{tkr}</span>
    <span class="wl-card-name"> &nbsp;{nom}</span></div>
    <div style="color:#374151;font-size:0.72rem">{origine} · {str(created)[:10]}</div>
  </div>
  <div style="margin-top:8px">{trend_b}{rsi_b}{vr_b}{qs_b}{ser_b}{fv_b}{sq_b}{wb_b}</div>
</div>""",unsafe_allow_html=True)
                with row_c[2]:
                    st.write("")
                    if st.button("🗑️",key=f"del_{rid}",help=f"Elimina {tkr}"):
                        gh_delete_from_watchlist([rid]); st.rerun()

            if selected_ids:
                ac1,ac2,ac3=st.columns(3)
                with ac1:
                    if st.button(f"➡️ Sposta in '{move_dest}'",key="do_mv_c"):
                        gh_move_watchlist_rows(selected_ids,move_dest); st.rerun()
                with ac2:
                    if st.button(f"📋 Copia in '{copy_dest2}'",key="do_cp_c"):
                        rows_s=df_wl_disp[df_wl_disp["id"].isin(selected_ids)]
                        gh_add_to_watchlist(rows_s[tcol].tolist(),
                            rows_s[ncol].tolist() if ncol in rows_s.columns else rows_s[tcol].tolist(),
                            "Copia","da selezione","LONG",copy_dest2)
                        st.success("✅ Copiati."); st.rerun()
                with ac3:
                    if st.button("🗑️ Elimina sel.",key="do_dl_c",type="secondary"):
                        gh_delete_from_watchlist(selected_ids); st.rerun()

        # ── Grafici ticker selezionato ────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-pill">📊 ANALISI TICKER</div>',unsafe_allow_html=True)
        if not df_wl.empty and tcol in df_wl.columns:
            _wl_df=df_wl[[tcol,ncol]].drop_duplicates(tcol).sort_values(ncol)
            _wl_labels=[f"{r[tcol]}  —  {r[ncol]}" for _,r in _wl_df.iterrows()]
            _wl_tickers=_wl_df[tcol].tolist()
        else:
            _wl_labels=[]; _wl_tickers=[]
        if _wl_tickers:
            _sel_idx=st.selectbox("🔍 Seleziona ticker",
                options=range(len(_wl_labels)),format_func=lambda i:_wl_labels[i],key="wl_tkr_sel")
            sel_wl=_wl_tickers[_sel_idx]
            row_wl=None
            for src in [df_ep,df_rea]:
                if src.empty or "Ticker" not in src.columns: continue
                m=src[src["Ticker"]==sel_wl]
                if not m.empty: row_wl=m.iloc[0]; break
            if row_wl is not None: show_charts(row_wl,key_suffix="wl")
            else: st.info(f"📭 Dati non disponibili per **{sel_wl}**. Esegui lo scanner.")

    # ── Info DB path + Backup/Restore ──────────────────────────────────────
    with st.expander("💾 Backup & Restore Watchlist", expanded=False):
        try:
            from utils.db import DB_PATH as _DBPATH
            st.caption(f"📂 DB attivo: `{_DBPATH}`")
            _db_ok = _DBPATH.exists()
            _db_sz = round(_DBPATH.stat().st_size/1024,1) if _db_ok else 0
            st.caption(f"{'✅' if _db_ok else '❌'} File {'presente' if _db_ok else 'non trovato'} — {_db_sz} KB")
        except Exception as _e:
            st.caption(f"⚠️ DB path non disponibile: {_e}")

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**📤 Esporta**")
            if st.button("📥 Scarica backup JSON", key="wl_export"):
                try:
                    _df_exp = load_watchlist()
                    if not _df_exp.empty:
                        import json as _json
                        _exp = _df_exp.to_dict(orient="records")
                        st.download_button(
                            "💾 Salva watchlist.json",
                            data=_json.dumps(_exp, indent=2, default=str),
                            file_name="watchlist_backup.json",
                            mime="application/json",
                            key="wl_dl"
                        )
                    else:
                        st.warning("Watchlist vuota.")
                except Exception as _e:
                    st.error(f"Errore export: {_e}")

        with bc2:
            st.markdown("**📥 Importa**")
            _up = st.file_uploader("Carica watchlist.json", type="json", key="wl_import")
            if _up and st.button("⬆️ Ripristina dal backup", key="wl_restore"):
                try:
                    import json as _json
                    _rows = _json.loads(_up.read().decode())
                    conn = sqlite3.connect(str(DB_PATH))
                    for _r in _rows:
                        _ticker  = str(_r.get("ticker",""))
                        _lname   = str(_r.get("list_name","DEFAULT"))
                        if not _ticker: continue
                        _exists = conn.execute(
                            "SELECT id FROM watchlist WHERE ticker=? AND list_name=?",
                            (_ticker, _lname)
                        ).fetchone()
                        if not _exists:
                            conn.execute(
                                "INSERT INTO watchlist (ticker,name,trend,origine,note,list_name,created_at) "
                                "VALUES (?,?,?,?,?,?,?)",
                                (_ticker, _r.get("name",""), _r.get("trend",""),
                                 _r.get("origine",""), _r.get("note",""), _lname,
                                 _r.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            )
                    conn.commit(); conn.close()
                    st.success(f"✅ Ripristinati {len(_rows)} ticker. Clicca Refresh.")
                except Exception as _e:
                    st.error(f"Errore import: {_e}")

    # Export TradingView — solo ticker, un per riga
    _df_tv = load_watchlist()
    _tv_cur = _df_tv[_df_tv["list_name"]==st.session_state.current_list_name] if not _df_tv.empty else pd.DataFrame()
    if not _tv_cur.empty:
        _tc = "ticker" if "ticker" in _tv_cur.columns else "Ticker"
        _tv_lines = _tv_cur[_tc].dropna().unique().tolist()
        st.download_button(
            label="📺 Export TradingView CSV",
            data=chr(10).join(_tv_lines),
            file_name=f"watchlist_{st.session_state.current_list_name}_tradingview.csv",
            mime="text/plain",
            key="wl_tv_export",
            help="Un ticker per riga — importabile direttamente in TradingView Watchlist"
        )
    if st.button("🔄 Refresh",key="wl_ref"): st.rerun()

    # ── Strategy Chart ────────────────────────────────────────────────────
    try:
        from utils.backtest_tab import strategy_chart_widget as _scw
        df_wl_sc = load_watchlist(list_name=st.session_state.get("current_list_name","DEFAULT"))
        _wl_tkrs = df_wl_sc["ticker"].dropna().tolist() if not df_wl_sc.empty and "ticker" in df_wl_sc.columns else []
        st.markdown("---")
        _scw(tickers=_wl_tkrs, key_suffix="WL")
    except Exception:
        pass

# =========================================================================
# STORICO
# =========================================================================
with tab_bt:
    render_backtest_tab()

with tab_of:
    try:
        if _of_render:
            # Passa df_ep dallo scanner se disponibile
            _df_of = df_ep if "df_ep" in dir() else None
            _of_render(df_scanner=_df_of)
        else:
            from utils.orderflow_tab import render_orderflow_tab
            render_orderflow_tab()
    except Exception as _ofe:
        import traceback
        st.error(f"Order Flow error: {_ofe}")
        st.code(traceback.format_exc())


with tab_bcd:
    try:
        from utils.bluechip_dip import render_bluechip_dip
        render_bluechip_dip()
    except ImportError:
        st.info("💎 bluechip_dip.py non trovato in utils/")
    except Exception as _bce:
        import traceback
        st.error(f"Blue Chip Dip error: {_bce}")
        st.code(traceback.format_exc())

    st.markdown('<div class="section-pill">📜 STORICO SCANSIONI</div>',unsafe_allow_html=True)

    # ── Legenda Storico ────────────────────────────────────────────────
    with st.expander("📖 Come leggere lo Storico — Guida completa", expanded=False):
        st.markdown("""
## 📜 Storico Scansioni — Guida Operativa

Lo **Storico** registra ogni scansione eseguita nel database locale.
Ogni riga corrisponde a una singola esecuzione dello scanner con timestamp, mercati scansionati
e numero di segnali trovati.

---

### 📊 Colonne della tabella

| Colonna | Tipo | Significato |
|---------|------|-------------|
| **id** | numero | ID progressivo scansione |
| **scanned_at** | datetime | Data e ora esecuzione (UTC) |
| **markets** | testo | Mercati inclusi (US, ETF, Crypto…) |
| **n_tickers** | intero | Titoli totali analizzati nello scan |
| **n_early** | intero | Titoli con `Stato_Early = EARLY` trovati |
| **n_pro** | intero | Titoli con `Stato_Pro = PRO` trovati |
| **n_rea** | intero | Titoli con `Stato = HOT` (REA) trovati |
| **elapsed_s** | secondi | Tempo impiegato per la scansione |
| **params** | JSON | Parametri usati (soglie, top, indicatori) |

---

### 🔍 Confronto Snapshot — Come funziona

Il **confronto** permette di analizzare l'evoluzione del mercato tra due momenti diversi:

- **🆕 Nuovi in B**: ticker apparsi in B ma non in A → **nuovi segnali emergenti**
- **❌ Usciti da A**: ticker che erano in A ma non in B → **segnali deteriorati o usciti**
- **✅ Persistenti**: ticker presenti in entrambe le scan → **segnali solidi e confermati**

**Caso d'uso tipico:**
1. Scan mattina 09:00 → salva come A
2. Scan pomeriggio 15:30 → salva come B
3. Confronta → vedi quali nuovi titoli sono entrati in segnale nel corso della giornata

**Interpretazione:**
- Molti *Nuovi* con pochi *Persistenti* → mercato in rotazione, cautela
- Pochi *Nuovi* con molti *Persistenti* → trend solido, conferma
- Tutti *Usciti* → deterioramento rapido, possibile fine trend

---

### 💡 Consigli operativi

- **Frequenza ideale**: 1-3 scan al giorno (apertura, metà seduta, chiusura)
- **Reset storico**: usa il pulsante 🗑️ solo se vuoi cancellare tutto. I dati del DB watchlist
  rimangono intatti — viene cancellato solo lo storico delle scansioni.
- **Backup**: esporta i dati importanti dalla Watchlist prima di fare reset
- **Limite**: vengono mostrate le ultime **20 scansioni**. Le più vecchie rimangono nel DB
  ma non vengono visualizzate (modifica `load_scan_history(20)` per aumentare).
""")

    _,col_rst=st.columns([4,1])
    with col_rst:
        if st.button("🗑️ Reset",key="rst_hist",type="secondary"):
            conn=sqlite3.connect(str(DB_PATH)); conn.execute("DELETE FROM scan_history")
            conn.commit();conn.close(); st.success("Storico cancellato!"); st.rerun()
    df_hist=load_scan_history(20)
    if df_hist.empty:
        st.info("""
📭 **Nessuna scansione salvata ancora.**

Per popolare lo storico:
1. Vai nella sidebar
2. Seleziona i mercati da scansionare
3. Clicca **▶️ Avvia Scanner**

Ogni scansione viene automaticamente salvata qui con timestamp, mercati, segnali trovati e tempi.
""")
    else:
        # Formatta colonne
        disp_hist = df_hist.copy()
        if "elapsed_s" in disp_hist.columns:
            disp_hist["elapsed_s"] = disp_hist["elapsed_s"].apply(
                lambda x: f"{x:.0f}s" if pd.notna(x) else "—")

        # Metriche aggregate
        _m1,_m2,_m3,_m4 = st.columns(4)
        _m1.metric("📋 Scan totali", len(df_hist))
        if "n_early" in df_hist.columns:
            _m2.metric("📡 Max EARLY", int(df_hist["n_early"].max()))
        if "n_pro" in df_hist.columns:
            _m3.metric("💪 Max PRO", int(df_hist["n_pro"].max()))
        if "n_tickers" in df_hist.columns:
            _m4.metric("🔭 Titoli medi", f"{df_hist['n_tickers'].mean():.0f}")

        st.markdown("**📋 Ultime 20 scansioni:**")
        st.dataframe(disp_hist,use_container_width=True)
        st.markdown("---")
        st.subheader("🔍 Confronto Snapshot")
        st.caption("Seleziona due scansioni per confrontare quali ticker sono entrati/usciti dai segnali.")
        hc1,hc2=st.columns(2)
        def _slbl(row):
            dt=str(row.get("scanned_at",""))[:16]
            ep=int(row.get("n_early",0)); pr=int(row.get("n_pro",0))
            mkt=str(row.get("markets",""))[:20]
            return f"{dt}  |  E:{ep} P:{pr}  [{mkt}]"
        _smap={row["id"]:_slbl(row) for _,row in df_hist.iterrows()}
        _ids=list(_smap.keys())
        with hc1:
            id_a=st.selectbox("📅 Scansione A (baseline)",_ids,format_func=lambda i:_smap[i],key="sn_a")
        with hc2:
            id_b=st.selectbox("📅 Scansione B (più recente)",_ids,format_func=lambda i:_smap[i],
                index=min(1,len(_ids)-1),key="sn_b")
        if st.button("🔍 Confronta le due scansioni", use_container_width=False):
            ea,_=load_scan_snapshot(id_a); eb,_=load_scan_snapshot(id_b)
            if ea.empty or eb.empty: st.warning("Dati non disponibili per uno dei due snapshot.")
            else:
                ta=set(ea.get("Ticker",pd.Series()).tolist())
                tb=set(eb.get("Ticker",pd.Series()).tolist())
                sc1,sc2,sc3,sc4=st.columns(4)
                sc1.metric("🆕 Nuovi in B",len(tb-ta),help="Ticker apparsi in B ma non in A")
                sc2.metric("❌ Usciti da A",len(ta-tb),help="Ticker che erano in A ma non in B")
                sc3.metric("✅ Persistenti",len(ta&tb),help="Presenti in entrambe le scan")
                sc4.metric("📊 Overlap %",f"{len(ta&tb)/max(len(ta|tb),1)*100:.0f}%")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    if tb-ta:
                        st.markdown("**🆕 Nuovi ticker in B:**")
                        st.code("  ".join(sorted(tb-ta)))
                    if ta-tb:
                        st.markdown("**❌ Ticker usciti da A:**")
                        st.code("  ".join(sorted(ta-tb)))
                with col_r2:
                    if ta&tb:
                        st.markdown(f"**✅ Ticker persistenti ({len(ta&tb)}):**")
                        st.code("  ".join(sorted(ta&tb)))


# =========================================================================
# EXPORT GLOBALI
# =========================================================================
st.markdown("---")
st.markdown('<div class="section-pill">💾 EXPORT GLOBALI</div>',unsafe_allow_html=True)
df_conf_exp=pd.DataFrame()
if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
    df_conf_exp=df_ep[(df_ep["Stato_Early"]=="EARLY")&(df_ep["Stato_Pro"]=="PRO")].copy()
df_wl_exp=load_watchlist()
df_wl_exp=df_wl_exp[df_wl_exp["list_name"]==st.session_state.current_list_name]
all_exp={"EARLY":df_ep,"PRO":df_ep,"REA-HOT":df_rea,"CONFLUENCE":df_conf_exp,"Watchlist":df_wl_exp}
cur_tab=st.session_state.get("last_active_tab","EARLY")
df_cur=all_exp.get(cur_tab,pd.DataFrame())

ec1,ec2,ec3,ec4=st.columns(4)
with ec1:
    st.download_button("📊 XLSX Tutti",to_excel_bytes(all_exp),
        "TradingScanner_v29_Tutti.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="xlsx_all")
with ec2:
    tv_rows=[]
    for n,df_t in all_exp.items():
        if isinstance(df_t,pd.DataFrame) and not df_t.empty and "Ticker" in df_t.columns:
            tks=df_t["Ticker"].tolist()
            tv_rows.append(pd.DataFrame({"Tab":[n]*len(tks),"Ticker":tks}))
    if tv_rows:
        df_tv=pd.concat(tv_rows,ignore_index=True).drop_duplicates("Ticker")
        st.download_button("📈 CSV TV Tutti",df_tv.to_csv(index=False).encode(),
            "TradingScanner_v29_TV.csv","text/csv",key="csv_tv_all")
with ec3:
    st.download_button(f"📊 XLSX {cur_tab}",to_excel_bytes({cur_tab:df_cur}),
        f"TradingScanner_v29_{cur_tab}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="xlsx_curr")
with ec4:
    if not df_cur.empty and "Ticker" in df_cur.columns:
        st.download_button(f"📈 CSV TV {cur_tab}",make_tv_csv(df_cur,cur_tab),
            f"TradingScanner_v29_{cur_tab}_TV.csv","text/csv",key="csv_tv_curr")
