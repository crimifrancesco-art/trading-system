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

from utils.db import (
    init_db, reset_watchlist_db, add_to_watchlist, load_watchlist,
    DB_PATH, save_scan_history, load_scan_history, load_scan_snapshot,
    delete_from_watchlist, move_watchlist_rows, rename_watchlist,
    update_watchlist_note,
    save_signals, cache_stats, cache_clear,
)
from utils.scanner import load_universe, scan_universe
from utils.backtest_tab import render_backtest_tab

# =========================================================================
# CSS
# =========================================================================
DARK_CSS = """
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="block-container"]{
    background-color:#0a0e1a !important; color:#c9d1d9 !important;}
[data-testid="stSidebar"]{background-color:#0d1117 !important;border-right:1px solid #1f2937 !important;}
[data-testid="stSidebar"] *{color:#c9d1d9 !important;}
h1{color:#00ff88 !important;font-family:'Courier New',monospace !important;
   letter-spacing:2px;text-shadow:0 0 20px #00ff8855;}
h2,h3{color:#58a6ff !important;font-family:'Courier New',monospace !important;}
.stCaption,small{color:#6b7280 !important;}
[data-testid="stTabs"] button{background:#0d1117 !important;color:#8b949e !important;
    border-bottom:2px solid transparent !important;
    font-family:'Courier New',monospace !important;font-size:0.82rem !important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#00ff88 !important;border-bottom:2px solid #00ff88 !important;}
[data-testid="stMetric"]{background:#0d1117 !important;border:1px solid #1f2937 !important;
    border-radius:8px !important;padding:12px 16px !important;}
[data-testid="stMetricLabel"]{color:#6b7280 !important;font-size:0.75rem !important;}
[data-testid="stMetricValue"]{color:#00ff88 !important;font-size:1.6rem !important;
    font-family:'Courier New',monospace !important;}
[data-testid="stButton"]>button{background:linear-gradient(135deg,#0d1117,#1a2233) !important;
    color:#00ff88 !important;border:1px solid #00ff8855 !important;
    border-radius:6px !important;font-family:'Courier New',monospace !important;transition:all 0.2s;}
[data-testid="stButton"]>button:hover{border-color:#00ff88 !important;color:#ffffff !important;}
[data-testid="stButton"]>button[kind="primary"]{background:linear-gradient(135deg,#00401f,#006633) !important;
    border-color:#00ff88 !important;color:#00ff88 !important;font-size:1rem !important;}
[data-testid="stButton"]>button[kind="secondary"]{background:linear-gradient(135deg,#1a0a0a,#2d1010) !important;
    color:#ef4444 !important;border:1px solid #ef444455 !important;}
[data-testid="stDownloadButton"]>button{background:#0d1117 !important;color:#58a6ff !important;
    border:1px solid #1f3a5f !important;border-radius:6px !important;}
[data-testid="stExpander"]{background:#0d1117 !important;border:1px solid #1f2937 !important;border-radius:8px !important;}
[data-testid="stExpander"] summary{color:#58a6ff !important;}
hr{border-color:#1f2937 !important;}
.ag-root-wrapper{background:#0d1117 !important;border:1px solid #1f2937 !important;}
.ag-header{background:#0a0e1a !important;border-bottom:1px solid #1f2937 !important;}
.ag-header-cell-label{color:#58a6ff !important;font-family:'Courier New',monospace !important;
    font-size:0.78rem !important;letter-spacing:1px;}
.ag-header-cell-resize{background:#374151 !important;}
.ag-row{background:#0d1117 !important;border-bottom:1px solid #1a2233 !important;}
.ag-row:hover{background:#131d2e !important;}
.ag-row-selected{background:#0d2d1e !important;}
.ag-cell{color:#c9d1d9 !important;font-family:'Courier New',monospace !important;font-size:0.82rem !important;}
.ag-paging-panel{background:#0a0e1a !important;color:#6b7280 !important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0a0e1a;}
::-webkit-scrollbar-thumb{background:#1f2937;border-radius:3px;}
.section-pill{display:inline-block;background:linear-gradient(90deg,#003320,#001a10);
    border:1px solid #00ff8844;border-radius:20px;padding:4px 16px;
    font-family:'Courier New',monospace;font-size:0.8rem;color:#00ff88;
    letter-spacing:2px;margin-bottom:12px;}
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
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Courier New"),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
)
# =========================================================================
# FORMATTING HELPERS  (inline — non richiedono utils.formatting)
# =========================================================================
def _fmt_large(v):
    """Abbrevia numeri grandi: 1234567 → '1.2M', 12345678901 → '12.3B'"""
    try:
        v = float(v)
        if v >= 1e12: return f"{v/1e12:.1f}T"
        if v >= 1e9:  return f"{v/1e9:.1f}B"
        if v >= 1e6:  return f"{v/1e6:.1f}M"
        if v >= 1e3:  return f"{v/1e3:.0f}K"
        return str(int(v))
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
            lambda x: _fmt_large(x) if pd.notna(x) else "—")
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
def build_full_chart(row: pd.Series, indicators: list) -> go.Figure:
    cd=row.get("_chart_data")
    if not cd or not isinstance(cd,dict): return None
    dates=cd.get("dates",[]); opens=cd.get("open",[])
    highs=cd.get("high",[]); lows=cd.get("low",[])
    closes=cd.get("close",[]); vols=cd.get("volume",[])
    ema20=cd.get("ema20",[]); ema50=cd.get("ema50",[])
    bb_up=cd.get("bb_up",[]); bb_dn=cd.get("bb_dn",[])
    if not dates or not closes: return None

    show_sma=("SMA 9 & 21 + RSI" in indicators)
    show_macd=("MACD" in indicators)
    show_sar=("Parabolic SAR" in indicators)

    cur=2; row_rsi=None; row_macd=None
    if show_sma:  row_rsi=cur;  cur+=1
    if show_macd: row_macd=cur; cur+=1
    row_vol=cur; n_rows=cur

    ht={2:[0.65,0.15],3:[0.52,0.18,0.13],4:[0.44,0.17,0.17,0.13]}
    heights=ht.get(n_rows,[0.38,0.15,0.15,0.17,0.13])[:n_rows]
    s=sum(heights); heights=[h/s for h in heights]

    fig=make_subplots(rows=n_rows,cols=1,shared_xaxes=True,
                      row_heights=heights,vertical_spacing=0.025)
    fig.add_trace(go.Candlestick(x=dates,open=opens,high=highs,low=lows,close=closes,
        increasing_line_color="#22c55e",increasing_fillcolor="rgba(34,197,94,0.33)",
        decreasing_line_color="#ef4444",decreasing_fillcolor="rgba(239,68,68,0.33)",
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

    if vols:
        fig.add_trace(go.Bar(x=dates,y=vols,
            marker_color=["rgba(0,255,136,0.4)" if c>=o else "rgba(239,68,68,0.4)" for c,o in zip(closes,opens)],
            name="Volume",showlegend=False),row=row_vol,col=1)
        fig.update_yaxes(title_text="Vol",tickfont=dict(size=8),row=row_vol,col=1)

    tkr=row.get("Ticker",""); sq="  🔥" if row.get("Squeeze") else ""
    fig.update_layout(**PLOTLY_DARK,
        title=dict(text=f"<b>{tkr}</b> — {row.get('Nome','')}  |  {row.get('Prezzo','')}  |  RSI {row.get('RSI','')}{sq}",
            font=dict(color="#00ff88",size=13)),
        height=160+180*n_rows,xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.01,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
        margin=dict(l=0,r=0,t=55,b=0),hovermode="x unified")
    for r in range(1,n_rows+1):
        fig.update_xaxes(gridcolor="#1f2937",row=r,col=1)
        fig.update_yaxes(gridcolor="#1f2937",row=r,col=1)
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
    ind_opts=["SMA 9 & 21 + RSI","MACD","Parabolic SAR"]
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

# =========================================================================
# JS RENDERERS
# =========================================================================
name_dblclick_renderer=JsCode("""class N{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value||'';const t=p.data.Ticker||p.data.ticker;if(!t)return;
this.eGui.style.cursor='pointer';this.eGui.title='Doppio click → TradingView';
this.eGui.ondblclick=()=>window.open("https://www.tradingview.com/chart/?symbol="+String(t).split(".")[0],"_blank");}
getGui(){return this.eGui;}}""")

rsi_renderer=JsCode("""class R{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'-':v.toFixed(1);
this.eGui.style.fontWeight='bold';this.eGui.style.fontFamily='Courier New';
if(v<30)this.eGui.style.color='#60a5fa';
else if(v<40)this.eGui.style.color='#93c5fd';
else if(v<=65)this.eGui.style.color='#00ff88';
else if(v<=70)this.eGui.style.color='#f59e0b';
else this.eGui.style.color='#ef4444';}getGui(){return this.eGui;}}""")

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
let txt='-';let color='#6b7280';
if(!isNaN(v)){
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
st.set_page_config(page_title="Trading Scanner PRO 27.0",layout="wide",page_icon="🧠")
st.markdown(DARK_CSS,unsafe_allow_html=True)
st.markdown("# 🧠 Trading Scanner PRO 28.0")
st.markdown('<div class="section-pill">CACHE · BACKTEST · FINVIZ · MULTI-WATCHLIST · v28.0</div>',unsafe_allow_html=True)
init_db()

# =========================================================================
# SESSION STATE
# =========================================================================
defaults=dict(
    mSP500=True,mNasdaq=True,mFTSE=True,mEurostoxx=False,
    mDow=False,mRussell=False,mStoxxEmerging=False,mUSSmallCap=False,
    eh=0.02,prmin=40,prmax=70,rpoc=0.02,vol_ratio_hot=1.5,top=15,
    min_early_score=2.0,min_quality=3,min_pro_score=2.0,
    current_list_name="DEFAULT",last_active_tab="EARLY",
    active_indicators=["SMA 9 & 21 + RSI","MACD","Parabolic SAR"],
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
    ind_opts_all=["SMA 9 & 21 + RSI","MACD","Parabolic SAR"]
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
st.sidebar.subheader("⚡ Scanner v28")
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
    if st.button("🚀 AVVIA SCANNER PRO 28.0",type="primary",use_container_width=True):
        universe=load_universe(sel)
        if not universe: st.warning("Seleziona almeno un mercato!")
        else:
            pb=st.progress(0); status=st.empty()
            use_cache   = st.session_state.get("use_cache",True)
            use_finviz  = st.session_state.get("use_finviz",False)
            n_wk        = st.session_state.get("n_workers",8)

            def _progress(done, total, tkr):
                pb.progress(done/total)
                hit = "⚡ cache" if use_cache else ""
                status.text(f"Analisi {done}/{total}: {tkr}  {hit}")

            df_ep_new, df_rea_new, scan_stats = scan_universe(
                universe, eh, prmin, prmax, rpoc, vol_ratio_hot,
                cache_enabled=use_cache, finviz_enabled=use_finviz,
                n_workers=n_wk, progress_callback=_progress
            )
            st.session_state.df_ep     = df_ep_new
            st.session_state.df_rea    = df_rea_new
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
            st.session_state.scan_stats= scan_stats

            scan_id = save_scan_history(
                sel, df_ep_new, df_rea_new,
                elapsed_s   = scan_stats["elapsed_s"],
                cache_hits  = scan_stats["cache_hits"],
            )
            save_signals(scan_id, df_ep_new, df_rea_new, sel)

            n_h=len(df_rea_new); n_c=0
            if not df_ep_new.empty and "Stato_Early" in df_ep_new.columns:
                n_c=int(((df_ep_new["Stato_Early"]=="EARLY")&(df_ep_new["Stato_Pro"]=="PRO")).sum())
            if n_h>=5: st.toast(f"🔥 {n_h} HOT!",icon="🔥")
            if n_c>=3: st.toast(f"⭐ {n_c} CONFLUENCE!",icon="⭐")
            elapsed = scan_stats["elapsed_s"]
            hits    = scan_stats["cache_hits"]
            dl      = scan_stats["downloaded"]
            st.toast(f"⏱️ {elapsed}s  |  ⚡ {hits} cache  |  ☁️ {dl} scaricati",icon="✅")
            st.rerun()

df_ep =st.session_state.get("df_ep", pd.DataFrame())
df_rea=st.session_state.get("df_rea",pd.DataFrame())
if "last_scan" in st.session_state:
    st.caption(f"⏱️ Ultima scansione: {st.session_state.last_scan}")
render_kpi_bar(df_ep,df_rea)
st.markdown("---")

# =========================================================================
# AGGRID BUILDER  — resize + sort + filter
# =========================================================================
def build_aggrid(df_disp: pd.DataFrame, grid_key: str, height: int=480,
                 editable_cols: list=None):
    gb=GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(sortable=True,resizable=True,filterable=True,
                                 editable=False,wrapText=False,suppressSizeToFit=False)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple",use_checkbox=True)

    if editable_cols:
        for ec in editable_cols:
            if ec in df_disp.columns:
                gb.configure_column(ec,editable=True)

    col_w={"Ticker":80,"Nome":160,"Prezzo":85,"Prezzo_fmt":85,"MarketCap":110,"MarketCap_fmt":110,
           "Early_Score":95,"Pro_Score":80,"Quality_Score":130,"Ser_Score":90,"FV_Score":90,
           "RSI":65,"Vol_Ratio":85,"Squeeze":70,"RSI_Div":85,
           "Weekly_Bull":80,"Stato_Early":85,"Stato_Pro":80,
           "Vol_Today":85,"Vol_7d_Avg":85,"Avg_Vol_20":85,
           "trend":100,"note":200,"origine":90,"created_at":100,
           "EPS_NY_Gr":90,"EPS_5Y_Gr":90,"PE":70,"Fwd_PE":75,
           "Earnings_Soon":90,"Optionable":85,"OBV_Trend":80,
           "EMA20":80,"EMA50":80,"EMA200":85,"ATR":70,"Rel_Vol":75}
    for c,w in col_w.items():
        if c in df_disp.columns: gb.configure_column(c,width=w)
    # Nascondi colonne interne che non devono apparire in griglia
    hide_cols=["id","_chart_data","_quality_components","_ser_criteri","_fv_criteri",
               "Ser_OK","FV_OK","ATR_Exp","Stato"]
    for c in hide_cols:
        if c in df_disp.columns: gb.configure_column(c,hide=True)

    rmap={"Nome":name_dblclick_renderer,"RSI":rsi_renderer,
          "Vol_Ratio":vol_ratio_renderer,"Quality_Score":quality_renderer,
          "Ser_Score":ser_score_renderer,"FV_Score":fv_score_renderer,
          "Squeeze":squeeze_renderer,"RSI_Div":rsi_div_renderer,
          "Weekly_Bull":weekly_renderer,"Prezzo_fmt":price_renderer,"Prezzo":price_renderer,
          "trend":trend_renderer,
          "Vol_Today":vol_abbrev_renderer,"Vol_7d_Avg":vol_abbrev_renderer,"Avg_Vol_20":vol_abbrev_renderer,
          "MarketCap":mcap_renderer,"MarketCap_fmt":mcap_renderer,
          "EPS_NY_Gr":pct_renderer,"EPS_5Y_Gr":pct_renderer,
          "ROE":pct_renderer,"Gross_Mgn":pct_renderer,"Op_Mgn":pct_renderer,
          "Earnings_Soon":bool_renderer,"Optionable":bool_renderer,
          "Ser_OK":bool_renderer,"FV_OK":bool_renderer}
    for c,r in rmap.items():
        if c in df_disp.columns: gb.configure_column(c,cellRenderer=r)

    if "Ticker" in df_disp.columns: gb.configure_column("Ticker",pinned="left")
    if "Nome"   in df_disp.columns: gb.configure_column("Nome",  pinned="left")

    go_opts=gb.build()
    # Salva larghezza colonne e ordinamento nel localStorage del browser
    # (persiste tra refresh/scanner/sessioni finché non si svuota la cache)
    state_key=f"grid_state_{grid_key}"
    go_opts["onFirstDataRendered"]=JsCode(f"""
function(p){{
  const key='{state_key}';
  try{{
    const saved=localStorage.getItem(key);
    if(saved){{
      const st=JSON.parse(saved);
      if(st.colState) p.columnApi.applyColumnState({{state:st.colState,applyOrder:true}});
      if(st.sortState) p.api.setSortModel(st.sortState);
    }} else {{
      p.api.sizeColumnsToFit();
    }}
  }}catch(e){{p.api.sizeColumnsToFit();}}
}}""")
    go_opts["onColumnResized"]=JsCode(f"""
function(p){{
  if(!p.finished)return;
  const key='{state_key}';
  try{{
    const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
    saved.colState=p.columnApi.getColumnState();
    localStorage.setItem(key,JSON.stringify(saved));
  }}catch(e){{}}
}}""")
    go_opts["onSortChanged"]=JsCode(f"""
function(p){{
  const key='{state_key}';
  try{{
    const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
    saved.sortState=p.api.getSortModel();
    localStorage.setItem(key,JSON.stringify(saved));
  }}catch(e){{}}
}}""")
    go_opts["onColumnMoved"]=JsCode(f"""
function(p){{
  const key='{state_key}';
  try{{
    const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
    saved.colState=p.columnApi.getColumnState();
    localStorage.setItem(key,JSON.stringify(saved));
  }}catch(e){{}}
}}""")

    update=GridUpdateMode.VALUE_CHANGED if editable_cols else GridUpdateMode.SELECTION_CHANGED
    return AgGrid(df_disp,gridOptions=go_opts,height=height,
                  update_mode=update,
                  data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                  fit_columns_on_grid_load=False,theme="streamlit",
                  allow_unsafe_jscode=True,key=grid_key)

# =========================================================================
# LEGENDE
# =========================================================================
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
    if df.empty:
        st.info(f"Nessun dato. Esegui lo scanner per popolare **{title}**."); return

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
        if "Stato" not in df.columns: st.warning("Colonna Stato mancante."); return
        df_f=df[df["Stato"]=="HOT"].copy()

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
            st.warning("Colonna Ser_OK non trovata. Riesegui scanner v27.0."); return
        df_f=df[df["Ser_OK"].isin([True,"True","true"])].copy()
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="FINVIZ_PRO":
        if "FV_Score" not in df.columns:
            st.warning("Colonna FV_Score non trovata. Riesegui scanner v27.0."); return
        df_f=df[df["FV_OK"].isin([True,"True","true"])].copy()
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    else:
        df_f=df.copy()

    if df_f.empty:
        st.warning(f"⚠️ Nessun segnale **{title}** con le soglie attuali.\n"
                   "💡 Prova **🔓 Nessun Filtro** o abbassa le soglie nella sidebar."); return

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
    # Ordine: Ticker, Nome, Prezzo subito dopo, poi il resto
    cols=list(df_disp.columns)
    priority=["Ticker","Nome","Prezzo","Prezzo_fmt"]
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
            add_to_watchlist(tickers,names,title,"Scanner","LONG",st.session_state.current_list_name)
            st.success(f"✅ Aggiunti {len(tickers)} titoli a '{st.session_state.current_list_name}'.")
            time.sleep(0.8); st.rerun()
        else: st.warning("Seleziona almeno una riga.")

    if not selected_df.empty:
        ticker_sel=selected_df.iloc[0].get("Ticker","")
        match=df_f[df_f["Ticker"]==ticker_sel]
        if not match.empty: show_charts(match.iloc[0],key_suffix=title)

# =========================================================================
# TABS
# =========================================================================
tabs=st.tabs(["📡 EARLY","💪 PRO","🔥 REA-HOT","⭐ CONFLUENCE",
              "🚀 Momentum","🌐 Multi-TF","🔎 Finviz",
              "🎯 Serafini","🔎 Finviz Pro",
              "📋 Watchlist","📈 Backtest","📜 Storico"])
(tab_e,tab_p,tab_r,tab_conf,tab_regime,tab_mtf,
 tab_finviz,tab_ser,tab_fvpro,tab_w,tab_bt,tab_hist)=tabs

with tab_e:
    st.session_state.last_active_tab="EARLY"; show_legend("EARLY")
    render_scan_tab(df_ep,"EARLY",["Early_Score","RSI"],[False,True],"EARLY")

with tab_p:
    st.session_state.last_active_tab="PRO"; show_legend("PRO")
    render_scan_tab(df_ep,"PRO",["Quality_Score","Pro_Score","RSI"],[False,False,True],"PRO")

with tab_r:
    st.session_state.last_active_tab="REA-HOT"; show_legend("REA-HOT")
    render_scan_tab(df_rea,"HOT",["Vol_Ratio","Dist_POC_%"],[False,True],"REA-HOT")

with tab_conf:
    st.session_state.last_active_tab="CONFLUENCE"; show_legend("⭐ CONFLUENCE")
    render_scan_tab(df_ep,"CONFLUENCE",["Quality_Score","Early_Score","Pro_Score"],[False,False,False],"CONFLUENCE")

with tab_regime:
    show_legend("Regime Momentum")
    render_scan_tab(df_ep,"REGIME",["Pro_Score"],[False],"Regime Momentum")

with tab_mtf:
    show_legend("Multi-Timeframe")
    render_scan_tab(df_ep,"MTF",["Quality_Score","Pro_Score"],[False,False],"Multi-Timeframe")

with tab_finviz:
    show_legend("Finviz")
    sp=df_ep.get("Stato_Pro")
    df_fv=df_ep[sp=="PRO"].copy() if sp is not None and not df_ep.empty else df_ep.copy()
    if not df_fv.empty:
        if "MarketCap" in df_fv.columns: df_fv=df_fv[df_fv["MarketCap"]>=df_fv["MarketCap"].median()]
        if "Vol_Ratio" in df_fv.columns: df_fv=df_fv[df_fv["Vol_Ratio"]>1.2]
    render_scan_tab(df_fv,"PRO",["Quality_Score","Pro_Score"],[False,False],"Finviz")

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
# WATCHLIST — AgGrid + cards + multi-lista
# =========================================================================
with tab_w:
    st.markdown(f'<div class="section-pill">📋 WATCHLIST MANAGER — {st.session_state.current_list_name}</div>',
                unsafe_allow_html=True)

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
                rename_watchlist(ren_src,ren_dst.strip())
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
                    add_to_watchlist(df_src[tc].tolist(),
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
                        move_watchlist_rows(selected_ids,move_dest); st.rerun()
                with ac2:
                    if st.button(f"📋 Copia in '{copy_dest2}'",key="do_cp_g"):
                        rows_s=df_wl_disp[df_wl_disp["id"].isin(selected_ids)]
                        add_to_watchlist(rows_s[tcol].tolist(),
                            rows_s[ncol].tolist() if ncol in rows_s.columns else rows_s[tcol].tolist(),
                            "Copia","da selezione","LONG",copy_dest2)
                        st.success("✅ Copiati."); st.rerun()
                with ac3:
                    if st.button("🗑️ Elimina sel.",key="do_dl_g",type="secondary"):
                        delete_from_watchlist(selected_ids); st.rerun()

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
                        delete_from_watchlist([rid]); st.rerun()

            if selected_ids:
                ac1,ac2,ac3=st.columns(3)
                with ac1:
                    if st.button(f"➡️ Sposta in '{move_dest}'",key="do_mv_c"):
                        move_watchlist_rows(selected_ids,move_dest); st.rerun()
                with ac2:
                    if st.button(f"📋 Copia in '{copy_dest2}'",key="do_cp_c"):
                        rows_s=df_wl_disp[df_wl_disp["id"].isin(selected_ids)]
                        add_to_watchlist(rows_s[tcol].tolist(),
                            rows_s[ncol].tolist() if ncol in rows_s.columns else rows_s[tcol].tolist(),
                            "Copia","da selezione","LONG",copy_dest2)
                        st.success("✅ Copiati."); st.rerun()
                with ac3:
                    if st.button("🗑️ Elimina sel.",key="do_dl_c",type="secondary"):
                        delete_from_watchlist(selected_ids); st.rerun()

        # ── Grafici ticker selezionato ────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-pill">📊 ANALISI TICKER</div>',unsafe_allow_html=True)
        tickers_wl=df_wl[tcol].dropna().unique().tolist()
        if tickers_wl:
            sel_wl=st.selectbox("🔍 Seleziona ticker",options=tickers_wl,key="wl_tkr_sel")
            row_wl=None
            for src in [df_ep,df_rea]:
                if src.empty or "Ticker" not in src.columns: continue
                m=src[src["Ticker"]==sel_wl]
                if not m.empty: row_wl=m.iloc[0]; break
            if row_wl is not None: show_charts(row_wl,key_suffix="wl")
            else: st.info(f"📭 Dati non disponibili per **{sel_wl}**. Esegui lo scanner.")

    if st.button("🔄 Refresh",key="wl_ref"): st.rerun()

# =========================================================================
# STORICO
# =========================================================================
with tab_bt:
    render_backtest_tab()

with tab_hist:
    st.markdown('<div class="section-pill">📜 STORICO SCANSIONI</div>',unsafe_allow_html=True)
    _,col_rst=st.columns([4,1])
    with col_rst:
        if st.button("🗑️ Reset",key="rst_hist",type="secondary"):
            conn=sqlite3.connect(str(DB_PATH)); conn.execute("DELETE FROM scan_history")
            conn.commit();conn.close(); st.success("Storico cancellato!"); st.rerun()
    df_hist=load_scan_history(20)
    if df_hist.empty: st.info("Nessuna scansione salvata.")
    else:
        # Formatta colonne nuove se presenti
        disp_hist = df_hist.copy()
        if "elapsed_s" in disp_hist.columns:
            disp_hist["elapsed_s"] = disp_hist["elapsed_s"].apply(
                lambda x: f"{x:.0f}s" if pd.notna(x) else "—")
        st.dataframe(disp_hist,use_container_width=True)
        st.markdown("---"); st.subheader("🔍 Confronto Snapshot")
        hc1,hc2=st.columns(2)
        with hc1: id_a=st.selectbox("Scansione A",df_hist["id"].tolist(),key="sn_a")
        with hc2: id_b=st.selectbox("Scansione B",df_hist["id"].tolist(),index=min(1,len(df_hist)-1),key="sn_b")
        if st.button("🔍 Confronta"):
            ea,_=load_scan_snapshot(id_a); eb,_=load_scan_snapshot(id_b)
            if ea.empty or eb.empty: st.warning("Dati non disponibili.")
            else:
                ta=set(ea.get("Ticker",[])); tb=set(eb.get("Ticker",[]))
                sc1,sc2,sc3=st.columns(3)
                sc1.metric("🆕 Nuovi",len(tb-ta)); sc2.metric("❌ Usciti",len(ta-tb)); sc3.metric("✅ Persistenti",len(ta&tb))
                if tb-ta: st.markdown("**🆕** "+" ".join(sorted(tb-ta)))
                if ta-tb: st.markdown("**❌** "+" ".join(sorted(ta-tb)))

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
        "TradingScanner_v27_Tutti.xlsx",
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
            "TradingScanner_v27_TV.csv","text/csv",key="csv_tv_all")
with ec3:
    st.download_button(f"📊 XLSX {cur_tab}",to_excel_bytes({cur_tab:df_cur}),
        f"TradingScanner_v27_{cur_tab}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="xlsx_curr")
with ec4:
    if not df_cur.empty and "Ticker" in df_cur.columns:
        st.download_button(f"📈 CSV TV {cur_tab}",make_tv_csv(df_cur,cur_tab),
            f"TradingScanner_v27_{cur_tab}_TV.csv","text/csv",key="csv_tv_curr")
