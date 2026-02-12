import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

st.set_page_config(page_title="Trading Dashboard", layout="wide", page_icon="📊")

st.sidebar.title("⚙️ Configurazione")
st.sidebar.header("📈 Selezione Mercati")

m = {
    "Eurostoxx": st.sidebar.checkbox("🇪🇺 Eurostoxx 600", True),
    "FTSE": st.sidebar.checkbox("🇮🇹 FTSE MIB", True),
    "SP500": st.sidebar.checkbox("🇺🇸 S&P 500", True),
    "Nasdaq": st.sidebar.checkbox("🇺🇸 Nasdaq 100", False),
    "Dow": st.sidebar.checkbox("🇺🇸 Dow Jones", False),
    "Russell": st.sidebar.checkbox("🇺🇸 Russell 2000", False),
    "Commodities": st.sidebar.checkbox("🛢️ Materie Prime", False),
    "ETF": st.sidebar.checkbox("📦 ETF", False),
    "Crypto": st.sidebar.checkbox("₿ Crypto", False),
    "Emerging": st.sidebar.checkbox("🌍 Emergenti", False)
}
sel = [k for k,v in m.items() if v]

st.sidebar.divider()
st.sidebar.header("🎛️ Parametri Scanner")
e_h = st.sidebar.slider("EARLY - Distanza EMA20 (%)", 0.0, 10.0, 2.0, 0.5)/100
p_rmin = st.sidebar.slider("PRO - RSI minimo", 0, 100, 40, 5)
p_rmax = st.sidebar.slider("PRO - RSI massimo", 0, 100, 70, 5)
r_poc = st.sidebar.slider("REA - Distanza POC (%)", 0.0, 10.0, 2.0, 0.5)/100
top = st.sidebar.number_input("TOP N titoli", 5, 50, 15, 5)

st.title("📊 Trading System Dashboard")
st.markdown("**Scanner EARLY + PRO + REA-QUANT con selezione mercati**")

if not sel:
    st.warning("⚠️ Seleziona almeno un mercato dalla sidebar!")
    st.stop()

st.info(f"🎯 Mercati selezionati: **{', '.join(sel)}**")

@st.cache_data(ttl=3600)
def load(markets):
    t = []
    if "SP500" in markets:
        t += pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")["Symbol"].tolist()
    if "Nasdaq" in markets:
        t += ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","NFLX","ADBE","COST","PEP","CSCO","INTC","AMD"]
    if "Dow" in markets:
        t += ["AAPL","MSFT","JPM","V","UNH","JNJ","WMT","PG","HD","DIS","KO","MCD","BA","CAT","GS"]
    if "Russell" in markets:
        t += ["IWM","VTWO"]
    if "FTSE" in markets:
        t += ["UCG.MI","ISP.MI","ENEL.MI","ENI.MI","LDO.MI","PRY.MI","STM.MI","TEN.MI","A2A.MI","AMP.MI"]
    if "Eurostoxx" in markets:
        t += ["ASML.AS","NESN.SW","SAN.PA","TTE.PA","AIR.PA","MC.PA","OR.PA","SU.PA"]
    if "Commodities" in markets:
        t += ["GC=F","CL=F","SI=F","NG=F","HG=F"]
    if "ETF" in markets:
        t += ["SPY","QQQ","IWM","GLD","TLT","VTI","EEM"]
    if "Crypto" in markets:
        t += ["BTC-USD","ETH-USD","BNB-USD","XRP-USD","SOL-USD"]
    if "Emerging" in markets:
        t += ["EEM","EWZ","INDA","FXI"]
    return list(dict.fromkeys(t))

def scan(ticker, e_h, p_rmin, p_rmax, r_poc):
    try:
        d = yf.Ticker(ticker).history(period="6mo")
        if len(d)<40: 
            return None, None
        
        c, v = d["Close"], d["Volume"]
        info = yf.Ticker(ticker).info
        n = info.get("longName", info.get("shortName", ticker))[:25]
        p = float(c.iloc[-1])
        e = float(c.ewm(20).mean().iloc[-1])
        
        # EARLY
        dist = abs(p-e)/e
        early_score = 8 if dist<e_h else 0
        
        # PRO
        pro_score = 3 if p>e else 0
        delta = c.diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = -delta.where(delta<0,0).rolling(14).mean()
        rsi = 100-(100/(1+(gain/loss)))
        rv = float(rsi.iloc[-1])
        pro_score += 3 if p_rmin<rv<p_rmax else 0
        vr = float(v.iloc[-1]/v.rolling(20).mean().iloc[-1])
        pro_score += 2 if vr>1.2 else 0
        
        stato_ep = "PRO" if pro_score>=8 else ("EARLY" if early_score>=8 else "-")
        
        # REA
        h, l = d["High"], d["Low"]
        tp = (h+l+c)/3
        bins = np.linspace(float(l.min()),float(h.max()),50)
        price_bins = pd.cut(tp, bins, labels=bins[:-1])
        vp = pd.DataFrame({"P":price_bins,"V":v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dpoc = abs(p-poc)/poc
        rea_score = 7 if dpoc<r_poc and vr>1.5 else 0
        
        stato_rea = "HOT" if rea_score>=7 else "-"
        
        result_ep = {
            "Nome": n,
            "Ticker": ticker,
            "Prezzo": round(p,2),
            "Early_Score": early_score,
            "Pro_Score": pro_score,
            "RSI": round(rv,1),
            "Vol": round(vr,2),
            "Stato": stato_ep
        }
        
        result_rea = {
            "Nome": n,
            "Ticker": ticker,
            "Prezzo": round(p,2),
            "Rea_Score": rea_score,
            "POC": round(poc,2),
            "Dist_POC": round(dpoc*100,1),
            "Vol": round(vr,2),
            "Stato": stato_rea
        }
        
        return result_ep, result_rea
        
    except Exception as e:
        return None, None

if st.button("🚀 AVVIA SCANNER", type="primary", use_container_width=True):
    tickers = load(sel)
    st.info(f"⏳ Scansione di **{len(tickers)} titoli** in corso...")
    
    pb = st.progress(0)
    status = st.empty()
    
    r1, r2 = [], []
    
    for i, ticker in enumerate(tickers):
        status.text(f"Analisi: {ticker} ({i+1}/{len(tickers)})")
        pb.progress((i+1)/len(tickers))
        
        ep, rea = scan(ticker, e_h, p_rmin, p_rmax, r_poc)
        if ep: r1.append(ep)
        if rea: r2.append(rea)
        
        if (i+1)%10==0: 
            time.sleep(0.2)
    
    status.text("✅ Scansione completata!")
    pb.empty()
    
    st.session_state['df_ep'] = pd.DataFrame(r1)
    st.session_state['df_rea'] = pd.DataFrame(r2)
    st.session_state['done'] = True
    st.rerun()

if st.session_state.get('done'):
    df_ep = st.session_state['df_ep']
    df_rea = st.session_state['df_rea']
    
    st.success(f"✅ Analizzati {len(df_ep)} titoli")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Titoli EARLY", len(df_ep[df_ep["Stato"]=="EARLY"]))
    with col2:
        st.metric("Titoli PRO", len(df_ep[df_ep["Stato"]=="PRO"]))
    with col3:
        st.metric("Titoli REA-HOT", len(df_rea[df_rea["Stato"]=="HOT"]))
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["🔵 EARLY","🟢 PRO","🎯 REA-QUANT"])
    
    with tab1:
        st.header("🔵 TOP EARLY")
        top_early = df_ep.sort_values("Early_Score", ascending=False).head(top)
        st.dataframe(top_early, use_container_width=True, height=400)
        st.download_button(
            "📥 Scarica CSV",
            top_early.to_csv(index=False).encode('utf-8'),
            "early_top.csv",
            "text/csv"
        )
        
    with tab2:
        st.header("🟢 TOP PRO")
        top_pro = df_ep.sort_values("Pro_Score", ascending=False).head(top)
        st.dataframe(top_pro, use_container_width=True, height=400)
        st.download_button(
            "📥 Scarica CSV",
            top_pro.to_csv(index=False).encode('utf-8'),
            "pro_top.csv",
            "text/csv"
        )
        
    with tab3:
        st.header("🎯 TOP REA-QUANT")
        top_rea = df_rea.sort_values("Rea_Score", ascending=False).head(top)
        st.dataframe(top_rea, use_container_width=True, height=400)
        st.download_button(
            "📥 Scarica CSV",
            top_rea.to_csv(index=False).encode('utf-8'),
            "rea_top.csv",
            "text/csv"
        )

