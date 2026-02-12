import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

st.set_page_config(page_title="Trading Dashboard", layout="wide", page_icon="📊")

st.sidebar.title("⚙️ Config")
st.sidebar.header("📈 Mercati")

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
e_h = st.sidebar.slider("EARLY dist%", 0.0, 10.0, 2.0)/100
p_rmin = st.sidebar.slider("PRO RSI min", 0, 100, 40)
p_rmax = st.sidebar.slider("PRO RSI max", 0, 100, 70)
r_poc = st.sidebar.slider("REA POC%", 0.0, 10.0, 2.0)/100
top = st.sidebar.number_input("TOP N", 5, 50, 15)

st.title("📊 Trading Dashboard")
if not sel:
    st.warning("Seleziona mercati!")
    st.stop()
st.info(f"Mercati: {', '.join(sel)}")

@st.cache_data(ttl=3600)
def load(markets):
    t = []
    if "SP500" in markets:
        t += pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")["Symbol"].tolist()
    if "Nasdaq" in markets:
        t += ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","NFLX","ADBE"]
    if "Dow" in markets:
        t += ["AAPL","MSFT","JPM","V","UNH","JNJ","WMT","PG","HD","DIS"]
    if "FTSE" in markets:
        t += ["UCG.MI","ISP.MI","ENEL.MI","ENI.MI","LDO.MI","PRY.MI","STM.MI"]
    if "Eurostoxx" in markets:
        t += ["ASML.AS","NESN.SW","SAN.PA","TTE.PA","AIR.PA"]
    if "Commodities" in markets:
        t += ["GC=F","CL=F","SI=F","NG=F"]
    if "ETF" in markets:
        t += ["SPY","QQQ","IWM","GLD","TLT"]
    if "Crypto" in markets:
        t += ["BTC-USD","ETH-USD","BNB-USD","XRP-USD"]
    if "Emerging" in markets:
        t += ["EEM","EWZ","INDA","FXI"]
    return list(dict.fromkeys(t))

def scan(t, e_h, p_rmin, p_rmax, r_poc):
    try:
        d = yf.Ticker(t).history(period="6mo")
        if len(d)<40: return None, None
        c, v = d["Close"], d["Volume"]
        n = yf.Ticker(t).info.get("longName",t)[:25]
        p = float(c.iloc[-1])
        e = float(c.ewm(20).mean().iloc[-1])
        dist = abs(p-e)/e
        early = 8 if dist<e_h else 0
        pro = 3 if p>e else 0
        delta = c.diff()
        rsi = 100-(100/(1+(delta.where(delta>0,0).rolling(14).mean()/(-delta.where(delta<0,0).rolling(14).mean()))))
        rv = float(rsi.iloc[-1])
        pro += 3 if p_rmin<rv<p_rmax else 0
        vr = float(v.iloc[-1]/v.rolling(20).mean().iloc[-1])
        pro += 2 if vr>1.2 else 0
        st1 = "PRO" if pro>=8 else ("EARLY" if early>=8 else "-")
        
        h, l = d["High"], d["Low"]
        tp = (h+l+c)/3
        bins = np.linspace(float(l.min()),float(h.max()),50)
        vp = pd.DataFrame({"P":pd.cut(tp,bins,labels=bins[:-1]),"V":v}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dpoc = abs(p-poc)/poc
        rea = 7 if dpoc<r_poc and vr>1.5 else 0
        st2 = "HOT" if rea>=7 else "-"
        
        return {
            "Nome":n,"Ticker":t,"Prezzo":round(p,2),"Early":early,"Pro":pro,"RSI":round(rv,1),"Stato":st1
        }, {
            "Nome":n,"Ticker":t,"Prezzo":round(p,2),"Rea":rea,"POC":round(poc,2),"Vol":round(vr,2),"Stato":st2
        }
    except:
        return None, None

if st.button("🚀 SCAN", type="primary"):
    tickers = load(sel)
    st.info(f"Scanning {len(tickers)} tickers...")
    pb = st.progress(0)
    tx = st.empty()
    r1, r2 = [], []
    for i,t in enumerate(tickers):
        tx.text(f"{t} ({i+1}/{len(tickers)})")
        pb.progress((i+1)/len(tickers))
        x, y = scan(t, e_h, p_rmin, p_rmax, r_poc)
        if x: r1.append(x)
        if y: r2.append(y)
        if (i+1)%10==0: time.sleep(0.2)
    tx.text("✅ Done!")
    pb.empty()
    st.session_state['df_ep'] = pd.DataFrame(r1)
    st.session_state['df_rea'] = pd.DataFrame(r2)
    st.session_state['done'] = True

if st.session_state.get('done'):
    df_ep = st.session_state['df_ep']
    df_rea = st.session_state['df_rea']
    
    tab1, tab2, tab3 = st.tabs(["EARLY","PRO","REA"])
    
    with tab1:
        st.header("🔵 TOP EARLY")
        t1 = df_ep.sort_values("Early",ascending=False).head(top)
        st.dataframe(t1, use_container_width=True)
        st.download_button("CSV", t1.to_csv(index=False), "early.csv")
        
    with tab2:
        st.header("🟢 TOP PRO")
        t2 = df_ep.sort_values("Pro",ascending=False).head(top)
        st.dataframe(t2, use_container_width=True)
        st.download_button("CSV", t2.to_csv(index=False), "pro.csv")
        
    with tab3:
        st.header("🎯 TOP REA")
        t3 = df_rea.sort_values("Rea",ascending=False).head(top)
        st.dataframe(t3, use_container_width=True)
        st.download_button("CSV", t3.to_csv(index=False), "rea.csv")
