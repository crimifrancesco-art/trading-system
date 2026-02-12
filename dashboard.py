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
# ============================================================================
# SEZIONE METRICHE SCANNER - CODICE FIXATO
# ============================================================================

st.header("📊 Risultati Scanner")

# Creazione di 3 colonne per le metriche
col1, col2, col3 = st.columns(3)

# ----------------------------------------------------------------------------
# METRICA 1: EARLY Scanner
# ----------------------------------------------------------------------------
with col1:
    try:
        if df_epidf_ep is None:
            st.metric("Titoli EARLY", 0)
            st.caption("⚠️ DataFrame non inizializzato")
        elif df_epidf_ep.empty:
            st.metric("Titoli EARLY", 0)
            st.caption("ℹ️ Nessun titolo trovato")
        elif "Stato" not in df_epidf_ep.columns:
            st.metric("Titoli EARLY", 0)
            st.caption(f"❌ Colonna 'Stato' mancante")
            # Debug info (rimuovi in produzione)
            with st.expander("🔍 Debug Info"):
                st.write(f"Colonne disponibili: {df_epidf_ep.columns.tolist()}")
        else:
            # CALCOLO CORRETTO
            n_early = len(df_epidf_ep[df_epidf_ep["Stato"] == "EARLY"])
            st.metric("Titoli EARLY", n_early)
            
            if n_early > 0:
                st.caption(f"✅ {n_early} opportunità trovate")
            else:
                st.caption("ℹ️ Nessun segnale al momento")
                
    except KeyError as e:
        st.metric("Titoli EARLY", 0)
        st.caption(f"❌ Errore colonna: {str(e)}")
    except Exception as e:
        st.metric("Titoli EARLY", 0)
        st.caption(f"❌ Errore: {str(e)}")

# ----------------------------------------------------------------------------
# METRICA 2: PRO Scanner
# ----------------------------------------------------------------------------
with col2:
    try:
        if df_pro is None:
            st.metric("Titoli PRO", 0)
            st.caption("⚠️ DataFrame non inizializzato")
        elif df_pro.empty:
            st.metric("Titoli PRO", 0)
            st.caption("ℹ️ Nessun titolo trovato")
        elif "Stato" not in df_pro.columns:
            st.metric("Titoli PRO", 0)
            st.caption(f"❌ Colonna 'Stato' mancante")
        else:
            # CALCOLO CORRETTO
            n_pro = len(df_pro[df_pro["Stato"] == "PRO"])
            st.metric("Titoli PRO", n_pro)
            
            if n_pro > 0:
                st.caption(f"✅ {n_pro} opportunità trovate")
            else:
                st.caption("ℹ️ Nessun segnale al momento")
                
    except KeyError as e:
        st.metric("Titoli PRO", 0)
        st.caption(f"❌ Errore colonna: {str(e)}")
    except Exception as e:
        st.metric("Titoli PRO", 0)
        st.caption(f"❌ Errore: {str(e)}")

# ----------------------------------------------------------------------------
# METRICA 3: REA-QUANT Scanner
# ----------------------------------------------------------------------------
with col3:
    try:
        if df_rea is None:
            st.metric("Titoli REA-QUANT", 0)
            st.caption("⚠️ DataFrame non inizializzato")
        elif df_rea.empty:
            st.metric("Titoli REA-QUANT", 0)
            st.caption("ℹ️ Nessun titolo trovato")
        elif "Stato" not in df_rea.columns:
            st.metric("Titoli REA-QUANT", 0)
            st.caption(f"❌ Colonna 'Stato' mancante")
        else:
            # CALCOLO CORRETTO - REA usa "distanza_poc" invece di "Stato"
            # Adatta in base alla tua logica
            if "distanza_poc" in df_rea.columns:
                n_rea = len(df_rea[df_rea["distanza_poc"] <= 2.00])  # Usa il tuo threshold
            else:
                n_rea = len(df_rea)  # Se non hai filtro specifico
                
            st.metric("Titoli REA-QUANT", n_rea)
            
            if n_rea > 0:
                st.caption(f"✅ {n_rea} opportunità trovate")
            else:
                st.caption("ℹ️ Nessun segnale al momento")
                
    except KeyError as e:
        st.metric("Titoli REA-QUANT", 0)
        st.caption(f"❌ Errore colonna: {str(e)}")
    except Exception as e:
        st.metric("Titoli REA-QUANT", 0)
        st.caption(f"❌ Errore: {str(e)}")

# ============================================================================
# SEZIONE VISUALIZZAZIONE RISULTATI
# ============================================================================

st.divider()

# ----------------------------------------------------------------------------
# TABS PER I 3 SCANNER
# ----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 EARLY Scanner", "💎 PRO Scanner", "📊 REA-QUANT Scanner"])

# TAB 1: EARLY Scanner
with tab1:
    st.subheader("🎯 Risultati EARLY Scanner")
    
    try:
        if df_epidf_ep is not None and not df_epidf_ep.empty and "Stato" in df_epidf_ep.columns:
            df_early_filtered = df_epidf_ep[df_epidf_ep["Stato"] == "EARLY"]
            
            if not df_early_filtered.empty:
                st.success(f"✅ Trovati {len(df_early_filtered)} titoli EARLY")
                
                # Ordina per volume o RSI (adatta alle tue colonne)
                if "Volume" in df_early_filtered.columns:
                    df_early_filtered = df_early_filtered.sort_values("Volume", ascending=False)
                
                # Mostra tabella
                st.dataframe(
                    df_early_filtered,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Download CSV
                csv = df_early_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"early_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ Nessun titolo EARLY trovato con i parametri attuali")
                st.write("**Suggerimenti:**")
                st.write("- Aumenta la distanza EMA20 (attuale: 2%)")
                st.write("- Allarga il range RSI minimo/massimo")
                st.write("- Verifica che il mercato sia aperto")
        else:
            st.warning("⚠️ Dati EARLY non disponibili. Clicca su 'AVVIA SCANNER' per caricare i dati.")
            
    except Exception as e:
        st.error(f"❌ Errore nella visualizzazione EARLY: {str(e)}")

# TAB 2: PRO Scanner
with tab2:
    st.subheader("💎 Risultati PRO Scanner")
    
    try:
        if df_pro is not None and not df_pro.empty and "Stato" in df_pro.columns:
            df_pro_filtered = df_pro[df_pro["Stato"] == "PRO"]
            
            if not df_pro_filtered.empty:
                st.success(f"✅ Trovati {len(df_pro_filtered)} titoli PRO")
                
                # Ordina per RSI
                if "RSI" in df_pro_filtered.columns:
                    df_pro_filtered = df_pro_filtered.sort_values("RSI", ascending=True)
                
                st.dataframe(
                    df_pro_filtered,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Download CSV
                csv = df_pro_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"pro_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ Nessun titolo PRO trovato con i parametri attuali")
                st.write("**Suggerimenti:**")
                st.write("- Aumenta RSI massimo (attuale: 70)")
                st.write("- Riduci RSI minimo (attuale: 40)")
        else:
            st.warning("⚠️ Dati PRO non disponibili. Clicca su 'AVVIA SCANNER' per caricare i dati.")
            
    except Exception as e:
        st.error(f"❌ Errore nella visualizzazione PRO: {str(e)}")

# TAB 3: REA-QUANT Scanner
with tab3:
    st.subheader("📊 Risultati REA-QUANT Scanner")
    
    try:
        if df_rea is not None and not df_rea.empty:
            # REA usa distanza_poc, non "Stato"
            if "distanza_poc" in df_rea.columns:
                df_rea_filtered = df_rea[df_rea["distanza_poc"] <= 2.00]  # Adatta threshold
            else:
                df_rea_filtered = df_rea
            
            if not df_rea_filtered.empty:
                st.success(f"✅ Trovati {len(df_rea_filtered)} titoli REA-QUANT")
                
                # Ordina per distanza POC
                if "distanza_poc" in df_rea_filtered.columns:
                    df_rea_filtered = df_rea_filtered.sort_values("distanza_poc", ascending=True)
                
                st.dataframe(
                    df_rea_filtered,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Download CSV
                csv = df_rea_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"rea_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ Nessun titolo REA-QUANT trovato con i parametri attuali")
                st.write("**Suggerimenti:**")
                st.write("- Aumenta distanza POC massima (attuale: 2%)")
        else:
            st.warning("⚠️ Dati REA-QUANT non disponibili. Clicca su 'AVVIA SCANNER' per caricare i dati.")
            
    except Exception as e:
        st.error(f"❌ Errore nella visualizzazione REA-QUANT: {str(e)}")

# ============================================================================
# SEZIONE ANALISI COMBINATA (BONUS)
# ============================================================================

st.divider()
st.header("🎯 Analisi Combinata")

try:
    # Trova titoli presenti in più scanner (segnali forti)
    titoli_early = set()
    titoli_pro = set()
    titoli_rea = set()
    
    if df_epidf_ep is not None and not df_epidf_ep.empty and "Ticker" in df_epidf_ep.columns:
        titoli_early = set(df_epidf_ep[df_epidf_ep["Stato"] == "EARLY"]["Ticker"].tolist())
    
    if df_pro is not None and not df_pro.empty and "Ticker" in df_pro.columns:
        titoli_pro = set(df_pro[df_pro["Stato"] == "PRO"]["Ticker"].tolist())
    
    if df_rea is not None and not df_rea.empty and "Ticker" in df_rea.columns:
        titoli_rea = set(df_rea["Ticker"].tolist())
    
    # Titoli in 2+ scanner
    titoli_multipli = (titoli_early & titoli_pro) | (titoli_early & titoli_rea) | (titoli_pro & titoli_rea)
    
    # Titoli in tutti e 3 i scanner (SEGNALE FORTISSIMO)
    titoli_tutti = titoli_early & titoli_pro & titoli_rea
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric("🔥 Titoli in 2+ Scanner", len(titoli_multipli))
        if titoli_multipli:
            st.write(", ".join(sorted(titoli_multipli)))
    
    with col_b:
        st.metric("⭐ Titoli in TUTTI i Scanner", len(titoli_tutti))
        if titoli_tutti:
            st.write(", ".join(sorted(titoli_tutti)))
            st.success("🎯 SEGNALI FORTISSIMI - Priorità massima!")
    
except Exception as e:
    st.error(f"❌ Errore nell'analisi combinata: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(f"📅 Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
st.caption("💡 Ricorda: Questi sono segnali automatici. Fai sempre la tua analisi prima di operare.")
