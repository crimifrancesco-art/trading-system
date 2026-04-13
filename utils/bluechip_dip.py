"""
Blue Chip Dip Scanner - Versione 40.0
Identifica blue chip in forte sconto dai massimi 52 settimane
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def render_bluechip_dip():
    """Renderizza la dashboard Blue Chip Dip"""
    
    st.markdown('<div class="section-pill">💎 BLUE CHIP DIP SCANNER</div>', unsafe_allow_html=True)
    
    # Blue chips da monitorare
    BLUE_CHIPS = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft', 
        'GOOGL': 'Alphabet',
        'AMZN': 'Amazon',
        'NVDA': 'NVIDIA',
        'META': 'Meta',
        'TSLA': 'Tesla',
        'BRK-B': 'Berkshire',
        'V': 'Visa',
        'JNJ': 'Johnson & Johnson',
        'WMT': 'Walmart',
        'JPM': 'JPMorgan',
        'MA': 'Mastercard',
        'PG': 'Procter & Gamble',
        'UNH': 'UnitedHealth',
        'HD': 'Home Depot',
        'DIS': 'Disney',
        'BAC': 'Bank of America',
        'ADBE': 'Adobe',
        'CRM': 'Salesforce',
        'NFLX': 'Netflix',
        'CSCO': 'Cisco',
        'PFE': 'Pfizer',
        'KO': 'Coca-Cola',
        'PEP': 'PepsiCo',
        'INTC': 'Intel',
        'ORCL': 'Oracle',
        'NKE': 'Nike',
        'MRK': 'Merck',
        'ABBV': 'AbbVie'
    }
    
    # Parametri
    col1, col2, col3 = st.columns(3)
    with col1:
        dip_threshold = st.slider("📉 Sconto Minimo da Max 52w", 5, 50, 15, 5, 
                                  help="Percentuale minima di sconto dai massimi")
    with col2:
        min_market_cap = st.number_input("💰 Cap. Min (Miliardi $)", 10, 500, 50, 10,
                                         help="Capitalizzazione minima in miliardi")
    with col3:
        show_all = st.checkbox("📊 Mostra Tutti", False,
                              help="Mostra anche quelli sopra soglia")
    
    if st.button("🔍 SCANSIONA", use_container_width=True):
        with st.spinner("Scansione blue chips in corso..."):
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (ticker, name) in enumerate(BLUE_CHIPS.items()):
                try:
                    status_text.text(f"Analizzando {name} ({ticker})...")
                    progress_bar.progress((idx + 1) / len(BLUE_CHIPS))
                    
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1y")
                    
                    if hist.empty:
                        continue
                    
                    current_price = hist['Close'].iloc[-1]
                    max_52w = hist['High'].max()
                    pct_from_max = ((current_price - max_52w) / max_52w) * 100
                    
                    # Variazioni
                    var_1d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) > 1 else 0
                    var_5d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-6]) - 1) * 100 if len(hist) > 5 else 0
                    
                    # Market cap
                    info = stock.info
                    market_cap = info.get('marketCap', 0) / 1e9  # in miliardi
                    
                    is_dip = pct_from_max <= -dip_threshold and market_cap >= min_market_cap
                    
                    if show_all or is_dip:
                        results.append({
                            'Ticker': ticker,
                            'Nome': name,
                            'Prezzo Attuale': current_price,
                            'Max 52w': max_52w,
                            '% da Max': pct_from_max,
                            'Var. 1g': var_1d,
                            'Var. 5g': var_5d,
                            'Cap. (B)': market_cap,
                            'Blue Chip Dip': '✅ Sì' if is_dip else '❌ No'
                        })
                        
                except Exception as e:
                    st.warning(f"Errore su {ticker}: {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                df = pd.DataFrame(results)
                df = df.sort_values('% da Max', ascending=True)
                
                # Metriche riepilogative
                num_dips = len(df[df['Blue Chip Dip'].str.contains('Sì')])
                avg_discount = df[df['Blue Chip Dip'].str.contains('Sì')]['% da Max'].mean() if num_dips > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 Opportunità Trovate", num_dips)
                col2.metric("📊 Totale Analizzati", len(df))
                col3.metric("📉 Sconto Medio", f"{avg_discount:.1f}%")
                col4.metric("💰 Cap. Media", f"${df['Cap. (B)'].mean():.1f}B")
                
                st.markdown("---")
                
                # Tabella con styling
                df_display = df.copy()
                
                # CORREZIONE: Usa .map invece di .applymap per pandas >= 2.1.0
                styled = df_display.style\
                    .map(lambda x: 'background-color: #90EE90' if isinstance(x, str) and 'Sì' in x else '', subset=['Blue Chip Dip'])\
                    .map(lambda x: 'background-color: #FFB6C6' if isinstance(x, str) and 'No' in x else '', subset=['Blue Chip Dip'])\
                    .format({
                        'Prezzo Attuale': '${:.2f}',
                        'Max 52w': '${:.2f}',
                        '% da Max': '{:.1f}%',
                        'Var. 1g': '{:+.2f}%',
                        'Var. 5g': '{:+.2f}%',
                        'Cap. (B)': '${:.1f}B'
                    })
                
                st.dataframe(styled, use_container_width=True, height=600)
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Scarica CSV",
                    csv,
                    f"blue_chip_dip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
            else:
                st.warning("Nessun risultato trovato con i criteri selezionati")
    
    # Info
    with st.expander("ℹ️ Come funziona"):
        st.markdown("""
        ### Blue Chip Dip Scanner
        
        Questo strumento identifica **blue chip** (titoli a grande capitalizzazione) che sono 
        in **forte sconto** rispetto ai loro massimi a 52 settimane.
        
        **Criteri:**
        - 🏢 Solo aziende con cap > soglia impostata
        - 📉 Sconto minimo configurabile dai massimi 52w
        - ✅ Indicatore visivo per opportunità rilevate
        
        **Metriche:**
        - **% da Max**: Distanza percentuale dal massimo annuale (negativo = sconto)
        - **Var. 1g/5g**: Performance recente per valutare momentum
        - **Cap. (B)**: Capitalizzazione in miliardi di dollari
        
        💡 **Strategia**: Acquistare blue chip di qualità quando sono sottovalutate può offrire 
        opportunità di lungo termine con rischio relativamente contenuto.
        """)

