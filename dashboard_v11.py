# ==========================================================
# DASHBOARD V11 — PROFESSIONAL SCANNER (MODULAR)
# ==========================================================
import streamlit as st
import pandas as pd
import json
import io
import numpy as np
from pathlib import Path
from datetime import datetime
from run_scan import run_scan

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner V11 Professional",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner — V11 Professional")

# ----------------------------------------------------------
# UTILITIES & FORMATTING
# ----------------------------------------------------------
def fmt_currency(value, currency="USD"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    symbol = "$" if currency == "USD" else "€"
    # Format: $1.234,56 or €1.234,56 (European style)
    try:
        res = f"{symbol}{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return res
    except:
        return str(value)

def fmt_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        return f"{int(value):,}".replace(",", ".")
    except:
        return str(value)

def fmt_market_cap(value):
    if not value or (isinstance(value, float) and np.isnan(value)): return "N/A"
    try:
        if value >= 1e12: return f"{value/1e12:.2f}T".replace(".", ",")
        if value >= 1e9: return f"{value/1e9:.2f}B".replace(".", ",")
        if value >= 1e6: return f"{value/1e6:.2f}M".replace(".", ",")
        return str(value)
    except:
        return str(value)

# ----------------------------------------------------------
# SIDEBAR & MARKETS
# ----------------------------------------------------------
st.sidebar.header("⚙️ CONFIGURAZIONE")

MARKETS = {
    "🇺🇸 USA (S&P 500)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V"],
    "🇮🇹 ITALIA (FTSE MIB)": ["ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI", "STLAM.MI", "G.MI", "FER.MI", "PST.MI", "A2A.MI", "PRY.MI"],
    "🇪🇺 EUROPA": ["ASML", "MC.PA", "SAP", "OR.PA", "TTE.PA", "SIE.DE", "NESN.SW", "NOVN.SW", "ROG.SW"],
    "⚡ CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
}

market_choice = st.sidebar.multiselect("Seleziona Mercati", list(MARKETS.keys()), default=[list(MARKETS.keys())[0]])
custom_tickers = st.sidebar.text_input("Tickers Manuali (separati da virgola)", "")

tickers_to_scan = []
for m in market_choice:
    tickers_to_scan.extend(MARKETS[m])
if custom_tickers:
    tickers_to_scan.extend([t.strip().upper() for t in custom_tickers.split(",") if t.strip()])

# Save universe for scanner
runtime_path = Path("data/runtime_universe.json")
runtime_path.parent.mkdir(parents=True, exist_ok=True)
runtime_path.write_text(json.dumps({"tickers": list(set(tickers_to_scan))}))

if st.sidebar.button("🚀 AVVIA SCANNER", use_container_width=True):
    with st.spinner("Scansione in corso..."):
        try:
            run_scan()
            st.success("Scansione completata!")
            st.rerun()
        except Exception as e:
            st.error(f"Errore durante lo scan: {e}")

# ----------------------------------------------------------
# LEGENDA FILTRI
# ----------------------------------------------------------
with st.expander("ℹ️ LEGENDA FILTRI & SEGNALI"):
    st.markdown("""
    **Criteri di Selezione (Score 0-5):**
    1.  **Trend:** EMA50 sopra il prezzo di 5 giorni fa (Trend Rialzista).
    2.  **RSI Momentum:** RSI in crescita e sotto 70 (No Ipercomprato).
    3.  **MACD Cross:** Linea MACD sopra la linea Signal.
    4.  **Volume:** Volume odierno superiore alla media 20 giorni.
    5.  **Volatility:** ATR/Price tra 0.5% e 10% (Volatilità sana).

    **Segnali:**
    - 🟢 **STRONG BUY:** Tutti i 5 criteri soddisfatti.
    - 🟡 **BUY:** Almeno 3 criteri soddisfatti.
    - ⚪ **NONE:** Meno di 3 criteri soddisfatti.
    """)

# ----------------------------------------------------------
# RESULTS DASHBOARD
# ----------------------------------------------------------
results_file = Path("data/scan_results.json")

if results_file.exists():
    try:
        data = json.loads(results_file.read_text())
        if not data:
            st.warning("Nessun titolo trovato nell'ultima scansione.")
        else:
            df = pd.DataFrame(data)
            
            # Defensive check for new columns (fix KeyError 'vol_today')
            required_cols = ['name', 'vol_today', 'vol_7d_avg', 'market_cap']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.warning(f"⚠️ Dati obsoleti rilevati. Esegui 'AVVIA SCANNER' per aggiornare i risultati (Mancano: {', '.join(missing)}).")
                for m in missing: df[m] = None

            # Filters in Main Page
            col1, col2 = st.columns(2)
            with col1:
                min_score = st.slider("Min Score", 0, 5, 3)
            with col2:
                signal_filter = st.multiselect("Segnale", ["STRONG BUY", "BUY", "NONE"], default=["STRONG BUY", "BUY"])
            
            # Apply Filters
            filtered_df = df[(df['score'] >= min_score) & (df['signal'].isin(signal_filter))].copy()
            
            st.subheader(f"🔍 Risultati ({len(filtered_df)})")
            
            if not filtered_df.empty:
                # Formatting Table
                display_df = pd.DataFrame()
                display_df["Nome"] = filtered_df["name"] if "name" in filtered_df.columns else "N/A"
                display_df["Ticker"] = filtered_df["ticker"]
                display_df["Prezzo"] = filtered_df.apply(lambda x: fmt_currency(x["price"], x.get("currency", "USD")), axis=1)
                display_df["Market Cap"] = filtered_df["market_cap"].apply(fmt_market_cap)
                display_df["Vol Giorno"] = filtered_df["vol_today"].apply(fmt_int)
                display_df["Vol Medio 7g"] = filtered_df["vol_7d_avg"].apply(fmt_int)
                display_df["RSI"] = filtered_df["rsi"]
                display_df["Score"] = filtered_df["score"]
                display_df["Segnale"] = filtered_df["signal"]
                
                # Links column
                display_df["Links"] = filtered_df["ticker"].apply(lambda t: 
                    f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">Yahoo</a> | '
                    f'<a href="https://it.tradingview.com/symbols/{t.replace("-", "")}/" target="_blank">TV</a>'
                )

                # Style
                def color_signal(val):
                    if val == "STRONG BUY": return "background-color: #2ecc71; color: white;"
                    if val == "BUY": return "background-color: #f1c40f; color: black;"
                    return ""

                st.write(display_df.style.applymap(color_signal, subset=["Segnale"]).to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Export
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Scarica CSV", csv, "scan_results.csv", "text/csv")
                with c2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='Sheet1')
                    st.download_button("📥 Scarica Excel", buffer.getvalue(), "scan_results.xlsx", "application/vnd.ms-excel")
            else:
                st.info("Nessun titolo soddisfa i filtri selezionati.")
                
    except Exception as e:
        st.error(f"Errore lettura risultati: {e}")
else:
    st.info("Benvenuto! Clicca su 'AVVIA SCANNER' nella sidebar per iniziare la prima scansione.")
