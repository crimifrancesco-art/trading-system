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
    if value is None:
        return ""
    try:
        if np.isnan(float(value)):
            return ""
    except:
        return str(value)
    symbol = "$" if currency == "USD" else "€"
    try:
        return f"{symbol}{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(value)

def fmt_int(value):
    if value is None:
        return ""
    try:
        v = float(value)
        if np.isnan(v):
            return ""
        return f"{int(v):,}".replace(",", ".")
    except:
        return str(value)

def fmt_market_cap(value):
    if not value:
        return "N/A"
    try:
        v = float(value)
        if np.isnan(v) or v == 0:
            return "N/A"
        if v >= 1e12:
            return f"{v/1e12:.2f}T".replace(".", ",")
        if v >= 1e9:
            return f"{v/1e9:.2f}B".replace(".", ",")
        if v >= 1e6:
            return f"{v/1e6:.2f}M".replace(".", ",")
        return str(int(v))
    except:
        return "N/A"

# ----------------------------------------------------------
# SIDEBAR & MARKETS
# ----------------------------------------------------------
st.sidebar.header("⚙️ CONFIGURAZIONE")

MARKETS = {
    "🇺🇸 USA (S&P 500)": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
        "BRK-B", "JPM", "V", "UNH", "JNJ", "XOM", "PG", "MA"
    ],
    "🇮🇹 ITALIA (FTSE MIB)": [
        "ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI", "STLAM.MI",
        "G.MI", "FER.MI", "PST.MI", "A2A.MI", "PRY.MI",
        "TIT.MI", "MB.MI", "BMED.MI"
    ],
    "🇪🇺 EUROPA": [
        "ASML", "MC.PA", "SAP", "OR.PA", "TTE.PA",
        "SIE.DE", "NESN.SW", "NOVN.SW", "ROG.SW", "LVMH.PA"
    ],
    "⚡ CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
}

market_choice = st.sidebar.multiselect(
    "Seleziona Mercati",
    list(MARKETS.keys()),
    default=[list(MARKETS.keys())[0]]
)
custom_tickers = st.sidebar.text_input(
    "Tickers Manuali (separati da virgola)", ""
)

active_tickers = []
for m in market_choice:
    active_tickers.extend(MARKETS[m])
if custom_tickers:
    active_tickers.extend(
        [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    )

# Save universe for scanner
runtime_path = Path("data/runtime_universe.json")
runtime_path.parent.mkdir(parents=True, exist_ok=True)
runtime_path.write_text(json.dumps({"tickers": list(set(active_tickers))}))

run_scan_btn = st.sidebar.button("🚀 AVVIA SCANNER", use_container_width=True)

# ----------------------------------------------------------
# LEGENDA FILTRI
# ----------------------------------------------------------
with st.expander("ℹ️ LEGENDA FILTRI & SEGNALI"):
    st.markdown("""
    **Criteri di Selezione (Score 0-5):**

    | # | Criterio | Descrizione |
    |---|----------|-------------|
    | 1 | **Trend** | EMA50 sopra il prezzo di 5 giorni fa |
    | 2 | **RSI Momentum** | RSI in crescita e sotto 70 (no ipercomprato) |
    | 3 | **MACD Cross** | Linea MACD sopra la linea Signal |
    | 4 | **Volume** | Volume odierno > media 20 giorni |
    | 5 | **Volatilità** | ATR/Price tra 0.5% e 10% |

    **Segnali:**
    - 🟢 **STRONG BUY** — Score 5/5
    - 🟡 **BUY** — Score 3-4/5
    - ⚪ **NONE** — Score < 3
    """)

# ----------------------------------------------------------
# RUN SCAN
# ----------------------------------------------------------
if run_scan_btn:
    if not active_tickers:
        st.warning("Seleziona almeno un mercato.")
        st.stop()
    with st.spinner("Scansione mercati in corso..."):
        try:
            run_scan()
            st.success("Scan completato ✅")
            st.rerun()
        except Exception as e:
            st.error(f"Errore durante lo scan: {e}")

# ----------------------------------------------------------
# RESULTS & DISPLAY
# ----------------------------------------------------------
result_path = Path("data/scan_results.json")
tab_results, tab_legend = st.tabs(["🗓️ Risultati Scan", "📊 Legenda Filtri"])

with tab_results:
    if result_path.exists():
        try:
            results = json.loads(result_path.read_text())
            df = pd.DataFrame(results)

            if not df.empty:
                # --- Compatibility fix: add missing new columns ---
                for col in ["name", "vol_today", "vol_7d_avg", "market_cap"]:
                    if col not in df.columns:
                        df[col] = None
                if any(c not in results[0] for c in ["vol_today", "vol_7d_avg", "name"] if results):
                    st.warning("⚠️ Dati obsoleti: esegui AVVIA SCANNER per aggiornare tutti i campi.")

                # Add links
                df["Yahoo"] = df["ticker"].apply(lambda x: f"https://finance.yahoo.com/quote/{x}")
                df["TradingView"] = df["ticker"].apply(lambda x: f"https://www.tradingview.com/chart/?symbol={x.split('.')[0]}")

                # Apply Formatting for Display
                df_display = df.copy()
                df_display["Prezzo"] = df.apply(lambda r: fmt_currency(r["price"], r.get("currency", "USD")), axis=1)
                df_display["Market Cap"] = df["market_cap"].apply(fmt_market_cap)
                df_display["Vol giorno"] = df["vol_today"].apply(fmt_int)
                df_display["Vol medio 7g"] = df["vol_7d_avg"].apply(fmt_int)

                # Metrics
                c1, c2, c3 = st.columns(3)
                n_strong = (df["signal"] == "STRONG BUY").sum()
                n_buy = (df["signal"] == "BUY").sum()

                c1.metric("STRONG BUY", int(n_strong))
                c2.metric("BUY", int(n_buy))
                c3.metric("Assets scansionati", len(df))

                # Table styling
                def color_signal(val):
                    if val == "STRONG BUY": return "background-color:#0f5132;color:white"
                    if val == "BUY": return "background-color:#664d03;color:white"
                    return ""

                COLS = [
                    "name", "ticker", "Prezzo", "Market Cap",
                    "Vol giorno", "Vol medio 7g",
                    "rsi", "score", "signal"
                ]
                rename_map = {"name": "Nome", "ticker": "Ticker", "rsi": "RSI", "score": "Score", "signal": "Segnale"}
                show_df = df_display[COLS].rename(columns=rename_map)

                st.dataframe(
                    show_df.style.applymap(color_signal, subset=["Segnale"]),
                    use_container_width=True,
                    hide_index=True
                )

                # Export
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Scarica CSV", csv, "scan_results.csv", "text/csv")
                with c2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False)
                    st.download_button("📥 Scarica Excel", buffer.getvalue(), "scan_results.xlsx")

        except Exception as e:
            st.error(f"Errore lettura risultati: {e}")
    else:
        st.info("👋 Benvenuto! Premi 'AVVIA SCANNER' nella sidebar per avviare la prima scansione.")

with tab_legend:
    st.markdown("""
    ### Come funziona il sistema

    Il sistema analizza ogni titolo su **5 indicatori tecnici** e assegna uno score da 0 a 5.

    | Indicatore | Logica | Peso |
    |------------|--------|------|
    | EMA50 Trend | EMA50 > prezzo 5gg fa | 1 |
    | RSI Momentum | RSI crescente e < 70 | 1 |
    | MACD Cross | MACD > Signal | 1 |
    | Volume | Vol odierno > media 20gg | 1 |
    | Volatilità ATR | 0.5% < ATR% < 10% | 1 |

    ### Segnali generati
    | Segnale | Condizione | Colore |
    |---------|------------|--------|
    | STRONG BUY | Score = 5 | 🟢 Verde |
    | BUY | Score ≥ 3 | 🟡 Giallo |
    | NONE | Score < 3 | ⚪ Grigio |

    ### Legenda colonne
    - **Nome**: Nome completo del titolo da Yahoo Finance
    - **Ticker**: Codice del titolo (es. AAPL, ENI.MI)
    - **Prezzo**: Ultimo prezzo di chiusura (formato europeo)
    - **Market Cap**: Capitalizzazione di mercato (T=trilioni, B=miliardi, M=milioni)
    - **Vol Giorno**: Volume scambiato oggi
    - **Vol Medio 7g**: Media volumi ultimi 7 giorni
    - **RSI**: Relative Strength Index (14 periodi)
    - **Score**: Punteggio totale (0-5)
    - **OBV Trend**: Tendenza On-Balance Volume (UP/DOWN)
    """)
