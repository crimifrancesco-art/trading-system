import pandas as pd
import numpy as np

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Aggiunge colonne formattate per la visualizzazione (mantenendo le originali per i calcoli)
    df = df.copy()
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df["MarketCap"].apply(lambda x: f"{x/1e9:.2f}B" if x > 1e9 else f"{x/1e6:.2f}M")
    return df

def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    # Seleziona e riordina le colonne per la tabella finale
    cols = [c for c in ["Ticker", "Nome", "Prezzo", "MarketCap_fmt", "Stato", "RSI"] if c in df.columns]
    return df[cols]

def add_links(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Utilizzo dell'icona 🔗 e link HTML
    df["Yahoo"] = df["Ticker"].apply(lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">🔗 Apri</a>')
    df["TradingView"] = df["Ticker"].apply(lambda t: f'<a href="https://www.tradingview.com/chart/?symbol={t}" target="_blank">🔗 Apri</a>')
    return df
