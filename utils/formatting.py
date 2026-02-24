import pandas as pd
import numpy as np

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df["MarketCap"].apply(lambda x: f"{x/1e9:.2f}B" if x > 1e9 else f"{x/1e6:.2f}M")
    return df

def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    # Mostra le colonne fondamentali
    cols = ["Ticker", "Nome", "Prezzo", "RSI", "Stato_Early", "Stato_Pro", "Stato"]
    return df[[c for c in cols if c in df.columns]]

def add_links(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Colonne con parola "Apri" e icona come richiesto
    df["Yahoo"] = df["Ticker"].apply(lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">🔗 Apri</a>')
    df["Tradingview"] = df["Ticker"].apply(lambda t: f'<a href="https://www.tradingview.com/chart/?symbol={t}" target="_blank">📈 Apri</a>')
    return df
