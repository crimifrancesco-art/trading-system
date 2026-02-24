import pandas as pd
import numpy as np

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df["MarketCap"].apply(lambda x: f"{x/1e9:.2f}B" if x > 1e9 else f"{x/1e6:.2f}M")
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df["Prezzo"].map('{:.2f}'.format)
    return df

def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    # Seleziona le colonne desiderate per la tabella
    cols_to_show = ["Ticker", "Nome", "Prezzo", "RSI", "Stato"]
    existing = [c for c in cols_to_show if c in df.columns]
    return df[existing]

def add_links(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Link con icone e target blank per AgGrid
    df["Yahoo"] = df["Ticker"].apply(lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">🔗 Apri</a>')
    df["Tradingview"] = df["Ticker"].apply(lambda t: f'<a href="https://www.tradingview.com/chart/?symbol={t}" target="_blank">📈 Apri</a>')
    return df
