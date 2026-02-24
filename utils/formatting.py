import pandas as pd
import numpy as np

def add_links(df: pd.DataFrame) -> pd.DataFrame:
    col = "Ticker" if "Ticker" in df.columns else "ticker"
    if col not in df.columns:
        return df
    # Formato HTML con icona per AgGrid
    df["Yahoo"] = df[col].apply(
        lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">🔗 Apri</a>'
    )
    df["TradingView"] = df[col].apply(
        lambda t: f'<a href="https://www.tradingview.com/chart/?symbol={str(t).split(".")[0]}" target="_blank">🔗 Apri</a>'
    )
    return df

# ... mantieni le altre funzioni (fmt_currency, prepare_display_df, ecc.) come nel file originale ...
