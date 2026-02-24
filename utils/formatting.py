import numpy as np
import pandas as pd


def _nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def fmt_currency(v, sym="$"):
    if _nan(v):
        return ""
    return f"{sym}{v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")


def fmt_int(v):
    if _nan(v):
        return ""
    return f"{int(v):,}".replace(",", ".")


def add_formatted_cols(df: pd.DataFrame):

    df = df.copy()

    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df["Prezzo"].apply(fmt_currency)

    if "Vol_Today" in df.columns:
        df["Vol_Today_fmt"] = df["Vol_Today"].apply(fmt_int)

    if "Vol_7d_Avg" in df.columns:
        df["Vol_7d_Avg_fmt"] = df["Vol_7d_Avg"].apply(fmt_int)

    return df


def prepare_display_df(df):

    priority = [
        "Ticker",
        "Nome",
        "Prezzo_fmt",
        "Vol_Today_fmt",
        "Vol_7d_Avg_fmt",
    ]

    ordered = [c for c in priority if c in df.columns]
    others = [c for c in df.columns if c not in ordered]

    return df[ordered + others]


def add_links(df):

    df = df.copy()

    if "Ticker" not in df.columns:
        return df

    df["Yahoo"] = df["Ticker"].apply(
        lambda t: f"https://finance.yahoo.com/quote/{t}"
    )

    df["TradingView"] = df["Ticker"].apply(
        lambda t: f"https://www.tradingview.com/chart/?symbol={t.split('.')[0]}"
    )

    return df
