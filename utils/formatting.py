import locale
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# LOCALE
# -----------------------------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _is_nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


# -----------------------------------------------------------------------------
# FORMATTERS
# -----------------------------------------------------------------------------
def fmt_currency(value, symbol="€"):
    if _is_nan(value):
        return ""

    return (
        f"{symbol}{float(value):,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def fmt_int(value):
    if _is_nan(value):
        return ""
    return f"{int(value):,}".replace(",", ".")


def fmt_marketcap(value, symbol="€"):
    if _is_nan(value):
        return ""

    v = float(value)

    if v >= 1_000_000_000:
        txt = f"{v/1_000_000_000:,.2f}B"
    elif v >= 1_000_000:
        txt = f"{v/1_000_000:,.2f}M"
    elif v >= 1_000:
        txt = f"{v/1_000:,.2f}K"
    else:
        return fmt_currency(v, symbol)

    return f"{symbol}{txt}".replace(",", "X").replace(".", ",").replace("X", ".")


# -----------------------------------------------------------------------------
# FORMATTAZIONE COLONNE
# -----------------------------------------------------------------------------
def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Currency" not in df.columns:
        df["Currency"] = "USD"

    # Prezzo
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df.apply(
            lambda r: fmt_currency(
                r["Prezzo"],
                "€" if r["Currency"] == "EUR" else "$"
            ),
            axis=1
        )

    # MarketCap
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df.apply(
            lambda r: fmt_marketcap(
                r["MarketCap"],
                "€" if r["Currency"] == "EUR" else "$"
            ),
            axis=1
        )

    if "Vol_Today" in df.columns:
        df["Vol_Today_fmt"] = df["Vol_Today"].apply(fmt_int)

    if "Vol_7d_Avg" in df.columns:
        df["Vol_7d_Avg_fmt"] = df["Vol_7d_Avg"].apply(fmt_int)

    return df


# -----------------------------------------------------------------------------
# PREPARAZIONE DISPLAY (AGGRID SAFE)
# -----------------------------------------------------------------------------
_COLS_TO_HIDE = [
    "Currency",
    "Stato_Early",
    "Stato_Pro",
]


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara dataframe per AgGrid senza rompere ordinamenti.
    Mantiene colonne RAW ma nasconde quelle inutili.
    """

    df = df.copy()

    # elimina solo colonne tecniche
    drop_cols = [c for c in _COLS_TO_HIDE if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # riordino intelligente
    priority = [
        "Ticker",
        "Nome",
        "Prezzo_fmt",
        "MarketCap_fmt",
        "Vol_Today_fmt",
        "Vol_7d_Avg_fmt",
    ]

    ordered = [c for c in priority if c in df.columns]
    others = [c for c in df.columns if c not in ordered]

    return df[ordered + others]


# -----------------------------------------------------------------------------
# LINK (AGGRID COMPATIBLE)
# -----------------------------------------------------------------------------
def add_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANTE:
    NON inseriamo HTML.
    Restituiamo URL puliti -> renderer JS farà il bottone "Apri".
    """

    df = df.copy()

    col = "Ticker" if "Ticker" in df.columns else "ticker"

    if col not in df.columns:
        return df

    df["Yahoo"] = df[col].astype(str).apply(
        lambda t: f"https://finance.yahoo.com/quote/{t}"
    )

    df["TradingView"] = df[col].astype(str).apply(
        lambda t: f"https://www.tradingview.com/chart/?symbol={t.split('.')[0]}"
    )

    return df
