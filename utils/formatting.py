import locale
import numpy as np
import pandas as pd

try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass


def fmt_currency(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return (
        f"{symbol}{value:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def fmt_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{int(value):,}".replace(",", ".")


def fmt_marketcap(value, symbol="€"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    v = float(value)
    if v >= 1_000_000_000:
        return (
            f"{symbol}{v / 1_000_000_000:,.2f}B"
            .replace(",", "X").replace(".", ",").replace("X", ".")
        )
    if v >= 1_000_000:
        return (
            f"{symbol}{v / 1_000_000:,.2f}M"
            .replace(",", "X").replace(".", ",").replace("X", ".")
        )
    if v >= 1_000:
        return (
            f"{symbol}{v / 1_000:,.2f}K"
            .replace(",", "X").replace(".", ",").replace("X", ".")
        )
    return fmt_currency(v, symbol)


# Colonne raw da eliminare dalla visualizzazione
_COLS_TO_DROP = [
    "MarketCap", "Vol_Today", "Vol_7d_Avg",
    "Currency", "Stato_Early", "Stato_Pro", "Prezzo_fmt",
]


def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Currency" not in df.columns:
        df["Currency"] = "USD"
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df.apply(
            lambda r: fmt_currency(r["Prezzo"], "\u20ac" if r["Currency"] == "EUR" else "$"),
            axis=1
        )
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df.apply(
            lambda r: fmt_marketcap(r["MarketCap"], "\u20ac" if r["Currency"] == "EUR" else "$"),
            axis=1
        )
    if "Vol_Today" in df.columns:
        df["Vol_Today_fmt"] = df["Vol_Today"].apply(fmt_int)
    if "Vol_7d_Avg" in df.columns:
        df["Vol_7d_Avg_fmt"] = df["Vol_7d_Avg"].apply(fmt_int)
    return df


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Riordina e pulisce il DataFrame per la visualizzazione:
    - Rimuove colonne raw (MarketCap, Vol_Today, Vol_7d_Avg, Currency, Stato_Early, Stato_Pro, Prezzo_fmt)
    - Sposta MarketCap_fmt, Vol_Today_fmt, Vol_7d_Avg_fmt in posizione 3, 4, 5
    """
    df = df.copy()

    # Elimina colonne indesiderate
    cols_drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_drop)

    # Rinomina Prezzo_fmt -> Prezzo se presente (sostituisce la colonna raw)
    # (Prezzo_fmt e' gia' stato eliminato sopra, usiamo la colonna Prezzo raw
    #  che va sostituita con il valore formattato)
    # Nota: Prezzo raw rimane per ordinamento ma lo rinominiamo in display
    # Rinomina per display
    rename_map = {}
    if "Prezzo" in df.columns and "MarketCap_fmt" in df.columns:
        # Prezzo e' ancora grezzo: lo formattiamo al volo per il display
        pass  # viene gestito tramite add_formatted_cols prima

    # Riordina: metti MarketCap_fmt, Vol_Today_fmt, Vol_7d_Avg_fmt in pos 3,4,5
    cols = list(df.columns)
    insert_cols = [c for c in ["MarketCap_fmt", "Vol_Today_fmt", "Vol_7d_Avg_fmt"] if c in cols]
    remaining = [c for c in cols if c not in insert_cols]

    # Costruisce l'ordine finale: prime 2 colonne + insert_cols + resto
    prefix = remaining[:2]
    suffix = remaining[2:]
    new_order = prefix + insert_cols + suffix
    df = df[new_order]

    return df


def add_links(df: pd.DataFrame) -> pd.DataFrame:
    col = "Ticker" if "Ticker" in df.columns else "ticker"
    if col not in df.columns:
        return df
    df["Yahoo"] = df[col].astype(str).apply(
        lambda t: f'<a href="https://finance.yahoo.com/quote/{t}" target="_blank">Apri</a>'
    )
    df["TradingView"] = df[col].astype(str).apply(
        lambda t: f'<a href="https://www.tradingview.com/chart/?symbol={t.split(".")[0]}" target="_blank">Apri</a>'
    )
    return df
