from __future__ import annotations

import io
import re
import unicodedata
from typing import Iterable

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# Mappatura manuale SOLO per casi ambigui che il parser automatico non può
# dedurre da "Strumento"/"Titolo". Identificata per ISIN quando possibile.
# Questa mappa NON deve essere la fonte primaria dei titoli: serve solo
# a correggere/arricchire la classificazione automatica.
# ─────────────────────────────────────────────────────────────────────────
_MANUAL_OVERRIDES: dict[str, dict[str, str]] = {
    # "IE00B4L5Y983": {"Asset_Class": "ETF", "Settore": "Azionario diversificato"},
}

_SYMBOL_OVERRIDES = {
    "nvidia": "NASDAQ:NVDA",
    "alphabet": "NASDAQ:GOOGL",
    "alphabet-a": "NASDAQ:GOOGL",
    "amazon": "NASDAQ:AMZN",
    "amazon.com": "NASDAQ:AMZN",
    "microsoft": "NASDAQ:MSFT",
    "apple": "NASDAQ:AAPL",
    "generali": "MIL:G",
    "wisdomtree physical bitcoin": "XETR:WBIT",
    "vanguard ftse all-world": "XETR:VWCE",
    "ishares physical gold": "MIL:SGLD",
    "ishares physical gold etc": "MIL:SGLD",
    "amundi core nasdaq-100 swap ucits etf acc": "XETR:LYMS",
    "amundi is s&p 500 swap ucits etf eur acc": "XETR:AUM5",
    "amundi stoxx europe 600 utilities ucits etf acc": "XETR:LUTI",
}

# Colonne del portafoglio DETTAGLIATO Fineco (con quantità, prezzi, valori).
_FINECO_COLUMNS = {
    "isin": [
        "isin",
    ],
    "ticker": [
        "simbolo",
        "ticker",
        "symbol",
    ],
    "market": [
        "mercato",
        "exchange",
    ],
    "instrument": [
        "strumento",
        "tipo strumento",
    ],
    "currency": [
        "valuta",
        "currency",
    ],
    "quantity": [
        "quantita",
        "q.ta",
        "qta",
    ],
    "avg_cost": [
        "p.zo medio di carico",
        "prezzo medio di carico",
        "pm carico",
    ],
    "market_price": [
        "p.zo di mercato",
        "prezzo di mercato",
        "p.mkt",
    ],
    "book_value": [
        "valore di carico",
    ],
    "market_value": [
        "valore di mercato",
        "val mercato",
    ],
    "return_pct": [
        "var%",
        "var %",
    ],
    "return_eur": [
        "var",
    ],
    "return_currency": [
        "var in valuta",
    ],
    "accrued": [
        "rateo",
    ],
}


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "").strip())


def _normalize_name(value) -> str:
    text = _clean(value).lower().replace("®", "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(".", "").replace("%", "pct")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_column(columns: Iterable[str], candidates: Iterable[str]):
    normalized = {_normalize_name(c): c for c in columns}

    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    for column in columns:
        column_key = _normalize_name(column)
        for candidate in candidates:
            if _normalize_name(candidate) in column_key:
                return column

    return None


def _map_fineco_columns(columns: Iterable[str]) -> dict[str, str]:
    """Ritorna {campo_logico: nome_colonna_reale} per le colonne trovate."""
    mapped: dict[str, str] = {}
    for logical, candidates in _FINECO_COLUMNS.items():
        found = _find_column(columns, candidates)
        if found:
            mapped[logical] = found
    return mapped


def _read_upload(uploaded_file) -> pd.DataFrame:
    """
    Legge l'export caricato dall'utente. Prova prima a leggerlo come
    tabella reale (CSV/Excel). Solo se non produce nulla di utile,
    ricade sul parser testuale del vecchio "portafoglio di sintesi".
    """
    filename = str(getattr(uploaded_file, "name", "")).lower()
    raw = uploaded_file.getvalue()

    frame = pd.DataFrame()

    if filename.endswith(".csv") or filename.endswith(".txt"):
        for separator in [",", ";", "\t", "|"]:
            try:
                candidate = pd.read_csv(io.BytesIO(raw), sep=separator)
                if candidate.shape[1] > 1:
                    frame = candidate
                    break
            except Exception:
                continue

        if frame.empty:
            try:
                frame = pd.read_csv(io.BytesIO(raw))
            except Exception:
                frame = pd.DataFrame()

    elif filename.endswith(".xls"):
        try:
            candidate = pd.read_excel(io.BytesIO(raw), engine="xlrd")
            if candidate is not None and not candidate.empty:
                frame = candidate
        except Exception:
            frame = pd.DataFrame()

    else:
        try:
            candidate = pd.read_excel(io.BytesIO(raw))
            if candidate is not None and not candidate.empty:
                frame = candidate
        except Exception:
            frame = pd.DataFrame()

    if frame is not None and not frame.empty:
        mapped = _map_fineco_columns(frame.columns)
        # Se troviamo ISIN o Quantità o Valore di mercato, è un export
        # dettagliato utilizzabile: lo ritorniamo subito.
        if any(k in mapped for k in ("isin", "quantity", "market_value")):
            return frame
        # Altrimenti potrebbe comunque essere una tabella valida con
        # intestazioni diverse (es. "Titolo","Nome"...): la ritorniamo
        # e lasciamo decidere a normalize_portfolio.
        if frame.shape[1] > 1:
            return frame

    # Fallback: file di riepilogo testuale (vecchio formato "sintesi").
    fallback = _read_fineco_summary(raw)
    if fallback is not None and not fallback.empty:
        return fallback

    return frame if frame is not None else pd.DataFrame()


def _read_fineco_summary(raw: bytes) -> pd.DataFrame:
    """Parser per il vecchio export 'Portafoglio di sintesi' (solo nomi)."""
    decoded_variants = []
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            decoded_variants.append(raw.decode(encoding, errors="ignore"))
        except Exception:
            pass

    text = "\n".join(decoded_variants)
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"<[^>]+>", " ", line)
        line = line.replace("\x00", " ")

        parts = re.split(r"[|;\t]", line)
        parts = [_clean(part.strip("# ")) for part in parts]
        parts = [part for part in parts if part]

        if parts:
            rows.append(parts)

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    frame = pd.DataFrame([row + [""] * (width - len(row)) for row in rows])

    # Rimuove righe di intestazione/rumore tipiche dell'export Fineco.
    def _is_noise(value: str) -> bool:
        norm = _normalize_name(value)
        return norm in {
            "titolo",
            "totale",
            "eur",
            "portafoglio di sintesi",
            "--",
        } or "finecobank" in norm or value.startswith("&")

    mask = ~frame.apply(
        lambda row: all(_is_noise(v) or not _clean(v) for v in row),
        axis=1,
    )
    frame = frame[mask]

    # Elimina singole celle di rumore mantenendo la riga se ha altro contenuto.
    frame = frame.map(lambda v: "" if _is_noise(str(v)) else v)  # type: ignore

    if frame.shape[1] == 1:
        frame.columns = ["Titolo"]
    return frame.reset_index(drop=True)


def _infer_asset_class(name: str, instrument: str = "") -> str:
    text = _normalize_name(f"{name} {instrument}")

    if any(token in text for token in ["bitcoin", "crypto", "ethereum"]):
        return "Crypto"
    if any(token in text for token in ["gold", "oro", "physical"]):
        return "Oro/Commodities"
    if any(token in text for token in [
        "bond", "obbligaz", "treasury", "fixed income", "obbligazione"
    ]):
        return "Obbligazionario"
    if any(token in text for token in [
        "certificate", "certificato", "structured", "strutturato", "call", "put"
    ]):
        return "Strutturato"
    if any(token in text for token in [
        "etf", "ishares", "amundi", "vanguard", "wisdomtree", "etc", "etp"
    ]):
        return "ETF"
    if any(token in text for token in ["fondo", "fund", "sicav"]):
        return "Fondo"
    return "Azionario"


def _infer_sector(name: str, asset_class: str) -> str:
    text = _normalize_name(name)

    if asset_class in {"Crypto", "Oro/Commodities", "Obbligazionario", "Strutturato", "Fondo"}:
        return asset_class
    if any(token in text for token in [
        "nvidia", "microsoft", "apple", "amazon", "alphabet", "nasdaq", "meta", "google"
    ]):
        return "Tecnologia/Mega-cap"
    if "utilit" in text:
        return "Utilities"
    if any(token in text for token in ["sp 500", "s&p", "all-world", "world", "msci"]):
        return "Azionario diversificato"
    return "Da classificare"


def _resolve_position_id(row: dict) -> str:
    """ISIN > Mercato:Simbolo > Simbolo > Nome — in questo ordine di priorità."""
    isin = _clean(row.get("ISIN", ""))
    if isin:
        return f"ISIN:{isin.upper()}"

    market = _clean(row.get("Mercato", ""))
    ticker = _clean(row.get("Ticker", ""))
    if ticker:
        return f"SYM:{market.upper()}:{ticker.upper()}" if market else f"SYM:{ticker.upper()}"

    name = _clean(row.get("Nome", ""))
    return f"NAME:{_normalize_name(name)}"


def normalize_portfolio(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza qualunque export (dettagliato o di sintesi) in uno schema
    unico. Non assume un elenco fisso di titoli: legge sempre le colonne
    presenti nel file caricato, quindi supporta portafogli che cambiano
    completamente nel tempo.
    """
    columns_out = [
        "Position_ID", "Nome", "ISIN", "Ticker", "Mercato", "Strumento",
        "Valuta", "Quantità", "Prezzo_medio_carico", "Valore_carico",
        "Prezzo_mercato", "Valore_mercato", "Var_pct", "Var_eur",
        "Var_valuta", "Rateo", "Peso_%", "Asset_Class", "Settore", "Azione",
    ]

    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns_out)

    source = frame.copy()
    source.columns = [_clean(c) for c in source.columns]

    mapped = _map_fineco_columns(source.columns)

    name_col = _find_column(
        source.columns,
        ["Titolo", "Nome", "Descrizione", "Strumento", "Instrument", "Security"],
    )

    result = pd.DataFrame(index=source.index)
    result["Nome"] = source[name_col].map(_clean) if name_col else ""

    result["ISIN"] = source[mapped["isin"]].map(_clean) if "isin" in mapped else ""
    result["Ticker"] = source[mapped["ticker"]].map(_clean) if "ticker" in mapped else ""
    result["Mercato"] = source[mapped["market"]].map(_clean) if "market" in mapped else ""
    result["Strumento"] = source[mapped["instrument"]].map(_clean) if "instrument" in mapped else ""
    result["Valuta"] = source[mapped["currency"]].map(_clean) if "currency" in mapped else ""

    def _parse_number(value):
        if value is None or pd.isna(value):
            return float("nan")

        if isinstance(value, (int, float)):
            return float(value)

        value = str(value).strip()
        if not value:
            return float("nan")

        value = (
            value.replace("€", "")
            .replace("$", "")
            .replace("£", "")
            .replace("%", "")
            .replace("\u00a0", "")
            .strip()
        )

        # Gestione formato italiano: 1.234,56
        if "," in value and "." in value:
            if value.rfind(",") > value.rfind("."):
                value = value.replace(".", "").replace(",", ".")
            else:
                value = value.replace(",", "")
        elif "," in value:
            value = value.replace(",", ".")
        else:
            # Mantiene il punto come separatore decimale.
            value = value.replace(" ", "")

        try:
            return float(value)
        except ValueError:
            return float("nan")

    def _num(col_key):
        if col_key not in mapped:
            return pd.Series(
                [float("nan")] * len(source),
                index=source.index,
                dtype="float64",
            )

        return source[mapped[col_key]].map(_parse_number).astype("float64")

    result["Quantità"] = _num("quantity")
    result["Prezzo_medio_carico"] = _num("avg_cost")
    result["Valore_carico"] = _num("book_value")
    result["Prezzo_mercato"] = _num("market_price")
    result["Valore_mercato"] = _num("market_value")
    result["Var_pct"] = _num("return_pct")
    result["Var_eur"] = _num("return_eur")
    result["Var_valuta"] = _num("return_currency")
    result["Rateo"] = _num("accrued")

    # Se manca il valore di mercato ma abbiamo quantità e prezzo, calcoliamolo.
    needs_calc = result["Valore_mercato"].isna() & result["Quantità"].notna() & result["Prezzo_mercato"].notna()
    result.loc[needs_calc, "Valore_mercato"] = (
        result.loc[needs_calc, "Quantità"] * result.loc[needs_calc, "Prezzo_mercato"]
    )

    # Ticker di fallback per nomi noti, solo se il file non fornisce ISIN/Simbolo.
    missing_ticker = result["Ticker"].eq("") & result["ISIN"].eq("")
    result.loc[missing_ticker, "Ticker"] = result.loc[missing_ticker, "Nome"].map(
        lambda n: _SYMBOL_OVERRIDES.get(_normalize_name(n), "")
    )

    result = result[result["Nome"].ne("") | result["ISIN"].ne("") | result["Ticker"].ne("")]
    result = result.reset_index(drop=True)

    result["Valore_mercato"] = pd.to_numeric(result["Valore_mercato"], errors="coerce").fillna(0.0)

    result["Asset_Class"] = [
        _infer_asset_class(n, s) for n, s in zip(result["Nome"], result["Strumento"])
    ]
    result["Settore"] = [
        _infer_sector(n, a) for n, a in zip(result["Nome"], result["Asset_Class"])
    ]

    # Applica override manuali per ISIN, se presenti.
    for idx, isin in result["ISIN"].items():
        key = _clean(isin).upper()
        if key in _MANUAL_OVERRIDES:
            for field, value in _MANUAL_OVERRIDES[key].items():
                result.at[idx, field] = value

    total = float(result["Valore_mercato"].sum())
    result["Peso_%"] = (
        (result["Valore_mercato"] / total * 100).round(2) if total > 0 else 0.0
    )

    result["Position_ID"] = [
        _resolve_position_id(row) for row in result.to_dict("records")
    ]

    result["Azione"] = "Mantenere"

    return result[columns_out].reset_index(drop=True)


def has_market_values(portfolio: pd.DataFrame) -> bool:
    if portfolio is None or portfolio.empty:
        return False
    return float(pd.to_numeric(portfolio["Valore_mercato"], errors="coerce").fillna(0).sum()) > 0


def diff_portfolio_snapshots(previous_ids: set[str], current_ids: set[str]) -> dict[str, set[str]]:
    """Confronta due snapshot per Position_ID e segnala titoli nuovi/usciti."""
    return {
        "nuovi": current_ids - previous_ids,
        "usciti": previous_ids - current_ids,
        "invariati": current_ids & previous_ids,
    }


def build_rebalance(
    portfolio: pd.DataFrame,
    targets: dict[str, float],
) -> pd.DataFrame:
    """
    Calcola il riequilibrio per asset class.

    Scostamento_pp:
        peso attuale meno target.

    Importo_teorico:
        valore target meno valore attuale.
        Positivo = capitale teoricamente da aggiungere.
        Negativo = capitale teoricamente da ridurre.
    """
    if portfolio is None or portfolio.empty:
        return pd.DataFrame()

    data = portfolio.copy()

    if "Valore_mercato" not in data.columns:
        data["Valore_mercato"] = 0.0

    data["Valore_mercato"] = pd.to_numeric(
        data["Valore_mercato"],
        errors="coerce",
    ).fillna(0.0)

    total_value = float(data["Valore_mercato"].sum())

    if total_value > 0:
        data["Peso_%"] = (
            data["Valore_mercato"] / total_value * 100
        )
    elif "Peso_%" not in data.columns:
        data["Peso_%"] = 0.0

    grouped = (
        data.groupby("Asset_Class", as_index=False)
        .agg(
            Peso_attuale=("Peso_%", "sum"),
            Valore_attuale=("Valore_mercato", "sum"),
        )
    )

    grouped["Target_%"] = (
        grouped["Asset_Class"]
        .map(targets)
        .fillna(0.0)
    )

    grouped["Scostamento_pp"] = (
        grouped["Peso_attuale"] - grouped["Target_%"]
    ).round(2)

    grouped["Valore_target"] = (
        total_value * grouped["Target_%"] / 100
    ).round(2)

    grouped["Importo_teorico"] = (
        grouped["Valore_target"] - grouped["Valore_attuale"]
    ).round(2)

    def _action(value):
        if value >= 7:
            return "🔴 Ridurre — priorità alta"
        if value >= 3:
            return "🟠 Ridurre gradualmente"
        if value <= -7:
            return "🟢 Aumentare — priorità alta"
        if value <= -3:
            return "🔵 Aumentare gradualmente"
        return "✅ In linea"

    def _priority(value):
        value = abs(float(value))
        if value >= 7:
            return "Alta"
        if value >= 3:
            return "Media"
        return "Bassa"

    grouped["Azione"] = grouped["Scostamento_pp"].map(_action)
    grouped["Priorità"] = grouped["Scostamento_pp"].map(_priority)

    columns = [
        "Asset_Class",
        "Peso_attuale",
        "Target_%",
        "Scostamento_pp",
        "Valore_attuale",
        "Valore_target",
        "Importo_teorico",
        "Priorità",
        "Azione",
    ]

    return (
        grouped[columns]
        .sort_values(
            "Scostamento_pp",
            key=lambda column: column.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )



def portfolio_to_tradingview(portfolio: pd.DataFrame) -> str:
    if portfolio is None or portfolio.empty:
        return "###PORTAFOGLIO,"

    tickers = []
    for _, row in portfolio.iterrows():
        market = _clean(row.get("Mercato", ""))
        ticker = _clean(row.get("Ticker", ""))
        if ticker and ":" not in ticker and market:
            tickers.append(f"{market.upper()}:{ticker.upper()}")
        elif ticker:
            tickers.append(ticker)

    return "###PORTAFOGLIO," + ",".join(tickers) + ","


def portfolio_to_csv(portfolio: pd.DataFrame) -> bytes:
    return portfolio.to_csv(index=False).encode("utf-8")


def portfolio_to_xlsx(portfolio: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        portfolio.to_excel(writer, index=False, sheet_name="Portafoglio")
    return output.getvalue()
