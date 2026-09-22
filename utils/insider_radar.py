"""
utils/insider_radar.py — V45.09
Modulo Insider Radar: scarica, analizza e classifica le transazioni
insider (Form 4 SEC) per individuare segnali di acquisto significativi
da parte di CEO, CFO, director e azionisti con oltre il 10% delle quote.

Fonte dati: SEC EDGAR (https://www.sec.gov)
- Feed "latest filings" (Form 4) per il flusso quasi in tempo reale
- XML del singolo filing per i dettagli della transazione

Nessun database esterno richiesto: caching tramite st.cache_data(ttl=...),
in linea con il resto del progetto (macro_regime.py, scanner.py).

IMPORTANTE — Compliance SEC:
La SEC richiede un User-Agent che identifichi realmente il richiedente
(nome + email di contatto), altrimenti le richieste possono essere
bloccate con errore "Undeclared Automated Tool".
Limite di accesso SEC: massimo 10 richieste/secondo (qui throttled a ~6-7/s).
"""

import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st

# ── Configurazione ──────────────────────────────────────────────────────
SEC_USER_AGENT = "TradingSystemPersonale crimi.francesco@gmail.com"
SEC_BASE = "https://www.sec.gov"
HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

TRANSACTION_CODES = {
    "P": "Acquisto sul mercato",
    "S": "Vendita sul mercato",
    "A": "Assegnazione (Grant/Award)",
    "M": "Esercizio opzioni",
    "F": "Pagamento tasse (cessione titoli)",
    "G": "Donazione",
    "C": "Conversione derivato",
    "D": "Cessione a terzi",
    "X": "Esercizio diritti",
}

_MIN_REQUEST_INTERVAL = 0.15  # ~6-7 req/s, sotto il limite SEC di 10 req/s
_last_request_time = [0.0]


def _throttled_get(url, **kwargs):
    elapsed = time.time() - _last_request_time[0]
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    resp = requests.get(url, headers=HEADERS, timeout=15, **kwargs)
    _last_request_time[0] = time.time()
    return resp


def _extract_accession(index_url: str) -> str:
    """Estrae il numero di accession (identificativo univoco del filing)
    dall'URL della index page, indipendentemente dal CIK usato nel path."""
    m = re.search(r"(\d{10}-\d{2}-\d{6})", index_url)
    if m:
        return m.group(1)
    m = re.search(r"/(\d{18})[/-]", index_url)
    return m.group(1) if m else index_url


# ── Step 1: feed degli ultimi filing Form 4 ─────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_latest_form4_feed(count: int = 100) -> pd.DataFrame:
    """
    Recupera l'elenco degli ultimi filing Form 4 pubblicati su EDGAR
    (feed 'getcurrent', quasi in tempo reale), deduplicato per accession
    number (lo stesso filing compare più volte se coinvolge più CIK).
    Colonne: company_raw, company, filing_date, index_url, accession.
    """
    url = (
        f"{SEC_BASE}/cgi-bin/browse-edgar"
        f"?action=getcurrent&type=4&company=&dateb=&owner=include"
        f"&count={count}&output=atom"
    )
    try:
        resp = _throttled_get(url)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return pd.DataFrame()

    ns = {"a": "http://www.w3.org/2005/Atom"}
    rows = []
    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns) or ""
        link_el = entry.find("a:link", ns)
        link = link_el.get("href") if link_el is not None else ""
        updated = entry.findtext("a:updated", default="", namespaces=ns) or ""
        m = re.match(r"^(.*?)\s*\(", title)
        company = m.group(1).strip() if m else title
        if link:
            rows.append({
                "company_raw": title,
                "company": company,
                "filing_date": updated,
                "index_url": link,
                "accession": _extract_accession(link),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset="accession", keep="first").reset_index(drop=True)


def _find_xml_doc(index_url: str):
    """Trova l'URL del documento XML primario del Form 4 dalla index page."""
    try:
        resp = _throttled_get(index_url)
        resp.raise_for_status()
    except Exception:
        return None
    matches = re.findall(r'href="([^"]+\.xml)"', resp.text, flags=re.IGNORECASE)
    if not matches:
        return None
    for m in matches:
        low = m.lower()
        if "xslf345" in low:
            continue
        if "form4" in low or "doc4" in low or re.search(r"\d{18}\.xml$", low):
            return urljoin(index_url, m)
    return urljoin(index_url, matches[0])


def _text(el, path, default=""):
    if el is None:
        return default
    found = el.find(path)
    return found.text.strip() if found is not None and found.text else default


# ── Step 2: parsing del singolo Form 4 XML ──────────────────────────────
def parse_form4_xml(xml_bytes: bytes):
    """
    Effettua il parsing dello Form 4 XML e restituisce una lista di
    transazioni normalizzate (una riga per ogni transazione non-derivata).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    issuer = root.find("issuer")
    ticker = _text(issuer, "issuerTradingSymbol")
    issuer_name = _text(issuer, "issuerName")

    owner = root.find("reportingOwner")
    owner_name = _text(owner, "reportingOwnerId/rptOwnerName")
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    is_director = _text(rel, "isDirector") == "1"
    is_officer = _text(rel, "isOfficer") == "1"
    is_ten_pct = _text(rel, "isTenPercentOwner") == "1"
    officer_title = _text(rel, "officerTitle")

    role = "Director" if is_director else ""
    if is_officer:
        role = officer_title or "Officer"
    if is_ten_pct:
        role = (role + " / 10% Owner") if role else "10% Owner"
    role = role or "Other"

    has_footnote = root.find(".//footnoteId") is not None

    rows = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")
        tx_code = _text(tx, "transactionCoding/transactionCode")
        shares = _text(tx, "transactionAmounts/transactionShares/value")
        price = _text(tx, "transactionAmounts/transactionPricePerShare/value")
        date = _text(tx, "transactionDate/value")
        shares_after = _text(tx, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        direct = _text(tx, "ownershipNature/directOrIndirectOwnership/value")

        try:
            shares_f = float(shares) if shares else 0.0
            price_f = float(price) if price else 0.0
            shares_after_f = float(shares_after) if shares_after else None
        except ValueError:
            shares_f, price_f, shares_after_f = 0.0, 0.0, None

        rows.append({
            "Ticker": ticker,
            "Issuer": issuer_name,
            "Insider": owner_name,
            "Ruolo": role,
            "CodiceSEC": tx_code,
            "Tipo": TRANSACTION_CODES.get(tx_code, tx_code or "N/D"),
            "AcquistoDisposizione": code,
            "DataTransazione": date,
            "Azioni": shares_f,
            "Prezzo": price_f,
            "Valore": round(shares_f * price_f, 2),
            "AzioniPossedute": shares_after_f,
            "Diretto": direct,
            "Is10b5_1": 1 if has_footnote else 0,
        })
    return rows


# ── Step 3: pipeline completa ────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_recent_insider_transactions(max_filings: int = 60) -> pd.DataFrame:
    """
    Pipeline completa: recupera gli ultimi filing Form 4 (deduplicati per
    accession number), scarica e analizza ciascun XML, restituisce un
    DataFrame aggregato di transazioni insider normalizzate.
    """
    feed = fetch_latest_form4_feed(count=max_filings)
    if feed.empty:
        return pd.DataFrame()

    all_rows = []
    for _, row in feed.iterrows():
        xml_url = _find_xml_doc(row["index_url"])
        if not xml_url:
            continue
        try:
            resp = _throttled_get(xml_url)
            resp.raise_for_status()
        except Exception:
            continue
        parsed = parse_form4_xml(resp.content)
        for p in parsed:
            p["Accession"] = row["accession"]
            p["FilingIndex"] = row["index_url"]
            p["FilingDate"] = row["filing_date"]
            if not p.get("Issuer"):
                p["Issuer"] = row["company"]
        all_rows.extend(parsed)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df[df["Ticker"].astype(str).str.strip() != ""]
    dedup_cols = ["Accession", "Insider", "CodiceSEC", "DataTransazione", "Azioni", "Prezzo"]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    return df.reset_index(drop=True)


# ── Step 4: Insider Score e classificazione ──────────────────────────────
def compute_insider_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola un Insider Score (0-100) aggregato per ticker basandosi su:
    - valore netto degli acquisti sul mercato (40%)
    - numero di insider distinti coinvolti / cluster (25%)
    - presenza di ruoli chiave CEO/CFO (20%)
    - numero di transazioni ravvicinate (10%)
    - assenza di segnali automatici / footnote 10b5-1 (5%)
    """
    if df.empty:
        return pd.DataFrame()

    buys = df[df["CodiceSEC"] == "P"].copy()
    if buys.empty:
        return pd.DataFrame()

    agg = buys.groupby("Ticker").agg(
        Issuer=("Issuer", "first"),
        Valore_Netto=("Valore", "sum"),
        N_Insider=("Insider", "nunique"),
        N_Transazioni=("Insider", "count"),
        Is10b5_1_pct=("Is10b5_1", "mean"),
    ).reset_index()

    def _has_key_role(tkr):
        sub = buys.loc[buys["Ticker"] == tkr, "Ruolo"].str.upper()
        return sub.str.contains("CEO|CFO|CHIEF EXECUTIVE|CHIEF FINANCIAL", regex=True).any()

    agg["CEO_CFO_Coinvolto"] = agg["Ticker"].apply(_has_key_role)

    v_max = agg["Valore_Netto"].max() or 1
    n_max = agg["N_Insider"].max() or 1

    agg["Score_Valore"] = (agg["Valore_Netto"] / v_max * 40).clip(0, 40)
    agg["Score_Insider"] = (agg["N_Insider"] / n_max * 25).clip(0, 25)
    agg["Score_Ruolo"] = agg["CEO_CFO_Coinvolto"].map({True: 20, False: 5})
    agg["Score_Cluster"] = (agg["N_Transazioni"].clip(upper=5) / 5 * 10).clip(0, 10)
    agg["Score_Automatico"] = ((1 - agg["Is10b5_1_pct"]) * 5).clip(0, 5)

    agg["Insider_Score"] = (
        agg["Score_Valore"] + agg["Score_Insider"] + agg["Score_Ruolo"]
        + agg["Score_Cluster"] + agg["Score_Automatico"]
    ).round(1)

    def _classify(s):
        if s >= 75:
            return "🟢 STRONG INSIDER"
        if s >= 55:
            return "🟡 POSITIVE"
        if s >= 35:
            return "⚪ WATCH"
        return "⚫ NEUTRAL"

    agg["Livello"] = agg["Insider_Score"].apply(_classify)
    agg["Cluster"] = agg["N_Insider"] >= 2

    cols = [
        "Ticker", "Issuer", "Livello", "Insider_Score", "Valore_Netto",
        "N_Insider", "N_Transazioni", "CEO_CFO_Coinvolto", "Cluster",
    ]
    return agg[cols].sort_values("Insider_Score", ascending=False).reset_index(drop=True)
