"""
utils/insider_radar.py — V45.10 (Insider Radar Pro)
Modulo Insider Radar: scarica, analizza e classifica le transazioni
insider (Form 4 SEC) per individuare segnali di acquisto significativi
da parte di CEO, CFO, director e azionisti con oltre il 10% delle quote.

Novità V45.10:
- Mappatura ruoli più precisa (CEO, CFO, President, COO, Chairman, Director,
  10% Owner, Other), con normalizzazione dei titoli ufficiali.
- Insider Score a soglie fisse: il punteggio di un acquisto non dipende più
  dal campione analizzato (60, 150 o 300 filing), quindi è confrontabile
  tra Feed Rapido e Storico Esteso e nel tempo.
- Cluster buying con finestra temporale reale: richiede almeno 2 insider
  distinti che comprano entro N giorni (default 5), non semplicemente
  "più di un insider nel campione", indipendentemente da quando.

Due modalità di raccolta dati:
1. Feed rapido — 'getcurrent' atom feed: ultimi ~100 filing Form 4 in
   assoluto su tutto EDGAR (istantaneo, ma con mercati attivi copre
   solo poche ore).
2. Storico estesa — 'daily-index': indice giornaliero completo di ogni
   filing SEC. Permette di analizzare più giorni consecutivi (es. 7 o
   14 giorni) e quindi un campione realmente esteso, non ridotto.

Fonte dati: SEC EDGAR (https://www.sec.gov)
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
from datetime import datetime, timedelta
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
    """Estrae il numero di accession dall'URL della index page, indipendentemente dal CIK usato nel path."""
    m = re.search(r"(\d{10}-\d{2}-\d{6})", index_url)
    if m:
        return m.group(1)
    m = re.search(r"/(\d{18})[/-]", index_url)
    return m.group(1) if m else index_url


# ── Modalità 1: feed rapido (ultimi filing in assoluto) ─────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_latest_form4_feed(count: int = 100) -> pd.DataFrame:
    """
    Recupera l'elenco degli ultimi filing Form 4 pubblicati su EDGAR
    (feed 'getcurrent', quasi in tempo reale, max ~100 risultati totali).
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


# ── Modalità 2: storico estesa (daily-index, più giorni) ────────────────
def _quarter_of(dt: datetime) -> int:
    return (dt.month - 1) // 3 + 1


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_form4_daily_index(date_str: str) -> pd.DataFrame:
    """
    Scarica e analizza l'indice giornaliero SEC (form.YYYYMMDD.idx) per una
    singola data, filtrando solo i filing di tipo Form 4.
    date_str: formato 'YYYYMMDD'.
    Colonne: company, cik, filing_date, index_url, accession.
    """
    dt = datetime.strptime(date_str, "%Y%m%d")
    url = f"{SEC_BASE}/Archives/edgar/daily-index/{dt.year}/QTR{_quarter_of(dt)}/form.{date_str}.idx"
    try:
        resp = _throttled_get(url)
        if resp.status_code != 200:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    lines = resp.text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("---"):
            start_idx = i + 1
            break
    if start_idx is None:
        return pd.DataFrame()

    rows = []
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5:
            continue
        form_type, company, cik, date_filed, file_name = parts[:5]
        if form_type.strip() != "4":
            continue
        m = re.search(r"(\d{10}-\d{2}-\d{6})", file_name)
        accession = m.group(1) if m else None
        if not accession:
            continue
        index_url = f"{SEC_BASE}/Archives/edgar/data/{cik.strip()}/{accession}-index.htm"
        rows.append({
            "company": company.strip(),
            "cik": cik.strip(),
            "filing_date": date_filed.strip(),
            "index_url": index_url,
            "accession": accession,
        })
    return pd.DataFrame(rows)


def _business_days_back(n_days: int, end_date=None):
    """Restituisce le date (stringhe YYYYMMDD) degli ultimi n_days giorni lavorativi (lun-ven)."""
    if end_date is None:
        end_date = datetime.now()
    dates = []
    cursor = end_date
    while len(dates) < n_days:
        if cursor.weekday() < 5:  # 0=lunedì ... 4=venerdì
            dates.append(cursor.strftime("%Y%m%d"))
        cursor -= timedelta(days=1)
    return dates


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_form4_extended_feed(days_back: int = 7) -> pd.DataFrame:
    """
    Aggrega l'indice giornaliero SEC su più giorni lavorativi per ottenere
    un elenco esteso di filing Form 4 (non limitato ai soli ultimi ~100).
    Colonne: company, cik, filing_date, index_url, accession.
    """
    dates = _business_days_back(days_back)
    frames = []
    for d in dates:
        df_day = fetch_form4_daily_index(d)
        if not df_day.empty:
            frames.append(df_day)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
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


# ── V45.10: mappatura ruoli normalizzata ─────────────────────────────────
_ROLE_PATTERNS = [
    (re.compile(r"CHIEF\s+EXECUTIVE|^CEO$|\bCEO\b", re.I), "CEO"),
    (re.compile(r"CHIEF\s+FINANCIAL|^CFO$|\bCFO\b", re.I), "CFO"),
    (re.compile(r"CHIEF\s+OPERATING|^COO$|\bCOO\b", re.I), "COO"),
    (re.compile(r"PRESIDENT", re.I), "President"),
    (re.compile(r"CHAIRMAN|CHAIR\b", re.I), "Chairman"),
    (re.compile(r"CHIEF\s+\w+\s+OFFICER|^C[A-Z]O$", re.I), "Other Officer"),
]


def _normalize_role(is_director, is_officer, is_ten_pct, officer_title):
    """
    Normalizza il ruolo dell'insider in una categoria stabile:
    CEO, CFO, COO, President, Chairman, Other Officer, Director,
    10% Owner, Other. Un insider può avere più ruoli (es. CEO e Director);
    la priorità di visualizzazione va al ruolo più operativo/rilevante.
    """
    tags = []
    title_upper = str(officer_title or "").upper()

    if is_officer and officer_title:
        matched = False
        for pattern, label in _ROLE_PATTERNS:
            if pattern.search(title_upper):
                tags.append(label)
                matched = True
                break
        if not matched:
            tags.append("Other Officer")
    elif is_officer:
        tags.append("Other Officer")

    if is_director:
        tags.append("Director")
    if is_ten_pct:
        tags.append("10% Owner")

    if not tags:
        return "Other"

    priority = ["CEO", "CFO", "COO", "President", "Chairman", "Other Officer", "Director", "10% Owner"]
    tags_sorted = sorted(set(tags), key=lambda t: priority.index(t) if t in priority else 99)
    return " / ".join(tags_sorted)


def role_bucket(role_str: str) -> str:
    """Raggruppa il ruolo normalizzato in una macro-categoria per i filtri UI."""
    r = str(role_str).upper()
    if "CEO" in r or "CFO" in r:
        return "CEO/CFO"
    if "10% OWNER" in r:
        return "10% Owner"
    if "DIRECTOR" in r or "PRESIDENT" in r or "CHAIRMAN" in r or "COO" in r or "OFFICER" in r:
        return "Director/Officer"
    return "Altro"


# ── Parsing del singolo Form 4 XML ──────────────────────────────────────
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

    role = _normalize_role(is_director, is_officer, is_ten_pct, officer_title)

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


def _process_filing_list(feed: pd.DataFrame) -> pd.DataFrame:
    """Scarica e analizza l'XML di ciascun filing in 'feed', restituendo
    un DataFrame aggregato di transazioni normalizzate e deduplicate."""
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


@st.cache_data(ttl=600, show_spinner=False)
def fetch_recent_insider_transactions(max_filings: int = 60) -> pd.DataFrame:
    """
    Pipeline rapida: usa il feed 'getcurrent' (ultimi filing in assoluto),
    utile per un aggiornamento veloce ma limitato nel tempo.
    """
    feed = fetch_latest_form4_feed(count=max_filings)
    return _process_filing_list(feed)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_insider_transactions_extended(days_back: int = 7, max_filings: int = 500) -> pd.DataFrame:
    """
    Pipeline estesa: usa l'indice giornaliero SEC per analizzare più
    giorni consecutivi. Restituisce un campione realmente esteso di
    transazioni insider, non limitato agli ultimi ~100 filing assoluti.
    max_filings limita il numero di filing scaricati (per contenere i
    tempi di esecuzione: ogni filing richiede 2 richieste HTTP).
    """
    feed = fetch_form4_extended_feed(days_back=days_back)
    if feed.empty:
        return pd.DataFrame()
    if len(feed) > max_filings:
        feed = feed.sample(n=max_filings, random_state=42).reset_index(drop=True)
    return _process_filing_list(feed)


# ── V45.10: Insider Score a soglie fisse ─────────────────────────────────
def _score_valore_fisso(valore: float) -> float:
    """Punteggio (0-40) basato su soglie fisse di valore, non relative al
    campione. Così lo score è confrontabile tra Feed Rapido e Storico Esteso
    e nel tempo (lo stesso acquisto ottiene sempre lo stesso punteggio)."""
    if valore >= 1_000_000:
        return 40.0
    if valore >= 500_000:
        return 32.0
    if valore >= 250_000:
        return 26.0
    if valore >= 100_000:
        return 20.0
    if valore >= 25_000:
        return 10.0
    return 4.0


def _score_ruolo_fisso(role_bucket_val: str) -> float:
    if role_bucket_val == "CEO/CFO":
        return 20.0
    if role_bucket_val == "Director/Officer":
        return 12.0
    if role_bucket_val == "10% Owner":
        return 10.0
    return 4.0


def _detect_cluster(buys: pd.DataFrame, window_days: int = 5):
    """
    Individua per ciascun ticker se esiste un cluster di acquisti:
    almeno 2 insider distinti che comprano entro 'window_days' giorni
    l'uno dall'altro (non semplicemente "più di un insider nel campione
    complessivo", indipendentemente da quando hanno comprato).
    Restituisce un dict {ticker: (is_cluster: bool, n_insider_in_window: int)}.
    """
    result = {}
    buys = buys.copy()
    buys["_data_dt"] = pd.to_datetime(buys["DataTransazione"], errors="coerce")

    for tkr, grp in buys.groupby("Ticker"):
        grp = grp.dropna(subset=["_data_dt"]).sort_values("_data_dt")
        if grp.empty:
            result[tkr] = (False, grp["Insider"].nunique())
            continue

        best_n = 1
        dates = grp["_data_dt"].tolist()
        insiders = grp["Insider"].tolist()
        for i in range(len(dates)):
            window_end = dates[i] + timedelta(days=window_days)
            in_window = {
                insiders[j] for j in range(len(dates))
                if dates[i] <= dates[j] <= window_end
            }
            best_n = max(best_n, len(in_window))

        result[tkr] = (best_n >= 2, best_n)

    return result


def compute_insider_score(df: pd.DataFrame, cluster_window_days: int = 5) -> pd.DataFrame:
    """
    Calcola un Insider Score (0-100) aggregato per ticker basandosi su:
    - valore netto degli acquisti sul mercato, a soglie fisse (40%)
    - ruolo chiave CEO/CFO > Director/Officer > 10% Owner > Other (20%)
    - cluster buying con finestra temporale reale (25%)
    - numero di transazioni (10%)
    - assenza di segnali automatici / footnote 10b5-1 (5%)

    A differenza delle versioni precedenti, il punteggio NON dipende dal
    numero di filing analizzati nel campione: lo stesso acquisto riceve
    sempre lo stesso punteggio, sia nel Feed Rapido che nello Storico Esteso.
    """
    if df.empty:
        return pd.DataFrame()

    buys = df[df["CodiceSEC"] == "P"].copy()
    if buys.empty:
        return pd.DataFrame()

    buys["RuoloGruppo"] = buys["Ruolo"].apply(role_bucket)
    cluster_info = _detect_cluster(buys, window_days=cluster_window_days)

    agg = buys.groupby("Ticker").agg(
        Issuer=("Issuer", "first"),
        Valore_Netto=("Valore", "sum"),
        Valore_Massimo=("Valore", "max"),
        N_Insider=("Insider", "nunique"),
        N_Transazioni=("Insider", "count"),
        Is10b5_1_pct=("Is10b5_1", "mean"),
    ).reset_index()

    def _top_role(tkr):
        sub = buys.loc[buys["Ticker"] == tkr, "RuoloGruppo"]
        priority = ["CEO/CFO", "Director/Officer", "10% Owner", "Altro"]
        present = [r for r in priority if r in sub.values]
        return present[0] if present else "Altro"

    agg["RuoloTop"] = agg["Ticker"].apply(_top_role)
    agg["CEO_CFO_Coinvolto"] = agg["RuoloTop"] == "CEO/CFO"
    agg["Cluster"] = agg["Ticker"].map(lambda t: cluster_info.get(t, (False, 1))[0])
    agg["N_Insider_Cluster"] = agg["Ticker"].map(lambda t: cluster_info.get(t, (False, 1))[1])

    agg["Score_Valore"] = agg["Valore_Massimo"].apply(_score_valore_fisso)
    agg["Score_Ruolo"] = agg["RuoloTop"].apply(_score_ruolo_fisso)
    agg["Score_Cluster"] = agg["N_Insider_Cluster"].apply(
        lambda n: 25.0 if n >= 3 else (15.0 if n >= 2 else 5.0)
    )
    agg["Score_Transazioni"] = (agg["N_Transazioni"].clip(upper=5) / 5 * 10).clip(0, 10)
    agg["Score_Automatico"] = ((1 - agg["Is10b5_1_pct"]) * 5).clip(0, 5)

    agg["Insider_Score"] = (
        agg["Score_Valore"] + agg["Score_Ruolo"] + agg["Score_Cluster"]
        + agg["Score_Transazioni"] + agg["Score_Automatico"]
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

    cols = [
        "Ticker", "Issuer", "Livello", "Insider_Score", "Valore_Netto",
        "RuoloTop", "N_Insider", "N_Insider_Cluster", "N_Transazioni",
        "CEO_CFO_Coinvolto", "Cluster",
    ]
    return agg[cols].sort_values("Insider_Score", ascending=False).reset_index(drop=True)
