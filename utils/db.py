import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------------
# DB PATH — percorso assoluto stabile su Streamlit Cloud
# -------------------------------------------------------------------------
# Usa la directory del file db.py stesso come radice,
# così il path è sempre lo stesso indipendentemente dalla working directory.

DB_PATH = Path(__file__).resolve().parent.parent / "watchlist.db"


# -------------------------------------------------------------------------
# HELPER: serializzazione sicura di DataFrame con colonne complesse
# -------------------------------------------------------------------------

def _safe_df_to_json(df: pd.DataFrame) -> str:
    """
    Serializza un DataFrame in JSON escludendo le colonne con oggetti
    complessi (dict/list) che non sono serializzabili in modo stabile.
    Queste colonne (_chart_data, _quality_components) non servono
    nello storico — i grafici si rigenerano dallo scanner.
    """
    if df.empty:
        return "[]"
    cols_to_drop = [c for c in df.columns if c.startswith("_")]
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")

    # Converti colonne problematiche (bool numpy, NaN, inf)
    df_clean = df_clean.copy()
    for col in df_clean.columns:
        try:
            df_clean[col] = df_clean[col].apply(
                lambda x: bool(x) if isinstance(x, (np.bool_,)) else
                          float(x) if isinstance(x, (np.floating,)) else
                          int(x)   if isinstance(x, (np.integer,)) else
                          None     if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else
                          x
            )
        except Exception:
            pass

    try:
        return df_clean.to_json(orient="records", default_handler=str)
    except Exception:
        return "[]"


def _safe_json_to_df(json_str: str) -> pd.DataFrame:
    """Deserializza in modo sicuro, ritorna DataFrame vuoto in caso di errore."""
    if not json_str or json_str in ("[]", "null", ""):
        return pd.DataFrame()
    try:
        return pd.read_json(json_str, orient="records")
    except Exception:
        return pd.DataFrame()


# -------------------------------------------------------------------------
# INIT DB
# -------------------------------------------------------------------------

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Watchlist
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker     TEXT NOT NULL,
            name       TEXT,
            trend      TEXT,
            origine    TEXT,
            note       TEXT,
            list_name  TEXT DEFAULT 'DEFAULT',
            created_at TEXT
        )
    """)

    # Aggiungi colonne mancanti in modo sicuro (idempotente)
    existing = {row[1] for row in c.execute("PRAGMA table_info(watchlist)").fetchall()}
    for col_def, col_name in [
        ("trend TEXT",              "trend"),
        ("list_name TEXT",          "list_name"),
        ("origine TEXT",            "origine"),
        ("note TEXT",               "note"),
    ]:
        if col_name not in existing:
            try:
                c.execute(f"ALTER TABLE watchlist ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass

    # Storico scansioni
    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at   TEXT NOT NULL,
            markets      TEXT,
            n_early      INTEGER DEFAULT 0,
            n_pro        INTEGER DEFAULT 0,
            n_rea        INTEGER DEFAULT 0,
            n_confluence INTEGER DEFAULT 0,
            df_ep_json   TEXT,
            df_rea_json  TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------------------------------
# WATCHLIST
# -------------------------------------------------------------------------

def reset_watchlist_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()
    init_db()


def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    """Aggiunge ticker alla watchlist evitando duplicati per (ticker, list_name)."""
    if not tickers:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        # Controlla se esiste già
        existing = c.execute(
            "SELECT id FROM watchlist WHERE ticker=? AND list_name=?", (t, list_name)
        ).fetchone()
        if existing:
            # Aggiorna dati esistenti invece di duplicare
            c.execute("""
                UPDATE watchlist SET name=?, trend=?, origine=?, note=?, created_at=?
                WHERE ticker=? AND list_name=?
            """, (n, trend, origine, note, now, t, list_name))
        else:
            c.execute("""
                INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (t, n, trend, origine, note, list_name, now))
    conn.commit()
    conn.close()


def load_watchlist() -> pd.DataFrame:
    """Carica tutta la watchlist dal DB."""
    if not DB_PATH.exists():
        init_db()
        return pd.DataFrame(columns=[
            "id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"
        ])
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            "SELECT * FROM watchlist ORDER BY list_name, created_at DESC", conn
        )
        conn.close()
    except Exception:
        return pd.DataFrame(columns=[
            "id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"
        ])

    # Garantisce tutte le colonne attese
    for col in ["id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"]:
        if col not in df.columns:
            df[col] = "" if col != "id" else np.nan

    # Rinomina per compatibilità con il dashboard (ticker → Ticker, name → Nome)
    rename = {}
    if "ticker" in df.columns and "Ticker" not in df.columns:
        rename["ticker"] = "Ticker"
    if "name" in df.columns and "Nome" not in df.columns:
        rename["name"] = "Nome"
    if rename:
        df = df.rename(columns=rename)

    return df


def delete_from_watchlist(ids):
    if not ids:
        return
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.executemany("DELETE FROM watchlist WHERE id = ?", [(int(i),) for i in ids])
    conn.commit()
    conn.close()


def move_watchlist_rows(ids, dest_list):
    if not ids:
        return
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.executemany(
        "UPDATE watchlist SET list_name = ? WHERE id = ?",
        [(dest_list, int(i)) for i in ids]
    )
    conn.commit()
    conn.close()


def rename_watchlist(old_name, new_name):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "UPDATE watchlist SET list_name = ? WHERE list_name = ?", (new_name, old_name)
    )
    conn.commit()
    conn.close()


def reset_watchlist_by_name(list_name):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM watchlist WHERE list_name = ?", (list_name,))
    conn.commit()
    conn.close()


def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("UPDATE watchlist SET note = ? WHERE id = ?", (new_note, int(row_id)))
    conn.commit()
    conn.close()


# -------------------------------------------------------------------------
# STORICO SCANSIONI
# -------------------------------------------------------------------------

def save_scan_history(markets: list, df_ep: pd.DataFrame, df_rea: pd.DataFrame):
    """
    Salva i risultati della scansione nel DB.
    Le colonne _chart_data e _quality_components vengono escluse
    perché contengono oggetti Python non serializzabili in JSON stabile.
    """
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_early = int((df_ep.get("Stato_Early", pd.Series()) == "EARLY").sum()) \
                  if not df_ep.empty else 0
        n_pro   = int((df_ep.get("Stato_Pro",   pd.Series()) == "PRO"  ).sum()) \
                  if not df_ep.empty else 0
        n_rea   = len(df_rea) if not df_rea.empty else 0
        n_conf  = 0
        if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
            n_conf = int(
                ((df_ep["Stato_Early"] == "EARLY") & (df_ep["Stato_Pro"] == "PRO")).sum()
            )

        ep_json  = _safe_df_to_json(df_ep)
        rea_json = _safe_df_to_json(df_rea)

        c.execute("""
            INSERT INTO scan_history
                (scanned_at, markets, n_early, n_pro, n_rea, n_confluence, df_ep_json, df_rea_json)
            VALUES (?,?,?,?,?,?,?,?)
        """, (now, json.dumps(markets), n_early, n_pro, n_rea, n_conf, ep_json, rea_json))

        # Mantieni solo le ultime 50 scansioni per non far crescere il DB
        c.execute("""
            DELETE FROM scan_history WHERE id NOT IN (
                SELECT id FROM scan_history ORDER BY id DESC LIMIT 50
            )
        """)

        conn.commit()
        conn.close()
    except Exception as e:
        # Log esplicito invece di pass silenzioso
        import traceback
        print(f"[save_scan_history ERROR] {e}\n{traceback.format_exc()}")


def load_scan_history(limit: int = 20) -> pd.DataFrame:
    """Carica le ultime N scansioni (metadati, senza JSON dei dati)."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            """SELECT id, scanned_at, markets, n_early, n_pro, n_rea, n_confluence
               FROM scan_history ORDER BY id DESC LIMIT ?""",
            conn, params=(limit,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_scan_snapshot(scan_id: int):
    """Carica i DataFrame di una scansione specifica dal DB."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "SELECT df_ep_json, df_rea_json FROM scan_history WHERE id = ?", (int(scan_id),)
        )
        row = c.fetchone()
        conn.close()
        if row:
            return _safe_json_to_df(row[0]), _safe_json_to_df(row[1])
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()


# -------------------------------------------------------------------------
# Inizializza DB al caricamento del modulo
# -------------------------------------------------------------------------
init_db()
