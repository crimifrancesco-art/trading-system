import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path("watchlist.db")


# -------------------------------------------------------------------------
# INIT DB
# -------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Watchlist
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            name TEXT,
            trend TEXT,
            origine TEXT,
            note TEXT,
            list_name TEXT,
            created_at TEXT
        )
    """)
    for col_def in ["trend TEXT", "list_name TEXT"]:
        try:
            c.execute(f"ALTER TABLE watchlist ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    # Storico scansioni
    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at TEXT NOT NULL,
            markets TEXT,
            n_early INTEGER DEFAULT 0,
            n_pro INTEGER DEFAULT 0,
            n_rea INTEGER DEFAULT 0,
            n_confluence INTEGER DEFAULT 0,
            df_ep_json TEXT,
            df_rea_json TEXT,
            elapsed_s REAL,
            cache_hits INTEGER DEFAULT 0
        )
    """)
    # Aggiungi colonne nuove se mancanti (retrocompatibilita')
    for col_def in ["elapsed_s REAL", "cache_hits INTEGER DEFAULT 0"]:
        try:
            c.execute(f"ALTER TABLE scan_history ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


# -------------------------------------------------------------------------
# WATCHLIST
# -------------------------------------------------------------------------

def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()
    init_db()


def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute("""
            INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (t, n, trend, origine, note, list_name, now))
    conn.commit()
    conn.close()


def load_watchlist() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["id","ticker","name","trend","origine","note","list_name","created_at"])
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
        conn.close()
    except Exception:
        return pd.DataFrame(columns=["id","ticker","name","trend","origine","note","list_name","created_at"])
    # Colonne alias per compatibilita' con dashboard (usa "Ticker","Nome")
    if "ticker" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})
    if "name" in df.columns and "Nome" not in df.columns:
        df = df.rename(columns={"name": "Nome"})
    for col in ["id","Ticker","Nome","trend","origine","note","list_name","created_at"]:
        if col not in df.columns:
            df[col] = "" if col != "id" else np.nan
    return df


def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE watchlist SET note = ? WHERE id = ?", (new_note, int(row_id)))
    conn.commit()
    conn.close()


def delete_from_watchlist(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("DELETE FROM watchlist WHERE id = ?", [(int(i),) for i in ids])
    conn.commit()
    conn.close()


def move_watchlist_rows(ids, dest_list):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("UPDATE watchlist SET list_name = ? WHERE id = ?",
                     [(dest_list, int(i)) for i in ids])
    conn.commit()
    conn.close()


def rename_watchlist(old_name, new_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE watchlist SET list_name = ? WHERE list_name = ?", (new_name, old_name))
    conn.commit()
    conn.close()


def reset_watchlist_by_name(list_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM watchlist WHERE list_name = ?", (list_name,))
    conn.commit()
    conn.close()


# -------------------------------------------------------------------------
# STORICO SCANSIONI
# -------------------------------------------------------------------------

def _df_to_json_safe(df: pd.DataFrame) -> str:
    """Converte DataFrame in JSON gestendo tipi numpy non serializzabili."""
    if df is None or df.empty:
        return "[]"
    df2 = df.copy()
    # Rimuovi colonne con oggetti non serializzabili (dict, list annidati)
    drop_cols = [c for c in df2.columns if c.startswith("_")]
    df2 = df2.drop(columns=drop_cols, errors="ignore")
    # Converti tipi numpy
    for col in df2.columns:
        try:
            df2[col] = df2[col].apply(
                lambda x: bool(x)  if isinstance(x, (np.bool_,))     else
                          float(x) if isinstance(x, np.floating)       else
                          int(x)   if isinstance(x, np.integer)        else
                          None     if isinstance(x, float) and (np.isnan(x) or np.isinf(x))
                          else x
            )
        except Exception:
            pass
    try:
        return df2.to_json(orient="records", default_handler=str)
    except Exception:
        return "[]"


def save_scan_history(markets: list, df_ep: pd.DataFrame, df_rea: pd.DataFrame,
                      elapsed_s: float = 0.0, cache_hits: int = 0) -> int:
    """
    Salva risultati scansione nel DB.
    Ritorna l'ID del record inserito (usato da save_signals).
    Compatibile con chiamate a 3 argomenti (elapsed_s e cache_hits opzionali).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Conta segnali (dopo _enrich_df, quindi Stato_Pro ha soglia 6)
        n_early = 0
        n_pro   = 0
        n_rea   = len(df_rea) if not df_rea.empty else 0
        n_conf  = 0

        if not df_ep.empty:
            if "Stato_Early" in df_ep.columns:
                n_early = int((df_ep["Stato_Early"] == "EARLY").sum())
            if "Stato_Pro" in df_ep.columns:
                n_pro = int((df_ep["Stato_Pro"] == "PRO").sum())
            if "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
                n_conf = int(
                    ((df_ep["Stato_Early"] == "EARLY") & (df_ep["Stato_Pro"] == "PRO")).sum()
                )

        ep_json  = _df_to_json_safe(df_ep)
        rea_json = _df_to_json_safe(df_rea)

        c.execute("""
            INSERT INTO scan_history
                (scanned_at, markets, n_early, n_pro, n_rea, n_confluence,
                 df_ep_json, df_rea_json, elapsed_s, cache_hits)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (now, json.dumps(markets), n_early, n_pro, n_rea, n_conf,
              ep_json, rea_json, float(elapsed_s), int(cache_hits)))
        conn.commit()
        scan_id = c.lastrowid
        conn.close()
        return scan_id

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0


def load_scan_history(limit: int = 20) -> pd.DataFrame:
    """Carica le ultime N scansioni (senza JSON dei dati)."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """SELECT id, scanned_at, markets, n_early, n_pro, n_rea, n_confluence,
                      elapsed_s, cache_hits
               FROM scan_history ORDER BY id DESC LIMIT ?""",
            conn, params=(limit,)
        )
        conn.close()
        return df
    except Exception:
        # Fallback senza colonne nuove
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(
                "SELECT id, scanned_at, markets, n_early, n_pro, n_rea, n_confluence FROM scan_history ORDER BY id DESC LIMIT ?",
                conn, params=(limit,)
            )
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()


def load_scan_snapshot(scan_id: int):
    """Carica i DataFrame di una scansione specifica dal DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT df_ep_json, df_rea_json FROM scan_history WHERE id = ?", (scan_id,))
        row = c.fetchone()
        conn.close()
        if row:
            import io
            df_ep  = pd.read_json(io.StringIO(row[0])) if row[0] and row[0] != "[]" else pd.DataFrame()
            df_rea = pd.read_json(io.StringIO(row[1])) if row[1] and row[1] != "[]" else pd.DataFrame()
            return df_ep, df_rea
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()


# -------------------------------------------------------------------------
# SIGNALS (v28 — opzionale, usato per backtest)
# -------------------------------------------------------------------------

def save_signals(scan_id, df_ep: pd.DataFrame, df_rea: pd.DataFrame, markets: list):
    """Stub compatibile — implementazione completa nel db_v28.py opzionale."""
    pass


# -------------------------------------------------------------------------
# CACHE (v28 — stub, implementazione in db_v28.py)
# -------------------------------------------------------------------------

def cache_stats() -> dict:
    return {"fresh": 0, "stale": 0, "size_mb": 0, "total_entries": 0}


def cache_clear(*a, **k):
    pass


# -------------------------------------------------------------------------
# INIT al caricamento del modulo
# -------------------------------------------------------------------------

init_db()

