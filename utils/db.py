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

    # Storico scansioni (NUOVO v21.0)
    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at TEXT NOT NULL,
            markets TEXT,
            n_early INTEGER,
            n_pro INTEGER,
            n_rea INTEGER,
            n_confluence INTEGER,
            df_ep_json TEXT,
            df_rea_json TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------------------------------
# WATCHLIST
# -------------------------------------------------------------------------

def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS watchlist")
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
        return pd.DataFrame(columns=["id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    for col in ["id", "ticker", "name", "trend", "origine", "note", "list_name", "created_at"]:
        if col not in df.columns:
            df[col] = "" if col != "id" else np.nan
    return df


def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE watchlist SET note = ? WHERE id = ?", (new_note, int(row_id)))
    conn.commit()
    conn.close()


def delete_from_watchlist(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany("DELETE FROM watchlist WHERE id = ?", [(int(i),) for i in ids])
    conn.commit()
    conn.close()


def move_watchlist_rows(ids, dest_list):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany("UPDATE watchlist SET list_name = ? WHERE id = ?",
                  [(dest_list, int(i)) for i in ids])
    conn.commit()
    conn.close()


def reset_watchlist_by_name(list_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE list_name = ?", (list_name,))
    conn.commit()
    conn.close()


def rename_watchlist(old_name, new_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE watchlist SET list_name = ? WHERE list_name = ?", (new_name, old_name))
    conn.commit()
    conn.close()


# -------------------------------------------------------------------------
# STORICO SCANSIONI (NUOVO v21.0)
# -------------------------------------------------------------------------

def save_scan_history(markets: list, df_ep: pd.DataFrame, df_rea: pd.DataFrame):
    """Salva i risultati della scansione nel DB per confronto storico."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_early = int((df_ep.get("Stato_Early", pd.Series()) == "EARLY").sum()) if not df_ep.empty else 0
        n_pro = int((df_ep.get("Stato_Pro", pd.Series()) == "PRO").sum()) if not df_ep.empty else 0
        n_rea = len(df_rea) if not df_rea.empty else 0

        # Confluence: EARLY + PRO contemporaneamente
        n_confluence = 0
        if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
            n_confluence = int(
                ((df_ep["Stato_Early"] == "EARLY") & (df_ep["Stato_Pro"] == "PRO")).sum()
            )

        ep_json = df_ep.to_json(orient="records") if not df_ep.empty else "[]"
        rea_json = df_rea.to_json(orient="records") if not df_rea.empty else "[]"

        c.execute("""
            INSERT INTO scan_history (scanned_at, markets, n_early, n_pro, n_rea, n_confluence, df_ep_json, df_rea_json)
            VALUES (?,?,?,?,?,?,?,?)
        """, (now, json.dumps(markets), n_early, n_pro, n_rea, n_confluence, ep_json, rea_json))
        conn.commit()
        conn.close()
    except Exception:
        pass


def load_scan_history(limit=20) -> pd.DataFrame:
    """Carica le ultime N scansioni (senza JSON dei dati)."""
    if not DB_PATH.exists():
        return pd.DataFrame()
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
            df_ep = pd.read_json(row[0]) if row[0] and row[0] != "[]" else pd.DataFrame()
            df_rea = pd.read_json(row[1]) if row[1] and row[1] != "[]" else pd.DataFrame()
            return df_ep, df_rea
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()


# Inizializza DB al caricamento del modulo
init_db()

