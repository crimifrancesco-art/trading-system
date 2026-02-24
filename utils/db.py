import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path("watchlist.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
    conn.commit()
    conn.close()


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


# Inizializza DB al caricamento del modulo
init_db()
