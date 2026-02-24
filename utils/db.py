import sqlite3
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
            ticker TEXT,
            name TEXT,
            trend TEXT,
            origine TEXT,
            note TEXT,
            list_name TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_to_watchlist(tickers, names, origine,
                     note="Scanner",
                     trend="LONG",
                     list_name="DEFAULT"):

    if not tickers:
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    for t, n in zip(tickers, names):
        c.execute("""
            INSERT INTO watchlist
            (ticker,name,trend,origine,note,list_name,created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (t, n, trend, origine, note, list_name, now))

    conn.commit()
    conn.close()


def load_watchlist():
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    return df


def delete_from_watchlist(ids):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executemany(
        "DELETE FROM watchlist WHERE id=?",
        [(int(i),) for i in ids]
    )

    conn.commit()
    conn.close()


init_db()
