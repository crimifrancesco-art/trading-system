import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------
# DB PATH
# -----------------------------------------------------------------------------
DB_PATH = Path("watchlist.db")


# -----------------------------------------------------------------------------
# INIT DB
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# RESET DB
# -----------------------------------------------------------------------------
def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()
    init_db()


# -----------------------------------------------------------------------------
# ADD ROWS
# -----------------------------------------------------------------------------
def add_to_watchlist(
    tickers,
    names,
    origine="Scanner",
    note="",
    trend="LONG",
    list_name="DEFAULT",
):
    if not tickers:
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    for t, n in zip(tickers, names):
        c.execute("""
            INSERT INTO watchlist
            (ticker, name, trend, origine, note, list_name, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (t, n, trend, origine, note, list_name, now))

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# LOAD WATCHLIST
# -----------------------------------------------------------------------------
def load_watchlist():

    if not DB_PATH.exists():
        return pd.DataFrame(columns=[
            "id","ticker","name","trend",
            "origine","note","list_name","created_at"
        ])

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM watchlist ORDER BY created_at DESC",
        conn
    )
    conn.close()

    return df


# -----------------------------------------------------------------------------
# DELETE
# -----------------------------------------------------------------------------
def delete_from_watchlist(ids):

    if not ids:
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executemany(
        "DELETE FROM watchlist WHERE id=?",
        [(int(i),) for i in ids],
    )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# INIT ON IMPORT
# -----------------------------------------------------------------------------
init_db()
