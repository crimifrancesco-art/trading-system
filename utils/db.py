# db.py

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path.home() / ".trading_scanner_watchlist.db"


def init_db():

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        ticker TEXT PRIMARY KEY,
        added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def add_ticker(ticker):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        "INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)",
        (ticker,)
    )

    conn.commit()
    conn.close()


def remove_ticker(ticker):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        "DELETE FROM watchlist WHERE ticker=?",
        (ticker,)
    )

    conn.commit()
    conn.close()


def get_watchlist():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        "SELECT * FROM watchlist",
        conn
    )

    conn.close()

    return df
