import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "watchlist.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, name TEXT, 
                  origine TEXT, note TEXT, created_at TEXT)""")
    conn.commit()
    conn.close()

def add_to_watchlist(tickers, names, origine, note):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute("INSERT INTO watchlist (ticker, name, origine, note, created_at) VALUES (?,?,?,?,?)",
                  (t, n, origine, note, now))
    conn.commit()
    conn.close()

def load_watchlist():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
    conn.close()
    return df
