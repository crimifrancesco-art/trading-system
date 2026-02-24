import sqlite3
import pandas as pd
from datetime import datetime

def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers: return
    conn = sqlite3.connect("watchlist.db")
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute("""INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at)
                     VALUES (?,?,?,?,?,?,?)""", (t, n, trend, origine, note, list_name, now))
    conn.commit()
    conn.close()
