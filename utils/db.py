import sqlite3
import json
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
        except sqlite3.OperationalError: pass
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
    for col_def in ["elapsed_s REAL", "cache_hits INTEGER DEFAULT 0"]:
        try:
            c.execute(f"ALTER TABLE scan_history ADD COLUMN {col_def}")
        except sqlite3.OperationalError: pass
    conn.commit()
    conn.close()

def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()
    init_db()

def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute("INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at) VALUES (?,?,?,?,?,?,?)", (t, n, trend, origine, note, list_name, now))
    conn.commit()
    conn.close()

def load_watchlist() -> pd.DataFrame:
    if not DB_PATH.exists(): return pd.DataFrame(columns=["id","Ticker","Nome","trend","origine","note","list_name","created_at"])
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY created_at DESC", conn)
        conn.close()
        if "ticker" in df.columns: df = df.rename(columns={"ticker": "Ticker"})
        if "name" in df.columns: df = df.rename(columns={"name": "Nome"})
        return df
    except Exception: return pd.DataFrame(columns=["id","Ticker","Nome","trend","origine","note","list_name","created_at"])

def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE watchlist SET note = ? WHERE id = ?", (new_note, int(row_id)))
    conn.commit()
    conn.close()

def delete_from_watchlist(ids):
    if not ids: return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("DELETE FROM watchlist WHERE id = ?", [(int(i),) for i in ids])
    conn.commit()
    conn.close()

def move_watchlist_rows(ids, dest_list):
    if not ids: return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("UPDATE watchlist SET list_name = ? WHERE id = ?", [(dest_list, int(i)) for i in ids])
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

def _df_to_json_safe(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "[]"
    df2 = df.copy()
    drop_cols = [c for c in df2.columns if c.startswith("_")]
    df2 = df2.drop(columns=drop_cols, errors="ignore")
    for col in df2.columns:
        try:
            df2[col] = df2[col].apply(lambda x: bool(x) if isinstance(x, (np.bool_)) else float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else None if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x)
        except Exception: pass
    try: return df2.to_json(orient="records", default_handler=str)
    except Exception: return "[]"

def save_scan_history(markets: list, df_ep: pd.DataFrame, df_rea: pd.DataFrame, elapsed_s: float = 0.0, cache_hits: int = 0) -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_early, n_pro, n_conf = 0, 0, 0
        n_rea = len(df_rea) if not df_rea.empty else 0
        if not df_ep.empty:
            if "StatoEarly" in df_ep.columns: n_early = int((df_ep["StatoEarly"] == "EARLY").sum())
            if "StatoPro" in df_ep.columns: n_pro = int((df_ep["StatoPro"] == "PRO").sum())
            if "StatoEarly" in df_ep.columns and "StatoPro" in df_ep.columns:
                n_conf = int(((df_ep["StatoEarly"] == "EARLY") & (df_ep["StatoPro"] == "PRO")).sum())
        ep_json = _df_to_json_safe(df_ep)
        rea_json = _df_to_json_safe(df_rea)
        c.execute("INSERT INTO scan_history (scanned_at, markets, n_early, n_pro, n_rea, n_confluence, df_ep_json, df_rea_json, elapsed_s, cache_hits) VALUES (?,?,?,?,?,?,?,?,?,?)", (now, json.dumps(markets), n_early, n_pro, n_rea, n_conf, ep_json, rea_json, float(elapsed_s), int(cache_hits)))
        conn.commit()
        scan_id = c.lastrowid
        conn.close()
        return scan_id
    except Exception:
        import traceback; traceback.print_exc(); return 0

def load_scan_history(limit: int = 20) -> pd.DataFrame:
    if not DB_PATH.exists(): return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT id, scanned_at, markets, n_early, n_pro, n_rea, n_confluence, elapsed_s, cache_hits FROM scan_history ORDER BY id DESC LIMIT ?", conn, params=(limit,))
        conn.close()
        return df
    except Exception: return pd.DataFrame()

def load_scan_snapshot(scan_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT df_ep_json, df_rea_json FROM scan_history WHERE id = ?", (scan_id,))
        row = c.fetchone()
        conn.close()
        if row:
            import io
            df_ep = pd.read_json(io.StringIO(row[0])) if row[0] and row[0] != "[]" else pd.DataFrame()
            df_rea = pd.read_json(io.StringIO(row[1])) if row[1] and row[1] != "[]" else pd.DataFrame()
            return df_ep, df_rea
    except Exception: pass
    return pd.DataFrame(), pd.DataFrame()

def save_signals(scan_id, df_ep, df_rea, markets): pass
def cache_stats(): return {"fresh": 0, "stale": 0, "size_mb": 0, "total_entries": 0}
def cache_clear(*a, **k): pass

init_db()
