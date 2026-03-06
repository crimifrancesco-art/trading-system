import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Path persistente: usa la home dell'utente se disponibile (Streamlit Cloud),
# altrimenti cartella locale al modulo
_HERE = Path(__file__).parent  # utils/
_HOME_DB = Path.home() / ".streamlit_trading_scanner" / "watchlist.db"
_LOCAL_DB = _HERE / "watchlist.db"

def _get_db_path() -> Path:
    """Ritorna path DB persistente. Priorità: home > locale."""
    # Prova cartella home (persiste tra restart su Streamlit Cloud)
    try:
        _HOME_DB.parent.mkdir(parents=True, exist_ok=True)
        # Test scrittura
        test = _HOME_DB.parent / ".write_test"
        test.write_text("ok"); test.unlink()
        return _HOME_DB
    except Exception:
        pass
    # Fallback: stessa cartella di db.py
    try:
        _HERE.mkdir(parents=True, exist_ok=True)
        return _LOCAL_DB
    except Exception:
        pass
    # Ultimo fallback: /tmp (non persiste tra restart)
    return Path("/tmp/watchlist.db")

DB_PATH = _get_db_path()

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
    # Crea tabella signals per backtest
    _ensure_signals_table(conn)
    conn.commit()
    conn.close()

def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()
    
def _ensure_signals_table(conn):
    """Crea tabella signals se non esiste."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id     INTEGER,
            scanned_at  TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            signal_type TEXT,
            prezzo      REAL,
            markets     TEXT,
            ret_1d      REAL,
            ret_5d      REAL,
            ret_10d     REAL,
            ret_20d     REAL,
            updated_at  TEXT
        )
    """)
    conn.commit()


def save_signals(scan_id: int, df_ep: pd.DataFrame,
                 df_rea: pd.DataFrame, markets: list):
    """Salva segnali EP e REA nella tabella signals."""
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mkt = json.dumps(markets) if markets else "[]"
        rows = []
        for df, stype_col, default_type in [
            (df_ep,  "Stato_Early", "EARLY"),
            (df_rea, "Stato",       "HOT"),
        ]:
            if df is None or df.empty: continue
            for _, row in df.iterrows():
                ticker = str(row.get("Ticker", ""))
                if not ticker: continue
                stype = str(row.get(stype_col, default_type))
                if stype == "-" or not stype:
                    stype = default_type
                prezzo = float(row.get("Prezzo", 0) or 0)
                rows.append((scan_id, now, ticker, stype, prezzo, mkt))
        if rows:
            conn.executemany(
                "INSERT INTO signals (scan_id,scanned_at,ticker,signal_type,"
                "prezzo,markets) VALUES (?,?,?,?,?,?)",
                rows
            )
            conn.commit()
        conn.close()
    except Exception:
        import traceback; traceback.print_exc()


def load_signals(signal_type: str = None, days_back: int = 90,
                 with_perf: bool = True) -> pd.DataFrame:
    """Carica segnali dal DB, opzionalmente filtrati per tipo e periodo."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        where = []
        params = []
        if signal_type and signal_type != "Tutti":
            where.append("signal_type = ?"); params.append(signal_type)
        if days_back:
            where.append("scanned_at >= datetime('now', ?)")
            params.append(f"-{days_back} days")
        sql = "SELECT * FROM signals"
        if where: sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scanned_at DESC"
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def signal_summary_stats(days_back: int = 90) -> pd.DataFrame:
    """Statistiche aggregate: win rate e avg return per tipo segnale."""
    df = load_signals(days_back=days_back, with_perf=True)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for stype, grp in df.groupby("signal_type"):
        n = len(grp)
        for col, label in [("ret_1d","1g"),("ret_5d","5g"),
                           ("ret_10d","10g"),("ret_20d","20g")]:
            if col not in grp.columns: continue
            vals = grp[col].dropna()
            if vals.empty: continue
            rows.append({
                "Tipo": stype, "Periodo": label, "N": n,
                "Win%":  round((vals > 0).mean() * 100, 1),
                "Avg%":  round(vals.mean(), 2),
                "Med%":  round(vals.median(), 2),
                "Max%":  round(vals.max(), 2),
                "Min%":  round(vals.min(), 2),
            })
    return pd.DataFrame(rows)


def update_signal_performance(max_signals: int = 300) -> int:
    """Aggiorna prezzi forward +1/5/10/20g per segnali senza performance."""
    if not DB_PATH.exists():
        return 0
    try:
        import yfinance as _yf
    except ImportError:
        return 0
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        # Carica segnali senza performance completa
        df = pd.read_sql_query(
            "SELECT * FROM signals WHERE ret_20d IS NULL "
            "ORDER BY scanned_at DESC LIMIT ?",
            conn, params=(max_signals,)
        )
        if df.empty:
            conn.close(); return 0

        updated = 0
        for _, row in df.iterrows():
            try:
                tkr  = row["ticker"]
                date = pd.to_datetime(row["scanned_at"])
                p0   = float(row["prezzo"] or 0)
                if p0 <= 0: continue

                hist = _yf.Ticker(tkr).history(
                    start=date.strftime("%Y-%m-%d"),
                    end=(date + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True
                )
                if hist.empty: continue
                closes = hist["Close"].dropna()
                if len(closes) < 2: continue

                def _ret(n):
                    idx = min(n, len(closes)-1)
                    return round((float(closes.iloc[idx]) / p0 - 1) * 100, 2)

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "UPDATE signals SET ret_1d=?,ret_5d=?,ret_10d=?,ret_20d=?,"
                    "updated_at=? WHERE id=?",
                    (_ret(1),_ret(5),_ret(10),_ret(20), now, int(row["id"]))
                )
                updated += 1
            except Exception:
                continue
        conn.commit()
        conn.close()
        return updated
    except Exception:
        import traceback; traceback.print_exc()
        return 0

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
            if "Stato_Early" in df_ep.columns: n_early = int((df_ep["Stato_Early"] == "EARLY").sum())
            if "Stato_Pro" in df_ep.columns: n_pro = int((df_ep["Stato_Pro"] == "PRO").sum())
            if "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
                n_conf = int(((df_ep["Stato_Early"] == "EARLY") & (df_ep["Stato_Pro"] == "PRO")).sum())
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


def _ensure_signals_table(conn):
    """Crea tabella signals se non esiste."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id     INTEGER,
            scanned_at  TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            signal_type TEXT,
            prezzo      REAL,
            markets     TEXT,
            ret_1d      REAL,
            ret_5d      REAL,
            ret_10d     REAL,
            ret_20d     REAL,
            updated_at  TEXT
        )
    """)
    conn.commit()


def save_signals(scan_id: int, df_ep: pd.DataFrame,
                 df_rea: pd.DataFrame, markets: list):
    """Salva segnali EP e REA nella tabella signals."""
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mkt = json.dumps(markets) if markets else "[]"
        rows = []
        for df, stype_col, default_type in [
            (df_ep,  "Stato_Early", "EARLY"),
            (df_rea, "Stato",       "HOT"),
        ]:
            if df is None or df.empty: continue
            for _, row in df.iterrows():
                ticker = str(row.get("Ticker", ""))
                if not ticker: continue
                stype = str(row.get(stype_col, default_type))
                if stype == "-" or not stype:
                    stype = default_type
                prezzo = float(row.get("Prezzo", 0) or 0)
                rows.append((scan_id, now, ticker, stype, prezzo, mkt))
        if rows:
            conn.executemany(
                "INSERT INTO signals (scan_id,scanned_at,ticker,signal_type,"
                "prezzo,markets) VALUES (?,?,?,?,?,?)",
                rows
            )
            conn.commit()
        conn.close()
    except Exception:
        import traceback; traceback.print_exc()


def load_signals(signal_type: str = None, days_back: int = 90,
                 with_perf: bool = True) -> pd.DataFrame:
    """Carica segnali dal DB, opzionalmente filtrati per tipo e periodo."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        where = []
        params = []
        if signal_type and signal_type != "Tutti":
            where.append("signal_type = ?"); params.append(signal_type)
        if days_back:
            where.append("scanned_at >= datetime('now', ?)")
            params.append(f"-{days_back} days")
        sql = "SELECT * FROM signals"
        if where: sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scanned_at DESC"
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def signal_summary_stats(days_back: int = 90) -> pd.DataFrame:
    """Statistiche aggregate: win rate e avg return per tipo segnale."""
    df = load_signals(days_back=days_back, with_perf=True)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for stype, grp in df.groupby("signal_type"):
        n = len(grp)
        for col, label in [("ret_1d","1g"),("ret_5d","5g"),
                           ("ret_10d","10g"),("ret_20d","20g")]:
            if col not in grp.columns: continue
            vals = grp[col].dropna()
            if vals.empty: continue
            rows.append({
                "Tipo": stype, "Periodo": label, "N": n,
                "Win%":  round((vals > 0).mean() * 100, 1),
                "Avg%":  round(vals.mean(), 2),
                "Med%":  round(vals.median(), 2),
                "Max%":  round(vals.max(), 2),
                "Min%":  round(vals.min(), 2),
            })
    return pd.DataFrame(rows)


def update_signal_performance(max_signals: int = 300) -> int:
    """Aggiorna prezzi forward +1/5/10/20g per segnali senza performance."""
    if not DB_PATH.exists():
        return 0
    try:
        import yfinance as _yf
    except ImportError:
        return 0
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        # Carica segnali senza performance completa
        df = pd.read_sql_query(
            "SELECT * FROM signals WHERE ret_20d IS NULL "
            "ORDER BY scanned_at DESC LIMIT ?",
            conn, params=(max_signals,)
        )
        if df.empty:
            conn.close(); return 0

        updated = 0
        for _, row in df.iterrows():
            try:
                tkr  = row["ticker"]
                date = pd.to_datetime(row["scanned_at"])
                p0   = float(row["prezzo"] or 0)
                if p0 <= 0: continue

                hist = _yf.Ticker(tkr).history(
                    start=date.strftime("%Y-%m-%d"),
                    end=(date + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True
                )
                if hist.empty: continue
                closes = hist["Close"].dropna()
                if len(closes) < 2: continue

                def _ret(n):
                    idx = min(n, len(closes)-1)
                    return round((float(closes.iloc[idx]) / p0 - 1) * 100, 2)

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "UPDATE signals SET ret_1d=?,ret_5d=?,ret_10d=?,ret_20d=?,"
                    "updated_at=? WHERE id=?",
                    (_ret(1),_ret(5),_ret(10),_ret(20), now, int(row["id"]))
                )
                updated += 1
            except Exception:
                continue
        conn.commit()
        conn.close()
        return updated
    except Exception:
        import traceback; traceback.print_exc()
        return 0

init_db()
