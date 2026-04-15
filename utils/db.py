# -*- coding: utf-8 -*-
import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── DB Path ────────────────────────────────────────────────────────────────
# Su Streamlit Cloud: /mount/src/<repo>/ è READ-ONLY, /home/appuser/ è scrivibile
# Su locale: usa la home utente
# DB_PATH è fisso a runtime — non cambia mai durante la sessione.

_HERE = Path(__file__).parent

def _get_db_path() -> Path:
    """Path fisso e scrivibile per il DB watchlist.
    Priorità:
      1. $TRADING_DB_PATH  (variabile d'ambiente opzionale)
      2. /home/appuser/.trading_scanner/  (Streamlit Cloud)
      3. ~/.trading_scanner/              (locale / home generica)
      4. /tmp/                            (fallback assoluto)
    """
    import os
    # Priorità 1: variabile d'ambiente esplicita
    env_path = os.environ.get("TRADING_DB_PATH")
    if env_path:
        p = Path(env_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass

    # Priorità 2-4: cerca un path scrivibile
    candidates = [
        Path("/home/appuser/.trading_scanner/watchlist.db"),  # Streamlit Cloud
        Path.home() / ".trading_scanner" / "watchlist.db",   # locale
        Path("/tmp/trading_scanner_watchlist.db"),            # fallback
    ]
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            # Verifica scrittura reale
            _t = p.with_suffix(".tmp")
            _t.write_text("test"); _t.unlink()
            # Se esiste già un DB a /tmp con dati, migra
            _tmp = Path("/tmp/trading_scanner_watchlist.db")
            _old = Path("/tmp/watchlist.db")
            for _src in [_tmp, _old]:
                if _src != p and _src.exists() and _src.stat().st_size > 8192:
                    if not p.exists() or p.stat().st_size < _src.stat().st_size:
                        try:
                            import shutil; shutil.copy2(_src, p)
                        except Exception:
                            pass
                    break
            return p
        except Exception:
            continue
    return Path("/tmp/trading_scanner_watchlist.db")

DB_PATH = _get_db_path()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── Tabella Watchlist ──────────────────────────────────────────────────
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

    # ── Tabella Settings (API Keys, Preferences) ───────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT,
            updated_at TEXT
        )
    """)

    # ── Tabella Journal (Paper Trading) ────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            nome TEXT,
            entry_price REAL,
            exit_price REAL,
            size INTEGER,
            direction TEXT,
            setup_type TEXT,
            entry_date TEXT,
            exit_date TEXT,
            pnl REAL,
            pnl_pct REAL,
            notes TEXT,
            outcome TEXT,
            created_at TEXT
        )
    """)

    # ── Tabella Positions (P&L Tracker) ───────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            entry_price REAL,
            current_price REAL,
            size INTEGER,
            direction TEXT,
            stop_loss REAL,
            take_profit REAL,
            opened_at TEXT,
            updated_at TEXT,
            notes TEXT
        )
    """)

    # ── Tabella Alerts (Momentum Alerts) ───────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            alert_type TEXT,
            value REAL,
            priority TEXT,
            triggered_at TEXT,
            acknowledged INTEGER DEFAULT 0,
            notes TEXT
        )
    """)

    # ── Tabella Chat History (AI Assistant) ────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)

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
            nome        TEXT,
            signal_type TEXT,
            prezzo      REAL,
            markets     TEXT,
            rsi         REAL,
            quality_score REAL,
            ser_score   REAL,
            fv_score    REAL,
            squeeze     INTEGER,
            weekly_bull INTEGER,
            ret_1d      REAL,
            ret_5d      REAL,
            ret_10d     REAL,
            ret_20d     REAL,
            updated_at  TEXT
        )
    """)
    conn.commit()
    # Migrazione: aggiunge colonne mancanti a DB esistenti
    for _col, _ctype in [
        ('nome','TEXT'), ('rsi','REAL'), ('quality_score','REAL'),
        ('ser_score','REAL'), ('fv_score','REAL'),
        ('squeeze','INTEGER'), ('weekly_bull','INTEGER'),
    ]:
        try:
            conn.execute(f'ALTER TABLE signals ADD COLUMN {_col} {_ctype}')
            conn.commit()
        except Exception:
            pass  # colonna già presente


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
                nome   = str(row.get("Nome", "") or row.get("name", "") or "")
                rsi_v  = float(row.get("RSI", 0) or 0)
                qual_v = float(row.get("Quality_Score", 0) or 0)
                ser_v  = float(row.get("Ser_Score", 0) or 0)
                fv_v   = float(row.get("FV_Score", 0) or 0)
                sq_v   = 1 if row.get("Squeeze") in [True,"True","true",1] else 0
                wb_v   = 1 if row.get("Weekly_Bull") in [True,"True","true",1] else 0
                rows.append((scan_id, now, ticker, nome, stype, prezzo, mkt,
                             rsi_v, qual_v, ser_v, fv_v, sq_v, wb_v))
        if rows:
            conn.executemany(
                "INSERT INTO signals (scan_id,scanned_at,ticker,nome,signal_type,"
                "prezzo,markets,rsi,quality_score,ser_score,fv_score,squeeze,weekly_bull) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
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

# =========================================================================
# v42 — FUNZIONI PER DATABASE PERMANENTE
# =========================================================================

# ── Settings (API Keys, Preferences) ───────────────────────────────────────
def save_setting(key: str, value: str):
    """Salva una impostazione nel DB."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, now)
    )
    conn.commit()
    conn.close()

def load_setting(key: str, default: str = "") -> str:
    """Carica una impostazione dal DB."""
    if not DB_PATH.exists():
        return default
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else default
    except Exception:
        return default

def load_all_settings() -> dict:
    """Carica tutte le impostazioni."""
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT key, value FROM settings", conn)
        conn.close()
        return dict(zip(df["key"], df["value"]))
    except Exception:
        return {}

# ── Journal (Paper Trading) ────────────────────────────────────────────────
def add_journal_entry(ticker, nome, entry_price, exit_price, size, direction,
                      setup_type, entry_date, exit_date, pnl, pnl_pct, notes, outcome):
    """Aggiungi una entry al journal."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO journal (ticker, nome, entry_price, exit_price, size, direction,
        setup_type, entry_date, exit_date, pnl, pnl_pct, notes, outcome, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, nome, entry_price, exit_price, size, direction, setup_type,
          entry_date, exit_date, pnl, pnl_pct, notes, outcome, now))
    conn.commit()
    conn.close()

def load_journal(limit: int = 100) -> pd.DataFrame:
    """Carica journal entries."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM journal ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def delete_journal_entry(id: int):
    """Elimina entry journal."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM journal WHERE id = ?", (id,))
    conn.commit()
    conn.close()

# ── Positions (P&L Tracker) ───────────────────────────────────────────────
def save_position(ticker, entry_price, current_price, size, direction,
                  stop_loss, take_profit, opened_at, notes=""):
    """Salva o aggiorna una posizione."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Check if exists
    cur = conn.execute("SELECT id FROM positions WHERE ticker = ? AND direction = ?",
                       (ticker, direction))
    if cur.fetchone():
        conn.execute("""
            UPDATE positions SET entry_price=?, current_price=?, size=?,
            stop_loss=?, take_profit=?, updated_at=?, notes=?
            WHERE ticker=? AND direction=?
        """, (entry_price, current_price, size, stop_loss, take_profit, now, notes, ticker, direction))
    else:
        conn.execute("""
            INSERT INTO positions (ticker, entry_price, current_price, size, direction,
            stop_loss, take_profit, opened_at, updated_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, entry_price, current_price, size, direction,
              stop_loss, take_profit, opened_at, now, notes))
    conn.commit()
    conn.close()

def load_positions() -> pd.DataFrame:
    """Carica tutte le posizioni."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM positions ORDER BY opened_at DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def delete_position(id: int):
    """Elimina una posizione."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM positions WHERE id = ?", (id,))
    conn.commit()
    conn.close()

# ── Alerts (Momentum Alerts) ───────────────────────────────────────────────
def save_alert(ticker, alert_type, value, priority, notes=""):
    """Salva un alert."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO alerts (ticker, alert_type, value, priority, triggered_at, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker, alert_type, value, priority, now, notes))
    conn.commit()
    conn.close()

def load_alerts(limit: int = 100) -> pd.DataFrame:
    """Carica alerts."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY triggered_at DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def acknowledge_alert(id: int):
    """Segna alert come letto."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (id,))
    conn.commit()
    conn.close()

def delete_alert(id: int):
    """Elimina alert."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM alerts WHERE id = ?", (id,))
    conn.commit()
    conn.close()

# ── Chat History (AI Assistant) ────────────────────────────────────────────
def save_chat_message(role: str, content: str):
    """Salva messaggio chat."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO chat_history (role, content, created_at) VALUES (?, ?, ?)",
        (role, content, now)
    )
    conn.commit()
    conn.close()

def load_chat_history(limit: int = 50) -> list:
    """Carica cronologia chat."""
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df.to_dict("records") if not df.empty else []
    except Exception:
        return []

def clear_chat_history():
    """Pulisci cronologia chat."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chat_history")
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
            nome        TEXT,
            signal_type TEXT,
            prezzo      REAL,
            markets     TEXT,
            rsi         REAL,
            quality_score REAL,
            ser_score   REAL,
            fv_score    REAL,
            squeeze     INTEGER,
            weekly_bull INTEGER,
            ret_1d      REAL,
            ret_5d      REAL,
            ret_10d     REAL,
            ret_20d     REAL,
            updated_at  TEXT
        )
    """)
    conn.commit()
    # Migrazione: aggiunge colonne mancanti a DB esistenti
    for _col, _ctype in [
        ('nome','TEXT'), ('rsi','REAL'), ('quality_score','REAL'),
        ('ser_score','REAL'), ('fv_score','REAL'),
        ('squeeze','INTEGER'), ('weekly_bull','INTEGER'),
    ]:
        try:
            conn.execute(f'ALTER TABLE signals ADD COLUMN {_col} {_ctype}')
            conn.commit()
        except Exception:
            pass  # colonna già presente


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
                nome   = str(row.get("Nome", "") or row.get("name", "") or "")
                rsi_v  = float(row.get("RSI", 0) or 0)
                qual_v = float(row.get("Quality_Score", 0) or 0)
                ser_v  = float(row.get("Ser_Score", 0) or 0)
                fv_v   = float(row.get("FV_Score", 0) or 0)
                sq_v   = 1 if row.get("Squeeze") in [True,"True","true",1] else 0
                wb_v   = 1 if row.get("Weekly_Bull") in [True,"True","true",1] else 0
                rows.append((scan_id, now, ticker, nome, stype, prezzo, mkt,
                             rsi_v, qual_v, ser_v, fv_v, sq_v, wb_v))
        if rows:
            conn.executemany(
                "INSERT INTO signals (scan_id,scanned_at,ticker,nome,signal_type,"
                "prezzo,markets,rsi,quality_score,ser_score,fv_score,squeeze,weekly_bull) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
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

# ─────────────────────────────────────────────────────────────────
# LAYOUT GRIGLIA — persistenza larghezze/ordinamento colonne AgGrid
# ─────────────────────────────────────────────────────────────────
def _ensure_grid_layouts_table(conn):
    """Crea tabella grid_layouts se non esiste."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS grid_layouts (
            grid_key    TEXT PRIMARY KEY,
            layout_json TEXT,
            updated_at  TEXT
        )
    """)
    conn.commit()


def save_grid_layout(grid_key: str, layout: dict | None):
    """
    Salva il layout (colState/sortState) di una griglia nel DB.
    Se layout=None cancella il layout salvato (reset).
    """
    import json
    from datetime import datetime
    conn = sqlite3.connect(str(_get_db_path()))
    try:
        _ensure_grid_layouts_table(conn)
        if layout is None:
            conn.execute("DELETE FROM grid_layouts WHERE grid_key=?", (grid_key,))
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("""
                INSERT INTO grid_layouts (grid_key, layout_json, updated_at)
                VALUES (?,?,?)
                ON CONFLICT(grid_key) DO UPDATE SET
                    layout_json=excluded.layout_json,
                    updated_at=excluded.updated_at
            """, (grid_key, json.dumps(layout, default=str), now))
        conn.commit()
    finally:
        conn.close()


def load_grid_layout(grid_key: str) -> dict | None:
    """
    Carica il layout salvato per una griglia.
    Ritorna None se non esiste.
    """
    import json
    conn = sqlite3.connect(str(_get_db_path()))
    try:
        _ensure_grid_layouts_table(conn)
        row = conn.execute(
            "SELECT layout_json FROM grid_layouts WHERE grid_key=?", (grid_key,)
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None
    except Exception:
        return None
    finally:
        conn.close()

