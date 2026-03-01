"""
utils/db.py  —  v28.0
======================
Novità rispetto alla v27:
  • Tabella  yf_cache      → cache yfinance su SQLite (funziona su Streamlit Cloud)
  • Tabella  signals       → registra ogni segnale emesso per backtest
  • Tabella  signal_perf   → performance aggiornate dei segnali (prezzi forward)
  • Funzioni cache_*       → get/set/clear per history e info
  • Funzioni signal_*      → salva segnali, carica, aggiorna performance
"""

import json
import sqlite3
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PERCORSO DB  (assoluto, stabile su Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "watchlist.db"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER  serializzazione DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _safe_df_to_json(df: pd.DataFrame) -> str:
    if df.empty:
        return "[]"
    drop = [c for c in df.columns if c.startswith("_")]
    df2  = df.drop(columns=drop, errors="ignore").copy()
    for col in df2.columns:
        try:
            df2[col] = df2[col].apply(
                lambda x: bool(x)  if isinstance(x, np.bool_)    else
                          float(x) if isinstance(x, np.floating)  else
                          int(x)   if isinstance(x, np.integer)   else
                          None     if isinstance(x, float) and (np.isnan(x) or np.isinf(x))
                          else x
            )
        except Exception:
            pass
    try:
        return df2.to_json(orient="records", default_handler=str)
    except Exception:
        return "[]"


def _safe_json_to_df(s: str) -> pd.DataFrame:
    if not s or s in ("[]", "null", ""):
        return pd.DataFrame()
    try:
        return pd.read_json(s, orient="records")
    except Exception:
        return pd.DataFrame()


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH), timeout=30,
                           check_same_thread=False)


# ─────────────────────────────────────────────────────────────────────────────
# INIT DB
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _conn()
    c    = conn.cursor()

    # ── Watchlist ─────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker     TEXT NOT NULL,
            name       TEXT,
            trend      TEXT,
            origine    TEXT,
            note       TEXT,
            list_name  TEXT DEFAULT 'DEFAULT',
            created_at TEXT
        )
    """)
    existing_wl = {r[1] for r in c.execute("PRAGMA table_info(watchlist)").fetchall()}
    for col_def, col_name in [
        ("trend TEXT",    "trend"),
        ("list_name TEXT","list_name"),
        ("origine TEXT",  "origine"),
        ("note TEXT",     "note"),
    ]:
        if col_name not in existing_wl:
            try: c.execute(f"ALTER TABLE watchlist ADD COLUMN {col_def}")
            except sqlite3.OperationalError: pass

    # ── Storico scansioni ─────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at   TEXT NOT NULL,
            markets      TEXT,
            n_early      INTEGER DEFAULT 0,
            n_pro        INTEGER DEFAULT 0,
            n_rea        INTEGER DEFAULT 0,
            n_confluence INTEGER DEFAULT 0,
            elapsed_s    REAL    DEFAULT 0,
            cache_hits   INTEGER DEFAULT 0,
            df_ep_json   TEXT,
            df_rea_json  TEXT
        )
    """)
    existing_sh = {r[1] for r in c.execute("PRAGMA table_info(scan_history)").fetchall()}
    for col_def, col_name in [
        ("elapsed_s REAL DEFAULT 0",   "elapsed_s"),
        ("cache_hits INTEGER DEFAULT 0","cache_hits"),
    ]:
        if col_name not in existing_sh:
            try: c.execute(f"ALTER TABLE scan_history ADD COLUMN {col_def}")
            except sqlite3.OperationalError: pass

    # ── Cache yfinance  ───────────────────────────────────────────────────
    # Chiave: ticker + kind ('history_9mo'/'history_6mo_weekly'/'info'/'calendar')
    # Value: JSON/bytes compresso
    # expires_at: TTL come stringa ISO
    c.execute("""
        CREATE TABLE IF NOT EXISTS yf_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            kind        TEXT    NOT NULL,
            value_json  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL,
            expires_at  TEXT    NOT NULL,
            UNIQUE(ticker, kind) ON CONFLICT REPLACE
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_yf_cache_tk ON yf_cache(ticker, kind)")

    # ── Segnali (per backtest) ────────────────────────────────────────────
    # Ogni riga = un segnale emesso in una scansione
    # entry_price = prezzo al momento del segnale
    # perf_* vengono aggiornati successivamente
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id       INTEGER NOT NULL,
            scanned_at    TEXT    NOT NULL,
            ticker        TEXT    NOT NULL,
            nome          TEXT,
            signal_type   TEXT    NOT NULL,  -- 'EARLY','PRO','HOT','CONFLUENCE','SERAFINI','FINVIZ'
            entry_price   REAL    NOT NULL,
            rsi           REAL,
            quality_score REAL,
            early_score   REAL,
            pro_score     REAL,
            ser_score     REAL,
            fv_score      REAL,
            squeeze       INTEGER,
            weekly_bull   INTEGER,
            markets       TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_tk  ON signals(ticker)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_sc  ON signals(scan_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_tp  ON signals(signal_type)")

    # ── Performance segnali ───────────────────────────────────────────────
    # Per ogni segnale, memorizza prezzi forward a +1d, +5d, +10d, +20d
    c.execute("""
        CREATE TABLE IF NOT EXISTS signal_perf (
            signal_id     INTEGER PRIMARY KEY,
            ticker        TEXT,
            entry_price   REAL,
            price_1d      REAL,
            price_5d      REAL,
            price_10d     REAL,
            price_20d     REAL,
            ret_1d        REAL,
            ret_5d        REAL,
            ret_10d       REAL,
            ret_20d       REAL,
            last_updated  TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CACHE YFINANCE  (SQLite, funziona su Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────

_TTL = {
    "history_9mo":       timedelta(hours=4),
    "history_6mo_weekly":timedelta(hours=8),
    "info":              timedelta(hours=12),
    "calendar":          timedelta(hours=24),
    "finviz":            timedelta(hours=6),
}


def cache_get(ticker: str, kind: str):
    """
    Ritorna valore dalla cache se fresco, altrimenti None.
    kind: 'history_9mo' | 'history_6mo_weekly' | 'info' | 'calendar' | 'finviz'
    """
    try:
        conn = _conn()
        row  = conn.execute(
            "SELECT value_json, expires_at FROM yf_cache WHERE ticker=? AND kind=?",
            (ticker, kind)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        value_json, expires_at = row
        if datetime.fromisoformat(expires_at) < datetime.now():
            return None                 # scaduto
        return json.loads(value_json)
    except Exception:
        return None


def cache_set(ticker: str, kind: str, value):
    """
    Salva valore in cache.
    value può essere dict, list, o qualsiasi JSON-serializzabile.
    """
    try:
        ttl        = _TTL.get(kind, timedelta(hours=6))
        now        = datetime.now()
        expires_at = (now + ttl).isoformat()
        value_json = json.dumps(value, default=str)
        conn       = _conn()
        conn.execute("""
            INSERT OR REPLACE INTO yf_cache (ticker, kind, value_json, updated_at, expires_at)
            VALUES (?,?,?,?,?)
        """, (ticker, kind, value_json, now.isoformat(), expires_at))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[cache_set ERROR] {ticker}/{kind}: {e}")


def cache_clear(ticker: str = None, kind: str = None):
    """
    Cancella cache. ticker=None → cancella tutto.
    """
    conn = _conn()
    if ticker is None:
        conn.execute("DELETE FROM yf_cache")
    elif kind is None:
        conn.execute("DELETE FROM yf_cache WHERE ticker=?", (ticker,))
    else:
        conn.execute("DELETE FROM yf_cache WHERE ticker=? AND kind=?", (ticker, kind))
    conn.commit()
    conn.close()


def cache_stats() -> dict:
    """Statistiche sulla cache corrente."""
    try:
        conn  = _conn()
        total = conn.execute("SELECT COUNT(*) FROM yf_cache").fetchone()[0]
        fresh = conn.execute(
            "SELECT COUNT(*) FROM yf_cache WHERE expires_at > ?",
            (datetime.now().isoformat(),)
        ).fetchone()[0]
        stale = total - fresh
        # Stima dimensione in MB
        size_bytes = conn.execute(
            "SELECT SUM(LENGTH(value_json)) FROM yf_cache"
        ).fetchone()[0] or 0
        conn.close()
        return {
            "total_entries": total,
            "fresh":         fresh,
            "stale":         stale,
            "size_mb":       round(size_bytes / 1024 / 1024, 2),
        }
    except Exception:
        return {"total_entries": 0, "fresh": 0, "stale": 0, "size_mb": 0}


def df_to_cache_json(df: pd.DataFrame) -> list:
    """Converte DataFrame → lista di dict JSON-safe."""
    if df is None or df.empty:
        return []
    df2 = df.copy()
    # Converti index datetime → string
    if hasattr(df2.index, 'strftime'):
        df2.index = df2.index.strftime("%Y-%m-%d")
    df2 = df2.reset_index()
    return json.loads(df2.to_json(orient="records", default_handler=str))


def cache_json_to_df(data: list, index_col: str = None) -> pd.DataFrame:
    """Ricostruisce DataFrame da lista di dict."""
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if index_col and index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST  (identica a v27, mantenuta per compatibilità)
# ─────────────────────────────────────────────────────────────────────────────

def reset_watchlist_db():
    conn = _conn()
    conn.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit(); conn.close()
    init_db()


def add_to_watchlist(tickers, names, origine, note, trend="LONG", list_name="DEFAULT"):
    if not tickers:
        return
    conn = _conn(); c = conn.cursor()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        ex = c.execute(
            "SELECT id FROM watchlist WHERE ticker=? AND list_name=?", (t, list_name)
        ).fetchone()
        if ex:
            c.execute("""UPDATE watchlist SET name=?,trend=?,origine=?,note=?,created_at=?
                         WHERE ticker=? AND list_name=?""",
                      (n, trend, origine, note, now, t, list_name))
        else:
            c.execute("""INSERT INTO watchlist (ticker,name,trend,origine,note,list_name,created_at)
                         VALUES (?,?,?,?,?,?,?)""",
                      (t, n, trend, origine, note, list_name, now))
    conn.commit(); conn.close()


def load_watchlist() -> pd.DataFrame:
    if not DB_PATH.exists():
        init_db()
        return pd.DataFrame(columns=["id","ticker","name","trend","origine","note","list_name","created_at"])
    try:
        conn = _conn()
        df   = pd.read_sql_query(
            "SELECT * FROM watchlist ORDER BY list_name, created_at DESC", conn)
        conn.close()
    except Exception:
        return pd.DataFrame(columns=["id","ticker","name","trend","origine","note","list_name","created_at"])
    for col in ["id","ticker","name","trend","origine","note","list_name","created_at"]:
        if col not in df.columns: df[col] = "" if col != "id" else np.nan
    rename = {}
    if "ticker" in df.columns and "Ticker" not in df.columns: rename["ticker"] = "Ticker"
    if "name"   in df.columns and "Nome"   not in df.columns: rename["name"]   = "Nome"
    if rename: df = df.rename(columns=rename)
    return df


def delete_from_watchlist(ids):
    if not ids: return
    conn = _conn(); c = conn.cursor()
    c.executemany("DELETE FROM watchlist WHERE id=?", [(int(i),) for i in ids])
    conn.commit(); conn.close()


def move_watchlist_rows(ids, dest_list):
    if not ids: return
    conn = _conn(); c = conn.cursor()
    c.executemany("UPDATE watchlist SET list_name=? WHERE id=?",
                  [(dest_list, int(i)) for i in ids])
    conn.commit(); conn.close()


def rename_watchlist(old_name, new_name):
    conn = _conn()
    conn.execute("UPDATE watchlist SET list_name=? WHERE list_name=?", (new_name, old_name))
    conn.commit(); conn.close()


def update_watchlist_note(row_id, new_note):
    conn = _conn()
    conn.execute("UPDATE watchlist SET note=? WHERE id=?", (new_note, int(row_id)))
    conn.commit(); conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# STORICO SCANSIONI
# ─────────────────────────────────────────────────────────────────────────────

def save_scan_history(markets: list, df_ep: pd.DataFrame, df_rea: pd.DataFrame,
                      elapsed_s: float = 0, cache_hits: int = 0):
    try:
        conn = _conn(); c = conn.cursor()
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_early = int((df_ep.get("Stato_Early", pd.Series()) == "EARLY").sum()) if not df_ep.empty else 0
        n_pro   = int((df_ep.get("Stato_Pro",   pd.Series()) == "PRO"  ).sum()) if not df_ep.empty else 0
        n_rea   = len(df_rea) if not df_rea.empty else 0
        n_conf  = 0
        if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
            n_conf = int(((df_ep["Stato_Early"]=="EARLY")&(df_ep["Stato_Pro"]=="PRO")).sum())

        ep_json  = _safe_df_to_json(df_ep)
        rea_json = _safe_df_to_json(df_rea)

        c.execute("""
            INSERT INTO scan_history
                (scanned_at, markets, n_early, n_pro, n_rea, n_confluence,
                 elapsed_s, cache_hits, df_ep_json, df_rea_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (now, json.dumps(markets), n_early, n_pro, n_rea, n_conf,
              round(elapsed_s, 1), cache_hits, ep_json, rea_json))

        scan_id = c.lastrowid

        # Mantieni ultime 50 scansioni
        c.execute("""DELETE FROM scan_history WHERE id NOT IN
                     (SELECT id FROM scan_history ORDER BY id DESC LIMIT 50)""")

        conn.commit(); conn.close()
        return scan_id
    except Exception as e:
        print(f"[save_scan_history ERROR] {e}\n{traceback.format_exc()}")
        return None


def load_scan_history(limit: int = 20) -> pd.DataFrame:
    if not DB_PATH.exists(): return pd.DataFrame()
    try:
        conn = _conn()
        df   = pd.read_sql_query("""
            SELECT id, scanned_at, markets, n_early, n_pro, n_rea, n_confluence,
                   elapsed_s, cache_hits
            FROM scan_history ORDER BY id DESC LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_scan_snapshot(scan_id: int):
    try:
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT df_ep_json, df_rea_json FROM scan_history WHERE id=?", (int(scan_id),))
        row = c.fetchone(); conn.close()
        if row: return _safe_json_to_df(row[0]), _safe_json_to_df(row[1])
    except Exception: pass
    return pd.DataFrame(), pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SEGNALI  (per backtest)
# ─────────────────────────────────────────────────────────────────────────────

def save_signals(scan_id: int, df_ep: pd.DataFrame, df_rea: pd.DataFrame,
                 markets: list, scanned_at: str = None):
    """
    Registra i segnali di questa scansione nella tabella signals.
    Viene chiamato da save_scan_history — ogni segnale EARLY/PRO/HOT/CONFLUENCE/SERAFINI/FINVIZ
    viene salvato con il prezzo di entrata corrente.
    Evita duplicati: non re-inserisce segnali (ticker, signal_type) già presenti
    nelle ultime 24 ore (stesso segnale non si ripete in giornata).
    """
    if scan_id is None:
        return
    now = scanned_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mkts = json.dumps(markets)
    conn = _conn(); c = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    def _insert(row, stype):
        tkr   = row.get("Ticker", row.get("ticker", ""))
        price = row.get("Prezzo", 0)
        if not tkr or not price:
            return
        # Evita duplicati nelle ultime 24h per stesso ticker+tipo
        dup = c.execute("""SELECT id FROM signals
                           WHERE ticker=? AND signal_type=? AND scanned_at>?""",
                        (tkr, stype, cutoff)).fetchone()
        if dup:
            return
        c.execute("""
            INSERT INTO signals
                (scan_id, scanned_at, ticker, nome, signal_type,
                 entry_price, rsi, quality_score, early_score, pro_score,
                 ser_score, fv_score, squeeze, weekly_bull, markets)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            scan_id, now, tkr,
            str(row.get("Nome", ""))[:80],
            stype,
            float(price),
            float(row.get("RSI", 0) or 0),
            float(row.get("Quality_Score", 0) or 0),
            float(row.get("Early_Score",   0) or 0),
            float(row.get("Pro_Score",     0) or 0),
            float(row.get("Ser_Score",     0) or 0),
            float(row.get("FV_Score",      0) or 0),
            int(bool(row.get("Squeeze",    False))),
            int(bool(row.get("Weekly_Bull",False))),
            mkts,
        ))

    if not df_ep.empty:
        for _, row in df_ep.iterrows():
            stato_e = str(row.get("Stato_Early", "-"))
            stato_p = str(row.get("Stato_Pro",   "-"))
            ser_ok  = row.get("Ser_OK", False)
            fv_ok   = row.get("FV_OK",  False)
            if stato_e == "EARLY":
                _insert(row, "EARLY")
            if stato_p == "PRO":
                _insert(row, "PRO")
            if stato_e == "EARLY" and stato_p == "PRO":
                _insert(row, "CONFLUENCE")
            if ser_ok is True or str(ser_ok).lower() == "true":
                _insert(row, "SERAFINI")
            if fv_ok is True or str(fv_ok).lower() == "true":
                _insert(row, "FINVIZ")

    if not df_rea.empty:
        for _, row in df_rea.iterrows():
            if str(row.get("Stato", "")) == "HOT":
                _insert(row, "HOT")

    conn.commit(); conn.close()


def update_signal_performance(max_signals: int = 500):
    """
    Aggiorna i prezzi forward (+1d,+5d,+10d,+20d) per i segnali
    che non li hanno ancora.
    Chiamato dal dashboard in background (non blocca lo scanner).
    Usa yfinance in batch per ridurre le chiamate API.
    """
    import yfinance as yf

    conn = _conn()
    # Prendi segnali senza performance o con performance parziale
    rows = conn.execute("""
        SELECT s.id, s.ticker, s.scanned_at, s.entry_price,
               p.price_1d, p.price_5d, p.price_10d, p.price_20d
        FROM signals s
        LEFT JOIN signal_perf p ON s.id = p.signal_id
        WHERE p.ret_20d IS NULL
        ORDER BY s.scanned_at DESC
        LIMIT ?
    """, (max_signals,)).fetchall()
    conn.close()

    if not rows:
        return 0

    # Raggruppa per ticker per scaricare in batch
    by_ticker: dict = {}
    for row in rows:
        sig_id, ticker, scanned_at, entry, p1, p5, p10, p20 = row
        by_ticker.setdefault(ticker, []).append({
            "id": sig_id, "scanned_at": scanned_at,
            "entry": entry, "p1": p1, "p5": p5, "p10": p10, "p20": p20
        })

    updated = 0
    conn = _conn(); c = conn.cursor()

    for ticker, sigs in by_ticker.items():
        try:
            # Scarica 3 mesi per coprire tutti i forward day
            hist = yf.Ticker(ticker).history(period="3mo")
            if hist.empty:
                continue
            closes = hist["Close"]
            dates  = pd.DatetimeIndex(closes.index.date)

            for sig in sigs:
                sig_date = pd.Timestamp(sig["scanned_at"][:10]).date()
                # Trova indice del giorno del segnale (o più recente disponibile)
                idx_arr = np.searchsorted(dates, sig_date)
                if idx_arr >= len(closes):
                    continue

                def price_at(offset):
                    i = idx_arr + offset
                    return float(closes.iloc[i]) if i < len(closes) else None

                p1  = sig["p1"]  or price_at(1)
                p5  = sig["p5"]  or price_at(5)
                p10 = sig["p10"] or price_at(10)
                p20 = sig["p20"] or price_at(20)
                ep  = sig["entry"]

                def ret(p):
                    return round((p - ep) / ep * 100, 2) if p and ep else None

                c.execute("""
                    INSERT OR REPLACE INTO signal_perf
                        (signal_id, ticker, entry_price,
                         price_1d, price_5d, price_10d, price_20d,
                         ret_1d, ret_5d, ret_10d, ret_20d, last_updated)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (sig["id"], ticker, ep, p1, p5, p10, p20,
                      ret(p1), ret(p5), ret(p10), ret(p20),
                      datetime.now().isoformat()))
                updated += 1
        except Exception as e:
            print(f"[update_signal_perf] {ticker}: {e}")

    conn.commit(); conn.close()
    return updated


def load_signals(signal_type: str = None, days_back: int = 90,
                 with_perf: bool = True) -> pd.DataFrame:
    """
    Carica segnali dal DB con performance opzionale.
    signal_type: 'EARLY'|'PRO'|'HOT'|'CONFLUENCE'|'SERAFINI'|'FINVIZ'|None (tutti)
    """
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn   = _conn()
    if with_perf:
        query = """
            SELECT s.*, p.price_1d, p.price_5d, p.price_10d, p.price_20d,
                   p.ret_1d, p.ret_5d, p.ret_10d, p.ret_20d
            FROM signals s
            LEFT JOIN signal_perf p ON s.id = p.signal_id
            WHERE s.scanned_at >= ?
        """
    else:
        query = "SELECT * FROM signals WHERE scanned_at >= ?"
    params = [cutoff]
    if signal_type:
        query  += " AND s.signal_type = ?" if with_perf else " AND signal_type = ?"
        params.append(signal_type)
    query += " ORDER BY s.scanned_at DESC" if with_perf else " ORDER BY scanned_at DESC"
    try:
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()


def signal_summary_stats(days_back: int = 90) -> pd.DataFrame:
    """
    Statistiche aggregate per tipo di segnale:
    win_rate, avg_ret_1d/5d/10d/20d, n_signals, n_with_perf
    """
    df = load_signals(days_back=days_back, with_perf=True)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for stype in df["signal_type"].unique():
        sub = df[df["signal_type"] == stype]
        row = {"Signal": stype, "N": len(sub)}
        for col, label in [("ret_1d","Avg +1d%"), ("ret_5d","Avg +5d%"),
                            ("ret_10d","Avg +10d%"), ("ret_20d","Avg +20d%")]:
            valid = sub[col].dropna()
            row[label]          = round(valid.mean(), 2) if len(valid) else None
            row[f"Win%_{col}"]  = round((valid > 0).mean() * 100, 1) if len(valid) else None
            row[f"N_{col}"]     = len(valid)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Avg +20d%", ascending=False, na_position="last")


# ─────────────────────────────────────────────────────────────────────────────
# Init automatico
# ─────────────────────────────────────────────────────────────────────────────
init_db()
