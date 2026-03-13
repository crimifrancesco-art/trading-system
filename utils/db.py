# -*- coding: utf-8 -*-
"""
db.py — v32.0
=============
Gestione DB:
- watchlist
- storico scansioni (scan_history)
- segnali per backtest (signals) con performance forward

Esporta:
- DB_PATH, init_db
- save_scan_history, load_scan_history, load_scan_snapshot
- save_signals, load_signals, signal_summary_stats, update_signal_performance
- funzioni watchlist (add_to_watchlist, load_watchlist, ecc.)
- cache_stats, cache_clear (stub / compatibilità)
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── DB Path ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent


def _get_db_path() -> Path:
    """Path fisso e scrivibile per il DB.

    Priorità:
    1. $TRADING_DB_PATH (variabile d'ambiente opzionale)
    2. /home/appuser/.trading_scanner/ (Streamlit Cloud)
    3. ~/.trading_scanner/ (locale / home generica)
    4. /tmp/ (fallback assoluto)
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
        Path.home() / ".trading_scanner" / "watchlist.db",    # locale
        Path("/tmp/trading_scanner_watchlist.db"),            # fallback
    ]

    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            # Verifica scrittura reale
            _t = p.with_suffix(".tmp")
            _t.write_text("test")
            _t.unlink()

            # Migrazione eventuale da /tmp
            _tmp = Path("/tmp/trading_scanner_watchlist.db")
            _old = Path("/tmp/watchlist.db")
            for _src in [_tmp, _old]:
                if _src != p and _src.exists() and _src.stat().st_size > 8192:
                    if (not p.exists()) or (p.stat().st_size < _src.stat().st_size):
                        try:
                            import shutil
                            shutil.copy2(_src, p)
                        except Exception:
                            pass
                    break
            return p
        except Exception:
            continue
    return Path("/tmp/trading_scanner_watchlist.db")


DB_PATH = _get_db_path()

# ── Init DB ────────────────────────────────────────────────────────────────


def _ensure_signals_table(conn: sqlite3.Connection) -> None:
    """Crea tabella signals se non esiste + migrazioni colonne."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            scanned_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            nome TEXT,
            signal_type TEXT,
            prezzo REAL,
            markets TEXT,
            rsi REAL,
            quality_score REAL,
            ser_score REAL,
            fv_score REAL,
            squeeze INTEGER,
            weekly_bull INTEGER,
            ret_1d REAL,
            ret_5d REAL,
            ret_10d REAL,
            ret_20d REAL,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    # Migrazione: aggiunge colonne mancanti (in caso di DB vecchio)
    for _col, _ctype in [
        ("nome", "TEXT"),
        ("rsi", "REAL"),
        ("quality_score", "REAL"),
        ("ser_score", "REAL"),
        ("fv_score", "REAL"),
        ("squeeze", "INTEGER"),
        ("weekly_bull", "INTEGER"),
        ("ret_1d", "REAL"),
        ("ret_5d", "REAL"),
        ("ret_10d", "REAL"),
        ("ret_20d", "REAL"),
        ("updated_at", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {_col} {_ctype}")
            conn.commit()
        except Exception:
            # colonna già presente
            pass


def init_db():
    """Crea tutte le tabelle necessarie se non esistono."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Tabella watchlist
    c.execute(
        """
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
        """
    )
    # Migrazione colonne aggiuntive
    for col_def in ["trend TEXT", "list_name TEXT"]:
        try:
            c.execute(f"ALTER TABLE watchlist ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    # Tabella scan_history
    c.execute(
        """
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
        """
    )
    for col_def in ["elapsed_s REAL", "cache_hits INTEGER DEFAULT 0"]:
        try:
            c.execute(f"ALTER TABLE scan_history ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    # Tabella signals per backtest
    _ensure_signals_table(conn)

    conn.commit()
    conn.close()


def reset_watchlist_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS watchlist")
    conn.commit()
    conn.close()

# ── Watchlist API ─────────────────────────────────────────────────────────


def add_to_watchlist(
    tickers,
    names,
    origine,
    note,
    trend: str = "LONG",
    list_name: str = "DEFAULT",
):
    if not tickers:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for t, n in zip(tickers, names):
        c.execute(
            """
            INSERT INTO watchlist
            (ticker, name, trend, origine, note, list_name, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (t, n, trend, origine, note, list_name, now),
        )
    conn.commit()
    conn.close()


def load_watchlist() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=[
                "id",
                "Ticker",
                "Nome",
                "trend",
                "origine",
                "note",
                "list_name",
                "created_at",
            ]
        )
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM watchlist ORDER BY created_at DESC", conn
        )
        conn.close()
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "Ticker"})
        if "name" in df.columns:
            df = df.rename(columns={"name": "Nome"})
        return df
    except Exception:
        return pd.DataFrame(
            columns=[
                "id",
                "Ticker",
                "Nome",
                "trend",
                "origine",
                "note",
                "list_name",
                "created_at",
            ]
        )


def update_watchlist_note(row_id, new_note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE watchlist SET note = ? WHERE id = ?",
        (new_note, int(row_id)),
    )
    conn.commit()
    conn.close()


def delete_from_watchlist(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "DELETE FROM watchlist WHERE id = ?",
        [(int(i),) for i in ids],
    )
    conn.commit()
    conn.close()


def move_watchlist_rows(ids, dest_list: str):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "UPDATE watchlist SET list_name = ? WHERE id = ?",
        [(dest_list, int(i)) for i in ids],
    )
    conn.commit()
    conn.close()


def rename_watchlist(old_name: str, new_name: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE watchlist SET list_name = ? WHERE list_name = ?",
        (new_name, old_name),
    )
    conn.commit()
    conn.close()


def reset_watchlist_by_name(list_name: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM watchlist WHERE list_name = ?", (list_name,))
    conn.commit()
    conn.close()

# ── Helpers JSON per scan_history ─────────────────────────────────────────


def _df_to_json_safe(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "[]"
    df2 = df.copy()
    drop_cols = [c for c in df2.columns if c.startswith("_")]
    df2 = df2.drop(columns=drop_cols, errors="ignore")
    for col in df2.columns:
        try:
            df2[col] = df2[col].apply(
                lambda x: bool(x)
                if isinstance(x, (np.bool_,))
                else float(x)
                if isinstance(x, np.floating)
                else int(x)
                if isinstance(x, np.integer)
                else None
                if isinstance(x, float) and (np.isnan(x) or np.isinf(x))
                else x
            )
        except Exception:
            pass
    try:
        return df2.to_json(orient="records", default_handler=str)
    except Exception:
        return "[]"

# ── Scan history API ──────────────────────────────────────────────────────


def save_scan_history(
    markets: list,
    df_ep: pd.DataFrame,
    df_rea: pd.DataFrame,
    elapsed_s: float = 0.0,
    cache_hits: int = 0,
) -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_early, n_pro, n_conf = 0, 0, 0
        n_rea = len(df_rea) if df_rea is not None and not df_rea.empty else 0
        if df_ep is not None and not df_ep.empty:
            if "StatoEarly" in df_ep.columns:
                n_early = int((df_ep["StatoEarly"] == "EARLY").sum())
            if "StatoPro" in df_ep.columns:
                n_pro = int((df_ep["StatoPro"] == "PRO").sum())
            if "StatoEarly" in df_ep.columns and "StatoPro" in df_ep.columns:
                n_conf = int(
                    ((df_ep["StatoEarly"] == "EARLY") & (df_ep["StatoPro"] == "PRO")).sum()
                )

        ep_json = _df_to_json_safe(df_ep)
        rea_json = _df_to_json_safe(df_rea)

        c.execute(
            """
            INSERT INTO scan_history (
                scanned_at, markets, n_early, n_pro, n_rea, n_confluence,
                df_ep_json, df_rea_json, elapsed_s, cache_hits
            )
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now,
                json.dumps(markets or []),
                n_early,
                n_pro,
                n_rea,
                n_conf,
                ep_json,
                rea_json,
                float(elapsed_s),
                int(cache_hits),
            ),
        )
        conn.commit()
        scan_id = c.lastrowid
        conn.close()
        return scan_id
    except Exception:
        import traceback

        traceback.print_exc()
        return 0


def load_scan_history(limit: int = 20) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT id, scanned_at, markets,
                   n_early, n_pro, n_rea, n_confluence,
                   elapsed_s, cache_hits
            FROM scan_history
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_scan_snapshot(scan_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT df_ep_json, df_rea_json FROM scan_history WHERE id = ?",
            (scan_id,),
        )
        row = c.fetchone()
        conn.close()
        if row:
            import io

            df_ep = (
                pd.read_json(io.StringIO(row[0]))
                if row[0] and row[0] != "[]"
                else pd.DataFrame()
            )
            df_rea = (
                pd.read_json(io.StringIO(row[1]))
                if row[1] and row[1] != "[]"
                else pd.DataFrame()
            )
            return df_ep, df_rea
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()

# ── Signals API (scanner → backtest) ───────────────────────────────────────


def save_signals(
    scan_id: int,
    df_ep: pd.DataFrame,
    df_rea: pd.DataFrame,
    markets: list,
) -> None:
    """Salva segnali EARLY / HOT / PRO ecc. nella tabella signals."""
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mkt = json.dumps(markets) if markets else "[]"

        rows = []
        # df_ep: contiene EARLY / PRO / CONFLUENCE (StatoEarly, StatoPro)
        if df_ep is not None and not df_ep.empty:
            for _, row in df_ep.iterrows():
                ticker = str(row.get("Ticker", "") or row.get("ticker", ""))
                if not ticker:
                    continue
                nome = str(
                    row.get("Nome", "")
                    or row.get("name", "")
                    or row.get("NomeTicker", "")
                    or ""
                )
                prezzo = float(row.get("Prezzo", 0) or 0)
                # Se hai colonne di tipo specifico, puoi codificarle, qui uso genericamente "EARLY"/"PRO"
                stato_early = str(row.get("StatoEarly", "")).upper()
                stato_pro = str(row.get("StatoPro", "")).upper()
                if stato_early == "EARLY" and stato_pro == "PRO":
                    stype = "CONFLUENCE"
                elif stato_pro == "PRO":
                    stype = "PRO"
                elif stato_early == "EARLY":
                    stype = "EARLY"
                else:
                    stype = "EARLY"  # fallback

                rsi_v = float(row.get("RSI", 0) or 0)
                qual_v = float(row.get("QualityScore", 0) or 0)
                ser_v = float(row.get("SerScore", 0) or 0)
                fv_v = float(row.get("FVScore", 0) or 0)
                sq_v = 1 if row.get("Squeeze") in [True, "True", "true", 1] else 0
                wb_v = 1 if row.get("WeeklyBull") in [True, "True", "true", 1] else 0

                rows.append(
                    (
                        scan_id,
                        now,
                        ticker,
                        nome,
                        stype,
                        prezzo,
                        mkt,
                        rsi_v,
                        qual_v,
                        ser_v,
                        fv_v,
                        sq_v,
                        wb_v,
                    )
                )

        # df_rea: HOT / REA-HOT
        if df_rea is not None and not df_rea.empty:
            for _, row in df_rea.iterrows():
                ticker = str(row.get("Ticker", "") or row.get("ticker", ""))
                if not ticker:
                    continue
                nome = str(
                    row.get("Nome", "")
                    or row.get("name", "")
                    or row.get("NomeTicker", "")
                    or ""
                )
                prezzo = float(row.get("Prezzo", 0) or 0)
                stype = "HOT"

                rsi_v = float(row.get("RSI", 0) or 0)
                qual_v = float(row.get("QualityScore", 0) or 0)
                ser_v = float(row.get("SerScore", 0) or 0)
                fv_v = float(row.get("FVScore", 0) or 0)
                sq_v = 1 if row.get("Squeeze") in [True, "True", "true", 1] else 0
                wb_v = 1 if row.get("WeeklyBull") in [True, "True", "true", 1] else 0

                rows.append(
                    (
                        scan_id,
                        now,
                        ticker,
                        nome,
                        stype,
                        prezzo,
                        mkt,
                        rsi_v,
                        qual_v,
                        ser_v,
                        fv_v,
                        sq_v,
                        wb_v,
                    )
                )

        if rows:
            conn.executemany(
                """
                INSERT INTO signals (
                    scan_id, scanned_at, ticker, nome, signal_type,
                    prezzo, markets, rsi, quality_score, ser_score,
                    fv_score, squeeze, weekly_bull
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            conn.commit()
        conn.close()
    except Exception:
        import traceback

        traceback.print_exc()


def load_signals(
    signal_type: str = None,
    days_back: int = 90,
    with_perf: bool = True,
) -> pd.DataFrame:
    """Carica segnali dal DB, opzionalmente filtrati per tipo e periodo."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)
        where = []
        params = []
        if signal_type and signal_type != "Tutti":
            where.append("signal_type = ?")
            params.append(signal_type)
        if days_back:
            where.append("scanned_at >= datetime('now', ?)")
            params.append(f"-{days_back} days")
        sql = "SELECT * FROM signals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scanned_at DESC"
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def signal_summary_stats(days_back: int = 90) -> pd.DataFrame:
    """Statistiche aggregate avanzate per tipo segnale e orizzonte."""
    df = load_signals(days_back=days_back, with_perf=True)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for stype, grp in df.groupby("signal_type"):
        n_all = len(grp)
        for col, label in [
            ("ret_1d", "1g"),
            ("ret_5d", "5g"),
            ("ret_10d", "10g"),
            ("ret_20d", "20g"),
        ]:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            if vals.empty:
                continue

            n = len(vals)
            win = (vals > 0).mean() * 100.0
            avg = vals.mean()
            med = vals.median()
            std = vals.std()
            p25 = vals.quantile(0.25)
            p75 = vals.quantile(0.75)
            vmax = vals.max()
            vmin = vals.min()
            sharpe = avg / std if std and std > 0 else 0.0

            rows.append(
                dict(
                    Signal=stype,
                    Periodo=label,
                    N=n,
                    N_tot=n_all,
                    Win=round(win, 1),
                    Avg=round(avg, 2),
                    Med=round(med, 2),
                    Std=round(std, 2) if std == std else None,
                    P25=round(p25, 2),
                    P75=round(p75, 2),
                    Max=round(vmax, 2),
                    Min=round(vmin, 2),
                    Sharpe=round(sharpe, 2),
                )
            )

    return pd.DataFrame(rows)


def update_signal_performance(max_signals: int = 300) -> int:
    """Aggiorna prezzi forward +1/5/10/20g per segnali senza performance completa."""
    if not DB_PATH.exists():
        return 0
    try:
        import yfinance as _yf
    except ImportError:
        return 0

    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_signals_table(conn)

        df = pd.read_sql_query(
            """
            SELECT * FROM signals
            WHERE ret_20d IS NULL
            ORDER BY scanned_at DESC
            LIMIT ?
            """,
            conn,
            params=(max_signals,),
        )
        if df.empty:
            conn.close()
            return 0

        updated = 0
        for _, row in df.iterrows():
            try:
                tkr = row["ticker"]
                date = pd.to_datetime(row["scanned_at"])
                p0 = float(row["prezzo"] or 0)
                if p0 <= 0:
                    continue

                hist = _yf.Ticker(tkr).history(
                    start=date.strftime("%Y-%m-%d"),
                    end=(date + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if hist.empty:
                    continue
                closes = hist["Close"].dropna()
                if len(closes) < 2:
                    continue

                def _ret(n: int) -> float:
                    idx = min(n, len(closes) - 1)
                    return round((float(closes.iloc[idx]) / p0 - 1) * 100, 2)

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    UPDATE signals
                    SET ret_1d=?, ret_5d=?, ret_10d=?, ret_20d=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        _ret(1),
                        _ret(5),
                        _ret(10),
                        _ret(20),
                        now,
                        int(row["id"]),
                    ),
                )
                updated += 1
            except Exception:
                continue

        conn.commit()
        conn.close()
        return updated
    except Exception:
        import traceback

        traceback.print_exc()
        return 0

# ── Cache info (stub compatibilità) ───────────────────────────────────────


def cache_stats():
    """Stub compatibilità: ritorna dizionario vuoto/placeholder."""
    return {"fresh": 0, "stale": 0, "size_mb": 0, "total_entries": 0}


def cache_clear(*a, **k):
    """Stub compatibilità: non fa nulla."""
    return None


# Inizializza DB all'import
init_db()
