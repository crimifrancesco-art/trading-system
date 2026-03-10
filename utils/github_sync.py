# -*- coding: utf-8 -*-
"""
github_sync.py  —  Watchlist GitHub Sync  v29.0
═══════════════════════════════════════════════════
Salva la watchlist come JSON in un repo GitHub privato.
Si integra SOPRA il db.py esistente: SQLite resta come cache locale,
GitHub è la fonte di verità persistente tra i deploy.

SETUP (una tantum):
  1. Crea un repo GitHub privato (es. "trading-watchlist-data")
  2. Genera un Personal Access Token con scope "repo":
     GitHub → Settings → Developer settings → Personal access tokens → Fine-grained
  3. In Streamlit Cloud → App settings → Secrets:

     [github]
     token = "ghp_xxxxxxxxxxxxxxxx"
     repo  = "tuousername/trading-watchlist-data"
     path  = "watchlist.json"   # percorso nel repo

  4. L'app sincronizza automaticamente ad ogni modifica watchlist.

ARCHITETTURA:
  ┌─────────────────────────────────────────┐
  │  App                                    │
  │  ├── add/delete/rename watchlist        │
  │  │     ↓ (immediato)                   │
  │  ├── SQLite locale (veloce, cache)      │
  │  │     ↓ (background, ~1s)             │
  │  └── GitHub JSON (persistente)          │
  └─────────────────────────────────────────┘

  Al BOOT dell'app:
  1. Tenta di caricare da GitHub → sovrascrive SQLite locale
  2. Se GitHub non disponibile → usa SQLite locale (degraded mode)
"""

import json
import base64
import threading
import time
import sqlite3
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────

def _get_github_config() -> Optional[dict]:
    """
    Legge la config GitHub da Streamlit secrets o variabili d'ambiente.
    Ritorna None se non configurato (modalità solo-SQLite).
    """
    try:
        import streamlit as st
        cfg = st.secrets.get("github", {})
        token = cfg.get("token","")
        repo  = cfg.get("repo","")
        path  = cfg.get("path","watchlist.json")
        if token and repo:
            return {"token": token, "repo": repo, "path": path}
    except Exception:
        pass
    # Fallback: variabili d'ambiente
    import os
    token = os.environ.get("GITHUB_TOKEN","")
    repo  = os.environ.get("GITHUB_REPO","")
    path  = os.environ.get("GITHUB_WL_PATH","watchlist.json")
    if token and repo:
        return {"token": token, "repo": repo, "path": path}
    return None


# ── GitHub API helpers ────────────────────────────────────────────────────

def _github_get(cfg: dict) -> Optional[dict]:
    """
    Scarica il file JSON dal repo GitHub.
    Ritorna dict con 'content' (decoded) e 'sha' (per l'update).
    """
    import urllib.request, urllib.error
    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['path']}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {cfg['token']}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "TradingScanner-v29",
    })
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            content = base64.b64decode(data["content"]).decode("utf-8")
            return {"content": json.loads(content), "sha": data["sha"]}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"content": [], "sha": None}   # file non esiste ancora
        raise
    except Exception:
        return None


def _github_put(cfg: dict, payload: list, sha: Optional[str]) -> bool:
    """
    Scrive/aggiorna il file JSON nel repo GitHub.
    sha=None → crea nuovo file, sha=str → aggiorna esistente.
    Ritorna True se successo.
    """
    import urllib.request, urllib.error
    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['path']}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = {
        "message": f"watchlist sync {now}",
        "content": base64.b64encode(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("ascii"),
    }
    if sha:
        body["sha"] = sha
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="PUT",
        headers={
            "Authorization": f"token {cfg['token']}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "TradingScanner-v29",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 201)
    except Exception:
        traceback.print_exc()
        return False


# ── SHA cache in-process (evita GET extra) ────────────────────────────────

_sha_cache: dict = {}   # {"repo/path": "sha_string"}


# ── SQLite → JSON export ──────────────────────────────────────────────────

def _sqlite_to_list(db_path: Path) -> list:
    """Esporta tutta la watchlist da SQLite come lista di dict."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ticker, name, trend, origine, note, list_name, created_at "
            "FROM watchlist ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _list_to_sqlite(db_path: Path, rows: list):
    """Importa una lista di dict nella watchlist SQLite (replace completo)."""
    if not db_path.exists():
        return
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM watchlist")
        if rows:
            conn.executemany(
                "INSERT INTO watchlist (ticker, name, trend, origine, note, list_name, created_at) "
                "VALUES (:ticker, :name, :trend, :origine, :note, :list_name, :created_at)",
                rows
            )
        conn.commit()
        conn.close()
    except Exception:
        traceback.print_exc()


# ── Core pubbliche ────────────────────────────────────────────────────────

def push_watchlist(db_path: Path) -> bool:
    """
    Legge SQLite e carica su GitHub in background.
    Non-blocking: lancia un thread e ritorna subito True.
    Ritorna False se GitHub non configurato.
    """
    cfg = _get_github_config()
    if not cfg:
        return False

    def _worker():
        try:
            payload = _sqlite_to_list(db_path)
            cache_key = f"{cfg['repo']}/{cfg['path']}"
            # Ottieni SHA attuale (necessario per update)
            if cache_key not in _sha_cache:
                result = _github_get(cfg)
                if result:
                    _sha_cache[cache_key] = result.get("sha")
            sha = _sha_cache.get(cache_key)
            ok = _github_put(cfg, payload, sha)
            if ok:
                # Aggiorna SHA cache dopo scrittura
                result2 = _github_get(cfg)
                if result2:
                    _sha_cache[cache_key] = result2.get("sha")
        except Exception:
            traceback.print_exc()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return True


def pull_watchlist(db_path: Path) -> tuple:
    """
    Scarica la watchlist da GitHub e sovrascrive SQLite.
    Chiamare al boot dell'app.
    Ritorna (ok: bool, n_rows: int, source: str)
    """
    cfg = _get_github_config()
    if not cfg:
        return False, 0, "no_github"

    try:
        result = _github_get(cfg)
        if result is None:
            return False, 0, "github_error"

        rows = result.get("content", [])
        cache_key = f"{cfg['repo']}/{cfg['path']}"
        _sha_cache[cache_key] = result.get("sha")

        if not isinstance(rows, list) or not rows:
            return True, 0, "github_empty"

        _list_to_sqlite(db_path, rows)
        return True, len(rows), "github"

    except Exception:
        traceback.print_exc()
        return False, 0, "exception"


def sync_status(db_path: Path) -> dict:
    """
    Ritorna lo stato della sync per mostrare nell'UI.
    """
    cfg = _get_github_config()
    if not cfg:
        return {
            "configured": False,
            "message": "⚠️ GitHub sync non configurato — watchlist solo locale",
            "color": "#f59e0b",
        }
    return {
        "configured": True,
        "repo": cfg["repo"],
        "path": cfg["path"],
        "message": f"☁️ Sync GitHub: `{cfg['repo']}/{cfg['path']}`",
        "color": "#26a69a",
    }


# ── Wrapper per db.py ─────────────────────────────────────────────────────
# Queste funzioni sostituiscono quelle di db.py con auto-sync GitHub.
# Importa da qui invece che da db.py per avere la persistenza.

def _get_db_path_safe():
    """Importa DB_PATH da db.py."""
    try:
        from utils.db import DB_PATH
        return DB_PATH
    except ImportError:
        from db import DB_PATH
        return DB_PATH


def gh_add_to_watchlist(tickers, names, origine, note,
                         trend="LONG", list_name="DEFAULT"):
    """add_to_watchlist + GitHub push."""
    try:
        from utils.db import add_to_watchlist
    except ImportError:
        from db import add_to_watchlist
    add_to_watchlist(tickers, names, origine, note, trend, list_name)
    push_watchlist(_get_db_path_safe())


def gh_delete_from_watchlist(ids):
    """delete_from_watchlist + GitHub push."""
    try:
        from utils.db import delete_from_watchlist
    except ImportError:
        from db import delete_from_watchlist
    delete_from_watchlist(ids)
    push_watchlist(_get_db_path_safe())


def gh_rename_watchlist(old_name, new_name):
    """rename_watchlist + GitHub push."""
    try:
        from utils.db import rename_watchlist
    except ImportError:
        from db import rename_watchlist
    rename_watchlist(old_name, new_name)
    push_watchlist(_get_db_path_safe())


def gh_move_watchlist_rows(ids, dest_list):
    """move_watchlist_rows + GitHub push."""
    try:
        from utils.db import move_watchlist_rows
    except ImportError:
        from db import move_watchlist_rows
    move_watchlist_rows(ids, dest_list)
    push_watchlist(_get_db_path_safe())


def gh_update_watchlist_note(row_id, new_note):
    """update_watchlist_note + GitHub push."""
    try:
        from utils.db import update_watchlist_note
    except ImportError:
        from db import update_watchlist_note
    update_watchlist_note(row_id, new_note)
    push_watchlist(_get_db_path_safe())


def gh_reset_watchlist_by_name(list_name):
    """reset_watchlist_by_name + GitHub push."""
    try:
        from utils.db import reset_watchlist_by_name
    except ImportError:
        from db import reset_watchlist_by_name
    reset_watchlist_by_name(list_name)
    push_watchlist(_get_db_path_safe())
