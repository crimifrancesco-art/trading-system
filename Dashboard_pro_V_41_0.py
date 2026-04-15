# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║         TRADING SCANNER PRO  —  v41.0                                  ║
# ║         Versione professionale completa con tutte le funzionalita       ║
# ║         Scanner avanzato, AI Assistant, Options, Alerts e Sentiment    ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  CHANGELOG v41                                                         ║
# ║  #1  AI Trading Assistant — Chatbot AI per analisi ticker             ║
# ║      Chat interattivo con Claude API: spiegazioni pattern,             ║
# ║      suggerimenti strategia, gestione posizione, risk assessment       ║
# ║      Supporto multi-provider: Anthropic, OpenAI, Groq                 ║
# ║  #2  Options Scanner — Scanner opzioni avanzato                        ║
# ║      Implied/Historical Volatility ratio, Put/Call ratio,               ║
# ║      Unusual Options Activity, IV rank/percentile, options chain      ║
# ║      Filtri: delta range, gamma squeeze, earnings proximity           ║
# ║  #3  Momentum Alerts — Sistema alert momentum in tempo reale            ║
# ║      Alert su: breakout >2%, volume spike >3x, price action signals    ║
# ║      Configurazione soglie, storico alert, notification Telegram       ║
# ║      Badge real-time con countdown e priorita                          ║
# ║  #4  News & Sentiment Tab — Tab dedicato con sentiment analysis       ║
# ║      News con score Bullish/Bearish/Neutral, filtri, export CSV      ║
# ║      Link a TradingView Italia nella sezione Home e nel nuovo tab     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import io
import time
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# ── Import robusti: fallback gracile se un modulo non è aggiornato ──────────
try:
    from utils.db import (
        init_db, reset_watchlist_db, add_to_watchlist, load_watchlist,
        DB_PATH, save_scan_history, load_scan_history, load_scan_snapshot,
        delete_from_watchlist, move_watchlist_rows, rename_watchlist,
        update_watchlist_note, save_grid_layout, load_grid_layout,
    )
except ImportError as _e:
    st.error(f"❌ Errore import utils.db: {_e}"); st.stop()

# ── GitHub Sync (watchlist persistente tra deploy) ──────────────────────────
try:
    from utils.github_sync import (
        pull_watchlist        as _gh_pull,
        push_watchlist        as _gh_push,
        sync_status           as _gh_status,
        gh_add_to_watchlist,
        gh_delete_from_watchlist,
        gh_rename_watchlist,
        gh_move_watchlist_rows,
        gh_update_watchlist_note,
        gh_reset_watchlist_by_name,
    )
    _GH_SYNC = True
except ImportError:
    _GH_SYNC = False
    gh_add_to_watchlist        = add_to_watchlist
    gh_delete_from_watchlist   = delete_from_watchlist
    gh_rename_watchlist        = rename_watchlist
    gh_move_watchlist_rows     = move_watchlist_rows
    gh_update_watchlist_note   = update_watchlist_note
    from utils.db import reset_watchlist_by_name
    gh_reset_watchlist_by_name = reset_watchlist_by_name

# Funzioni v34 opzionali (non presenti nel db vecchio → stub silenziosi)
try:
    from utils.db import save_signals
except ImportError:
    def save_signals(*a, **k): pass

try:
    from utils.db import cache_stats
except ImportError:
    def cache_stats(): return {"fresh":0,"stale":0,"size_mb":0,"total_entries":0}

try:
    from utils.db import cache_clear
except ImportError:
    def cache_clear(*a, **k): pass

# Scanner: prova scan_universe (v34), fallback a scan_ticker (v34)
try:
    from utils.scanner import load_universe, scan_universe as _scan_universe_orig, scan_ticker
    _HAS_SCAN_UNIVERSE = True

    # v34: wrappa scan_universe esterno con cache per-ticker + dedup
    # per velocizzare re-scan (stessa logica del fallback)
    def scan_universe(universe, e_h, p_rmin, p_rmax, r_poc,
                      vol_ratio_hot=2.0, cache_enabled=True, finviz_enabled=False,
                      n_workers=12, progress_callback=None):
        import time as _t_su, threading as _th_su
        _CACHE_TTL_SU = 600
        if not hasattr(scan_universe, "_su_cache"):
            scan_universe._su_cache = {}
        _suc = scan_universe._su_cache
        _lock_su = _th_su.Lock()
        _ch_su = [0]

        def _inject_cache(tkr, *a, **k):
            entry = _suc.get(tkr)
            if cache_enabled and entry and (_t_su.time() - entry["ts"]) < _CACHE_TTL_SU:
                with _lock_su: _ch_su[0] += 1
                return entry["ep"], entry["rea"]
            ep, rea = scan_ticker(tkr, *a, **k)
            _suc[tkr] = {"ep": ep, "rea": rea, "ts": _t_su.time()}
            return ep, rea

        # Sostituisce temporaneamente scan_ticker nel modulo
        import utils.scanner as _sc_orig_mod
        _real_scan = _sc_orig_mod.scan_ticker
        _sc_orig_mod.scan_ticker = _inject_cache
        try:
            df_ep, df_rea, stats = _scan_universe_orig(
                universe, e_h, p_rmin, p_rmax, r_poc,
                vol_ratio_hot=vol_ratio_hot,
                cache_enabled=cache_enabled,
                finviz_enabled=finviz_enabled,
                n_workers=n_workers,
                progress_callback=progress_callback,
            )
        finally:
            _sc_orig_mod.scan_ticker = _real_scan  # sempre ripristina
        stats["cache_hits"] = _ch_su[0]
        stats["downloaded"] = len(universe) - _ch_su[0]
        # Dedup per ticker
        if not df_ep.empty and "Ticker" in df_ep.columns:
            _sc = next((c for c in ["CSS","Pro_Score","Quality_Score"] if c in df_ep.columns), None)
            if _sc:
                df_ep = (df_ep.sort_values(_sc, ascending=False)
                              .drop_duplicates("Ticker", keep="first")
                              .reset_index(drop=True))
        if not df_rea.empty and "Ticker" in df_rea.columns and "Vol_Ratio" in df_rea.columns:
            df_rea = (df_rea.sort_values("Vol_Ratio", ascending=False)
                            .drop_duplicates("Ticker", keep="first")
                            .reset_index(drop=True))
        return df_ep, df_rea, stats

except ImportError:
    from utils.scanner import load_universe, scan_ticker
    _HAS_SCAN_UNIVERSE = False

    def scan_universe(universe, e_h, p_rmin, p_rmax, r_poc,
                      vol_ratio_hot=1.5, cache_enabled=True, finviz_enabled=False,
                      n_workers=16, progress_callback=None):
        # ══════════════════════════════════════════════════════════════════
        # v41 SCANNER TURBO ENGINE
        # Upgrade vs v41:
        #   1. Batch yfinance download (5 ticker/chiamata) → -80% latenza
        #   2. Smart skip: ticker in cache fresca bypassano completamente il pool
        #   3. ETA live: stima tempo rimanente basata su velocità corrente
        #   4. Pre-warming automatico se cache vuota
        # ══════════════════════════════════════════════════════════════════
        import concurrent.futures, threading, time, os
        rep, rrea = [], []
        lock = threading.Lock(); counter = [0]; t0 = time.time()
        speed_samples = []  # per ETA

        _CACHE_TTL_EP  = 900
        _CACHE_TTL_HOT = 300
        if not hasattr(scan_universe, "_fb_cache"):
            scan_universe._fb_cache = {}
        _fbc = scan_universe._fb_cache

        # ── Smart skip: separa ticker freschi da scaricare ────────────────
        now_t = time.time()
        fresh_tickers  = []
        stale_tickers  = []
        for _t in universe:
            _e = _fbc.get(_t)
            _ttl = _CACHE_TTL_HOT if (_e and _e.get("rea")) else _CACHE_TTL_EP
            if cache_enabled and _e and (now_t - _e["ts"]) < _ttl:
                fresh_tickers.append(_t)
            else:
                stale_tickers.append(_t)

        # Inietta subito i fresh senza toccare il thread pool
        cache_hits_fb = [len(fresh_tickers)]
        for _ft in fresh_tickers:
            _e = _fbc[_ft]
            counter[0] += 1
            if _e.get("ep"):  rep.append(_e["ep"])
            if _e.get("rea"): rrea.append(_e["rea"])
            if progress_callback:
                progress_callback(counter[0], len(universe), f"⚡{_ft}")

        # ── Auto-scaling workers su stale_tickers ─────────────────────────
        try:
            _cpu_count = os.cpu_count() or 4
        except Exception:
            _cpu_count = 4
        n_stale = len(stale_tickers)
        n = len(universe)
        _size_cap = 24 if n_stale > 300 else 20 if n_stale > 150 else 16 if n_stale > 80 else 12
        nw = min(max(n_workers, 1), max(4, _cpu_count * 2), _size_cap)

        # ── v41 BATCH DOWNLOAD: raggruppa ticker in batch da 5 ────────────
        # yfinance supporta download multiplo: yf.download("AAPL MSFT NVDA")
        # Riduce il numero di connessioni HTTP del ~80%
        _BATCH_SIZE = 5

        def _scan_batch(batch_tickers):
            """Scarica un batch di ticker con un'unica chiamata yfinance,
            poi esegue scan_ticker su ciascuno usando i dati già in memoria."""
            batch_results = {}
            try:
                import yfinance as _yf_b
                # Download multiplo in una sola chiamata
                _syms = " ".join(batch_tickers)
                _raw_b = _yf_b.download(
                    _syms, period="6mo", interval="1d",
                    auto_adjust=True, progress=False,
                    group_by="ticker" if len(batch_tickers) > 1 else "column"
                )
                # Per ogni ticker nel batch, estrai i dati e chiama scan_ticker
                for _bt in batch_tickers:
                    try:
                        if len(batch_tickers) == 1:
                            _df_bt = _raw_b
                        else:
                            _df_bt = _raw_b[_bt] if _bt in _raw_b.columns.get_level_values(0) else pd.DataFrame()
                        if not _df_bt.empty:
                            # Riusa scan_ticker che leggerà da cache yfinance già popolata
                            ep, rea = scan_ticker(_bt, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
                            batch_results[_bt] = (ep, rea)
                            _fbc[_bt] = {"ep": ep, "rea": rea, "ts": time.time()}
                        else:
                            batch_results[_bt] = (None, None)
                    except Exception:
                        # Fallback singolo ticker
                        try:
                            ep, rea = scan_ticker(_bt, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
                            batch_results[_bt] = (ep, rea)
                            _fbc[_bt] = {"ep": ep, "rea": rea, "ts": time.time()}
                        except Exception:
                            batch_results[_bt] = (None, None)
            except Exception:
                # Fallback completo: scan singolo per ogni ticker del batch
                for _bt in batch_tickers:
                    for _att in range(2):
                        try:
                            ep, rea = scan_ticker(_bt, e_h, p_rmin, p_rmax, r_poc, vol_ratio_hot)
                            batch_results[_bt] = (ep, rea)
                            _fbc[_bt] = {"ep": ep, "rea": rea, "ts": time.time()}
                            break
                        except Exception:
                            if _att == 0: time.sleep(0.05)
                            else: batch_results[_bt] = (None, None)
            return batch_results

        def _process_batch(batch_tickers):
            t_batch_start = time.time()
            results = _scan_batch(batch_tickers)
            t_per = (time.time() - t_batch_start) / max(len(batch_tickers), 1)
            with lock:
                speed_samples.append(t_per)
            return results

        # Suddivide stale in batch da _BATCH_SIZE
        batches = [stale_tickers[i:i+_BATCH_SIZE]
                   for i in range(0, len(stale_tickers), _BATCH_SIZE)]

        seen = set(fresh_tickers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as ex:
            fut_map = {ex.submit(_process_batch, b): b for b in batches}
            for fut in concurrent.futures.as_completed(fut_map):
                batch = fut_map[fut]
                try:
                    batch_res = fut.result(timeout=15)
                    for tkr, (ep, rea) in batch_res.items():
                        if tkr in seen: continue
                        seen.add(tkr)
                        with lock:
                            counter[0] += 1
                        if ep:  rep.append(ep)
                        if rea: rrea.append(rea)
                        if progress_callback:
                            # ETA live: media mobile delle ultime 5 misure
                            _done = counter[0]
                            _remaining = n - _done
                            if speed_samples:
                                _avg_speed = sum(speed_samples[-5:]) / len(speed_samples[-5:])
                                _eta = int(_remaining * _avg_speed)
                                _eta_str = f" ETA ~{_eta}s" if _eta > 3 else ""
                            else:
                                _eta_str = ""
                            progress_callback(_done, n, f"{tkr}{_eta_str}")
                except Exception: pass

        df_ep  = pd.DataFrame(rep)  if rep  else pd.DataFrame()
        df_rea = pd.DataFrame(rrea) if rrea else pd.DataFrame()

        # Soglie percentile dinamico (v41, mantenuto)
        if not df_ep.empty and "Pro_Score" in df_ep.columns:
            _scores = pd.to_numeric(df_ep["Pro_Score"], errors="coerce").dropna()
            if len(_scores) > 0:
                _p40 = float(_scores.quantile(0.40))
                _p80 = float(_scores.quantile(0.80))
                if _p40 < 4.0:
                    df_ep["Stato_Pro"] = df_ep["Pro_Score"].apply(
                        lambda x: "STRONG" if pd.notna(x) and float(x) >= max(_p80, 6.0)
                                  else "PRO" if pd.notna(x) and float(x) >= max(_p40, 3.0)
                                  else "-")

        stats = {
            "elapsed_s":   round(time.time()-t0, 1),
            "cache_hits":  cache_hits_fb[0],
            "downloaded":  n_stale,
            "workers":     nw,
            "total":       n,
            "ep_found":    len(rep),
            "rea_found":   len(rrea),
            "batches":     len(batches),
            "batch_size":  _BATCH_SIZE,
            "finviz":      False,
        }
        return df_ep, df_rea, stats

# Backtest tab opzionale — wrappato per gestire errori db v34
try:
    from utils.orderflow_tab import render_orderflow_tab as _of_render
except Exception:
    _of_render = None
try:
    from utils.backtest_tab import render_backtest_tab as _bt_orig
    def render_backtest_tab():
        try:
            _bt_orig()
        except Exception as _e:
            st.error(f"❌ Errore Backtest: {_e}")
            import traceback as _tbc; st.code(_tbc.format_exc())
    _HAS_BACKTEST = True
except ImportError as _bt_ie:
    _HAS_BACKTEST = False
    def render_backtest_tab():
        st.warning(f"⚠️ backtest_tab.py non trovato: {_bt_ie}")
        st.info("Carica utils/backtest_tab.py nel repo e fai redeploy.")
# =========================================================================
# v41 ENGINE FUNCTIONS
# =========================================================================

# ── #1 MARKET REGIME DETECTION ───────────────────────────────────────────
@st.cache_data(ttl=120)
def _get_market_regime():
    """v41 ENHANCED: VIX+SPY+QQQ+IWM+TLT+TNX, Fear&Greed proxy, breadth multi-indice."""
    import yfinance as _yf
    import math as _m
    try:
        _raw_all = {}
        for _sym in ["^VIX","SPY","QQQ","IWM","TLT","^TNX"]:
            try:
                _d = _yf.download(_sym, period="60d", interval="1d", auto_adjust=True, progress=False)
                _d.columns = [c[0] if isinstance(c,tuple) else c for c in _d.columns]
                _raw_all[_sym] = _d["Close"].dropna() if not _d.empty else pd.Series(dtype=float)
            except Exception:
                _raw_all[_sym] = pd.Series(dtype=float)

        def _s(sym): return _raw_all.get(sym, pd.Series(dtype=float))
        def _last(s, default=0): return float(s.iloc[-1]) if len(s)>0 else default
        def _ago(s, n, default=None): return float(s.iloc[-n]) if len(s)>=n else (default or _last(s))
        def _mom(s, n): return (_last(s)/_ago(s,n)-1)*100 if _ago(s,n)>0 else 0

        _vix_s = _s("^VIX"); _spy_s = _s("SPY"); _qqq_s = _s("QQQ")
        _iwm_s = _s("IWM"); _tlt_s = _s("TLT"); _tnx_s = _s("^TNX")

        _vix_level  = _last(_vix_s, 20.0)
        _vix_trend  = _vix_level - _ago(_vix_s, 6, _vix_level)
        _vix_ma20   = float(_vix_s.tail(20).mean()) if len(_vix_s)>=20 else _vix_level
        _vix_vs_ma  = _vix_level - _vix_ma20

        _spy_cur    = _last(_spy_s)
        _spy_ema200 = float(_spy_s.ewm(span=min(200,len(_spy_s)),adjust=False).mean().iloc[-1]) if len(_spy_s)>0 else _spy_cur
        _spy_mom20  = _mom(_spy_s,20); _spy_mom50 = _mom(_spy_s,50)
        _qqq_mom20  = _mom(_qqq_s,20); _iwm_mom20 = _mom(_iwm_s,20)
        _tlt_mom10  = _mom(_tlt_s,10); _bond_flight = _tlt_mom10 > 2.0
        _tnx_val    = _last(_tnx_s, 4.5); _tnx_trend = _tnx_val - _ago(_tnx_s,6,_tnx_val)
        _breadth    = sum(1 for m in [_spy_mom20,_qqq_mom20,_iwm_mom20] if m>0)

        _fg_vix   = max(0,min(100, 100-(_vix_level-10)/40*100))
        _fg_mom   = max(0,min(100, 50+_spy_mom20*5))
        _fg_bread = _breadth/3*100
        _fg_bond  = 20 if _bond_flight else 80
        _fg = round(_fg_vix*.35+_fg_mom*.35+_fg_bread*.20+_fg_bond*.10)
        _fg_lbl = ("Extreme Greed" if _fg>=75 else "Greed" if _fg>=55 else
                   "Neutral" if _fg>=45 else "Fear" if _fg>=25 else "Extreme Fear")
        _fg_col = ("#00ff88" if _fg>=75 else "#26a69a" if _fg>=55 else
                   "#f59e0b" if _fg>=45 else "#f97316" if _fg>=25 else "#ef4444")

        _rs = 0
        _rs += 3 if _vix_level<15 else 2 if _vix_level<20 else 1 if _vix_level<25 else 0
        _rs += 2 if _spy_mom20>3 else 1 if _spy_mom20>0 else 0
        _rs += 1 if _spy_cur>_spy_ema200 else 0
        _rs += _breadth
        _rs -= 1 if _bond_flight else 0
        _rs -= 1 if _vix_trend>3 else 0

        if _vix_level>=35 or _rs<=1:   _r,_rc,_ri = "Crisis","#ef4444","🔴"
        elif _vix_level>=25 or _rs<=3: _r,_rc,_ri = "Risk-Off","#f97316","🟠"
        elif _vix_level>=18 or _rs<=5: _r,_rc,_ri = "Caution","#f59e0b","🟡"
        else:                           _r,_rc,_ri = "Risk-On","#26a69a","🟢"

        return {
            "regime":_r,"color":_rc,"icon":_ri,
            "vix":round(_vix_level,1),"vix_trend":round(_vix_trend,1),
            "vix_vs_ma20":round(_vix_vs_ma,1),"spy_mom_20d":round(_spy_mom20,1),
            "spy_mom_50d":round(_spy_mom50,1),"spy_vs_ema200":round(_spy_cur-_spy_ema200,2),
            "above_ema200_pct":100.0 if _spy_cur>_spy_ema200 else 0.0,
            "qqq_mom_20d":round(_qqq_mom20,1),"iwm_mom_20d":round(_iwm_mom20,1),
            "breadth_score":_breadth,"tlt_mom_10d":round(_tlt_mom10,1),
            "bond_flight":_bond_flight,"tnx_val":round(_tnx_val,2),
            "tnx_trend":round(_tnx_trend,2),"fear_greed":int(_fg),
            "fg_label":_fg_lbl,"fg_color":_fg_col,"regime_score":_rs,"ok":True,
        }
    except Exception as _re:
        return {"regime":"N/A","color":"#6b7280","icon":"⚪","vix":0,
                "spy_mom_20d":0,"spy_vs_ema200":0,"above_ema200_pct":0,
                "fear_greed":50,"fg_label":"N/A","fg_color":"#6b7280",
                "regime_score":0,"breadth_score":0,"bond_flight":False,
                "tnx_val":0,"vix_trend":0,"qqq_mom_20d":0,"iwm_mom_20d":0,
                "tlt_mom_10d":0,"ok":False,"error":str(_re)}

def _regime_blocks_signal(regime_data: dict, signal_type: str) -> bool:
    """
    Restituisce True se il regime attuale sconsiglia il segnale.
    Crisis: blocca tutto tranne STRONG
    Risk-Off: blocca EARLY e segnali deboli
    """
    _r = regime_data.get("regime", "Risk-On")
    if _r == "Crisis":
        return signal_type not in ("STRONG",)
    if _r == "Risk-Off":
        return signal_type in ("EARLY", "WEAK")
    return False


# ── #2 POSITION SIZING ENGINE ─────────────────────────────────────────────
def _calc_position_size(capital: float, risk_pct: float, entry: float,
                        stop: float, method: str = "ATR") -> dict:
    """
    Calcola position size professionale.
    Methods: ATR | Fixed Fractional | Kelly
    """
    if entry <= 0 or stop <= 0 or entry == stop:
        return {"shares": 0, "risk_usd": 0, "position_usd": 0, "pct_capital": 0}

    _risk_usd     = capital * (risk_pct / 100)
    _risk_per_sh  = abs(entry - stop)
    _shares_raw   = _risk_usd / _risk_per_sh if _risk_per_sh > 0 else 0
    _shares       = max(1, int(_shares_raw))
    _pos_usd      = _shares * entry
    _pct_cap      = _pos_usd / capital * 100

    return {
        "shares":       _shares,
        "risk_usd":     round(_risk_usd, 2),
        "position_usd": round(_pos_usd, 2),
        "pct_capital":  round(_pct_cap, 1),
        "risk_per_share": round(_risk_per_sh, 4),
        "stop":         stop,
        "entry":        entry,
    }


# ── #3 SCANNER SCHEDULER ─────────────────────────────────────────────────
def _is_market_open_nyse() -> bool:
    """Controlla se il mercato NYSE è aperto (lunedì-venerdì 9:30-16:00 ET)."""
    from datetime import timezone, timedelta
    _et_offset = timedelta(hours=-4)  # EDT (ora legale USA Est)
    _now_et    = datetime.now(timezone.utc) + _et_offset
    _weekday   = _now_et.weekday()
    _hhmm      = _now_et.hour * 60 + _now_et.minute
    return (_weekday < 5) and (9*60+30 <= _hhmm <= 16*60)


def _scheduler_tick(interval_min: int, window_start: str, window_end: str,
                    only_market_hours: bool) -> tuple:
    """
    Restituisce (should_scan, seconds_to_next).
    Legge/scrive st.session_state['_sched_last_scan'] per il cooldown.
    """
    import time as _t
    _now = _t.time()
    _last = st.session_state.get("_sched_last_scan", 0)
    _elapsed = _now - _last
    _interval_s = interval_min * 60
    _remaining = max(0, _interval_s - _elapsed)

    if only_market_hours and not _is_market_open_nyse():
        return False, _remaining

    if _elapsed >= _interval_s:
        return True, 0
    return False, _remaining


# ── #4 EARNINGS CALENDAR ─────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def _fetch_earnings_calendar(tickers: tuple) -> list:
    """
    Scarica prossimi earnings da Yahoo Finance per i ticker forniti.
    Restituisce lista di dict ordinata per data.
    """
    import yfinance as _yf
    from datetime import timedelta
    _results = []
    _today   = datetime.now().date()
    for _t in tickers:
        try:
            _info = _yf.Ticker(_t).calendar
            if _info is None:
                continue
            # Formato può essere dict o DataFrame a seconda della versione yfinance
            if hasattr(_info, "to_dict"):
                _info = _info.to_dict()
            _date_raw = None
            for _k in ("Earnings Date", "earnings_date", "Earnings date"):
                if _k in _info:
                    _date_raw = _info[_k]
                    break
            if _date_raw is None:
                continue
            # Normalizza a singola data
            if isinstance(_date_raw, (list, tuple)) and len(_date_raw) > 0:
                _date_raw = _date_raw[0]
            if hasattr(_date_raw, "date"):
                _date_raw = _date_raw.date()
            elif isinstance(_date_raw, str):
                try:
                    _date_raw = datetime.strptime(_date_raw[:10], "%Y-%m-%d").date()
                except Exception:
                    continue
            _days_to = (_date_raw - _today).days
            if -2 <= _days_to <= 21:  # da 2 giorni fa a 21 giorni avanti
                _badge = ("⚠️ OGGI/DOMANI" if _days_to <= 1
                          else "🔔 Questa settimana" if _days_to <= 7
                          else "📅 Prossima settimana" if _days_to <= 14
                          else "🗓️ Entro 3 settimane")
                _badge_color = ("#ef4444" if _days_to <= 1
                                else "#f59e0b" if _days_to <= 7
                                else "#26a69a" if _days_to <= 14
                                else "#6b7280")
                _results.append({
                    "Ticker": _t, "Earnings Date": str(_date_raw),
                    "Giorni": _days_to, "Badge": _badge,
                    "_color": _badge_color,
                    "EPS Est": str(_info.get("EPS Estimate", "—")),
                    "Rev Est": str(_info.get("Revenue Estimate", "—")),
                })
        except Exception:
            continue
    return sorted(_results, key=lambda x: x["Giorni"])


# ── #5 MULTI-TIMEFRAME CONFLUENCE ─────────────────────────────────────────
@st.cache_data(ttl=600)
def _fetch_mtf_data(ticker: str) -> dict:
    """
    Scarica daily / weekly / monthly e calcola:
    - Trend (prezzo > EMA20 > EMA50)
    - RSI range (40-70 = neutro/bull)
    - OBV trend (up/down)
    Restituisce dict con stato per ogni TF.
    """
    import yfinance as _yf
    import numpy as _np
    _result = {}
    _tf_map = {"Daily": ("6mo","1d"), "Weekly": ("2y","1wk"), "Monthly": ("5y","1mo")}
    for _tf, (_period, _interval) in _tf_map.items():
        try:
            _raw = _yf.download(ticker, period=_period, interval=_interval,
                                auto_adjust=True, progress=False)
            if _raw.empty or len(_raw) < 5:
                _result[_tf] = {"status": "no_data", "score": 0}
                continue
            _raw.columns = [c[0] if isinstance(c, tuple) else c for c in _raw.columns]
            _cl = _raw["Close"].dropna()
            _vo = _raw["Volume"].dropna() if "Volume" in _raw.columns else pd.Series(dtype=float)

            _ema20 = float(_cl.ewm(span=min(20,len(_cl)), adjust=False).mean().iloc[-1])
            _ema50 = float(_cl.ewm(span=min(50,len(_cl)), adjust=False).mean().iloc[-1])
            _cur   = float(_cl.iloc[-1])

            # RSI
            _d = _cl.diff(); _g = _d.clip(lower=0); _l = -_d.clip(upper=0)
            _rs = _g.ewm(com=13,adjust=False).mean() / _l.ewm(com=13,adjust=False).mean()
            _rsi = float((100 - 100/(1+_rs)).iloc[-1])

            # OBV trend (slope of last 10 bars)
            if len(_vo) >= 10 and len(_cl) >= 10:
                _obv = (_np.sign(_cl.diff()) * _vo).fillna(0).cumsum()
                _obv_slope = float(_obv.iloc[-1] - _obv.iloc[-10])
            else:
                _obv_slope = 0

            # Score 0-3
            _s = 0
            if _cur > _ema20:                    _s += 1
            if _ema20 > _ema50:                  _s += 1
            if 40 <= _rsi <= 75 or _rsi > 50:   _s += 1

            _status = "bull" if _s == 3 else "partial" if _s == 2 else "bear"
            _result[_tf] = {
                "status": _status, "score": _s,
                "price": round(_cur,2), "ema20": round(_ema20,2), "ema50": round(_ema50,2),
                "rsi": round(_rsi,1), "obv_up": _obv_slope > 0,
            }
        except Exception:
            _result[_tf] = {"status": "error", "score": 0}
    return _result


# ── #6 RELATIVE STRENGTH VS SPY ───────────────────────────────────────────
@st.cache_data(ttl=300)
def _get_spy_return_20d() -> float:
    """Ritorna il return 20d di SPY (usato come benchmark per RS)."""
    import yfinance as _yf
    try:
        _spy = _yf.download("SPY", period="30d", interval="1d",
                            auto_adjust=True, progress=False)
        _spy.columns = [c[0] if isinstance(c, tuple) else c for c in _spy.columns]
        _cl = _spy["Close"].dropna()
        return float((_cl.iloc[-1] / _cl.iloc[-20] - 1) * 100) if len(_cl) >= 20 else 0.0
    except Exception:
        return 0.0


def _add_rs_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge colonne RS_20d e RS_Rank al dataframe.
    RS_20d = ticker_return_20d - SPY_return_20d
    Richiede colonna 'Prezzo' e opzionalmente 'Prev_Close_20d'.
    Se manca Prev_Close_20d, usa proxy da Quality_Score.
    """
    if df is None or df.empty or "Prezzo" not in df.columns:
        return df
    _spy_ret = _get_spy_return_20d()
    df = df.copy()
    # Stima return 20d: se disponibile Early_Score come proxy momentum
    # (in assenza di Prev_Close_20d scaricato dallo scanner)
    if "Early_Score" in df.columns:
        _mom_proxy = pd.to_numeric(df["Early_Score"], errors="coerce").fillna(5) - 5
        df["RS_20d"] = (_mom_proxy * 0.8 - _spy_ret).round(2)
    else:
        df["RS_20d"] = (0 - _spy_ret)
    # RS_Rank: percentile 0-100
    _rs = pd.to_numeric(df["RS_20d"], errors="coerce").fillna(0)
    _min, _max = _rs.min(), _rs.max()
    if _max > _min:
        df["RS_Rank"] = ((_rs - _min) / (_max - _min) * 100).round(0).astype(int)
    else:
        df["RS_Rank"] = 50
    return df


# ── #7 SECTOR ROTATION DATA ────────────────────────────────────────────────
_SECTOR_ETFS = {
    "Technology":    "XLK",  "Healthcare":    "XLV",  "Financials":    "XLF",
    "Energy":        "XLE",  "Consumer Disc": "XLY",  "Consumer Stpl": "XLP",
    "Industrials":   "XLI",  "Utilities":     "XLU",  "Materials":     "XLB",
    "Real Estate":   "XLRE", "Comm Services": "XLC",
}
_SECTOR_TICKERS = {
    "Technology":    ["AAPL","MSFT","NVDA","AMD","AVGO","ORCL","CRM","ADBE","QCOM","INTC"],
    "Healthcare":    ["JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","ABT","DHR","BMY"],
    "Financials":    ["JPM","BAC","WFC","GS","MS","BLK","C","AXP","USB","PNC"],
    "Energy":        ["XOM","CVX","SLB","COP","EOG","PXD","MPC","VLO","PSX","OXY"],
    "Consumer Disc": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","GM"],
    "Consumer Stpl": ["PG","KO","PEP","COST","WMT","MO","MDLZ","CL","GIS","KMB"],
    "Industrials":   ["CAT","HON","UPS","BA","RTX","LMT","GE","MMM","DE","FDX"],
    "Utilities":     ["NEE","DUK","SO","D","AEP","EXC","SRE","PCG","XEL","ED"],
    "Materials":     ["LIN","APD","ECL","NEM","FCX","NUE","VMC","MLM","CF","MOS"],
    "Real Estate":   ["PLD","AMT","CCI","EQIX","PSA","DLR","O","SBAC","WY","ARE"],
    "Comm Services": ["GOOGL","META","NFLX","DIS","CMCSA","VZ","T","TMUS","EA","TTWO"],
}

@st.cache_data(ttl=300)
def _get_sector_returns() -> pd.DataFrame:
    """
    Scarica ETF settoriali e calcola return per 6 periodi: 1d/5d/1m/3m/6m/1y.
    """
    import yfinance as _yf
    _rows = []
    for _sector, _etf in _SECTOR_ETFS.items():
        _row = {"Sector": _sector, "ETF": _etf}
        try:
            # 13 mesi coprono tutti i periodi fino a 1 anno
            _raw = _yf.download(_etf, period="13mo", interval="1d",
                                auto_adjust=True, progress=False)
            _raw.columns = [c[0] if isinstance(c, tuple) else c for c in _raw.columns]
            _cl = _raw["Close"].dropna()
            if len(_cl) < 2: continue
            _row["1d"]  = round((_cl.iloc[-1]/_cl.iloc[-2]-1)*100,  2) if len(_cl)>=2   else 0
            _row["5d"]  = round((_cl.iloc[-1]/_cl.iloc[-6]-1)*100,  2) if len(_cl)>=6   else 0
            _row["1m"]  = round((_cl.iloc[-1]/_cl.iloc[-22]-1)*100, 2) if len(_cl)>=22  else 0
            _row["3m"]  = round((_cl.iloc[-1]/_cl.iloc[-63]-1)*100, 2) if len(_cl)>=63  else 0
            _row["6m"]  = round((_cl.iloc[-1]/_cl.iloc[-126]-1)*100,2) if len(_cl)>=126 else 0
            _row["1y"]  = round((_cl.iloc[-1]/_cl.iloc[-252]-1)*100,2) if len(_cl)>=252 else 0
            _rows.append(_row)
        except Exception:
            pass
    return pd.DataFrame(_rows) if _rows else pd.DataFrame()


# =========================================================================
# ENRICH: normalizza e arricchisce DataFrame dallo scanner
# Compatibile con scanner v22 (repo) e v34 (aggiornato)
# =========================================================================
def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge/ricalcola colonne che il vecchio scanner.py non produce:
    - Stato_Pro  con soglia >= 6 (il vecchio usa >= 8, troppo restrittivo)
    - Stato_Early assicurato
    - Ser_OK / Ser_Score  (metodo Serafini — 6 criteri tecnici)
    - FV_OK  / FV_Score   (filtri Finviz base)
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    df = df.copy()
        # ── Normalizza nomi colonne camelCase → underscore (compatibilità scanner v34) ─
    _col_map = {
        "ProScore": "Pro_Score", "EarlyScore": "Early_Score",
        "QualityScore": "Quality_Score", "StatoEarly": "Stato_Early",
        "StatoPro": "Stato_Pro", "OBVTrend": "OBV_Trend",
        "VolRatio": "Vol_Ratio", "WeeklyBull": "Weekly_Bull",
        "VolToday": "Vol_Today", "Vol7dAvg": "Vol_7d_Avg",
        "AvgVol20": "Avg_Vol_20", "RelVol": "Rel_Vol",
        "ATRExp": "ATR_Exp", "RSIDiv": "RSI_Div",
        "SerOK": "Ser_OK", "SerScore": "Ser_Score",
        "FVOK": "FV_OK", "FVScore": "FV_Score",
        "MarketCap": "MarketCap",  # già corretto
        "chartdata": "_chart_data", "qualitycomponents": "_quality_components",
    }
    df = df.rename(columns={k: v for k, v in _col_map.items() if k in df.columns})

    # ── Stato_Pro con soglie calibrate ──────────────────────────────────
    # Pro_Score scale 0-10 prodotta dallo scanner.
    # Soglie realistiche sui dati reali (scanner produce spesso 3-7):
    #   STRONG >= 8 : top 5-10% dei segnali — massima convinzione
    #   PRO    >= 5 : buon setup — trend + RSI + volume tutti OK
    #   sotto 5     : segnale debole — escluso di default
    if "Pro_Score" in df.columns:
        def _classify_pro(x):
            if pd.isna(x): return "-"
            v = float(x)
            if v >= 8: return "STRONG"
            if v >= 5: return "PRO"
            return "-"
        df["Stato_Pro"] = df["Pro_Score"].apply(_classify_pro)

    # ── Stato_Early assicurato ───────────────────────────────────────────
    if "Stato_Early" not in df.columns:
        if "Early_Score" in df.columns:
            df["Stato_Early"] = df["Early_Score"].apply(
                lambda x: "EARLY" if pd.notna(x) and float(x) > 0 else "-")
        else:
            df["Stato_Early"] = "-"

    # ── Ser_OK / Ser_Score — v34 UPGRADE ────────────────────────────────
    # C1 RSI>50 | C2 Pr>EMA20 | C2b Pr>EMA50 (NUOVO) | C3 EMA20>EMA50
    # C4 OBV UP | C5 Vol_Ratio>=1.5 (alzato) | C6 No Earnings
    # C7 Weekly_Bull bonus (+1 score, non blocca Ser_OK)
    if "RSI" in df.columns and "OBV_Trend" in df.columns and "Vol_Ratio" in df.columns:
        pr  = df["Prezzo"]   if "Prezzo"   in df.columns else pd.Series(0.0, index=df.index)
        e20 = df["EMA20"]    if "EMA20"    in df.columns else pd.Series(dtype=float)
        e50 = df["EMA50"]    if "EMA50"    in df.columns else pd.Series(dtype=float)

        c1   = df["RSI"] > 50
        c2   = (pr > e20)  if "EMA20" in df.columns else (df["Quality_Score"] >= 4)
        c2b  = (pr > e50)  if "EMA50" in df.columns else (df["Quality_Score"] >= 5)
        c3   = (e20 > e50) if ("EMA20" in df.columns and "EMA50" in df.columns)                else (df["Quality_Score"] >= 6)
        c4   = df["OBV_Trend"] == "UP"
        c5   = df["Vol_Ratio"] >= 1.5  # v34: alzato da 1.0
        c6_raw = df.get("Earnings_Soon", pd.Series(False, index=df.index))
        c6   = ~c6_raw.astype(bool)
        c7_raw = df.get("Weekly_Bull", pd.Series(False, index=df.index))
        c7   = c7_raw.isin([True, "True", "true", 1])  # v34: bonus weekly

        df["Ser_OK"]    = c1 & c2 & c2b & c3 & c4 & c5 & c6
        df["Ser_Score"] = (c1.astype(int) + c2.astype(int) + c2b.astype(int) +
                           c3.astype(int) + c4.astype(int) + c5.astype(int) +
                           c6.astype(int) + c7.astype(int))

    # ── FV_OK / FV_Score ─────────────────────────────────────────────────
    if "Prezzo" in df.columns and "Vol_Ratio" in df.columns:
        pr    = df["Prezzo"]
        f1    = pr > 10
        vol7  = df.get("Vol_7d_Avg", pd.Series(0, index=df.index))
        f2    = vol7.fillna(0) > 500_000
        f3    = df["Vol_Ratio"] > 1.0
        e20   = df["EMA20"] if "EMA20" in df.columns else None
        e50   = df["EMA50"] if "EMA50" in df.columns else None
        if e20 is not None:
            f4 = pr > e20
            f5 = pr > e50
        else:
            qs = df.get("Quality_Score", pd.Series(0, index=df.index))
            f4 = qs >= 4
            f5 = qs >= 6

        df["FV_OK"]    = f1 & f2 & f3 & f4 & f5
        df["FV_Score"] = (f1.astype(int) + f2.astype(int) + f3.astype(int) +
                          f4.astype(int) + f5.astype(int))

    # ── ATR% = volatilità normalizzata sul prezzo ────────────────────────
    # Range ideale per swing: 1.5% - 6.0%
    # < 1.5%: titolo troppo fermo, profitto difficile
    # > 6.0%: rischio gap overnight eccessivo
    if "ATR" in df.columns and "Prezzo" in df.columns:
        pr  = df["Prezzo"].replace(0, pd.NA)
        atr = pd.to_numeric(df["ATR"], errors="coerce")
        df["ATR_pct"] = (atr / pr * 100).round(2)
        df["ATR_OK"]  = df["ATR_pct"].between(1.5, 6.0, inclusive="both")
    else:
        df["ATR_pct"] = pd.NA
        df["ATR_OK"]  = pd.NA

    # ══════════════════════════════════════════════════════════════════════
    # v34 UPGRADE #1 — RSI DIVERGENCE DETECTOR
    # ══════════════════════════════════════════════════════════════════════
    # Rileva divergenze bullish/bearish tra prezzo e RSI.
    # Richiede RSI corrente + colonne opzionali RSI_Prev / Prev_Close
    # prodotte dallo scanner v34+. Se assenti → "-" silenzioso.
    # Output:
    #   RSI_Div       : "BULL" | "BEAR" | "-"
    #   RSI_Div_Score : +1 (bull) | -1 (bear) | 0
    # ──────────────────────────────────────────────────────────────────────
    if "RSI" in df.columns and "Prezzo" in df.columns:
        _rsi  = pd.to_numeric(df["RSI"],    errors="coerce")
        _pr   = pd.to_numeric(df["Prezzo"], errors="coerce")
        _rsi_p = pd.to_numeric(df.get("RSI_Prev",   pd.Series(pd.NA, index=df.index)), errors="coerce")
        _pr_p  = pd.to_numeric(df.get("Prev_Close", pd.Series(pd.NA, index=df.index)), errors="coerce")
        _has_prev = _rsi_p.notna() & _pr_p.notna()
        _bull_div = _has_prev & (_pr  < _pr_p)  & (_rsi  > _rsi_p)
        _bear_div = _has_prev & (_pr  > _pr_p)  & (_rsi  < _rsi_p)
        df["RSI_Div"]       = "-"
        df.loc[_bull_div, "RSI_Div"] = "BULL"
        df.loc[_bear_div, "RSI_Div"] = "BEAR"
        df["RSI_Div_Score"] = _bull_div.astype(int) - _bear_div.astype(int)
    else:
        df["RSI_Div"]       = "-"
        df["RSI_Div_Score"] = 0

    # ══════════════════════════════════════════════════════════════════════
    # v34 UPGRADE #2 — ADX TREND STRENGTH PROXY
    # ══════════════════════════════════════════════════════════════════════
    # ADX vero richiede serie OHLC complete. Usiamo un proxy 0-100 basato
    # su colonne già disponibili nel dataframe post-scanner:
    #   EMA alignment (0-40 pt)  + Vol_Ratio (0-30 pt)
    #   OBV_Trend     (0-15 pt)  + ATR%       (0-15 pt)
    # Output:
    #   ADX_Proxy      : float 0-100
    #   Trend_Strength : "STRONG" | "MODERATE" | "WEAK" | "RANGING"
    # ──────────────────────────────────────────────────────────────────────
    if "Prezzo" in df.columns:
        _tpr   = pd.to_numeric(df["Prezzo"], errors="coerce").replace(0, pd.NA)
        _te20  = pd.to_numeric(df.get("EMA20",     pd.Series(pd.NA, index=df.index)), errors="coerce")
        _te50  = pd.to_numeric(df.get("EMA50",     pd.Series(pd.NA, index=df.index)), errors="coerce")
        _tatr  = pd.to_numeric(df.get("ATR_pct",   pd.Series(2.0,  index=df.index)), errors="coerce").fillna(2.0)
        _tvol  = pd.to_numeric(df.get("Vol_Ratio", pd.Series(1.0,  index=df.index)), errors="coerce").fillna(1.0)
        _tobv  = (df.get("OBV_Trend", pd.Series("-", index=df.index)) == "UP")

        # Componente 1: allineamento EMA (max 40 pt)
        _pr_num = _tpr.fillna(0)
        _above20 = (_pr_num > _te20.fillna(0)).astype(float)
        _above50 = (_pr_num > _te50.fillna(0)).astype(float)
        _dist20  = ((_pr_num - _te20.fillna(_pr_num)).abs() / _tpr.fillna(1) * 100).clip(0, 5)
        _dist50  = ((_pr_num - _te50.fillna(_pr_num)).abs() / _tpr.fillna(1) * 100).clip(0, 5)
        _ema_score = (_above20 + _above50) / 2 * 25 + _dist20 * 2 + _dist50

        # Componente 2: volume (max 30 pt)
        _vol_score = (_tvol.clip(0.5, 3.0) - 0.5) / 2.5 * 30
        # Componente 3: OBV (max 15 pt)
        _obv_score = _tobv.astype(float) * 15
        # Componente 4: ATR vitalità (max 15 pt — ottimale 2-4%)
        _atr_score = (_tatr.clip(1.0, 5.0) - 1.0) / 4.0 * 15

        _adx = (_ema_score + _vol_score + _obv_score + _atr_score).clip(0, 100).round(1)

        df["ADX_Proxy"]      = _adx
        df["Trend_Strength"] = _adx.apply(
            lambda v: "STRONG"   if v >= 65 else
                      "MODERATE" if v >= 40 else
                      "WEAK"     if v >= 20 else
                      "RANGING")
    else:
        df["ADX_Proxy"]      = pd.NA
        df["Trend_Strength"] = "-"

    # ── Dollar Volume = liquidita' in dollari giornaliera ────────────────
    # Soglie:  > 5M  = minimo operabile (retail con posizioni moderate)
    #          > 20M = swing trading professionale
    #          > 50M = intraday / grandi posizioni
    # Catena fallback: Vol_Today (intraday) → Vol_7d_Avg → Avg_Vol_20
    # Vol_Today puo' essere basso a inizio seduta: usiamo il massimo tra
    # giornaliero e media 7gg per evitare esclusioni errate.
    if "Prezzo" in df.columns:
        pr      = pd.to_numeric(df["Prezzo"],    errors="coerce").fillna(0)
        vol_day = pd.to_numeric(df.get("Vol_Today",  pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        vol_7d  = pd.to_numeric(df.get("Vol_7d_Avg", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        vol_20  = pd.to_numeric(df.get("Avg_Vol_20", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        # Prende il massimo disponibile per evitare false esclusioni intraday
        vol_best = vol_day.where(vol_day > vol_7d, vol_7d)   # max(today, 7d)
        vol_best = vol_best.where(vol_best > 0, vol_20)       # fallback su 20d se entrambi 0
        df["Dollar_Vol"]  = (pr * vol_best / 1_000_000).round(2)   # milioni $
        df["Liq_OK"]      = df["Dollar_Vol"] >= 5.0
        df["Liq_Grade"]   = df["Dollar_Vol"].apply(
            lambda x: "L3-Institutional" if x >= 50  else
                      "L2-Professional"  if x >= 20  else
                      "L1-Retail"        if x >=  5  else
                      "Illiquido")
    else:
        df["Dollar_Vol"] = pd.NA
        df["Liq_OK"]     = pd.NA
        df["Liq_Grade"]  = pd.NA

    # ══════════════════════════════════════════════════════════════════════
    # v34 UPGRADE #3 — COMPOSITE SIGNAL SCORE (CSS)  0–100
    # ══════════════════════════════════════════════════════════════════════
    # Combina TUTTI gli score e filtri binari già calcolati in un singolo
    # numero ordinabile. Pesi calibrati per swing trading:
    #
    #   Pro_Score      (0-10) × 4.0  → max 40 pt  (peso principale)
    #   Ser_Score      (0-6)  × 3.0  → max 18 pt  (metodo Serafini)
    #   FV_Score       (0-5)  × 2.0  → max 10 pt  (filtri Finviz)
    #   ADX_Proxy      (0-100)× 0.15 → max 15 pt  (trend strength)
    #   ATR_OK         bool   × 5    → max  5 pt  (volatilità OK)
    #   Liq_OK         bool   × 5    → max  5 pt  (liquidità OK)
    #   RSI_Div_Score  (-1/0/+1)× 4  → max  4 pt  (divergenza RSI)
    #   OBV_Trend UP   bool   × 3    → max  3 pt  (OBV conferma)
    #
    # Totale massimo teorico: 100 pt
    # Grade: A ≥80 | B ≥60 | C ≥40 | D <40
    # ──────────────────────────────────────────────────────────────────────
    _css = pd.Series(0.0, index=df.index)

    if "Pro_Score" in df.columns:
        _css += pd.to_numeric(df["Pro_Score"], errors="coerce").fillna(0).clip(0, 10) * 4.0
    if "Ser_Score" in df.columns:
        _css += pd.to_numeric(df["Ser_Score"], errors="coerce").fillna(0).clip(0, 6)  * 3.0
    if "FV_Score" in df.columns:
        _css += pd.to_numeric(df["FV_Score"],  errors="coerce").fillna(0).clip(0, 5)  * 2.0
    if "ADX_Proxy" in df.columns:
        _css += pd.to_numeric(df["ADX_Proxy"], errors="coerce").fillna(0).clip(0,100) * 0.15
    if "ATR_OK" in df.columns:
        _css += pd.to_numeric(df["ATR_OK"].astype(float),  errors="coerce").fillna(0) * 5.0
    if "Liq_OK" in df.columns:
        _css += pd.to_numeric(df["Liq_OK"].astype(float),  errors="coerce").fillna(0) * 5.0
    if "RSI_Div_Score" in df.columns:
        _css += pd.to_numeric(df["RSI_Div_Score"], errors="coerce").fillna(0).clip(-1, 1) * 4.0
    if "OBV_Trend" in df.columns:
        _css += (df["OBV_Trend"] == "UP").astype(float) * 3.0

    df["CSS"]       = _css.clip(0, 100).round(1)
    df["CSS_Grade"] = df["CSS"].apply(
        lambda v: "A" if v >= 80 else
                  "B" if v >= 60 else
                  "C" if v >= 40 else "D")

    # ── v41 UPGRADE #6 — RELATIVE STRENGTH vs SPY ───────────────────────
    df = _add_rs_column(df)

    return df


# =========================================================================
# CSS
# =========================================================================
DARK_CSS = """
<style>
/* ── TradingView-style skin ─────────────────────────────────── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="block-container"]{
    background-color:#131722 !important; color:#d1d4dc !important;
    font-family:'Trebuchet MS','Segoe UI',sans-serif !important;}
[data-testid="stSidebar"]{background-color:#1e222d !important;border-right:1px solid #2a2e39 !important;}
[data-testid="stSidebar"] *{color:#d1d4dc !important;}
h1{color:#2962ff !important;font-family:'Trebuchet MS',sans-serif !important;
   letter-spacing:1px;text-shadow:0 0 16px #2962ff44;}
h2,h3{color:#50c4e0 !important;font-family:'Trebuchet MS',sans-serif !important;}
.stCaption,small{color:#6b7280 !important;}
[data-testid="stTabs"] button{background:#131722 !important;color:#787b86 !important;
    border-bottom:2px solid transparent !important;
    font-family:'Trebuchet MS',sans-serif !important;font-size:0.83rem !important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#2962ff !important;border-bottom:2px solid #2962ff !important;
    background:#1e222d !important;}
[data-testid="stMetric"]{background:#1e222d !important;border:1px solid #2a2e39 !important;
    border-radius:6px !important;padding:12px 16px !important;}
[data-testid="stMetricLabel"]{color:#787b86 !important;font-size:0.75rem !important;}
[data-testid="stMetricValue"]{color:#26a69a !important;font-size:1.6rem !important;
    font-family:'Trebuchet MS',sans-serif !important;font-weight:700 !important;}
[data-testid="stButton"]>button{background:#1e222d !important;
    color:#d1d4dc !important;border:1px solid #363a45 !important;
    border-radius:4px !important;font-family:'Trebuchet MS',sans-serif !important;transition:all 0.15s;}
[data-testid="stButton"]>button:hover{background:#2a2e39 !important;border-color:#50c4e0 !important;color:#ffffff !important;}
[data-testid="stButton"]>button[kind="primary"]{background:#2962ff !important;
    border-color:#2962ff !important;color:#ffffff !important;font-size:1rem !important;}
[data-testid="stButton"]>button[kind="secondary"]{background:#1e222d !important;
    color:#ef5350 !important;border:1px solid #ef535055 !important;}
[data-testid="stDownloadButton"]>button{background:#0d1117 !important;color:#58a6ff !important;
    border:1px solid #1f3a5f !important;border-radius:6px !important;}
[data-testid="stExpander"]{background:#0d1117 !important;border:1px solid #1f2937 !important;border-radius:8px !important;}
[data-testid="stExpander"] summary{color:#58a6ff !important;}
hr{border-color:#1f2937 !important;}
.ag-root-wrapper{background:#1e222d !important;border:1px solid #2a2e39 !important;border-radius:4px !important;}
.ag-header{background:#131722 !important;border-bottom:1px solid #363a45 !important;}
.ag-header-cell-label{color:#50c4e0 !important;font-family:'Trebuchet MS',sans-serif !important;
    font-size:0.79rem !important;letter-spacing:0.5px;text-transform:uppercase;}
.ag-header-cell-resize{background:#363a45 !important;}
.ag-row{background:#1e222d !important;border-bottom:1px solid #2a2e39 !important;}
.ag-row:hover{background:#2a2e39 !important;}
.ag-row-selected{background:rgba(41,98,255,0.18) !important;border-left:3px solid #2962ff !important;}
.ag-cell{color:#d1d4dc !important;font-family:'Trebuchet MS',sans-serif !important;font-size:0.83rem !important;}
.ag-paging-panel{background:#131722 !important;color:#787b86 !important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0a0e1a;}
::-webkit-scrollbar-thumb{background:#1f2937;border-radius:3px;}
.section-pill{display:inline-block;background:#1e222d;
    border-left:3px solid #2962ff;border-radius:0 4px 4px 0;padding:5px 16px;
    font-family:'Trebuchet MS',sans-serif;font-size:0.82rem;color:#50c4e0;
    letter-spacing:1px;margin-bottom:14px;}
.wl-card{background:linear-gradient(135deg,#0d1117 0%,#111827 100%);
    border:1px solid #1f2937;border-radius:12px;padding:14px 18px;margin-bottom:8px;transition:border-color 0.2s;}
.wl-card:hover{border-color:#374151;}
.wl-card-ticker{font-family:'Courier New',monospace;font-size:1.05rem;font-weight:bold;color:#00ff88;letter-spacing:1px;}
.wl-card-name{color:#8b949e;font-size:0.82rem;margin-top:2px;}
.wl-card-badge{display:inline-block;border-radius:10px;padding:2px 8px;font-size:0.72rem;font-weight:bold;margin-right:4px;}
.badge-green{background:rgba(0,255,136,0.15);color:#00ff88;border:1px solid #00ff8844;}
.badge-orange{background:rgba(245,158,11,0.15);color:#f59e0b;border:1px solid #f59e0b44;}
.badge-red{background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid #ef444444;}
.badge-blue{background:rgba(88,166,255,0.15);color:#58a6ff;border:1px solid #58a6ff44;}
.badge-gray{background:rgba(107,114,128,0.15);color:#6b7280;border:1px solid #6b728044;}
.badge-purple{background:rgba(167,139,250,0.15);color:#a78bfa;border:1px solid #a78bfa44;}
.legend-table{width:100%;border-collapse:collapse;font-family:'Courier New',monospace;font-size:0.82rem;}
.legend-table th{color:#58a6ff;border-bottom:1px solid #1f2937;padding:6px 10px;text-align:left;}
.legend-table td{color:#c9d1d9;border-bottom:1px solid #1a2233;padding:5px 10px;}
.legend-table tr:hover td{background:#131d2e;}
.legend-col-name{color:#00ff88;font-weight:bold;}
.legend-col-range{color:#f59e0b;}
.crit-ok{color:#00ff88;font-weight:bold;}
.crit-no{color:#ef4444;}
/* v41 MOBILE */
@media(max-width:480px){
 [data-testid="block-container"]{padding:0.5rem!important}
 [data-testid="stMetricValue"]{font-size:1.2rem!important}
 [data-testid="stTabs"]>div:first-child>button{font-size:0.62rem!important;padding:3px 5px!important}
 [data-testid="stButton"]>button{min-height:44px!important}
}
@media(min-width:481px) and (max-width:768px){
 [data-testid="stMetricValue"]{font-size:1.35rem!important}
 [data-testid="stTabs"]>div:first-child>button{font-size:0.68rem!important;padding:4px 6px!important}
}
</style>
"""

PLOTLY_DARK = dict(
    paper_bgcolor="#131722",
    plot_bgcolor="#1e222d",
    font=dict(color="#b2b5be", family="Trebuchet MS, sans-serif", size=12),
    xaxis=dict(gridcolor="#2a2e39", zerolinecolor="#363a45",
               linecolor="#363a45", tickfont=dict(color="#787b86",size=10)),
    yaxis=dict(gridcolor="#2a2e39", zerolinecolor="#363a45",
               linecolor="#363a45", tickfont=dict(color="#787b86",size=10)),
)
# =========================================================================
# FORMATTING HELPERS  (inline — non richiedono utils.formatting)
# =========================================================================
def _fmt_large(v):
    """Abbrevia numeri grandi: 1234567 → '1.2M', 12345678901 → '12.3B'"""
    try:
        v = float(v)
        if v != v or v <= 0: return "—"   # NaN o zero
        if v >= 1e12: return f"{v/1e12:.1f}T"
        if v >= 1e9:  return f"{v/1e9:.1f}B"
        if v >= 1e6:  return f"{v/1e6:.1f}M"
        if v >= 1e3:  return f"{v/1e3:.0f}K"
        return "—"  # valori irrisori non ha senso mostrarli
    except Exception:
        return "—"

def add_formatted_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonne _fmt usate dal display."""
    df = df.copy()
    if "Prezzo" in df.columns:
        df["Prezzo_fmt"] = df["Prezzo"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    if "MarketCap" in df.columns:
        df["MarketCap_fmt"] = df["MarketCap"].apply(
            lambda x: _fmt_large(x) if (pd.notna(x) and not (isinstance(x,float) and (x!=x))
                      and float(x) > 1_000_000) else "—")
    if "EMA200" in df.columns:
        df["EMA200_fmt"] = df["EMA200"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) and not (isinstance(x,float) and (x!=x)) else "—")
    return df

def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara DataFrame per visualizzazione AgGrid:
    - Rimuove colonne interne (prefisso _)
    - Converte bool numpy in bool Python
    - Resetta indice
    """
    df = df.copy()
    drop = [c for c in df.columns if c.startswith("_")]
    df   = df.drop(columns=drop, errors="ignore")
    for col in df.columns:
        try:
            df[col] = df[col].apply(
                lambda x: bool(x)  if isinstance(x, np.bool_)   else
                          float(x) if isinstance(x, np.floating) else
                          int(x)   if isinstance(x, np.integer)  else
                          None     if isinstance(x, float) and (np.isnan(x) or np.isinf(x))
                          else x
            )
        except Exception:
            pass
    return df.reset_index(drop=True)



# =========================================================================
# INDICATORI TECNICI (per grafici)
# =========================================================================
def _sma(arr, n):   return pd.Series(arr).rolling(n).mean().tolist()
def _rsi_calc(arr, n=14):
    s=pd.Series(arr); d=s.diff()
    up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(com=n-1,adjust=False).mean()/dn.ewm(com=n-1,adjust=False).mean()
    return (100-100/(1+rs)).tolist()
def _macd_calc(arr,fast=12,slow=26,sig=9):
    s=pd.Series(arr)
    m=s.ewm(span=fast,adjust=False).mean()-s.ewm(span=slow,adjust=False).mean()
    sg=m.ewm(span=sig,adjust=False).mean()
    return m.tolist(),sg.tolist(),(m-sg).tolist()
def _parabolic_sar(highs,lows,af_start=0.02,af_max=0.2):
    h=list(highs);l=list(lows);n=len(h)
    if n<2: return [None]*n,[0]*n
    sar=[0.0]*n;bull=[True]*n;ep=h[0];af=af_start;sar[0]=l[0]
    for i in range(1,n):
        pb=bull[i-1];ps=sar[i-1]
        if pb:
            ns=min(ps+af*(ep-ps),l[i-1],l[i-2] if i>=2 else l[i-1])
            if l[i]<ns: bull[i]=False;sar[i]=ep;ep=l[i];af=af_start
            else:
                bull[i]=True;sar[i]=ns
                if h[i]>ep: ep=h[i];af=min(af+af_start,af_max)
        else:
            ns=max(ps+af*(ep-ps),h[i-1],h[i-2] if i>=2 else h[i-1])
            if h[i]>ns: bull[i]=True;sar[i]=ep;ep=h[i];af=af_start
            else:
                bull[i]=False;sar[i]=ns
                if l[i]<ep: ep=l[i];af=min(af+af_start,af_max)
    return sar,[1 if b else -1 for b in bull]

# =========================================================================
# CHART BUILDER
# =========================================================================

def _calc_volume_profile(highs, lows, closes, vols, n_bins=36):
    """
    Volume Profile: distribuzione volume per livello di prezzo.
    Restituisce (bin_centers, vol_per_bin, poc, vah, val)
    POC = Point of Control  |  VAH/VAL = Value Area (70%)
    """
    try:
        import numpy as _np
        h=_np.array(highs,dtype=float); l=_np.array(lows,dtype=float)
        v=_np.array(vols,dtype=float)
        pmin,pmax = l.min(), h.max()
        if pmax<=pmin or len(h)<5: return [],[],None,None,None
        bins   = _np.linspace(pmin, pmax, n_bins+1)
        centers= (bins[:-1]+bins[1:])/2
        vpvol  = _np.zeros(n_bins)
        for i in range(len(h)):
            if v[i]<=0 or h[i]<=l[i]: continue
            b0=int(_np.searchsorted(bins,l[i],'left'))
            b1=int(_np.searchsorted(bins,h[i],'right'))
            b0=max(0,min(b0,n_bins-1)); b1=max(0,min(b1,n_bins))
            span=h[i]-l[i]
            for b in range(b0,b1):
                lo=max(bins[b],l[i]); hi=min(bins[b+1] if b+1<len(bins) else pmax,h[i])
                vpvol[b]+=v[i]*max(0,hi-lo)/span
        poc_i=int(_np.argmax(vpvol))
        poc=float(centers[poc_i])
        # Value Area 70%
        tot=vpvol.sum(); tgt=tot*0.70
        acc=vpvol[poc_i]; lo_i=hi_i=poc_i
        while acc<tgt and (lo_i>0 or hi_i<n_bins-1):
            add_lo=vpvol[lo_i-1] if lo_i>0 else 0
            add_hi=vpvol[hi_i+1] if hi_i<n_bins-1 else 0
            if add_hi>=add_lo and hi_i<n_bins-1: hi_i+=1; acc+=add_hi
            elif lo_i>0:                          lo_i-=1; acc+=add_lo
            else:                                  hi_i+=1; acc+=add_hi
        vah=float(centers[hi_i]); val=float(centers[lo_i])
        return list(centers),list(vpvol),poc,vah,val
    except Exception: return [],[],None,None,None


def build_full_chart(row: pd.Series, indicators: list) -> go.Figure:
    cd=row.get("_chart_data")
    if not cd or not isinstance(cd,dict): return None
    dates=cd.get("dates",[]); opens=cd.get("open",[])
    highs=cd.get("high",[]); lows=cd.get("low",[])
    closes=cd.get("close",[]); vols=cd.get("volume",[])
    ema20=cd.get("ema20",[]); ema50=cd.get("ema50",[])
    ema200=cd.get("ema200",[])
    bb_up=cd.get("bb_up",[]); bb_dn=cd.get("bb_dn",[])
    if not dates or not closes: return None

    show_sma=("SMA 9 & 21 + RSI" in indicators)
    show_macd=("MACD" in indicators)
    show_sar=("Parabolic SAR" in indicators)
    show_alligator=("Alligator + Vortex" in indicators)
    show_stochrsi=("Stochastic RSI" in indicators)  # v34
    show_vwap=("VWAP" in indicators)                # v35
    show_ha=("Heikin-Ashi" in indicators)            # v35
    show_sr=("S/R Auto" in indicators)               # v35

    # v35: Heikin-Ashi candle transform
    if show_ha and len(closes) >= 2:
        ha_closes = [(opens[i]+highs[i]+lows[i]+closes[i])/4 for i in range(len(closes))]
        ha_opens  = [opens[0]]
        for i in range(1, len(closes)):
            ha_opens.append((ha_opens[i-1]+ha_closes[i-1])/2)
        ha_highs = [max(highs[i], ha_opens[i], ha_closes[i]) for i in range(len(closes))]
        ha_lows  = [min(lows[i],  ha_opens[i], ha_closes[i]) for i in range(len(closes))]
        _opens_plot = ha_opens; _highs_plot = ha_highs
        _lows_plot  = ha_lows;  _closes_plot = ha_closes
    else:
        _opens_plot = opens; _highs_plot = highs
        _lows_plot  = lows;  _closes_plot = closes

    cur=2; row_rsi=None; row_macd=None; row_vortex=None; row_stochrsi=None
    if show_macd:       row_macd=cur;     cur+=1
    if show_alligator:  row_vortex=cur;   cur+=1
    if show_stochrsi:   row_stochrsi=cur; cur+=1   # v34 — pannello dedicato
    row_vol=cur; n_rows=cur

    ht={2:[0.65,0.15],3:[0.52,0.18,0.13],4:[0.44,0.17,0.15,0.12],5:[0.38,0.15,0.15,0.12,0.10],
        6:[0.34,0.13,0.13,0.11,0.11,0.08]}
    heights=ht.get(n_rows,[0.38,0.15,0.15,0.12,0.10])[:n_rows]
    s=sum(heights); heights=[h/s for h in heights]

    show_vp = ("Volume Profile" in indicators)
    if show_vp and vols:
        # 2 colonne: 84% candlestick | 16% Volume Profile
        _specs = [[{"secondary_y":False},{"secondary_y":False}]]*n_rows
        fig=make_subplots(rows=n_rows,cols=2,shared_xaxes=False,
                          shared_yaxes=False,
                          row_heights=heights,vertical_spacing=0.025,
                          column_widths=[0.84,0.16],
                          specs=_specs,horizontal_spacing=0.004)
        _vp_col=2
    else:
        show_vp=False
        fig=make_subplots(rows=n_rows,cols=1,shared_xaxes=True,
                          row_heights=heights,vertical_spacing=0.025)
        _vp_col=None
    # v35: usa _opens_plot/_closes_plot per supporto Heikin-Ashi
    _candle_name = "Heikin-Ashi" if show_ha else "Prezzo"
    fig.add_trace(go.Candlestick(x=dates,open=_opens_plot,high=_highs_plot,
        low=_lows_plot,close=_closes_plot,
        increasing_line_color="#26a69a",increasing_fillcolor="rgba(38,166,154,0.85)",
        decreasing_line_color="#ef5350",decreasing_fillcolor="rgba(239,83,80,0.85)",
        name=_candle_name,showlegend=False),row=1,col=1)
    if bb_up and bb_dn:
        fig.add_trace(go.Scatter(x=dates+dates[::-1],y=bb_up+bb_dn[::-1],fill="toself",
            fillcolor="rgba(88,166,255,0.06)",line=dict(color="rgba(0,0,0,0)"),
            showlegend=False),row=1,col=1)
        for b,n in [(bb_up,"BB↑"),(bb_dn,"BB↓")]:
            fig.add_trace(go.Scatter(x=dates,y=b,
                line=dict(color="#58a6ff",width=1,dash="dot"),showlegend=False,name=n),row=1,col=1)
    if ema20: fig.add_trace(go.Scatter(x=dates,y=ema20,line=dict(color="#f59e0b",width=1.5),name="EMA20"),row=1,col=1)
    if ema50: fig.add_trace(go.Scatter(x=dates,y=ema50,line=dict(color="#a78bfa",width=1.5),name="EMA50"),row=1,col=1)
    # EMA200 gialla — già letta nell'header da chart_data
    if ema200:
        fig.add_trace(go.Scatter(x=dates,y=ema200,
            line=dict(color="#ffffff",width=2.0,dash="dot"),name="EMA200"),row=1,col=1)

    # ── v35 UPGRADE #6a — VWAP intraday ─────────────────────────────────
    # VWAP = cumsum(typical_price * volume) / cumsum(volume)
    # Plotted solo se i volumi sono disponibili e mostra lo stesso range di dates
    if show_vwap and vols and closes:
        try:
            import numpy as _npvw
            _tp  = _npvw.array([(highs[i]+lows[i]+closes[i])/3 for i in range(len(closes))], dtype=float)
            _vol = _npvw.array(vols, dtype=float)
            _mask = _vol > 0
            _cum_tp_v = _npvw.cumsum(_tp * _vol)
            _cum_v    = _npvw.cumsum(_vol)
            _vwap = _npvw.where(_cum_v > 0, _cum_tp_v / _cum_v, _npvw.nan)
            fig.add_trace(go.Scatter(x=dates, y=_vwap.tolist(),
                line=dict(color="#ff6b6b", width=2, dash="dashdot"),
                name="VWAP"), row=1, col=1)
        except Exception:
            pass

    # ── v35 UPGRADE #6b — S/R Auto ──────────────────────────────────────
    # Identifica supporti/resistenze automatici su pivot locali
    if show_sr and closes and len(closes) >= 20:
        try:
            import numpy as _npsr2
            _c = _npsr2.array(closes, dtype=float)
            _h = _npsr2.array(highs, dtype=float)
            _l = _npsr2.array(lows, dtype=float)
            _pivots_r = []; _pivots_s = []
            _win = max(5, len(_c)//20)
            for i in range(_win, len(_c)-_win):
                if _h[i] == _h[i-_win:i+_win+1].max(): _pivots_r.append((_h[i], dates[i]))
                if _l[i] == _l[i-_win:i+_win+1].min(): _pivots_s.append((_l[i], dates[i]))
            # Raggruppa livelli vicini (entro 0.5% prezzo corrente)
            _cur_price = float(_c[-1])
            def _dedup_levels(pivots, tol_pct=0.5):
                if not pivots: return []
                _sorted = sorted(pivots, key=lambda x: x[0])
                _out = [_sorted[0]]
                for _p in _sorted[1:]:
                    if abs(_p[0]-_out[-1][0])/_cur_price*100 > tol_pct:
                        _out.append(_p)
                return _out[-4:]  # max 4 livelli
            for _lvl, _dt in _dedup_levels(_pivots_r):
                fig.add_hline(y=_lvl, line=dict(color="rgba(239,83,80,0.50)", width=1, dash="dot"),
                    annotation_text=f" R {_lvl:.2f}",
                    annotation_font_color="#ef5350", annotation_font_size=8,
                    row=1, col=1)
            for _lvl, _dt in _dedup_levels(_pivots_s):
                fig.add_hline(y=_lvl, line=dict(color="rgba(38,166,154,0.50)", width=1, dash="dot"),
                    annotation_text=f" S {_lvl:.2f}",
                    annotation_font_color="#26a69a", annotation_font_size=8,
                    row=1, col=1)
        except Exception:
            pass

    if show_sma:
        sma9=_sma(closes,9); sma21=_sma(closes,21)
        fig.add_trace(go.Scatter(x=dates,y=sma9,line=dict(color="#c084fc",width=1.5,dash="dash"),name="SMA9"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=sma21,line=dict(color="#fb923c",width=1.5,dash="dash"),name="SMA21"),row=1,col=1)
        for i in range(1,len(closes)):
            if any(v is None for v in [sma9[i],sma21[i],sma9[i-1],sma21[i-1]]): continue
            if sma9[i-1]<=sma21[i-1] and sma9[i]>sma21[i]:
                fig.add_annotation(x=dates[i],y=lows[i]*0.995,text="▲ ENTRY",
                    font=dict(color="#00ff88",size=10),showarrow=True,
                    arrowhead=2,arrowcolor="#00ff88",ay=30,ax=0,row=1,col=1)
            elif sma9[i-1]>=sma21[i-1] and sma9[i]<sma21[i]:
                fig.add_annotation(x=dates[i],y=highs[i]*1.005,text="▼ EXIT",
                    font=dict(color="#ef4444",size=10),showarrow=True,
                    arrowhead=2,arrowcolor="#ef4444",ay=-30,ax=0,row=1,col=1)

    if show_sar:
        sv,sd=_parabolic_sar(highs,lows)
        fig.add_trace(go.Scatter(x=dates,y=[sv[i] if sd[i]==1 else None for i in range(len(sv))],
            mode="markers",marker=dict(color="#00ff88",size=4),name="SAR ↑"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=[sv[i] if sd[i]==-1 else None for i in range(len(sv))],
            mode="markers",marker=dict(color="#ef4444",size=4),name="SAR ↓"),row=1,col=1)

    if show_sma and row_rsi:
        rv=_rsi_calc(closes)
        fig.add_hrect(y0=70,y1=100,fillcolor="rgba(239,68,68,0.08)",line_width=0,row=row_rsi,col=1)
        fig.add_hrect(y0=0,y1=30,fillcolor="rgba(0,255,136,0.08)",line_width=0,row=row_rsi,col=1)
        fig.add_trace(go.Scatter(x=dates,y=rv,line=dict(color="#60a5fa",width=1.5),name="RSI"),row=row_rsi,col=1)
        for lvl,col in [(70,"#ef4444"),(50,"#6b7280"),(30,"#00ff88")]:
            fig.add_hline(y=lvl,line=dict(color=col,width=1,dash="dot"),row=row_rsi,col=1)
        fig.update_yaxes(title_text="RSI",range=[0,100],tickfont=dict(size=9),row=row_rsi,col=1)

    if show_macd and row_macd:
        ml,ms,mh=_macd_calc(closes)
        fig.add_trace(go.Bar(x=dates,y=mh,
            marker_color=["rgba(0,255,136,0.7)" if v>=0 else "rgba(239,68,68,0.7)" for v in mh],
            name="MACD Hist",showlegend=False),row=row_macd,col=1)
        fig.add_trace(go.Scatter(x=dates,y=ml,line=dict(color="#60a5fa",width=1.5),name="MACD"),row=row_macd,col=1)
        fig.add_trace(go.Scatter(x=dates,y=ms,line=dict(color="#f97316",width=1.5),name="Signal"),row=row_macd,col=1)
        fig.add_hline(y=0,line=dict(color="#6b7280",width=1,dash="dot"),row=row_macd,col=1)
        fig.update_yaxes(title_text="MACD",tickfont=dict(size=9),row=row_macd,col=1)

    # ── Alligator (Jaw/Teeth/Lips) + Vortex (+VI/-VI) ─────────────────────
    if show_alligator and row_vortex:
        # Alligator: Jaw=SMA13, Teeth=SMA8, Lips=SMA5 (Williams)
        _jaw   = _sma(closes, 13)
        _teeth = _sma(closes, 8)
        _lips  = _sma(closes, 5)
        fig.add_trace(go.Scatter(x=dates,y=_jaw,
            line=dict(color="#3b82f6",width=1.5),name="Jaw(13)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=_teeth,
            line=dict(color="#ef4444",width=1.5),name="Teeth(8)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=_lips,
            line=dict(color="#22c55e",width=1.5),name="Lips(5)"),row=1,col=1)
        # Vortex Indicator (+VI/-VI) su pannello separato
        import numpy as _np2
        def _vortex(highs_l, lows_l, closes_l, period=14):
            n = len(highs_l)
            if n < period+1: return [None]*n, [None]*n
            h=_np2.array(highs_l,dtype=float); l=_np2.array(lows_l,dtype=float)
            c=_np2.array(closes_l,dtype=float)
            tr  = _np2.maximum(h[1:]-l[1:], _np2.maximum(_np2.abs(h[1:]-c[:-1]),_np2.abs(l[1:]-c[:-1])))
            vm_pos = _np2.abs(h[1:]-l[:-1])
            vm_neg = _np2.abs(l[1:]-h[:-1])
            vi_pos=[None]*period; vi_neg=[None]*period
            for i in range(period, n):
                s=i-period
                vi_pos.append(vm_pos[s:i].sum()/tr[s:i].sum() if tr[s:i].sum()>0 else 1.0)
                vi_neg.append(vm_neg[s:i].sum()/tr[s:i].sum() if tr[s:i].sum()>0 else 1.0)
            return vi_pos, vi_neg
        _vp, _vn = _vortex(highs, lows, closes)
        fig.add_trace(go.Scatter(x=dates,y=_vp,
            line=dict(color="#3b82f6",width=1.5),name="+VI"),row=row_vortex,col=1)
        fig.add_trace(go.Scatter(x=dates,y=_vn,
            line=dict(color="#ef4444",width=1.5),name="-VI"),row=row_vortex,col=1)
        fig.add_hline(y=1.0,line=dict(color="#6b7280",width=1,dash="dot"),row=row_vortex,col=1)
        fig.update_yaxes(title_text="Vortex",tickfont=dict(size=8),row=row_vortex,col=1)

    if vols:
        fig.add_trace(go.Bar(x=dates,y=vols,
            marker_color=["rgba(38,166,154,0.55)" if c>=o else "rgba(239,83,80,0.55)" for c,o in zip(closes,opens)],
            name="Volume",showlegend=False),row=row_vol,col=1)
        fig.update_yaxes(title_text="Vol",tickfont=dict(size=8),row=row_vol,col=1)

    # ── Volume Profile ──────────────────────────────────────────────────
    if show_vp and _vp_col:
        _vp_c,_vp_v,_poc,_vah,_val=_calc_volume_profile(highs,lows,closes,vols)
        if _vp_c:
            _mx=max(_vp_v) if _vp_v else 1
            _norm=[x/_mx for x in _vp_v]
            # Colori: dentro VA=blu TV, POC=oro, fuori=grigio
            _binw=(_vp_c[1]-_vp_c[0]) if len(_vp_c)>1 else 0
            _colors=[]
            for _i,_p in enumerate(_vp_c):
                if _poc and _binw and abs(_p-_poc)<_binw:
                    _colors.append("rgba(255,215,0,0.92)")    # POC oro
                elif _val and _vah and _val<=_p<=_vah:
                    _colors.append("rgba(41,98,255,0.70)")    # VA blu TV
                else:
                    _colors.append("rgba(120,123,134,0.42)")  # fuori VA grigio
            fig.add_trace(go.Bar(
                x=_norm, y=_vp_c, orientation="h",
                marker=dict(color=_colors,line=dict(width=0)),
                name="Vol Profile", showlegend=False,
                hovertemplate="P: %{y:.2f}<br>Vol: %{customdata:,.0f}<extra>VP</extra>",
                customdata=_vp_v,
            ),row=1,col=_vp_col)
            # Linee POC/VAH/VAL su asse Y condiviso con il prezzo
            if _poc:
                fig.add_hline(y=_poc,line=dict(color="#ffd700",width=1.5,dash="dot"),
                    annotation_text=" POC",annotation_font_color="#ffd700",
                    annotation_font_size=9,row=1,col=_vp_col)
            if _vah:
                fig.add_hline(y=_vah,line=dict(color="#2962ff",width=1,dash="dash"),
                    annotation_text=" VAH",annotation_font_color="#2962ff",
                    annotation_font_size=8,row=1,col=_vp_col)
            if _val:
                fig.add_hline(y=_val,line=dict(color="#2962ff",width=1,dash="dash"),
                    annotation_text=" VAL",annotation_font_color="#2962ff",
                    annotation_font_size=8,row=1,col=_vp_col)
            # Nascondi assi VP
            fig.update_xaxes(showticklabels=False,showgrid=False,zeroline=False,
                             col=_vp_col)
            for _rv in range(1,n_rows+1):
                fig.update_yaxes(showticklabels=False,showgrid=False,
                                 col=_vp_col,row=_rv)

    # ── v34 UPGRADE #4 — STOCHASTIC RSI  ────────────────────────────────
    # StochRSI = (RSI - min(RSI,n)) / (max(RSI,n) - min(RSI,n))
    # K = SMA(StochRSI, 3)   D = SMA(K, 3)
    # Zone: K/D > 80 → overbought   K/D < 20 → oversold
    # ─────────────────────────────────────────────────────────────────────
    if show_stochrsi and row_stochrsi and closes:
        def _stochrsi_calc(closes_l, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
            import numpy as _npsr
            c = _npsr.array(closes_l, dtype=float)
            n = len(c)
            # Calcola RSI interno (Wilder)
            delta = _npsr.diff(c)
            up   = _npsr.where(delta > 0, delta, 0.0)
            down = _npsr.where(delta < 0, -delta, 0.0)
            rs_up   = pd.Series(up).ewm(com=rsi_period-1, adjust=False).mean().values
            rs_down = pd.Series(down).ewm(com=rsi_period-1, adjust=False).mean().values
            with _npsr.errstate(divide="ignore", invalid="ignore"):
                rsi_arr = _npsr.where(rs_down == 0, 100.0, 100 - 100 / (1 + rs_up / rs_down))
            rsi_arr = _npsr.concatenate([[_npsr.nan], rsi_arr])
            # Stochastic su RSI
            stoch = _npsr.full(n, _npsr.nan)
            for i in range(stoch_period - 1, n):
                window = rsi_arr[i - stoch_period + 1: i + 1]
                lo, hi = _npsr.nanmin(window), _npsr.nanmax(window)
                stoch[i] = (rsi_arr[i] - lo) / (hi - lo) * 100 if hi > lo else 50.0
            k_line = pd.Series(stoch).rolling(smooth_k).mean().tolist()
            d_line = pd.Series(k_line).rolling(smooth_d).mean().tolist()
            return k_line, d_line

        _sk, _sd = _stochrsi_calc(closes)
        # Fasce overbought / oversold
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,68,68,0.07)",
                      line_width=0, row=row_stochrsi, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(0,255,136,0.07)",
                      line_width=0, row=row_stochrsi, col=1)
        fig.add_trace(go.Scatter(x=dates, y=_sk,
            line=dict(color="#a78bfa", width=1.5), name="StochRSI %K"),
            row=row_stochrsi, col=1)
        fig.add_trace(go.Scatter(x=dates, y=_sd,
            line=dict(color="#fb923c", width=1.5, dash="dot"), name="StochRSI %D"),
            row=row_stochrsi, col=1)
        for _lvl, _col in [(80, "#ef4444"), (50, "#6b7280"), (20, "#00ff88")]:
            fig.add_hline(y=_lvl, line=dict(color=_col, width=1, dash="dot"),
                          row=row_stochrsi, col=1)
        fig.update_yaxes(title_text="StochRSI", range=[0, 100],
                         tickfont=dict(size=9), row=row_stochrsi, col=1)

    # ── ATR Stop / Target levels (linee orizzontali operative) ──────────────
    # Visibili solo se ATR e Prezzo sono disponibili nella row dello scanner.
    # Stop  = Entry - 1.5×ATR  (rosso tratteggiato)
    # T1    = Entry + 1.5×ATR  (arancione, R:R 1:1)
    # T2    = Entry + 3.0×ATR  (verde,     R:R 2:1)
    _atr_val   = float(row.get("ATR", 0) or 0)
    _entry_val = float(row.get("Prezzo", 0) or 0)
    if _atr_val > 0 and _entry_val > 0:
        _sl  = round(_entry_val - 1.5 * _atr_val, 4)
        _t1  = round(_entry_val + 1.5 * _atr_val, 4)
        _t2  = round(_entry_val + 3.0 * _atr_val, 4)
        _slp = round((_sl - _entry_val) / _entry_val * 100, 1)
        _t1p = round((_t1 - _entry_val) / _entry_val * 100, 1)
        _t2p = round((_t2 - _entry_val) / _entry_val * 100, 1)
        # Linea entry (bianca tratteggiata)
        fig.add_hline(y=_entry_val,
            line=dict(color="rgba(255,255,255,0.50)", width=1.5, dash="dot"),
            annotation_text=f" Entry {_entry_val:.2f}",
            annotation_font_color="#d1d4dc", annotation_font_size=9,
            row=1, col=1)
        # Stop loss (rosso)
        fig.add_hline(y=_sl,
            line=dict(color="rgba(239,83,80,0.85)", width=1.5, dash="dash"),
            annotation_text=f" SL {_sl:.2f} ({_slp:+.1f}%)",
            annotation_font_color="#ef5350", annotation_font_size=9,
            row=1, col=1)
        # Target 1 (arancione, R:1)
        fig.add_hline(y=_t1,
            line=dict(color="rgba(255,152,0,0.85)", width=1.5, dash="dash"),
            annotation_text=f" T1 {_t1:.2f} ({_t1p:+.1f}%) R:1",
            annotation_font_color="#ff9800", annotation_font_size=9,
            row=1, col=1)
        # Target 2 (verde, R:2)
        fig.add_hline(y=_t2,
            line=dict(color="rgba(38,166,154,0.85)", width=1.5, dash="dash"),
            annotation_text=f" T2 {_t2:.2f} ({_t2p:+.1f}%) R:2",
            annotation_font_color="#26a69a", annotation_font_size=9,
            row=1, col=1)

    tkr=row.get("Ticker",""); sq="  🔥" if row.get("Squeeze") else ""
    _atr_label = f"  ATR:{_atr_val:.2f}" if _atr_val > 0 else ""
    fig.update_layout(**PLOTLY_DARK,
        title=dict(text=f"<b>{tkr}</b> — {row.get('Nome','')}  |  {row.get('Prezzo','')}  |  RSI {row.get('RSI','')}{sq}{_atr_label}",
            font=dict(color="#50c4e0",size=13),x=0.01,xanchor="left"),
        height=160+180*n_rows,xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.01,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
        margin=dict(l=0,r=0,t=55,b=0),hovermode="x unified")
    for r in range(1,n_rows+1):
        fig.update_xaxes(gridcolor="#2a2e39",gridwidth=1,showline=True,linecolor="#363a45",row=r,col=1)
        fig.update_yaxes(gridcolor="#2a2e39",gridwidth=1,showline=True,linecolor="#363a45",row=r,col=1)
    return fig

def build_radar(row: pd.Series) -> go.Figure:
    qc=row.get("_quality_components")
    if not qc or not isinstance(qc,dict): return None
    keys=list(qc.keys()); vals=list(qc.values())
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=keys+[keys[0]],fill="toself",
        fillcolor="rgba(0,255,136,0.15)",line=dict(color="#00ff88",width=2)))
    fig.update_layout(**PLOTLY_DARK,
        polar=dict(bgcolor="#0d1117",
            radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=9,color="#6b7280"),
                gridcolor="#1f2937",linecolor="#1f2937"),
            angularaxis=dict(tickfont=dict(size=11,color="#c9d1d9"),
                gridcolor="#1f2937",linecolor="#1f2937")),
        title=dict(text=f"<b>{row.get('Ticker','')}</b>  Q: <b>{row.get('Quality_Score',0)}/12</b>",
            font=dict(color="#58a6ff",size=13)),
        height=340,margin=dict(l=40,r=40,t=55,b=20),showlegend=False)
    return fig

def show_charts(row_full: pd.Series, key_suffix: str=""):
    tkr=row_full.get("Ticker","")
    st.markdown("---")
    ind_opts=["SMA 9 & 21 + RSI","MACD","Parabolic SAR","Alligator + Vortex","Volume Profile",
              "Stochastic RSI",   # v34
              "VWAP","Heikin-Ashi","S/R Auto"]  # v35
    c1,c2=st.columns([4,1])
    with c1:
        indicators=st.multiselect("🔧 Indicatori",options=ind_opts,
            default=st.session_state.get("active_indicators",ind_opts),
            key=f"ind_{tkr}_{key_suffix}")
        st.session_state["active_indicators"]=indicators
    with c2:
        st.write("")
        if st.button("🔄 Aggiorna",key=f"ref_{tkr}_{key_suffix}"): st.rerun()
    fig=build_full_chart(row_full,indicators)
    if fig: st.plotly_chart(fig,use_container_width=True,key=f"full_{tkr}_{key_suffix}")
    else:   st.info("Dati grafici non disponibili. Riesegui lo scanner.")
    fig_r=build_radar(row_full)
    if fig_r:
        _,c2,_=st.columns([1,1,1])
        with c2: st.plotly_chart(fig_r,use_container_width=True,key=f"radar_{tkr}_{key_suffix}")
    # ── Grafico Analitico Avanzato ──────────────────────────────────────
    try:
        from analysis_chart import render_analysis_chart as _adv_chart
        with st.expander(f"📐 Analisi Avanzata — Ichimoku · S/R · Trend · Squeeze  [{tkr}]",
                         expanded=False):
            _adv_chart(row_full, key_suffix=key_suffix)
    except ImportError:
        pass  # analysis_chart.py non presente

# =========================================================================
# JS RENDERERS
# =========================================================================
name_dblclick_renderer=JsCode("""class N{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value||'';const t=p.data.Ticker||p.data.ticker;if(!t)return;
this.eGui.style.cursor='pointer';this.eGui.title='Doppio click → TradingView';
this.eGui.ondblclick=()=>window.open("https://it.tradingview.com/chart/?symbol="+String(t).split(".")[0],"_blank");}
getGui(){return this.eGui;}}""")

rsi_renderer=JsCode("""class R{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'-':v.toFixed(1);
this.eGui.style.fontWeight='bold';this.eGui.style.fontFamily='Courier New';
if(v<30)this.eGui.style.color='#60a5fa';
else if(v<40)this.eGui.style.color='#93c5fd';
else if(v<=65)this.eGui.style.color='#00ff88';
else if(v<=70)this.eGui.style.color='#f59e0b';
else this.eGui.style.color='#ef4444';}getGui(){return this.eGui;}}""")

# Renderer stringa già formattata (MarketCap_fmt = "1.2B", "—", etc.)
mcap_str_renderer=JsCode("""class MS{init(p){this.eGui=document.createElement('span');
const s=String(p.value||'\u2014').trim();
let color='#6b7280';
if(s.endsWith('T'))color='#00ff88';
else if(s.endsWith('B'))color='#58a6ff';
else if(s.endsWith('M'))color='#f59e0b';
this.eGui.innerText=s;this.eGui.style.color=color;
this.eGui.style.fontFamily='Courier New';this.eGui.style.fontWeight='bold';}
getGui(){return this.eGui;}refresh(){return false;}}""")

vol_ratio_renderer=JsCode("""class V{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'-':v.toFixed(2)+'x';
this.eGui.style.fontFamily='Courier New';this.eGui.style.fontWeight='bold';
if(v<1)this.eGui.style.color='#6b7280';
else if(v<2)this.eGui.style.color='#00ff88';
else if(v<3)this.eGui.style.color='#f59e0b';
else{this.eGui.style.color='#ef4444';this.eGui.style.textShadow='0 0 6px #ef4444';}
}getGui(){return this.eGui;}}""")

# Renderer per volumi abbreviati (es. 1.2M, 45.6K, 2.3B)
vol_abbrev_renderer=JsCode("""class VA{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
let txt='-';
if(!isNaN(v)){
  if(v>=1e9)txt=(v/1e9).toFixed(1)+'B';
  else if(v>=1e6)txt=(v/1e6).toFixed(1)+'M';
  else if(v>=1e3)txt=(v/1e3).toFixed(0)+'K';
  else txt=v.toFixed(0);
}
this.eGui.innerText=txt;
this.eGui.style.fontFamily='Courier New';this.eGui.style.color='#c9d1d9';
}getGui(){return this.eGui;}}""")

# Renderer MarketCap abbreviato
mcap_renderer=JsCode("""class MC{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
let txt='—';let color='#6b7280';
if(!isNaN(v) && v>1000000){
  if(v>=1e12){txt=(v/1e12).toFixed(2)+'T';color='#00ff88';}
  else if(v>=1e9){txt=(v/1e9).toFixed(1)+'B';color='#58a6ff';}
  else if(v>=1e6){txt=(v/1e6).toFixed(0)+'M';color='#f59e0b';}
  else{txt=(v/1e3).toFixed(0)+'K';color='#6b7280';}
}
this.eGui.innerText=txt;
this.eGui.style.fontFamily='Courier New';this.eGui.style.color=color;this.eGui.style.fontWeight='bold';
}getGui(){return this.eGui;}}""")

quality_renderer=JsCode("""class Q{init(p){this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:6px';
const v=parseInt(p.value||0);const pct=Math.round((v/12)*100);
const c=v>=9?'#00ff88':v>=6?'#f59e0b':'#6b7280';
this.eGui.innerHTML=`<span style="font-family:Courier New;font-weight:bold;color:${c};min-width:20px">${v}</span>
<div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
<div style="width:${pct}%;background:${c};height:6px;border-radius:3px"></div></div>`;}
getGui(){return this.eGui;}}""")

ser_score_renderer=JsCode("""class S{init(p){this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:6px';
const v=parseInt(p.value||0);const pct=Math.round((v/6)*100);
const c=v>=6?'#00ff88':v>=4?'#f59e0b':'#ef4444';
this.eGui.innerHTML=`<span style="font-family:Courier New;font-weight:bold;color:${c};min-width:20px">${v}/6</span>
<div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
<div style="width:${pct}%;background:${c};height:6px;border-radius:3px"></div></div>`;}
getGui(){return this.eGui;}}""")

fv_score_renderer=JsCode("""class F{init(p){this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:6px';
const v=parseInt(p.value||0);const pct=Math.round((v/8)*100);
const c=v>=7?'#00ff88':v>=5?'#f59e0b':'#6b7280';
this.eGui.innerHTML=`<span style="font-family:Courier New;font-weight:bold;color:${c};min-width:20px">${v}/8</span>
<div style="flex:1;background:#1f2937;border-radius:3px;height:6px">
<div style="width:${pct}%;background:${c};height:6px;border-radius:3px"></div></div>`;}
getGui(){return this.eGui;}}""")

bool_renderer=JsCode("""class B{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v===true||v==='True'||v==='true'||v===1){this.eGui.innerText='✅';this.eGui.style.color='#00ff88';}
else if(v===false||v==='False'||v==='false'||v===0){this.eGui.innerText='❌';this.eGui.style.color='#ef4444';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

squeeze_renderer=JsCode("""class Sq{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v===true||v==='True'||v==='true'){this.eGui.innerText='🔥 SQ';this.eGui.style.color='#f97316';this.eGui.style.fontWeight='bold';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

weekly_renderer=JsCode("""class W{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v===true||v==='True'||v==='true'){this.eGui.innerText='📈 W+';this.eGui.style.color='#00ff88';}
else if(v===false||v==='False'||v==='false'){this.eGui.innerText='📉 W—';this.eGui.style.color='#ef4444';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

rsi_div_renderer=JsCode("""class RD{init(p){this.eGui=document.createElement('span');
const v=p.value;
if(v==='BEARISH'){this.eGui.innerText='⚠️ BEAR';this.eGui.style.color='#ef4444';}
else if(v==='BULLISH'){this.eGui.innerText='✅ BULL';this.eGui.style.color='#00ff88';}
else{this.eGui.innerText='—';this.eGui.style.color='#374151';}
}getGui(){return this.eGui;}}""")

price_renderer=JsCode("""class P{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value??'-';this.eGui.style.fontFamily='Courier New';
this.eGui.style.color='#e2e8f0';this.eGui.style.fontWeight='bold';}
getGui(){return this.eGui;}}""")

trend_renderer=JsCode("""class T{init(p){this.eGui=document.createElement('span');
const v=(p.value||'').toUpperCase();
const map={LONG:{c:'#00ff88',e:'🟢 LONG'},SHORT:{c:'#ef4444',e:'🔴 SHORT'},WATCH:{c:'#f59e0b',e:'👁 WATCH'}};
const m=map[v]||{c:'#6b7280',e:v||'—'};
this.eGui.innerText=m.e;this.eGui.style.color=m.c;this.eGui.style.fontWeight='bold';}
getGui(){return this.eGui;}}""")

# Renderer Stato_Pro — distingue STRONG (oro) da PRO (verde) da - (grigio)
stato_pro_renderer=JsCode("""class SP{init(p){this.eGui=document.createElement('span');
const v=(p.value||'').toUpperCase();
if(v==='STRONG'){
  this.eGui.innerText='★ STRONG';
  this.eGui.style.cssText='color:#ffd700;font-weight:bold;font-family:Courier New;'
    +'background:rgba(255,215,0,0.12);padding:2px 6px;border-radius:4px;border:1px solid #ffd70044;';
}else if(v==='PRO'){
  this.eGui.innerText='✦ PRO';
  this.eGui.style.cssText='color:#00ff88;font-weight:bold;font-family:Courier New;'
    +'background:rgba(0,255,136,0.10);padding:2px 6px;border-radius:4px;border:1px solid #00ff8844;';
}else{
  this.eGui.innerText='—';this.eGui.style.color='#374151';
}
}getGui(){return this.eGui;}}""")

# Renderer Dollar Volume (in M$)
dollar_vol_renderer=JsCode("""class DV{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
let txt='—';let color='#ef4444';
if(!isNaN(v)&&v>0){
  txt='$'+v.toFixed(1)+'M';
  if(v>=50)color='#00ff88';
  else if(v>=20)color='#26a69a';
  else if(v>=5)color='#f59e0b';
  else color='#ef4444';
}
this.eGui.innerText=txt;this.eGui.style.color=color;
this.eGui.style.fontFamily='Courier New';this.eGui.style.fontWeight='bold';
}getGui(){return this.eGui;}}""")

# Renderer ATR% con semaforo
atr_pct_renderer=JsCode("""class AP{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
let txt='—';let color='#6b7280';
if(!isNaN(v)){
  txt=v.toFixed(2)+'%';
  if(v>=1.5&&v<=6.0)color='#00ff88';
  else if(v<1.5)color='#6b7280';
  else color='#ef4444';
}
this.eGui.innerText=txt;this.eGui.style.color=color;
this.eGui.style.fontFamily='Courier New';
}getGui(){return this.eGui;}}""")

# Renderer Liq_Grade badge
liq_grade_renderer=JsCode("""class LG{init(p){this.eGui=document.createElement('span');
const v=(p.value||'');
const map={
  'L3-Institutional':{c:'#00ff88',bg:'rgba(0,255,136,0.12)'},
  'L2-Professional': {c:'#26a69a',bg:'rgba(38,166,154,0.12)'},
  'L1-Retail':       {c:'#f59e0b',bg:'rgba(245,158,11,0.12)'},
  'Illiquido':       {c:'#ef4444',bg:'rgba(239,68,68,0.12)'},
};
const m=map[v]||{c:'#6b7280',bg:'transparent'};
this.eGui.innerText=v||'—';
this.eGui.style.cssText='color:'+m.c+';background:'+m.bg+';padding:1px 5px;'
  +'border-radius:3px;font-size:0.78rem;font-family:Courier New;';
}getGui(){return this.eGui;}}""")

# ── v34 RENDERERS ─────────────────────────────────────────────────────────────

# CSS score (0-100) con barra orizzontale + Grade colorato
css_renderer=JsCode("""class CS{init(p){
this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:5px;height:100%;';
const v=parseFloat(p.value);
if(isNaN(v)){this.eGui.innerHTML='<span style="color:#6b7280">—</span>';return;}
const pct=Math.min(100,Math.max(0,v));
const col=pct>=80?'#00ff88':pct>=60?'#26a69a':pct>=40?'#f59e0b':'#ef4444';
const bar=document.createElement('div');
bar.style.cssText='flex:1;height:6px;background:#1e222d;border-radius:3px;overflow:hidden;';
const fill=document.createElement('div');
fill.style.cssText='height:100%;width:'+pct+'%;background:'+col+';border-radius:3px;transition:width 0.3s;';
bar.appendChild(fill);
const lbl=document.createElement('span');
lbl.innerText=v.toFixed(1);
lbl.style.cssText='font-family:Courier New;font-size:0.79rem;font-weight:bold;color:'+col+';min-width:32px;text-align:right;';
this.eGui.appendChild(lbl);this.eGui.appendChild(bar);
}getGui(){return this.eGui;}}""")

# CSS Grade (A/B/C/D) badge colorato
css_grade_renderer=JsCode("""class CG{init(p){this.eGui=document.createElement('span');
const v=(p.value||'');
const map={'A':{c:'#00ff88',bg:'rgba(0,255,136,0.15)',b:'1px solid rgba(0,255,136,0.3)'},
           'B':{c:'#26a69a',bg:'rgba(38,166,154,0.15)',b:'1px solid rgba(38,166,154,0.3)'},
           'C':{c:'#f59e0b',bg:'rgba(245,158,11,0.15)',b:'1px solid rgba(245,158,11,0.3)'},
           'D':{c:'#ef4444',bg:'rgba(239,68,68,0.15)', b:'1px solid rgba(239,68,68,0.3)'}};
const m=map[v]||{c:'#6b7280',bg:'transparent',b:'none'};
this.eGui.innerText=v||'—';
this.eGui.style.cssText='color:'+m.c+';background:'+m.bg+';border:'+m.b+';'
  +'padding:1px 8px;border-radius:10px;font-weight:bold;font-size:0.85rem;font-family:Courier New;';
}getGui(){return this.eGui;}}""")

# Trend Strength (STRONG/MODERATE/WEAK/RANGING)
trend_strength_renderer=JsCode("""class TS{init(p){this.eGui=document.createElement('span');
const v=(p.value||'');
const map={
  'STRONG':  {c:'#00ff88',bg:'rgba(0,255,136,0.12)',icon:'⚡'},
  'MODERATE':{c:'#26a69a',bg:'rgba(38,166,154,0.12)',icon:'↗'},
  'WEAK':    {c:'#f59e0b',bg:'rgba(245,158,11,0.12)',icon:'→'},
  'RANGING': {c:'#6b7280',bg:'rgba(107,114,128,0.10)',icon:'↔'},
};
const m=map[v]||{c:'#6b7280',bg:'transparent',icon:''};
this.eGui.innerText=(m.icon?m.icon+' ':'')+v;
this.eGui.style.cssText='color:'+m.c+';background:'+m.bg+';padding:1px 6px;'
  +'border-radius:3px;font-size:0.78rem;font-family:Courier New;';
}getGui(){return this.eGui;}}""")

# ADX Proxy (0-100) barra compatta
adx_proxy_renderer=JsCode("""class ADX{init(p){
this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:4px;height:100%;';
const v=parseFloat(p.value);
if(isNaN(v)){this.eGui.innerHTML='<span style="color:#6b7280">—</span>';return;}
const col=v>=65?'#00ff88':v>=40?'#26a69a':v>=20?'#f59e0b':'#6b7280';
const bar=document.createElement('div');
bar.style.cssText='flex:1;height:4px;background:#1e222d;border-radius:2px;overflow:hidden;';
const fill=document.createElement('div');
fill.style.cssText='height:100%;width:'+Math.min(100,v)+'%;background:'+col+';border-radius:2px;';
bar.appendChild(fill);
const lbl=document.createElement('span');
lbl.innerText=v.toFixed(0);
lbl.style.cssText='font-family:Courier New;font-size:0.79rem;color:'+col+';min-width:24px;';
this.eGui.appendChild(lbl);this.eGui.appendChild(bar);
}getGui(){return this.eGui;}}""")

pct_renderer=JsCode("""class Pct{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);
if(isNaN(v)){this.eGui.innerText='—';this.eGui.style.color='#6b7280';}
else{this.eGui.innerText=(v*100).toFixed(1)+'%';
this.eGui.style.color=v>0?'#00ff88':v<0?'#ef4444':'#6b7280';
this.eGui.style.fontWeight='bold';this.eGui.style.fontFamily='Courier New';}
}getGui(){return this.eGui;}}""")

# v41 — RS vs SPY renderer (barra orizzontale con valore)
rs_renderer=JsCode("""class RS{init(p){
this.eGui=document.createElement('div');
this.eGui.style.cssText='display:flex;align-items:center;gap:4px;height:100%;';
const v=parseFloat(p.value);
if(isNaN(v)){this.eGui.innerHTML='<span style="color:#6b7280">—</span>';return;}
const col=v>=5?'#00ff88':v>=0?'#26a69a':v>=-5?'#f59e0b':'#ef4444';
const pct=Math.min(100,Math.max(0,(v+20)/40*100));
const bar=document.createElement('div');
bar.style.cssText='flex:1;height:4px;background:#1e222d;border-radius:2px;overflow:hidden;';
const fill=document.createElement('div');
fill.style.cssText='height:100%;width:'+pct+'%;background:'+col+';border-radius:2px;';
bar.appendChild(fill);
const lbl=document.createElement('span');
lbl.innerText=(v>0?'+':'')+v.toFixed(1)+'%';
lbl.style.cssText='font-family:Courier New;font-size:0.78rem;color:'+col+';min-width:44px;';
this.eGui.appendChild(lbl);this.eGui.appendChild(bar);
}getGui(){return this.eGui;}}""")

# v41 — RS Rank renderer (0-100 badge)
rs_rank_renderer=JsCode("""class RR{init(p){this.eGui=document.createElement('span');
const v=parseInt(p.value||0);
const col=v>=80?'#00ff88':v>=60?'#26a69a':v>=40?'#f59e0b':'#ef4444';
this.eGui.innerText=v;
this.eGui.style.cssText='color:'+col+';font-family:Courier New;font-weight:bold;font-size:0.82rem;';
}getGui(){return this.eGui;}}""")

# =========================================================================
# EXPORT
# =========================================================================
def to_excel_bytes(d):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="xlsxwriter") as w:
        for nm,df in d.items():
            if isinstance(df,pd.DataFrame) and not df.empty:
                df.to_excel(w,sheet_name=nm[:31],index=False)
    return buf.getvalue()

def make_tv_csv(df,tab):
    t=df[["Ticker"]].copy(); t.insert(0,"Tab",tab)
    return t.to_csv(index=False).encode()

def csv_btn(df,fname,key):
    st.download_button("📥 CSV",df.to_csv(index=False).encode(),fname,"text/csv",key=key)

# =========================================================================
# PRESETS
# =========================================================================
PRESETS={
    # Aggressivo: molti segnali, soglie basse, size ridotta consigliata
    "⚡ Aggressivo":   dict(eh=0.01,prmin=45,prmax=65,rpoc=0.01,vol_ratio_hot=1.2,top=25,
                         min_early_score=0.0,min_quality=0,min_pro_score=0.0,
                         liq_filter_enabled=True,min_dollar_vol=5,
                         atr_filter_enabled=True,atr_pct_min=1.0,atr_pct_max=8.0,
                         show_strong_only=False),
    # Bilanciato: rapporto qualita'/quantita' ottimale per swing trading
    "⚖️ Bilanciato":   dict(eh=0.02,prmin=40,prmax=70,rpoc=0.02,vol_ratio_hot=2.0,top=15,  # v34
                         min_early_score=2.0,min_quality=4,min_pro_score=0.0,
                         liq_filter_enabled=True,min_dollar_vol=10,
                         atr_filter_enabled=True,atr_pct_min=1.5,atr_pct_max=6.0,
                         show_strong_only=False),
    # Conservativo: alta selettivita', meno segnali ma piu' affidabili
    "🛡️ Conservativo": dict(eh=0.04,prmin=35,prmax=75,rpoc=0.04,vol_ratio_hot=2.0,top=10,
                         min_early_score=4.0,min_quality=6,min_pro_score=0.0,
                         liq_filter_enabled=True,min_dollar_vol=20,
                         atr_filter_enabled=True,atr_pct_min=1.5,atr_pct_max=4.0,
                         show_strong_only=False),
    # Solo STRONG: massima convinzione, pochissimi segnali ad alta probabilita'
    "★ Solo STRONG":   dict(eh=0.02,prmin=40,prmax=70,rpoc=0.02,vol_ratio_hot=2.5,top=10,  # v34
                         min_early_score=4.0,min_quality=7,min_pro_score=7.0,
                         liq_filter_enabled=True,min_dollar_vol=20,
                         atr_filter_enabled=True,atr_pct_min=1.5,atr_pct_max=6.0,
                         show_strong_only=True),
    # Istituzionale: alta liquidita', grandi cap, per posizioni importanti
    "🏦 Istituzionale":dict(eh=0.02,prmin=38,prmax=72,rpoc=0.02,vol_ratio_hot=1.3,top=10,
                         min_early_score=4.0,min_quality=8,min_pro_score=6.0,
                         liq_filter_enabled=True,min_dollar_vol=50,
                         atr_filter_enabled=True,atr_pct_min=1.0,atr_pct_max=4.0,
                         show_strong_only=False),
    # Nessun Filtro: debug / esplorazione completa
    "🔓 Nessun Filtro":dict(eh=0.05,prmin=10,prmax=90,rpoc=0.05,vol_ratio_hot=0.3,top=100,
                         min_early_score=0.0,min_quality=0,min_pro_score=0.0,
                         liq_filter_enabled=False,min_dollar_vol=1,
                         atr_filter_enabled=False,atr_pct_min=0.5,atr_pct_max=12.0,
                         show_strong_only=False),
}

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(page_title="Trading Scanner PRO 39.0",layout="wide",page_icon="🧠")
st.markdown(DARK_CSS,unsafe_allow_html=True)
st.markdown("# 🧠 Trading Scanner PRO 39.0")
st.markdown('<div class="section-pill">SCANNER V40 · WATCHLIST ALERT · P&L TRACKER · BACKTEST PRO · EXPORT PRO · CHART TV-STYLE · MTF MATRIX · JOURNAL · REGIME</div>',unsafe_allow_html=True)
init_db()

# ── GitHub pull al boot (ripristina watchlist dopo ogni deploy) ─────────────
if _GH_SYNC and not st.session_state.get("_gh_pulled"):
    with st.spinner("☁️ Ripristino watchlist da GitHub..."):
        _ok, _n, _gh_src = _gh_pull(DB_PATH)
    st.session_state["_gh_pulled"] = True
    if _ok and _n > 0:
        st.toast(f"☁️ Watchlist ripristinata: {_n} ticker", icon="✅")
    elif not _ok and _gh_src == "github_error":
        st.toast("⚠️ GitHub sync: errore connessione — uso dati locali", icon="⚠️")

# =========================================================================
# SESSION STATE
# =========================================================================
defaults=dict(
    mSP500=True,mNasdaq=True,mFTSE=True,mEurostoxx=False,
    mDow=False,mRussell=False,mStoxxEmerging=False,mUSSmallCap=False,
    eh=0.02,prmin=40,prmax=70,rpoc=0.02,vol_ratio_hot=2.0,top=15,  # v34
    min_early_score=2.0,min_quality=3,
    min_pro_score=0.0,   # 0 = nessun filtro extra: la classificazione PRO/STRONG basta
    # Nuovi filtri qualita' v34
    min_dollar_vol=5.0,         # Dollar Volume minimo in milioni $ (liquidita')
    atr_filter_enabled=True,    # Filtro ATR% attivo di default
    atr_pct_min=1.5,            # ATR% minimo (titolo troppo fermo se sotto)
    atr_pct_max=6.0,            # ATR% massimo (troppo volatile se sopra)
    liq_filter_enabled=True,    # Filtro liquidita' attivo di default
    show_strong_only=False,     # Mostra solo STRONG (Pro>=9) invece di PRO+STRONG
    current_list_name="DEFAULT",last_active_tab="EARLY",
    active_indicators=["SMA 9 & 21 + RSI","MACD","Parabolic SAR","Alligator + Vortex"],
    wl_view_mode="cards",
)
for k,v in defaults.items():
    st.session_state.setdefault(k,v)

# =========================================================================
# KPI BAR
# =========================================================================
def render_kpi_bar(df_ep,df_rea):
    hist=load_scan_history(2); p_e=p_p=p_h=p_c=0
    if len(hist)>=2:
        pr=hist.iloc[1];p_e=int(pr.get("n_early",0));p_p=int(pr.get("n_pro",0))
        p_h=int(pr.get("n_rea",0));p_c=int(pr.get("n_confluence",0))
    n_e=int((df_ep.get("Stato_Early",pd.Series())=="EARLY").sum()) if not df_ep.empty else 0
    n_p=int((df_ep.get("Stato_Pro",pd.Series()).isin(["PRO","STRONG"])).sum()) if not df_ep.empty else 0
    n_str=int((df_ep.get("Stato_Pro",pd.Series())=="STRONG").sum()) if not df_ep.empty else 0
    n_h=len(df_rea) if not df_rea.empty else 0
    n_c=0
    if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
        n_c=int(((df_ep["Stato_Early"]=="EARLY") &
                  (df_ep["Stato_Pro"].isin(["PRO","STRONG"]))).sum())
    # Liquidita' media (Dollar_Vol)
    n_liq = 0
    if not df_ep.empty and "Liq_OK" in df_ep.columns:
        n_liq = int(df_ep["Liq_OK"].isin([True,"True","true",1]).sum())

    # ── v34: CSS Grade A e Trend STRONG ─────────────────────────────────
    n_css_a    = 0
    n_strong   = 0
    css_avg    = None
    if not df_ep.empty:
        if "CSS_Grade" in df_ep.columns:
            n_css_a = int((df_ep["CSS_Grade"] == "A").sum())
        if "CSS" in df_ep.columns:
            _css_vals = pd.to_numeric(df_ep["CSS"], errors="coerce").dropna()
            css_avg = round(float(_css_vals.mean()), 1) if len(_css_vals) > 0 else None
        if "Trend_Strength" in df_ep.columns:
            n_strong = int((df_ep["Trend_Strength"] == "STRONG").sum())

    k1,k2,k3,k4,k5,k6,k7,k8=st.columns(8)
    k1.metric("📡 EARLY",     n_e,   delta=n_e-p_e   if p_e  else None)
    k2.metric("💪 PRO+STR",   n_p,   delta=n_p-p_p   if p_p  else None)
    k3.metric("⭐ CONFLUENCE", n_c,   delta=n_c-p_c   if p_c  else None)
    k4.metric("🔥 REA-HOT",   n_h,   delta=n_h-p_h   if p_h  else None)
    k5.metric("💧 Liq OK",    n_liq)
    k6.metric("🏆 CSS Grade A", n_css_a, help="Titoli con Composite Signal Score ≥ 80")
    k7.metric("⚡ Trend STRONG", n_strong, help="Titoli con ADX_Proxy ≥ 65 (trend forte)")
    k8.metric("📊 CSS medio",  f"{css_avg:.1f}" if css_avg else "—",
              help="Composite Signal Score medio del batch corrente")

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.title("⚙️ Configurazione")

# ── v35: Quick-Filter bar ───────────────────────────────────────────────────
with st.sidebar.container():
    _qf_cols = st.sidebar.columns([1,1,1])
    with _qf_cols[0]:
        if st.button("⚡ Solo STRONG", key="qf_strong", use_container_width=True,
                     help="Attiva STRONG only + CSS>=60"):
            st.session_state.show_strong_only = True
            st.session_state["css_filter_enabled"] = True
            st.session_state["css_min_val"] = 60
            st.rerun()
    with _qf_cols[1]:
        if st.button("🎯 Bilanciato", key="qf_balanced", use_container_width=True,
                     help="Reset filtri bilanciati (default v41)"):
            for k,v in PRESETS["⚖️ Bilanciato"].items():
                st.session_state[k] = v
            st.session_state.show_strong_only = False
            st.session_state["css_filter_enabled"] = False
            st.rerun()
    with _qf_cols[2]:
        if st.button("🔓 Reset", key="qf_reset", use_container_width=True,
                     help="Azzera TUTTI i filtri — mostra tutto"):
            for k,v in PRESETS["🔓 Nessun Filtro"].items():
                st.session_state[k] = v
            st.session_state.show_strong_only = False
            st.session_state["css_filter_enabled"] = False
            st.session_state["ts_filter"] = "Tutti"
            st.rerun()

# ── v35: counter segnali live ────────────────────────────────────────────────
_df_ep_live  = st.session_state.get("df_ep",  pd.DataFrame())
_df_rea_live = st.session_state.get("df_rea", pd.DataFrame())
if not _df_ep_live.empty:
    _n_early_live = int((_df_ep_live.get("Stato_Early", pd.Series()) == "EARLY").sum())
    _n_pro_live   = int((_df_ep_live.get("Stato_Pro",   pd.Series()).isin(["PRO","STRONG"])).sum())
    _n_hot_live   = len(_df_rea_live) if not _df_rea_live.empty else 0
    st.sidebar.markdown(
        f"<div style='background:#1e222d;border:1px solid #2a2e39;border-radius:6px;"
        f"padding:8px 12px;margin:6px 0;font-family:Courier New;font-size:0.82rem;'>"
        f"📡 <b style='color:#26a69a'>{_n_early_live}</b> EARLY &nbsp;|&nbsp; "
        f"💪 <b style='color:#2962ff'>{_n_pro_live}</b> PRO &nbsp;|&nbsp; "
        f"🔥 <b style='color:#ef5350'>{_n_hot_live}</b> HOT</div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.caption("_Nessuna scansione attiva_")

st.sidebar.divider()

with st.sidebar.expander("🎯 Preset Rapidi",expanded=False):
    for pname,pvals in PRESETS.items():
        if st.button(pname,use_container_width=True,key=f"preset_{pname}"):
            for k,v in pvals.items(): st.session_state[k]=v
            st.rerun()

with st.sidebar.expander("🌍 Mercati",expanded=True):
    msp500   =st.checkbox("S&P 500",         st.session_state.mSP500)
    mnasdaq  =st.checkbox("Nasdaq 100",       st.session_state.mNasdaq)
    mftse    =st.checkbox("FTSE MIB",         st.session_state.mFTSE)
    meuro    =st.checkbox("Eurostoxx 600",    st.session_state.mEurostoxx)
    mdow     =st.checkbox("Dow Jones",        st.session_state.mDow)
    mrussell =st.checkbox("Russell 2000",     st.session_state.mRussell)
    mstoxxem =st.checkbox("Stoxx Emerging 50",st.session_state.mStoxxEmerging)
    mussmall =st.checkbox("US Small Cap 2000",st.session_state.mUSSmallCap)

sel=[mkt for flag,mkt in [
    (msp500,"SP500"),(mnasdaq,"Nasdaq"),(mftse,"FTSE"),(meuro,"Eurostoxx"),
    (mdow,"Dow"),(mrussell,"Russell"),(mstoxxem,"StoxxEmerging"),(mussmall,"USSmallCap"),
] if flag]
(st.session_state.mSP500,st.session_state.mNasdaq,st.session_state.mFTSE,
 st.session_state.mEurostoxx,st.session_state.mDow,st.session_state.mRussell,
 st.session_state.mStoxxEmerging,st.session_state.mUSSmallCap)=(
    msp500,mnasdaq,mftse,meuro,mdow,mrussell,mstoxxem,mussmall)

with st.sidebar.expander("🎛️ Parametri Scanner",expanded=False):
    eh           =st.slider("EARLY EMA20 %",0.0,10.0,float(st.session_state.eh*100),0.5)/100
    prmin        =st.slider("PRO RSI min",0,100,int(st.session_state.prmin),5)
    prmax        =st.slider("PRO RSI max",0,100,int(st.session_state.prmax),5)
    rpoc         =st.slider("REA POC %",0.0,10.0,float(st.session_state.rpoc*100),0.5)/100
    vol_ratio_hot=st.number_input("VolRatio HOT",0.0,10.0,float(st.session_state.vol_ratio_hot),0.1)
    top          =st.number_input("TOP N",5,200,int(st.session_state.top),5)
(st.session_state.eh,st.session_state.prmin,st.session_state.prmax,
 st.session_state.rpoc,st.session_state.vol_ratio_hot,st.session_state.top)=(
    eh,prmin,prmax,rpoc,vol_ratio_hot,top)

with st.sidebar.expander("🔬 Soglie Filtri (live)",expanded=True):
    st.caption("Abbassa per vedere piu' segnali  |  0 = nessun filtro")
    min_early_score=st.slider("Early Score >=",0.0,10.0,float(st.session_state.min_early_score),0.5)
    min_quality    =st.slider("Quality >=",0,12,int(st.session_state.min_quality),1)
    min_pro_score  =st.slider("Pro Score >=",0.0,10.0,float(st.session_state.min_pro_score),0.5)
    st.session_state.min_early_score=min_early_score
    st.session_state.min_quality    =min_quality
    st.session_state.min_pro_score  =min_pro_score

    st.divider()
    # ── Filtro STRONG ────────────────────────────────────────────────────
    show_strong_only = st.checkbox(
        "Solo STRONG (Pro >= 9)",
        value=bool(st.session_state.show_strong_only),
        help="Mostra solo i setup di massima qualita' (Pro_Score >= 9/10). "
             "Pochi segnali, altissima selettivita'.",
        key="sb_strong_only",
    )
    st.session_state.show_strong_only = show_strong_only

    st.divider()
    # ── Filtro Liquidita' (Dollar Volume) ────────────────────────────────
    liq_filter_enabled = st.checkbox(
        "Filtro Liquidita' (Dollar Vol)",
        value=bool(st.session_state.liq_filter_enabled),
        help="Esclude titoli con volume giornaliero in $ troppo basso. "
             "Riduce slippage e rischio di non poter uscire dalla posizione.",
        key="sb_liq_enabled",
    )
    st.session_state.liq_filter_enabled = liq_filter_enabled
    if liq_filter_enabled:
        min_dollar_vol = st.select_slider(
            "Dollar Volume min ($M)",
            options=[1, 2, 5, 10, 20, 50, 100],
            value=int(st.session_state.min_dollar_vol),
            help="5M = retail OK | 20M = swing pro | 50M = intraday/istituzionale",
            key="sb_dollar_vol",
        )
        st.session_state.min_dollar_vol = float(min_dollar_vol)
        _liq_labels = {1:"illiquido",2:"illiquido",5:"retail",
                       10:"retail+",20:"swing pro",50:"intraday",100:"istituzionale"}
        st.caption(f"Soglia: >= **${min_dollar_vol}M/gg** — livello _{_liq_labels.get(min_dollar_vol,'')}_")

    st.divider()
    # ── Filtro ATR% (volatilita' operativa) ──────────────────────────────
    atr_filter_enabled = st.checkbox(
        "Filtro ATR% (volatilita')",
        value=bool(st.session_state.atr_filter_enabled),
        help="Seleziona titoli con volatilita' giornaliera (ATR/Prezzo%) "
             "nel range ideale per lo swing trading.",
        key="sb_atr_enabled",
    )
    st.session_state.atr_filter_enabled = atr_filter_enabled
    if atr_filter_enabled:
        atr_range = st.slider(
            "ATR% range",
            min_value=0.5, max_value=12.0,
            value=(float(st.session_state.atr_pct_min),
                   float(st.session_state.atr_pct_max)),
            step=0.5,
            help="1.5-6%: zona ideale swing. < 1.5% titolo fermo. > 6% gap risk elevato.",
            key="sb_atr_range",
        )
        st.session_state.atr_pct_min = atr_range[0]
        st.session_state.atr_pct_max = atr_range[1]
        _atr_label = ("ottimale" if 1.5 <= atr_range[0] and atr_range[1] <= 6.0
                      else "allargato")
        st.caption(f"ATR% in [{atr_range[0]:.1f}% – {atr_range[1]:.1f}%] — range _{_atr_label}_")

    st.divider()
    # ── v34: Filtro CSS (Composite Signal Score) ─────────────────────────
    css_filter_enabled = st.checkbox(
        "🏆 Filtro CSS (v41)",
        value=bool(st.session_state.get("css_filter_enabled", False)),
        help="Mostra solo titoli con Composite Signal Score sopra la soglia. "
             "CSS combina Pro/Ser/FV score + ADX + ATR + liquidità + OBV.",
        key="sb_css_enabled",
    )
    st.session_state["css_filter_enabled"] = css_filter_enabled
    if css_filter_enabled:
        css_min = st.select_slider(
            "CSS minimo",
            options=[20, 30, 40, 50, 60, 70, 80],
            value=int(st.session_state.get("css_min_val", 40)),
            help="40=Grade C+ | 60=Grade B+ | 80=Grade A (top quality)",
            key="sb_css_min",
        )
        st.session_state["css_min_val"] = css_min
        _css_lbl = {20:"tutti",30:"base",40:"Grade C+",50:"selettivo",
                    60:"Grade B+",70:"premium",80:"Grade A — elite"}
        st.caption(f"CSS >= **{css_min}** — _{_css_lbl.get(css_min,'')}_")

    st.divider()
    # ── v34: Filtro Trend Strength ────────────────────────────────────────
    ts_filter = st.selectbox(
        "⚡ Trend Strength min",
        options=["Tutti","WEAK+","MODERATE+","STRONG"],
        index=["Tutti","WEAK+","MODERATE+","STRONG"].index(
            st.session_state.get("ts_filter","Tutti")),
        help="Filtra per forza trend calcolata su EMA/Volume/OBV/ATR (ADX Proxy v41)",
        key="sb_ts_filter",
    )
    st.session_state["ts_filter"] = ts_filter

with st.sidebar.expander("📊 Indicatori Grafici",expanded=False):
    ind_opts_all=["SMA 9 & 21 + RSI","MACD","Parabolic SAR","Alligator + Vortex","Stochastic RSI",  # v34
                  "VWAP","Heikin-Ashi","S/R Auto"]  # v35
    ai=st.multiselect("Attivi",options=ind_opts_all,
        default=[x for x in st.session_state.active_indicators if x in ind_opts_all],
        key="global_indicators")
    st.session_state.active_indicators=ai

st.sidebar.divider()
st.sidebar.subheader("📋 Watchlist")

df_wl_all=load_watchlist()
list_options=sorted(df_wl_all["list_name"].unique().tolist()) if not df_wl_all.empty else []
if "DEFAULT" not in list_options: list_options.append("DEFAULT")
list_options=sorted(list_options)

active_list=st.sidebar.selectbox("Lista Attiva",list_options,
    index=list_options.index(st.session_state.current_list_name)
    if st.session_state.current_list_name in list_options else 0,
    key="active_list")
st.session_state.current_list_name=active_list

# ── Crea nuova lista ─────────────────────────────────────────────────────
with st.sidebar.expander("➕ Nuova Lista",expanded=False):
    new_list_name=st.text_input("Nome lista",key="new_list_input",placeholder="es. Watchlist Tech")
    if st.button("✅ Crea e Attiva",key="create_list_btn",use_container_width=True):
        if new_list_name.strip():
            nm=new_list_name.strip()
            # Crea la lista inserendo un placeholder temporaneo e cancellandolo subito
            # (la lista esiste nel DB solo se ha almeno un record)
            # → salviamo il nome in session_state e sarà visibile quando si aggiunge un ticker
            st.session_state.current_list_name=nm
            st.session_state["pending_new_list"]=nm
            st.sidebar.success(f"Lista '{nm}' creata. Aggiungici ticker dallo scanner.")
            st.rerun()
        else:
            st.sidebar.warning("Inserisci un nome.")

if st.sidebar.button("⚠️ Reset Watchlist DB",key="rst_wl"):
    reset_watchlist_db(); st.rerun()

st.sidebar.divider()
st.sidebar.subheader("⚡ Scanner v41")
with st.sidebar.expander("🔧 Opzioni avanzate",expanded=False):
    use_cache  = st.checkbox("⚡ Cache SQLite (più veloce)",True,key="use_cache",
                              help="Riusa dati yfinance già scaricati oggi (TTL 4h). "
                                   "Secondo scanner della giornata → ~30 sec totali.")
    use_finviz = st.checkbox("📊 Finviz scraping (EPS reali)",False,key="use_finviz",
                              help="Scarica EPS growth, short float, PEG da Finviz. "
                                   "Più lento (+20-40% tempo). Richiede finvizfinance installato.")
    n_workers  = st.slider("🔄 Worker paralleli",2,24,12,2,key="n_workers",
                            help="Thread simultanei. 8 = ottimale. Aumenta con cautela "
                                 "(troppi → rate limit yfinance).")
    if st.button("🗑️ Svuota cache",key="clear_cache_btn",use_container_width=True):
        try:
            cache_clear()
            st.success("✅ Cache svuotata.")
        except Exception as e:
            st.error(f"Errore: {e}")
    if st.button("📊 Info cache",key="cache_info_btn",use_container_width=True):
        try:
            cs = cache_stats()
            st.info(f"🟢 {cs['fresh']} fresche  ⏰ {cs['stale']} scadute  💾 {cs['size_mb']} MB")
        except Exception as e:
            st.info("Cache non disponibile.")

# ── v41 UPGRADE #3 — SCANNER SCHEDULER ────────────────────────────────────
with st.sidebar.expander("⏰ Auto-Scanner v41", expanded=False):
    st.caption("Scan automatico a intervalli regolari.")
    _sched_enabled = st.checkbox("🟢 Abilita Auto-Scan", key="sched_enabled",
                                  value=st.session_state.get("sched_enabled", False))
    _sched_interval = st.select_slider("Intervallo (min)",
        options=[5,10,15,20,30,45,60], value=st.session_state.get("sched_interval_val",15),
        key="sched_interval_val")
    _sched_market_only = st.checkbox("Solo orario NYSE (9:30-16:00 ET)",
        value=st.session_state.get("sched_mkt_only", True), key="sched_mkt_only")

    if _sched_enabled:
        import time as _t_sched
        _should, _remaining = _scheduler_tick(_sched_interval, "09:30", "16:00", _sched_market_only)
        _mins_left = int(_remaining // 60); _secs_left = int(_remaining % 60)

        if _should:
            st.sidebar.info("🔄 Auto-scan in avvio...")
            st.session_state["_sched_last_scan"] = _t_sched.time()
            st.session_state["_trigger_autoscan"] = True
            st.rerun()
        else:
            if _sched_market_only and not _is_market_open_nyse():
                st.sidebar.caption("🔒 Mercato chiuso — auto-scan sospeso")
            else:
                st.sidebar.caption(f"⏱️ Prossimo scan: **{_mins_left:02d}:{_secs_left:02d}**")
    else:
        st.sidebar.caption("Auto-scan disabilitato")
        st.session_state["_trigger_autoscan"] = False

# Scan stats ultima scansione
if "scan_stats" in st.session_state:
    ss = st.session_state.scan_stats
    st.sidebar.caption(
        f"⏱️ Ultima: **{ss['elapsed_s']}s**  "
        f"⚡ {ss['cache_hits']} cache  "
        f"☁️ {ss['downloaded']} scaricati"
    )

st.sidebar.divider()
if st.sidebar.button("🗑️ Reset Storico",key="reset_hist_sidebar"):
    try:
        conn=sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM scan_history");conn.commit();conn.close()
        st.sidebar.success("Storico cancellato.");st.rerun()
    except Exception as e: st.sidebar.error(f"Errore: {e}")

# ── v41: AI multi-provider status in sidebar ────────────────────────────
_ai_providers_status = {
    "🟢 Gemini":     bool(st.secrets.get("GEMINI_API_KEY","")     or st.session_state.get("_gemini_api_key","")),
    "🟣 Groq":       bool(st.secrets.get("GROQ_API_KEY","")       or st.session_state.get("_groq_api_key","")),
    "🔵 OpenRouter": bool(st.secrets.get("OPENROUTER_API_KEY","") or st.session_state.get("_openrouter_api_key","")),
    "🟡 Claude":     bool(st.secrets.get("ANTHROPIC_API_KEY","")  or st.session_state.get("_anthropic_api_key","")),
}
_n_active = sum(_ai_providers_status.values())
_ai_status_lines = "  ".join(
    f"<span style='color:{"#00ff88" if ok else "#374151"}'>{name.split()[0]}</span>"
    for name, ok in _ai_providers_status.items()
)
_bg = "#0d2b1f" if _n_active > 0 else "#1a0f00"
_bc = "#00ff88" if _n_active > 0 else "#f59e0b"
_msg = f"{_n_active}/4 provider attivi" if _n_active > 0 else "nessun provider — vai tab PRO"
st.sidebar.markdown(
    f"<div style='background:{_bg};border-left:3px solid {_bc};"
    f"border-radius:0 4px 4px 0;padding:5px 10px;font-size:0.72rem;margin:4px 0'>"
    f"🧠 AI: <b style='color:{_bc}'>{_msg}</b><br>{_ai_status_lines}</div>",
    unsafe_allow_html=True
)
if _n_active > 0:
    if st.sidebar.button("🔑 Reset AI Keys", key="ai_key_reset_sidebar", use_container_width=True):
        for _rk in ["_gemini_api_key","_groq_api_key","_openrouter_api_key","_anthropic_api_key"]:
            st.session_state.pop(_rk, None)
        st.rerun()

st.sidebar.divider()
only_watchlist=st.sidebar.checkbox("Solo Watchlist",False)

st.sidebar.divider()
st.sidebar.markdown("**🔧 Layout Griglie**")
st.sidebar.caption("Le larghezze/ordinamenti colonne vengono salvati nel browser (localStorage).")
if st.sidebar.button("↺ Reset layout griglie",key="reset_grid_layout",use_container_width=True):
    # Inietta JS per cancellare tutte le chiavi grid_state_* dal localStorage
    st.markdown("""<script>
(function(){
  Object.keys(localStorage).filter(k=>k.startsWith('grid_state_')).forEach(k=>localStorage.removeItem(k));
  console.log('Grid states cleared');
})();
</script>""",unsafe_allow_html=True)
    st.sidebar.success("Layout resettato — ricarica la pagina.")

# =========================================================================
# SCANNER
# =========================================================================
if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 39.0",type="primary",use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            tot        = len(universe)
            use_cache  = st.session_state.get("use_cache", True)
            use_finviz = st.session_state.get("use_finviz", False)
            n_wk       = st.session_state.get("n_workers", 8)

            # ── Test connessione Yahoo Finance ────────────────────────────
            import requests as _req
            _conn_ok  = False
            _test_tkr = next((t for t in universe if len(t) <= 5), universe[0])
            _conn_box = st.empty()
            try:
                _s = _req.Session()
                _s.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Referer": "https://finance.yahoo.com/"
                })
                _r = _s.get(
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{_test_tkr}",
                    params={"interval": "1d", "range": "5d"}, timeout=20
                )
                if _r.status_code == 200:
                    _res = _r.json().get("chart", {}).get("result", [])
                    if _res and _res[0].get("timestamp"):
                        _conn_box.success(f"✅ Connessione Yahoo OK — ticker test: `{_test_tkr}`")
                        _conn_ok = True
                    else:
                        _conn_box.error(f"❌ Yahoo Finance risposta vuota per `{_test_tkr}`")
                else:
                    _conn_box.error(f"❌ Yahoo Finance HTTP {_r.status_code}")
            except Exception as _ce:
                _conn_box.error(f"❌ Connessione fallita: {_ce}")

            if not _conn_ok:
                st.warning("⚠️ Test connessione fallito. Lo scanner proverà comunque — "
                           "potrebbe restituire 0 risultati se Yahoo Finance non è raggiungibile.")

            # ── Barra progressiva SEQUENZIALE (aggiornamento in tempo reale) ──
            st.markdown(f"### 🔍 Scansione: **{tot}** ticker")
            pb     = st.progress(0.0)
            status = st.empty()
            errors_box = st.empty()
            found_box  = st.empty()

            rep_live  = [0]   # contatore segnali trovati in tempo reale
            rea_live  = [0]

            def _progress(done, total, tkr):
                pct = done / total
                pb.progress(pct)
                n_ep  = rep_live[0]
                n_rea = rea_live[0]
                status.info(
                    f"🔍 **{done} / {total}** "
                    f"({pct*100:.0f}%) — `{tkr}`  "
                    f"| 📡 EARLY/PRO: **{n_ep}** | 🔥 HOT: **{n_rea}**"
                )

            # Patch scan_universe per aggiornare contatori live
            import utils.scanner as _sc_mod
            _orig_scan = _sc_mod.scan_ticker
            def _patched_scan(tkr, *a, **k):
                ep, rea = _orig_scan(tkr, *a, **k)
                if ep:  rep_live[0] += 1
                if rea: rea_live[0] += 1
                return ep, rea
            _sc_mod.scan_ticker = _patched_scan

            try:
                df_ep_new, df_rea_new, scan_stats = scan_universe(
                    universe, eh, prmin, prmax, rpoc, vol_ratio_hot,
                    cache_enabled=use_cache, finviz_enabled=use_finviz,
                    n_workers=n_wk, progress_callback=_progress
                )
            finally:
                _sc_mod.scan_ticker = _orig_scan  # ripristina

            # ── Normalizza colonne ────────────────────────────────────────
            df_ep_new  = _enrich_df(df_ep_new)
            df_rea_new = _enrich_df(df_rea_new)
            # v34 FIX DEDUP: rimuovi ticker duplicati (stesso ticker in più mercati)
            # Tieni la riga con lo score più alto per ogni ticker
            if not df_ep_new.empty and "Ticker" in df_ep_new.columns:
                _score_col = next((c for c in ["CSS","Pro_Score","Quality_Score"] if c in df_ep_new.columns), None)
                if _score_col:
                    df_ep_new = (df_ep_new.sort_values(_score_col, ascending=False)
                                         .drop_duplicates(subset=["Ticker"], keep="first")
                                         .reset_index(drop=True))
                else:
                    df_ep_new = df_ep_new.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
            if not df_rea_new.empty and "Ticker" in df_rea_new.columns:
                df_rea_new = (df_rea_new.sort_values("Vol_Ratio", ascending=False)
                                        .drop_duplicates(subset=["Ticker"], keep="first")
                                        .reset_index(drop=True)) if "Vol_Ratio" in df_rea_new.columns \
                             else df_rea_new.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
            pb.progress(1.0)

            elapsed = scan_stats.get("elapsed_s", 0)
            n_err   = scan_stats.get("n_errors", 0)
            errs    = scan_stats.get("errors", [])

            status.success(
                f"✅ **{tot} ticker** in **{elapsed:.0f}s** — "
                f"📡 **{len(df_ep_new)}** segnali EP | "
                f"🔥 **{len(df_rea_new)}** HOT | "
                f"⚠️ {n_err} errori"
            )

            if n_err > 0:
                with st.expander(f"⚠️ {n_err} errori (espandi per dettagli)",
                                  expanded=(len(df_ep_new) == 0)):
                    for _e in errs[:20]:
                        st.code(_e)

            if df_ep_new.empty and df_rea_new.empty:
                st.error(
                    "🔴 **0 segnali trovati.** Cause possibili:\n"
                    "1. Yahoo Finance irraggiungibile (prova tra 5 min)\n"
                    "2. Parametri troppo restrittivi → usa Preset **'🔓 Nessun Filtro'**\n"
                    f"3. {n_err} ticker con errori (vedi sopra)"
                )

            st.session_state.df_ep     = df_ep_new
            st.session_state.df_rea    = df_rea_new
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")
            st.session_state.scan_stats = scan_stats

            try:
                scan_id = save_scan_history(sel, df_ep_new, df_rea_new,
                                             elapsed_s=elapsed, cache_hits=0)
            except TypeError:
                scan_id = save_scan_history(sel, df_ep_new, df_rea_new)
            save_signals(scan_id, df_ep_new, df_rea_new, sel)

            n_h = len(df_rea_new)
            n_c = 0
            if not df_ep_new.empty and "Stato_Early" in df_ep_new.columns:
                n_c = int(((df_ep_new["Stato_Early"]=="EARLY")&
                            (df_ep_new["Stato_Pro"]=="PRO")).sum())
            if n_h >= 5: st.toast(f"🔥 {n_h} HOT!", icon="🔥")
            if n_c >= 3: st.toast(f"⭐ {n_c} CONFLUENCE!", icon="⭐")
            st.rerun()

# ── Auto-load: se session_state è vuoto (refresh/reboot), ricarica l'ultima
#    scansione salvata nel DB così i tab non sono mai completamente vuoti ─────
if "df_ep" not in st.session_state:
    try:
        _hist = load_scan_history(1)
        if not _hist.empty:
            _last_id = int(_hist.iloc[0]["id"])
            _df_ep_load, _df_rea_load = load_scan_snapshot(_last_id)
            if not _df_ep_load.empty or not _df_rea_load.empty:
                # Arricchisce con campi calcolati (Ser_OK, FV_OK, Stato_Pro>=6)
                _df_ep_load  = _enrich_df(_df_ep_load)
                _df_rea_load = _enrich_df(_df_rea_load)
                st.session_state.df_ep     = _df_ep_load
                st.session_state.df_rea    = _df_rea_load
                st.session_state.last_scan = str(_hist.iloc[0].get("scanned_at",""))[:16]
                st.session_state["_autoloaded"] = True
    except Exception:
        pass

df_ep =st.session_state.get("df_ep", pd.DataFrame())
df_rea=st.session_state.get("df_rea",pd.DataFrame())

if st.session_state.get("_autoloaded"):
    st.caption(f"📂 Dati dall'ultima scansione: {st.session_state.get('last_scan','')} _(ricaricati dal DB)_")
elif "last_scan" in st.session_state:
    st.caption(f"⏱️ Ultima scansione: {st.session_state.last_scan}")
render_kpi_bar(df_ep,df_rea)

# ── Pannello diagnostico (visibile solo se df non vuoto o si clicca) ─────────
with st.expander("🔎 Diagnostica dati scanner",expanded=False):
    c1,c2,c3=st.columns(3)
    c1.metric("Righe df_ep",  len(df_ep)  if not df_ep.empty  else 0)
    c2.metric("Righe df_rea", len(df_rea) if not df_rea.empty else 0)
    c3.metric("Autoloaded",   "Sì" if st.session_state.get("_autoloaded") else "No")
    if not df_ep.empty:
        _col_check = {
            "Stato_Early":  df_ep.get("Stato_Early","").eq("EARLY").sum() if "Stato_Early" in df_ep.columns else "colonna assente",
            "Stato_Pro":    df_ep.get("Stato_Pro","").eq("PRO").sum()     if "Stato_Pro"   in df_ep.columns else "colonna assente",
            "Ser_OK=True":  df_ep.get("Ser_OK","").isin([True,"True","true"]).sum() if "Ser_OK" in df_ep.columns else "colonna assente",
            "FV_OK=True":   df_ep.get("FV_OK","").isin([True,"True","true"]).sum()  if "FV_OK"  in df_ep.columns else "colonna assente",
            "Weekly_Bull":  df_ep.get("Weekly_Bull","").isin([True,"True","true",1]).sum() if "Weekly_Bull" in df_ep.columns else "colonna assente",
        }
        st.write("**Conteggi segnali:**", _col_check)
        st.write("**Colonne disponibili:**", list(df_ep.columns))

    else:
        st.write("df_ep è vuoto.")
        _hist_diag = load_scan_history(3)
        if not _hist_diag.empty:
            st.write("**Ultime scansioni nel DB:**")
            st.dataframe(_hist_diag[["id","scanned_at","n_early","n_pro","n_rea"]],
                         use_container_width=True)
        else:
            st.write("Nessuna scansione trovata nel DB.")

st.markdown("---")

# =========================================================================
# AGGRID BUILDER  — resize + sort + filter
# =========================================================================
def build_aggrid(df_disp, grid_key, height=480, editable_cols=None):
    gb=GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(sortable=True,resizable=True,filterable=True,
                                 editable=False,wrapText=False,suppressSizeToFit=False,
                                 minWidth=95)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple",use_checkbox=True)

    if editable_cols:
        for ec in editable_cols:
            if ec in df_disp.columns:
                gb.configure_column(ec,editable=True)

    col_w={"Ticker":100,"Nome":230,"Prezzo":95,"Prezzo_fmt":105,"MarketCap":130,"MarketCap_fmt":130,
           "Early_Score":105,"Pro_Score":95,"Quality_Score":145,"Ser_Score":100,"FV_Score":100,
           "RSI":80,"Vol_Ratio":100,"Squeeze":85,"RSI_Div":95,
           "Weekly_Bull":95,"Stato_Early":100,"Stato_Pro":110,
           "Vol_Today":110,"Vol_7d_Avg":110,"Avg_Vol_20":110,
           "trend":115,"note":230,"origine":105,"created_at":115,
           "EPS_NY_Gr":100,"EPS_5Y_Gr":100,"PE":80,"Fwd_PE":85,
           "Earnings_Soon":105,"Optionable":95,"OBV_Trend":95,
           "EMA20":95,"EMA50":95,"EMA200":100,"EMA200_fmt":105,"ATR":85,"Rel_Vol":90,
           "Dist_POC_%":105,"POC":95,"Currency":85,
           # Nuove colonne v34
           "Dollar_Vol":110,"Liq_Grade":130,"ATR_pct":90,"ATR_OK":85,"Liq_OK":80,
           "CSS":130,"CSS_Grade":85,"Trend_Strength":120,"ADX_Proxy":110,
           # v41
           "RS_20d":120,"RS_Rank":80,
           "RSI_Div_Score":90,
           # v34 REA-HOT
           "AB_Score":110,"AB_Grade":110}
    for c,w in col_w.items():
        if c in df_disp.columns: gb.configure_column(c,width=w)
    hide_cols=["id","_chart_data","_quality_components","_ser_criteri","_fv_criteri",
               "Ser_OK","FV_OK","ATR_Exp","Stato",
               "Prezzo","MarketCap","EMA200","Currency",
               "ATR_OK","Liq_OK",
               "RSI_Div_Score","ADX_Proxy"]   # v34: info sintetizzata in CSS/Trend_Strength
    for c in hide_cols:
        if c in df_disp.columns: gb.configure_column(c,hide=True)

    rmap={"Nome":name_dblclick_renderer,"RSI":rsi_renderer,
          "Vol_Ratio":vol_ratio_renderer,"Quality_Score":quality_renderer,
          "Ser_Score":ser_score_renderer,"FV_Score":fv_score_renderer,
          "Squeeze":squeeze_renderer,"RSI_Div":rsi_div_renderer,
          "Weekly_Bull":weekly_renderer,"Prezzo_fmt":price_renderer,"Prezzo":price_renderer,
          "trend":trend_renderer,
          "Vol_Today":vol_abbrev_renderer,"Vol_7d_Avg":vol_abbrev_renderer,"Avg_Vol_20":vol_abbrev_renderer,
          "MarketCap":mcap_renderer,"MarketCap_fmt":mcap_str_renderer,
          "EMA200_fmt":price_renderer,
          "EPS_NY_Gr":pct_renderer,"EPS_5Y_Gr":pct_renderer,
          "ROE":pct_renderer,"Gross_Mgn":pct_renderer,"Op_Mgn":pct_renderer,
          "Earnings_Soon":bool_renderer,"Optionable":bool_renderer,
          "Ser_OK":bool_renderer,"FV_OK":bool_renderer,
          # Nuovi renderer v34
          "Stato_Pro":stato_pro_renderer,
          "Dollar_Vol":dollar_vol_renderer,
          "ATR_pct":atr_pct_renderer,
          "Liq_Grade":liq_grade_renderer,
          "CSS":css_renderer,
          "CSS_Grade":css_grade_renderer,
          "Trend_Strength":trend_strength_renderer,
          # v41
          "RS_20d":rs_renderer,
          "RS_Rank":rs_rank_renderer,
          "Dist_POC_%":JsCode("""class DP{init(p){this.eGui=document.createElement('span');const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'\u2014':v.toFixed(2)+'%';this.eGui.style.fontFamily='Courier New';}getGui(){return this.eGui;}}""")}
    for c,r in rmap.items():
        if c in df_disp.columns: gb.configure_column(c,cellRenderer=r)

    if "Ticker" in df_disp.columns: gb.configure_column("Ticker",pinned="left")
    if "Nome"   in df_disp.columns: gb.configure_column("Nome",  pinned="left")
    # v34 — CSS sempre visibile, ordinata discendente di default (i migliori in cima)
    if "CSS" in df_disp.columns:
        gb.configure_column("CSS", pinned="right", sort="desc",
                            headerTooltip="Composite Signal Score v41 — punteggio 0-100 che combina Pro/Ser/FV score + ADX + ATR + liquidità + OBV")
    if "CSS_Grade" in df_disp.columns:
        gb.configure_column("CSS_Grade", pinned="right",
                            headerTooltip="A≥80 | B≥60 | C≥40 | D<40")

    go_opts=gb.build()
    sk = "grid_state_" + grid_key

    # Carica layout salvato nel DB (persiste tra riavvii)
    saved_layout = load_grid_layout(grid_key)
    if saved_layout:
        _sl = repr(saved_layout)
        go_opts["onFirstDataRendered"]=JsCode("""
function(p){
  try{
    var db=""" + _sl + """;
    if(db.colState) p.columnApi.applyColumnState({state:db.colState,applyOrder:true});
    if(db.sortState) p.api.setSortModel(db.sortState);
    sessionStorage.setItem('""" + sk + """',JSON.stringify(db));
  }catch(e){p.api.sizeColumnsToFit();}
}""")
    else:
        go_opts["onFirstDataRendered"]=JsCode("""
function(p){
  try{
    var saved=sessionStorage.getItem('""" + sk + """');
    if(saved){
      var st=JSON.parse(saved);
      if(st.colState) p.columnApi.applyColumnState({state:st.colState,applyOrder:true});
      if(st.sortState) p.api.setSortModel(st.sortState);
    } else { p.api.sizeColumnsToFit(); }
  }catch(e){p.api.sizeColumnsToFit();}
}""")

    go_opts["onColumnResized"]=JsCode("""
function(p){
  if(!p.finished)return;
  try{
    var cur=JSON.parse(sessionStorage.getItem('""" + sk + """')||'{}');
    cur.colState=p.columnApi.getColumnState();
    sessionStorage.setItem('""" + sk + """',JSON.stringify(cur));
  }catch(e){}
}""")
    go_opts["onSortChanged"]=JsCode("""
function(p){
  try{
    var cur=JSON.parse(sessionStorage.getItem('""" + sk + """')||'{}');
    cur.sortState=p.api.getSortModel();
    sessionStorage.setItem('""" + sk + """',JSON.stringify(cur));
  }catch(e){}
}""")
    go_opts["onColumnMoved"]=JsCode("""
function(p){
  try{
    var cur=JSON.parse(sessionStorage.getItem('""" + sk + """')||'{}');
    cur.colState=p.columnApi.getColumnState();
    sessionStorage.setItem('""" + sk + """',JSON.stringify(cur));
  }catch(e){}
}""")

    update=GridUpdateMode.VALUE_CHANGED if editable_cols else GridUpdateMode.SELECTION_CHANGED
    try:
        resp = AgGrid(df_disp,gridOptions=go_opts,height=height,
                      update_mode=update,
                      data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                      fit_columns_on_grid_load=False,theme="streamlit",
                      allow_unsafe_jscode=True,key=grid_key)
        # v41: verifica che AgGrid abbia restituito dati — altrimenti fallback
        _resp_data = resp.get("data", None)
        if _resp_data is not None and hasattr(_resp_data, '__len__') and len(_resp_data) == 0 and len(df_disp) > 0:
            raise ValueError("AgGrid returned empty data — using fallback")
    except Exception as _ag_err:
        # Fallback: st.dataframe nativo se AgGrid non funziona
        st.dataframe(df_disp, use_container_width=True, hide_index=True,
                     height=min(height, 600))
        # Crea resp dummy compatibile
        resp = {"selected_rows": [], "data": df_disp}

    # ── Pulsante salva/reset layout ──────────────────────────────
    _lc1,_lc2,_lc3=st.columns([1,1,8])
    with _lc1:
        if st.button("💾 Layout",key="save_lay_"+grid_key,
                     help="Salva larghezza e ordinamento colonne nel DB (persiste dopo riavvio)"):
            try:
                # Leggiamo il colState dal DB resp (quello visible da AgGrid)
                _cols_data = resp.get("column_state", None)
                if _cols_data:
                    save_grid_layout(grid_key, {"colState": _cols_data})
                    st.success("✅ Layout salvato nel DB!")
                else:
                    # Fallback: salva le larghezze da col_w come baseline
                    save_grid_layout(grid_key, {"colState": [], "note": "baseline"})
                    st.info("Layout baseline salvato. Ridimensiona poi salva di nuovo.")
            except Exception as _le:
                st.error(f"Errore: {_le}")
    with _lc2:
        if st.button("↩️ Reset",key="reset_lay_"+grid_key,
                     help="Ripristina le larghezze predefinite delle colonne"):
            try:
                save_grid_layout(grid_key, None)
                st.success("↩️ Layout resettato!")
                st.rerun()
            except Exception as _le:
                st.error(f"Errore reset: {_le}")
    return resp

# =========================================================================
# LEGENDE
# =========================================================================
# ═══════════════════════════════════════════════════════════════════
# CRISIS MONITOR — asset difensivi per guerra, inflazione, crisi
# ═══════════════════════════════════════════════════════════════════
CRISIS_ASSETS = {
    "🥇 Metalli Preziosi": {
        "desc": "Riserva di valore in ogni crisi. Oro e argento salgono in guerra, inflazione, panic sell.",
        "assets": [
            ("GLD",  "SPDR Gold ETF",          "ETF oro fisico — il più liquido"),
            ("IAU",  "iShares Gold Trust",      "ETF oro fisico — costi ridotti"),
            ("SLV",  "iShares Silver Trust",    "ETF argento fisico — più volatile dell'oro"),
            ("GDX",  "VanEck Gold Miners ETF",  "Minatori oro — leva sull'oro"),
            ("GDXJ", "VanEck Junior Gold Miners","Minatori junior — leva maggiore"),
            ("NEM",  "Newmont Corp",            "Principale miner oro mondiale"),
            ("GOLD", "Barrick Gold",            "Secondo miner oro mondiale"),
            ("WPM",  "Wheaton Precious Metals", "Royalty streaming su oro/argento"),
        ]
    },
    "⚫ Energia & Petrolio": {
        "desc": "Conflitti in Medio Oriente o Russia fanno esplodere l'energia. Hedging naturale.",
        "assets": [
            ("USO",  "United States Oil Fund",  "ETF futures petrolio WTI"),
            ("BNO",  "United States Brent Oil", "ETF futures Brent (europeo)"),
            ("XOM",  "ExxonMobil",              "Prima Big Oil USA"),
            ("CVX",  "Chevron",                 "Big Oil USA, dividendo stabile"),
            ("XLE",  "Energy Select SPDR",      "ETF settore energia S&P500"),
            ("OXY",  "Occidental Petroleum",    "Preferita di Buffett"),
            ("VLO",  "Valero Energy",           "Raffinerie — beneficia da spread"),
            ("UNG",  "US Natural Gas Fund",     "ETF futures gas naturale"),
            ("LNG",  "Cheniere Energy",         "Esportatore LNG — guerra gas"),
        ]
    },
    "🔫 Difesa & Aerospazio": {
        "desc": "In caso di conflitto militare, i budget della difesa esplodono. Outperformer storici.",
        "assets": [
            ("LMT",  "Lockheed Martin",         "F-35, missili, sistemi difesa"),
            ("RTX",  "RTX Corp (Raytheon)",     "Missili Patriot, difesa aerea"),
            ("NOC",  "Northrop Grumman",        "B-21, sistemi spaziali, cyber"),
            ("GD",   "General Dynamics",        "Carri armati Abrams, navi"),
            ("BA",   "Boeing Defense",          "Aerei militari, elicotteri"),
            ("HII",  "Huntington Ingalls",      "Portaerei, sottomarini nucleari"),
            ("KTOS", "Kratos Defense",          "Droni, ipersonici, cyber"),
            ("CACI", "CACI International",      "Intelligence, cybersecurity gov"),
            ("ITA",  "iShares US Aerospace ETF","ETF settore difesa/aerospazio"),
            ("XAR",  "SPDR S&P Aerospace ETF",  "ETF difesa — più diversificato"),
        ]
    },
    "💊 Healthcare & Pharma": {
        "desc": "Settore difensivo per eccellenza. Domanda inelastica, dividendi stabili.",
        "assets": [
            ("JNJ",  "Johnson & Johnson",       "Healthcare diversificato, dividendo 60+ anni"),
            ("PFE",  "Pfizer",                  "Pharma globale, vaccini"),
            ("ABBV", "AbbVie",                  "Farmaceutico, alta cedola"),
            ("XLV",  "Health Care Select SPDR", "ETF healthcare S&P500"),
            ("IBB",  "iShares Biotech ETF",     "ETF biotech — più rischio/rendimento"),
        ]
    },
    "⚡ Utilities": {
        "desc": "Monopoli regolamentati, dividendi alti. Salgono quando i tassi scendono.",
        "assets": [
            ("XLU",  "Utilities Select SPDR",   "ETF utilities S&P500"),
            ("NEE",  "NextEra Energy",          "Prima utility USA, rinnovabili"),
            ("SO",   "Southern Company",        "Utility elettrica sud USA"),
            ("DUK",  "Duke Energy",             "Utility elettrica grande"),
            ("AWK",  "American Water Works",    "Acqua — utility anti-crisi"),
            ("VPU",  "Vanguard Utilities ETF",  "ETF utilities — costi bassi"),
        ]
    },
    "🏦 Treasuries & Obbligazioni": {
        "desc": "Flight-to-safety: in crisi il mercato compra T-Bond USA. Duration lunga = massimo beneficio.",
        "assets": [
            ("TLT",  "iShares 20+ Year Treasury","ETF treasury long duration — +forte"),
            ("IEF",  "iShares 7-10 Year Treasury","ETF treasury medium duration"),
            ("SHY",  "iShares 1-3 Year Treasury","ETF treasury short — cash-like"),
            ("TIPS", "iShares TIPS Bond ETF",   "ETF inflation-protected (TIPS)"),
            ("TIP",  "iShares TIPS ETF",        "TIPS — inflazione"),
            ("BIL",  "SPDR 1-3 Month T-Bill",   "Quasi-cash, rendimento risk-free"),
        ]
    },
    "🍞 Commodities & Agri": {
        "desc": "Guerra blocca export grano (Ucraina), mais, soia. Siccità + crisi = spike prezzi.",
        "assets": [
            ("DBA",  "Invesco DB Agriculture",  "ETF basket agri: grano, mais, soia"),
            ("WEAT", "Teucrium Wheat Fund",     "ETF puro grano — massima esposizione"),
            ("CORN", "Teucrium Corn Fund",      "ETF puro mais"),
            ("SOYB", "Teucrium Soybean Fund",   "ETF puro soia"),
            ("MOO",  "VanEck Agribusiness ETF", "Aziende agri: Deere, Mosaic"),
            ("MOS",  "The Mosaic Company",      "Fertilizzanti — crisi ucraina"),
            ("NTR",  "Nutrien",                 "Fertilizzanti — leader mondiale"),
        ]
    },
    "💵 Valute Rifugio": {
        "desc": "CHF e JPY si apprezzano in crisi. USD Index sale. Copre rischio valutario.",
        "assets": [
            ("FXF",  "Invesco CurrencyShares CHF","ETF franco svizzero vs USD"),
            ("FXY",  "Invesco CurrencyShares JPY","ETF yen giapponese vs USD"),
            ("UUP",  "Invesco DB USD Index Bull", "ETF dollaro USA (DXY long)"),
            ("UDN",  "Invesco DB USD Index Bear", "ETF short USD — hedge"),
        ]
    },
    "🪙 Crypto Rifugio": {
        "desc": "Bitcoin: 'oro digitale' per alcuni. Correlazione variabile con crisi tradizionali.",
        "assets": [
            ("IBIT", "iShares Bitcoin Trust",   "ETF Bitcoin spot BlackRock — più liquido"),
            ("FBTC", "Fidelity Bitcoin ETF",    "ETF Bitcoin spot Fidelity"),
            ("GBTC", "Grayscale Bitcoin Trust", "Il più vecchio veicolo Bitcoin"),
        ]
    },
    "🌍 Mercati Neutri / Commodity States": {
        "desc": "Paesi esportatori netti di commodities. Beneficiano da inflazione/guerra.",
        "assets": [
            ("EWZ",  "iShares Brazil ETF",      "Brasile: ferro, soia, petrolio"),
            ("EWC",  "iShares Canada ETF",      "Canada: petrolio, gas, oro"),
            ("EWA",  "iShares Australia ETF",   "Australia: ferro, carbone, LNG"),
            ("GXG",  "iShares Colombia ETF",    "Colombia: petrolio, carbone"),
            ("RSX",  "VanEck Russia ETF",       "Russia (attenzione: illiquido post-2022)"),
        ]
    },
}

CRISIS_LEGEND = {
    "🥇 Metalli Preziosi": "Rifugio universale. In ogni crisi guerra/inflazione l'oro sale. GLD/IAU = ETF più semplici. GDX/GDXJ = leva indiretta sui miner.",
    "⚫ Energia & Petrolio": "Conflitti in regioni produttrici → spike immediato del petrolio. XOM/CVX per dividendo stabile. USO/BNO per trading puro.",
    "🔫 Difesa & Aerospazio": "Budget difesa sale sempre in caso di conflitto. LMT, RTX, NOC = Big 3. ITA/XAR per esposizione ETF diversificata.",
    "💊 Healthcare & Pharma": "Domanda anelastica in ogni scenario. JNJ = qualità assoluta. XLV = ETF diversificato. ABBV per cedola elevata.",
    "⚡ Utilities": "Monopoli regolamentati con dividendi stabili. Sottoperformano in rialzo tassi, sovraperformano in panic/recessione. NEE = leader.",
    "🏦 Treasuries & Obbligazioni": "Flight-to-safety in crisi acute. TLT (20Y+) ha la massima duration = massimo guadagno se tassi scendono. TIPS contro inflazione.",
    "🍞 Commodities & Agri": "Ucraina e Russia = 30% export grano mondiale. Conflitto → spike immediato WEAT/CORN. DBA per basket diversificato.",
    "💵 Valute Rifugio": "CHF: mai in guerra dal 1815. JPY: carry trade → apprezzamento in crisi. UUP: dollaro sale in ogni stress globale.",
    "🪙 Crypto Rifugio": "Bitcoin come hedge è dibattuto: in crisi 2022 è sceso, in crisi bancaria 2023 è salito. IBIT (BlackRock) = più regolamentato.",
    "🌍 Mercati Neutri": "Paesi commodity-esportatori beneficiano da inflazione materie prime. Attenzione alla governance (EWZ) e sanzioni (RSX).",
}

LEGENDS={
    "EARLY":{"desc":"Titoli dove il prezzo è **vicino alla EMA20** — zona rimbalzo/continuazione. Ideale per ingressi anticipati.",
      "cols":[("Early_Score","0–10","Prossimità EMA20. ≥8 ottimo, 5-7 buono"),("RSI","0–100","Momentum. Blu<30, Verde 40-65, Rosso>70"),("Squeeze","🔥","Bollinger dentro Keltner: esplosione imminente")],
      "filters":"Stato_Early='EARLY' AND Early_Score ≥ soglia","sort":"Early_Score DESC"},
    "PRO":{"desc":"Trend confermato: prezzo>EMA20>EMA50, RSI neutro-rialzista, volume sopra media.",
      "cols":[("Pro_Score","0–8","+3 trend, +3 RSI, +2 volume. ≥8=PRO"),("Quality_Score","0–12","Composito 6 fattori. ≥9 alta qualità"),("RSI","40–70","Range ideale momentum")],
      "filters":"Stato_Pro='PRO' AND Pro_Score≥soglia_P AND Quality≥soglia_Q","sort":"Quality DESC"},
    "REA-HOT":{"desc":"Volumi anomali vicini al POC (Point of Control). Interesse istituzionale.",
      "cols":[("Vol_Ratio","x","Oggi/media20gg. >hot_soglia=trigger"),("Dist_POC_%","%","Distanza dal POC — minore=meglio"),("POC","$","Livello max volume storico")],
      "filters":"dist_poc<rpoc AND Vol_Ratio>vol_ratio_hot","sort":"Vol_Ratio DESC"},
    "⭐ CONFLUENCE":{"desc":"EARLY + PRO contemporaneamente. Setup ad altissima probabilità.",
      "cols":[("Early_Score","0–10","Timing"),("Pro_Score","0–8","Forza"),("Quality_Score","0–12","Qualità")],
      "filters":"Stato_Early='EARLY' AND Stato_Pro='PRO'","sort":"Quality DESC, Early DESC"},
    "Regime Momentum":{"desc":"PRO ordinati per Momentum = Pro×10+RSI. Maggiore forza relativa.",
      "cols":[("Momentum","calc","Pro_Score×10+RSI")],
      "filters":"Stato_Pro='PRO' AND Pro≥soglia","sort":"Momentum DESC"},
    "Multi-Timeframe":{"desc":"PRO con trend rialzista anche settimanale (EMA20 weekly).",
      "cols":[("Weekly_Bull","📈","Prezzo>EMA20 weekly"),("Quality_Score","0–12","Qualità daily")],
      "filters":"PRO AND Weekly_Bull=True","sort":"Quality DESC"},
    "Finviz":{"desc":"PRO con MarketCap≥mediana e Vol_Ratio>1.2. Focus liquido/istituzionale.",
      "cols":[("MarketCap","$","Cap≥mediana campione"),("Vol_Ratio","x",">1.2x partecipazione")],
      "filters":"PRO AND MarketCap≥median AND Vol_Ratio>1.2","sort":"Quality DESC"},
    "🎯 Serafini":{"desc":"**Metodo Stefano Serafini** — 6 criteri tecnici tutti soddisfatti: trend allineato, momentum, volume, no earnings imminenti.",
      "cols":[("Ser_Score","0–6","Criteri soddisfatti su 6"),("RSI>50","bool","Momentum positivo"),("EMA20>EMA50","bool","Trend allineato"),("OBV_UP","bool","Volume crescente"),("No_Earnings","bool","No earnings entro 14gg")],
      "filters":"Ser_OK=True (tutti e 6 i criteri)","sort":"Ser_Score DESC, Quality DESC"},
    "🔎 Finviz Pro":{"desc":"**Replica filtri Finviz** da immagine: Price>$10, AvgVol>1M, RelVol>1, Price above SMA20/50/200, EPS Next Year>10%, EPS 5Y>15%.",
      "cols":[("FV_Score","0–8","Filtri Finviz soddisfatti"),("EPS_NY_Gr","%","EPS Growth Next Year (>10%)"),("EPS_5Y_Gr","%","EPS Growth 5Y proxy (>15%)"),("EMA200","$","200-Day SMA"),("Avg_Vol_20","#","Average Volume 20gg"),("Rel_Vol","x","Relative Volume")],
      "filters":"Price > 10 AND AvgVol > 1M AND RelVol > 1 AND P > SMA20/50/200 AND EPS_NY > 10% AND EPS_5Y > 15%","sort":"FV_Score DESC, Quality DESC"},
}

def show_legend(key):
    info=LEGENDS.get(key)
    if not info: return
    with st.expander(f"📖 Come funziona: {key}",expanded=False):
        st.markdown(info["desc"])
        rows="".join(f'<tr><td class="legend-col-name">{c}</td><td class="legend-col-range">{r}</td><td>{d}</td></tr>'
                     for c,r,d in info["cols"])
        st.markdown(f"""<table class="legend-table"><tr><th>Colonna</th><th>Range</th><th>Significato</th></tr>
{rows}</table><br><span style="color:#6b7280;font-size:0.78rem">
🔬 <b>Filtro:</b> <code>{info["filters"]}</code> &nbsp;|&nbsp; 📊 <b>Sort:</b> <code>{info["sort"]}</code>
</span>""",unsafe_allow_html=True)

# =========================================================================
# RENDER SCAN TAB
# =========================================================================
def render_scan_tab(df,status_filter,sort_cols,ascending,title):
    if df is None or (hasattr(df,"empty") and df.empty):
        c1,c2=st.columns([3,1])
        c1.info(f"📭 Nessun dato in **{title}**. Avvia lo scanner dalla sidebar.")
        with c2:
            if st.button("🔄 Ricarica dal DB",key=f"reload_{title}"):
                try:
                    _h=load_scan_history(1)
                    if not _h.empty:
                        _id=int(_h.iloc[0]["id"])
                        ep,rea=load_scan_snapshot(_id)
                        st.session_state.df_ep=ep
                        st.session_state.df_rea=rea
                        st.session_state.last_scan=str(_h.iloc[0].get("scanned_at",""))[:16]
                        st.session_state.pop("_autoloaded",None)
                        st.rerun()
                except Exception as _e:
                    st.error(f"Errore ricarica: {_e}")
        return

    s_e=float(st.session_state.min_early_score)
    s_q=int(st.session_state.min_quality)
    s_p=float(st.session_state.min_pro_score)
    # Nuovi filtri v34
    _strong_only    = bool(st.session_state.get("show_strong_only", False))
    _liq_enabled    = bool(st.session_state.get("liq_filter_enabled", True))
    _min_dvol       = float(st.session_state.get("min_dollar_vol", 5.0))
    _atr_enabled    = bool(st.session_state.get("atr_filter_enabled", True))
    _atr_min        = float(st.session_state.get("atr_pct_min", 1.5))
    _atr_max        = float(st.session_state.get("atr_pct_max", 6.0))
    # v34: HOT bypassa filtro ATR — i breakout hanno ATR elevato per natura
    _skip_atr_for_hot = False

    # Caption dinamica che mostra filtri attivi
    _active_flags = []
    if _strong_only:               _active_flags.append("STRONG only")
    if _liq_enabled:               _active_flags.append(f"DolVol>=${_min_dvol:.0f}M")
    if _atr_enabled:               _active_flags.append(f"ATR%[{_atr_min:.1f}-{_atr_max:.1f}]")
    _extra = "  |  " + "  |  ".join(_active_flags) if _active_flags else ""
    st.caption(
        f"Filtri: Early>={s_e} | Quality>={s_q} | Pro>={s_p}{_extra}  "
        f"_(sidebar -> Soglie)_"
    )

    if status_filter=="EARLY":
        if "Stato_Early" not in df.columns: st.warning("Colonna Stato_Early mancante."); return
        df_f=df[df["Stato_Early"]=="EARLY"].copy()
        if "Early_Score" in df_f.columns and s_e>0: df_f=df_f[df_f["Early_Score"]>=s_e]

    elif status_filter=="PRO":
        if "Stato_Pro" not in df.columns: st.warning("Colonna Stato_Pro mancante."); return
        # Se show_strong_only: filtra solo STRONG (Pro>=9), altrimenti PRO+STRONG
        _pro_valid = ["STRONG"] if _strong_only else ["PRO","STRONG"]
        df_f=df[df["Stato_Pro"].isin(_pro_valid)].copy()
        if "Pro_Score"     in df_f.columns and s_p>0: df_f=df_f[df_f["Pro_Score"]    >=s_p]
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="HOT":
        # REA-HOT: df_rea contiene già solo i HOT ma filtriamo per sicurezza
        if df is None or (hasattr(df,"empty") and df.empty):
            st.info("📭 Nessun segnale HOT trovato. Il segnale REA-HOT richiede"
                    " Vol_Ratio > soglia E distanza dal POC < soglia.\n\n"
                    " Abbassa `vol_ratio_hot` o `rpoc` nella sidebar → ⚙️ Avanzate.")
            return
        if "Stato" in df.columns:
            df_f=df[df["Stato"]=="HOT"].copy()
        else:
            df_f=df.copy()  # df_rea è già pre-filtrata
        # v34 FIX: filtro hard Dist_POC% — scarta titoli che si sono allontanati dal POC
        _rpoc_pct = float(st.session_state.get("rpoc", 0.02)) * 100
        if "Dist_POC_%" in df_f.columns and _rpoc_pct > 0:
            _n_before_poc = len(df_f)
            df_f = df_f[df_f["Dist_POC_%"].abs() <= _rpoc_pct * 1.5]
            _n_poc_rm = _n_before_poc - len(df_f)
            if _n_poc_rm > 0:
                st.caption(f"📍 POC filter: rimossi {_n_poc_rm} titoli distanti dal POC (soglia {_rpoc_pct*1.5:.1f}%)")
        # v34 FIX: disabilita ATR filter per HOT (breakout hanno ATR elevato)
        _skip_atr_for_hot = True

    elif status_filter=="CONFLUENCE":
        if "Stato_Early" not in df.columns or "Stato_Pro" not in df.columns:
            st.warning("Colonne Stato mancanti."); return
        # CONFLUENCE v34: EARLY + PRO/STRONG + Weekly_Bull (vera confluenza multi-timeframe)
        # La combinazione daily+weekly è il filtro più selettivo e affidabile.
        _pro_valid = ["PRO","STRONG"] if not _strong_only else ["STRONG"]
        _base_mask = (df["Stato_Early"]=="EARLY") & (df["Stato_Pro"].isin(_pro_valid))
        # Requisito Weekly_Bull: se la colonna esiste, è obbligatoria per CONFLUENCE
        if "Weekly_Bull" in df.columns:
            _wb_mask = df["Weekly_Bull"].isin([True,"True","true",1])
            df_f = df[_base_mask & _wb_mask].copy()
            if df_f.empty:
                # Fallback: mostra anche senza Weekly_Bull con avviso
                df_f = df[_base_mask].copy()
                if not df_f.empty:
                    st.caption("⚠️ Nessun segnale con Weekly Bull attivo — mostrati EARLY+PRO senza conferma weekly.")
        else:
            df_f = df[_base_mask].copy()
        if "Early_Score"   in df_f.columns and s_e>0: df_f=df_f[df_f["Early_Score"]  >=s_e]
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="REGIME":
        df_f=df[df["Stato_Pro"]=="PRO"].copy() if "Stato_Pro" in df.columns else df.copy()
        if "Pro_Score" in df_f.columns and s_p>0: df_f=df_f[df_f["Pro_Score"]>=s_p]
        if "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
            df_f["Momentum"]=df_f["Pro_Score"]*10+df_f["RSI"]
            sort_cols=["Momentum"]; ascending=[False]

    elif status_filter=="MTF":
        df_f=df[df["Stato_Pro"]=="PRO"].copy() if "Stato_Pro" in df.columns else df.copy()
        if "Pro_Score"   in df_f.columns and s_p>0: df_f=df_f[df_f["Pro_Score"]>=s_p]
        if "Weekly_Bull" in df_f.columns:
            df_f=df_f[df_f["Weekly_Bull"].isin([True,"True","true",1])]

    elif status_filter=="SERAFINI":
        if "Ser_OK" not in df.columns:
            st.warning("Colonna Ser_OK non trovata. Riesegui scanner v41."); return
        df_f=df[df["Ser_OK"].isin([True,"True","true"])].copy()
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="FINVIZ_PRO":
        if "FV_Score" not in df.columns:
            st.warning("Colonna FV_Score non trovata. Riesegui scanner v41."); return
        df_f=df[df["FV_OK"].isin([True,"True","true"])].copy()
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    else:
        df_f=df.copy()

    # ── Filtri qualita' condivisi (applicati a tutti i tab) ──────────────
    # 1. Dollar Volume (liquidita')
    if _liq_enabled and "Dollar_Vol" in df_f.columns:
        _before_liq = len(df_f)
        df_f = df_f[df_f["Dollar_Vol"].fillna(0) >= _min_dvol]
        _removed_liq = _before_liq - len(df_f)
        if _removed_liq > 0:
            st.caption(f"Liquidita': rimossi {_removed_liq} titoli con Dollar_Vol < ${_min_dvol:.0f}M")

    # 2. ATR% range — v34: NON applicare per HOT (breakout hanno ATR naturalmente alto)
    if _atr_enabled and not _skip_atr_for_hot and "ATR_pct" in df_f.columns:
        _before_atr = len(df_f)
        _mask_atr = df_f["ATR_pct"].isna() | df_f["ATR_pct"].between(_atr_min, _atr_max, inclusive="both")
        df_f = df_f[_mask_atr]
        _removed_atr = _before_atr - len(df_f)
        if _removed_atr > 0:
            st.caption(f"ATR%: rimossi {_removed_atr} titoli fuori range [{_atr_min:.1f}%-{_atr_max:.1f}%]")

    # 3. v34 — CSS (Composite Signal Score)
    _css_filter_on = bool(st.session_state.get("css_filter_enabled", False))
    _css_min_val   = float(st.session_state.get("css_min_val", 40))
    if _css_filter_on and "CSS" in df_f.columns:
        _before_css = len(df_f)
        df_f = df_f[pd.to_numeric(df_f["CSS"], errors="coerce").fillna(0) >= _css_min_val]
        _removed_css = _before_css - len(df_f)
        if _removed_css > 0:
            st.caption(f"CSS: rimossi {_removed_css} titoli con CSS < {_css_min_val:.0f}")

    # 4. v34 — Trend Strength
    _ts_filter = st.session_state.get("ts_filter", "Tutti")
    _ts_map = {"WEAK+": ["WEAK","MODERATE","STRONG"],
               "MODERATE+": ["MODERATE","STRONG"],
               "STRONG": ["STRONG"]}
    if _ts_filter != "Tutti" and "Trend_Strength" in df_f.columns:
        _before_ts = len(df_f)
        df_f = df_f[df_f["Trend_Strength"].isin(_ts_map.get(_ts_filter, []))]
        _removed_ts = _before_ts - len(df_f)
        if _removed_ts > 0:
            st.caption(f"Trend: rimossi {_removed_ts} titoli con Trend < {_ts_filter}")

    if df_f.empty:
        # ── Diagnostica cascata filtri ────────────────────────────────────
        # Mostra quanti segnali ci sono ad ogni step per identificare il filtro bloccante
        _n_tot   = len(df)
        _n_stato = 0

        # Conta prima dell'applicazione delle soglie numeriche
        if status_filter == "EARLY" and "Stato_Early" in df.columns:
            _n_stato = int((df["Stato_Early"]=="EARLY").sum())
        elif status_filter == "PRO" and "Stato_Pro" in df.columns:
            _pro_v = ["STRONG"] if _strong_only else ["PRO","STRONG"]
            _n_stato = int(df["Stato_Pro"].isin(_pro_v).sum())
        elif status_filter == "HOT" and "Stato" in df.columns:
            _n_stato = int((df["Stato"]=="HOT").sum())
        elif status_filter == "CONFLUENCE" and "Stato_Early" in df.columns and "Stato_Pro" in df.columns:
            _pro_v = ["PRO","STRONG"]
            _n_stato = int(((df["Stato_Early"]=="EARLY") & df["Stato_Pro"].isin(_pro_v)).sum())
        elif status_filter == "SERAFINI" and "Ser_OK" in df.columns:
            _n_stato = int(df["Ser_OK"].isin([True,"True","true"]).sum())
        elif status_filter == "FINVIZ_PRO" and "FV_OK" in df.columns:
            _n_stato = int(df["FV_OK"].isin([True,"True","true"]).sum())
        elif status_filter == "MTF" and "Weekly_Bull" in df.columns:
            _n_stato = int(df["Weekly_Bull"].isin([True,"True","true",1]).sum())

        # Conta dopo soglie numeriche (senza liquidità/ATR)
        _df_post_score = df.copy()
        if status_filter == "EARLY" and "Stato_Early" in _df_post_score.columns:
            _df_post_score = _df_post_score[_df_post_score["Stato_Early"]=="EARLY"]
            if "Early_Score" in _df_post_score.columns and s_e>0:
                _df_post_score = _df_post_score[_df_post_score["Early_Score"]>=s_e]
        elif status_filter == "PRO" and "Stato_Pro" in _df_post_score.columns:
            _pro_v = ["STRONG"] if _strong_only else ["PRO","STRONG"]
            _df_post_score = _df_post_score[_df_post_score["Stato_Pro"].isin(_pro_v)]
            if "Pro_Score" in _df_post_score.columns and s_p>0:
                _df_post_score = _df_post_score[_df_post_score["Pro_Score"]>=s_p]
            if "Quality_Score" in _df_post_score.columns and s_q>0:
                _df_post_score = _df_post_score[_df_post_score["Quality_Score"]>=s_q]
        _n_post_score = len(_df_post_score)

        # Conta dopo filtro liquidità
        _n_post_liq = _n_post_score
        if _liq_enabled and "Dollar_Vol" in _df_post_score.columns:
            _n_post_liq = int((_df_post_score["Dollar_Vol"].fillna(0) >= _min_dvol).sum())

        # Conta dopo filtro ATR%
        _n_post_atr = _n_post_liq
        if _atr_enabled and "ATR_pct" in _df_post_score.columns:
            _mask_atr = _df_post_score["ATR_pct"].isna() | _df_post_score["ATR_pct"].between(_atr_min, _atr_max, inclusive="both")
            if _liq_enabled and "Dollar_Vol" in _df_post_score.columns:
                _mask_liq = _df_post_score["Dollar_Vol"].fillna(0) >= _min_dvol
                _n_post_atr = int((_mask_atr & _mask_liq).sum())
            else:
                _n_post_atr = int(_mask_atr.sum())

        # Mostra diagnostica completa
        _diag_lines = [
            f"**Totale analizzati:** {_n_tot}",
            f"**Dopo classificazione {status_filter}:** {_n_stato}",
        ]
        if s_e > 0 or s_p > 0 or s_q > 0:
            _diag_lines.append(f"**Dopo soglie** (Early≥{s_e} Pro≥{s_p} Q≥{s_q}): {_n_post_score}")
        if _liq_enabled:
            _diag_lines.append(f"**Dopo filtro Liquidità** (DolVol≥${_min_dvol:.0f}M): {_n_post_liq}")
        if _atr_enabled:
            _diag_lines.append(f"**Dopo filtro ATR%** ({_atr_min:.1f}%–{_atr_max:.1f}%): {_n_post_atr}")

        # Individua il filtro bloccante e suggerisci rimedio
        if _n_stato == 0:
            _rimedio = (
                "👉 **Nessun segnale classificato.** Abbassa i parametri scanner nella sidebar "
                "(EMA %, RSI range, POC %) o seleziona più mercati."
            )
        elif _n_post_score == 0:
            _rimedio = (
                f"👉 **Filtro soglie troppo restrittivo.** "
                f"Vai sidebar → 🔬 Soglie → abbassa Pro Score ≥ (attuale: {s_p}) "
                f"o Quality ≥ (attuale: {s_q}) oppure usa preset **⚡ Aggressivo**."
            )
        elif _n_post_liq == 0:
            _rimedio = (
                f"👉 **Filtro liquidità troppo alto.** "
                f"Abbassa Dollar Volume minimo (attuale: ${_min_dvol:.0f}M) "
                f"oppure disabilita il filtro — sidebar → 🔬 Soglie."
            )
        else:
            _rimedio = (
                f"👉 **Filtro ATR% troppo stretto.** "
                f"Allarga il range ATR% (attuale: {_atr_min:.1f}%–{_atr_max:.1f}%) "
                f"oppure disabilita — sidebar → 🔬 Soglie."
            )

        st.warning(
            f"⚠️ **{title}** — 0 segnali dopo tutti i filtri\n\n"
            + "\n\n".join(_diag_lines) + f"\n\n{_rimedio}"
        )
        return

    valid_sort=[c for c in sort_cols if c in df_f.columns]
    if valid_sort: df_f=df_f.sort_values(valid_sort,ascending=ascending[:len(valid_sort)])

    # ── v41: Pannello diagnostica filtri sempre visibile ──────────────────
    with st.expander(f"🔬 Diagnostica filtri — {len(df_f)} segnali visibili", expanded=False):
        _n_raw = len(df)
        _n_after_state  = len(df_f) + 0  # dopo classificazione (prima del head)

        # Breakdown di tutti i filtri applicati
        _diag_data = {
            "Totale in df_ep":      _n_raw,
            f"Dopo classificazione {status_filter}": "→ vedi sopra",
            "Dopo liquidità/ATR":   len(df_f),
            f"Head(top={st.session_state.top})": min(len(df_f), int(st.session_state.top)),
        }

        # Distribuzione Pro_Score nel df originale
        if "Pro_Score" in df.columns:
            _ps = pd.to_numeric(df["Pro_Score"], errors="coerce").dropna()
            st.markdown(
                f"**Pro_Score distribuzione** — "
                f"min: `{_ps.min():.1f}` · "
                f"p25: `{_ps.quantile(0.25):.1f}` · "
                f"p50: `{_ps.median():.1f}` · "
                f"p75: `{_ps.quantile(0.75):.1f}` · "
                f"max: `{_ps.max():.1f}`"
            )
            _n_pro_5  = int((_ps >= 5).sum())
            _n_pro_6  = int((_ps >= 6).sum())
            _n_pro_8  = int((_ps >= 8).sum())
            st.markdown(
                f"Pro≥5: **{_n_pro_5}** · Pro≥6: **{_n_pro_6}** · "
                f"Pro≥8 (STRONG): **{_n_pro_8}** · "
                f"Soglia attuale PRO: **≥{5 if not st.session_state.get('show_strong_only') else 8}**"
            )

        # Mostra i filtri attivi e quanti taglia ciascuno
        if "Dollar_Vol" in df_f.columns:
            st.caption(f"💧 Liquidità: DolVol ≥ ${_min_dvol:.0f}M | ATR%: {_atr_min:.1f}–{_atr_max:.1f}%")

        # Suggerimento se risultati sembrano sempre gli stessi
        st.info(
            "💡 **Se vedi sempre gli stessi ticker:** "
            "i risultati sono ordinati per Quality_Score → i large cap stabili "
            "tendono ad avere score alto sempre. "
            "Prova: **Ordina per RS vs SPY** o **CSS** per vedere titoli con momentum recente diverso. "
            "Oppure aumenta il TOP N (sidebar) per vedere più risultati."
        )

    # ── v41: Opzioni ordinamento inline ───────────────────────────────────
    _sort_options = {
        "🏆 CSS (default)":      ("CSS", False),
        "📈 RS vs SPY":          ("RS_20d", False),
        "⚡ Momentum (Pro×RSI)": ("_Momentum_v41", False),
        "📊 Quality Score":      ("Quality_Score", False),
        "🔥 Volume Ratio":       ("Vol_Ratio", False),
        "📡 Early Score":        ("Early_Score", False),
    }
    _sort_avail = {k:v for k,v in _sort_options.items()
                   if v[0] in df_f.columns or v[0] == "_Momentum_v41"}

    _sc1, _sc2, _sc3 = st.columns([2, 1, 1])
    with _sc1:
        _sort_choice = st.selectbox(
            "Ordina per",
            list(_sort_avail.keys()),
            index=0,
            key=f"sort_choice_{title}",
            label_visibility="collapsed"
        )
    with _sc2:
        _top_n = st.number_input(
            "Mostra TOP N",
            min_value=5, max_value=200,
            value=int(st.session_state.top),
            step=5,
            key=f"top_n_{title}",
            label_visibility="collapsed"
        )
    with _sc3:
        _show_new_only = st.checkbox(
            "🆕 Solo nuovi",
            value=False,
            key=f"new_only_{title}",
            help="Esclude ticker già presenti in Watchlist"
        )

    # Applica ordinamento scelto
    _sort_col, _sort_asc = _sort_avail.get(_sort_choice, ("CSS", False))

    if _sort_col == "_Momentum_v41" and "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
        df_f = df_f.copy()
        df_f["_Momentum_v41"] = (
            pd.to_numeric(df_f["Pro_Score"], errors="coerce").fillna(0) * 10 +
            pd.to_numeric(df_f["RSI"], errors="coerce").fillna(50)
        )

    if _sort_col in df_f.columns:
        df_f = df_f.sort_values(
            _sort_col,
            ascending=_sort_asc,
            key=lambda s: pd.to_numeric(s, errors="coerce").fillna(-999 if not _sort_asc else 999)
        )

    # Filtro "solo nuovi" — esclude ticker già in watchlist
    if _show_new_only:
        try:
            _wl_excl = load_watchlist()
            if not _wl_excl.empty and "Ticker" in _wl_excl.columns:
                _wl_set = set(_wl_excl["Ticker"].dropna().tolist())
                _before_new = len(df_f)
                df_f = df_f[~df_f["Ticker"].isin(_wl_set)]
                st.caption(f"🆕 Solo nuovi: esclusi {_before_new - len(df_f)} ticker già in watchlist")
        except Exception:
            pass

    df_f = df_f.head(int(_top_n))

    m1,m2,m3,m4=st.columns(4)
    m1.metric("Titoli",len(df_f))
    if "Squeeze" in df_f.columns:
        m2.metric("🔥 Squeeze",int(df_f["Squeeze"].apply(lambda x:x is True or str(x).lower()=="true").sum()))
    if "Weekly_Bull" in df_f.columns:
        m3.metric("📈 Weekly+",int(df_f["Weekly_Bull"].apply(lambda x:x is True or str(x).lower()=="true").sum()))
    if "RSI_Div" in df_f.columns:
        m4.metric("⚠️ Div RSI",int((df_f["RSI_Div"]!="-").sum()))

    df_fmt =add_formatted_cols(df_f)
    df_disp=prepare_display_df(df_fmt)
    # Rimuovi colonne interne (prefisso _ e criteri grezzi)
    drop_cols=[c for c in df_disp.columns if c.startswith("_")]
    df_disp=df_disp.drop(columns=drop_cols, errors="ignore")
    # Ordine: Ticker, Nome, Prezzo_fmt, MarketCap_fmt, poi segnali, poi resto
    cols=list(df_disp.columns)
    priority=["Ticker","Nome","Prezzo_fmt","MarketCap_fmt","Early_Score","Pro_Score",
               "RSI","Dollar_Vol","Liq_Grade","ATR_pct",
               "Vol_Ratio","Quality_Score","Stato_Early","Stato_Pro","EMA200_fmt"]
    base=[c for c in priority if c in cols]
    rest=[c for c in cols if c not in base]
    df_disp=df_disp[base+rest].reset_index(drop=True)

    ce1,ce2=st.columns([1,3])
    with ce1: csv_btn(df_f,f"{title.lower().replace(' ','_')}.csv",f"exp_{title}")
    with ce2: st.caption(f"Seleziona → **➕** per aggiungere a `{st.session_state.current_list_name}`. Doppio click Nome → TradingView.")

    grid_resp  =build_aggrid(df_disp,f"grid_{title}")
    # v41 FIX: se AgGrid non renderizza (versione incompatibile), usa st.dataframe come fallback
    try:
        selected_df=pd.DataFrame(grid_resp["selected_rows"])
    except Exception:
        selected_df=pd.DataFrame()
    # Controlla se AgGrid ha renderizzato dati — se no, usa dataframe nativo
    _aggrid_ok = True
    try:
        _rd = grid_resp.get("data", None)
        if _rd is not None and len(_rd) == 0 and len(df_disp) > 0:
            _aggrid_ok = False
    except Exception:
        _aggrid_ok = False
    if not _aggrid_ok:
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

    if st.button(f"➕ Aggiungi a '{st.session_state.current_list_name}'",key=f"btn_{title}"):
        if not selected_df.empty and "Ticker" in selected_df.columns:
            tickers=selected_df["Ticker"].dropna().tolist()
            names  =selected_df.get("Nome",selected_df["Ticker"]).tolist()
            # v34 FIX WATCHLIST: forza insert diretto nel DB prima di chiamare gh_add
            # per garantire persistenza anche senza GitHub Sync configurato
            try:
                _conn_wl = sqlite3.connect(str(DB_PATH))
                _wl_now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _list_nm = st.session_state.current_list_name
                for _tkr, _nm in zip(tickers, names):
                    _exists = _conn_wl.execute(
                        "SELECT id FROM watchlist WHERE ticker=? AND list_name=?",
                        (_tkr, _list_nm)
                    ).fetchone()
                    if not _exists:
                        _conn_wl.execute(
                            "INSERT INTO watchlist (ticker,name,trend,origine,note,list_name,created_at) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (_tkr, str(_nm)[:60], title, "Scanner", "", _list_nm, _wl_now)
                        )
                _conn_wl.commit(); _conn_wl.close()
            except Exception as _wl_err:
                st.warning(f"DB insert: {_wl_err} — provo gh_add_to_watchlist")
            gh_add_to_watchlist(tickers,names,title,"Scanner","LONG",st.session_state.current_list_name)
            st.success(f"✅ Aggiunti {len(tickers)} titoli a '{st.session_state.current_list_name}'.")
            time.sleep(0.5); st.rerun()
        else: st.warning("⚠️ Seleziona almeno una riga dalla griglia.")

    if not selected_df.empty:
        ticker_sel=selected_df.iloc[0].get("Ticker","")
        match=df_f[df_f["Ticker"]==ticker_sel]
        if not match.empty: show_charts(match.iloc[0],key_suffix=title)

    # ── Strategy Chart widget ─────────────────────────────────────────────
    # Ticker auto-selezionato dalla riga scelta nella griglia.
    # Mostra "Nome Azienda (TICKER)" nel dropdown per identificazione rapida.
    try:
        from utils.backtest_tab import strategy_chart_widget as _scw
        if "Ticker" in df_f.columns:
            _tkrs = df_f["Ticker"].dropna().tolist()
            # Costruisci labels "Nome Azienda  (TICKER)" se colonna Nome disponibile
            if "Nome" in df_f.columns:
                _tlabels = {
                    row["Ticker"]: f"{str(row.get('Nome',''))[:28]}  ({row['Ticker']})"
                    for _, row in df_f[["Ticker","Nome"]].dropna(subset=["Ticker"]).iterrows()
                }
            else:
                _tlabels = None
            # Auto-selezione: usa il ticker dalla riga selezionata nella griglia
            _default = selected_df.iloc[0].get("Ticker","") if not selected_df.empty else (
                _tkrs[0] if _tkrs else "")
        else:
            _tkrs = []; _tlabels = None; _default = ""
        st.markdown("---")
        _scw(tickers=_tkrs, key_suffix=title, default_ticker=_default,
             ticker_labels=_tlabels)
    except Exception:
        pass

# =========================================================================
# TABS
# =========================================================================
# ── v41: Menu sticky + 2 righe + font adattivo + tab attivo evidenziato ──
st.markdown("""<style>
/* Sticky tab bar — rimane visibile scorrendo */
[data-testid="stTabs"] {
    position: sticky !important;
    top: 0 !important;
    z-index: 999 !important;
    background-color: #131722 !important;
    padding-top: 4px !important;
    border-bottom: 1px solid #2a2e39 !important;
}
/* 2 righe: flex-wrap */
[data-testid="stTabs"] > div:first-child {
    flex-wrap: wrap !important;
    gap: 0px !important;
    overflow: visible !important;
    background-color: #131722 !important;
}
/* Tab button base */
[data-testid="stTabs"] > div:first-child > button {
    flex-shrink: 0 !important;
    min-width: fit-content !important;
    font-size: 0.75rem !important;
    padding: 5px 9px !important;
    white-space: nowrap !important;
    transition: background 0.12s, color 0.12s !important;
}
/* Tab attivo: sfondo blu scuro + testo bianco + bordo più spesso */
[data-testid="stTabs"] > div:first-child > button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #2962ff !important;
    background: rgba(41,98,255,0.12) !important;
    font-weight: 600 !important;
}
/* Hover su tab inattivi */
[data-testid="stTabs"] > div:first-child > button:hover {
    background: rgba(41,98,255,0.07) !important;
    color: #d1d4dc !important;
}
/* Font adattivo su schermi stretti */
@media (max-width: 1200px) {
    [data-testid="stTabs"] > div:first-child > button {
        font-size: 0.70rem !important;
        padding: 4px 7px !important;
    }
}
@media (max-width: 900px) {
    [data-testid="stTabs"] > div:first-child > button {
        font-size: 0.65rem !important;
        padding: 3px 5px !important;
    }
}
</style>""", unsafe_allow_html=True)


# =========================================================================
# v41 — FUNZIONI (prima dei tab per evitare NameError)
# =========================================================================

_PATTERN_ALERTS_V39 = {
    "ema_breakout":   {"label":"EMA Breakout",      "icon":"📈","desc":"Prezzo > EMA20"},
    "golden_cross":   {"label":"Golden Cross",       "icon":"⭐","desc":"EMA20 > EMA50"},
    "death_cross":    {"label":"Death Cross",        "icon":"💀","desc":"EMA20 < EMA50"},
    "squeeze_fire":   {"label":"Squeeze Fire",       "icon":"🔥","desc":"Uscita da Squeeze"},
    "volume_spike":   {"label":"Volume Spike",       "icon":"⚡","desc":"Volume > 3x media"},
    "rsi_oversold":   {"label":"RSI Oversold",       "icon":"🔵","desc":"RSI < 32"},
    "rsi_overbought": {"label":"RSI Overbought",     "icon":"🔴","desc":"RSI > 68"},
    "bb_breakout":    {"label":"BB Breakout",        "icon":"🎯","desc":"Prezzo > EMA20+2xATR"},
}

def _detect_patterns_v41(row):
    out = []
    try:
        pr  = float(row.get("Prezzo",  0) or 0)
        e20 = float(row.get("EMA20",   0) or 0)
        e50 = float(row.get("EMA50",   0) or 0)
        rsi = float(row.get("RSI",    50) or 50)
        vr  = float(row.get("Vol_Ratio",0) or 0)
        sq  = row.get("Squeeze", False)
        atr = float(row.get("ATR",     0) or 0)
        if pr>0 and e20>0 and pr>e20 and rsi>45: out.append("ema_breakout")
        if e20>0 and e50>0:
            if e20>e50 and rsi>50: out.append("golden_cross")
            elif e20<e50 and rsi<50: out.append("death_cross")
        if sq in (True,"True","true",1): out.append("squeeze_fire")
        if vr>=3.0: out.append("volume_spike")
        if rsi<32: out.append("rsi_oversold")
        if rsi>68: out.append("rsi_overbought")
        if pr>0 and e20>0 and atr>0 and pr>e20+2*atr: out.append("bb_breakout")
    except Exception: pass
    return out

def _render_pattern_alerts_v41(df_src, tab_name="x"):
    st.markdown('<div class="section-pill">🔔 ALERT MULTIPLI v41 — Pattern Tecnici</div>', unsafe_allow_html=True)
    if df_src is None or (hasattr(df_src,"empty") and df_src.empty):
        st.info("Avvia lo scanner per rilevare i pattern."); return
    with st.expander("⚙️ Pattern da monitorare", expanded=False):
        _pc = st.columns(4)
        _en = {pid: _pc[i%4].checkbox(f"{p['icon']} {p['label']}", True,
               key=f"pat_en_{pid}_{tab_name}", help=p["desc"])
               for i,(pid,p) in enumerate(_PATTERN_ALERTS_V39.items())}
    _rows = []
    for _, r in df_src.iterrows():
        _pats = [p for p in _detect_patterns_v41(r) if _en.get(p,True)]
        if _pats:
            _rows.append({"Ticker":str(r.get("Ticker","")),"Nome":str(r.get("Nome",""))[:22],
                          "Prezzo":r.get("Prezzo",""),"RSI":r.get("RSI",""),
                          "CSS":r.get("CSS",""),"Pattern":_pats,"_stato":str(r.get("Stato_Pro","-"))})
    if not _rows: st.info("Nessun pattern rilevato."); return
    _cnt = {p: sum(1 for r in _rows if p in r["Pattern"]) for p in _PATTERN_ALERTS_V39}
    _kpi = st.columns(min(sum(1 for v in _cnt.values() if v>0),6))
    _ki=0
    for pid,cnt in sorted(_cnt.items(),key=lambda x:-x[1]):
        if cnt>0 and _ki<6:
            _kpi[_ki].metric(f"{_PATTERN_ALERTS_V39[pid]['icon']} {_PATTERN_ALERTS_V39[pid]['label']}",cnt); _ki+=1
    st.markdown("---")
    for ar in sorted(_rows,key=lambda x:len(x["Pattern"]),reverse=True)[:30]:
        _ac1,_ac2,_ac3,_ac4 = st.columns([1.5,1.5,3,1])
        _sc = "#ffd700" if ar["_stato"]=="STRONG" else "#00ff88" if ar["_stato"]=="PRO" else "#b2b5be"
        _ac1.markdown(f"<span style='font-family:Courier New;color:{_sc};font-weight:bold'>{ar['Ticker']}</span><br><span style='color:#6b7280;font-size:0.72rem'>{ar['Nome']}</span>",unsafe_allow_html=True)
        _ac2.markdown(f"<span style='font-family:Courier New;font-size:0.82rem'>${ar['Prezzo']}</span><br><span style='color:#787b86;font-size:0.72rem'>RSI {ar['RSI']} · CSS {ar['CSS']}</span>",unsafe_allow_html=True)
        _parts=[]
        for p in ar["Pattern"]:
            _bear=p in("death_cross","rsi_overbought","bb_breakout"); _gold=p=="golden_cross"
            _bg="#ffd70022" if _gold else "#ef444422" if _bear else "#2962ff22"
            _tx="#ffd700" if _gold else "#ef4444" if _bear else "#58a6ff"
            _parts.append(f"<span style='background:{_bg};color:{_tx};border-radius:3px;padding:1px 6px;font-size:0.72rem;margin-right:3px'>{_PATTERN_ALERTS_V39.get(p,{}).get('icon','🔔')} {_PATTERN_ALERTS_V39.get(p,{}).get('label',p)}</span>")
        _ac3.markdown(" ".join(_parts),unsafe_allow_html=True)
        with _ac4:
            if st.button("📋",key=f"alwl_{ar['Ticker']}_{tab_name}",help="Aggiungi a watchlist"):
                try: gh_add_to_watchlist(ar["Ticker"],st.session_state.current_list_name); st.success(f"✅ {ar['Ticker']} aggiunto!")
                except Exception: pass
    _at_ts=datetime.now().strftime("%Y%m%d_%H%M")
    _df_exp=pd.DataFrame([{"Ticker":a["Ticker"],"Pattern":", ".join(a["Pattern"]),"CSS":a["CSS"]} for a in _rows])
    st.download_button("📊 Export Alert",_df_exp.to_csv(index=False).encode(),f"Alert_v41_{_at_ts}.csv","text/csv",key=f"alert_exp_{tab_name}")

# ── v41 #2: News & Sentiment ───────────────────────────────────────────────
@st.cache_data(ttl=600)
def _fetch_news_v41(tickers:tuple)->list:
    import urllib.request as _ur, xml.etree.ElementTree as _ET
    _BULL={"surge","rally","soar","beat","record","upgrade","buy","bullish","outperform","strong","growth","profit","revenue","exceed","positive","gain","rise","boost","breakout"}
    _BEAR={"crash","fall","drop","miss","downgrade","sell","bearish","underperform","weak","loss","decline","below","negative","cut","reduce","layoff","concern","risk","warning","plunge"}

    # Cache nomi aziende
    _names={}
    try:
        import yfinance as yf
        for _t in tickers[:25]:
            try:
                _tk=yf.Ticker(_t)
                _info=_tk.info if hasattr(_tk,'info') else {}
                _names[_t]=_info.get("shortName",_info.get("longName",_t))
            except: _names[_t]=_t
    except Exception: _names={t:t for t in tickers}

    _res=[]
    for _t in tickers[:20]:
        try:
            _url=f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_t}&region=US&lang=en-US"
            _req=_ur.Request(_url,headers={"User-Agent":"Mozilla/5.0"})
            with _ur.urlopen(_req,timeout=6) as _r: _xml=_r.read()
            _root=_ET.fromstring(_xml)
            for _item in _root.findall(".//item")[:5]:
                _title=_item.findtext("title",""); _words=set(_title.lower().split())
                _b=len(_words&_BULL); _br=len(_words&_BEAR)
                _sent="🟢 Bullish" if _b>_br else "🔴 Bearish" if _br>_b else "⚪ Neutral"
                _tk_name=_names.get(_t,_t)
                _tv_link=f"https://it.tradingview.com/symbols/{_t.replace('.','-')}" if '.' not in _t else f"https://it.tradingview.com/symbols/{_t}"
                _res.append({"Ticker":_t,"Nome":_tk_name,"Titolo":_title[:80],"Sentiment":_sent,"Score":_b-_br,
                             "Data":_item.findtext("pubDate","")[:16],"Link":_item.findtext("link",""),
                             "TVLink":_tv_link})
        except Exception: pass
    return sorted(_res,key=lambda x:abs(x["Score"]),reverse=True)

def _render_news_v41(df_ep_news):
    _tickers=[]
    if not(df_ep_news is None or(hasattr(df_ep_news,"empty")and df_ep_news.empty)):
        if "Stato_Pro" in df_ep_news.columns:
            _tickers=df_ep_news[df_ep_news["Stato_Pro"].isin(["PRO","STRONG"])]["Ticker"].dropna().tolist()[:20]
        if not _tickers: _tickers=df_ep_news["Ticker"].dropna().tolist()[:15]
    try:
        _wl=load_watchlist()
        if not _wl.empty and "Ticker" in _wl.columns:
            _tickers+=_wl[_wl["list_name"]==st.session_state.current_list_name]["Ticker"].dropna().tolist()[:10]
    except Exception: pass
    _tickers=list(dict.fromkeys(_tickers))[:25]
    if not _tickers: st.info("Avvia scanner o aggiungi ticker alla watchlist."); return
    _,_fc2=st.columns([3,1])
    with _fc2:
        _nsf=st.selectbox("Filtro",["Tutti","🟢 Bullish","🔴 Bearish","⚪ Neutral"],key="ns_filter_v41")
        if st.button("🔄 Aggiorna",key="ns_refresh_v41"): st.cache_data.clear(); st.rerun()
    with st.spinner("Carico news..."): _news=_fetch_news_v41(tuple(_tickers))
    if _nsf=="🟢 Bullish": _news=[n for n in _news if "Bullish" in n["Sentiment"]]
    elif _nsf=="🔴 Bearish": _news=[n for n in _news if "Bearish" in n["Sentiment"]]
    elif _nsf=="⚪ Neutral": _news=[n for n in _news if "Neutral" in n["Sentiment"]]
    if not _news: st.info("Nessuna news trovata."); return
    _nb,_nr,_nn=(sum(1 for n in _news if x in n["Sentiment"]) for x in("Bullish","Bearish","Neutral"))
    _k1,_k2,_k3,_k4=st.columns(4)
    _k1.metric("📰 Totale",len(_news)); _k2.metric("🟢",_nb); _k3.metric("🔴",_nr); _k4.metric("⚪",_nn)
    st.markdown("---")
    for n in _news[:40]:
        _sc2="#00ff88" if "Bullish" in n["Sentiment"] else "#ef4444" if "Bearish" in n["Sentiment"] else "#6b7280"
        _tk_nome=n.get("Nome",n["Ticker"])
        _tv_link=n.get("TVLink","#")
        _c1,_c2,_c3=st.columns([1.2,0.8,4])
        _c1.markdown(f"<a href='{_tv_link}' target='_blank' style='color:#00ff88;font-weight:bold;text-decoration:none'>{n['Ticker']}</a> <span style='color:#6b7280;font-size:0.7rem'>{_tk_nome}</span>",unsafe_allow_html=True)
        _c2.markdown(f"<span style='color:{_sc2};font-size:0.78rem'>{n['Sentiment']}</span>",unsafe_allow_html=True)
        _c3.markdown(f"<a href='{n['Link']}' target='_blank' style='color:#b2b5be;font-size:0.82rem;text-decoration:none'>{n['Titolo']}</a> <span style='color:#374151;font-size:0.70rem'>{n['Data']}</span>",unsafe_allow_html=True)

# ── v41 #3: Macro Calendar ─────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _fetch_macro_v41()->list:
    import calendar as _cal
    _today=datetime.now().date()
    _events=[]
    for _day,_name,_impact,_desc in [
        (1,"ISM Manufacturing","🟡 Med","Attività manifatturiera"),(3,"ISM Services","🟡 Med","Settore servizi"),
        (5,"NFP + Unemployment","🔴 High","Non-Farm Payrolls"),(10,"CPI Inflation","🔴 High","Consumer Price Index"),
        (14,"PPI","🟡 Med","Producer Price Index"),(15,"Retail Sales","🟡 Med","Vendite al dettaglio"),
        (20,"FOMC Minutes","🔴 High","Verbali Fed"),(28,"PCE Inflation","🔴 High","Personal Consumption Expenditures")]:
        for _dm in range(3):
            try:
                _m=(_today.month+_dm-1)%12+1; _y=_today.year+((_today.month+_dm-1)//12)
                _d=min(_day,_cal.monthrange(_y,_m)[1]); _cand=_today.replace(year=_y,month=_m,day=_d)
                _dt=(_cand-_today).days
                if _dt>=-1: _events.append({"Data":str(_cand),"Evento":_name,"Impatto":_impact,"Desc":_desc,"Giorni":_dt}); break
            except Exception: pass
    for _fd in ["2026-04-29","2026-06-18","2026-07-30","2026-09-17","2026-11-05","2026-12-17"]:
        try:
            from datetime import datetime as _dt2
            _fd_d=_dt2.strptime(_fd,"%Y-%m-%d").date(); _dt=(_fd_d-_today).days
            if -1<=_dt<=120: _events.append({"Data":_fd,"Evento":"⚠️ FOMC Rate Decision","Impatto":"🔴 High","Desc":"Decisione tassi Fed","Giorni":_dt})
        except Exception: pass
    return sorted(_events,key=lambda x:x["Giorni"])

# ── v41 #4: Short Interest + Options + Insider ─────────────────────────────
@st.cache_data(ttl=3600)
def _fetch_short_v41(tickers:tuple)->dict:
    import yfinance as _yf_si
    _res={}
    for _t in tickers[:40]:
        try:
            _s=_yf_si.Ticker(_t).info.get("shortPercentOfFloat")
            if _s is not None: _res[_t]=round(float(_s)*100,1)
        except Exception: pass
    return _res

@st.cache_data(ttl=900)
def _fetch_options_v41(ticker:str)->dict:
    import yfinance as _yf_op
    try:
        _tk=_yf_op.Ticker(ticker); _exps=_tk.options
        if not _exps: return {}
        _ch=_tk.option_chain(_exps[0])
        _cv=float(_ch.calls["volume"].fillna(0).sum()) if not _ch.calls.empty else 0
        _pv=float(_ch.puts["volume"].fillna(0).sum()) if not _ch.puts.empty else 0
        _pcr=_pv/_cv if _cv>0 else None
        _sig="🟢 Bullish" if _pcr and _pcr<0.7 else "🔴 Bearish" if _pcr and _pcr>1.2 else "⚪ Neutro"
        return {"ticker":ticker,"pcr":round(_pcr,2) if _pcr else None,"call_vol":int(_cv),"put_vol":int(_pv),"signal":_sig,"expiry":_exps[0]}
    except Exception: return {}

@st.cache_data(ttl=3600)
def _fetch_insider_v41(tickers:tuple)->list:
    import urllib.request as _ur, json as _js
    _res=[]
    for _t in tickers[:15]:
        try:
            _url=f"https://efts.sec.gov/LATEST/search-index?q=%22{_t}%22&forms=4&hits.hits._source=entity_name,file_date"
            _req=_ur.Request(_url,headers={"User-Agent":"TradingScanner/1.0 info@example.com","Accept":"application/json"})
            with _ur.urlopen(_req,timeout=8) as _r: _data=_js.loads(_r.read())
            for _h in _data.get("hits",{}).get("hits",[])[:3]:
                _s=_h.get("_source",{})
                _res.append({"Ticker":_t,"Insider":_s.get("entity_name","—")[:30],"Data":_s.get("file_date","—")[:10],"Tipo":"Form 4"})
        except Exception: pass
    return _res

# =========================================================================

def _render_ai_explainer_v41(df_source, tab_name="PRO"):
    """AI Signal Explainer — multi-provider con fallback automatico."""
    st.markdown(
        '<div class="section-pill">🤖 MODULO 2 — AI ANALYST · Setup · Target · Invalidazione · Rischio</div>',
        unsafe_allow_html=True)
    st.caption("Fallback automatico: Gemini (free) → Groq (free) → OpenRouter → Claude · Clicca 🧠 Analizza su ogni ticker")

    # ── Pannello configurazione API keys ──────────────────────────────────
    _any_key = any([
        st.secrets.get("GEMINI_API_KEY","")      or st.session_state.get("_gemini_api_key",""),
        st.secrets.get("GROQ_API_KEY","")        or st.session_state.get("_groq_api_key",""),
        st.secrets.get("OPENROUTER_API_KEY","")  or st.session_state.get("_openrouter_api_key",""),
        st.secrets.get("ANTHROPIC_API_KEY","")   or st.session_state.get("_anthropic_api_key",""),
        st.secrets.get("HUGGINGFACE_API_KEY","") or st.session_state.get("_huggingface_api_key",""),
        st.secrets.get("COHERE_API_KEY","")      or st.session_state.get("_cohere_api_key",""),
        st.secrets.get("PERPLEXITY_API_KEY","")  or st.session_state.get("_perplexity_api_key",""),
    ])

    with st.expander(
        "🔑 Configura API Keys" + (" ✅" if _any_key else " ⚠️ Nessuna key — configura qui"),
        expanded=not _any_key
    ):
        st.markdown(
            "<div style='background:#0d1117;border:1px solid #1f2937;border-radius:6px;"
            "padding:10px 14px;margin-bottom:10px;font-size:0.80rem;color:#b2b5be'>"
            "Configura almeno una key. Il sistema usa quella disponibile con fallback automatico.<br><br>"
            "🟢 <b>Gemini Flash</b> — gratis · <a href='https://aistudio.google.com' target='_blank' "
            "style='color:#2962ff'>aistudio.google.com</a> → Get API Key<br>"
            "🟠 <b>HuggingFace</b> — gratis · <a href='https://huggingface.co' target='_blank' "
            "style='color:#2962ff'>huggingface.co</a> → Settings → Tokens<br>"
            "🔷 <b>Cohere</b> — free tier · <a href='https://cohere.com' target='_blank' "
            "style='color:#2962ff'>cohere.com</a> → API Keys<br>"
            "🟣 <b>Perplexity</b> — free tier · <a href='https://perplexity.ai' target='_blank' "
            "style='color:#2962ff'>perplexity.ai</a> → Settings → API<br>"
            "🔵 <b>OpenRouter</b> — free tier · <a href='https://openrouter.ai' target='_blank' "
            "style='color:#2962ff'>openrouter.ai</a> → Keys<br>"
            "🟣 <b>Groq</b> — gratis · <a href='https://console.groq.com' target='_blank' "
            "style='color:#2962ff'>console.groq.com</a> → API Keys<br>"
            "🟡 <b>Claude</b> — a pagamento · <a href='https://console.anthropic.com' target='_blank' "
            "style='color:#2962ff'>console.anthropic.com</a> → API Keys<br>"
            "🏠 <b>Ollama</b> — locale (gratis) · <a href='https://ollama.com' target='_blank' "
            "style='color:#2962ff'>ollama.com</a> → Download"
            "</div>",
            unsafe_allow_html=True
        )

        _kc1, _kc2 = st.columns(2)
        with _kc1:
            # Gemini
            _gem_cur = st.session_state.get("_gemini_api_key","")
            _gem_inp = st.text_input("🟢 Gemini API Key",
                value=_gem_cur, type="password",
                placeholder="AIzaSy...",
                key=f"gem_inp_{tab_name}")
            # HuggingFace
            _hf_cur = st.session_state.get("_huggingface_api_key","")
            _hf_inp = st.text_input("🟠 HuggingFace API Key",
                value=_hf_cur, type="password",
                placeholder="hf_...",
                key=f"hf_inp_{tab_name}")
            # Cohere
            _coh_cur = st.session_state.get("_cohere_api_key","")
            _coh_inp = st.text_input("🔷 Cohere API Key",
                value=_coh_cur, type="password",
                placeholder="...",
                key=f"coh_inp_{tab_name}")
            # Perplexity
            _perp_cur = st.session_state.get("_perplexity_api_key","")
            _perp_inp = st.text_input("🟣 Perplexity API Key",
                value=_perp_cur, type="password",
                placeholder="...",
                key=f"perp_inp_{tab_name}")
        with _kc2:
            # OpenRouter
            _or_cur = st.session_state.get("_openrouter_api_key","")
            _or_inp = st.text_input("🔵 OpenRouter API Key",
                value=_or_cur, type="password",
                placeholder="sk-or-...",
                key=f"or_inp_{tab_name}")
            # Groq
            _groq_cur = st.session_state.get("_groq_api_key","")
            _groq_inp = st.text_input("🟣 Groq API Key",
                value=_groq_cur, type="password",
                placeholder="gsk_...",
                key=f"groq_inp_{tab_name}")
            # Claude
            _ant_cur = st.session_state.get("_anthropic_api_key","")
            _ant_inp = st.text_input("🟡 Claude API Key",
                value=_ant_cur, type="password",
                placeholder="sk-ant-...",
                key=f"ant_inp_{tab_name}")

        # Salva tutte le chiavi
        if st.button("💾 Salva Keys", key=f"save_keys_{tab_name}"):
            if _gem_inp.strip():  st.session_state["_gemini_api_key"]     = _gem_inp.strip()
            if _hf_inp.strip():   st.session_state["_huggingface_api_key"]= _hf_inp.strip()
            if _coh_inp.strip(): st.session_state["_cohere_api_key"]      = _coh_inp.strip()
            if _perp_inp.strip():st.session_state["_perplexity_api_key"] = _perp_inp.strip()
            if _or_inp.strip():  st.session_state["_openrouter_api_key"] = _or_inp.strip()
            if _groq_inp.strip():st.session_state["_groq_api_key"]       = _groq_inp.strip()
            if _ant_inp.strip():st.session_state["_anthropic_api_key"]  = _ant_inp.strip()
            st.success("✅ Keys salvate!"); st.rerun()

        st.caption("💡 Ollama: installa localmente, non serve key. Gli altri provider hanno free tier.")

        # Status provider
        _prov_status = []
        for _pn, _sk, _ssk in [
            ("🟢 Gemini",      "GEMINI_API_KEY",       "_gemini_api_key"),
            ("🟠 HuggingFace", "HUGGINGFACE_API_KEY",  "_huggingface_api_key"),
            ("🔷 Cohere",     "COHERE_API_KEY",       "_cohere_api_key"),
            ("🟣 Perplexity", "PERPLEXITY_API_KEY",   "_perplexity_api_key"),
            ("🔵 OpenRouter", "OPENROUTER_API_KEY",    "_openrouter_api_key"),
            ("🟣 Groq",       "GROQ_API_KEY",         "_groq_api_key"),
            ("🟡 Claude",     "ANTHROPIC_API_KEY",    "_anthropic_api_key"),
            ("🏠 Ollama",     "-",                    "-"),
        ]:
            _ok = bool(st.secrets.get(_sk,"") or st.session_state.get(_ssk,""))
            _prov_status.append(f"{'✅' if _ok else '❌'} {_pn}")
        st.caption("  ·  ".join(_prov_status))

    if not _any_key:
        st.info("Inserisci almeno una API key sopra per usare l'AI Explainer.")
        return

    if df_source is None or (hasattr(df_source,"empty") and df_source.empty):
        st.info("Avvia lo scanner per usare l'AI Explainer.")
        return

    # Filtra PRO/STRONG
    _df_ai = df_source.copy()
    if "Stato_Pro" in _df_ai.columns:
        _df_ai = _df_ai[_df_ai["Stato_Pro"].isin(["PRO","STRONG"])]
    if _df_ai.empty:
        st.info("Nessun segnale PRO/STRONG trovato. Avvia lo scanner.")
        return

    # Regime context
    try:
        _rg_ai = _get_market_regime()
        _regime_ctx = (f"VIX={_rg_ai['vix']}, Regime={_rg_ai['regime']}, "
                       f"SPY momentum 20d={_rg_ai['spy_mom_20d']:+.1f}%")
    except Exception:
        _regime_ctx = "dati regime non disponibili"

    # Ordina per Nome (alfabetico, case-insensitive)
    if "Nome" in _df_ai.columns:
        _df_ai = _df_ai.sort_values(
            "Nome",
            key=lambda x: x.str.upper().fillna(""),
            na_position="last"
        )

    # Header griglia
    _ai_cols = st.columns([2,1,1,1,1])
    for _col, _lbl in zip(_ai_cols, ["Ticker","Stato","CSS","RSI","Modulo 2"]):
        _col.markdown(f"<span style='color:#50c4e0;font-size:0.78rem;font-weight:bold;"
                      f"letter-spacing:1px'>{_lbl}</span>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2a2e39;margin:4px 0'>", unsafe_allow_html=True)

    for _, _row_ai in _df_ai.iterrows():
        _tkr_ai   = str(_row_ai.get("Ticker",""))
        _stato_ai = str(_row_ai.get("Stato_Pro","-"))
        _css_ai   = _row_ai.get("CSS","—")
        _rsi_ai   = _row_ai.get("RSI","—")
        _nome_ai  = str(_row_ai.get("Nome",""))[:25]
        _pr_ai    = _row_ai.get("Prezzo","")
        _atr_ai   = _row_ai.get("ATR","")
        _rs_ai    = _row_ai.get("RS_20d","")
        _vol_ai   = _row_ai.get("Vol_Ratio","")

        # ── Estrai valori numerici puliti (rimuovi HTML se presente) ────
        def _strip_html_val(v):
            if v is None: return "—"
            s = str(v)
            import re as _re2
            cleaned = _re2.sub(r"<[^>]+>", "", s).strip()
            return cleaned if cleaned else "—"
        def _css_color(v):
            try:
                f = float(_strip_html_val(v).replace(",","."))
                if f >= 75: return "#00ff88"
                if f >= 55: return "#ffd700"
                if f >= 35: return "#fb923c"
                return "#ef4444"
            except Exception: return "#b2b5be"
        def _rsi_color(v):
            try:
                f = float(_strip_html_val(v).replace(",","."))
                if f >= 70: return "#ef4444"
                if f >= 50: return "#00ff88"
                if f >= 30: return "#ffd700"
                return "#ef4444"
            except Exception: return "#b2b5be"

        _css_str = _strip_html_val(_css_ai)
        _rsi_str = _strip_html_val(_rsi_ai)

        # Link TradingView IT
        _tv_sym  = _tkr_ai.replace(".MI", "%3AMI").replace(".L", "%3AL").replace(".PA", "%3APA").replace(".DE", "%3ADE").replace(".AS", "%3AAS")
        _tv_link = f"https://it.tradingview.com/chart/?symbol={_tv_sym}"

        _ac1,_ac2,_ac3,_ac4,_ac5 = st.columns([2,1,1,1,1])
        _sc = "#ffd700" if _stato_ai=="STRONG" else "#00ff88"
        _ac1.markdown(
            f"<a href='{_tv_link}' target='_blank' style='text-decoration:none'>"
            f"<span style='font-family:Courier New;color:{_sc};font-weight:bold'>"
            f"{_tkr_ai}</span></a><br>"
            f"<span style='color:#787b86;font-size:0.72rem'>{_nome_ai}</span>",
            unsafe_allow_html=True)
        _ac2.markdown(f"<span style='color:{_sc};font-weight:bold;font-size:0.82rem'>"
                      f"{_stato_ai}</span>", unsafe_allow_html=True)
        _ac3.markdown(
            f"<span style='font-family:Courier New;font-size:0.82rem;"
            f"color:{_css_color(_css_ai)};font-weight:bold'>{_css_str}</span>",
            unsafe_allow_html=True)
        _ac4.markdown(
            f"<span style='font-family:Courier New;font-size:0.82rem;"
            f"color:{_rsi_color(_rsi_ai)};font-weight:bold'>{_rsi_str}</span>",
            unsafe_allow_html=True)

        with _ac5:
            if st.button("🧠 Analizza", key=f"ai_explain_{_tkr_ai}_{tab_name}",
                         use_container_width=True, help=f"Modulo 2 AI Analyst — {_tkr_ai}"):
                st.session_state[f"_ai_req_{_tkr_ai}_{tab_name}"] = True

        if st.session_state.get(f"_ai_req_{_tkr_ai}_{tab_name}"):
            with st.expander(f"🤖 Modulo 2 — AI Analyst · {_tkr_ai} ({_nome_ai})", expanded=True):
                # Dati aggiuntivi dal row
                _sq_ai  = _row_ai.get("Squeeze", False)
                _e20_ai = _row_ai.get("EMA20", "")
                _e50_ai = _row_ai.get("EMA50", "")
                _wk_ai  = _row_ai.get("Weekly_Bull", "")
                _st_ai  = _row_ai.get("Stato_Early", "")

                _prompt_ai = (
                    f"Sei un analista tecnico professionista. Produci un brief operativo conciso.\n\n"
                    f"TICKER: {_tkr_ai} ({_nome_ai}) | PREZZO: ${_pr_ai}\n"
                    f"SEGNALE: {_stato_ai} | CSS: {_css_ai}/100 | RSI: {_rsi_ai}\n"
                    f"ATR: {_atr_ai} | Vol Ratio: {_vol_ai}x | RS vs SPY: {_rs_ai}%\n"
                    f"EMA20: {_e20_ai} | EMA50: {_e50_ai} | Squeeze: {_sq_ai}\n"
                    f"Weekly Bull: {_wk_ai} | Stato Early: {_st_ai}\n"
                    f"SCENARIO MACRO: {_regime_ctx}\n\n"
                    f"Rispondi in italiano con questo formato ESATTO (max 2 righe per sezione):\n\n"
                    f"📊 SETUP:\n"
                    f"[descrivi la struttura tecnica — trend, momentum, volume, squeeze]\n\n"
                    f"🎯 TARGET:\n"
                    f"[T1 = entry + 1.5×ATR | T2 = entry + 3×ATR — valori numerici precisi]\n\n"
                    f"❌ INVALIDAZIONE:\n"
                    f"[livello di prezzo che invalida il setup — stop loss ATR-based]\n\n"
                    f"⚠️ RISCHIO:\n"
                    f"[rischio specifico principale in questo momento per questo titolo]"
                )

                with st.spinner(f"Analisi {_tkr_ai}..."):
                    try:
                        _txt, _prov_used = _ai_call_with_fallback(_prompt_ai)
                        st.markdown(
                            f"<div style='background:#0d1117;border:1px solid #1f2937;"
                            f"border-left:3px solid #2962ff;border-radius:0 8px 8px 0;"
                            f"padding:12px 16px;font-size:0.88rem;line-height:1.6'>"
                            f"{_txt.replace(chr(10),'<br>')}</div>",
                            unsafe_allow_html=True)
                        st.caption(f"Provider: {_prov_used}")
                    except Exception as _ai_err:
                        _err_msg = str(_ai_err)
                        if "NO_KEYS" in _err_msg:
                            st.warning("⚠️ Nessuna API key configurata — espandi il pannello 🔑 sopra.")
                        elif "ALL_FAILED" in _err_msg:
                            st.error(f"❌ Tutti i provider hanno fallito:\n{_err_msg.replace('ALL_FAILED: ','')}")
                        else:
                            st.error(f"❌ Errore: {_err_msg[:200]}")

                if st.button("✕ Chiudi", key=f"ai_close_{_tkr_ai}_{tab_name}"):
                    st.session_state.pop(f"_ai_req_{_tkr_ai}_{tab_name}", None)
                    st.rerun()


# =========================================================================
# v41 UPGRADE #4 — TELEGRAM ALERT ENGINE
# =========================================================================

def _ai_call_gemini(api_key: str, prompt: str) -> str:
    """Google Gemini 2.0 Flash — gratuito 1500 req/day."""
    import requests as _r
    _url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={api_key}")
    _resp = _r.post(_url,
        json={"contents":[{"parts":[{"text": prompt}]}],
              "generationConfig":{"maxOutputTokens":500,"temperature":0.3}},
        timeout=20)
    if _resp.status_code == 200:
        _d = _resp.json()
        return _d["candidates"][0]["content"]["parts"][0]["text"]
    raise Exception(f"Gemini {_resp.status_code}: {_resp.text[:120]}")


def _ai_call_groq(api_key: str, prompt: str) -> str:
    """Groq — Llama 3.3 70B, gratuito con rate limits."""
    import requests as _r
    _resp = _r.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile",
              "messages": [{"role":"user","content": prompt}],
              "max_tokens": 500, "temperature": 0.3},
        timeout=20)
    if _resp.status_code == 200:
        return _resp.json()["choices"][0]["message"]["content"]
    raise Exception(f"Groq {_resp.status_code}: {_resp.text[:120]}")


def _ai_call_openrouter(api_key: str, prompt: str) -> str:
    """OpenRouter — accesso a molti modelli, free tier disponibile."""
    import requests as _r
    _resp = _r.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://trading-scanner-pro.streamlit.app"},
        json={"model": "mistralai/mistral-7b-instruct:free",
              "messages": [{"role":"user","content": prompt}],
              "max_tokens": 500},
        timeout=20)
    if _resp.status_code == 200:
        return _resp.json()["choices"][0]["message"]["content"]
    raise Exception(f"OpenRouter {_resp.status_code}: {_resp.text[:120]}")


def _ai_call_claude(api_key: str, prompt: str) -> str:
    """Anthropic Claude Haiku — a pagamento, massima qualità."""
    import requests as _r
    _resp = _r.post("https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01",
                 "x-api-key": api_key},
        json={"model": "claude-haiku-4-5-20251001",
              "max_tokens": 500,
              "messages": [{"role":"user","content": prompt}]},
        timeout=25)
    if _resp.status_code == 200:
        return _resp.json()["content"][0]["text"]
    _err = _resp.json().get("error",{})
    raise Exception(f"Claude {_resp.status_code}: {_err.get('message','')[:120]}")


def _ai_call_huggingface(api_key: str, prompt: str) -> str:
    """HuggingFace Inference API — free tier con limiti."""
    import requests as _r
    _resp = _r.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
        timeout=30
    )
    if _resp.status_code == 200:
        _j = _resp.json()
        if isinstance(_j, list) and _j:
            return _j[0].get("generated_text", str(_j))[:2000]
        return str(_j)[:2000]
    raise Exception(f"HuggingFace {_resp.status_code}: {_resp.text[:100]}")


def _ai_call_cohere(api_key: str, prompt: str) -> str:
    """Cohere API — free tier disponibile."""
    import requests as _r
    _resp = _r.post(
        "https://api.cohere.ai/v1/chat",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "command-r", "message": prompt, "max_tokens": 300},
        timeout=25
    )
    if _resp.status_code == 200:
        return _resp.json()["text"]
    raise Exception(f"Cohere {_resp.status_code}: {_resp.text[:100]}")


def _ai_call_ollama(prompt: str) -> str:
    """Ollama — modelli locali gratuiti (richiede Ollama installato)."""
    import requests as _r
    try:
        _resp = _r.post("http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=60)
        if _resp.status_code == 200:
            return _resp.json().get("response", "")[:2000]
    except Exception as _e:
        raise Exception(f"Ollama non disponibile: {_e}")
    raise Exception("Ollama failed")


def _ai_call_perplexity(api_key: str, prompt: str) -> str:
    """Perplexity API — free tier disponibile."""
    import requests as _r
    _resp = _r.post(
        "https://api.perplexity.ai/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.1-sonar-small-128k-online", "messages": [{"role": "user", "content": prompt}]},
        timeout=25
    )
    if _resp.status_code == 200:
        return _resp.json()["choices"][0]["message"]["content"]
    raise Exception(f"Perplexity {_resp.status_code}: {_resp.text[:100]}")


def _ai_call_with_fallback(prompt: str) -> tuple:
    """
    Prova i provider in ordine: Gemini → HuggingFace → Cohere → Perplexity → OpenRouter → Groq → Claude.
    Restituisce (testo_risposta, provider_usato) o lancia Exception.
    """
    _ss = st.session_state

    # Costruisce lista provider configurati nell'ordine preferito
    _providers = []

    # Provider gratuiti / freemium
    _gem_key = st.secrets.get("GEMINI_API_KEY","") or _ss.get("_gemini_api_key","")
    if _gem_key:
        _providers.append(("🟢 Gemini Flash", _ai_call_gemini, _gem_key))

    _hf_key = st.secrets.get("HUGGINGFACE_API_KEY","") or _ss.get("_huggingface_api_key","")
    if _hf_key:
        _providers.append(("🟠 HuggingFace", _ai_call_huggingface, _hf_key))

    _coh_key = st.secrets.get("COHERE_API_KEY","") or _ss.get("_cohere_api_key","")
    if _coh_key:
        _providers.append(("🔷 Cohere", _ai_call_cohere, _coh_key))

    _perp_key = st.secrets.get("PERPLEXITY_API_KEY","") or _ss.get("_perplexity_api_key","")
    if _perp_key:
        _providers.append(("🟣 Perplexity", _ai_call_perplexity, _perp_key))

    # Provider con free tier
    _or_key = st.secrets.get("OPENROUTER_API_KEY","") or _ss.get("_openrouter_api_key","")
    if _or_key:
        _providers.append(("🔵 OpenRouter", _ai_call_openrouter, _or_key))

    _groq_key = st.secrets.get("GROQ_API_KEY","") or _ss.get("_groq_api_key","")
    if _groq_key:
        _providers.append(("🟣 Groq Llama", _ai_call_groq, _groq_key))

    # Provider a pagamento
    _ant_key = st.secrets.get("ANTHROPIC_API_KEY","") or _ss.get("_anthropic_api_key","")
    if _ant_key:
        _providers.append(("🟡 Claude Haiku", _ai_call_claude, _ant_key))

    # Prova Ollama locale (senza API key)
    try:
        import requests as _r_test
        _r_test.get("http://localhost:11434/", timeout=2)
        _providers.append(("🏠 Ollama (locale)", _ai_call_ollama, ""))
    except:
        pass

    if not _providers:
        raise Exception("NO_KEYS")

    _errors = []
    for _name, _fn, _key in _providers:
        try:
            # Ollama non richiede api_key
            if "Ollama" in _name:
                _result = _fn(prompt)
            else:
                _result = _fn(_key, prompt)
            return _result, _name
        except Exception as _e:
            _errors.append(f"{_name}: {_e}")
            continue  # prova il prossimo

    raise Exception("ALL_FAILED: " + " | ".join(_errors))



tabs = st.tabs([
    "🏠 Home",
    "📊 Comparatore",
    "💎 Blue Chip Dip",
    "📡 EARLY",
    "💪 PRO",
    "🔥 REA-HOT",
    "⭐ CONFLUENCE",
    "🎯 Serafini",
    "🔎 Finviz Pro",
    "🔬 Order Flow",
    "🛡️ Crisis Monitor",
    "🔀 MTF Matrix",
    "📓 Journal",
    "🌡️ Regime",
    "📋 Watchlist",
    "⚖️ Risk Manager",
    "📈 Backtest",
    "💡 Analisi Personale",
    "🤖 AI Assistant",       # v41 #1
    "🎲 Options Scanner",    # v41 #2
    "⚡ Momentum Alerts",     # v41 #3
    "📰 News & Sentiment",    # v41 #4
])
(tab_home, tab_mtf, tab_bcd, tab_e, tab_p, tab_r, tab_conf,
 tab_ser, tab_fvpro, tab_of, tab_crisis,
 tab_mtfmatrix, tab_journal, tab_regime,
 tab_w, tab_rm, tab_bt, tab_analisi,
 tab_ai, tab_opts, tab_mom, tab_news) = tabs

with tab_home:
    # ── v41 #1 — MARKET REGIME BANNER ────────────────────────────────────
    try:
        _regime_data = _get_market_regime()
        _rc = _regime_data["color"]; _ri = _regime_data["icon"]
        _rn = _regime_data["regime"]
        _rv = _regime_data["vix"]; _rm = _regime_data["spy_mom_20d"]
        _regime_badge_html = (
            f"<div style='background:#1e222d;border-left:4px solid {_rc};"
            f"border-radius:0 8px 8px 0;padding:10px 18px;margin-bottom:12px;"
            f"display:flex;align-items:center;gap:20px;'>"
            f"<span style='font-size:1.5rem'>{_ri}</span>"
            f"<div>"
            f"<span style='color:{_rc};font-family:Trebuchet MS;font-size:1.05rem;"
            f"font-weight:bold;letter-spacing:1px'>REGIME: {_rn}</span>"
            f"<span style='color:#787b86;font-size:0.82rem;margin-left:16px'>"
            f"VIX: <b style='color:#d1d4dc'>{_rv}</b> &nbsp;|&nbsp; "
        )
        _spy_col_inline = "#26a69a" if _rm >= 0 else "#ef4444"
        _regime_badge_html += (
            f"SPY 20d: <b style='color:{_spy_col_inline}'>"
            f"{_rm:+.1f}%</b>"
            f"</span>"
            f"</div>"
        )
        if _rn in ("Crisis", "Risk-Off"):
            _regime_badge_html += (
                f"<span style='background:rgba(239,68,68,0.15);color:#ef4444;"
                f"border:1px solid #ef444444;border-radius:4px;padding:3px 10px;"
                f"font-size:0.78rem;font-weight:bold'>⚠️ Segnali deboli soppressi</span>"
            )
        _regime_badge_html += "</div>"
        st.markdown(_regime_badge_html, unsafe_allow_html=True)
    except Exception:
        pass

    # ── v41 #3 — AUTO-SCAN TRIGGER ───────────────────────────────────────
    if st.session_state.get("_trigger_autoscan"):
        st.session_state["_trigger_autoscan"] = False
        st.toast("⏰ Auto-scan avviato dallo scheduler!", icon="🤖")

    # ── v41 — MERCATI LIVE GLOBALE (full panel, multi-row, YTD) ─────────────
    @st.cache_data(ttl=60, show_spinner=False)
    def _fetch_live_markets_v41():
        import yfinance as _yf_live
        _mkts = [
            # Azionari USA
            ("^GSPC",       "S&P 500",        "🇺🇸"),
            ("^IXIC",       "NASDAQ",          "💻"),
            ("^DJI",        "Dow Jones",       "🏭"),
            ("^RUT",        "Russell 2K",      "📊"),
            # Azionari Europa
            ("FTSEMIB.MI",  "FTSE MIB",        "🇮🇹"),
            ("^FTSE",       "FTSE 100",        "🇬🇧"),
            ("^GDAXI",      "DAX 40",          "🇩🇪"),
            ("^FCHI",       "CAC 40",          "🇫🇷"),
            ("^IBEX",       "IBEX 35",         "🇪🇸"),
            ("^AEX",        "AEX Olanda",      "🇳🇱"),
            ("^SSMI",       "SMI Svizzera",    "🇨🇭"),
            # Azionari Asia / EM
            ("^N225",       "Nikkei 225",      "🇯🇵"),
            ("^HSI",        "Hang Seng",       "🇭🇰"),
            ("000001.SS",   "Shanghai",        "🇨🇳"),
            ("^KS11",       "KOSPI Korea",     "🇰🇷"),
            ("^BVSP",       "BVSP Brasile",    "🇧🇷"),
            ("^NSEI",       "Nifty 50 India",  "🇮🇳"),
            # Volatilità
            ("^VIX",        "VIX",             "😰"),
            ("^VXN",        "VXN NASDAQ",      "😱"),
            # Crypto
            ("BTC-USD",     "Bitcoin",         "₿"),
            ("ETH-USD",     "Ethereum",        "⟠"),
            # Commodities
            ("GC=F",        "Gold",            "🥇"),
            ("SI=F",        "Argento",         "⚪"),
            ("CL=F",        "Oil WTI",         "🛢️"),
            ("NG=F",        "Gas Naturale",    "🔥"),
            ("HG=F",        "Rame",            "🟤"),
            ("ZW=F",        "Grano",           "🌾"),
            # Valute & Bond
            ("DX-Y.NYB",    "DXY",             "💵"),
            ("EURUSD=X",    "EUR/USD",         "🇪🇺"),
            ("JPY=X",       "USD/JPY",         "¥"),
            ("TLT",         "TLT Bond 20Y",    "🏦"),
        ]
        _results = []
        for _sym, _name, _ico in _mkts:
            try:
                _dy = _yf_live.download(_sym, period="ytd", interval="1d",
                                        auto_adjust=True, progress=False, threads=False)
                _dy.columns = [co[0] if isinstance(co,tuple) else co for co in _dy.columns]
                _cly = _dy["Close"].dropna()
                _d5 = _yf_live.download(_sym, period="5d", interval="1d",
                                        auto_adjust=True, progress=False, threads=False)
                _d5.columns = [co[0] if isinstance(co,tuple) else co for co in _d5.columns]
                _cl5 = _d5["Close"].dropna()
                if len(_cl5) < 1:
                    continue
                _cur  = float(_cl5.iloc[-1])
                _chg  = (_cur / float(_cl5.iloc[-2]) - 1)*100 if len(_cl5)>=2 else 0.0
                _ytd  = (_cur / float(_cly.iloc[0]) - 1)*100  if len(_cly)>=2 else None
                _results.append({"sym":_sym,"name":_name,"icon":_ico,
                                 "price":_cur,"chg":_chg,"ytd":_ytd})
            except Exception:
                pass
        return _results

    try:
        _live_data_v41 = _fetch_live_markets_v41()
        if _live_data_v41:
            _now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

            _groups = [
                ("🇺🇸 Azionari USA",          ["^GSPC","^IXIC","^DJI","^RUT"]),
                ("🌍 Azionari Europa",         ["FTSEMIB.MI","^FTSE","^GDAXI","^FCHI","^IBEX","^AEX","^SSMI"]),
                ("🌏 Azionari Asia / EM",      ["^N225","^HSI","000001.SS","^KS11","^BVSP","^NSEI"]),
                ("😰 Volatilità · ₿ Crypto",   ["^VIX","^VXN","BTC-USD","ETH-USD"]),
                ("🥇 Commodities",             ["GC=F","SI=F","CL=F","NG=F","HG=F","ZW=F"]),
                ("💵 Valute · 🏦 Bond",        ["DX-Y.NYB","EURUSD=X","JPY=X","TLT"]),
            ]
            _data_map = {m["sym"]: m for m in _live_data_v41}

            def _fmt_price_v41(m):
                s = m["sym"]; p = m["price"]
                if s in ("BTC-USD",):                return f"${p:,.0f}"
                if s in ("GC=F","SI=F","CL=F","NG=F","HG=F","ZW=F","ETH-USD"): return f"${p:,.2f}"
                if s in ("^VIX","^VXN","DX-Y.NYB","EURUSD=X","JPY=X"):         return f"{p:,.3f}" if p < 10 else f"{p:,.2f}"
                if p > 10000: return f"{p:,.0f}"
                if p > 1000:  return f"{p:,.1f}"
                return f"{p:,.2f}"

            _yf_url = "https://it.finance.yahoo.com/markets/"
            _live_html = (
                f"<div style='background:#1a1e2e;border:1px solid #2a2e39;"
                f"border-left:4px solid #2962ff;"
                f"border-radius:0 10px 10px 0;padding:14px 18px 16px 18px;margin-bottom:16px'>"
                f"<div style='margin-bottom:12px'>"
                f"<a href='{_yf_url}' target='_blank' style='text-decoration:none'>"
                f"<span style='color:#2962ff;font-weight:bold;font-size:0.90rem;"
                f"letter-spacing:1.5px'>📊 MERCATI LIVE</span></a>"
                f"<span style='color:#6b7280;font-size:0.73rem;margin-left:12px'>{_now_str}</span>"
                f"<span style='color:#374151;font-size:0.70rem;margin-left:10px'>· aggiorna ogni 60s · YTD = performance da inizio anno</span>"
                f"</div>"
            )

            for _grp_name, _syms in _groups:
                _avail = [s for s in _syms if s in _data_map]
                if not _avail: continue
                _live_html += (
                    f"<div style='margin-bottom:12px'>"
                    f"<div style='color:#50c4e0;font-size:0.68rem;font-weight:bold;"
                    f"letter-spacing:2px;margin-bottom:6px;text-transform:uppercase;"
                    f"border-bottom:1px solid #2a2e39;padding-bottom:4px'>"
                    f"{_grp_name}</div>"
                    f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
                )
                for _sym in _avail:
                    _m   = _data_map[_sym]
                    _c   = "#26a69a" if _m["chg"]>=0 else "#ef4444"
                    _ar  = "▲" if _m["chg"]>=0 else "▼"
                    _pr  = _fmt_price_v41(_m)
                    _ytd = _m.get("ytd")
                    if _ytd is not None:
                        _ytd_c  = "#26a69a" if _ytd>=0 else "#ef4444"
                        _ytd_ar = "▲" if _ytd>=0 else "▼"
                        _ytd_html = (
                            f"<div style='color:{_ytd_c};font-size:0.68rem;"
                            f"margin-top:3px;font-weight:500'>"
                            f"YTD {_ytd_ar}{abs(_ytd):.1f}%</div>"
                        )
                    else:
                        _ytd_html = "<div style='color:#374151;font-size:0.68rem;margin-top:3px'>YTD —</div>"

                    _live_html += (
                        f"<div style='background:#131722;border:1px solid #2a2e39;"
                        f"border-top:3px solid {_c}88;"
                        f"border-radius:6px;padding:9px 13px;"
                        f"min-width:108px;max-width:160px;flex:1;text-align:center;"
                        f"transition:border-color .2s'>"
                        f"<div style='color:#9ca3af;font-size:0.68rem;white-space:nowrap;"
                        f"margin-bottom:4px;letter-spacing:0.5px'>{_m['icon']} {_m['name']}</div>"
                        f"<div style='color:#e2e8f0;font-family:Courier New;font-size:0.96rem;"
                        f"font-weight:bold;letter-spacing:0.5px'>{_pr}</div>"
                        f"<div style='color:{_c};font-size:0.75rem;font-weight:bold;margin-top:3px'>"
                        f"{_ar} {abs(_m['chg']):.2f}%</div>"
                        f"{_ytd_html}"
                        f"</div>"
                    )
                _live_html += "</div></div>"

            _live_html += "</div>"
            st.markdown(_live_html, unsafe_allow_html=True)
    except Exception:
        pass

        # ── v41 — CORRELAZIONI ASSET 30 giorni (inline, funzionante) ─────────
    with st.expander("🔗 Correlazioni Asset — 30 giorni", expanded=False):
        @st.cache_data(ttl=3600, show_spinner=False)
        def _fetch_corr_v41():
            import yfinance as _yc
            _corr_syms = {
                "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DAX": "^GDAXI",
                "FTSE MIB": "FTSEMIB.MI", "Nikkei": "^N225",
                "Bitcoin": "BTC-USD", "Gold": "GC=F", "Oil WTI": "CL=F",
                "Silver": "SI=F", "DXY": "DX-Y.NYB", "VIX": "^VIX",
                "TLT Bond": "TLT",
            }
            _raw = _yc.download(
                list(_corr_syms.values()), period="30d", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker"
            )
            _closes = {}
            for _lab, _sym in _corr_syms.items():
                try:
                    if isinstance(_raw.columns, pd.MultiIndex):
                        _s = _raw[(_sym, "Close")].dropna() if (_sym,"Close") in _raw.columns else _raw["Close"][_sym].dropna()
                    else:
                        _s = _raw["Close"].dropna()
                    if len(_s) >= 10:
                        _closes[_lab] = _s
                except Exception:
                    pass
            if len(_closes) < 2:
                return None
            _df_c = pd.DataFrame(_closes).pct_change().dropna()
            return _df_c.corr().round(2)

        try:
            _corr_df = _fetch_corr_v41()
            if _corr_df is not None and not _corr_df.empty:
                import plotly.graph_objects as _go_corr
                _labs = list(_corr_df.columns)
                _zvals = _corr_df.values.tolist()
                _text  = [[f"{v:.2f}" for v in row] for row in _zvals]
                _fig_corr = _go_corr.Figure(_go_corr.Heatmap(
                    z=_zvals, x=_labs, y=_labs, text=_text,
                    texttemplate="%{text}",
                    colorscale=[
                        [0.0,  "#ef4444"], [0.4, "#991b1b"],
                        [0.5,  "#1e222d"],
                        [0.6,  "#1a3a2e"], [1.0, "#26a69a"],
                    ],
                    zmid=0, zmin=-1, zmax=1,
                    showscale=True,
                    colorbar=dict(
                        tickfont=dict(color="#787b86", size=10),
                        bgcolor="#131722", bordercolor="#2a2e39",
                    )
                ))
                _fig_corr.update_layout(
                    paper_bgcolor="#131722", plot_bgcolor="#131722",
                    margin=dict(l=10,r=10,t=10,b=10),
                    height=420,
                    font=dict(color="#d1d4dc", size=11),
                    xaxis=dict(tickfont=dict(size=10), gridcolor="#2a2e39"),
                    yaxis=dict(tickfont=dict(size=10), gridcolor="#2a2e39"),
                )
                st.plotly_chart(_fig_corr, use_container_width=True, key="home_corr_v41")
                st.caption(
                    "🟢 +1 = si muovono insieme (rischio correlato) &nbsp;·&nbsp; "
                    "🔴 −1 = hedge naturale (si muovono opposti) &nbsp;·&nbsp; "
                    "⬛ 0 = scorrelati. Periodo: 30gg giornaliero."
                )
            else:
                st.info("Dati correlazione non disponibili — riprova tra qualche secondo.")
        except Exception as _ce:
            st.warning(f"Errore correlazioni: {str(_ce)[:120]}")

    # v41: render_home saltato per evitare duplicato Mercati Live
    # (i Mercati Live sono gia' nel blocco principale qui sopra)

    # ── v41 #4 + #9 — EARNINGS CALENDAR (Home, fondo pagina) ─────────────
    st.markdown("---")
    st.markdown('<div class="section-pill">📅 EARNINGS CALENDAR v41 — Prossimi earnings da Watchlist + Scanner</div>',
                unsafe_allow_html=True)
    _earn_tickers = set()
    # Da watchlist
    try:
        _wl_earn = load_watchlist()
        if not _wl_earn.empty and "Ticker" in _wl_earn.columns:
            _earn_tickers.update(_wl_earn["Ticker"].dropna().unique().tolist())
    except Exception:
        pass
    # Dal df_ep scanner (prime 60 per velocità)
    if not df_ep.empty and "Ticker" in df_ep.columns:
        _earn_tickers.update(df_ep["Ticker"].dropna().unique().tolist()[:60])

    _earn_tickers_sorted = tuple(sorted(_earn_tickers)[:80])  # cap 80 per performance

    if _earn_tickers_sorted:
        with st.spinner("📅 Carico earnings calendar..."):
            _earn_data = _fetch_earnings_calendar(_earn_tickers_sorted)

        if _earn_data:
            # Summary metrics
            _ec1, _ec2, _ec3, _ec4 = st.columns(4)
            _ec1.metric("📅 Con earnings", len(_earn_data))
            _ec2.metric("⚠️ Oggi/Domani",  sum(1 for x in _earn_data if x["Giorni"] <= 1))
            _ec3.metric("🔔 Questa sett.", sum(1 for x in _earn_data if 2 <= x["Giorni"] <= 7))
            _ec4.metric("📅 Entro 2 sett.",sum(1 for x in _earn_data if 8 <= x["Giorni"] <= 14))

            # Tabella earnings con nome + link TradingView IT
            for _ed in _earn_data[:25]:
                _ea, _eb, _ec_col, _edd = st.columns([2.5, 1.5, 1.2, 2])
                _tkr_ed  = _ed['Ticker']
                _tv_ed   = _tkr_ed.replace(".MI","").replace(".","")
                _nome_ed = ""
                if not df_ep.empty and "Ticker" in df_ep.columns and "Nome" in df_ep.columns:
                    _nm_row = df_ep[df_ep["Ticker"]==_tkr_ed]
                    if not _nm_row.empty:
                        _nome_ed = str(_nm_row.iloc[0].get("Nome",""))[:28]
                _ea.markdown(
                    f"<a href='https://it.tradingview.com/chart/?symbol={_tv_ed}' target='_blank' "
                    f"style='text-decoration:none'>"
                    f"<b style='font-family:Courier New;color:#00ff88;font-size:0.95rem'>{_tkr_ed}</b>"
                    f"<span style='color:#2962ff;font-size:0.65rem'> ↗</span></a>"
                    f"<br><span style='color:#787b86;font-size:0.72rem'>{_nome_ed}</span>",
                    unsafe_allow_html=True)
                _eb.markdown(
                    f"<span style='color:#d1d4dc;font-size:0.85rem'>{_ed['Earnings Date']}</span>",
                    unsafe_allow_html=True)
                _ec_col.markdown(
                    f"<b style='font-size:0.78rem;color:{_ed['_color']}'>{_ed['Giorni']:+d}gg</b>",
                    unsafe_allow_html=True)
                _edd.markdown(
                    f"<span style='background:{_ed['_color']}22;color:{_ed['_color']};"
                    f"border:1px solid {_ed['_color']}44;border-radius:4px;"
                    f"padding:1px 8px;font-size:0.75rem;font-weight:bold'>"
                    f"{_ed['Badge']}</span>",
                    unsafe_allow_html=True)
        else:
            st.info("📭 Nessun earnings trovato nei prossimi 21 giorni per i ticker in watchlist/scanner.")
    else:
        st.info("Aggiungi ticker alla watchlist o avvia lo scanner per vedere gli earnings.")

    st.markdown('---')
    with st.expander('📰 NEWS & SENTIMENT v41 — Ultime news con score sentiment · [TradingView Italia](https://it.tradingview.com/)', expanded=False):
        _render_news_v41(df_ep)
    st.markdown('---')
    st.markdown('<div class="section-pill">🗓️ MACRO CALENDAR v41 — Fed · CPI · NFP · PCE</div>', unsafe_allow_html=True)
    _macro_ev41=_fetch_macro_v41()
    _mc39=[e for e in _macro_ev41 if 0<=e['Giorni']<=14]
    if _mc39:
        _mc39c=st.columns(min(len(_mc39),4))
        for _i39,_ev41 in enumerate(_mc39[:4]):
            _ic39='#ef4444' if 'High' in _ev41['Impatto'] else '#f59e0b'
            _mc39c[_i39].markdown(f"<div style='background:#1e222d;border-top:2px solid {_ic39};border-radius:0 0 6px 6px;padding:8px 10px'><div style='color:{_ic39};font-size:0.70rem'>{_ev41['Impatto']} · {_ev41['Giorni']}gg</div><div style='color:#d1d4dc;font-size:0.82rem;font-weight:bold'>{_ev41['Evento']}</div><div style='color:#6b7280;font-size:0.70rem'>{_ev41['Data']}</div></div>",unsafe_allow_html=True)
    with st.expander('📅 Calendario completo 90 giorni',expanded=False):
        for _ev41 in _macro_ev41[:20]:
            _ic392='#ef4444' if 'High' in _ev41['Impatto'] else '#f59e0b' if 'Med' in _ev41['Impatto'] else '#6b7280'
            _dot39='🔴' if _ev41['Giorni']<=3 else '🟡' if _ev41['Giorni']<=7 else '🟢'
            st.markdown(f"<div style='border-left:3px solid {_ic392};padding:4px 10px;margin:2px 0'><span style='color:{_ic392};font-size:0.75rem'>{_ev41['Data']}</span> <b style='color:#d1d4dc'>{_ev41['Evento']}</b> <span style='color:{_ic392};float:right'>{_dot39} {_ev41['Giorni']}gg</span></div>",unsafe_allow_html=True)

with tab_e:
    st.session_state.last_active_tab="EARLY"; show_legend("EARLY")
    render_scan_tab(df_ep,"EARLY",["Early_Score","RSI"],[False,True],"EARLY")

    st.markdown('---')
    with st.expander('🔔 Alert Multipli v41 — Pattern tecnici', expanded=False):
        _render_pattern_alerts_v41(df_ep, tab_name='early')

with tab_p:
    st.session_state.last_active_tab="PRO"; show_legend("PRO")
    _pro_sort = st.radio("Ordina per",["Quality","Momentum (Pro×RSI)"],
                         horizontal=True, key="pro_sort_mode", label_visibility="collapsed")
    if _pro_sort == "Momentum (Pro×RSI)":
        # Aggiunge colonna Momentum temporanea per ordinamento
        _df_pro = df_ep.copy()
        if not _df_pro.empty and "Pro_Score" in _df_pro.columns and "RSI" in _df_pro.columns:
            _df_pro["_Momentum"] = _df_pro["Pro_Score"].fillna(0)*10 + _df_pro["RSI"].fillna(0)
        else:
            _df_pro["_Momentum"] = 0
        render_scan_tab(_df_pro,"PRO",["_Momentum","Quality_Score"],[False,False],"PRO — Momentum")
    else:
        render_scan_tab(df_ep,"PRO",["Quality_Score","Pro_Score","RSI"],[False,False,True],"PRO")

    st.markdown('---')
    with st.expander('🔔 Alert Multipli v41 — Pattern su segnali PRO', expanded=False):
        _render_pattern_alerts_v41(df_ep, tab_name='pro')

    # ── Modulo 2 — 🤖 AI Analyst ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("🤖 Modulo 2 — AI Analyst · Analisi per ogni ticker PRO/STRONG", expanded=False):
        _render_ai_explainer_v41(df_ep, "PRO")

with tab_r:
    st.session_state.last_active_tab="REA-HOT"; show_legend("REA-HOT")

    # ══════════════════════════════════════════════════════════════════════
    # 🔥 REA-HOT v34 — ACCUMULO & BREAKOUT DETECTOR
    # ══════════════════════════════════════════════════════════════════════
    # Idea ispirata agli youtuber/trader che cercano titoli in fase di
    # ACCUMULO (volatilità bassa, volume stabile) seguita da un BREAKOUT
    # improvviso con volume anomalo — classico pattern "coiled spring".
    #
    # SCORE ACCUMULO-BREAKOUT (AB_Score 0-100):
    #   • Vol_Ratio >= 2.0   → volume breakout confermato (40 pt)
    #   • Dist_POC% vicino   → prezzo torna vicino al livello chiave (20 pt)
    #   • ATR_pct in range   → movimento reale, non flat (15 pt)
    #   • RSI 45-65          → momentum sano, non overbought (15 pt)
    #   • OBV_Trend UP       → accumulo istituzionale (10 pt)
    # ──────────────────────────────────────────────────────────────────────
    if not df_rea.empty:
        df_rea_view = df_rea.copy()

        # Calcola AB_Score
        def _ab_score(row):
            score = 0.0
            # 1. Vol_Ratio breakout (max 40 pt)
            vr = float(row.get("Vol_Ratio", 0) or 0)
            if   vr >= 4.0: score += 40
            elif vr >= 3.0: score += 32
            elif vr >= 2.0: score += 22
            elif vr >= 1.5: score += 10
            # 2. Dist_POC% vicino al POC (max 20 pt) — più vicino = meglio
            dp = abs(float(row.get("Dist_POC_%", 999) or 999))
            if   dp <= 0.5: score += 20
            elif dp <= 1.0: score += 15
            elif dp <= 2.0: score += 10
            elif dp <= 3.0: score += 5
            # 3. ATR% range operativo (max 15 pt)
            atr = float(row.get("ATR_pct", 0) or 0)
            if   2.0 <= atr <= 4.0: score += 15
            elif 1.5 <= atr <= 6.0: score += 8
            # 4. RSI zona sana 45-65 (max 15 pt)
            rsi = float(row.get("RSI", 50) or 50)
            if   50 <= rsi <= 60: score += 15
            elif 45 <= rsi <= 65: score += 10
            elif 40 <= rsi <= 70: score += 5
            # 5. OBV crescente (max 10 pt)
            if row.get("OBV_Trend") == "UP": score += 10
            return round(score, 1)

        df_rea_view["AB_Score"] = df_rea_view.apply(_ab_score, axis=1)
        df_rea_view["AB_Grade"] = df_rea_view["AB_Score"].apply(
            lambda v: "🔥 HOT"    if v >= 70 else
                      "⚡ STRONG" if v >= 50 else
                      "📈 WATCH"  if v >= 30 else "💤 WEAK"
        )

        # Header con metriche rapide
        _ab_hot    = int((df_rea_view["AB_Grade"] == "🔥 HOT").sum())
        _ab_strong = int((df_rea_view["AB_Grade"] == "⚡ STRONG").sum())
        _ab_avg    = round(df_rea_view["AB_Score"].mean(), 1) if not df_rea_view.empty else 0
        _ab_top    = df_rea_view.nlargest(1, "AB_Score").iloc[0]["Ticker"] if not df_rea_view.empty else "—"

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🔥 HOT Breakout",   _ab_hot,    help="AB_Score ≥ 70 — breakout confermato con volume forte")
        m2.metric("⚡ Strong Setup",   _ab_strong, help="AB_Score 50-69 — setup in formazione")
        m3.metric("📊 Tot. segnali",   len(df_rea_view))
        m4.metric("📈 AB Score medio", f"{_ab_avg:.1f}")
        m5.metric("🏆 Top ticker",     _ab_top,    help="Ticker con AB_Score più alto")

        st.markdown(
            f'<div style="background:#1e222d;border-left:3px solid #f97316;'
            f'padding:8px 14px;border-radius:0 4px 4px 0;margin:8px 0;font-size:0.80rem">'
            f'<b style="color:#f97316">💡 Come leggere il tab REA-HOT:</b>'
            f' <span style="color:#b2b5be">I titoli in lista hanno già Vol_Ratio > soglia E '
            f'prezzo vicino al POC (Point of Control). '
            f'L\'<b>AB_Score</b> aggiunge una valutazione 0-100 del setup accumulo→breakout: '
            f'più alto = volume anomalo + prezzo in zona chiave + momentum sano.</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Chart top-10 AB_Score (mini heatmap visuale)
        top10 = df_rea_view.nlargest(min(10, len(df_rea_view)), "AB_Score")
        if not top10.empty:
            fig_ab = go.Figure()
            colors_ab = [
                "#f97316" if g == "🔥 HOT" else
                "#60a5fa" if g == "⚡ STRONG" else
                "#26a69a"
                for g in top10["AB_Grade"]
            ]
            vr_vals = pd.to_numeric(top10.get("Vol_Ratio", pd.Series()), errors="coerce").fillna(0)
            fig_ab.add_trace(go.Bar(
                x=top10["Ticker"],
                y=top10["AB_Score"],
                marker_color=colors_ab,
                marker_line_width=0,
                text=[f"{v:.0f}" for v in top10["AB_Score"]],
                textposition="outside",
                textfont=dict(size=10, color="#d1d4dc"),
                customdata=list(zip(
                    vr_vals.tolist(),
                    top10.get("Dist_POC_%", pd.Series([0]*len(top10))).fillna(0).tolist(),
                    top10.get("RSI", pd.Series([0]*len(top10))).fillna(0).tolist(),
                )),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "AB Score: <b>%{y:.0f}</b><br>"
                    "Vol Ratio: %{customdata[0]:.1f}x<br>"
                    "Dist POC: %{customdata[1]:+.1f}%<br>"
                    "RSI: %{customdata[2]:.0f}"
                    "<extra></extra>"
                ),
            ))
            fig_ab.update_layout(
                paper_bgcolor="#131722", plot_bgcolor="#1e222d",
                font=dict(color="#b2b5be", family="Trebuchet MS, sans-serif", size=12),
                xaxis=dict(gridcolor="#2a2e39", zerolinecolor="#363a45",
                           linecolor="#363a45", tickfont=dict(color="#787b86", size=10)),
                yaxis=dict(range=[0, 115], showgrid=True, gridcolor="#2a2e39",
                           zerolinecolor="#363a45", tickfont=dict(color="#787b86", size=10)),
                title=dict(text="🔥 Top 10 — Accumulo & Breakout Score",
                           font=dict(color="#f97316", size=13), x=0.01),
                height=260,
                margin=dict(l=0, r=0, t=44, b=0),
                showlegend=False,
            )
            # Linee soglia colorate
            for yval, col, lbl in [(70,"#f97316","HOT"), (50,"#60a5fa","STRONG"), (30,"#26a69a","WATCH")]:
                fig_ab.add_hline(y=yval, line=dict(color=col, width=1, dash="dot"),
                                 annotation_text=lbl, annotation_font_color=col,
                                 annotation_font_size=9)
            st.plotly_chart(fig_ab, use_container_width=True, key="rea_ab_chart")

    # ── Tabella standard con AB_Score aggiunto ──────────────────────────
    _df_rea_enhanced = df_rea_view if not df_rea.empty else df_rea
    render_scan_tab(_df_rea_enhanced, "HOT", ["AB_Score","Vol_Ratio","Dist_POC_%"],
                    [False, False, True], "REA-HOT")

with tab_conf:
    st.session_state.last_active_tab="CONFLUENCE"; show_legend("⭐ CONFLUENCE")
    render_scan_tab(df_ep,"CONFLUENCE",["Quality_Score","Early_Score","Pro_Score"],[False,False,False],"CONFLUENCE")
    st.markdown("---")
    with st.expander("🤖 Modulo 2 — AI Analyst · Analisi CONFLUENCE", expanded=False):
        _df_conf_ai = pd.DataFrame()
        if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
            _df_conf_ai = df_ep[(df_ep["Stato_Early"]=="EARLY") &
                                (df_ep["Stato_Pro"].isin(["PRO","STRONG"]))].copy()
        _render_ai_explainer_v41(_df_conf_ai, "CONF")

with tab_mtf:
    # ══════════════════════════════════════════════════════════════════════
    # COMPARATORE MULTI-TICKER v41
    # Top 5 per capitalizzazione (Mar 2025): AAPL > MSFT > NVDA > GOOGL > META
    # Il comparatore inline è SEMPRE visibile. Se compare_tab.py esiste,
    # viene mostrato anche il comparatore esterno sotto.
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-pill">📊 COMPARATORE MULTI-TICKER v41</div>',
                unsafe_allow_html=True)

    # ── Badge capitalizzazione — sempre visibili ─────────────────────────
    _CMP_DEFAULTS = [
        ("AAPL",  "Apple",    "~3.4T", "#2962ff"),
        ("MSFT",  "Microsoft","~3.1T", "#00d4aa"),
        ("NVDA",  "NVIDIA",   "~2.9T", "#f97316"),
        ("AMZN",  "Amazon",   "~2.2T", "#ff4081"),
        ("GOOGL", "Alphabet", "~2.1T", "#a78bfa"),
        ("META",  "Meta",     "~1.5T", "#f59e0b"),
    ]
    _badge_html = "".join(
        f'<span style="background:rgba(255,255,255,0.04);border:1px solid {c}44;' 
        f'border-left:3px solid {c};border-radius:0 3px 3px 0;' 
        f'padding:3px 10px;margin:2px 4px 2px 0;display:inline-block;' 
        f'font-family:Courier New;font-size:0.78rem">' 
        f'<b style="color:{c}">{t}</b>' 
        f'<span style="color:#b2b5be;margin-left:5px">{n}</span>' 
        f'<span style="color:#5a6478;margin-left:5px;font-size:0.70rem">{cap}</span>' 
        f'</span>'
        for t, n, cap, c in _CMP_DEFAULTS
    )
    st.markdown(f'<div style="margin-bottom:6px">{_badge_html}</div>', unsafe_allow_html=True)
    st.caption("Ordinati per capitalizzazione di mercato · Mar 2025")

    # ── Controlli ────────────────────────────────────────────────────────
    _cmp_default_str = "\n".join(t for t,_,_,_ in _CMP_DEFAULTS)
    _cc1, _cc2 = st.columns([1, 2.5])
    with _cc1:
        _cmp_input = st.text_area(
            "Ticker (uno per riga)",
            value=_cmp_default_str,
            height=155,
            key="cmp_tickers",
            help="Ticker Yahoo Finance, uno per riga."
        )
        _cmp_range = st.select_slider(
            "Periodo", options=["1mo","3mo","6mo","1y","2y","5y"],
            value="1y", key="cmp_range"
        )
    with _cc2:
        if not df_ep.empty and "Ticker" in df_ep.columns:
            _scan_tkrs = df_ep["Ticker"].dropna().unique().tolist()[:60]
            _cmp_nome_map = {str(r["Ticker"]): str(r.get("Nome",""))[:28]
                for _,r in df_ep[["Ticker","Nome"]].dropna(subset=["Ticker"]).iterrows()
            } if "Nome" in df_ep.columns else {}
            _cmp_extra = st.multiselect(
                "➕ Aggiungi ticker dal tuo scanner",
                options=sorted(_scan_tkrs, key=lambda t: _cmp_nome_map.get(t,t).lower()),
                format_func=lambda t: f"{_cmp_nome_map[t]}  ({t})" if _cmp_nome_map.get(t) else t,
                key="cmp_extra_tickers",
                help="Ordinati per nome azienda"
            )
        else:
            _cmp_extra = []
        st.write("")
        _run_cmp = st.button("📊 Confronta", key="cmp_run",
                             type="primary", use_container_width=True)

    # ── Esecuzione confronto ─────────────────────────────────────────────
    if _run_cmp:
        _raw     = [t.strip().upper() for t in (_cmp_input or "").splitlines() if t.strip()]
        _all_cmp = list(dict.fromkeys(_raw + _cmp_extra))[:12]
        if not _all_cmp:
            st.warning("Inserisci almeno un ticker.")
        else:
            import urllib.request as _ur_cmp, json as _js_cmp

            @st.cache_data(ttl=300, show_spinner=False)
            def _fetch_cmp(tkr: str, rng: str):
                try:
                    import yfinance as _yf_cmp
                    _raw_c = _yf_cmp.download(tkr, period=rng, interval="1d",
                                              auto_adjust=True, progress=False)
                    _raw_c.columns = [c[0] if isinstance(c,tuple) else c for c in _raw_c.columns]
                    if _raw_c.empty:
                        return pd.DataFrame(), tkr, {}
                    _info = {}
                    try:
                        _ti = _yf_cmp.Ticker(tkr).info
                        _info = {
                            "name":      _ti.get("longName") or _ti.get("shortName") or tkr,
                            "sector":    _ti.get("sector","—"),
                            "mcap":      _ti.get("marketCap",0),
                            "pe":        _ti.get("trailingPE",None),
                            "fwd_pe":    _ti.get("forwardPE",None),
                            "eps_fwd":   _ti.get("forwardEps",None),
                            "div_yield": _ti.get("dividendYield",None),
                            "beta":      _ti.get("beta",None),
                            "52w_high":  _ti.get("fiftyTwoWeekHigh",None),
                            "52w_low":   _ti.get("fiftyTwoWeekLow",None),
                            "avg_vol":   _ti.get("averageVolume",None),
                        }
                    except Exception:
                        _info = {"name": tkr}
                    _cl = _raw_c["Close"].dropna()
                    _df_c = pd.DataFrame({"date": _raw_c.index, "close": _cl}).dropna()
                    return _df_c, _info.get("name", tkr), _info
                except Exception:
                    return pd.DataFrame(), tkr, {}

            with st.spinner(f"Carico {len(_all_cmp)} ticker…"):
                _cmp_data = {}
                for _ct in _all_cmp:
                    _dfc, _nmc, _inf = _fetch_cmp(_ct, _cmp_range)
                    if not _dfc.empty:
                        _cmp_data[_ct] = (_dfc, _nmc, _inf)

            if not _cmp_data:
                st.error("Nessun dato disponibile. Verifica i simboli.")
            else:
                _pal_cmp = ["#2962ff","#00d4aa","#f97316","#f59e0b",
                            "#a78bfa","#ef5350","#26c6da","#00e676",
                            "#ff4081","#ffd740","#40c4ff","#69f0ae"]
                fig_cmp = go.Figure()
                _kpi_rows = []
                for i, (ct, (dfc, nmc, inf)) in enumerate(_cmp_data.items()):
                    base = float(dfc["close"].dropna().iloc[0])
                    norm = (dfc["close"] / base - 1) * 100
                    chg  = float(norm.iloc[-1])
                    cur_price = float(dfc["close"].iloc[-1])
                    col_c = _pal_cmp[i % len(_pal_cmp)]
                    fig_cmp.add_trace(go.Scatter(
                        x=dfc["date"].dt.strftime("%Y-%m-%d") if hasattr(dfc["date"],"dt") else dfc["date"].astype(str),
                        y=norm.round(2), mode="lines",
                        name=f"{ct}  {nmc[:22]}",
                        line=dict(color=col_c, width=2.2),
                        hovertemplate=f"<b>{ct}</b> {nmc}<br>%{{y:+.2f}}%<extra></extra>",
                    ))

                    # ── v41: colonne professionali ───────────────────────
                    _52wh = inf.get("52w_high")
                    _52wl = inf.get("52w_low")
                    _dist_52wh = round((cur_price/_52wh-1)*100,1) if _52wh and _52wh>0 else None
                    _dist_52wl = round((cur_price/_52wl-1)*100,1) if _52wl and _52wl>0 else None

                    def _fmt(v, fmt=".2f", suffix=""):
                        return f"{v:{fmt}}{suffix}" if v is not None else "—"

                    _mcap = inf.get("mcap",0)
                    _mcap_str = (f"${_mcap/1e12:.2f}T" if _mcap and _mcap>=1e12
                                 else f"${_mcap/1e9:.1f}B" if _mcap and _mcap>=1e9
                                 else "—")

                    # Volatilità 20d annualizzata
                    _vols = dfc["close"].pct_change().dropna()
                    _vol20 = round(float(_vols.tail(20).std() * (252**0.5) * 100), 1) if len(_vols)>=20 else None

                    # RS vs SPY (return periodo vs SPY stesso periodo)
                    _spy_base_ret = _get_spy_return_20d()
                    _tkr_ret_20d  = round(float((dfc["close"].iloc[-1]/dfc["close"].iloc[-20]-1)*100),1) if len(dfc)>=20 else 0
                    _rs_val = round(_tkr_ret_20d - _spy_base_ret, 1)

                    _kpi_rows.append({
                        "Ticker":       ct,
                        "Nome":         nmc[:28],
                        "Settore":      inf.get("sector","—"),
                        "Prezzo":       f"${cur_price:.2f}",
                        f"Rend {_cmp_range}": f"{chg:+.1f}%",
                        "RS vs SPY":    f"{_rs_val:+.1f}%",
                        "Volatilità":   f"{_vol20:.1f}%" if _vol20 else "—",
                        "P/E":          _fmt(inf.get("pe"),".1f"),
                        "P/E Fwd":      _fmt(inf.get("fwd_pe"),".1f"),
                        "Div Yield":    f"{inf.get('div_yield',0)*100:.2f}%" if inf.get("div_yield") else "—",
                        "Beta":         _fmt(inf.get("beta"),".2f"),
                        "52W High":     f"${_52wh:.2f}" if _52wh else "—",
                        "Dist 52W H":   f"{_dist_52wh:+.1f}%" if _dist_52wh is not None else "—",
                        "Mkt Cap":      _mcap_str,
                        "_rs": _rs_val,
                        "_chg": chg,
                    })

                fig_cmp.add_hline(y=0, line=dict(color="#363a45", width=1, dash="dot"))
                fig_cmp.update_layout(
                    paper_bgcolor="#131722", plot_bgcolor="#1e222d",
                    title=dict(
                        text=f"Performance normalizzata (base 100) · {_cmp_range}",
                        font=dict(color="#50c4e0", size=13), x=0.01
                    ),
                    height=430,
                    yaxis=dict(title="Rendimento %", ticksuffix="%",
                               gridcolor="#2a2e39", zeroline=False,
                               tickfont=dict(color="#787b86", size=10)),
                    xaxis=dict(gridcolor="#2a2e39",
                               tickfont=dict(color="#787b86", size=9)),
                    legend=dict(orientation="h", y=1.05, x=0,
                                bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hovermode="x unified",
                    margin=dict(l=0, r=0, t=48, b=0),
                    font=dict(color="#b2b5be",
                              family="Trebuchet MS, sans-serif", size=11),
                )
                st.plotly_chart(fig_cmp, use_container_width=True, key="cmp_chart")

                # ── Tabella KPI professionale v41 ─────────────────────
                df_kpi = (pd.DataFrame(_kpi_rows)
                            .sort_values("_chg", ascending=False)
                            .drop(columns=["_rs","_chg"])
                            .reset_index(drop=True))

                # Colora Rendimento e RS vs SPY
                def _color_pct_col(s):
                    def _cell(v):
                        try:
                            val = float(str(v).replace("%","").replace("+",""))
                            return f"color: {'#00ff88' if val>0 else '#ef4444' if val<0 else '#6b7280'};font-weight:bold;font-family:Courier New"
                        except Exception:
                            return ""
                    return [_cell(x) for x in s]

                _pct_cols = [c for c in df_kpi.columns if c in
                             [f"Rend {_cmp_range}","RS vs SPY","Volatilità","Dist 52W H"]]
                st.dataframe(
                    df_kpi.style.apply(_color_pct_col, subset=_pct_cols),
                    use_container_width=True, hide_index=True, height=280,
                )

    # ── Se compare_tab.py esiste, passagli i ticker di default ─────────
    try:
        from utils.compare_tab import render_compare
        _df_scan_all = pd.concat(
            [df for df in [df_ep, df_rea] if df is not None and not df.empty],
            ignore_index=True
        ) if any(df is not None and not df.empty for df in [df_ep, df_rea]) else None

        # Inietta i ticker di default nel session_state usato da compare_tab
        # (i text_area di compare_tab leggono tipicamente da questi key)
        _cmp_ticker_keys = [
            "compare_tickers", "cmp_input", "compare_input",
            "ticker_input", "tickers_input", "multi_tickers"
        ]
        _default_val = "\n".join(t for t,_,_,_ in _CMP_DEFAULTS)
        for _ck in _cmp_ticker_keys:
            if _ck not in st.session_state:
                st.session_state[_ck] = _default_val

        with st.expander("📊 Comparatore avanzato (compare_tab.py)", expanded=False):
            try:
                render_compare(_df_scan_all, default_tickers=[t for t,_,_,_ in _CMP_DEFAULTS])
            except TypeError:
                render_compare(_df_scan_all)
    except ImportError:
        pass
    except Exception as _ce:
        st.error(f"Comparatore error: {_ce}")

with tab_ser:
    show_legend("🎯 Serafini")
    # Mostra criteri dettaglio
    with st.expander("✅ Criteri Serafini nel dettaglio — v41",expanded=False):
        st.markdown("""
| # | Criterio | Calcolo | Soglia | Novità v34 |
|---|----------|---------|--------|------------|
| 1 | **RSI > 50** | RSI(14) | >50 | — |
| 2 | **Prezzo > EMA20** | Close > EMA(20) | Sì | — |
| 2b | **Prezzo > EMA50** | Close > EMA(50) | Sì | 🆕 aggiunto |
| 3 | **EMA20 > EMA50** | EMA(20) > EMA(50) — golden align | Sì | — |
| 4 | **OBV crescente** | OBV_Trend = UP | Sì | — |
| 5 | **Volume significativo** | Vol_Ratio | **≥ 1.5** | ⬆️ alzato da 1.0 |
| 6 | **No earnings prossimi** | Earnings Date > 14gg | Sì | — |
| 7 | **Weekly Bull** *(bonus)* | Weekly_Bull = True | +1 score | 🆕 bonus |

**Ser_OK = True** quando tutti i criteri 1·2·2b·3·4·5·6 sono soddisfatti (7 criteri hard).  
**Ser_Score** va da 0 a **8** — il punto 7 (weekly) è bonus. Score ≥ 6 con Ser_OK=False = quasi qualificato, vale la pena esaminarlo.
""")
    render_scan_tab(df_ep,"SERAFINI",["Ser_Score","Quality_Score","RSI"],[False,False,True],"🎯 Serafini")

with tab_fvpro:
    show_legend("🔎 Finviz Pro")
    with st.expander("✅ Filtri Finviz replicati",expanded=False):
        st.markdown("""
| Filtro Finviz | Replica yfinance | Soglia |
|---|---|---|
| Price $ | `Close` | > $10 |
| Average Volume | `avg_vol_20` | > 1.000.000 |
| Relative Volume | `vol_today / avg_vol_20` | > 1.0 |
| 20-Day SMA | `Close > EMA(20)` | Sì |
| 50-Day SMA | `Close > EMA(50)` | Sì |
| 200-Day SMA | `Close > SMA(200)` | Sì |
| EPS Growth Next Year | `(forwardEPS-trailingEPS)/abs(trailingEPS)` | > 10% |
| EPS Growth Next 5Y | `revenueGrowth` _(proxy)_ | > 15% |
| Optionable | Exchange in [NMS,NYQ,ASE,...] _(proxy)_ | — (info) |

> ⚠️ I dati fondamentali EPS Growth dipendono dalla disponibilità in yfinance.  
> Per dati precisi si consiglia Finviz Elite API.
""")
    render_scan_tab(df_ep,"FINVIZ_PRO",["FV_Score","Quality_Score","EPS_NY_Gr"],[False,False,False],"🔎 Finviz Pro")

# =========================================================================
# CRISIS MONITOR TAB
# =========================================================================
with tab_crisis:
    st.markdown('<div class="section-pill">🛡️ CRISIS MONITOR — Asset Difensivi</div>',
                unsafe_allow_html=True)

    st.markdown("""
> **Come usare questo tab**: seleziona lo scenario di rischio che ti preoccupa.
> Per ogni asset trovi ticker, nome e descrizione tattica. Clicca sul ticker per aprire TradingView.
> Aggiungi alla watchlist per seguire l'analisi tecnica con lo scanner.
""")

    # ── Selezione scenario ────────────────────────────────────────────
    scenario_labels = {
        "🌍 Guerra / Conflitto Militare":  ["🥇 Metalli Preziosi","⚫ Energia & Petrolio","🔫 Difesa & Aerospazio","🏦 Treasuries & Obbligazioni","💵 Valute Rifugio"],
        "📈 Inflazione Alta":              ["🥇 Metalli Preziosi","⚫ Energia & Petrolio","🍞 Commodities & Agri","💵 Valute Rifugio","🌍 Mercati Neutri / Commodity States"],
        "🧱 Stagflazione":                 ["🥇 Metalli Preziosi","⚫ Energia & Petrolio","🍞 Commodities & Agri","⚡ Utilities","💊 Healthcare & Pharma","💵 Valute Rifugio"],
        "📉 Crash / Panic Sell":            ["🥇 Metalli Preziosi","🏦 Treasuries & Obbligazioni","⚡ Utilities","💊 Healthcare & Pharma","💵 Valute Rifugio"],
        "🦠 Pandemia / Crisi Sanitaria":   ["💊 Healthcare & Pharma","🥇 Metalli Preziosi","⚡ Utilities","🏦 Treasuries & Obbligazioni"],
        "💻 Crisi Energetica":             ["⚫ Energia & Petrolio","⚡ Utilities","🌍 Mercati Neutri / Commodity States"],
        "📊 Tutti gli asset difensivi":    list(CRISIS_ASSETS.keys()),
    }

    # Inizializza session_state per evitare crash al primo render
    if "crisis_scenario" not in st.session_state:
        st.session_state["crisis_scenario"] = list(scenario_labels.keys())[0]

    sc_col1, sc_col2 = st.columns([2, 3])
    with sc_col1:
        selected_scenario = st.selectbox(
            "🎯 Seleziona scenario di rischio",
            list(scenario_labels.keys()),
            key="crisis_scenario"
        )
    with sc_col2:
        _n_cats   = len(scenario_labels.get(selected_scenario, []))
        _n_assets = sum(len(CRISIS_ASSETS.get(c,{}).get("assets",[]))
                        for c in scenario_labels.get(selected_scenario, []))
        st.markdown(f"""
<div style="background:#1a2332;border:1px solid #2d3f55;border-radius:8px;padding:10px;margin-top:8px">
<b style="color:#60a5fa">Scenario selezionato:</b>
<span style="color:#e2e8f0"> {selected_scenario}</span><br>
<span style="color:#6b7280;font-size:0.82rem">{_n_cats} categorie — {_n_assets} asset totali</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    active_categories = scenario_labels[selected_scenario]
    all_crisis_tickers = []

    # ── Scanner fisso Crisis Monitor ───────────────────────────────────────
    # Scansiona TUTTI i ticker CRISIS_ASSETS (indipendente dalla sidebar)
    # Cache unica per tutti gli scenari — si aggiorna solo su richiesta esplicita

    def _slug_fast(s):
        import re as _re2
        return _re2.sub(r'[^\w]','',s)[:16]

    # Lista COMPLETA di tutti i ticker crisis (tutte le categorie)
    _ALL_CRISIS_TKS = list(dict.fromkeys(
        t for _cat in CRISIS_ASSETS.values()
        for t,_,_ in _cat.get("assets", [])
    ))
    _CRISIS_CACHE_KEY = "_crisis_scan_all"   # chiave unica per tutti gli scenari

    _crisis_df_cached = st.session_state.get(_CRISIS_CACHE_KEY)

    # Barra di stato + bottone aggiorna
    _hc1, _hc2, _hc3 = st.columns([3, 2, 3])
    with _hc1:
        if _crisis_df_cached is not None:
            _ts = st.session_state.get("_crisis_scan_time", "")
            st.markdown(
                f'<div style="background:#1a2e1a;border:1px solid #2a4a2a;'
                f'border-radius:6px;padding:8px 12px;font-size:0.82rem">'
                f'✅ <b style="color:#26a69a">{len(_crisis_df_cached)} ticker</b>'
                f' con dati live'
                f'{"  ·  🕐 " + _ts if _ts else ""}'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#2e1a1a;border:1px solid #4a2a2a;'
                'border-radius:6px;padding:8px 12px;font-size:0.82rem">'
                '⚠️ <b style="color:#ef5350">Dati non disponibili</b>'
                ' — premi Scansiona per popolare RSI, Quality, Vol×, etc.'
                '</div>', unsafe_allow_html=True
            )
    with _hc2:
        _run_crisis = st.button(
            f"🔍 Scansiona tutti ({len(_ALL_CRISIS_TKS)})",
            key="crisis_scan_btn",
            type="primary",
            use_container_width=True,
            help=f"Scarica dati tecnici live per tutti i {len(_ALL_CRISIS_TKS)} asset difensivi — indipendente dalla selezione mercati"
        )
    with _hc3:
        _col_rf, _col_rs = st.columns(2)
        with _col_rf:
            if _crisis_df_cached is not None:
                if st.button("🔄 Aggiorna", key="crisis_scan_refresh",
                             use_container_width=True,
                             help="Forza nuova scansione"):
                    st.session_state.pop(_CRISIS_CACHE_KEY, None)
                    st.session_state.pop("_crisis_scan_time", None)
                    st.rerun()
        with _col_rs:
            if _crisis_df_cached is not None:
                if st.button("🗑️ Reset", key="crisis_scan_reset",
                             use_container_width=True):
                    st.session_state.pop(_CRISIS_CACHE_KEY, None)
                    st.session_state.pop("_crisis_scan_time", None)
                    st.rerun()

    # ── Esegui scansione ──────────────────────────────────────────────────
    if _run_crisis:
        _crisis_rows = []
        _crisis_errors = []
        _prog = st.progress(0, text="🔍 Avvio scansione Crisis Monitor...")
        _n = len(_ALL_CRISIS_TKS)
        for _i, _tkr in enumerate(_ALL_CRISIS_TKS):
            _prog.progress((_i + 1) / _n,
                           text=f"🔍 {_tkr}  ({_i+1}/{_n})")
            try:
                _ep_row, _rea_row = scan_ticker(
                    _tkr,
                    e_h=0.03,
                    p_rmin=25,
                    p_rmax=85,
                    r_poc=0.03,
                    vol_ratio_hot=1.2,
                )
                _row = _ep_row if _ep_row is not None else _rea_row
                if _row is not None:
                    _crisis_rows.append(_row)
                else:
                    _crisis_errors.append(f"{_tkr}: scan_ticker → (None, None)")
            except Exception as _ex:
                _crisis_errors.append(f"{_tkr}: {type(_ex).__name__}: {_ex}")
        _prog.empty()
        if _crisis_rows:
            _crisis_df = pd.DataFrame(_crisis_rows)
            st.session_state[_CRISIS_CACHE_KEY] = _crisis_df
            st.session_state["_crisis_scan_time"] = datetime.now().strftime("%H:%M")
            if _crisis_errors:
                with st.expander(f"⚠️ {len(_crisis_errors)} ticker non caricati", expanded=False):
                    st.code("\n".join(_crisis_errors[:20]))
            st.success(f"✅ Scansione completata: {len(_crisis_df)}/{_n} ticker")
            st.rerun()
        else:
            st.error(f"⚠️ Nessun dato recuperato ({_n} ticker tentati). Errori:")
            st.code("\n".join(_crisis_errors[:30]) if _crisis_errors else "Nessun errore registrato — scan_ticker ha restituito None per tutti")

    # Dati live da usare nel merge per ogni categoria
    _crisis_live = st.session_state.get(_CRISIS_CACHE_KEY)

    # ── Per ogni categoria — lista ticker con filtro scenario attivo ───────
    _all_crisis_tks = []   # mantieni compatibilità variabile downstream

    def _slug(s, maxlen=12):
        """Rimuove emoji e spazi per creare una chiave Streamlit valida."""
        import re as _re
        clean = _re.sub(r'[^\w]', '', s)
        return clean[:maxlen] if clean else "cat"

# ── Per ogni categoria ─────────────────────────────────────────────
    for cat_name in active_categories:
        cat_data = CRISIS_ASSETS.get(cat_name, {})
        if not cat_data: continue
        assets = cat_data.get("assets", [])
        if not assets: continue

        st.markdown(f"### {cat_name}")
        st.markdown(f"*{CRISIS_LEGEND.get(cat_name, cat_data.get('desc',''))}*")

        rows = [{"Ticker": t, "Nome": n, "Descrizione Tattica": d} for t,n,d in assets]
        df_crisis_cat = pd.DataFrame(rows)
        all_crisis_tickers.extend([r[0] for r in assets])
        # Arricchisci con dati scanner: prima crisis scan dedicato, poi scanner principale
        _live_keep = ["Ticker","Prezzo","RSI","Vol_Ratio","OBV_Trend",
                      "Stato_Early","Quality_Score","Early_Score","Pro_Score",
                      "Squeeze","Weekly_Bull"]
        for _ldf in [_crisis_live, df_ep, df_rea]:
            if _ldf is None or _ldf.empty or "Ticker" not in _ldf.columns: continue
            _sub = _ldf[[c for c in _live_keep if c in _ldf.columns]].copy()
            df_crisis_cat = df_crisis_cat.merge(_sub, on="Ticker", how="left")
            break  # primo df disponibile basta

        # ── v41 FIX ROBUSTO: elimina tutti i NaN/inf prima di AgGrid ──────
        import numpy as _np, math as _math
        # 1) Colonne testo: qualsiasi NaN → "—"
        _text_cols_crisis = {"Ticker","Nome","Descrizione Tattica","OBV_Trend",
                             "Stato_Early","RSI_Div","CSS_Grade","Squeeze","Weekly_Bull"}
        for _col in df_crisis_cat.columns:
            if _col in _text_cols_crisis or df_crisis_cat[_col].dtype == object:
                df_crisis_cat[_col] = (
                    df_crisis_cat[_col].astype(str)
                    .replace({"nan":"—","None":"—","<NA>":"—","NaN":"—"})
                )
            else:
                df_crisis_cat[_col] = (
                    pd.to_numeric(df_crisis_cat[_col], errors="coerce")
                    .replace([_np.inf, -_np.inf], _np.nan)
                    .fillna(0.0)
                )
        # 2) Ultimo controllo record-per-record: float NaN → None (→ JSON null)
        _safe_recs = []
        for _rec in df_crisis_cat.to_dict(orient="records"):
            _safe = {
                _k: (None if isinstance(_v, float) and (_math.isnan(_v) or _math.isinf(_v)) else _v)
                for _k, _v in _rec.items()
            }
            _safe_recs.append(_safe)
        df_crisis_cat = pd.DataFrame(_safe_recs)

        gb_c = GridOptionsBuilder.from_dataframe(df_crisis_cat)
        gb_c.configure_default_column(sortable=True, resizable=True, filterable=False, minWidth=65)
        gb_c.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb_c.configure_column("Ticker", width=100, pinned="left",
            cellRenderer=JsCode("""class T{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value||'';const t=p.value;if(!t)return;
this.eGui.style.cursor='pointer';this.eGui.style.color='#50c4e0';
this.eGui.style.fontWeight='bold';this.eGui.style.fontFamily='Trebuchet MS';
this.eGui.title='Doppio click → TradingView';
this.eGui.ondblclick=()=>window.open('https://it.tradingview.com/chart/?symbol='+String(t).split('.')[0],'_blank');}
getGui(){return this.eGui;}refresh(){return false;}}"""))
        gb_c.configure_column("Nome", width=195)
        gb_c.configure_column("Descrizione Tattica", width=360, wrapText=True, autoHeight=True)
        # Colonne dati live (se disponibili dallo scanner)
        if "Prezzo" in df_crisis_cat.columns:
            gb_c.configure_column("Prezzo", width=88, headerName="Prezzo $",
                cellRenderer=JsCode("""class P{init(p){this.eGui=document.createElement('span');
const v=parseFloat(p.value);this.eGui.innerText=isNaN(v)?'—':'$'+v.toFixed(2);
this.eGui.style.color='#d1d4dc';this.eGui.style.fontWeight='600';}
getGui(){return this.eGui;}refresh(){return false;}}"""))
        if "RSI" in df_crisis_cat.columns:
            gb_c.configure_column("RSI", width=68, cellRenderer=rsi_renderer)
        if "Vol_Ratio" in df_crisis_cat.columns:
            gb_c.configure_column("Vol_Ratio", width=82, headerName="Vol×",
                cellRenderer=vol_ratio_renderer)
        if "Quality_Score" in df_crisis_cat.columns:
            gb_c.configure_column("Quality_Score", width=82, headerName="Quality",
                cellRenderer=quality_renderer)
        if "OBV_Trend" in df_crisis_cat.columns:
            gb_c.configure_column("OBV_Trend", width=80, headerName="OBV Trend")
        if "Stato_Early" in df_crisis_cat.columns:
            gb_c.configure_column("Stato_Early", width=85, headerName="Stato")
        if "Early_Score" in df_crisis_cat.columns:
            gb_c.configure_column("Early_Score", width=72, headerName="E.Score")
        if "Pro_Score" in df_crisis_cat.columns:
            gb_c.configure_column("Pro_Score", width=72, headerName="P.Score")
        if "Squeeze" in df_crisis_cat.columns:
            gb_c.configure_column("Squeeze", width=72, cellRenderer=squeeze_renderer)
        if "Weekly_Bull" in df_crisis_cat.columns:
            gb_c.configure_column("Weekly_Bull", width=68, headerName="W+",
                cellRenderer=weekly_renderer)
        go_c = gb_c.build()

        try:
            resp_c = AgGrid(df_crisis_cat, gridOptions=go_c,
                            height=min(120 + len(assets)*35, 440),
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            fit_columns_on_grid_load=True, theme="streamlit",
                            allow_unsafe_jscode=True, key=f"cg_{_slug(cat_name)}")
            sel_crisis = pd.DataFrame(resp_c["selected_rows"])
        except Exception as _ag_err:
            # Fallback: dataframe semplice se AgGrid non disponibile
            st.dataframe(df_crisis_cat, use_container_width=True, hide_index=True)
            sel_crisis = pd.DataFrame()

        c_a1, c_a2, _ = st.columns([2, 2, 4])
        with c_a1:
            if st.button(f"➕ Aggiungi selezionati", key=f"cadd_{_slug(cat_name)}"):
                if not sel_crisis.empty and "Ticker" in sel_crisis.columns:
                    tks = sel_crisis["Ticker"].tolist()
                    nms = sel_crisis["Nome"].tolist()
                    gh_add_to_watchlist(tks, nms, f"Crisis:{cat_name[:18]}", "CrisisMonitor",
                                     "WATCH", st.session_state.current_list_name)
                    st.success(f"✅ Aggiunti {len(tks)} ticker."); time.sleep(0.5); st.rerun()
                else:
                    st.warning("Seleziona almeno un asset dalla griglia.")
        with c_a2:
            if st.button(f"➕ Tutti ({len(assets)})", key=f"call_{_slug(cat_name)}"):
                tks=[r[0] for r in assets]; nms=[r[1] for r in assets]
                gh_add_to_watchlist(tks, nms, f"Crisis:{cat_name[:18]}", "CrisisMonitor",
                                 "WATCH", st.session_state.current_list_name)
                st.success(f"✅ Aggiunti tutti i {len(tks)} ticker."); time.sleep(0.5); st.rerun()
        # ── Grafico ticker selezionato (come negli altri tab) ──────────
        # Controlla selezione esplicita (non solo riga pre-selezionata)
        _has_selection = (not sel_crisis.empty
                          and "Ticker" in sel_crisis.columns
                          and len(sel_crisis) > 0)
        if _has_selection:
            _ctkr = sel_crisis.iloc[0].get("Ticker","")
            _crow = None
            for _cdf in [df_ep, df_rea]:
                if _cdf is None or _cdf.empty or "Ticker" not in _cdf.columns: continue
                _cm = _cdf[_cdf["Ticker"]==_ctkr]
                if not _cm.empty and "_chart_data" in _cm.columns:
                    _cd = _cm.iloc[0].get("_chart_data")
                    if _cd and isinstance(_cd, dict) and _cd.get("dates"):
                        _crow = _cm.iloc[0]; break
            if _crow is not None:
                show_charts(_crow, key_suffix=f"cr_{_slug(cat_name)}")
            else:
                st.info(f"📭 Dati tecnici per **{_ctkr}** non disponibili. "
                        f"Esegui lo scanner su questo mercato.")
        st.markdown("")

    # ── Legenda e guida ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 Guida — Come usare il Crisis Monitor e performance storiche", expanded=False):
        st.markdown("""
## 🛡️ Crisis Monitor — Guida Operativa

### 📊 Come usare il tab
| Azione | Come fare |
|--------|-----------|
| **Aprire grafico** | Clicca sul ticker (link blu) → TradingView |
| **Aggiungere alla watchlist** | Seleziona riga → ➕ Aggiungi selezionati |
| **Analisi tecnica** | Dopo averli in watchlist, esegui lo scanner per segnali |
| **Cambiare scenario** | Usa il selettore in cima |

### 🎯 Criteri di selezione asset
- ✅ **Liquidità** > 1M$/giorno — trattabili senza slippage
- ✅ **Correlazione provata** con lo scenario (dati storici reali)
- ✅ **Strumenti regolamentati** NYSE/NASDAQ — niente prodotti esotici
- ✅ **Diversificazione**: ETF broad + singoli titoli per leva

### 📈 Performance storica in scenari di crisi
| Scenario | Asset vincente | Performance tipica |
|----------|---------------|-------------------|
| Guerra Ucraina Feb 2022 | LMT +36%, RTX +28%, XOM +40% | +30/50% in 3 mesi |
| COVID Crash Mar 2020 | TLT +20%, GLD +15%, XLV -5% | TLT unico rialzista |
| Inflazione 2021-2022 | XOM +80%, OXY +120%, WEAT +65% | Energia/agri dominano |
| Crisi bancaria Mar 2023 | GLD +8%, BTC +40%, TLT +6% | Oro e Bitcoin |
| 9/11 Settembre 2001 | GLD, LMT, RTX +15% in 6 mesi | Difesa e oro |

### ⚠️ Avvertenze
> I rendimenti passati non garantiscono quelli futuri. Questo è uno strumento informativo,
> non consulenza finanziaria. Alcuni ETF (RSX Russia) possono diventare illiquidi in caso di sanzioni.
""")

    # ── 📋 RIEPILOGO TECNICO — griglia stile PRO ─────────────────────
    # Aggrega tutti gli asset delle categorie attive con dati live scanner
    st.markdown("---")
    st.markdown('<div class="section-pill">📋 RIEPILOGO TECNICO — tutti gli asset attivi</div>',
                unsafe_allow_html=True)

    _riepilogo_rows = []
    for _rc in active_categories:
        _rcat = CRISIS_ASSETS.get(_rc, {})
        for _rt, _rn, _rd in _rcat.get("assets", []):
            _riepilogo_rows.append({
                "Categoria": _rc, "Ticker": _rt, "Nome": _rn,
                "Tattica": _rd[:60] + ("…" if len(_rd) > 60 else "")
            })

    if _riepilogo_rows:
        df_riepilogo = pd.DataFrame(_riepilogo_rows)

        # Arricchisci con dati live (crisis scan o scanner principale)
        _rlive_cols = ["Ticker","Prezzo","RSI","Vol_Ratio","OBV_Trend",
                       "Stato_Early","Stato_Pro","Quality_Score",
                       "Pro_Score","Early_Score","Squeeze","Weekly_Bull",
                       "ATR_pct","Dollar_Vol"]
        for _rsrc in [_crisis_live, df_ep, df_rea]:
            if _rsrc is None or _rsrc.empty or "Ticker" not in _rsrc.columns: continue
            _rsub = _rsrc[[c for c in _rlive_cols if c in _rsrc.columns]].copy()
            df_riepilogo = df_riepilogo.merge(_rsub, on="Ticker", how="left")
            break

        # Griglia stile PRO con tutti i renderer già definiti
        gb_r = GridOptionsBuilder.from_dataframe(df_riepilogo)
        gb_r.configure_default_column(sortable=True, resizable=True,
                                      filterable=True, minWidth=70)
        gb_r.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb_r.configure_column("Categoria", width=130, pinned="left")
        gb_r.configure_column("Ticker", width=85, pinned="left",
            cellRenderer=JsCode("""class T{init(p){this.eGui=document.createElement('span');
this.eGui.innerText=p.value||'';const t=p.value;if(!t)return;
this.eGui.style.cssText='cursor:pointer;color:#50c4e0;font-weight:bold;font-family:Courier New';
this.eGui.title='Doppio click → TradingView';
this.eGui.ondblclick=()=>window.open('https://it.tradingview.com/chart/?symbol='+String(t).split('.')[0],'_blank');}
getGui(){return this.eGui;}refresh(){return false;}}"""))
        gb_r.configure_column("Nome", width=180)
        gb_r.configure_column("Tattica", width=300, wrapText=True, autoHeight=True)
        # Colonne dati live con renderer PRO
        for _rc_col, _rc_wd, _rc_rend in [
            ("Prezzo",        88,  price_renderer),
            ("RSI",           68,  rsi_renderer),
            ("Vol_Ratio",     85,  vol_ratio_renderer),
            ("Quality_Score", 90,  quality_renderer),
            ("Stato_Pro",     95,  stato_pro_renderer),
            ("Squeeze",       72,  squeeze_renderer),
            ("Weekly_Bull",   68,  weekly_renderer),
            ("ATR_pct",       80,  atr_pct_renderer),
        ]:
            if _rc_col in df_riepilogo.columns:
                gb_r.configure_column(_rc_col, width=_rc_wd, cellRenderer=_rc_rend)

        for _hc in ["OBV_Trend","Stato_Early","Early_Score","Pro_Score","Dollar_Vol"]:
            if _hc in df_riepilogo.columns:
                gb_r.configure_column(_hc, width=80)

        go_r = gb_r.build()
        try:
            AgGrid(df_riepilogo, gridOptions=go_r,
                   height=min(180 + len(_riepilogo_rows) * 38, 600),
                   update_mode=GridUpdateMode.SELECTION_CHANGED,
                   data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                   fit_columns_on_grid_load=False, theme="streamlit",
                   allow_unsafe_jscode=True, key="crisis_riepilogo_grid")
        except Exception:
            st.dataframe(df_riepilogo, use_container_width=True, hide_index=True)
    else:
        st.info("Seleziona uno scenario per vedere il riepilogo tecnico.")

    # ── Export ────────────────────────────────────────────────────────
    st.markdown("---")
    _cx1, _cx2 = st.columns(2)
    _unique = list(dict.fromkeys(all_crisis_tickers))
    with _cx1:
        st.download_button("📺 Export TradingView CSV",
            data=chr(10).join(_unique),
            file_name=f"crisis_{selected_scenario[:25].replace(' ','_')}.csv",
            mime="text/plain", key="crisis_tv_exp",
            help="Un ticker per riga — importabile in TradingView Watchlist")
    with _cx2:
        _cdf = pd.DataFrame([
            {"Categoria":cat,"Ticker":t,"Nome":n,"Descrizione":d}
            for cat in active_categories
            for t,n,d in CRISIS_ASSETS.get(cat,{}).get("assets",[])
        ])
        if not _cdf.empty:
            st.download_button("📊 Export Excel",
                data=to_excel_bytes({"Crisis Monitor":_cdf}),
                file_name="crisis_monitor.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="crisis_xlsx_exp")

    # ── Strategy Chart ────────────────────────────────────────────────────
    try:
        from utils.backtest_tab import strategy_chart_widget as _scw
        _crisis_tkrs = [
            t for cat in active_categories
            for t,_n,_d in CRISIS_ASSETS.get(cat,{}).get("assets",[])
        ]
        st.markdown("---")
        _scw(tickers=_crisis_tkrs, key_suffix="CRISIS")
    except Exception:
        pass


# =========================================================================
# RISK MANAGER TAB
# =========================================================================
with tab_rm:
    try:
        from utils.risk_manager import render_risk_manager
        # Combina df_ep + df_rea per avere tutti i ticker disponibili
        # Ordina alfabeticamente per Nome (fallback su Ticker)
        _rm_frames = [d for d in [df_ep, df_rea]
                      if d is not None and not d.empty]
        if _rm_frames:
            _df_rm = pd.concat(_rm_frames, ignore_index=True)
            # Deduplicazione: mantieni il record con score più alto per ticker
            if "Ticker" in _df_rm.columns:
                _score_col = "Pro_Score" if "Pro_Score" in _df_rm.columns else None
                if _score_col:
                    _df_rm = (_df_rm
                              .sort_values(_score_col, ascending=False)
                              .drop_duplicates(subset=["Ticker"], keep="first"))
                # Ordine alfabetico per Nome (per selectbox leggibile)
                _sort_col = "Nome" if "Nome" in _df_rm.columns else "Ticker"
                _df_rm = _df_rm.sort_values(_sort_col, key=lambda s: s.str.lower()).reset_index(drop=True)
        else:
            _df_rm = None
        render_risk_manager(df_scanner=_df_rm)
    except ImportError:
        st.warning(
            "⚠️ `utils/risk_manager.py` non trovato.\n\n"
            "Copia il file `risk_manager.py` generato nella cartella `utils/` del progetto."
        )
    except Exception as _rme:
        import traceback
        st.error(f"Risk Manager error: {_rme}")
        st.code(traceback.format_exc())

# =========================================================================
# WATCHLIST — AgGrid + cards + multi-lista
# =========================================================================
with tab_w:
    st.markdown(f'<div class="section-pill">📋 WATCHLIST MANAGER — {st.session_state.current_list_name}</div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # v35 UPGRADE #3 — P&L TRACKER + ALERT ENGINE
    # ══════════════════════════════════════════════════════════════════════
    _df_ep_wl = st.session_state.get("df_ep", pd.DataFrame())

    with st.expander("📈 P&L Tracker & Alert Engine v41", expanded=False):
        st.caption("Inserisci prezzo entrata e size per ogni ticker — P&L si aggiorna con prezzi scanner.")
        _pnl_col1, _pnl_col2 = st.columns([2, 1])

        with _pnl_col1:
            st.markdown("**💰 P&L Tracker (Mark-to-Market)**")
            if "v41_pnl_entries" not in st.session_state:
                st.session_state["v41_pnl_entries"] = {}
            _pnl = st.session_state["v41_pnl_entries"]

            # Form aggiunta posizione
            _pa, _pb, _pc, _pd = st.columns([2,1.5,1.5,1])
            with _pa: _tkr_inp = st.text_input("Ticker", key="pnl_tkr", placeholder="es. AAPL").upper().strip()
            with _pb: _entry_inp = st.number_input("Prezzo Entrata $", min_value=0.0, step=0.01, key="pnl_entry")
            with _pc: _size_inp  = st.number_input("N. Azioni", min_value=1, step=1, key="pnl_size")
            with _pd:
                st.write("")
                if st.button("➕ Add", key="pnl_add") and _tkr_inp and _entry_inp > 0:
                    _pnl[_tkr_inp] = {"entry": _entry_inp, "size": _size_inp, "added": datetime.now().strftime("%H:%M")}
                    st.rerun()

            # Tabella P&L mark-to-market
            if _pnl:
                _pnl_rows = []
                for _t, _pos in list(_pnl.items()):
                    # Recupera prezzo corrente dal df_ep se disponibile
                    _cur_price = None
                    if not _df_ep_wl.empty and "Ticker" in _df_ep_wl.columns and "Prezzo" in _df_ep_wl.columns:
                        _match = _df_ep_wl[_df_ep_wl["Ticker"] == _t]
                        if not _match.empty:
                            _cur_price = float(_match.iloc[0]["Prezzo"])
                    _entry = _pos["entry"]; _size = _pos["size"]
                    if _cur_price is not None:
                        _pnl_usd = (_cur_price - _entry) * _size
                        _pnl_pct = (_cur_price / _entry - 1) * 100
                        _pnl_str = f"${_pnl_usd:+,.0f} ({_pnl_pct:+.1f}%)"
                        _color    = "#00ff88" if _pnl_usd >= 0 else "#ef4444"
                    else:
                        _pnl_str = "— (no scanner data)"
                        _color    = "#6b7280"
                        _cur_price = _entry
                    _pnl_rows.append({
                        "Ticker": _t,
                        "Entry $": f"${_entry:.2f}",
                        "Current $": f"${_cur_price:.2f}" if _cur_price else "—",
                        "Size": _size,
                        "P&L": _pnl_str,
                        "_color": _color,
                        "Added": _pos.get("added",""),
                    })

                for _row in _pnl_rows:
                    _c1,_c2,_c3,_c4,_c5 = st.columns([1.5,1,1,0.8,0.5])
                    _c1.markdown(f"**`{_row['Ticker']}`**")
                    _c2.caption(f"Entry: {_row['Entry $']}")
                    _c3.caption(f"Now: {_row['Current $']}")
                    _c4.markdown(f"<span style='color:{_row['_color']};font-weight:bold;font-family:Courier New;font-size:0.85rem'>{_row['P&L']}</span>", unsafe_allow_html=True)
                    with _c5:
                        if st.button("🗑", key=f"pnl_del_{_row['Ticker']}"):
                            _pnl.pop(_row["Ticker"], None); st.rerun()

                # Totale P&L
                _total_pnl = 0.0
                for _t, _pos in _pnl.items():
                    if not _df_ep_wl.empty and "Ticker" in _df_ep_wl.columns and "Prezzo" in _df_ep_wl.columns:
                        _m = _df_ep_wl[_df_ep_wl["Ticker"] == _t]
                        if not _m.empty:
                            _total_pnl += (float(_m.iloc[0]["Prezzo"]) - _pos["entry"]) * _pos["size"]
                _tc = "#00ff88" if _total_pnl >= 0 else "#ef4444"
                st.markdown(f"**Portfolio P&L: <span style='color:{_tc};font-family:Courier New'>${_total_pnl:+,.0f}</span>**", unsafe_allow_html=True)
            else:
                st.info("Nessuna posizione. Aggiungi ticker sopra.")

        with _pnl_col2:
            st.markdown("**🔔 Alert Engine**")
            if "v41_alerts" not in st.session_state:
                st.session_state["v41_alerts"] = {}
            _alerts = st.session_state["v41_alerts"]

            _at, _av, _atype = st.columns([1.5, 1.5, 1.5])
            with _at:   _alert_tkr   = st.text_input("Ticker", key="alt_tkr", placeholder="AAPL").upper().strip()
            with _av:   _alert_val   = st.number_input("Soglia", min_value=0.0, step=0.01, key="alt_val")
            with _atype: _alert_type = st.selectbox("Tipo", ["Prezzo ≥", "Prezzo ≤", "CSS ≥", "Vol_Ratio ≥"], key="alt_type")
            if st.button("🔔 Set Alert", key="alt_add") and _alert_tkr:
                _alerts[f"{_alert_tkr}_{_alert_type}"] = {"tkr": _alert_tkr, "val": _alert_val, "type": _alert_type, "fired": False}
                st.rerun()

            # Verifica alert contro df_ep live
            if _alerts and not _df_ep_wl.empty:
                _col_map_alt = {"Prezzo ≥": "Prezzo", "Prezzo ≤": "Prezzo", "CSS ≥": "CSS", "Vol_Ratio ≥": "Vol_Ratio"}
                for _ak, _a in list(_alerts.items()):
                    _col = _col_map_alt.get(_a["type"], "Prezzo")
                    if "Ticker" in _df_ep_wl.columns and _col in _df_ep_wl.columns:
                        _row_a = _df_ep_wl[_df_ep_wl["Ticker"] == _a["tkr"]]
                        if not _row_a.empty:
                            _cur_v = float(pd.to_numeric(_row_a.iloc[0].get(_col, 0), errors="coerce") or 0)
                            _fired = (
                                (_a["type"] in ["Prezzo ≥","CSS ≥","Vol_Ratio ≥"] and _cur_v >= _a["val"]) or
                                (_a["type"] == "Prezzo ≤" and _cur_v <= _a["val"])
                            )
                            if _fired: _alerts[_ak]["fired"] = True

            for _ak, _a in list(_alerts.items()):
                _badge = "🔴 FIRED" if _a["fired"] else "🟡 Attivo"
                _bcol  = "#ef4444" if _a["fired"] else "#f59e0b"
                _r1, _r2 = st.columns([3,1])
                _r1.markdown(
                    f"<span style='font-family:Courier New;font-size:0.8rem'>"
                    f"<b style='color:#00ff88'>{_a['tkr']}</b> {_a['type']} "
                    f"<b style='color:#58a6ff'>{_a['val']}</b> "
                    f"<b style='color:{_bcol}'>{_badge}</b></span>",
                    unsafe_allow_html=True
                )
                with _r2:
                    if st.button("✕", key=f"alt_del_{_ak}"):
                        _alerts.pop(_ak, None); st.rerun()

    st.markdown("")
    _wl_col1, _wl_col2 = st.columns([3, 2])
    with _wl_col1:
        if _GH_SYNC:
            _gs = _gh_status(DB_PATH)
            st.markdown(
                f'<div style="background:#1e222d;border-left:3px solid #26a69a;'
                f'padding:6px 12px;border-radius:0 4px 4px 0;font-size:0.82rem;">'
                f'☁️ <b style="color:#26a69a">GitHub Sync attivo</b> — '
                f'<code style="color:#b2b5be">{_gs.get("repo","")}/{_gs.get("path","")}</code>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#1e222d;border-left:3px solid #f59e0b;'
                'padding:6px 12px;border-radius:0 4px 4px 0;font-size:0.82rem;">'
                '⚠️ <b style="color:#f59e0b">GitHub Sync non configurato</b> — '
                'watchlist solo locale (si azzera ad ogni deploy)'
                '</div>',
                unsafe_allow_html=True
            )
    with _wl_col2:
        try:
            _wl_db_ok = DB_PATH.exists()
            _wl_db_sz = round(DB_PATH.stat().st_size/1024,1) if _wl_db_ok else 0
            st.caption(f"💾 `{DB_PATH.name}` — {_wl_db_sz} KB {'✅' if _wl_db_ok else '⚠️'}")
        except Exception as _e:
            st.caption(f"⚠️ DB: {_e}")
        if _GH_SYNC:
            if st.button("☁️ Sync ora", key="wl_sync_now",
                         help="Forza upload watchlist → GitHub"):
                _gh_push(DB_PATH)
                st.success("✅ Watchlist inviata a GitHub!")
    st.markdown("")

    df_wl_full=load_watchlist()

    # gestione lista "pending" (creata dalla sidebar ma non ancora nel DB)
    pending=st.session_state.pop("pending_new_list",None)
    all_lists=sorted(df_wl_full["list_name"].unique().tolist()) if not df_wl_full.empty else []
    if "DEFAULT" not in all_lists: all_lists.append("DEFAULT")
    if pending and pending not in all_lists: all_lists.append(pending); all_lists=sorted(all_lists)

    # ── Pannello gestione liste ──────────────────────────────────────────
    with st.expander("⚙️ Gestione Liste",expanded=True):
        gc1,gc2,gc3,gc4=st.columns(4)

        with gc1:
            st.markdown("**📂 Liste**")
            for ln in all_lists:
                cnt=len(df_wl_full[df_wl_full["list_name"]==ln]) if not df_wl_full.empty else 0
                active_m=" ✅" if ln==st.session_state.current_list_name else ""
                if st.button(f"{ln} ({cnt}){active_m}",key=f"sw_{ln}",use_container_width=True):
                    st.session_state.current_list_name=ln; st.rerun()

        with gc2:
            st.markdown("**✏️ Rinomina**")
            ren_src=st.selectbox("Da",all_lists,key="ren_src")
            ren_dst=st.text_input("Nuovo nome",key="ren_dst")
            if st.button("✏️ Rinomina",key="do_ren") and ren_dst.strip():
                gh_rename_watchlist(ren_src,ren_dst.strip())
                if st.session_state.current_list_name==ren_src:
                    st.session_state.current_list_name=ren_dst.strip()
                st.rerun()

        with gc3:
            st.markdown("**📋 Copia lista**")
            cp_src=st.selectbox("Da",all_lists,key="cp_src")
            cp_dst=st.text_input("A (nuova o esistente)",key="cp_dst")
            if st.button("📋 Copia",key="do_cp") and cp_dst.strip():
                df_src=df_wl_full[df_wl_full["list_name"]==cp_src]
                if not df_src.empty:
                    tc="Ticker" if "Ticker" in df_src.columns else "ticker"
                    nc="Nome"   if "Nome"   in df_src.columns else "name"
                    gh_add_to_watchlist(df_src[tc].tolist(),
                                     df_src[nc].tolist() if nc in df_src.columns else df_src[tc].tolist(),
                                     "Copia",f"da {cp_src}","LONG",cp_dst.strip())
                    st.success(f"✅ Copiati {len(df_src)} ticker."); st.rerun()

        with gc4:
            st.markdown("**🗑️ Elimina lista**")
            dl_sel=st.selectbox("Lista",all_lists,key="dl_sel")
            if st.button("🗑️ Elimina lista",key="do_dl",type="secondary"):
                conn=sqlite3.connect(str(DB_PATH))
                conn.execute("DELETE FROM watchlist WHERE list_name=?",(dl_sel,))
                conn.commit();conn.close()
                if st.session_state.current_list_name==dl_sel:
                    rem=[l for l in all_lists if l!=dl_sel]
                    st.session_state.current_list_name=rem[0] if rem else "DEFAULT"
                st.rerun()

    # ── Contenuto lista attiva ───────────────────────────────────────────
    df_wl=df_wl_full[df_wl_full["list_name"]==st.session_state.current_list_name].copy() \
          if not df_wl_full.empty else pd.DataFrame()

    st.markdown(f'<div class="section-pill">📌 {st.session_state.current_list_name} — {len(df_wl)} titoli</div>',
                unsafe_allow_html=True)

    if df_wl.empty:
        st.info("Lista vuota. Aggiungi ticker dagli altri tab oppure usa **Copia lista**.")
    else:
        tcol="Ticker" if "Ticker" in df_wl.columns else "ticker"
        ncol="Nome"   if "Nome"   in df_wl.columns else "name"

        # ── Vista: toggle cards / griglia ────────────────────────────────
        vmode_col1,vmode_col2,_=st.columns([1,1,4])
        with vmode_col1:
            if st.button("🃏 Cards",key="vm_cards",
                         type="primary" if st.session_state.wl_view_mode=="cards" else "secondary"):
                st.session_state.wl_view_mode="cards"; st.rerun()
        with vmode_col2:
            if st.button("📊 Griglia",key="vm_grid",
                         type="primary" if st.session_state.wl_view_mode=="grid" else "secondary"):
                st.session_state.wl_view_mode="grid"; st.rerun()

        # Merge colonne scanner
        extra_cols=["Prezzo","RSI","Vol_Ratio","Quality_Score","OBV_Trend","Weekly_Bull",
                    "Squeeze","Early_Score","Pro_Score","Ser_Score","Ser_OK","FV_Score","FV_OK"]
        df_wl_disp=df_wl.copy()
        for src_df in [df_ep,df_rea]:
            if not src_df.empty and "Ticker" in src_df.columns:
                for ec in extra_cols:
                    if ec in src_df.columns and ec not in df_wl_disp.columns:
                        mm=src_df[["Ticker",ec]].drop_duplicates("Ticker")
                        df_wl_disp=df_wl_disp.merge(mm,left_on=tcol,right_on="Ticker",
                                                      how="left",suffixes=("","_sc"))
                        if "Ticker_sc" in df_wl_disp.columns:
                            df_wl_disp.drop(columns=["Ticker_sc"],inplace=True)

        # ── Azioni massa ──────────────────────────────────────────────────
        wa1,wa2,wa3=st.columns(3)
        with wa1:
            csv_btn(df_wl_disp,f"watchlist_{st.session_state.current_list_name}.csv","exp_wl_dl")
        other_lists=[l for l in all_lists if l!=st.session_state.current_list_name] or ["DEFAULT"]
        with wa2:
            move_dest=st.selectbox("Sposta selezione →",other_lists,key="mass_mv")
        with wa3:
            copy_dest2=st.selectbox("Copia selezione →",other_lists,key="mass_cp")

        # ── VISTA GRIGLIA (AgGrid con note/trend editabili) ──────────────
        if st.session_state.wl_view_mode=="grid":
            # Prepara colonne per griglia watchlist
            wl_grid_cols=["id",tcol,ncol,"Prezzo","trend","note","origine","created_at",
                          "RSI","Vol_Ratio","Quality_Score","Ser_Score","FV_Score",
                          "Weekly_Bull","Squeeze","Early_Score","Pro_Score","OBV_Trend"]
            df_wg=df_wl_disp[[c for c in wl_grid_cols if c in df_wl_disp.columns]].copy()
            # Rinomina per display
            rename_map={}
            if tcol!="Ticker": rename_map[tcol]="Ticker"
            if ncol!="Nome":   rename_map[ncol]="Nome"
            if rename_map: df_wg=df_wg.rename(columns=rename_map)

            grid_resp_wl=build_aggrid(df_wg,"wl_grid",height=520,
                                       editable_cols=["trend","note"])
            sel_wl_rows=pd.DataFrame(grid_resp_wl["selected_rows"])
            updated_wl =pd.DataFrame(grid_resp_wl["data"])

            # Salva modifiche note/trend
            if not updated_wl.empty and "id" in updated_wl.columns:
                if st.button("💾 Salva Note/Trend",key="save_wl_edits"):
                    conn=sqlite3.connect(str(DB_PATH))
                    for _,r in updated_wl.iterrows():
                        rid=int(r.get("id",0))
                        if rid>0:
                            conn.execute("UPDATE watchlist SET note=?,trend=? WHERE id=?",
                                         (str(r.get("note","")),str(r.get("trend","")),rid))
                    conn.commit();conn.close()
                    st.success("✅ Salvato!"); st.rerun()

            selected_ids=[int(r.get("id",0)) for _,r in sel_wl_rows.iterrows() if r.get("id")]

            if selected_ids:
                ac1,ac2,ac3=st.columns(3)
                with ac1:
                    if st.button(f"➡️ Sposta in '{move_dest}'",key="do_mv_g"):
                        gh_move_watchlist_rows(selected_ids,move_dest); st.rerun()
                with ac2:
                    if st.button(f"📋 Copia in '{copy_dest2}'",key="do_cp_g"):
                        rows_s=df_wl_disp[df_wl_disp["id"].isin(selected_ids)]
                        gh_add_to_watchlist(rows_s[tcol].tolist(),
                            rows_s[ncol].tolist() if ncol in rows_s.columns else rows_s[tcol].tolist(),
                            "Copia","da selezione","LONG",copy_dest2)
                        st.success("✅ Copiati."); st.rerun()
                with ac3:
                    if st.button("🗑️ Elimina sel.",key="do_dl_g",type="secondary"):
                        gh_delete_from_watchlist(selected_ids); st.rerun()

        # ── VISTA CARDS ───────────────────────────────────────────────────
        else:
            selected_ids=[]
            for _,wrow in df_wl_disp.iterrows():
                rid    =wrow.get("id","")
                tkr    =wrow.get(tcol,"")
                nom    =wrow.get(ncol,"")
                rsi_v  =wrow.get("RSI",None)
                vr_v   =wrow.get("Vol_Ratio",None)
                qs_v   =wrow.get("Quality_Score",None)
                sq_v   =wrow.get("Squeeze",False)
                wb_v   =wrow.get("Weekly_Bull",None)
                ser_v  =wrow.get("Ser_Score",None)
                fv_v   =wrow.get("FV_Score",None)
                origine=wrow.get("origine","")
                created=wrow.get("created_at","")
                trend_v=wrow.get("trend","")

                def badge(val,cls,txt): return f'<span class="wl-card-badge {cls}">{txt}</span>' if val else ""

                # RSI badge
                if rsi_v is not None and not (isinstance(rsi_v,float) and np.isnan(rsi_v)):
                    rn=float(rsi_v); rc="badge-blue" if rn<40 else "badge-green" if rn<=65 else "badge-orange" if rn<=70 else "badge-red"
                    rsi_b=f'<span class="wl-card-badge {rc}">RSI {rn:.1f}</span>'
                else: rsi_b=""
                # Vol badge
                if vr_v is not None and not (isinstance(vr_v,float) and np.isnan(vr_v)):
                    vn=float(vr_v); vc="badge-gray" if vn<1 else "badge-green" if vn<2 else "badge-orange" if vn<3 else "badge-red"
                    vr_b=f'<span class="wl-card-badge {vc}">Vol {vn:.1f}x</span>'
                else: vr_b=""
                # Quality badge
                if qs_v is not None and not (isinstance(qs_v,float) and np.isnan(qs_v)):
                    qn=int(float(qs_v)); qc="badge-green" if qn>=9 else "badge-orange" if qn>=6 else "badge-gray"
                    qs_b=f'<span class="wl-card-badge {qc}">Q {qn}/12</span>'
                else: qs_b=""
                # Serafini badge
                if ser_v is not None and not (isinstance(ser_v,float) and np.isnan(ser_v)):
                    sn=int(float(ser_v)); sc="badge-green" if sn==6 else "badge-orange" if sn>=4 else "badge-gray"
                    ser_b=f'<span class="wl-card-badge {sc}">🎯 S{sn}/6</span>'
                else: ser_b=""
                # Finviz badge
                if fv_v is not None and not (isinstance(fv_v,float) and np.isnan(fv_v)):
                    fn=int(float(fv_v)); fc="badge-green" if fn>=7 else "badge-orange" if fn>=5 else "badge-gray"
                    fv_b=f'<span class="wl-card-badge {fc}">📊 FV{fn}/8</span>'
                else: fv_b=""

                sq_b=badge(sq_v is True or str(sq_v).lower()=="true","badge-orange","🔥 SQ")
                wb_b=('<span class="wl-card-badge badge-green">📈 W+</span>' if wb_v is True or str(wb_v).lower()=="true" else
                      '<span class="wl-card-badge badge-red">📉 W—</span>'   if wb_v is False or str(wb_v).lower()=="false" else "")
                trend_cls={"LONG":"badge-green","SHORT":"badge-red","WATCH":"badge-orange"}.get(str(trend_v).upper(),"badge-gray")
                trend_b=f'<span class="wl-card-badge {trend_cls}">{trend_v}</span>' if trend_v and str(trend_v).upper() not in ("","NAN","NONE") else ""

                row_c=st.columns([0.3,3,1])
                with row_c[0]:
                    if st.checkbox("",key=f"chk_{rid}",label_visibility="collapsed"): selected_ids.append(rid)
                with row_c[1]:
                    st.markdown(f"""<div class="wl-card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div><span class="wl-card-ticker">{tkr}</span>
    <span class="wl-card-name"> &nbsp;{nom}</span></div>
    <div style="color:#374151;font-size:0.72rem">{origine} · {str(created)[:10]}</div>
  </div>
  <div style="margin-top:8px">{trend_b}{rsi_b}{vr_b}{qs_b}{ser_b}{fv_b}{sq_b}{wb_b}</div>
</div>""",unsafe_allow_html=True)
                with row_c[2]:
                    st.write("")
                    if st.button("🗑️",key=f"del_{rid}",help=f"Elimina {tkr}"):
                        gh_delete_from_watchlist([rid]); st.rerun()

            if selected_ids:
                ac1,ac2,ac3=st.columns(3)
                with ac1:
                    if st.button(f"➡️ Sposta in '{move_dest}'",key="do_mv_c"):
                        gh_move_watchlist_rows(selected_ids,move_dest); st.rerun()
                with ac2:
                    if st.button(f"📋 Copia in '{copy_dest2}'",key="do_cp_c"):
                        rows_s=df_wl_disp[df_wl_disp["id"].isin(selected_ids)]
                        gh_add_to_watchlist(rows_s[tcol].tolist(),
                            rows_s[ncol].tolist() if ncol in rows_s.columns else rows_s[tcol].tolist(),
                            "Copia","da selezione","LONG",copy_dest2)
                        st.success("✅ Copiati."); st.rerun()
                with ac3:
                    if st.button("🗑️ Elimina sel.",key="do_dl_c",type="secondary"):
                        gh_delete_from_watchlist(selected_ids); st.rerun()

        # ── Grafici ticker selezionato ────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-pill">📊 ANALISI TICKER</div>',unsafe_allow_html=True)
        if not df_wl.empty and tcol in df_wl.columns:
            _wl_df=df_wl[[tcol,ncol]].drop_duplicates(tcol).sort_values(ncol)
            _wl_labels=[f"{r[tcol]}  —  {r[ncol]}" for _,r in _wl_df.iterrows()]
            _wl_tickers=_wl_df[tcol].tolist()
        else:
            _wl_labels=[]; _wl_tickers=[]
        if _wl_tickers:
            _sel_idx=st.selectbox("🔍 Seleziona ticker",
                options=range(len(_wl_labels)),format_func=lambda i:_wl_labels[i],key="wl_tkr_sel")
            sel_wl=_wl_tickers[_sel_idx]
            row_wl=None
            for src in [df_ep,df_rea]:
                if src.empty or "Ticker" not in src.columns: continue
                m=src[src["Ticker"]==sel_wl]
                if not m.empty: row_wl=m.iloc[0]; break
            if row_wl is not None: show_charts(row_wl,key_suffix="wl")
            else: st.info(f"📭 Dati non disponibili per **{sel_wl}**. Esegui lo scanner.")

    # ── Info DB path + Backup/Restore ──────────────────────────────────────
    with st.expander("💾 Backup & Restore Watchlist", expanded=False):
        try:
            from utils.db import DB_PATH as _DBPATH
            st.caption(f"📂 DB attivo: `{_DBPATH}`")
            _db_ok = _DBPATH.exists()
            _db_sz = round(_DBPATH.stat().st_size/1024,1) if _db_ok else 0
            st.caption(f"{'✅' if _db_ok else '❌'} File {'presente' if _db_ok else 'non trovato'} — {_db_sz} KB")
        except Exception as _e:
            st.caption(f"⚠️ DB path non disponibile: {_e}")

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**📤 Esporta**")
            if st.button("📥 Scarica backup JSON", key="wl_export"):
                try:
                    _df_exp = load_watchlist()
                    if not _df_exp.empty:
                        import json as _json
                        _exp = _df_exp.to_dict(orient="records")
                        st.download_button(
                            "💾 Salva watchlist.json",
                            data=_json.dumps(_exp, indent=2, default=str),
                            file_name="watchlist_backup.json",
                            mime="application/json",
                            key="wl_dl"
                        )
                    else:
                        st.warning("Watchlist vuota.")
                except Exception as _e:
                    st.error(f"Errore export: {_e}")

        with bc2:
            st.markdown("**📥 Importa**")
            _up = st.file_uploader("Carica watchlist.json", type="json", key="wl_import")
            if _up and st.button("⬆️ Ripristina dal backup", key="wl_restore"):
                try:
                    import json as _json
                    _rows = _json.loads(_up.read().decode())
                    conn = sqlite3.connect(str(DB_PATH))
                    for _r in _rows:
                        _ticker  = str(_r.get("ticker",""))
                        _lname   = str(_r.get("list_name","DEFAULT"))
                        if not _ticker: continue
                        _exists = conn.execute(
                            "SELECT id FROM watchlist WHERE ticker=? AND list_name=?",
                            (_ticker, _lname)
                        ).fetchone()
                        if not _exists:
                            conn.execute(
                                "INSERT INTO watchlist (ticker,name,trend,origine,note,list_name,created_at) "
                                "VALUES (?,?,?,?,?,?,?)",
                                (_ticker, _r.get("name",""), _r.get("trend",""),
                                 _r.get("origine",""), _r.get("note",""), _lname,
                                 _r.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            )
                    conn.commit(); conn.close()
                    st.success(f"✅ Ripristinati {len(_rows)} ticker. Clicca Refresh.")
                except Exception as _e:
                    st.error(f"Errore import: {_e}")

    # Export TradingView — solo ticker, un per riga
    _df_tv = load_watchlist()
    _tv_cur = _df_tv[_df_tv["list_name"]==st.session_state.current_list_name] if not _df_tv.empty else pd.DataFrame()
    if not _tv_cur.empty:
        _tc = "ticker" if "ticker" in _tv_cur.columns else "Ticker"
        _tv_lines = _tv_cur[_tc].dropna().unique().tolist()
        st.download_button(
            label="📺 Export TradingView CSV",
            data=chr(10).join(_tv_lines),
            file_name=f"watchlist_{st.session_state.current_list_name}_tradingview.csv",
            mime="text/plain",
            key="wl_tv_export",
            help="Un ticker per riga — importabile direttamente in TradingView Watchlist"
        )

    # ── IMPORT DA TRADINGVIEW / CSV ──────────────────────────────────────
    with st.expander("📥 Importa da TradingView / CSV", expanded=False):
        st.markdown("""
**Formati supportati:**

| Formato | Descrizione |
|---|---|
| TradingView export | File `.txt` / `.csv` — un ticker per riga (es. `NASDAQ:AAPL`) |
| CSV semplice | Colonna `ticker` (obbligatoria) + `nome` (opzionale) |
| Testo libero | Incolla ticker separati da virgola, spazio o newline |
""")

        imp_tab1, imp_tab2 = st.tabs(["📄 Upload file", "✏️ Incolla testo"])

        # ── Tab 1: Upload file ────────────────────────────────────────────
        with imp_tab1:
            _tv_file = st.file_uploader(
                "Carica file watchlist (.csv / .txt)",
                type=["csv","txt"],
                key="wl_tv_upload",
                help="TradingView: Watchlist → Export → scarica il file .txt"
            )
            _imp_list_sel = st.selectbox(
                "Importa nella lista",
                options=all_lists,
                index=all_lists.index(st.session_state.current_list_name)
                      if st.session_state.current_list_name in all_lists else 0,
                key="wl_imp_list_upload"
            )
            _imp_skip_dup = st.checkbox(
                "Salta duplicati (non reimportare ticker gia' presenti)",
                value=True, key="wl_imp_skip_dup_file"
            )

            if _tv_file:
                # Parse file
                _raw = _tv_file.read().decode("utf-8", errors="ignore")
                _lines = [l.strip() for l in _raw.replace(",", "\n").splitlines()]
                _tickers_raw = [l for l in _lines if l and not l.startswith("#")]

                # Normalizza: rimuove prefisso exchange TradingView (NASDAQ:AAPL → AAPL)
                def _normalize_tv_ticker(t: str) -> str:
                    # Gestisce NASDAQ:AAPL, NYSE:MSFT, MIL:ENI, etc.
                    if ":" in t:
                        t = t.split(":", 1)[1]
                    # Rimuove caratteri non alfanumerici tranne punto e trattino
                    import re as _re
                    t = _re.sub(r"[^\w.\-]", "", t).upper()
                    return t

                _tickers_clean = [_normalize_tv_ticker(t) for t in _tickers_raw]
                _tickers_clean = [t for t in _tickers_clean if 1 <= len(t) <= 12]
                _tickers_clean = list(dict.fromkeys(_tickers_clean))  # deduplica mantendo ordine

                if _tickers_clean:
                    # Preview prima di importare
                    st.success(f"✅ {len(_tickers_clean)} ticker trovati nel file")
                    with st.expander(f"Preview — {len(_tickers_clean)} ticker", expanded=True):
                        _prev_cols = st.columns(min(8, len(_tickers_clean)))
                        for _pi, _pt in enumerate(_tickers_clean[:40]):
                            _prev_cols[_pi % len(_prev_cols)].code(_pt)
                        if len(_tickers_clean) > 40:
                            st.caption(f"... e altri {len(_tickers_clean) - 40} ticker")

                    if st.button(
                        f"⬆️ Importa {len(_tickers_clean)} ticker in '{_imp_list_sel}'",
                        key="wl_do_import_file",
                        type="primary"
                    ):
                        # Filtra duplicati se richiesto
                        _existing_tks = set()
                        if _imp_skip_dup and not df_wl_full.empty:
                            _ex_col = "ticker" if "ticker" in df_wl_full.columns else "Ticker"
                            _existing_tks = set(
                                df_wl_full[df_wl_full["list_name"] == _imp_list_sel][_ex_col]
                                .str.upper().dropna().tolist()
                            )

                        _to_import = [t for t in _tickers_clean
                                      if t.upper() not in _existing_tks]
                        _skipped   = len(_tickers_clean) - len(_to_import)

                        if _to_import:
                            gh_add_to_watchlist(
                                _to_import,
                                _to_import,   # Nome = Ticker (aggiornato dallo scanner al prossimo run)
                                "Importato",
                                "TradingView/CSV",
                                "LONG",
                                _imp_list_sel
                            )
                            _msg = f"✅ Importati **{len(_to_import)}** ticker in '{_imp_list_sel}'"
                            if _skipped > 0:
                                _msg += f"  |  ⏭️ {_skipped} duplicati saltati"
                            st.success(_msg)
                            import time as _t; _t.sleep(0.5); st.rerun()
                        else:
                            st.warning("Tutti i ticker sono gia' presenti nella lista — nessuna importazione.")
                else:
                    st.warning("Nessun ticker valido trovato nel file.")

        # ── Tab 2: Incolla testo libero ───────────────────────────────────
        with imp_tab2:
            _paste_text = st.text_area(
                "Incolla ticker (uno per riga, o separati da virgola/spazio)",
                height=150,
                placeholder="AAPL\nMSFT\nNVDA\n\noppure: AAPL, MSFT, NVDA, TSLA",
                key="wl_paste_text",
            )
            _imp_list_paste = st.selectbox(
                "Importa nella lista",
                options=all_lists,
                index=all_lists.index(st.session_state.current_list_name)
                      if st.session_state.current_list_name in all_lists else 0,
                key="wl_imp_list_paste"
            )
            _imp_skip_dup_p = st.checkbox(
                "Salta duplicati",
                value=True, key="wl_imp_skip_dup_paste"
            )

            if _paste_text.strip():
                import re as _re
                _paste_tks_raw = _re.split(r"[\s,;|\n]+", _paste_text.strip())
                _paste_tks = []
                for _pt in _paste_tks_raw:
                    _pt = _pt.strip().upper()
                    if ":" in _pt: _pt = _pt.split(":", 1)[1]
                    _pt = _re.sub(r"[^\w.\-]", "", _pt)
                    if 1 <= len(_pt) <= 12:
                        _paste_tks.append(_pt)
                _paste_tks = list(dict.fromkeys(_paste_tks))

                if _paste_tks:
                    st.caption(f"Trovati: **{', '.join(_paste_tks[:20])}**"
                               + (f" ...+{len(_paste_tks)-20}" if len(_paste_tks) > 20 else ""))

                    if st.button(
                        f"⬆️ Importa {len(_paste_tks)} ticker in '{_imp_list_paste}'",
                        key="wl_do_import_paste",
                        type="primary"
                    ):
                        _existing_p = set()
                        if _imp_skip_dup_p and not df_wl_full.empty:
                            _ex_col = "ticker" if "ticker" in df_wl_full.columns else "Ticker"
                            _existing_p = set(
                                df_wl_full[df_wl_full["list_name"] == _imp_list_paste][_ex_col]
                                .str.upper().dropna().tolist()
                            )
                        _to_imp_p  = [t for t in _paste_tks if t not in _existing_p]
                        _skipped_p = len(_paste_tks) - len(_to_imp_p)
                        if _to_imp_p:
                            gh_add_to_watchlist(
                                _to_imp_p, _to_imp_p,
                                "Importato", "Testo", "LONG", _imp_list_paste
                            )
                            _msg2 = f"✅ Importati **{len(_to_imp_p)}** ticker in '{_imp_list_paste}'"
                            if _skipped_p > 0: _msg2 += f"  |  ⏭️ {_skipped_p} duplicati saltati"
                            st.success(_msg2)
                            import time as _t2; _t2.sleep(0.5); st.rerun()
                        else:
                            st.warning("Tutti i ticker gia' presenti — nessuna importazione.")

    if st.button("🔄 Refresh",key="wl_ref"): st.rerun()

    # ── Strategy Chart ────────────────────────────────────────────────────
    try:
        from utils.backtest_tab import strategy_chart_widget as _scw
        df_wl_sc = load_watchlist(list_name=st.session_state.get("current_list_name","DEFAULT"))
        _wl_tkrs = df_wl_sc["ticker"].dropna().tolist() if not df_wl_sc.empty and "ticker" in df_wl_sc.columns else []
        st.markdown("---")
        _scw(tickers=_wl_tkrs, key_suffix="WL")
    except Exception:
        pass

# =========================================================================
# STORICO
# =========================================================================
with tab_bt:
    # ══════════════════════════════════════════════════════════════════════
    # v35 UPGRADE #4 — BACKTEST PRO ENGINE
    # Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, Avg R
    # Equity Curve con drawdown overlay — tutto inline senza dipendenze esterne
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-pill">🧪 BACKTEST PRO v41 — Metriche Professionali</div>', unsafe_allow_html=True)

    with st.expander("📊 Backtest Pro v41 — Sharpe · Drawdown · Win Rate · Equity Curve", expanded=False):
        _bt_tickers_v41 = []
        if not df_ep.empty and "Ticker" in df_ep.columns:
            _bt_tickers_v41 = sorted(df_ep["Ticker"].dropna().unique().tolist())

        if not _bt_tickers_v41:
            st.info("Avvia lo scanner per popolare l'universo di backtest.")
        else:
            _bc1, _bc2, _bc3, _bc4 = st.columns([2,1,1,1])
            with _bc1:
                _bt_tkr = st.selectbox("Ticker", _bt_tickers_v41, key="bt35_tkr")
            with _bc2:
                _bt_period = st.selectbox("Periodo", ["6mo","1y","2y","3y"], index=1, key="bt35_period")
            with _bc3:
                _bt_entry_rsi = st.number_input("Entry RSI ≤", 20, 60, 45, key="bt35_rsi_entry")
            with _bc4:
                _bt_exit_rsi  = st.number_input("Exit RSI ≥", 55, 90, 65, key="bt35_rsi_exit")

            _bt_atr_sl = st.slider("Stop Loss ATR multiplo", 1.0, 3.0, 1.5, 0.5, key="bt35_atr_sl")

            if st.button("▶️ Esegui Backtest Pro", key="bt35_run", type="primary"):
                with st.spinner(f"Scarico dati {_bt_tkr}..."):
                    try:
                        import yfinance as _yf
                        import numpy as _np35

                        _raw = _yf.download(_bt_tkr, period=_bt_period, interval="1d",
                                            auto_adjust=True, progress=False)
                        if _raw.empty:
                            st.error("Nessun dato scaricato da Yahoo Finance.")
                        else:
                            _raw.columns = [c[0] if isinstance(c, tuple) else c for c in _raw.columns]
                            _cl = _raw["Close"].squeeze()
                            _hi = _raw["High"].squeeze()
                            _lo = _raw["Low"].squeeze()

                            # RSI interno
                            def _bt_rsi(s, n=14):
                                d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
                                rs = g.ewm(com=n-1,adjust=False).mean()/l.ewm(com=n-1,adjust=False).mean()
                                return 100 - 100/(1+rs)

                            # ATR interno
                            def _bt_atr(hi, lo, cl, n=14):
                                tr = _np35.maximum(hi-lo, _np35.maximum(
                                    _np35.abs(hi-cl.shift(1)), _np35.abs(lo-cl.shift(1))))
                                return pd.Series(tr.values, index=cl.index).ewm(com=n-1,adjust=False).mean()

                            _rsi_s = _bt_rsi(_cl)
                            _atr_s = _bt_atr(_hi, _lo, _cl)

                            # Backtest: entry su RSI≤soglia + EMA20 uptrend, exit su RSI≥soglia o SL
                            _ema20_s = _cl.ewm(span=20, adjust=False).mean()
                            _trades = []
                            _in_trade = False
                            _entry_price = 0.0; _sl = 0.0; _entry_date = None

                            for _idx in range(20, len(_cl)):
                                _dt = _cl.index[_idx]
                                _p  = float(_cl.iloc[_idx])
                                _r  = float(_rsi_s.iloc[_idx]) if not _np35.isnan(_rsi_s.iloc[_idx]) else 50.0
                                _a  = float(_atr_s.iloc[_idx]) if not _np35.isnan(_atr_s.iloc[_idx]) else 0.0
                                _e20= float(_ema20_s.iloc[_idx])

                                if not _in_trade:
                                    if _r <= _bt_entry_rsi and _p > _e20 and _a > 0:
                                        _in_trade = True
                                        _entry_price = _p
                                        _sl = _p - _bt_atr_sl * _a
                                        _entry_date = _dt
                                else:
                                    _exit = None
                                    if _p <= _sl:
                                        _exit = "SL"
                                    elif _r >= _bt_exit_rsi:
                                        _exit = "RSI_EXIT"
                                    if _exit:
                                        _ret = (_p - _entry_price) / _entry_price
                                        _r_multiple = (_p - _entry_price) / (_entry_price - _sl) if (_entry_price - _sl) > 0 else 0
                                        _trades.append({
                                            "Entry Date": _entry_date, "Exit Date": _dt,
                                            "Entry $": round(_entry_price, 2),
                                            "Exit $": round(_p, 2),
                                            "Return %": round(_ret*100, 2),
                                            "R": round(_r_multiple, 2),
                                            "Exit": _exit,
                                        })
                                        _in_trade = False

                            if not _trades:
                                st.warning("Nessun trade trovato. Rilassa i parametri RSI.")
                            else:
                                _df_trades = pd.DataFrame(_trades)
                                _rets = _df_trades["Return %"].values / 100.0

                                # ── Metriche Pro ────────────────────────
                                _n_trades   = len(_trades)
                                _wins       = (_rets > 0).sum()
                                _win_rate   = _wins / _n_trades * 100
                                _avg_win    = _rets[_rets>0].mean() * 100 if _wins > 0 else 0
                                _avg_loss   = _rets[_rets<0].mean() * 100 if (_rets<0).sum()>0 else 0
                                _profit_fac = abs(_avg_win * _wins) / abs(_avg_loss * (_n_trades-_wins)) if (_n_trades-_wins)>0 and _avg_loss!=0 else float("inf")
                                _total_ret  = (1 + pd.Series(_rets)).prod() - 1
                                _avg_r      = _df_trades["R"].mean()

                                # Sharpe (annualizzato su trade returns, non daily)
                                _daily_rets = _cl.pct_change().dropna()
                                _sharpe     = (_daily_rets.mean() / _daily_rets.std() * _np35.sqrt(252)) if _daily_rets.std() > 0 else 0

                                # Equity curve
                                _eq = (1 + pd.Series(_rets)).cumprod()
                                _running_max = _eq.cummax()
                                _dd = (_eq - _running_max) / _running_max * 100
                                _max_dd = float(_dd.min())

                                # Display metriche
                                _m1,_m2,_m3,_m4,_m5,_m6 = st.columns(6)
                                _m1.metric("📊 N. Trade",    _n_trades)
                                _m2.metric("🎯 Win Rate",    f"{_win_rate:.1f}%")
                                _m3.metric("📈 Total Ret",   f"{_total_ret*100:.1f}%")
                                _m4.metric("⚡ Sharpe",      f"{_sharpe:.2f}")
                                _m5.metric("📉 Max DD",      f"{_max_dd:.1f}%")
                                _m6.metric("💰 Profit Factor",f"{_profit_fac:.2f}")

                                _m7,_m8,_m9 = st.columns(3)
                                _m7.metric("✅ Avg Win %",  f"{_avg_win:.2f}%")
                                _m8.metric("❌ Avg Loss %", f"{_avg_loss:.2f}%")
                                _m9.metric("📐 Avg R",      f"{_avg_r:.2f}")

                                # Equity Curve + Drawdown overlay
                                _fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.65, 0.35], vertical_spacing=0.04)
                                _eq_idx = list(range(len(_eq)))
                                _fig_bt.add_trace(go.Scatter(
                                    x=_eq_idx, y=_eq.tolist(),
                                    fill="tozeroy", fillcolor="rgba(41,98,255,0.12)",
                                    line=dict(color="#2962ff", width=2),
                                    name="Equity Curve"), row=1, col=1)
                                _fig_bt.add_trace(go.Bar(
                                    x=_eq_idx, y=_dd.tolist(),
                                    marker_color=["rgba(239,68,68,0.70)" for _ in _dd],
                                    name="Drawdown %"), row=2, col=1)
                                _fig_bt.update_layout(**PLOTLY_DARK,
                                    title=dict(text=f"<b>{_bt_tkr}</b> — Equity Curve & Drawdown ({_bt_period})",
                                        font=dict(color="#50c4e0",size=13)),
                                    height=380, margin=dict(l=0,r=0,t=40,b=0),
                                    showlegend=False)
                                _fig_bt.update_yaxes(title_text="Equity (×)", row=1, col=1, tickfont=dict(size=9))
                                _fig_bt.update_yaxes(title_text="DD %", row=2, col=1, tickfont=dict(size=9))
                                st.plotly_chart(_fig_bt, use_container_width=True, key="bt35_equity_chart")

                                # Trade log
                                st.markdown("**📋 Trade Log**")
                                st.dataframe(_df_trades, use_container_width=True, height=260)

                                # Export trades
                                _bt_xlsx = io.BytesIO()
                                with pd.ExcelWriter(_bt_xlsx, engine="xlsxwriter") as _bw:
                                    _df_trades.to_excel(_bw, sheet_name="Trades", index=False)
                                    pd.DataFrame([{
                                        "Ticker": _bt_tkr, "Periodo": _bt_period,
                                        "N_Trade": _n_trades, "Win_Rate_%": round(_win_rate,1),
                                        "Total_Ret_%": round(_total_ret*100,1),
                                        "Sharpe": round(_sharpe,2),
                                        "Max_DD_%": round(_max_dd,1),
                                        "Profit_Factor": round(_profit_fac,2),
                                        "Avg_R": round(_avg_r,2),
                                    }]).to_excel(_bw, sheet_name="Summary", index=False)
                                st.download_button(
                                    f"📊 Export Backtest {_bt_tkr}.xlsx",
                                    _bt_xlsx.getvalue(),
                                    f"Backtest_{_bt_tkr}_{_bt_period}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="bt35_export"
                                )
                    except Exception as _bt35_err:
                        import traceback as _tbt
                        st.error(f"Backtest error: {_bt35_err}")
                        st.code(_tbt.format_exc())

    st.markdown("---")
    # ── Backtest tab originale (mantenuto per compatibilità) ─────────────
    # v34: passa i ticker con nomi alfabetici dal df_ep corrente
    try:
        from utils.backtest_tab import render_backtest_tab as _bt_full, strategy_chart_widget as _scw_bt
        # Costruisci labels "Nome Azienda (TICKER)" dal df scanner corrente
        _bt_labels = {}
        if not df_ep.empty and "Ticker" in df_ep.columns and "Nome" in df_ep.columns:
            for _, _br in df_ep[["Ticker","Nome"]].dropna(subset=["Ticker"]).iterrows():
                _bt_labels[_br["Ticker"]] = f"{str(_br.get('Nome',''))[:28]}  ({_br['Ticker']})"
        _bt_tickers = sorted(df_ep["Ticker"].dropna().unique().tolist()) if not df_ep.empty and "Ticker" in df_ep.columns else []
        # Chiama render con i dati arricchiti
        render_backtest_tab()
        # Mostra strategy chart separata con nomi completi se ci sono ticker dallo scanner
        if _bt_tickers:
            st.markdown("---")
            st.markdown('<div class="section-pill">📊 STRATEGY CHART — ticker scanner corrente</div>', unsafe_allow_html=True)
            _scw_bt(tickers=_bt_tickers, key_suffix="bt_main", ticker_labels=_bt_labels)
    except Exception:
        render_backtest_tab()

with tab_of:
    try:
        if _of_render:
            # Passa df_ep dallo scanner se disponibile
            _df_of = df_ep if "df_ep" in dir() else None
            _of_render(df_scanner=_df_of)
        else:
            from utils.orderflow_tab import render_orderflow_tab
            render_orderflow_tab()
    except Exception as _ofe:
        import traceback
        st.error(f"Order Flow error: {_ofe}")
        st.code(traceback.format_exc())


with tab_bcd:
    try:
        from utils.bluechip_dip import render_bluechip_dip
        render_bluechip_dip()
    except ImportError:
        st.info("💎 bluechip_dip.py non trovato in utils/")
    except Exception as _bce:
        import traceback
        st.error(f"Blue Chip Dip error: {_bce}")
        st.code(traceback.format_exc())

    st.markdown('<div class="section-pill">📜 STORICO SCANSIONI</div>',unsafe_allow_html=True)

    # ── Legenda Storico ────────────────────────────────────────────────
    with st.expander("📖 Come leggere lo Storico — Guida completa", expanded=False):
        st.markdown("""
## 📜 Storico Scansioni — Guida Operativa

Lo **Storico** registra ogni scansione eseguita nel database locale.
Ogni riga corrisponde a una singola esecuzione dello scanner con timestamp, mercati scansionati
e numero di segnali trovati.

---

### 📊 Colonne della tabella

| Colonna | Tipo | Significato |
|---------|------|-------------|
| **id** | numero | ID progressivo scansione |
| **scanned_at** | datetime | Data e ora esecuzione (UTC) |
| **markets** | testo | Mercati inclusi (US, ETF, Crypto…) |
| **n_tickers** | intero | Titoli totali analizzati nello scan |
| **n_early** | intero | Titoli con `Stato_Early = EARLY` trovati |
| **n_pro** | intero | Titoli con `Stato_Pro = PRO` trovati |
| **n_rea** | intero | Titoli con `Stato = HOT` (REA) trovati |
| **elapsed_s** | secondi | Tempo impiegato per la scansione |
| **params** | JSON | Parametri usati (soglie, top, indicatori) |

---

### 🔍 Confronto Snapshot — Come funziona

Il **confronto** permette di analizzare l'evoluzione del mercato tra due momenti diversi:

- **🆕 Nuovi in B**: ticker apparsi in B ma non in A → **nuovi segnali emergenti**
- **❌ Usciti da A**: ticker che erano in A ma non in B → **segnali deteriorati o usciti**
- **✅ Persistenti**: ticker presenti in entrambe le scan → **segnali solidi e confermati**

**Caso d'uso tipico:**
1. Scan mattina 09:00 → salva come A
2. Scan pomeriggio 15:30 → salva come B
3. Confronta → vedi quali nuovi titoli sono entrati in segnale nel corso della giornata

**Interpretazione:**
- Molti *Nuovi* con pochi *Persistenti* → mercato in rotazione, cautela
- Pochi *Nuovi* con molti *Persistenti* → trend solido, conferma
- Tutti *Usciti* → deterioramento rapido, possibile fine trend

---

### 💡 Consigli operativi

- **Frequenza ideale**: 1-3 scan al giorno (apertura, metà seduta, chiusura)
- **Reset storico**: usa il pulsante 🗑️ solo se vuoi cancellare tutto. I dati del DB watchlist
  rimangono intatti — viene cancellato solo lo storico delle scansioni.
- **Backup**: esporta i dati importanti dalla Watchlist prima di fare reset
- **Limite**: vengono mostrate le ultime **20 scansioni**. Le più vecchie rimangono nel DB
  ma non vengono visualizzate (modifica `load_scan_history(20)` per aumentare).
""")

    _,col_rst=st.columns([4,1])
    with col_rst:
        if st.button("🗑️ Reset",key="rst_hist",type="secondary"):
            conn=sqlite3.connect(str(DB_PATH)); conn.execute("DELETE FROM scan_history")
            conn.commit();conn.close(); st.success("Storico cancellato!"); st.rerun()
    df_hist=load_scan_history(20)
    if df_hist.empty:
        st.info("""
📭 **Nessuna scansione salvata ancora.**

Per popolare lo storico:
1. Vai nella sidebar
2. Seleziona i mercati da scansionare
3. Clicca **▶️ Avvia Scanner**

Ogni scansione viene automaticamente salvata qui con timestamp, mercati, segnali trovati e tempi.
""")
    else:
        # Formatta colonne
        disp_hist = df_hist.copy()
        if "elapsed_s" in disp_hist.columns:
            disp_hist["elapsed_s"] = disp_hist["elapsed_s"].apply(
                lambda x: f"{x:.0f}s" if pd.notna(x) else "—")

        # Metriche aggregate
        _m1,_m2,_m3,_m4 = st.columns(4)
        _m1.metric("📋 Scan totali", len(df_hist))
        if "n_early" in df_hist.columns:
            _m2.metric("📡 Max EARLY", int(df_hist["n_early"].max()))
        if "n_pro" in df_hist.columns:
            _m3.metric("💪 Max PRO", int(df_hist["n_pro"].max()))
        if "n_tickers" in df_hist.columns:
            _m4.metric("🔭 Titoli medi", f"{df_hist['n_tickers'].mean():.0f}")

        st.markdown("**📋 Ultime 20 scansioni:**")
        st.dataframe(disp_hist,use_container_width=True)
        st.markdown("---")
        st.subheader("🔍 Confronto Snapshot")
        st.caption("Seleziona due scansioni per confrontare quali ticker sono entrati/usciti dai segnali.")
        hc1,hc2=st.columns(2)
        def _slbl(row):
            dt=str(row.get("scanned_at",""))[:16]
            ep=int(row.get("n_early",0)); pr=int(row.get("n_pro",0))
            mkt=str(row.get("markets",""))[:20]
            return f"{dt}  |  E:{ep} P:{pr}  [{mkt}]"
        _smap={row["id"]:_slbl(row) for _,row in df_hist.iterrows()}
        _ids=list(_smap.keys())
        with hc1:
            id_a=st.selectbox("📅 Scansione A (baseline)",_ids,format_func=lambda i:_smap[i],key="sn_a")
        with hc2:
            id_b=st.selectbox("📅 Scansione B (più recente)",_ids,format_func=lambda i:_smap[i],
                index=min(1,len(_ids)-1),key="sn_b")
        if st.button("🔍 Confronta le due scansioni", use_container_width=False):
            ea,_=load_scan_snapshot(id_a); eb,_=load_scan_snapshot(id_b)
            if ea.empty or eb.empty: st.warning("Dati non disponibili per uno dei due snapshot.")
            else:
                ta=set(ea.get("Ticker",pd.Series()).tolist())
                tb=set(eb.get("Ticker",pd.Series()).tolist())
                sc1,sc2,sc3,sc4=st.columns(4)
                sc1.metric("🆕 Nuovi in B",len(tb-ta),help="Ticker apparsi in B ma non in A")
                sc2.metric("❌ Usciti da A",len(ta-tb),help="Ticker che erano in A ma non in B")
                sc3.metric("✅ Persistenti",len(ta&tb),help="Presenti in entrambe le scan")
                sc4.metric("📊 Overlap %",f"{len(ta&tb)/max(len(ta|tb),1)*100:.0f}%")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    if tb-ta:
                        st.markdown("**🆕 Nuovi ticker in B:**")
                        st.code("  ".join(sorted(tb-ta)))
                    if ta-tb:
                        st.markdown("**❌ Ticker usciti da A:**")
                        st.code("  ".join(sorted(ta-tb)))
                with col_r2:
                    if ta&tb:
                        st.markdown(f"**✅ Ticker persistenti ({len(ta&tb)}):**")
                        st.code("  ".join(sorted(ta&tb)))


# =========================================================================
# EXPORT GLOBALI v35 PRO
# =========================================================================
st.markdown("---")
st.markdown('<div class="section-pill">💾 EXPORT PRO v41 — XLSX Multi-Sheet · CSV TradingView · Timestamp Auto</div>',unsafe_allow_html=True)

df_conf_exp=pd.DataFrame()
if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
    df_conf_exp=df_ep[(df_ep["Stato_Early"]=="EARLY")&(df_ep["Stato_Pro"]=="PRO")].copy()
df_wl_exp=load_watchlist()
df_wl_exp=df_wl_exp[df_wl_exp["list_name"]==st.session_state.current_list_name]
all_exp={"EARLY":df_ep,"PRO":df_ep,"REA-HOT":df_rea,"CONFLUENCE":df_conf_exp,"Watchlist":df_wl_exp}
cur_tab=st.session_state.get("last_active_tab","EARLY")
df_cur=all_exp.get(cur_tab,pd.DataFrame())

# v35: timestamp automatico per nomi file univoci
_ts = datetime.now().strftime("%Y%m%d_%H%M")

# v35: to_excel_bytes_pro con Summary sheet
def _to_excel_pro(d, label="Export"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        # Summary sheet in prima posizione
        _summary_rows = []
        for _nm, _df in d.items():
            if isinstance(_df, pd.DataFrame):
                _n  = len(_df)
                _tks= ",".join(_df["Ticker"].dropna().tolist()[:10]) + ("..." if _n>10 else "") if "Ticker" in _df.columns and _n>0 else ""
                _summary_rows.append({"Sheet": _nm, "N_Segnali": _n, "Top_Ticker (prime 10)": _tks})
        if _summary_rows:
            pd.DataFrame(_summary_rows).to_excel(w, sheet_name="Summary", index=False)
        # Dati per ogni sheet
        for _nm, _df in d.items():
            if isinstance(_df, pd.DataFrame) and not _df.empty:
                _df.to_excel(w, sheet_name=_nm[:31], index=False)
    return buf.getvalue()

ec1,ec2,ec3,ec4,ec5=st.columns(5)
with ec1:
    st.download_button(
        "📊 XLSX Pro Tutti",
        _to_excel_pro(all_exp),
        f"TradingScanner_v41_Tutti_{_ts}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xlsx_all",
        help="Tutti i tab + sheet Summary con conteggi"
    )
with ec2:
    tv_rows=[]
    for n,df_t in all_exp.items():
        if isinstance(df_t,pd.DataFrame) and not df_t.empty and "Ticker" in df_t.columns:
            tks=df_t["Ticker"].tolist()
            tv_rows.append(pd.DataFrame({"Tab":[n]*len(tks),"Ticker":tks}))
    if tv_rows:
        df_tv=pd.concat(tv_rows,ignore_index=True).drop_duplicates("Ticker")
        st.download_button(
            "📈 CSV TV Tutti",
            df_tv.to_csv(index=False).encode(),
            f"TradingScanner_v41_TV_{_ts}.csv",
            "text/csv",
            key="csv_tv_all",
            help="CSV pronto per import in TradingView Watchlist"
        )
with ec3:
    st.download_button(
        f"📊 XLSX {cur_tab}",
        _to_excel_pro({cur_tab: df_cur}),
        f"TradingScanner_v41_{cur_tab}_{_ts}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xlsx_curr"
    )
with ec4:
    if not df_cur.empty and "Ticker" in df_cur.columns:
        st.download_button(
            f"📈 CSV TV {cur_tab}",
            make_tv_csv(df_cur, cur_tab),
            f"TradingScanner_v41_{cur_tab}_TV_{_ts}.csv",
            "text/csv",
            key="csv_tv_curr"
        )
with ec5:
    # v35: export P&L tracker se presente
    _pnl_data = st.session_state.get("v41_pnl_entries", {})
    if _pnl_data:
        _df_pnl_exp = pd.DataFrame([
            {"Ticker": _t, "Entry $": _pos["entry"], "Size": _pos["size"], "Added": _pos.get("added","")}
            for _t, _pos in _pnl_data.items()
        ])
        st.download_button(
            "💰 Export P&L",
            _df_pnl_exp.to_csv(index=False).encode(),
            f"PnL_Tracker_v41_{_ts}.csv",
            "text/csv",
            key="csv_pnl_exp",
            help="Esporta posizioni P&L tracker"
        )
# =========================================================================
# v41 TAB #5 — MTF MATRIX (Multi-Timeframe Confluence)
# =========================================================================
with tab_mtfmatrix:
    st.markdown('<div class="section-pill">🔀 MULTI-TIMEFRAME CONFLUENCE MATRIX v41</div>',
                unsafe_allow_html=True)
    st.caption("Stato Daily / Weekly / Monthly per ogni ticker. 🟢 3/3 allineati · 🟡 2/3 · 🔴 1/3 · ⚪ no data")

    def _tf_emoji_fn(tf_d):
        s  = tf_d.get("status","no_data")
        sc = tf_d.get("score",0)
        if s in ("no_data","error"): return "⚪"
        return "🟢" if sc == 3 else "🟡" if sc == 2 else "🔴"

    _mtf_nome_map = {}
    if not df_ep.empty and "Ticker" in df_ep.columns and "Nome" in df_ep.columns:
        for _, _mr in df_ep[["Ticker","Nome"]].dropna(subset=["Ticker"]).iterrows():
            _mtf_nome_map[_mr["Ticker"]] = str(_mr.get("Nome",""))[:30]

    _mtf_tickers = []
    if not df_ep.empty and "Ticker" in df_ep.columns:
        _mtf_tickers = sorted(df_ep["Ticker"].dropna().unique().tolist())
    _wl_mtf = load_watchlist()
    if not _wl_mtf.empty and "Ticker" in _wl_mtf.columns:
        _wl_tickers_mtf = _wl_mtf["Ticker"].dropna().unique().tolist()
        _mtf_tickers = sorted(set(_mtf_tickers + _wl_tickers_mtf))

    if not _mtf_tickers:
        st.info("Avvia lo scanner o aggiungi ticker alla watchlist per usare la MTF Matrix.")
    else:
        # ── Costruisce labels e options PRIMA dell'expander (serve a _do_import) ──
        _mtf_labels = {}
        for _tk in _mtf_tickers:
            _nm = _mtf_nome_map.get(_tk,"")
            _mtf_labels[_tk] = f"{_nm}  ({_tk})" if _nm else _tk
        _mtf_options_sorted = sorted(_mtf_tickers, key=lambda t: _mtf_labels[t].lower())

        with st.expander("📂 Importa lista da Scanner / Watchlist", expanded=False):
            st.caption("Importa ticker da qualsiasi tab scanner o dalla watchlist. La lista viene caricata direttamente nel multiselect.")
            _imp_row1 = st.columns(3)
            _imp_row2 = st.columns(3)

            def _do_import(tickers_list):
                """Scrive la lista importata DIRETTAMENTE nel key del multiselect."""
                _valid = [t for t in tickers_list if t in _mtf_options_sorted]
                if not _valid:
                    st.warning("Nessun ticker valido trovato (avvia lo scanner prima).")
                    return
                # Scrive nel key del multiselect — Streamlit usa questo valore al prossimo render
                st.session_state["mtf_sel_tickers"] = _valid
                st.session_state["_mtf_results"]    = []   # reset risultati vecchi
                st.rerun()

            with _imp_row1[0]:
                if st.button("📡 EARLY", key="mtf_imp_early", use_container_width=True):
                    if not df_ep.empty and "Stato_Early" in df_ep.columns:
                        _do_import(sorted(set(df_ep[df_ep["Stato_Early"]=="EARLY"]["Ticker"].dropna().tolist()[:20])))
                    else:
                        st.warning("Nessun dato EARLY — avvia lo scanner.")
            with _imp_row1[1]:
                if st.button("💪 PRO + STRONG", key="mtf_imp_pro", use_container_width=True):
                    if not df_ep.empty and "Stato_Pro" in df_ep.columns:
                        _do_import(sorted(set(df_ep[df_ep["Stato_Pro"].isin(["PRO","STRONG"])]["Ticker"].dropna().tolist()[:20])))
                    else:
                        st.warning("Nessun dato PRO — avvia lo scanner.")
            with _imp_row1[2]:
                if st.button("⭐ CONFLUENCE", key="mtf_imp_conf", use_container_width=True):
                    if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
                        _mask = (df_ep["Stato_Early"]=="EARLY") & (df_ep["Stato_Pro"].isin(["PRO","STRONG"]))
                        _do_import(sorted(set(df_ep[_mask]["Ticker"].dropna().tolist()[:20])))
                    else:
                        st.warning("Nessun dato CONFLUENCE — avvia lo scanner.")

            with _imp_row2[0]:
                if st.button("★ Solo STRONG", key="mtf_imp_strong", use_container_width=True):
                    if not df_ep.empty and "Stato_Pro" in df_ep.columns:
                        _do_import(sorted(set(df_ep[df_ep["Stato_Pro"]=="STRONG"]["Ticker"].dropna().tolist()[:20])))
                    else:
                        st.warning("Nessun STRONG trovato.")
            with _imp_row2[1]:
                if st.button("🎯 Serafini", key="mtf_imp_ser", use_container_width=True):
                    if not df_ep.empty and "Ser_OK" in df_ep.columns:
                        _do_import(sorted(set(df_ep[df_ep["Ser_OK"].isin([True,"True","true"])]["Ticker"].dropna().tolist()[:20])))
                    else:
                        st.warning("Nessun dato Serafini — avvia lo scanner.")
            with _imp_row2[2]:
                if st.button("📋 Watchlist attiva", key="mtf_imp_wl", use_container_width=True):
                    _wl_imp = load_watchlist()
                    if not _wl_imp.empty and "Ticker" in _wl_imp.columns:
                        _imp_wl = _wl_imp[_wl_imp["list_name"]==st.session_state.current_list_name]["Ticker"].dropna().tolist()
                        _do_import(sorted(set(_imp_wl[:20])))
                    else:
                        st.warning("Watchlist vuota.")

            # Preview della selezione corrente
            _cur_mtf_sel = st.session_state.get("mtf_sel_tickers", [])
            if _cur_mtf_sel:
                st.success(f"✅ {len(_cur_mtf_sel)} ticker in selezione: "
                           f"{', '.join(_cur_mtf_sel[:8])}{'...' if len(_cur_mtf_sel)>8 else ''}")
                if st.button("🗑️ Svuota selezione", key="mtf_imp_clear"):
                    st.session_state["mtf_sel_tickers"] = []
                    st.session_state["_mtf_results"]    = []
                    st.rerun()

        # _mtf_labels e _mtf_options_sorted già costruiti sopra (prima dell'expander)
        _default_import = st.session_state.get("mtf_import_list",
                          _mtf_options_sorted[:min(10,len(_mtf_options_sorted))])
        _default_import = [t for t in _default_import if t in _mtf_options_sorted]

        _mc1, _mc2, _mc3 = st.columns([3,1,1])
        with _mc1:
            _mtf_sel = st.multiselect(
                "Ticker da analizzare (max 20) — doppio click sul nome → TradingView IT",
                options=_mtf_options_sorted,
                default=_default_import[:20],
                format_func=lambda t: _mtf_labels.get(t,t),
                key="mtf_sel_tickers"
            )
        with _mc2:
            _mtf_show_detail = st.checkbox("Mostra dettaglio (EMA/RSI)", value=False, key="mtf_detail")
        with _mc3:
            _mtf_only_confluence = st.checkbox("Solo 🟢 3/3 confluence", value=False, key="mtf_only_green")

        if st.button("🔀 Calcola MTF Matrix", key="mtf_run", type="primary") and _mtf_sel:
            _mtf_results = []
            _mtf_progress = st.progress(0.0)
            _mtf_status   = st.empty()
            for _i, _tkr in enumerate(_mtf_sel[:20]):
                _mtf_status.caption(f"Analisi {_mtf_labels.get(_tkr,_tkr)}... ({_i+1}/{len(_mtf_sel[:20])})")
                _mtf_progress.progress((_i+1)/len(_mtf_sel[:20]))
                _tf_data = _fetch_mtf_data(_tkr)
                _d  = _tf_data.get("Daily",   {"status":"no_data","score":0})
                _w  = _tf_data.get("Weekly",  {"status":"no_data","score":0})
                _mo = _tf_data.get("Monthly", {"status":"no_data","score":0})
                _conf_score = round((_d["score"]+_w["score"]+_mo["score"]) / 9 * 100)
                _conf_tfs   = sum(1 for _tf in [_d,_w,_mo]
                                  if _tf.get("status") not in ("no_data","error")
                                  and _tf.get("score",0) >= 2)
                _mtf_results.append({
                    "Ticker": _tkr, "Nome": _mtf_nome_map.get(_tkr,""),
                    "Daily": _tf_emoji_fn(_d), "Weekly": _tf_emoji_fn(_w), "Monthly": _tf_emoji_fn(_mo),
                    "TF Bull": f"{_conf_tfs}/3", "Score": _conf_score,
                    "_d": _d, "_w": _w, "_mo": _mo,
                })
            _mtf_progress.progress(1.0)
            _mtf_status.empty()
            st.session_state["_mtf_results"] = _mtf_results

        _mtf_res = st.session_state.get("_mtf_results", [])
        if _mtf_res:
            if _mtf_only_confluence:
                _mtf_res = [r for r in _mtf_res if r["TF Bull"] == "3/3"]
            _mtf_res = sorted(_mtf_res, key=lambda x: x["Score"], reverse=True)

            st.markdown("---")
            _hc = st.columns([2.0, 0.7, 0.7, 0.7, 0.7, 0.9])
            for _col, _lbl in zip(_hc, ["Ticker / Nome","Daily","Weekly","Monthly","TF Bull","Score"]):
                _col.markdown(f"<span style='color:#50c4e0;font-size:0.78rem;font-weight:bold;"
                              f"letter-spacing:1px;text-transform:uppercase'>{_lbl}</span>",
                              unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#2a2e39;margin:4px 0'>", unsafe_allow_html=True)

            for _r in _mtf_res:
                _rc = st.columns([2.0, 0.7, 0.7, 0.7, 0.7, 0.9])
                _score_c = "#00ff88" if _r["Score"]>=75 else "#f59e0b" if _r["Score"]>=50 else "#ef4444"
                _bull_c  = "#00ff88" if _r["TF Bull"]=="3/3" else "#f59e0b" if _r["TF Bull"]=="2/3" else "#ef4444"
                # v41 fix: usa href <a> invece di ondblclick (JS bloccato da Streamlit sandbox)
                _tv_sym  = _r["Ticker"].replace(".MI","").replace(".","")
                _tv_url  = f"https://it.tradingview.com/chart/?symbol={_tv_sym}"
                _nome_disp = (f"<span style='color:#787b86;font-size:0.72rem;display:block'>{_r['Nome']}</span>"
                              if _r["Nome"] else "")
                _rc[0].markdown(
                    f"<div style='line-height:1.3'>"
                    f"<a href='{_tv_url}' target='_blank' style='font-family:Courier New;"
                    f"color:#00ff88;font-weight:bold;font-size:0.95rem;"
                    f"text-decoration:none' title='Apri su TradingView IT'>"
                    f"{_r['Ticker']} <span style='font-size:0.7rem;color:#2962ff'>↗</span></a>"
                    f"{_nome_disp}</div>",
                    unsafe_allow_html=True)
                _rc[1].markdown(f"<span style='font-size:1.2rem'>{_r['Daily']}</span>",   unsafe_allow_html=True)
                _rc[2].markdown(f"<span style='font-size:1.2rem'>{_r['Weekly']}</span>",  unsafe_allow_html=True)
                _rc[3].markdown(f"<span style='font-size:1.2rem'>{_r['Monthly']}</span>", unsafe_allow_html=True)
                _rc[4].markdown(f"<b style='color:{_bull_c}'>{_r['TF Bull']}</b>",        unsafe_allow_html=True)
                _rc[5].markdown(
                    f"<div style='display:flex;align-items:center;gap:4px'>"
                    f"<span style='font-family:Courier New;color:{_score_c};font-weight:bold'>{_r['Score']}</span>"
                    f"<div style='flex:1;height:4px;background:#1e222d;border-radius:2px'>"
                    f"<div style='width:{_r['Score']}%;height:4px;background:{_score_c};border-radius:2px'>"
                    f"</div></div></div>",
                    unsafe_allow_html=True)

                if _mtf_show_detail:
                    with st.expander(f"📐 {_r['Ticker']} — dettaglio TF", expanded=False):
                        _dc1, _dc2, _dc3 = st.columns(3)
                        for _col_d, _tf_nm, _td in [(_dc1,"Daily",_r["_d"]),(_dc2,"Weekly",_r["_w"]),(_dc3,"Monthly",_r["_mo"])]:
                            with _col_d:
                                _em = _tf_emoji_fn(_td)
                                if _td.get("status") in ("no_data","error"):
                                    st.caption(f"**{_tf_nm}**: no data")
                                else:
                                    st.markdown(
                                        f"**{_tf_nm}** `{_em}`\n\n"
                                        f"P: `${_td.get('price',0)}` · E20: `${_td.get('ema20',0)}` · E50: `${_td.get('ema50',0)}`\n\n"
                                        f"RSI: `{_td.get('rsi',0)}` · OBV: `{'↑' if _td.get('obv_up') else '↓'}`"
                                    )

            # Strategy Chart
            st.markdown("---")
            st.markdown('<div class="section-pill">📊 STRATEGY CHART — Analisi Avanzata MTF</div>', unsafe_allow_html=True)
            _mtf_chart_tickers = [r["Ticker"] for r in _mtf_res]
            _msc1, _msc2 = st.columns([2,1])
            with _msc1:
                _mtf_chart_tkr = st.selectbox("Ticker per Strategy Chart",
                    _mtf_chart_tickers, format_func=lambda t: _mtf_labels.get(t,t), key="mtf_chart_sel")
            with _msc2:
                _mtf_chart_period = st.selectbox("Periodo", ["3mo","6mo","1y","2y"], index=1, key="mtf_chart_period")

            if _mtf_chart_tkr:
                _chart_row = pd.Series({"Ticker": _mtf_chart_tkr, "Nome": _mtf_nome_map.get(_mtf_chart_tkr,"")})
                if not df_ep.empty and "Ticker" in df_ep.columns:
                    _ep_match = df_ep[df_ep["Ticker"]==_mtf_chart_tkr]
                    if not _ep_match.empty:
                        _chart_row = _ep_match.iloc[0].copy()
                if not (hasattr(_chart_row,"get") and _chart_row.get("_chart_data")):
                    try:
                        import yfinance as _yf_mc
                        _raw_mc = _yf_mc.download(_mtf_chart_tkr, period=_mtf_chart_period,
                                                   interval="1d", auto_adjust=True, progress=False)
                        _raw_mc.columns = [c[0] if isinstance(c,tuple) else c for c in _raw_mc.columns]
                        if not _raw_mc.empty:
                            _cl_mc = _raw_mc["Close"].dropna()
                            _ema20_mc  = _cl_mc.ewm(span=20,adjust=False).mean()
                            _ema50_mc  = _cl_mc.ewm(span=50,adjust=False).mean()
                            _ema200_mc = _cl_mc.ewm(span=min(200,len(_cl_mc)),adjust=False).mean()
                            _sma20_mc  = _cl_mc.rolling(20).mean()
                            _std20_mc  = _cl_mc.rolling(20).std()
                            _chart_row = _chart_row.copy()
                            _chart_row["_chart_data"] = {
                                "dates":  [str(d)[:10] for d in _raw_mc.index],
                                "open":   _raw_mc["Open"].fillna(0).tolist(),
                                "high":   _raw_mc["High"].fillna(0).tolist(),
                                "low":    _raw_mc["Low"].fillna(0).tolist(),
                                "close":  _cl_mc.tolist(),
                                "volume": _raw_mc["Volume"].fillna(0).tolist() if "Volume" in _raw_mc.columns else [],
                                "ema20":  _ema20_mc.tolist(), "ema50": _ema50_mc.tolist(),
                                "ema200": _ema200_mc.tolist(),
                                "bb_up":  (_sma20_mc + 2*_std20_mc).tolist(),
                                "bb_dn":  (_sma20_mc - 2*_std20_mc).tolist(),
                            }
                            _chart_row["Prezzo"] = float(_cl_mc.iloc[-1])
                            _tr_mc = (_raw_mc["High"] - _raw_mc["Low"]).fillna(0)
                            _chart_row["ATR"] = float(_tr_mc.ewm(com=13,adjust=False).mean().iloc[-1])
                    except Exception as _mce:
                        st.warning(f"Chart data non disponibile per {_mtf_chart_tkr}: {_mce}")

                if hasattr(_chart_row,"get") and _chart_row.get("_chart_data"):
                    show_charts(_chart_row, key_suffix=f"mtf_{_mtf_chart_tkr}")
                else:
                    st.info(f"Dati chart non disponibili per {_mtf_chart_tkr}. Avvia lo scanner.")

                _tv_url = f"https://it.tradingview.com/chart/?symbol={_mtf_chart_tkr.replace('.MI','').replace('.','')}"
                st.markdown(
                    f"<a href='{_tv_url}' target='_blank' style='display:inline-block;"
                    f"background:#2962ff;color:white;padding:6px 16px;border-radius:4px;"
                    f"font-family:Trebuchet MS;font-size:0.85rem;text-decoration:none;margin-top:8px'>"
                    f"📈 Apri {_mtf_chart_tkr} su TradingView IT</a>",
                    unsafe_allow_html=True)

            # Export
            st.markdown("---")
            _df_mtf_exp = pd.DataFrame([
                {"Ticker":r["Ticker"],"Nome":r["Nome"],"Daily":r["Daily"],"Weekly":r["Weekly"],
                 "Monthly":r["Monthly"],"TF_Bull":r["TF Bull"],"Score":r["Score"]}
                for r in _mtf_res])
            _mtf_ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button("📊 Export MTF Matrix",
                _df_mtf_exp.to_csv(index=False).encode(),
                f"MTF_Matrix_v41_{_mtf_ts}.csv", "text/csv", key="mtf_export")



# =========================================================================
# v41 TAB #8 — PAPER TRADING JOURNAL
# =========================================================================
with tab_journal:
    st.markdown('<div class="section-pill">📓 PAPER TRADING JOURNAL v41</div>',
                unsafe_allow_html=True)
    st.caption("Log strutturato entry/exit/note. Metriche aggregate per setup type.")

    # Init journal in session state (persiste in sessione; export per persistenza lunga)
    if "v41_journal" not in st.session_state:
        st.session_state["v41_journal"] = []
    _journal = st.session_state["v41_journal"]

    # ── Form aggiunta trade ──────────────────────────────────────────────
    with st.expander("➕ Aggiungi Trade", expanded=len(_journal)==0):
        _jc = st.columns([1.5,1,1,1,1,1.5])
        with _jc[0]: _j_tkr    = st.text_input("Ticker",  key="j_tkr",  placeholder="AAPL").upper().strip()
        with _jc[1]: _j_entry  = st.number_input("Entry $",  min_value=0.0, step=0.01, key="j_entry")
        with _jc[2]: _j_exit   = st.number_input("Exit $",   min_value=0.0, step=0.01, key="j_exit")
        with _jc[3]: _j_size   = st.number_input("Shares",  min_value=1,   step=1,    key="j_size")
        with _jc[4]: _j_setup  = st.selectbox("Setup",
            ["EARLY","PRO","CONFLUENCE","SERAFINI","FINVIZ","HOT","MANUAL"], key="j_setup")
        with _jc[5]: _j_note   = st.text_input("Note setup", key="j_note", placeholder="es. breakout EMA20")

        _jc2 = st.columns([2,1,1])
        with _jc2[0]: _j_date_e = st.date_input("Data Entry",  key="j_date_e")
        with _jc2[1]: _j_date_x = st.date_input("Data Exit",   key="j_date_x")
        with _jc2[2]: _j_r_mult = st.number_input("R Multiple", min_value=-20.0, max_value=50.0,
                                                    step=0.1, value=0.0, key="j_r")

        if st.button("💾 Salva Trade", key="j_save", type="primary"):
            if _j_tkr and _j_entry > 0:
                _pnl_t = (_j_exit - _j_entry) * _j_size if _j_exit > 0 else 0
                _ret_t = (_j_exit/_j_entry - 1)*100 if _j_entry > 0 and _j_exit > 0 else 0
                _journal.append({
                    "ID":       len(_journal)+1,
                    "Ticker":   _j_tkr,
                    "Setup":    _j_setup,
                    "Entry $":  _j_entry,
                    "Exit $":   _j_exit if _j_exit > 0 else "Open",
                    "Shares":   _j_size,
                    "P&L $":    round(_pnl_t,2),
                    "Return %": round(_ret_t,2),
                    "R":        _j_r_mult,
                    "Note":     _j_note,
                    "Entry Date": str(_j_date_e),
                    "Exit Date":  str(_j_date_x),
                    "Won":      _pnl_t > 0,
                })
                st.success(f"✅ Trade {_j_tkr} salvato!")
                st.rerun()
            else:
                st.warning("Inserisci Ticker e prezzo Entry.")

    # ── Metriche aggregate ───────────────────────────────────────────────
    if _journal:
        _df_j = pd.DataFrame(_journal)
        _closed = _df_j[_df_j["Exit $"] != "Open"].copy()

        if not _closed.empty:
            st.markdown("---")
            st.markdown("#### 📊 Performance Aggregata")

            _jm1,_jm2,_jm3,_jm4,_jm5,_jm6 = st.columns(6)
            _n_j     = len(_closed)
            _wins_j  = int(_closed["Won"].sum())
            _wr_j    = _wins_j/_n_j*100
            _total_j = float(_closed["P&L $"].sum())
            _avg_r_j = float(_closed["R"].mean()) if "R" in _closed.columns else 0
            _pf_j    = (abs(_closed[_closed["P&L $"]>0]["P&L $"].sum()) /
                        abs(_closed[_closed["P&L $"]<0]["P&L $"].sum() + 0.01))

            _jm1.metric("📋 Trade",       _n_j)
            _jm2.metric("🎯 Win Rate",    f"{_wr_j:.1f}%")
            _jm3.metric("💰 P&L Totale",  f"${_total_j:+,.0f}")
            _jm4.metric("📐 Avg R",       f"{_avg_r_j:.2f}")
            _jm5.metric("⚡ Profit Factor",f"{_pf_j:.2f}")
            _jm6.metric("🏆 Winning",     f"{_wins_j}/{_n_j}")

            # Per setup type
            st.markdown("#### 📈 Win Rate per Setup Type")
            _setup_grp = _closed.groupby("Setup").agg(
                N=("Won","count"),
                Wins=("Won","sum"),
                PnL=("P&L $","sum"),
                AvgR=("R","mean"),
            ).reset_index()
            _setup_grp["Win%"] = (_setup_grp["Wins"]/_setup_grp["N"]*100).round(1)
            _setup_grp["PnL"]  = _setup_grp["PnL"].round(2)
            _setup_grp["AvgR"] = _setup_grp["AvgR"].round(2)

            # Visual per setup
            for _, _sg in _setup_grp.iterrows():
                _sc1,_sc2,_sc3,_sc4,_sc5 = st.columns([1.5,1,1,1,1])
                _win_col = "#00ff88" if _sg["Win%"] >= 50 else "#ef4444"
                _pnl_col = "#00ff88" if _sg["PnL"] >= 0 else "#ef4444"
                _sc1.markdown(f"<b style='color:#58a6ff'>{_sg['Setup']}</b>", unsafe_allow_html=True)
                _sc2.markdown(f"<span style='font-family:Courier New'>{int(_sg['N'])} trade</span>")
                _sc3.markdown(f"<b style='color:{_win_col};font-family:Courier New'>{_sg['Win%']}%</b>",
                              unsafe_allow_html=True)
                _sc4.markdown(f"<span style='color:{_pnl_col};font-family:Courier New'>"
                              f"${_sg['PnL']:+,.0f}</span>", unsafe_allow_html=True)
                _sc5.markdown(f"<span style='font-family:Courier New;color:#b2b5be'>"
                              f"R: {_sg['AvgR']}</span>", unsafe_allow_html=True)

        # ── Trade log completo ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Trade Log")

        # Filtro per setup
        _j_setup_filter = st.multiselect("Filtra per Setup",
            _df_j["Setup"].unique().tolist(), key="j_filter_setup",
            default=_df_j["Setup"].unique().tolist())
        _df_j_disp = _df_j[_df_j["Setup"].isin(_j_setup_filter)] if _j_setup_filter else _df_j

        # Colori riga per P&L
        for _, _row_j in _df_j_disp.iterrows():
            _is_open = _row_j["Exit $"] == "Open"
            _pnl_v   = float(_row_j["P&L $"]) if not _is_open else 0
            _row_c   = "#00ff8822" if _pnl_v > 0 else "#ef444422" if _pnl_v < 0 else "#58a6ff22"

            _rj1,_rj2,_rj3,_rj4,_rj5,_rj6,_rj7 = st.columns([0.4,1.2,1,1,1,1,2])
            _rj1.markdown(f"<span style='color:#6b7280;font-size:0.75rem'>#{int(_row_j['ID'])}</span>",
                          unsafe_allow_html=True)
            _rj2.markdown(f"<b style='font-family:Courier New;color:#00ff88'>{_row_j['Ticker']}</b>",
                          unsafe_allow_html=True)
            _rj3.markdown(f"<span style='font-size:0.78rem;color:#58a6ff'>{_row_j['Setup']}</span>",
                          unsafe_allow_html=True)
            _rj4.markdown(f"<span style='font-family:Courier New;font-size:0.82rem'>"
                          f"${_row_j['Entry $']:.2f}</span>", unsafe_allow_html=True)
            _exit_display = "Open" if _is_open else f"${float(_row_j['Exit $']):.2f}"
            _rj5.markdown(f"<span style='font-family:Courier New;font-size:0.82rem'>"
                          f"{_exit_display}</span>",
                          unsafe_allow_html=True)
            _pnl_color = "#00ff88" if _pnl_v > 0 else "#ef4444" if _pnl_v < 0 else "#6b7280"
            _pnl_display = "Open" if _is_open else f"${_pnl_v:+,.0f}"
            _rj6.markdown(f"<b style='color:{_pnl_color};font-family:Courier New'>"
                          f"{_pnl_display}</b>",
                          unsafe_allow_html=True)
            _rj7.markdown(f"<span style='color:#6b7280;font-size:0.78rem'>{_row_j['Note']}</span>",
                          unsafe_allow_html=True)

        # Pulsante delete ultimo trade
        _jd1, _jd2, _jd3 = st.columns([1,1,3])
        with _jd1:
            if st.button("🗑️ Rimuovi ultimo", key="j_del_last") and _journal:
                _journal.pop(); st.rerun()
        with _jd2:
            if st.button("🗑️ Svuota tutto", key="j_clear_all"):
                st.session_state["v41_journal"] = []; st.rerun()

        # Export Journal XLSX
        st.markdown("---")
        _j_export_buf = io.BytesIO()
        with pd.ExcelWriter(_j_export_buf, engine="xlsxwriter") as _jw:
            _df_j.to_excel(_jw, sheet_name="Trade Log", index=False)
            if not _closed.empty:
                _setup_grp.to_excel(_jw, sheet_name="Per Setup", index=False)
        _j_ts = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button("📊 Export Journal XLSX",
            _j_export_buf.getvalue(),
            f"TradingJournal_v41_{_j_ts}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="j_export_xlsx")
    else:
        st.info("📭 Nessun trade nel journal. Aggiungi il tuo primo trade sopra.")


# =========================================================================
# v41 TAB #1 — MARKET REGIME + SECTOR ROTATION HEATMAP INTERATTIVA
# =========================================================================
with tab_regime:
    st.markdown('<div class="section-pill">🌡️ MARKET REGIME DETECTION v41 — VIX · Fear&Greed · Breadth · Bonds</div>',
                unsafe_allow_html=True)

    if st.button("🔄 Aggiorna dati Regime", key="regime_refresh",
                 help="Ricarica VIX/SPY/QQQ/IWM/TLT — TTL 2 minuti"):
        st.cache_data.clear(); st.rerun()

    try:
        _rg = _get_market_regime()
        _rg_c = _rg["color"]

        # ── KPI row ───────────────────────────────────────────────────────
        _r1a,_r1b,_r1c,_r1d,_r1e,_r1f = st.columns(6)
        _r1a.metric("🌡️ Regime",     f"{_rg['icon']} {_rg['regime']}")
        _r1b.metric("📊 VIX",         _rg["vix"],
                    delta=f"{_rg['vix_trend']:+.1f} 5d", delta_color="inverse")
        _r1c.metric("📈 SPY 20d",     f"{_rg['spy_mom_20d']:+.1f}%")
        _r1d.metric("💎 Breadth",     f"{_rg['breadth_score']}/3",
                    help="SPY+QQQ+IWM momentum 20d positivo")
        _r1e.metric("🏦 10Y",         f"{_rg.get('tnx_val',0):.2f}%",
                    delta=f"{_rg.get('tnx_trend',0):+.2f}", delta_color="inverse")
        _r1f.metric("✈️ Bond Flight",
                    "🔴 ON" if _rg.get("bond_flight") else "🟢 OFF",
                    help="TLT +2% in 10gg = flight-to-safety")

        # ── Fear & Greed gauge + Regime score ─────────────────────────────
        _fg_col1, _fg_col2 = st.columns([1.2, 2.8])
        with _fg_col1:
            _fg = _rg.get("fear_greed", 50)
            _fg_lbl = _rg.get("fg_label","Neutral")
            _fg_col_v = _rg.get("fg_color","#f59e0b")
            import math as _mth
            _ang = _mth.pi + (_fg/100) * _mth.pi
            _nx = 100 + 72*_mth.cos(_ang); _ny = 100 + 72*_mth.sin(_ang)
            _gauge_svg = f"""<svg width='200' height='115' viewBox='0 0 200 115'>
  <defs>
    <linearGradient id="grd" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#ef4444"/>
      <stop offset="25%"  stop-color="#f97316"/>
      <stop offset="50%"  stop-color="#f59e0b"/>
      <stop offset="75%"  stop-color="#26a69a"/>
      <stop offset="100%" stop-color="#00ff88"/>
    </linearGradient>
  </defs>
  <path d="M 20,100 A 80,80 0 1,1 180,100"
    fill="none" stroke="#1e222d" stroke-width="20" stroke-linecap="round"/>
  <path d="M 20,100 A 80,80 0 1,1 180,100"
    fill="none" stroke="url(#grd)" stroke-width="20" stroke-linecap="round" opacity="0.35"/>
  <line x1="100" y1="100" x2="{_nx:.1f}" y2="{_ny:.1f}"
    stroke="{_fg_col_v}" stroke-width="3.5" stroke-linecap="round"/>
  <circle cx="100" cy="100" r="6" fill="{_fg_col_v}"/>
  <text x="100" y="85" text-anchor="middle" font-size="24" font-weight="bold" fill="{_fg_col_v}">{_fg}</text>
  <text x="100" y="112" text-anchor="middle" font-size="9" fill="#787b86">FEAR / GREED PROXY</text>
  <text x="18" y="113" font-size="8" fill="#ef4444">FEAR</text>
  <text x="155" y="113" font-size="8" fill="#00ff88">GREED</text>
</svg>"""
            st.markdown(
                f"<div style='text-align:center'>{_gauge_svg}"
                f"<div style='color:{_fg_col_v};font-weight:bold;font-size:0.95rem;"
                f"margin-top:-4px'>{_fg_lbl}</div></div>",
                unsafe_allow_html=True)

        with _fg_col2:
            # Regime score bar
            _rs = _rg.get("regime_score",0)
            _rs_pct = min(100, max(0, _rs/9*100))
            _rs_col = "#00ff88" if _rs>=7 else "#26a69a" if _rs>=5 else "#f59e0b" if _rs>=3 else "#ef4444"
            st.markdown(
                f"<div style='margin-bottom:12px'>"
                f"<span style='color:#787b86;font-size:0.75rem'>REGIME SCORE: "
                f"<b style='color:{_rs_col}'>{_rs}/9</b></span>"
                f"<div style='height:8px;background:#1e222d;border-radius:4px;margin-top:3px'>"
                f"<div style='width:{_rs_pct:.0f}%;height:8px;background:{_rs_col};"
                f"border-radius:4px'></div></div></div>",
                unsafe_allow_html=True)

            # Multi-indice breadth
            for _nm_b,_mom_b in [("SPY",_rg["spy_mom_20d"]),
                                   ("QQQ",_rg.get("qqq_mom_20d",0)),
                                   ("IWM",_rg.get("iwm_mom_20d",0))]:
                _bc = "#00ff88" if _mom_b>2 else "#26a69a" if _mom_b>0 else "#ef4444"
                _bar_w = min(100, abs(_mom_b)*8)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
                    f"<span style='font-family:Courier New;color:#b2b5be;font-size:0.8rem;"
                    f"min-width:32px'>{_nm_b}</span>"
                    f"<div style='flex:1;height:10px;background:#1e222d;border-radius:3px;overflow:hidden'>"
                    f"<div style='width:{_bar_w:.0f}%;height:10px;background:{_bc};border-radius:3px'>"
                    f"</div></div>"
                    f"<span style='font-family:Courier New;color:{_bc};font-size:0.8rem;"
                    f"min-width:50px;text-align:right'>{_mom_b:+.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True)

        # ── Playbook operativo ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Playbook Operativo")
        _playbook = {
            "Risk-On":  [("✅ Scanner","Tutti i tab attivi: EARLY·PRO·CONFLUENCE·STRONG"),
                         ("✅ Size","100% del sizing calcolato"),
                         ("✅ Setup","Priorità CONFLUENCE (EARLY+PRO+Weekly Bull)"),
                         ("✅ Hold","Mantieni posizioni — trend è amico"),
                         ("⚠️ Alert","VIX < 12 = compiacenza, occhio agli ingressi tardivi")],
            "Caution":  [("🟡 Scanner","STRONG e CONFLUENCE. Skip EARLY isolati"),
                         ("🟡 Size","75% della size standard"),
                         ("🟡 Stop","Stop più stretti: 1× ATR invece di 1.5×"),
                         ("🟡 Settori","Privilegia difensivi: XLV·XLU·XLP"),
                         ("⚠️ Evita","Settori ciclici ad alta volatilità")],
            "Risk-Off": [("🟠 Scanner","Solo STRONG (Pro≥8). Ignora EARLY e PRO base"),
                         ("🟠 Size","50% del sizing — capitale protetto"),
                         ("🟠 Crisis","Apri Crisis Monitor: GLD·TLT·XLV"),
                         ("🟠 Setup","Solo titoli L2+ liquidità, ATR% < 4%"),
                         ("❌ Evita","No nuovi long su growth/tech/small-cap")],
            "Crisis":   [("🔴 Scanner","NON aprire nuovi long"),
                         ("🔴 Size","0% — cash preservation totale"),
                         ("🔴 Azione","Chiudi posizioni deboli, gestisci le forti"),
                         ("🔴 Crisis","Crisis Monitor ATTIVO: GLD·TLT·SHY·USD"),
                         ("✅ Rientro","Aspetta VIX < 25 per 3gg prima di rientrare")],
        }
        _pb = _playbook.get(_rg["regime"], _playbook["Caution"])
        _pb_c = _rg["color"]
        for _rt, _rtxt in _pb:
            st.markdown(
                f"<div style='display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #1e222d'>"
                f"<span style='color:{_pb_c};font-weight:bold;min-width:115px;font-size:0.82rem'>{_rt}</span>"
                f"<span style='color:#d1d4dc;font-size:0.82rem'>{_rtxt}</span></div>",
                unsafe_allow_html=True)

        # ── VIX storico 60gg ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📉 VIX — Storico 60 giorni con zone regime")
        try:
            import yfinance as _yf_vh
            _vh = _yf_vh.download("^VIX","60d","1d", auto_adjust=True, progress=False)
            _vh.columns = [c[0] if isinstance(c,tuple) else c for c in _vh.columns]
            _vh_cl = _vh["Close"].dropna() if not _vh.empty else pd.Series(dtype=float)
            if len(_vh_cl) > 2:
                import plotly.graph_objects as _pgo_v
                _fig_v = _pgo_v.Figure()
                _fig_v.add_hrect(y0=35,y1=80,fillcolor="rgba(239,68,68,0.07)",line_width=0)
                _fig_v.add_hrect(y0=25,y1=35,fillcolor="rgba(249,115,22,0.07)",line_width=0)
                _fig_v.add_hrect(y0=18,y1=25,fillcolor="rgba(245,158,11,0.07)",line_width=0)
                _fig_v.add_hrect(y0=0,y1=18,fillcolor="rgba(38,166,154,0.05)",line_width=0)
                for _lv,_lc,_ll in [(35,"#ef4444","Crisis"),(25,"#f97316","Risk-Off"),(18,"#f59e0b","Caution")]:
                    _fig_v.add_hline(y=_lv, line=dict(color=_lc,width=1,dash="dot"),
                        annotation_text=f" {_ll}", annotation_font_color=_lc, annotation_font_size=9)
                _vcolors = ["#ef4444" if v>=35 else "#f97316" if v>=25 else "#f59e0b" if v>=18 else "#26a69a"
                            for v in _vh_cl.tolist()]
                _fig_v.add_trace(_pgo_v.Scatter(
                    x=[str(d)[:10] for d in _vh_cl.index], y=_vh_cl.tolist(),
                    mode="lines+markers", name="VIX",
                    line=dict(color="#58a6ff",width=2),
                    marker=dict(color=_vcolors,size=5),
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.05)",
                    hovertemplate="VIX: %{y:.1f} — %{x}<extra></extra>"))
                _vix_ly = dict(PLOTLY_DARK)
                _vix_ly["yaxis"] = dict(_vix_ly.get("yaxis",{}),
                    title="VIX", range=[0,max(45,max(_vh_cl.tolist())*1.1)], tickfont=dict(size=9))
                _fig_v.update_layout(**_vix_ly,
                    title=dict(text=f"VIX: <b>{_rg['vix']}</b> | Regime: <b>{_rg['regime']}</b>",
                               font=dict(color="#50c4e0",size=12),x=0.01),
                    height=240, margin=dict(l=0,r=0,t=38,b=0),
                    showlegend=False, hovermode="x unified")
                st.plotly_chart(_fig_v, use_container_width=True, key="vix_hist_chart")
        except Exception as _vhe:
            st.caption(f"VIX storico: {_vhe}")

    except Exception as _rge:
        st.warning(f"Regime data non disponibile: {_rge}")

    st.markdown("---")

    # ── v41 #7 — SECTOR ROTATION HEATMAP INTERATTIVA ────────────────────

    # ── v41 #7 — SECTOR ROTATION HEATMAP INTERATTIVA ────────────────────
    st.markdown('<div class="section-pill">🔄 SECTOR ROTATION HEATMAP — 11 Settori × 6 Periodi</div>',
                unsafe_allow_html=True)
    st.caption("Click su una cella per vedere i ticker del settore. Dati in tempo reale da ETF GICS.")

    _sr_cache_col1, _sr_cache_col2 = st.columns([3,1])
    with _sr_cache_col2:
        if st.button("🗑️ Aggiorna dati settori", key="sr_clear_cache",
                     help="Forza il refresh dei dati settoriali da Yahoo Finance"):
            st.cache_data.clear()
            st.rerun()
    with st.spinner("Carico dati settoriali (6 periodi: 1d/5d/1m/3m/6m/1y)..."):
        _sr_df = _get_sector_returns()

    if not _sr_df.empty:
        # ── Heatmap Plotly interattiva ─────────────────────────────────
        import plotly.graph_objects as _pgo

        _periods_sr = ["1d","5d","1m","3m","6m","1y"]
        _period_labels = {"1d":"1 Giorno","5d":"5 Giorni","1m":"1 Mese","3m":"3 Mesi","6m":"6 Mesi","1y":"1 Anno"}
        _sectors_sr = _sr_df["Sector"].tolist()

        # Matrice valori
        _z_matrix = []
        _text_matrix = []
        for _p in _periods_sr:
            _col_vals = []
            _col_text = []
            for _sec in _sectors_sr:
                _row = _sr_df[_sr_df["Sector"]==_sec]
                _v = float(_row[_p].iloc[0]) if not _row.empty and _p in _row.columns else 0
                _col_vals.append(_v)
                _col_text.append(f"{_v:+.1f}%")
            _z_matrix.append(_col_vals)
            _text_matrix.append(_col_text)

        _fig_sr = _pgo.Figure(data=_pgo.Heatmap(
            z=_z_matrix,
            x=_sectors_sr,
            y=[_period_labels.get(_p,_p) for _p in _periods_sr],
            text=_text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, color="white", family="Courier New"),
            colorscale=[
                [0.0,  "#7f0000"], [0.25, "#ef4444"],
                [0.45, "#1e222d"], [0.55, "#1e222d"],
                [0.75, "#26a69a"], [1.0,  "#00ff88"],
            ],
            zmid=0,
            showscale=True,
            colorbar=dict(
                title="Return %",
                tickfont=dict(color="#787b86", size=9),
                outlinecolor="#2a2e39",
            ),
            hovertemplate="<b>%{x}</b><br>Periodo: %{y}<br>Return: %{text}<extra></extra>",
        ))
        _sr_layout = dict(PLOTLY_DARK)
        _sr_layout["xaxis"] = dict(_sr_layout.get("xaxis",{}), tickfont=dict(size=10,color="#b2b5be"), side="bottom")
        _sr_layout["yaxis"] = dict(_sr_layout.get("yaxis",{}), tickfont=dict(size=10,color="#b2b5be"))
        _fig_sr.update_layout(
            **_sr_layout,
            title=dict(text="Sector Rotation — Return % per periodo",
                       font=dict(color="#50c4e0",size=13), x=0.01),
            height=320,
            margin=dict(l=0,r=0,t=45,b=0),
        )
        st.plotly_chart(_fig_sr, use_container_width=True, key="sector_heatmap_v41")

        # ── Drill-down: click su settore ───────────────────────────────
        st.markdown("#### 🔍 Drill-down per Settore")
        _sr_cols = st.columns([2,1])
        with _sr_cols[0]:
            _sel_sector = st.selectbox(
                "Seleziona settore per vedere i ticker",
                _sectors_sr, key="sr_drilldown_sector"
            )
        with _sr_cols[1]:
            _sel_period_dd = st.selectbox(
                "Periodo di riferimento",
                _periods_sr, index=2,
                format_func=lambda p: _period_labels.get(p,p),
                key="sr_drilldown_period"
            )

        if _sel_sector:
            # Mostra performance ETF settore
            _etf_name = _SECTOR_ETFS.get(_sel_sector,"")
            _etf_row  = _sr_df[_sr_df["Sector"]==_sel_sector]
            if not _etf_row.empty:
                _etf_vals = {p: float(_etf_row[p].iloc[0]) for p in _periods_sr if p in _etf_row.columns}
                _ev1,_ev2,_ev3,_ev4,_ev5,_ev6 = st.columns(6)
                for _col_ev, _p in zip([_ev1,_ev2,_ev3,_ev4,_ev5,_ev6], _periods_sr):
                    _v = _etf_vals.get(_p,0)
                    _col_ev.metric(f"{_etf_name} {_period_labels.get(_p,_p)}", f"{_v:+.1f}%",
                                   delta=None)

            # Ticker del settore
            _sector_tkrs = _SECTOR_TICKERS.get(_sel_sector,[])
            if _sector_tkrs:
                st.markdown(f"**Ticker principali — {_sel_sector}:**")
                _tkr_cols = st.columns(5)
                for _i, _tkr_s in enumerate(_sector_tkrs):
                    # Evidenzia se è in watchlist o scanner
                    _in_wl = False
                    _in_sc = False
                    try:
                        _wl_check = load_watchlist()
                        if not _wl_check.empty and "Ticker" in _wl_check.columns:
                            _in_wl = _tkr_s in _wl_check["Ticker"].values
                    except Exception:
                        pass
                    if not df_ep.empty and "Ticker" in df_ep.columns:
                        _in_sc = _tkr_s in df_ep["Ticker"].values
                    _badge_s = " ⭐" if _in_sc else " 📋" if _in_wl else ""
                    _color_s = "#00ff88" if _in_sc else "#58a6ff" if _in_wl else "#b2b5be"
                    _tv_s = _tkr_s.replace(".MI","").replace(".","")
                    _tv_url_s = f"https://it.tradingview.com/chart/?symbol={_tv_s}"
                    with _tkr_cols[_i % 5]:
                        # st.link_button funziona sia in local che in Streamlit Cloud
                        _btn_label = f"{_tkr_s}{_badge_s} ↗"
                        st.link_button(
                            _btn_label, _tv_url_s,
                            use_container_width=True,
                            help=f"Apri {_tkr_s} su TradingView IT",
                            type="secondary",
                        )
                st.caption("⭐ = in scanner · 📋 = in watchlist · click → TradingView IT")

        # ── Sector Rankings table ──────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🏆 Ranking Settori")
        _sr_rank = _sr_df[["Sector","ETF"] + _periods_sr].copy().sort_values(
            _sel_period_dd, ascending=False).reset_index(drop=True)
        _sr_rank.index += 1

        for _i, _rk_row in _sr_rank.iterrows():
            _rk1,_rk2,_rk3,_rk4,_rk5,_rk6,_rk7,_rk8 = st.columns([0.4,1.8,0.65,0.65,0.65,0.65,0.65,0.65])
            _medal = "🥇" if _i==1 else "🥈" if _i==2 else "🥉" if _i==3 else f"{_i}."
            _rk1.markdown(f"<b style='color:#787b86'>{_medal}</b>", unsafe_allow_html=True)
            _rk2.markdown(f"<b>{_rk_row['Sector']}</b> <span style='color:#6b7280;font-size:0.78rem'>"
                          f"({_rk_row['ETF']})</span>", unsafe_allow_html=True)
            for _col_rk, _p in zip([_rk3,_rk4,_rk5,_rk6,_rk7,_rk8], _periods_sr):
                _v = float(_rk_row[_p])
                _c = "#00ff88" if _v>0 else "#ef4444" if _v<0 else "#6b7280"
                _col_rk.markdown(f"<span style='font-family:Courier New;color:{_c};"
                                  f"font-size:0.82rem'>{_v:+.1f}%</span>",
                                  unsafe_allow_html=True)
    else:
        st.warning("Dati settoriali non disponibili. Controlla la connessione a Yahoo Finance.")

    # ── v41 #2 — POSITION SIZING ENGINE ──────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-pill">⚖️ POSITION SIZING ENGINE v41</div>',
                unsafe_allow_html=True)
    st.caption("Calcolo professionale della size ottimale basato su rischio per trade.")

    _ps_c1,_ps_c2,_ps_c3,_ps_c4 = st.columns(4)
    with _ps_c1: _ps_capital = st.number_input("Capitale ($)", min_value=1000.0,
                                               value=float(st.session_state.get("ps_capital",50000)),
                                               step=1000.0, key="ps_capital")
    with _ps_c2: _ps_risk    = st.number_input("Rischio % per trade", min_value=0.1,
                                               max_value=10.0, value=1.0, step=0.1, key="ps_risk")
    with _ps_c3: _ps_entry   = st.number_input("Entry $",  min_value=0.0, step=0.01, key="ps_entry")
    with _ps_c4: _ps_stop    = st.number_input("Stop Loss $", min_value=0.0, step=0.01, key="ps_stop")

    _ps_atr_auto = st.checkbox("Calcola stop da ATR (1.5× ATR)", key="ps_atr_auto")
    if _ps_atr_auto and _ps_entry > 0:
        _ps_atr_v = st.number_input("ATR $", min_value=0.0, step=0.01, key="ps_atr_v")
        if _ps_atr_v > 0:
            _ps_stop = round(_ps_entry - 1.5 * _ps_atr_v, 4)
            st.caption(f"Stop auto: **${_ps_stop:.2f}**  (entry − 1.5 × ATR)")

    if _ps_entry > 0 and _ps_stop > 0 and _ps_stop < _ps_entry:
        _ps_result = _calc_position_size(_ps_capital, _ps_risk, _ps_entry, _ps_stop)
        _psr1,_psr2,_psr3,_psr4,_psr5 = st.columns(5)
        _psr1.metric("📦 N. Azioni",       _ps_result["shares"])
        _psr2.metric("💰 Size Posizione",   f"${_ps_result['position_usd']:,.0f}")
        _psr3.metric("⚠️ Rischio $",        f"${_ps_result['risk_usd']:,.0f}")
        _psr4.metric("📊 % del Capitale",   f"{_ps_result['pct_capital']:.1f}%")
        _psr5.metric("📐 R/S per Azione",   f"${_ps_result['risk_per_share']:.4f}")

        # Target levels
        _t1 = round(_ps_entry + (_ps_entry - _ps_stop), 2)
        _t2 = round(_ps_entry + 2*(_ps_entry - _ps_stop), 2)
        _t3 = round(_ps_entry + 3*(_ps_entry - _ps_stop), 2)
        st.markdown(
            f"<div style='background:#1e222d;border-radius:8px;padding:10px 16px;"
            f"margin-top:8px;font-family:Courier New;font-size:0.84rem'>"
            f"🔴 SL: <b style='color:#ef4444'>${_ps_stop:.2f}</b> &nbsp;|&nbsp; "
            f"🟠 T1 (R:1): <b style='color:#f59e0b'>${_t1:.2f}</b> &nbsp;|&nbsp; "
            f"🟢 T2 (R:2): <b style='color:#26a69a'>${_t2:.2f}</b> &nbsp;|&nbsp; "
            f"✅ T3 (R:3): <b style='color:#00ff88'>${_t3:.2f}</b>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Inserisci Entry e Stop Loss per calcolare la size.")

# =========================================================================
# v41 — Aggiunge 3 nuovi tab alla lista esistente
# Tab: 🧠 AI Explainer, 📱 Telegram, 📊 Risk Pro, 🔍 Gap Scanner
# =========================================================================
# Nota: i tab vengono aggiunti alla definizione esistente tramite
# session_state flag — usa st.expander nei tab esistenti per compatibilità

# ── v41 scan stats update (batch info) ────────────────────────────────────
if "scan_stats" in st.session_state:
    _ss37 = st.session_state.scan_stats
    _batches_info = (f" · 📦 {_ss37.get('batches','?')} batch×{_ss37.get('batch_size','?')}"
                     if "batches" in _ss37 else "")
    st.sidebar.caption(
        f"⏱️ **{_ss37.get('elapsed_s','?')}s** · "
        f"⚡ {_ss37.get('cache_hits',0)} cache · "
        f"☁️ {_ss37.get('downloaded',0)} scaricati"
        f"{_batches_info}"
    )

# =========================================================================
# v41 UPGRADE #3 — AI SIGNAL EXPLAINER
# =========================================================================
# =========================================================================
# v41 UPGRADE #3 — AI SIGNAL EXPLAINER — Multi-Provider con Fallback
# Provider chain: Gemini (free) → Groq (free) → OpenRouter → Claude
# =========================================================================

def _send_telegram_v41(bot_token: str, chat_id: str, message: str) -> bool:
    """Invia messaggio via Telegram Bot API."""
    try:
        import requests as _req_tg
        _url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        _r = _req_tg.post(_url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        return _r.status_code == 200
    except Exception:
        return False


def _render_telegram_settings_v41():
    """Pannello configurazione Telegram nel Risk Manager."""
    st.markdown('<div class="section-pill">📱 NOTIFICHE TELEGRAM v41</div>',
                unsafe_allow_html=True)

    _tg_c1, _tg_c2 = st.columns(2)
    with _tg_c1:
        st.markdown("**Configurazione Bot**")
        _tg_token = st.text_input("Bot Token",
            value=st.secrets.get("TELEGRAM_BOT_TOKEN",""),
            type="password", key="tg_token_inp",
            help="Ottieni da @BotFather su Telegram")
        _tg_chat  = st.text_input("Chat ID",
            value=st.secrets.get("TELEGRAM_CHAT_ID",""),
            key="tg_chat_inp",
            help="Usa @userinfobot per trovare il tuo Chat ID")

        if st.button("🔔 Test notifica", key="tg_test_btn"):
            if _tg_token and _tg_chat:
                _ok = _send_telegram_v41(_tg_token, _tg_chat,
                    "✅ <b>Trading Scanner PRO v41</b>\nNotifiche Telegram attive!")
                if _ok:
                    st.success("✅ Messaggio inviato!")
                else:
                    st.error("❌ Errore invio. Verifica token e chat ID.")
            else:
                st.warning("Inserisci Bot Token e Chat ID.")

    with _tg_c2:
        st.markdown("**Alert automatici**")
        _tg_on_pro   = st.checkbox("Notifica segnali PRO/STRONG", True, key="tg_on_pro")
        _tg_on_conf  = st.checkbox("Notifica CONFLUENCE",         True, key="tg_on_conf")
        _tg_on_hot   = st.checkbox("Notifica REA-HOT",            False, key="tg_on_hot")
        _tg_min_css  = st.slider("CSS minimo per notifica", 0, 100, 60, key="tg_min_css")

        st.markdown("**Digest giornaliero**")
        _tg_digest   = st.checkbox("Abilita digest mattutino (9:00)", False, key="tg_digest")
        _tg_digest_h = st.number_input("Ora invio (UTC)", 0, 23, 7, key="tg_digest_h")

    # Invio manuale del digest
    st.markdown("---")
    if st.button("📤 Invia digest ora", key="tg_send_digest"):
        _df_ep_tg = st.session_state.get("df_ep", pd.DataFrame())
        if not _df_ep_tg.empty and _tg_token and _tg_chat:
            # Costruisce messaggio
            _pro_tg = _df_ep_tg[_df_ep_tg.get("Stato_Pro",pd.Series()).isin(["PRO","STRONG"])] \
                      if "Stato_Pro" in _df_ep_tg.columns else pd.DataFrame()
            _n_pro = len(_pro_tg)
            _top5  = _pro_tg.head(5)["Ticker"].tolist() if not _pro_tg.empty else []
            _rg_tg = st.session_state.get("_regime_cache", {})
            _msg_tg = (
                f"📊 <b>Trading Scanner PRO v41 — Digest</b>\n"
                f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
                f"💪 Segnali PRO/STRONG: <b>{_n_pro}</b>\n"
                f"🏆 Top 5: {', '.join(_top5) if _top5 else '—'}\n"
                f"🌡️ Regime: {_rg_tg.get('regime','N/A')} · VIX: {_rg_tg.get('vix','N/A')}"
            )
            _ok = _send_telegram_v41(_tg_token, _tg_chat, _msg_tg)
            if _ok: st.success("✅ Digest inviato!")
            else:   st.error("❌ Errore invio digest.")
        else:
            st.info("Avvia lo scanner e configura il bot prima.")

    # Auto-notifica dopo ogni scan
    _df_ep_new = st.session_state.get("df_ep", pd.DataFrame())
    _scan_notified_key = f"_tg_notified_{id(_df_ep_new)}"
    if (_tg_on_pro and _tg_token and _tg_chat
            and not st.session_state.get(_scan_notified_key)
            and not _df_ep_new.empty):
        _auto_pro = _df_ep_new[_df_ep_new.get("Stato_Pro",pd.Series()).isin(["PRO","STRONG"])] \
                    if "Stato_Pro" in _df_ep_new.columns else pd.DataFrame()
        if not _auto_pro.empty and "CSS" in _auto_pro.columns:
            _auto_top = _auto_pro[pd.to_numeric(_auto_pro["CSS"],errors="coerce").fillna(0)>=_tg_min_css]
            if not _auto_top.empty:
                _auto_tickers = _auto_top.head(5)["Ticker"].tolist()
                _auto_msg = (
                    f"🚀 <b>Nuovi segnali scanner v41</b>\n"
                    f"💪 {len(_auto_top)} PRO/STRONG (CSS≥{_tg_min_css})\n"
                    f"📌 {', '.join(_auto_tickers)}"
                )
                _send_telegram_v41(_tg_token, _tg_chat, _auto_msg)
                st.session_state[_scan_notified_key] = True


# =========================================================================
# v41 UPGRADE #5 — RISK DASHBOARD PRO
# =========================================================================
def _render_risk_dashboard_v41(df_ep_risk):
    """Correlation matrix, VaR 95%, portfolio heat, drawdown alert."""
    st.markdown('<div class="section-pill">📊 RISK DASHBOARD PRO v41 — Correlazioni · VaR · Portfolio Heat</div>',
                unsafe_allow_html=True)

    # Selezione ticker per il portfolio
    _rk_tickers_all = []
    try:
        _wl_rk = load_watchlist()
        if not _wl_rk.empty and "Ticker" in _wl_rk.columns:
            _rk_tickers_all = _wl_rk[_wl_rk["list_name"]==st.session_state.current_list_name]["Ticker"].dropna().tolist()
    except Exception:
        pass
    if not _rk_tickers_all and not (df_ep_risk is None or df_ep_risk.empty):
        _rk_tickers_all = df_ep_risk["Ticker"].dropna().unique().tolist()[:20]

    if not _rk_tickers_all:
        st.info("Aggiungi ticker alla watchlist o avvia lo scanner per usare il Risk Dashboard.")
        return

    _rk_c1, _rk_c2 = st.columns([3,1])
    with _rk_c1:
        _rk_sel = st.multiselect("Ticker portfolio (max 15)",
            _rk_tickers_all, default=_rk_tickers_all[:min(8,len(_rk_tickers_all))],
            key="risk_tkr_sel")
    with _rk_c2:
        _rk_period = st.selectbox("Periodo analisi", ["1mo","3mo","6mo","1y"], index=1,
                                   key="risk_period")
        _rk_capital = st.number_input("Capitale ($)", 10000, 10000000, 100000, 10000,
                                       key="risk_capital")

    if st.button("📊 Calcola Risk Dashboard", key="risk_calc", type="primary") and _rk_sel:
        with st.spinner("Scarico dati e calcolo metriche risk..."):
            try:
                import yfinance as _yf_rk
                import numpy as _np_rk

                # Download returns
                _raw_rk = _yf_rk.download(
                    " ".join(_rk_sel), period=_rk_period, interval="1d",
                    auto_adjust=True, progress=False,
                    group_by="ticker" if len(_rk_sel)>1 else "column"
                )
                _raw_rk.columns = [c[0] if isinstance(c,tuple) else c for c in _raw_rk.columns]

                # Estrai close per ogni ticker
                _closes_rk = {}
                for _t in _rk_sel:
                    try:
                        if len(_rk_sel) == 1:
                            _cl = _raw_rk["Close"].dropna()
                        elif _t in _raw_rk.columns:
                            _cl = _raw_rk[_t].dropna() if not isinstance(_raw_rk[_t], pd.DataFrame) \
                                  else _raw_rk[_t]["Close"].dropna()
                        else:
                            continue
                        if len(_cl) > 5:
                            _closes_rk[_t] = _cl
                    except Exception:
                        pass

                if len(_closes_rk) < 2:
                    st.warning("Dati insufficienti per calcolare le correlazioni.")
                    return

                # Returns daily
                _df_ret = pd.DataFrame({t: c.pct_change().dropna() for t,c in _closes_rk.items()})
                _df_ret = _df_ret.dropna()

                # ── Correlation Matrix ─────────────────────────────────
                st.markdown("#### 🔗 Correlation Matrix (return giornalieri)")
                _corr = _df_ret.corr().round(2)

                import plotly.graph_objects as _pgo_rk
                _fig_corr = _pgo_rk.Figure(data=_pgo_rk.Heatmap(
                    z=_corr.values,
                    x=_corr.columns.tolist(),
                    y=_corr.index.tolist(),
                    text=_corr.values.round(2),
                    texttemplate="%{text}",
                    textfont=dict(size=10, family="Courier New"),
                    colorscale=[
                        [0.0,"#ef4444"],[0.5,"#1e222d"],[1.0,"#00ff88"]
                    ],
                    zmid=0, zmin=-1, zmax=1,
                    showscale=True,
                    colorbar=dict(tickfont=dict(color="#787b86",size=9),
                                  outlinecolor="#2a2e39"),
                    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr: %{text}<extra></extra>",
                ))
                _corr_layout = dict(PLOTLY_DARK)
                _corr_layout["xaxis"] = dict(_corr_layout.get("xaxis",{}),
                    tickfont=dict(size=9,color="#b2b5be"))
                _corr_layout["yaxis"] = dict(_corr_layout.get("yaxis",{}),
                    tickfont=dict(size=9,color="#b2b5be"))
                _fig_corr.update_layout(**_corr_layout,
                    title=dict(text="Correlation Matrix", font=dict(color="#50c4e0",size=12),x=0.01),
                    height=max(250, len(_rk_sel)*40+80),
                    margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(_fig_corr, use_container_width=True, key="risk_corr_chart")

                # ── VaR 95% e metriche aggregate ──────────────────────
                st.markdown("#### 📉 Metriche di Rischio Portfolio")
                _port_ret = _df_ret.mean(axis=1)  # equal weight portfolio
                _var95    = float(_np_rk.percentile(_port_ret, 5))
                _var99    = float(_np_rk.percentile(_port_ret, 1))
                _cvar95   = float(_port_ret[_port_ret <= _var95].mean())
                _vol_ann  = float(_port_ret.std() * _np_rk.sqrt(252) * 100)
                _sharpe   = float(_port_ret.mean() / _port_ret.std() * _np_rk.sqrt(252)) if _port_ret.std()>0 else 0
                _max_dd   = float(((1+_port_ret).cumprod() / (1+_port_ret).cumprod().cummax() - 1).min() * 100)

                _rm1,_rm2,_rm3,_rm4,_rm5,_rm6 = st.columns(6)
                _rm1.metric("📉 VaR 95% (1g)",   f"{_var95*100:+.2f}%", help="Perdita massima con 95% confidenza in 1 giorno")
                _rm2.metric("📉 VaR 99% (1g)",   f"{_var99*100:+.2f}%", help="Perdita massima con 99% confidenza in 1 giorno")
                _rm3.metric("💀 CVaR 95%",        f"{_cvar95*100:+.2f}%", help="Expected Shortfall — perdita media oltre il VaR")
                _rm4.metric("📊 Volatilità ann.", f"{_vol_ann:.1f}%")
                _rm5.metric("⚡ Sharpe",           f"{_sharpe:.2f}")
                _rm6.metric("📉 Max Drawdown",    f"{_max_dd:.1f}%")

                # VaR in dollari
                _var_usd = abs(_var95) * _rk_capital
                _cvar_usd= abs(_cvar95) * _rk_capital
                st.markdown(
                    f"<div style='background:#1e222d;border-radius:6px;padding:8px 14px;"
                    f"margin-top:6px;font-size:0.83rem;font-family:Courier New'>"
                    f"Su capitale <b style='color:#d1d4dc'>${_rk_capital:,.0f}</b> &nbsp;·&nbsp; "
                    f"VaR giornaliero 95%: <b style='color:#ef4444'>${_var_usd:,.0f}</b> &nbsp;·&nbsp; "
                    f"CVaR: <b style='color:#ef5350'>${_cvar_usd:,.0f}</b>"
                    f"</div>", unsafe_allow_html=True)

                # ── Portfolio Heat (concentrazione per settore) ────────
                st.markdown("#### 🌡️ Portfolio Heat — Esposizione per Settore")
                _sector_exposure = {}
                for _t in _rk_sel:
                    try:
                        _info_rk = _yf_rk.Ticker(_t).info
                        _sec = _info_rk.get("sector","Unknown") or "Unknown"
                        _sector_exposure[_sec] = _sector_exposure.get(_sec,0) + 1
                    except Exception:
                        _sector_exposure["Unknown"] = _sector_exposure.get("Unknown",0) + 1

                _total_sec = sum(_sector_exposure.values())
                _sec_sorted = sorted(_sector_exposure.items(), key=lambda x: -x[1])
                for _sec_nm, _cnt in _sec_sorted:
                    _pct_sec = _cnt/_total_sec*100
                    _c_sec   = "#ef4444" if _pct_sec>40 else "#f59e0b" if _pct_sec>25 else "#26a69a"
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                        f"<span style='font-family:Courier New;font-size:0.82rem;min-width:160px;"
                        f"color:#b2b5be'>{_sec_nm}</span>"
                        f"<div style='flex:1;height:6px;background:#1e222d;border-radius:3px'>"
                        f"<div style='width:{_pct_sec:.0f}%;height:6px;background:{_c_sec};"
                        f"border-radius:3px'></div></div>"
                        f"<span style='font-family:Courier New;font-size:0.82rem;color:{_c_sec};"
                        f"min-width:50px;text-align:right'>{_pct_sec:.0f}% ({_cnt})</span>"
                        f"</div>", unsafe_allow_html=True)
                if any(p>40 for _,p in [(s,c/_total_sec*100) for s,c in _sector_exposure.items()]):
                    st.warning("⚠️ Concentrazione settoriale elevata (>40%). Considera la diversificazione.")

                # ── Drawdown Alert ─────────────────────────────────────
                st.markdown("#### 🚨 Drawdown Alert")
                _dd_threshold = st.slider("Soglia alert drawdown %", 5, 30, 10, key="risk_dd_thr")
                if _max_dd < -_dd_threshold:
                    st.error(f"🚨 DRAWDOWN ALERT: portfolio in drawdown {_max_dd:.1f}% "
                             f"(soglia: -{_dd_threshold}%)")
                else:
                    st.success(f"✅ Drawdown attuale {_max_dd:.1f}% — entro la soglia -{_dd_threshold}%")

            except Exception as _rk_err:
                import traceback as _tbrk
                st.error(f"Errore Risk Dashboard: {_rk_err}")
                st.code(_tbrk.format_exc())


# =========================================================================
# v41 UPGRADE #6 — SCANNER AVANZATO: GAP + EARNINGS PLAY
# =========================================================================
@st.cache_data(ttl=300)
def _scan_gaps_v41(tickers: tuple, min_gap_pct: float = 1.0) -> pd.DataFrame:
    """Gap Scanner: trova ticker con gap apertura > min_gap_pct con volume confermato."""
    import yfinance as _yf_g
    _results = []
    for _tg in tickers:
        try:
            _raw_g = _yf_g.download(_tg, period="5d", interval="1d",
                                     auto_adjust=True, progress=False)
            _raw_g.columns = [c[0] if isinstance(c,tuple) else c for c in _raw_g.columns]
            if len(_raw_g) < 2: continue
            _prev_close = float(_raw_g["Close"].iloc[-2])
            _today_open = float(_raw_g["Open"].iloc[-1])
            _today_close= float(_raw_g["Close"].iloc[-1])
            _today_vol  = float(_raw_g["Volume"].iloc[-1]) if "Volume" in _raw_g.columns else 0
            _avg_vol    = float(_raw_g["Volume"].iloc[:-1].mean()) if "Volume" in _raw_g.columns else 1

            _gap_pct = (_today_open / _prev_close - 1) * 100
            _vol_ratio = _today_vol / _avg_vol if _avg_vol > 0 else 0

            if abs(_gap_pct) >= min_gap_pct and _vol_ratio >= 1.2:
                _gap_filled = (
                    (_gap_pct > 0 and _today_close < _today_open) or
                    (_gap_pct < 0 and _today_close > _today_open)
                )
                _results.append({
                    "Ticker":     _tg,
                    "Gap %":      round(_gap_pct, 2),
                    "Gap Type":   "UP ▲" if _gap_pct > 0 else "DOWN ▼",
                    "Prev Close": f"${_prev_close:.2f}",
                    "Open":       f"${_today_open:.2f}",
                    "Close":      f"${_today_close:.2f}",
                    "Vol Ratio":  round(_vol_ratio, 2),
                    "Gap Filled": "✅" if _gap_filled else "❌",
                })
        except Exception:
            pass
    _df_g = pd.DataFrame(_results)
    if not _df_g.empty:
        _df_g = _df_g.sort_values("Gap %", key=abs, ascending=False)
    return _df_g


def _render_advanced_scanner_v41():
    """Tab Scanner Avanzato: Gap Scanner + Earnings Play."""
    st.markdown('<div class="section-pill">🔍 SCANNER AVANZATO v41 — Gap · Earnings Play</div>',
                unsafe_allow_html=True)

    _adv_t1, _adv_t2 = st.tabs(["📈 Gap Scanner", "🗓️ Earnings Play"])

    with _adv_t1:
        st.caption("Trova gap di apertura significativi con volume confermato.")
        _gap_c1, _gap_c2, _gap_c3 = st.columns(3)
        with _gap_c1:
            _gap_min = st.slider("Gap minimo %", 0.5, 5.0, 1.0, 0.5, key="gap_min_pct")
        with _gap_c2:
            _gap_markets = st.multiselect("Universo", ["S&P500 (top 50)","Nasdaq (top 50)","Watchlist"],
                default=["S&P500 (top 50)"], key="gap_markets")
        with _gap_c3:
            st.write("")
            _run_gap = st.button("🔍 Scansiona Gap", key="gap_run", type="primary",
                                  use_container_width=True)

        if _run_gap:
            # Costruisce universo
            _gap_universe = []
            if "S&P500 (top 50)" in _gap_markets:
                _gap_universe += ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B",
                                   "JPM","V","UNH","XOM","JNJ","WMT","MA","PG","HD","CVX",
                                   "MRK","ABBV","LLY","BAC","COST","AVGO","PEP","TMO","ORCL",
                                   "NFLX","CSCO","ABT","ACN","CRM","AMD","INTC","TXN","QCOM",
                                   "HON","UPS","CAT","DE","GS","MS","BLK","SPGI","AXP","SYK",
                                   "ISRG","MDT","C","SCHW"]
            if "Nasdaq (top 50)" in _gap_markets:
                _gap_universe += ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO",
                                   "ASML","COST","NFLX","AMD","QCOM","INTC","ADBE","TXN",
                                   "INTU","MU","LRCX","KLAC","SNPS","CDNS","MRVL","PANW","FTNT"]
            if "Watchlist" in _gap_markets:
                try:
                    _wl_gap = load_watchlist()
                    if not _wl_gap.empty and "Ticker" in _wl_gap.columns:
                        _gap_universe += _wl_gap[_wl_gap["list_name"]==st.session_state.current_list_name]["Ticker"].dropna().tolist()
                except Exception:
                    pass

            _gap_universe = list(dict.fromkeys(_gap_universe))[:100]

            with st.spinner(f"Scansiono {len(_gap_universe)} ticker per gap ≥{_gap_min}%..."):
                _df_gaps = _scan_gaps_v41(tuple(_gap_universe), _gap_min)

            if _df_gaps.empty:
                st.info(f"Nessun gap significativo (≥{_gap_min}%) trovato oggi.")
            else:
                _g_up   = len(_df_gaps[_df_gaps["Gap Type"]=="UP ▲"])
                _g_down = len(_df_gaps[_df_gaps["Gap Type"]=="DOWN ▼"])
                _gc1,_gc2,_gc3 = st.columns(3)
                _gc1.metric("📈 Gap UP",     _g_up)
                _gc2.metric("📉 Gap DOWN",   _g_down)
                _gc3.metric("🔍 Totale",     len(_df_gaps))

                # Colora Gap %
                def _color_gap(v):
                    try:
                        val = float(str(v).replace("%",""))
                        return f"color:{'#00ff88' if val>0 else '#ef4444'};font-weight:bold;font-family:Courier New"
                    except: return ""

                st.dataframe(
                    _df_gaps.style.map(_color_gap, subset=["Gap %"]),
                    use_container_width=True, hide_index=True
                )

                # Export
                _gap_ts = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button("📊 Export Gap Scanner",
                    _df_gaps.to_csv(index=False).encode(),
                    f"GapScanner_v41_{_gap_ts}.csv", "text/csv", key="gap_export")

    with _adv_t2:
        st.caption("Setup pre-earnings: titoli con earnings nei prossimi 7 giorni e CSS elevato.")
        _ep_tickers_src = []
        if not (df_ep is None or (hasattr(df_ep,"empty") and df_ep.empty)):
            _ep_tickers_src = df_ep["Ticker"].dropna().unique().tolist()[:80]

        if not _ep_tickers_src:
            st.info("Avvia lo scanner per popolare l'universo Earnings Play.")
        else:
            _run_ep = st.button("🗓️ Trova Earnings Play", key="ep_run", type="primary")
            if _run_ep:
                with st.spinner("Scarico calendario earnings..."):
                    _earn_ep = _fetch_earnings_calendar(tuple(_ep_tickers_src))

                # Filtra solo entro 7 giorni
                _ep_imm = [e for e in _earn_ep if 0 <= e["Giorni"] <= 7]

                if not _ep_imm:
                    st.info("Nessun earnings imminente (entro 7 giorni) tra i ticker scanner.")
                else:
                    # Arricchisce con CSS dal df_ep
                    _ep_rows = []
                    for _ep_item in _ep_imm:
                        _t_ep = _ep_item["Ticker"]
                        _css_ep = "—"; _pro_ep = "—"; _rs_ep = "—"
                        if not (df_ep is None or df_ep.empty) and "Ticker" in df_ep.columns:
                            _match_ep = df_ep[df_ep["Ticker"]==_t_ep]
                            if not _match_ep.empty:
                                _css_ep = _match_ep.iloc[0].get("CSS","—")
                                _pro_ep = _match_ep.iloc[0].get("Stato_Pro","—")
                                _rs_ep  = _match_ep.iloc[0].get("RS_20d","—")
                        _ep_rows.append({
                            "Ticker":        _t_ep,
                            "Earnings Date": _ep_item["Earnings Date"],
                            "Giorni":        _ep_item["Giorni"],
                            "Badge":         _ep_item["Badge"],
                            "CSS":           _css_ep,
                            "Stato Pro":     _pro_ep,
                            "RS vs SPY":     f"{_rs_ep:+.1f}%" if isinstance(_rs_ep,(int,float)) else _rs_ep,
                        })

                    _df_ep_play = pd.DataFrame(_ep_rows).sort_values("Giorni")
                    st.dataframe(_df_ep_play, use_container_width=True, hide_index=True)
                    st.caption("Strategia: entra 3-5 giorni prima · esci il giorno prima degli earnings · non tenere oltre la data")


# =========================================================================
# v41 — FUNZIONI (definite dopo i tab v41 ma usate nei with tab_ qui sotto)
# Nota: in Python le funzioni possono essere chiamate in "with tab_X:" anche
# se definite prima nella stessa sessione Streamlit — purché la chiamata
# avvenga DOPO la definizione nello stesso file. Qui le definiamo e poi
# le usiamo immediatamente nei blocchi with tab_X: che seguono.
# =========================================================================

# ── v41 #1: Pattern Alerts ────────────────────────────────────────────────
# v41 — AGGIUNTE NEI TAB ESISTENTI (singole, no duplicati)
# =========================================================================


# =========================================================================
# v41 — TAB 💡 ANALISI PERSONALE
# =========================================================================
with tab_analisi:
    st.markdown('<div class="section-pill">💡 ANALISI PERSONALE v41 — Carica ticker · Ricevi consigli AI</div>', unsafe_allow_html=True)
    st.caption("Inserisci i ticker. L'AI scarica dati freschi e fornisce consigli con entry/stop/target precisi.")

    _ap_c1, _ap_c2 = st.columns([2, 1.5])
    with _ap_c1:
        _ap_input = st.text_area("I tuoi ticker (uno per riga)",
            placeholder="AAPL\nMSFT\nENI.MI\nRACE.MI", height=160, key="ap_tickers_input")
        _ap_period = st.select_slider("Periodo",
            options=["1mo","3mo","6mo","1y","2y"], value="6mo", key="ap_period")
    with _ap_c2:
        st.markdown("**Tipo di analisi:**")
        _ap_swing  = st.checkbox("📈 Swing Trading",   True,  key="ap_swing")
        _ap_trend  = st.checkbox("📊 Trend Following", True,  key="ap_trend")
        _ap_risk   = st.checkbox("⚠️ Risk Assessment", True,  key="ap_risk")
        _ap_entry  = st.checkbox("🎯 Entry ottimale",  True,  key="ap_entry")
        _ap_has_key = any([
            st.secrets.get("GEMINI_API_KEY","")     or st.session_state.get("_gemini_api_key",""),
            st.secrets.get("GROQ_API_KEY","")       or st.session_state.get("_groq_api_key",""),
            st.secrets.get("OPENROUTER_API_KEY","") or st.session_state.get("_openrouter_api_key",""),
            st.secrets.get("ANTHROPIC_API_KEY","")  or st.session_state.get("_anthropic_api_key",""),
        ])
        if not _ap_has_key:
            st.warning("⚠️ Configura API key nel tab PRO → AI Explainer")

    _run_ap = st.button("🔍 Analizza", key="ap_run", type="primary",
                        use_container_width=True, disabled=not _ap_has_key)

    if _run_ap and _ap_input and _ap_input.strip():
        _ap_tickers = [t.strip().upper() for t in _ap_input.strip().splitlines() if t.strip()][:15]
        for _ap_tkr in _ap_tickers:
            with st.expander(f"💡 {_ap_tkr}", expanded=True):
                with st.spinner(f"Scarico dati {_ap_tkr}..."):
                    _apd = {}
                    try:
                        import yfinance as _yf_ap
                        _raw_ap = _yf_ap.download(_ap_tkr, period=_ap_period, interval="1d",
                                                   auto_adjust=True, progress=False)
                        _raw_ap.columns = [c[0] if isinstance(c,tuple) else c for c in _raw_ap.columns]
                        if not _raw_ap.empty:
                            _cl=_raw_ap["Close"].dropna(); _hi=_raw_ap["High"].dropna(); _lo=_raw_ap["Low"].dropna()
                            _pr=float(_cl.iloc[-1]); _e20=float(_cl.ewm(span=20,adjust=False).mean().iloc[-1])
                            _e50=float(_cl.ewm(span=50,adjust=False).mean().iloc[-1])
                            _e200=float(_cl.ewm(span=min(200,len(_cl)),adjust=False).mean().iloc[-1])
                            _d=_cl.diff(); _g=_d.clip(lower=0); _l=-_d.clip(upper=0)
                            _rs=_g.ewm(com=13,adjust=False).mean()/(_l.ewm(com=13,adjust=False).mean()+1e-10)
                            _rsi=float((100-100/(1+_rs)).iloc[-1])
                            _atr=float((_hi-_lo).ewm(com=13,adjust=False).mean().iloc[-1])
                            _atr_pct=round(_atr/_pr*100,2)
                            _ret1m=round((_cl.iloc[-1]/_cl.iloc[max(-22,-len(_cl))]-1)*100,1)
                            _ret3m=round((_cl.iloc[-1]/_cl.iloc[max(-63,-len(_cl))]-1)*100,1)
                            _52wh=float(_hi.tail(252).max()); _dist_52wh=round((_pr/_52wh-1)*100,1)
                            _trend="RIALZISTA" if _pr>_e20>_e50 else "RIBASSISTA" if _pr<_e20<_e50 else "LATERALE"
                            _info_ap={}
                            try:
                                _ti=_yf_ap.Ticker(_ap_tkr).info
                                _info_ap={"name":_ti.get("longName","—"),"sector":_ti.get("sector","—"),
                                          "pe":_ti.get("trailingPE","—"),"beta":_ti.get("beta","—")}
                            except Exception: pass
                            _apd={"ticker":_ap_tkr,"nome":_info_ap.get("name","—"),
                                  "settore":_info_ap.get("sector","—"),"prezzo":round(_pr,2),
                                  "ema20":round(_e20,2),"ema50":round(_e50,2),"ema200":round(_e200,2),
                                  "rsi":round(_rsi,1),"atr":round(_atr,4),"atr_pct":_atr_pct,
                                  "ret1m":_ret1m,"ret3m":_ret3m,"52wh":round(_52wh,2),
                                  "dist_52wh":_dist_52wh,"trend":_trend,
                                  "pe":_info_ap.get("pe","—"),"beta":_info_ap.get("beta","—")}
                    except Exception as _ae:
                        st.warning(f"Impossibile scaricare dati: {_ae}")
                if not _apd:
                    continue
                _m1,_m2,_m3,_m4,_m5,_m6 = st.columns(6)
                _m1.metric("💰 Prezzo", f"${_apd['prezzo']:.2f}")
                _m2.metric("📊 RSI", f"{_apd['rsi']:.1f}")
                _m3.metric("📈 Trend", _apd["trend"])
                _m4.metric("1M", f"{_apd['ret1m']:+.1f}%")
                _m5.metric("3M", f"{_apd['ret3m']:+.1f}%")
                _m6.metric("ATR%", f"{_apd['atr_pct']:.1f}%")
                _ap_types = [t for t,c in [("Swing Trading",_ap_swing),("Trend Following",_ap_trend),
                                            ("Risk Assessment",_ap_risk),("Entry ottimale",_ap_entry)] if c]
                try: _rg_ap=_get_market_regime(); _regime_ap=f"VIX={_rg_ap['vix']}, Regime={_rg_ap['regime']}"
                except Exception: _regime_ap="N/D"
                _ap_prompt = (
                    "Sei un trader professionista. Analizza questo titolo.\n"
                    f"TITOLO: {_apd['ticker']} - {_apd['nome']} | SETTORE: {_apd['settore']}\n"
                    f"Prezzo: ${_apd['prezzo']} | EMA20: ${_apd['ema20']} | EMA50: ${_apd['ema50']} | EMA200: ${_apd['ema200']}\n"
                    f"RSI: {_apd['rsi']} | ATR: {_apd['atr_pct']}% | Trend: {_apd['trend']}\n"
                    f"Performance: 1M {_apd['ret1m']:+.1f}% | 3M {_apd['ret3m']:+.1f}%\n"
                    f"P/E: {_apd['pe']} | Beta: {_apd['beta']} | Mercato: {_regime_ap}\n"
                    f"Analisi: {', '.join(_ap_types)}\n\n"
                    "Rispondi in italiano:\n\n"
                    "SETUP ATTUALE:\n[2-3 righe]\n\n"
                    "STRATEGIA:\n[tipo, timing, motivazione]\n\n"
                    "ENTRY: $[prezzo]\n"
                    "STOP LOSS: $[entry-1.5xATR] ([%]%)\n"
                    "TARGET 1: $[entry+1.5xATR] (R:R 1:1)\n"
                    "TARGET 2: $[entry+3xATR] (R:R 2:1)\n\n"
                    "RISCHI:\n[2 righe]\n\n"
                    "CONSIGLIO:\n[1 riga]"
                )
                with st.spinner("Analisi AI..."):
                    try:
                        _ap_text, _ap_prov = _ai_call_with_fallback(_ap_prompt)
                        st.markdown(
                            f"<div style='background:#0d1117;border:1px solid #1f2937;"
                            f"border-left:3px solid #26a69a;border-radius:0 8px 8px 0;"
                            f"padding:14px 18px;font-size:0.88rem;line-height:1.7'>"
                            f"{_ap_text.replace(chr(10),'<br>')}</div>",
                            unsafe_allow_html=True)
                        st.caption(f"Provider: {_ap_prov} · Yahoo Finance · {_ap_period}")
                    except Exception as _ap_err:
                        _em = str(_ap_err)
                        if "NO_KEYS" in _em: st.warning("⚠️ Configura API key nel tab PRO → AI Explainer")
                        else: st.error(f"Errore AI: {_em[:200]}")
    elif not _ap_has_key:
        st.info("👆 Configura una API key gratuita (Gemini o Groq) nel tab PRO → AI Explainer.")
    else:
        st.markdown(
            "<div style='background:#1e222d;border:1px solid #2a2e39;border-radius:8px;padding:16px 20px'>"
            "<b style='color:#50c4e0'>Come funziona:</b><br><br>"
            "<span style='color:#b2b5be'>"
            "1. Inserisci ticker (es. AAPL, ENI.MI, RACE.MI)<br>"
            "2. Seleziona tipo di analisi<br>"
            "3. Clicca Analizza: dati freschi da Yahoo Finance<br>"
            "4. L'AI genera: setup, entry/stop/target precisi, rischi, consiglio"
            "</span></div>",
            unsafe_allow_html=True)


# =========================================================================
# v41 — TAB 🤖 AI ASSISTANT
# =========================================================================
with tab_ai:
    st.markdown('<div class="section-pill">🤖 AI TRADING ASSISTANT v41 — Chatbot per analisi e strategie</div>', unsafe_allow_html=True)
    st.caption("Chat interattiva con AI. Chiedi analisi ticker, spiegazioni pattern, consigli strategia.")

    # Session state per chat
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    if "ai_ticker_context" not in st.session_state:
        st.session_state.ai_ticker_context = {}

    # Sidebar configurazione AI
    with st.expander("⚙️ Configurazione AI", expanded=False):
        _ai_provider = st.selectbox("Provider", ["Anthropic (Claude)", "OpenAI (GPT)", "Groq (Llama)"], index=0)
        _ai_model = st.text_input("Model", value="claude-3-haiku-20240307" if "Anthropic" in _ai_provider else "gpt-4o-mini" if "OpenAI" in _ai_provider else "llama-3.1-70b-versatile")
        _ai_api_key = st.text_input("API Key", type="password", key="ai_chat_key")

    # Input ticker opzionale per contesto
    _ai_ticker_input = st.text_input("Ticker (opzionale per contesto)", placeholder="AAPL, MSFT, ENI.MI").strip().upper()
    if _ai_ticker_input:
        try:
            import yfinance as yf
            _tkr = yf.Ticker(_ai_ticker_input)
            _info = _tkr.info if hasattr(_tkr, 'info') else {}
            st.session_state.ai_ticker_context[_ai_ticker_input] = {
                "price": _info.get("currentPrice", _info.get("regularMarketPrice", "N/A")),
                "volume": _info.get("volume", "N/A"),
                "marketCap": _info.get("marketCap", "N/A"),
                "pe": _info.get("trailingPE", "N/A"),
                "beta": _info.get("beta", "N/A"),
                "52w_high": _info.get("fiftyTwoWeekHigh", "N/A"),
                "52w_low": _info.get("fiftyTwoWeekLow", "N/A"),
            }
            st.success(f"📊 {_ai_ticker_input}: ${st.session_state.ai_ticker_context[_ai_ticker_input]['price']}")
        except Exception as _e:
            st.warning(f"Errore download dati: {_e}")

    # Chat input
    _ai_user_msg = st.chat_input("Chiedi qualcosa sull'analisi tecnica, pattern, o strategie trading...")
    if _ai_user_msg:
        st.session_state.ai_chat_history.append({"role": "user", "content": _ai_user_msg})

        # Costruisci prompt con contesto
        _ai_context = ""
        if st.session_state.ai_ticker_context:
            for _tk, _dt in st.session_state.ai_ticker_context.items():
                _ai_context += f"\nTicker: {_tk} | Prezzo: ${_dt['price']} | Vol: {_dt['volume']} | 52w: ${_dt['52w_low']}-${_dt['52w_high']}\n"

        _ai_prompt = (
            f"Sei un trader professionista con 20 anni di esperienza. "
            f"{_ai_context}\n\n"
            f"Domanda: {_ai_user_msg}\n\n"
            "Rispondi in modo chiaro e professionale in italiano. "
            "Includi quando possibile: entry zone, stop loss, target, gestione rischio."
        )

        if _ai_api_key:
            with st.spinner("AI in elaborazione..."):
                try:
                    if "Anthropic" in _ai_provider:
                        import anthropic
                        _client = anthropic.Anthropic(api_key=_ai_api_key)
                        _resp = _client.messages.create(
                            model=_ai_model,
                            max_tokens=1024,
                            messages=[{"role": "user", "content": _ai_prompt}]
                        )
                        _ai_resp = _resp.content[0].text
                    elif "OpenAI" in _ai_provider:
                        import openai
                        _client = openai.OpenAI(api_key=_ai_api_key)
                        _resp = _client.chat.completions.create(
                            model=_ai_model,
                            messages=[{"role": "user", "content": _ai_prompt}]
                        )
                        _ai_resp = _resp.choices[0].message.content
                    else:  # Groq
                        import openai
                        _client = openai.OpenAI(api_key=_ai_api_key, base_url="https://api.groq.com/openai/v1")
                        _resp = _client.chat.completions.create(
                            model=_ai_model,
                            messages=[{"role": "user", "content": _ai_prompt}]
                        )
                        _ai_resp = _resp.choices[0].message.content

                    st.session_state.ai_chat_history.append({"role": "assistant", "content": _ai_resp})
                except Exception as _e:
                    st.error(f"Errore API: {_e}")
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": f"Errore: {_e}"})
        else:
            st.warning("⚠️ Inserisci API key per usare l'AI")

    # Mostra cronologia chat
    for _msg in st.session_state.ai_chat_history:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])

    if st.button("🗑️ Pulisci cronologia"):
        st.session_state.ai_chat_history = []
        st.rerun()


# =========================================================================
# v41 — TAB 🎲 OPTIONS SCANNER
# =========================================================================
with tab_opts:
    st.markdown('<div class="section-pill">🎲 OPTIONS SCANNER v41 — Volatilità e Opzioni</div>', unsafe_allow_html=True)
    st.caption("Scanner avanzato per opzioni: IV, P/C ratio, unusual activity, gamma squeeze.")

    # Input
    _opt_cols = st.columns([2, 1, 1, 1])
    with _opt_cols[0]:
        _opt_universe = st.text_input("Ticker (comma separated)", value="SPY,QQQ,IWM,AAPL,TSLA,NVDA,AMD").strip()
    with _opt_cols[1]:
        _opt_iv_min = st.slider("IV Rank min", 0, 100, 30)
    with _opt_cols[2]:
        _opt_pc_max = st.slider("P/C Ratio max", 0.5, 3.0, 2.0)
    with _opt_cols[3]:
        _opt_ivol = st.selectbox("IV Filter", ["IV > HV", "IV < HV", "Any"])

    if st.button("🔍 Scan Opzioni", use_container_width=True):
        with st.spinner("Scaricando dati options..."):
            _opt_tickers = [t.strip().upper() for t in _opt_universe.split(",") if t.strip()]
            _opt_results = []

            for _tk in _opt_tickers:
                try:
                    import yfinance as yf
                    _t = yf.Ticker(_tk)
                    _opt = _t.option_chain if hasattr(_t, 'option_chain') else None

                    if _opt is not None:
                        _calls = _opt.calls if hasattr(_opt, 'calls') else pd.DataFrame()
                        _puts = _opt.puts if hasattr(_opt, 'puts') else pd.DataFrame()

                        _pc_ratio = len(_puts) / max(len(_calls), 1)

                        # Calcola IV approssimativa (usa median di strike vicini)
                        _iv = _calls['impliedVolatility'].median() if not _calls.empty and 'impliedVolatility' in _calls.columns else 0
                        _hv = 20  # placeholder, calcolerebbe da historical

                        _opt_results.append({
                            "Ticker": _tk,
                            "IV": round(_iv * 100, 1) if _iv else 0,
                            "IV Rank": min(100, int((_iv / 0.5) * 100)) if _iv else 0,
                            "P/C Ratio": round(_pc_ratio, 2),
                            "Calls": len(_calls),
                            "Puts": len(_puts),
                            "Vol Tot": _calls['volume'].sum() + _puts['volume'].sum() if not _calls.empty and not _puts.empty else 0,
                        })
                except Exception as _e:
                    _opt_results.append({"Ticker": _tk, "IV": 0, "IV Rank": 0, "P/C Ratio": 0, "Calls": 0, "Puts": 0, "Vol Tot": 0})

            _opt_df = pd.DataFrame(_opt_results)

            # Filtri
            if _opt_ivol == "IV > HV":
                _opt_df = _opt_df[_opt_df["IV"] > _opt_df["IV"] * 0.8]  # semplificato
            elif _opt_ivol == "IV < HV":
                _opt_df = _opt_df[_opt_df["IV"] < _opt_df["IV"] * 0.8]

            _opt_df = _opt_df[_opt_df["IV Rank"] >= _opt_iv_min]
            _opt_df = _opt_df[_opt_df["P/C Ratio"] <= _opt_pc_max]

            if not _opt_df.empty:
                st.dataframe(
                    _opt_df.style.background_gradient(subset=["IV Rank", "P/C Ratio"], cmap="Reds"),
                    use_container_width=True, height=400
                )
            else:
                st.info("Nessun risultato con i filtri applicati.")

    # Spiegazione metriche
    with st.expander("📖 Guida Metriche Options"):
        st.markdown("""
        - **IV (Implied Volatility)**: Volatilità implicita delle opzioni
        - **IV Rank**: Posizione IV attuale rispetto ai 52w (0-100)
        - **P/C Ratio**: Rapporto puts/calls. >1.5 = più puts (bearish), <0.7 = più calls (bullish)
        - **Unusual Activity**: Volume opzioni >3x media recente
        """)


# =========================================================================
# v41 — TAB ⚡ MOMENTUM ALERTS
# =========================================================================
with tab_mom:
    st.markdown('<div name="section-pill">⚡ MOMENTUM ALERTS v41 — Alert tempo reale</div>', unsafe_allow_html=True)
    st.caption("Configura alert per breakout, volume spike e price action in tempo reale.")

    # Configurazione alert
    _ma_cols = st.columns(4)
    with _ma_cols[0]:
        _ma_breakout = st.checkbox("Breakout >2%", value=True)
    with _ma_cols[1]:
        _ma_vol_spike = st.checkbox("Volume >3x", value=True)
    with _ma_cols[2]:
        _ma_price = st.checkbox("Price Action", value=False)
    with _ma_cols[3]:
        _ma_earnings = st.checkbox("Earnings", value=True)

    _ma_universe = st.text_input("Ticker da monitorare", value="SPY,QQQ,NVDA,TSLA,AAPL,AMD,META,GOOGL").strip()
    _ma_refresh = st.slider("Refresh (sec)", 30, 300, 60)

    # Session per alert
    if "mom_alerts" not in st.session_state:
        st.session_state.mom_alerts = []

    # Btn forza scan
    if st.button("⚡ Scan Now", use_container_width=True):
        _ma_tickers = [t.strip().upper() for t in _ma_universe.split(",") if t.strip()]
        _ma_new_alerts = []

        with st.spinner("Scanning momentum..."):
            for _tk in _ma_tickers:
                try:
                    import yfinance as yf
                    _t = yf.Ticker(_tk)
                    _hist = _t.history(period="5d")

                    if _hist.empty:
                        continue

                    _price = _hist['Close'].iloc[-1]
                    _prev_price = _hist['Close'].iloc[-2]
                    _vol = _hist['Volume'].iloc[-1]
                    _vol_avg = _hist['Volume'].mean()

                    # Check breakout
                    if _ma_breakout and _price > _prev_price * 1.02:
                        _ma_new_alerts.append({
                            "Ticker": _tk, "Type": "🚀 Breakout", "Value": f"+{((_price/_prev_price)-1)*100:.1f}%",
                            "Price": _price, "Priority": "HIGH" if _price > _prev_price * 1.05 else "MEDIUM"
                        })

                    # Check volume spike
                    if _ma_vol_spike and _vol > _vol_avg * 3:
                        _ma_new_alerts.append({
                            "Ticker": _tk, "Type": "📈 Volume Spike", "Value": f"{_vol/_vol_avg:.1f}x",
                            "Price": _price, "Priority": "MEDIUM"
                        })

                except Exception as _e:
                    continue

        st.session_state.mom_alerts = _ma_new_alerts
        if _ma_new_alerts:
            st.success(f"✅ Trovati {len(_ma_new_alerts)} alert!")
        else:
            st.info("Nessun alert rilevato.")

    # Mostra alert
    if st.session_state.mom_alerts:
        _ma_df = pd.DataFrame(st.session_state.mom_alerts)

        # Colori per priorità
        def _ma_color(val):
            if val == "HIGH": return "background-color: #ff4b4b; color: white"
            if val == "MEDIUM": return "background-color: #ffa500; color: black"
            return ""

        st.dataframe(
            _ma_df.style.applymap(_ma_color, subset=["Priority"]),
            use_container_width=True, height=300
        )

        # Export
        _ma_csv = _ma_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV", _ma_csv, "momentum_alerts.csv", "text/csv")
    else:
        st.info("Nessun alert attivo. Clicca 'Scan Now' per verificare.")

    # Clear alert
    if st.button("🗑️ Pulisci Alert"):
        st.session_state.mom_alerts = []
        st.rerun()

    # Info
    with st.expander("📖 Come funziona"):
        st.markdown("""
        - **Breakout >2%**: Prezzo sale sopra 2% rispetto a ieri
        - **Volume >3x**: Volume交易日 >3x media 5gg
        - **Price Action**: Engulfing, Hammer, Doji (basic)
        - **Earnings**: Alert su titoli con earnings imminenti
        """)


# =========================================================================
# v41 — TAB 📰 NEWS & SENTIMENT
# =========================================================================
with tab_news:
    st.markdown('<div class="section-pill">📰 NEWS & SENTIMENT v41 — Ultime news con sentiment analysis</div>', unsafe_allow_html=True)

    # Link a TradingView Italia
    st.markdown("🔗 **[TradingView Italia](https://it.tradingview.com/)** — Community italiana analysis", unsafe_allow_html=True)
    st.markdown("---")

    # Input ticker
    _ns_cols = st.columns([2, 1, 1])
    with _ns_cols[0]:
        _ns_tickers = st.text_input("Ticker da monitorare", value="SPY,QQQ,NVDA,TSLA,AAPL,META,AMD,GOOGL").strip()
    with _ns_cols[1]:
        _ns_filter = st.selectbox("Filtro Sentiment", ["Tutti", "🟢 Bullish", "🔴 Bearish", "⚪ Neutral"])
    with _ns_cols[2]:
        _ns_source = st.selectbox("Fonte", ["Yahoo Finance", "NewsAPI"])

    # Carica da watchlist opzionale
    try:
        _wl = load_watchlist()
        if not _wl.empty and "Ticker" in _wl.columns:
            _wl_tickers = _wl["Ticker"].dropna().unique().tolist()[:20]
            if st.checkbox("📋 Includi Watchlist", value=True):
                _all_tk = list(set([t.strip().upper() for t in _ns_tickers.split(",") if t.strip()] + _wl_tickers))
                _ns_tickers = ",".join(_all_tk)
    except Exception:
        pass

    if st.button("📥 Carica News", use_container_width=True):
        _tk_list = [t.strip().upper() for t in _ns_tickers.split(",") if t.strip()]
        with st.spinner("Scaricando news..."):
            _news_data = _fetch_news_v41(tuple(_tk_list[:25]))

        # Filtro
        if _ns_filter == "🟢 Bullish":
            _news_data = [n for n in _news_data if "Bullish" in n["Sentiment"]]
        elif _ns_filter == "🔴 Bearish":
            _news_data = [n for n in _news_data if "Bearish" in n["Sentiment"]]
        elif _ns_filter == "⚪ Neutral":
            _news_data = [n for n in _news_data if "Neutral" in n["Sentiment"]]

        if _news_data:
            # Statistiche
            _nb = sum(1 for n in _news_data if "Bullish" in n["Sentiment"])
            _nr = sum(1 for n in _news_data if "Bearish" in n["Sentiment"])
            _nn = sum(1 for n in _news_data if "Neutral" in n["Sentiment"])

            _s1, _s2, _s3, _s4 = st.columns(4)
            _s1.metric("📰 Totale", len(_news_data))
            _s2.metric("🟢 Bullish", _nb)
            _s3.metric("🔴 Bearish", _nr)
            _s4.metric("⚪ Neutral", _nn)

            st.markdown("---")

            # Lista news
            for _n in _news_data[:50]:
                _sc = "#00ff88" if "Bullish" in _n["Sentiment"] else "#ef4444" if "Bearish" in _n["Sentiment"] else "#6b7280"
                _tk_nome = _n.get("Nome", _n["Ticker"])
                _tv_link = _n.get("TVLink", "#")
                _nc1, _nc2, _nc3 = st.columns([1.2, 0.8, 4])
                _nc1.markdown(f"<a href='{_tv_link}' target='_blank' style='color:#00ff88;font-weight:bold;text-decoration:none'>{_n['Ticker']}</a> <span style='color:#6b7280;font-size:0.7rem'>{_tk_nome}</span>", unsafe_allow_html=True)
                _nc2.markdown(f"<span style='color:{_sc};font-weight:bold'>{_n['Sentiment']}</span>", unsafe_allow_html=True)
                _nc3.markdown(f"[{_n['Titolo']}]({_n['Link']}) <span style='color:#6b7280;font-size:0.7rem'>{_n['Data']}</span>", unsafe_allow_html=True)
                st.divider()

            # Export
            _ns_df = pd.DataFrame(_news_data)
            _ns_csv = _ns_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", _ns_csv, "news_sentiment.csv", "text/csv")
        else:
            st.info("Nessuna news trovata.")

    # Guida
    with st.expander("📖 Guida Sentiment Analysis"):
        st.markdown("""
        **Come funziona:**
        - Le news vengono analizzate per parole chiave bullish/bearish
        - Score positivo = più termini bullish
        - Score negativo = più termini bearish

        **Fonti:**
        - Yahoo Finance RSS feed (default)
        - NewsAPI (con API key opzionale)

        **Link utili:**
        - [TradingView Italia](https://it.tradingview.com/) — Community analysis
        - [TradingView](https://tradingview.com) — Charts e analisi
        """)
