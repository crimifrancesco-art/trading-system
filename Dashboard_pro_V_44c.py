# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║         TRADING SCANNER PRO  —  v41.0c                                  ║
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

# ── v43c helper link TradingView ────────────────────────────────────────
def _tv_link43c(ticker, label=None):
    sym = (ticker.replace('.MI','%3AMI').replace('.L','%3AL')
                 .replace('.PA','%3APA').replace('.AS','%3AAS'))
    lbl = label or ticker
    return (f"<a style='color:#00ff88;font-family:Courier New;font-weight:bold;"
            f"text-decoration:none;font-size:0.88rem' "
            f"href='https://it.tradingview.com/chart/?symbol={sym}' "
            f"target='_blank'>{lbl} "
            f"<span style='color:#2962ff;font-size:0.65rem'>&#8599;</span></a>")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── v43d: AI config globale — definite prima di ogni uso ──────────────────
def _get_global_ai_keys():
    """Legge le API key AI — una sola fonte per tutti i moduli."""
    _ss = st.session_state
    return {
        "gemini":     st.secrets.get("GEMINI_API_KEY","")     or _ss.get("_gemini_api_key",""),
        "groq":       st.secrets.get("GROQ_API_KEY","")       or _ss.get("_groq_api_key",""),
        "openrouter": st.secrets.get("OPENROUTER_API_KEY","") or _ss.get("_openrouter_api_key",""),
        "anthropic":  st.secrets.get("ANTHROPIC_API_KEY","")  or _ss.get("_anthropic_api_key",""),
    }

def _has_global_ai_config():
    """True se almeno un provider AI è configurato."""
    return any(bool(v) for v in _get_global_ai_keys().values())
# ────────────────────────────────────────────────────────────────────────────
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
    _rate_limited = False
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
        except Exception as _earn_err:
            _msg = str(_earn_err).lower()
            if "too many requests" in _msg or "rate limit" in _msg or "429" in _msg:
                _rate_limited = True
            continue
    if _rate_limited and not _results:
        return [{"Ticker":"RATE_LIMIT","Earnings Date":"—","Giorni":999,
                 "Badge":"⏳ Yahoo rate limited — riprova tra poco","_color":"#f59e0b",
                 "EPS Est":"","Rev Est":""}]
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

BACK_TO_TOP_CSS = """<style>
#btt-btn {
    position:fixed !important;
    bottom:24px !important;
    left:50% !important;
    transform:translateX(-50%) translateY(20px) !important;
    z-index:2147483647 !important;
    background:#2962ff;
    color:#fff;
    border:none;
    border-radius:24px;
    width:auto;
    min-width:120px;
    height:44px;
    padding:0 22px;
    font-size:0.92rem;
    font-weight:bold;
    font-family:'Trebuchet MS',sans-serif;
    cursor:pointer;
    box-shadow:0 4px 24px rgba(41,98,255,0.55);
    display:flex !important;
    align-items:center;
    justify-content:center;
    gap:6px;
    opacity:0;
    transition:opacity .3s ease, transform .3s ease;
    pointer-events:none;
    white-space:nowrap;
    letter-spacing:0.3px;
}
#btt-btn.btt-visible {
    opacity:1 !important;
    transform:translateX(-50%) translateY(0) !important;
    pointer-events:all !important;
}
#btt-btn:hover {
    background:#1a3fd4 !important;
    box-shadow:0 6px 28px rgba(41,98,255,0.75) !important;
}
#btt-btn:active { transform:translateX(-50%) scale(0.96) !important; }

/* v44b — responsive tabs & layout */
[data-testid="stTabs"] button{white-space:nowrap;}
@media(max-width:1200px){[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:.4rem;}}
@media(max-width:768px){div[data-testid="metric-container"]{min-width:140px;}.stDataFrame,[data-testid="stTable"]{font-size:.82rem;}iframe,canvas,.js-plotly-plot{max-width:100%!important;}}
@media(max-width:640px){[data-testid="stTabs"] button{padding:.35rem .55rem!important;font-size:.78rem!important;}div[data-testid="column"]{width:100%!important;flex:1 1 100%!important;}.element-container .stButton>button{width:100%;}}
</style>
<button id='btt-btn' title='Torna all\'inizio'>&#8679; Torna su</button>
<script>
(function(){
  var _btt_attached=false;
  function _scroll_top(){
    var doc=window.parent.document;
    var selectors=[
      '[data-testid="stAppViewContainer"]',
      '.main','section.main',
      '[data-testid="block-container"]','.block-container'
    ];
    for(var s=0;s<selectors.length;s++){
      var el=doc.querySelector(selectors[s]);
      if(el && el.scrollTop>0){ el.scrollTo({top:0,behavior:'smooth'}); }
    }
    if(window.parent.pageYOffset>0) window.parent.scrollTo({top:0,behavior:'smooth'});
    doc.documentElement.scrollTo({top:0,behavior:'smooth'});
    doc.body.scrollTo({top:0,behavior:'smooth'});
    window.scrollTo({top:0,behavior:'smooth'});
  }
  function _getBtn(){
    return window.parent.document.getElementById('btt-btn')||document.getElementById('btt-btn');
  }
  function _getScrollTop(){
    var doc=window.parent.document;
    var selectors=[
      '[data-testid="stAppViewContainer"]',
      '.main','section.main',
      '[data-testid="block-container"]','.block-container'
    ];
    var max=0;
    for(var s=0;s<selectors.length;s++){
      var el=doc.querySelector(selectors[s]);
      if(el && el.scrollTop>max) max=el.scrollTop;
    }
    return Math.max(max, window.parent.pageYOffset||0, window.pageYOffset||0);
  }
  function _btt_check(){
    var b=_getBtn(); if(!b) return;
    if(_getScrollTop()>200) b.classList.add('btt-visible');
    else b.classList.remove('btt-visible');
  }
  function _btt_attach(){
    if(_btt_attached) return;
    var doc=window.parent.document;
    var selectors=[
      '[data-testid="stAppViewContainer"]',
      '.main','section.main',
      '[data-testid="block-container"]','.block-container'
    ];
    var attached=false;
    for(var s=0;s<selectors.length;s++){
      var el=doc.querySelector(selectors[s]);
      if(el){ el.addEventListener('scroll',_btt_check,{passive:true}); attached=true; }
    }
    window.parent.addEventListener('scroll',_btt_check,{passive:true});
    window.addEventListener('scroll',_btt_check,{passive:true});
    _btt_check();
    if(attached) _btt_attached=true;
  }
  function _btt_init(){
    var b=_getBtn();
    if(b && !b._btt_ready){
      b.addEventListener('click',_scroll_top);
      b._btt_ready=true;
    }
  }
  [100,300,700,1500,3000,6000].forEach(function(d){
    setTimeout(function(){ _btt_attach(); _btt_init(); },d);
  });
  try{
    var obs=new MutationObserver(function(){ _btt_attach(); _btt_init(); _btt_check(); });
    obs.observe(window.parent.document.body,{childList:true,subtree:false});
  }catch(e){}
})();
</script>"""
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
/* ══════════════════════════════════════════════════════════════════════════
   v43 RESPONSIVE & ADAPTIVE — Mobile-first · Fluid · All Tabs
   ══════════════════════════════════════════════════════════════════════════ */
*,*::before,*::after{box-sizing:border-box;}
[data-testid="block-container"]{
    max-width:100%!important;
    padding:clamp(0.5rem,2vw,1.5rem) clamp(0.5rem,2vw,2rem)!important;}
[data-testid="stTabs"]>div:first-child{
    overflow-x:auto!important;overflow-y:hidden!important;
    -webkit-overflow-scrolling:touch!important;
    scrollbar-width:none!important;
    flex-wrap:nowrap!important;white-space:nowrap!important;padding-bottom:2px!important;}
[data-testid="stTabs"]>div:first-child::-webkit-scrollbar{display:none!important;}
[data-testid="stTabs"] button{
    font-size:clamp(0.60rem,1.2vw,0.83rem)!important;
    padding:clamp(3px,0.8vw,7px) clamp(4px,1vw,12px)!important;
    white-space:nowrap!important;flex-shrink:0!important;}
@media(max-width:640px){
    [data-testid="column"]{width:100%!important;flex:0 0 100%!important;min-width:0!important;}}
@media(min-width:641px) and (max-width:900px){
    [data-testid="column"]{min-width:min(200px,45%)!important;}}
@media(max-width:480px){
    [data-testid="stMetric"]{padding:8px 10px!important;}
    [data-testid="stMetricValue"]{font-size:1.15rem!important;}
    [data-testid="stMetricLabel"]{font-size:0.65rem!important;}
    [data-testid="stMetricDelta"]{font-size:0.70rem!important;}
    [data-testid="stButton"]>button{min-height:44px!important;font-size:0.85rem!important;}
    [data-testid="stTextInput"] input,[data-testid="stSelectbox"] select,
    [data-testid="stNumberInput"] input{font-size:16px!important;min-height:44px!important;}
    [data-testid="stSlider"]{padding:6px 0!important;}
    .section-pill{font-size:0.72rem!important;padding:4px 10px!important;}
    .hide-mobile{display:none!important;}
    [data-testid="stExpander"]>div>div{padding:8px 10px!important;}}
@media(min-width:481px) and (max-width:768px){
    [data-testid="stMetricValue"]{font-size:1.3rem!important;}
    [data-testid="stButton"]>button{min-height:40px!important;}}
[data-testid="stDataFrame"],[data-testid="stTable"],.ag-root-wrapper{
    overflow-x:auto!important;-webkit-overflow-scrolling:touch!important;max-width:100%!important;}
@media(max-width:768px){[data-testid="stPlotlyChart"]{width:100%!important;}}
@media(max-width:768px){
    [data-testid="stSidebarContent"]{padding:0.5rem!important;}
    [data-testid="stSidebar"]{min-width:0!important;}}
.spark-row{display:flex;align-items:center;gap:6px;padding:4px 0;
    border-bottom:1px solid #1f2937;flex-wrap:wrap;}
.spark-row:last-child{border-bottom:none;}
.spark-ticker{font-family:'Courier New',monospace;font-weight:bold;
    color:#00ff88;font-size:0.88rem;min-width:70px;}
.spark-badge{display:inline-block;border-radius:3px;padding:1px 6px;
    font-size:0.68rem;font-weight:bold;margin-left:2px;}
.spark-css{color:#b2b5be;font-size:0.78rem;min-width:40px;}
.spark-svg{flex-shrink:0;}
.refresh-banner{display:flex;align-items:center;gap:8px;
    background:#0d1117;border:1px solid #1f2937;border-radius:6px;
    padding:6px 14px;font-size:0.78rem;color:#6b7280;margin-bottom:8px;}
.refresh-banner strong{color:#50c4e0;}
.earn-row{display:grid;grid-template-columns:90px 1fr 60px 46px 72px;
    gap:8px;align-items:center;padding:7px 10px;
    border-bottom:1px solid #1a2233;font-size:0.86rem;}
@media(max-width:768px){.earn-row{grid-template-columns:78px 1fr 56px 40px 64px;font-size:0.80rem;}}
@media(max-width:480px){.earn-row{grid-template-columns:70px 1fr 50px 36px;font-size:0.75rem;}}
.earn-row:hover{background:#131d2e;}
.earn-ticker{font-family:'Courier New',monospace;color:#00ff88;font-weight:bold;font-size:0.90rem;}
.earn-ticker a{color:#00ff88;text-decoration:none;}
.earn-ticker a:hover{color:#26c6da;text-decoration:underline;}
.earn-date{color:#50c4e0;font-size:0.80rem;}
#theme-toggle-btn{position:fixed;top:14px;right:70px;z-index:9999;
    background:#1e222d;border:1px solid #363a45;border-radius:20px;
    padding:4px 12px;font-size:0.75rem;color:#d1d4dc;cursor:pointer;transition:all 0.2s;}
#theme-toggle-btn:hover{border-color:#50c4e0;color:#50c4e0;}
body.light-theme [data-testid="stAppViewContainer"],
body.light-theme [data-testid="stMain"],
body.light-theme [data-testid="block-container"]{background-color:#f8fafc!important;color:#1e293b!important;}
body.light-theme [data-testid="stSidebar"]{background-color:#e2e8f0!important;}
body.light-theme [data-testid="stSidebar"] *{color:#334155!important;}
body.light-theme [data-testid="stMetric"]{background:#fff!important;border-color:#e2e8f0!important;}
body.light-theme [data-testid="stMetricValue"]{color:#0f766e!important;}
body.light-theme [data-testid="stTabs"] button{background:#f1f5f9!important;color:#64748b!important;}
body.light-theme [data-testid="stTabs"] button[aria-selected="true"]{color:#2962ff!important;background:#fff!important;}
body.light-theme h1{color:#2962ff!important;text-shadow:none!important;}
body.light-theme h2,body.light-theme h3{color:#0f766e!important;}
body.light-theme .section-pill{background:#e0f2fe;border-left-color:#2962ff;color:#0f766e;}
body.light-theme .ag-root-wrapper{background:#fff!important;}
body.light-theme .ag-header{background:#f1f5f9!important;}
body.light-theme .ag-row{background:#fff!important;border-bottom:1px solid #e2e8f0!important;}
body.light-theme .ag-row:hover{background:#f8fafc!important;}
body.light-theme .ag-cell{color:#1e293b!important;}
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

def make_tv_txt(df, tab=""):
    """Un ticker per riga — formato import TradingView."""
    if df is None or (hasattr(df, 'empty') and df.empty): return b""
    _col = next((c for c in ["Ticker","ticker","Symbol","symbol"] if hasattr(df,'columns') and c in df.columns), None)
    if _col is None: return b""
    _v = list(dict.fromkeys([str(x).strip().upper() for x in df[_col].dropna() if str(x).strip()]))
    return ("\n".join(_v)).encode("utf-8")

def tv_txt_btn(df, fname, key, label="📥 TXT TradingView"):
    """Download TXT TradingView (un ticker per riga)."""
    _f = fname[:-4]+".txt" if fname.lower().endswith(".csv") else fname
    st.download_button(label, make_tv_txt(df), _f, "text/plain", key=key)

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
st.set_page_config(page_title="Trading Scanner PRO 44.0b",layout="wide",page_icon="🧠")
st.markdown(DARK_CSS,unsafe_allow_html=True)
# ── v44a scroll-to-top ──────────────────────────────────────────────────
st.markdown("""
<style>
#v43c-top{position:fixed;right:16px;top:50%;transform:translateY(-50%);z-index:999999;
  background:#1e222d;border:1px solid #363a45;border-radius:999px;
  width:46px;height:46px;display:flex;align-items:center;justify-content:center;
  color:#50c4e0;font-size:20px;text-decoration:none;cursor:pointer;opacity:0;pointer-events:none;
  box-shadow:0 4px 16px rgba(0,0,0,0.45);transition:opacity .18s ease, background .18s ease, color .18s ease, transform .18s ease;
  line-height:1;}
#v43c-top.v43c-top-visible{opacity:1;pointer-events:auto;}
#v43c-top:hover{background:#2962ff;color:#fff;transform:translateY(-50%) scale(1.04);}
@media (max-width: 768px){
  #v43c-top{right:12px;width:44px;height:44px;top:auto;bottom:88px;transform:none;}
  #v43c-top:hover{transform:scale(1.03);}
}
</style>
<a id='v43c-top' href='javascript:void(0)' title='Torna in cima' aria-label='Torna in cima'>&#9650;</a>
<script>
(function(){
  function getBtn(){
    return document.getElementById('v43c-top') || (window.parent && window.parent.document ? window.parent.document.getElementById('v43c-top') : null);
  }
  function getScrollTargets(){
    var out=[];
    try{
      var pdoc=window.parent && window.parent.document ? window.parent.document : document;
      [
        pdoc.querySelector('.main'),
        pdoc.querySelector('section.main'),
        pdoc.querySelector('[data-testid="stAppViewContainer"]'),
        pdoc.querySelector('[data-testid="stAppViewBlockContainer"]'),
        pdoc.querySelector('.block-container'),
        pdoc.scrollingElement,
        pdoc.documentElement,
        pdoc.body,
        document.scrollingElement,
        document.documentElement,
        document.body
      ].forEach(function(el){ if(el && out.indexOf(el)===-1) out.push(el); });
    }catch(e){}
    return out;
  }
  function scrollTopNow(){
    getScrollTargets().forEach(function(el){ try{ el.scrollTo({top:0, behavior:'smooth'}); }catch(e){} try{ el.scrollTop=0; }catch(e){} });
    try{ window.scrollTo({top:0, behavior:'smooth'}); }catch(e){}
    try{ window.parent.scrollTo({top:0, behavior:'smooth'}); }catch(e){}
  }
  function currentScroll(){
    var mx=0;
    getScrollTargets().forEach(function(el){ try{ mx=Math.max(mx, el.scrollTop||0); }catch(e){} });
    try{ mx=Math.max(mx, window.pageYOffset||0); }catch(e){}
    try{ mx=Math.max(mx, window.parent.pageYOffset||0); }catch(e){}
    return mx;
  }
  function refreshBtn(){
    var b=getBtn(); if(!b) return;
    if(currentScroll()>220) b.classList.add('v43c-top-visible');
    else b.classList.remove('v43c-top-visible');
  }
  function attach(){
    var b=getBtn();
    if(!b || b.dataset.bound==='1') return;
    b.dataset.bound='1';
    b.addEventListener('click', function(ev){ ev.preventDefault(); scrollTopNow(); });
    getScrollTargets().forEach(function(el){ try{ el.addEventListener('scroll', refreshBtn, {passive:true}); }catch(e){} });
    try{ window.addEventListener('scroll', refreshBtn, {passive:true}); }catch(e){}
    try{ window.parent.addEventListener('scroll', refreshBtn, {passive:true}); }catch(e){}
    refreshBtn();
  }
  [100,300,800,1500,3000].forEach(function(t){ setTimeout(function(){ attach(); refreshBtn(); }, t); });
  try{
    var mo=new MutationObserver(function(){ attach(); refreshBtn(); });
    mo.observe(document.body,{childList:true,subtree:true});
  }catch(e){}
})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# v43c — WEB PUSH NOTIFICATIONS helper (JS browser)
# ══════════════════════════════════════════════════════════════════════════════
_PUSH_JS = """
<script>
(function(){
    const PUSH_KEY = 'v43c_push_enabled';
    window._v43cPushEnabled = sessionStorage.getItem(PUSH_KEY)==='1';
    window._v43cRequestPush = function(){
        if(!('Notification' in window)){return;}
        Notification.requestPermission().then(p=>{
            if(p==='granted'){
                sessionStorage.setItem(PUSH_KEY,'1');
                window._v43cPushEnabled=true;
                new Notification('Trading Scanner PRO 43b',{
                    body:'✅ Alert push attivati — segnali PRO/STRONG in tempo reale.',
                    icon:'https://cdn.simpleicons.org/bitcoin'
                });
            }
        });
    };
    window._v43cSendPush = function(title,body){
        if(window._v43cPushEnabled && Notification.permission==='granted'){
            new Notification(title,{body:body});
        }
    };
    document.addEventListener('DOMContentLoaded',function(){
        var btn=document.getElementById('v43c-push-btn');
        if(btn){
            btn.onclick=window._v43cRequestPush;
            if(window._v43cPushEnabled){btn.innerText='🔔 Push ON';}
        }
    });
})();
</script>
"""
st.markdown(_PUSH_JS, unsafe_allow_html=True)

st.markdown(BACK_TO_TOP_CSS,unsafe_allow_html=True)

# ── v43: TEMA TOGGLE persistente per sessione ────────────────────────────────
st.markdown("""
<button id='theme-toggle-btn'
  onclick='(function(){
    var b=window.parent.document.body||document.body;
    if(b.classList.contains("light-theme")){
        b.classList.remove("light-theme");
        document.getElementById("theme-toggle-btn").textContent="\u2600 Light";
    }else{
        b.classList.add("light-theme");
        document.getElementById("theme-toggle-btn").textContent="\U0001f319 Dark";
    }
  })()'>&#9728; Light</button>
""", unsafe_allow_html=True)

# ── v43: AUTO-REFRESH schedulato ─────────────────────────────────────────────
import time as _time_v43
if "v43_last_refresh" not in st.session_state:
    st.session_state["v43_last_refresh"] = _time_v43.time()
if "v43_autorefresh_mins" not in st.session_state:
    st.session_state["v43_autorefresh_mins"] = 0
_v43_ar_mins = st.session_state.get("v43_autorefresh_mins", 0)
if _v43_ar_mins > 0:
    _v43_elapsed = _time_v43.time() - st.session_state["v43_last_refresh"]
    if _v43_elapsed >= _v43_ar_mins * 60:
        st.session_state["v43_last_refresh"] = _time_v43.time()
        st.rerun()

st.markdown("# 🧠 Trading Scanner PRO 44.0b")
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

# ── v41e: Backup/Restore watchlist su file JSON locale ───────────────────────
# Garantisce persistenza anche senza GitHub Sync (Streamlit Cloud usa /tmp/
# ma i secrets possono montare un volume persistente)
import json as _json_wl, pathlib as _pl_wl, os as _os_wl

# Cerca cartella persistente: prima /data (se montata), poi /tmp
_WL_BACKUP_DIR = _pl_wl.Path("/data") if _pl_wl.Path("/data").exists() else _pl_wl.Path("/tmp")
_WL_BACKUP_FILE = _WL_BACKUP_DIR / "watchlist_backup_v41b.json"

def _wl_save_backup():
    """Salva watchlist completa in JSON per persistenza cross-deploy."""
    try:
        _df_bk = load_watchlist()
        if not _df_bk.empty:
            _records = _df_bk.to_dict(orient="records")
            _WL_BACKUP_FILE.write_text(
                _json_wl.dumps(_records, indent=2, default=str), encoding="utf-8")
    except Exception as _bke:
        pass  # silenzioso — non blocca l'app

def _wl_restore_backup():
    """Ripristina watchlist dal backup JSON se DB è vuoto."""
    try:
        if not _WL_BACKUP_FILE.exists():
            return 0
        _records = _json_wl.loads(_WL_BACKUP_FILE.read_text(encoding="utf-8"))
        if not _records:
            return 0
        _conn_bk = sqlite3.connect(str(DB_PATH))
        _restored = 0
        for _r in _records:
            _tkr_bk = str(_r.get("ticker","") or _r.get("Ticker",""))
            _lst_bk = str(_r.get("list_name","DEFAULT"))
            if not _tkr_bk:
                continue
            _exists = _conn_bk.execute(
                "SELECT id FROM watchlist WHERE ticker=? AND list_name=?",
                (_tkr_bk, _lst_bk)
            ).fetchone()
            if not _exists:
                _conn_bk.execute(
                    "INSERT INTO watchlist (ticker,name,trend,origine,note,list_name,created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (_tkr_bk,
                     str(_r.get("name","") or _r.get("Nome","")),
                     str(_r.get("trend","")),
                     str(_r.get("origine","backup")),
                     str(_r.get("note","")),
                     _lst_bk,
                     str(_r.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
                )
                _restored += 1
        _conn_bk.commit(); _conn_bk.close()
        return _restored
    except Exception:
        return 0

# Ripristina backup se il DB è vuoto al boot (es. dopo deploy)
if not st.session_state.get("_wl_backup_checked"):
    try:
        _wl_check = load_watchlist()
        if _wl_check.empty or len(_wl_check) == 0:
            _n_restored = _wl_restore_backup()
            if _n_restored > 0:
                st.toast(f"💾 Watchlist ripristinata dal backup locale: {_n_restored} ticker", icon="💾")
    except Exception:
        pass
    st.session_state["_wl_backup_checked"] = True

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
_ai_providers_status = {k: bool(v) for k,v in zip(
    ["🟢 Gemini","🟣 Groq","🔵 OpenRouter","🟡 Claude"],
    _get_global_ai_keys().values())}
_n_active = sum(_ai_providers_status.values())
_ai_status_lines = "  ".join(
    ("<span style='color:" + ("#00ff88" if ok else "#374151") + "'>" + name.split()[0] + "</span>")
    for name, ok in _ai_providers_status.items()
)
_bg = "#0d2b1f" if _n_active > 0 else "#1a0f00"
_bc = "#00ff88" if _n_active > 0 else "#f59e0b"
_msg = f"{_n_active}/4 provider attivi" if _n_active > 0 else "nessun provider — configura AI nel tab PRO"
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
    if st.button("🚀 AVVIA SCANNER PRO 41.0",type="primary",use_container_width=True):
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
    with ce1: tv_txt_btn(df_f,f"{title.lower().replace(' ','_')}.txt",f"exp_{title}")
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
            _wl_save_backup()  # v41e: backup locale immediato
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
# ── v41e: Menu sticky multi-riga — tutti i tab visibili su 3 righe ──────
st.markdown("""<style>
/* Sticky tab bar */
[data-testid="stTabs"] {
    position: sticky !important;
    top: 0 !important;
    z-index: 999 !important;
    background-color: #131722 !important;
    padding-top: 2px !important;
    border-bottom: 2px solid #2a2e39 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}
/* Multi-riga: wrap + altezza auto */
[data-testid="stTabs"] > div:first-child {
    flex-wrap: wrap !important;
    gap: 0px !important;
    overflow: visible !important;
    background-color: #131722 !important;
    max-height: none !important;
    height: auto !important;
}
/* Tab button base — compatto ma leggibile */
[data-testid="stTabs"] > div:first-child > button {
    flex-shrink: 0 !important;
    min-width: fit-content !important;
    max-width: 160px !important;
    font-size: 0.73rem !important;
    padding: 5px 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: background 0.12s, color 0.12s !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: 2px !important;
}
/* Tab attivo */
[data-testid="stTabs"] > div:first-child > button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #2962ff !important;
    background: rgba(41,98,255,0.15) !important;
    font-weight: 700 !important;
}
/* Hover */
[data-testid="stTabs"] > div:first-child > button:hover {
    background: rgba(41,98,255,0.08) !important;
    color: #e2e8f0 !important;
    border-bottom: 2px solid #2962ff66 !important;
}
/* Watchlist — evidenziata in fondo */
[data-testid="stTabs"] > div:first-child > button:last-child {
    color: #ffd700 !important;
    border-bottom: 2px solid #ffd70044 !important;
}
[data-testid="stTabs"] > div:first-child > button:last-child[aria-selected="true"] {
    color: #ffd700 !important;
    border-bottom: 3px solid #ffd700 !important;
    background: rgba(255,215,0,0.10) !important;
}
/* Font adattivo */
@media (max-width: 1400px) {
    [data-testid="stTabs"] > div:first-child > button { font-size: 0.70rem !important; padding: 4px 8px !important; }
}
@media (max-width: 1000px) {
    [data-testid="stTabs"] > div:first-child > button { font-size: 0.65rem !important; padding: 3px 6px !important; }
}
@media (max-width: 700px) {
    [data-testid="stTabs"] > div:first-child > button { font-size: 0.60rem !important; padding: 3px 5px !important; }
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
    st.download_button("📥 TXT TradingView",make_tv_txt(_df_exp),f"Alert_v41_{_at_ts}.txt","text/plain",key=f"alert_exp_{tab_name}")

# ── v41 #2: News & Sentiment ───────────────────────────────────────────────
@st.cache_data(ttl=600)
def _fetch_news_v41(tickers:tuple)->list:
    import urllib.request as _ur, xml.etree.ElementTree as _ET
    _BULL={"surge","rally","soar","beat","record","upgrade","buy","bullish","outperform","strong","growth","profit","revenue","exceed","positive","gain","rise","boost","breakout"}
    _BEAR={"crash","fall","drop","miss","downgrade","sell","bearish","underperform","weak","loss","decline","below","negative","cut","reduce","layoff","concern","risk","warning","plunge"}
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
                _res.append({"Ticker":_t,"Titolo":_title[:80],"Sentiment":_sent,"Score":_b-_br,
                             "Data":_item.findtext("pubDate","")[:16],"Link":_item.findtext("link","")})
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
        _c1,_c2,_c3=st.columns([1,0.8,4.5])
        _tv_sym_n = str(n['Ticker']).replace('.MI', '%3AMI')
        _nome_n = str(n.get('Nome', '')).strip()[:22]
        _nome_n_html = (f" <span style='color:#6b7280;font-size:0.70rem;font-style:italic'>{_nome_n}</span>" if _nome_n else '')
        _c1.markdown(
            f"<a href='https://it.tradingview.com/chart/?symbol={_tv_sym_n}' target='_blank' style='text-decoration:none'>"
            f"<span style='font-family:Courier New;color:#00ff88;font-weight:bold'>{n['Ticker']}</span></a>"
            f"{_nome_n_html}", unsafe_allow_html=True)
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
    st.markdown('<div class="section-pill">🤖 MODULO 2 — AI ANALYST · Setup · Target · Invalidazione · Rischio</div>', unsafe_allow_html=True)
    st.caption("Fallback automatico: Gemini (free) → Groq (free) → OpenRouter → Claude · Clicca 🧠 Analizza su ogni ticker")
    st.markdown("""
    <style>
    .ai2-grid{border:1px solid #1f2937;border-radius:12px;overflow:hidden;background:#0b1220}
    .ai2-grid .head,.ai2-grid .row{display:grid;grid-template-columns:1.35fr .72fr .68fr .68fr 1.05fr;gap:10px;align-items:center}
    .ai2-grid .head{padding:10px 12px;background:#0b1326;border-bottom:1px solid #1f2937;color:#38bdf8;font-size:.74rem;font-weight:700;letter-spacing:.04em}
    .ai2-grid .row{padding:12px 12px;border-bottom:1px solid rgba(42,46,57,.75)}
    .ai2-grid .row:last-child{border-bottom:none}
    .ai2-tkr{font-family:Courier New,monospace;font-weight:800;color:#00ff88;font-size:1.03rem;line-height:1.05}
    .ai2-nm{color:#8b95a7;font-size:.76rem;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:280px}
    .ai2-st{display:inline-block;padding:4px 10px;border-radius:8px;font-size:.74rem;font-weight:800;border:1px solid transparent}
    .ai2-st.pro{color:#00ff88;background:rgba(0,255,136,.08);border-color:rgba(0,255,136,.24)}
    .ai2-st.strong{color:#22c55e;background:rgba(34,197,94,.10);border-color:rgba(34,197,94,.26)}
    .ai2-st.confluence{color:#fbbf24;background:rgba(251,191,36,.10);border-color:rgba(251,191,36,.24)}
    .ai2-m{font-family:Courier New,monospace;font-weight:700}
    .ai2-btn{display:block;text-align:center;padding:10px 12px;border-radius:9px;border:1px solid #374151;background:#111827;color:#f9a8d4;font-weight:700}
    .ai2-btn:hover{border-color:#4b5563;background:#141c2b}
    .ai2-note{color:#787b86;font-size:.77rem}
    @media (max-width: 1080px){.ai2-grid .head,.ai2-grid .row{grid-template-columns:1.15fr .75fr .65fr .65fr 1.05fr}}
    </style>
    """, unsafe_allow_html=True)

    # ── Pannello configurazione API keys ──────────────────────────────────
    _any_key = _has_global_ai_config()

    with st.expander(
        "🔑 AI Config globale" + (" ✅ attiva" if _any_key else " ⚠️ nessuna key — configura qui"),
        expanded=not _any_key
    ):
        st.markdown(
            "<div style='background:#0d1117;border:1px solid #1f2937;border-radius:6px;"
            "padding:10px 14px;margin-bottom:10px;font-size:0.80rem;color:#b2b5be'>"
            "Configura almeno una key. Il sistema usa quella disponibile con fallback automatico.<br><br>"
            "🟢 <b>Gemini Flash</b> — gratis · <a href='https://aistudio.google.com' target='_blank' "
            "style='color:#2962ff'>aistudio.google.com</a> → Get API Key<br>"
            "🟣 <b>Groq</b> — gratis · <a href='https://console.groq.com' target='_blank' "
            "style='color:#2962ff'>console.groq.com</a> → API Keys → Create<br>"
            "🔵 <b>OpenRouter</b> — free tier · <a href='https://openrouter.ai' target='_blank' "
            "style='color:#2962ff'>openrouter.ai</a> → Keys → Create Key<br>"
            "🟡 <b>Claude</b> — a pagamento · <a href='https://console.anthropic.com' target='_blank' "
            "style='color:#2962ff'>console.anthropic.com</a> → API Keys"
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
            # Groq
            _groq_cur = st.session_state.get("_groq_api_key","")
            _groq_inp = st.text_input("🟣 Groq API Key",
                value=_groq_cur, type="password",
                placeholder="gsk_...",
                key=f"groq_inp_{tab_name}")
        with _kc2:
            # OpenRouter
            _or_cur = st.session_state.get("_openrouter_api_key","")
            _or_inp = st.text_input("🔵 OpenRouter API Key",
                value=_or_cur, type="password",
                placeholder="sk-or-...",
                key=f"or_inp_{tab_name}")
            # Claude
            _ant_cur = st.session_state.get("_anthropic_api_key","")
            _ant_inp = st.text_input("🟡 Claude API Key",
                value=_ant_cur, type="password",
                placeholder="sk-ant-api03-...",
                key=f"ant_inp_{tab_name}")

        _save_col, _reset_col, _ = st.columns([1,1,2])
        with _save_col:
            if st.button("💾 Salva keys", key=f"ai_keys_save_{tab_name}", type="primary",
                         use_container_width=True):
                if _gem_inp.strip():  st.session_state["_gemini_api_key"]     = _gem_inp.strip()
                if _groq_inp.strip(): st.session_state["_groq_api_key"]       = _groq_inp.strip()
                if _or_inp.strip():   st.session_state["_openrouter_api_key"] = _or_inp.strip()
                if _ant_inp.strip():  st.session_state["_anthropic_api_key"]  = _ant_inp.strip()
                st.success("✅ Configurazione AI globale salvata per questa sessione!")
                st.rerun()
        with _reset_col:
            if st.button("🗑️ Reset tutte", key=f"ai_keys_reset_{tab_name}",
                         use_container_width=True):
                for _k in ["_gemini_api_key","_groq_api_key",
                           "_openrouter_api_key","_anthropic_api_key"]:
                    st.session_state.pop(_k, None)
                st.rerun()

        # Status provider
        _prov_status = []
        for _pn, _sk, _ssk in [
            ("🟢 Gemini",    "GEMINI_API_KEY",     "_gemini_api_key"),
            ("🟣 Groq",      "GROQ_API_KEY",        "_groq_api_key"),
            ("🔵 OpenRouter","OPENROUTER_API_KEY",   "_openrouter_api_key"),
            ("🟡 Claude",    "ANTHROPIC_API_KEY",    "_anthropic_api_key"),
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


def _ai_call_with_fallback(prompt: str) -> tuple:
    """
    Prova i provider in ordine: Gemini → Groq → OpenRouter → Claude.
    Restituisce (testo_risposta, provider_usato) o lancia Exception.
    """
    _k = _get_global_ai_keys()
    _gem_key  = _k["gemini"]
    _groq_key = _k["groq"]
    _or_key   = _k["openrouter"]
    _ant_key  = _k["anthropic"]

    _providers = []
    if _gem_key:  _providers.append(("🟢 Gemini Flash",  _ai_call_gemini,     _gem_key))
    if _groq_key: _providers.append(("🟣 Groq Llama",    _ai_call_groq,       _groq_key))
    if _or_key:   _providers.append(("🔵 OpenRouter",    _ai_call_openrouter, _or_key))
    if _ant_key:  _providers.append(("🟡 Claude Haiku",  _ai_call_claude,     _ant_key))

    if not _providers:
        raise Exception("NO_KEYS")

    _errors = []
    for _name, _fn, _key in _providers:
        try:
            _result = _fn(_key, prompt)
            return _result, _name
        except Exception as _e:
            _errors.append(f"{_name}: {_e}")
            continue  # prova il prossimo

    raise Exception("ALL_FAILED: " + " | ".join(_errors))




# ── v41e: Tab su 2 righe ─────────────────────────────────
st.markdown("""
<style>
[data-testid="stTabs"] > div:first-child { display:flex !important; flex-wrap:wrap !important; gap:2px 3px !important; max-height:none !important; overflow:visible !important; width:100% !important; border-bottom:1px solid #2a2e39 !important; padding-bottom:6px !important;}
[data-testid="stTabs"] [data-baseweb="tab"] { white-space:nowrap !important; font-size:0.74rem !important; padding:4px 8px !important; flex-shrink:0 !important; margin-bottom:2px !important; max-width:160px !important; border-radius:4px 4px 0 0 !important; border:1px solid #2a2e3966 !important;}
[data-testid="stTabs"] [aria-selected="true"] { border-bottom:2px solid #2962ff !important; color:#2962ff !important; background:#131722 !important; font-weight:bold !important;}
[data-testid="stTabs"] > div:first-child > div { overflow:visible !important; max-height:none !important;}
[role='tablist'] { display:flex !important; flex-wrap:wrap !important; overflow:visible !important; max-height:none !important; height:auto !important;}
[data-baseweb='tab-list'] { display:flex !important; flex-wrap:wrap !important; overflow:visible !important; max-height:none !important; height:auto !important;}
</style>
<script>
(function fixTabWrap(){
  function apply(){
    ['[data-testid="stTabs"] > div:first-child','[role="tablist"]','[data-baseweb="tab-list"]']
      .forEach(function(s){document.querySelectorAll(s).forEach(function(el){
        el.style.setProperty('display','flex','important');
        el.style.setProperty('flex-wrap','wrap','important');
        el.style.setProperty('max-height','none','important');
        el.style.setProperty('overflow','visible','important');
        el.style.setProperty('height','auto','important');
        var p=el.parentElement;for(var i=0;i<4;i++){if(p){
          p.style.setProperty('overflow','visible','important');
          p.style.setProperty('max-height','none','important');
          p=p.parentElement;}}
      });});
    document.querySelectorAll('[data-baseweb="tab"]').forEach(function(t){
      t.style.setProperty('white-space','nowrap','important');
      t.style.setProperty('flex-shrink','0','important');
      t.style.setProperty('font-size','0.74rem','important');
    });
  }
  apply();
  [200,600,1200,2500].forEach(function(t){setTimeout(apply,t);});
  new MutationObserver(apply).observe(document.body,{childList:true,subtree:true});
})();
</script>
""", unsafe_allow_html=True)

tabs = st.tabs([
    # ── Riga 1: Panoramica e Segnali ───────────────────────
    "🏠 Home",
    "📡 EARLY",
    "💪 PRO",
    "🔥 REA-HOT",
    "⭐ CONFLUENCE",
    "🎯 Serafini",
    "🔎 Finviz Pro",
    "🛡️ Crisis Monitor",
    # ── Riga 2: Analisi e Strumenti ────────────────────────
    "🌡️ Regime",
    "🔀 MTF Matrix",
    "📊 Comparatore",
    "💎 Blue Chip Dip",
    "🔬 Order Flow",
    "⚖️ Risk Manager",
    "📈 Backtest",
    # ── Riga 3: AI e Feature Avanzate ──────────────────────
    "🤖 AI Assistant",
    "🤖 Modulo 2 AI",
    "🎲 Options Scanner",
    "⚡ Momentum Alerts",
    "📰 News & Sentiment",
    "💡 Analisi Personale",
    "📓 Journal",
    # ── Ultima: Watchlist (persistente) ────────────────────
    "📋 Watchlist",
])
(tab_home, tab_e, tab_p, tab_r, tab_conf,
 tab_ser, tab_fvpro, tab_crisis,
 tab_regime, tab_mtfmatrix, tab_mtf, tab_bcd, tab_of, tab_rm, tab_bt,
 tab_ai, tab_ai2, tab_opts, tab_mom, tab_news,
 tab_analisi, tab_journal,
 tab_w) = tabs

with tab_home:
    # ── v41e — ALERT BANNER: segnali STRONG visibili subito ──────────────
    try:
        _df_ep_banner = st.session_state.get("df_ep", pd.DataFrame())
        if not _df_ep_banner.empty and "Stato_Pro" in _df_ep_banner.columns:
            _strong_list = _df_ep_banner[_df_ep_banner["Stato_Pro"]=="STRONG"]["Ticker"].dropna().tolist()
            _pro_count   = int(_df_ep_banner["Stato_Pro"].isin(["PRO","STRONG"]).sum())
            _strong_count = len(_strong_list)
            if _strong_count > 0:
                _df_strong_map = _df_ep_banner.set_index("Ticker")["Nome"].to_dict() if "Nome" in _df_ep_banner.columns else {}
                _banner_tickers_html = "  ".join(
                    f"<a href='https://it.tradingview.com/chart/?symbol={t.replace('.MI','%3AMI')}' target='_blank' style='color:#ffd700;font-family:Courier New;font-weight:bold;text-decoration:none;font-size:0.88rem'>{t}</a>"
                    f"<span style='color:#9ca3af;font-size:0.70rem;font-style:italic'>{str(_df_strong_map.get(t,''))[:18]}</span>"
                    for t in _strong_list[:12]
                )
                _banner_extra = f" <span style='color:#9ca3af'>+{_strong_count-12} altri</span>" if _strong_count > 12 else ""
                st.markdown(
                    f"<div style='background:linear-gradient(90deg,#1a0f00,#1e1a00,#0d1a0d);"
                    f"border:1px solid #ffd70044;border-left:4px solid #ffd700;"
                    f"border-radius:0 8px 8px 0;padding:8px 16px;margin-bottom:10px;"
                    f"display:flex;align-items:center;gap:12px;flex-wrap:wrap'>"
                    f"<span style='color:#ffd700;font-weight:bold;font-size:0.82rem;"
                    f"white-space:nowrap'>★ STRONG ({_strong_count})</span>"
                    f"<span style='color:#6b7280;font-size:0.78rem'>{_banner_tickers_html}{_banner_extra}</span>"
                    f"<span style='color:#374151;font-size:0.72rem;margin-left:auto'>"
                    f"PRO+STRONG totali: <b style='color:#00ff88'>{_pro_count}</b></span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif _pro_count > 0:
                st.markdown(
                    f"<div style='background:#0d1a0d;border-left:4px solid #00ff88;"
                    f"border-radius:0 6px 6px 0;padding:6px 14px;margin-bottom:10px;"
                    f"font-size:0.80rem;color:#00ff88'>✦ {_pro_count} segnali PRO attivi — nessun STRONG</div>",
                    unsafe_allow_html=True
                )
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # v43 — BARRA CONTROLLO: Auto-refresh + Export CSV/Excel 1-click
    # ══════════════════════════════════════════════════════════════════════════
    try:
        _v43_c1,_v43_c2,_v43_c3,_v43_c4 = st.columns([2.5,0.8,0.9,0.8])
        with _v43_c1:
            _v43_ar_label = (f"ogni {st.session_state.get('v43_autorefresh_mins',0)} min"
                             if st.session_state.get("v43_autorefresh_mins",0)>0 else "disattivato")
            st.markdown(
                f"<div class='refresh-banner'>🔄 Auto-refresh: <strong>{_v43_ar_label}</strong>"
                f"&nbsp;|&nbsp; Aggiornato: <strong>{datetime.now().strftime('%H:%M:%S')}</strong></div>",
                unsafe_allow_html=True)
        with _v43_c2:
            st.session_state["v43_autorefresh_mins"] = st.selectbox(
                "⏱", [0,5,10,15,30], index=0,
                format_func=lambda x: "Off" if x==0 else f"{x}m",
                key="v43_ar_sel", label_visibility="collapsed")
        with _v43_c3:
            if not df_ep.empty:
                _v43_ec = [c for c in ["Ticker","Nome","Stato_Pro","CSS","RSI","Tipo"] if c in df_ep.columns]
                st.download_button("📥 TXT TradingView", make_tv_txt(df_ep[_v43_ec]),
                                   f"segnali_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                   "text/plain", key="v43d_dl_txt", use_container_width=True)
        with _v43_c4:
            if not df_ep.empty:
                try:
                    import io as _io43
                    _v43_xl = _io43.BytesIO()
                    _v43_xlc = [c for c in ["Ticker","Nome","Stato_Pro","CSS","RSI"] if c in df_ep.columns]
                    df_ep[_v43_xlc].to_excel(_v43_xl, index=False, engine="openpyxl")
                    st.download_button("📊 Excel", _v43_xl.getvalue(),
                                       f"segnali_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       key="v43_dl_xl", use_container_width=True)
                except Exception:
                    pass
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # v43 — TOP PRO/STRONG CON SPARKLINE SVG INLINE (ultimi 10 giorni)
    # ══════════════════════════════════════════════════════════════════════════
    try:
        if not df_ep.empty and "Stato_Pro" in df_ep.columns:
            _v43_top = df_ep[df_ep["Stato_Pro"].isin(["PRO","STRONG"])].copy()
            if not _v43_top.empty:
                _v43_srt = (_v43_top.sort_values("CSS", ascending=False)
                            if "CSS" in _v43_top.columns else _v43_top)
                with st.expander(f"⭐ Top PRO/STRONG ({len(_v43_top)}) — Sparkline 10d", expanded=True):
                    for _, _vtr in _v43_srt.head(15).iterrows():
                        _vtk = str(_vtr.get("Ticker",""))
                        _vnm = str(_vtr.get("Nome", _vtr.get("Company","")))[:28]
                        _vcss = _vtr.get("CSS","—")
                        _vrsi = _vtr.get("RSI","—")
                        _vst  = str(_vtr.get("Stato_Pro",""))
                        _vbc  = "#ffd700" if _vst=="STRONG" else "#00ff88"
                        try:
                            import yfinance as _yf43
                            _sp = _yf43.download(_vtk, period="12d", interval="1d",
                                                 auto_adjust=True, progress=False)
                            _sp.columns = [c[0] if isinstance(c,tuple) else c for c in _sp.columns]
                            _spc = _sp["Close"].dropna().tolist() if "Close" in _sp.columns else []
                        except Exception:
                            _spc = []
                        if len(_spc) >= 2:
                            _smn=min(_spc); _smx=max(_spc); _sr=max(_smx-_smn,0.01)
                            _W,_H = 80,22
                            _pts = " ".join(
                                f"{int(k*(_W/(len(_spc)-1)))},{int(_H-(_v-_smn)/_sr*(_H-4)+2)}"
                                for k,_v in enumerate(_spc))
                            _sc = "#00ff88" if _spc[-1]>=_spc[0] else "#ef4444"
                            _svg = (f"<svg class='spark-svg' width='{_W}' height='{_H}' "
                                    f"viewBox='0 0 {_W} {_H}'><polyline points='{_pts}' "
                                    f"fill='none' stroke='{_sc}' stroke-width='1.5' "
                                    f"stroke-linejoin='round'/></svg>")
                        else:
                            _svg = ("<svg class='spark-svg' width='80' height='22' "
                                    "viewBox='0 0 80 22'><line x1='0' y1='11' x2='80' y2='11' "
                                    "stroke='#2a2e39' stroke-width='1'/></svg>")
                        _tv = _vtk.replace(".MI","%3AMI").replace(".L","%3AL")
                        st.markdown(
                            f"<div class='spark-row'>"
                            f"<a href='https://it.tradingview.com/chart/?symbol={_tv}' "
                            f"target='_blank' style='text-decoration:none'>"
                            f"<span class='spark-ticker'>{_vtk}</span></a>"
                            f"<span class='spark-badge' style='background:{_vbc}22;"
                            f"color:{_vbc};border:1px solid {_vbc}44'>{_vst}</span>"
                            f"<span class='spark-css'>CSS:{_vcss}</span>"
                            f"<span class='spark-css'>RSI:{_vrsi}</span>"
                            f"{_svg}"
                            f"<span style='color:#6b7280;font-size:0.72rem;margin-left:auto'>"
                            f"{_vnm}</span></div>", unsafe_allow_html=True)
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # v43 — EARNINGS TRACKER COMPATTO con filtro giorni + export CSV
    # ══════════════════════════════════════════════════════════════════════════
    try:
        with st.expander("📅 Earnings Tracker — prossime uscite", expanded=False):
            _et_days = st.slider("Prossimi N giorni", 1, 30, 7, key="v43_et_days")
            import yfinance as _yf_et43
            from datetime import timedelta as _td43
            _et_tks = (df_ep["Ticker"].dropna().unique().tolist()[:40]
                       if not df_ep.empty and "Ticker" in df_ep.columns
                       else ["AAPL","NVDA","MSFT","AMD","META","GOOGL"])
            _et_res, _tod43 = [], datetime.now().date()
            _cut43 = _tod43 + _td43(days=_et_days)
            _et_prog = st.progress(0.0, text="Recupero date earnings…")
            for _ei,_etk in enumerate(_et_tks):
                _et_prog.progress((_ei+1)/max(len(_et_tks),1))
                try:
                    _cal = _yf_et43.Ticker(_etk).calendar
                    _ed  = None
                    if isinstance(_cal,dict):
                        _ed = _cal.get("Earnings Date")
                        if hasattr(_ed,"__iter__") and not isinstance(_ed,str):
                            _edl=list(_ed); _ed=_edl[0] if _edl else None
                    elif hasattr(_cal,"columns") and "Earnings Date" in _cal.columns:
                        _ed = _cal["Earnings Date"].iloc[0]
                    if _ed is None: continue
                    _edate = pd.Timestamp(_ed).date()
                    if _tod43 <= _edate <= _cut43:
                        _dn = (_edate-_tod43).days
                        _enm = (df_ep[df_ep["Ticker"]==_etk]["Nome"].iloc[0]
                                if not df_ep.empty and "Nome" in df_ep.columns
                                and not df_ep[df_ep["Ticker"]==_etk].empty else "")
                        _est = (df_ep[df_ep["Ticker"]==_etk]["Stato_Pro"].iloc[0]
                                if not df_ep.empty and "Stato_Pro" in df_ep.columns
                                and not df_ep[df_ep["Ticker"]==_etk].empty else "—")
                        _et_res.append({"Ticker":_etk,"Nome":str(_enm)[:24],
                                        "Data":_edate.strftime("%d/%m"),
                                        "Giorni":_dn,"Stato":str(_est)})
                except Exception:
                    continue
            _et_prog.empty()
            _et_res.sort(key=lambda x: x["Giorni"])
            if _et_res:
                st.markdown(
                    "<div class='earn-row' style='color:#50c4e0;font-weight:bold;"
                    "border-bottom:2px solid #2a2e39;font-size:0.83rem;"
                    "background:#0d1117;border-radius:6px 6px 0 0'>"
                    "<span>Ticker</span><span>Nome</span>"
                    "<span>Data</span><span>Gg</span><span>Stato</span></div>",
                    unsafe_allow_html=True)
                for _er in _et_res:
                    _ec=("#ffd700" if _er["Giorni"]<=2 else "#f59e0b" if _er["Giorni"]<=5 else "#6b7280")
                    _sc=("#ffd700" if _er["Stato"]=="STRONG" else "#00ff88" if _er["Stato"]=="PRO" else "#787b86")
                    _tv_s43c = _er["Ticker"].replace(".MI","%3AMI").replace(".L","%3AL").replace(".PA","%3APA")
                    _tv_u43c = f"https://it.tradingview.com/chart/?symbol={_tv_s43c}"
                    st.markdown(
                        f"<div class='earn-row'>"
                        f"<span class='earn-ticker'><a href='{_tv_u43c}' target='_blank'>{_er['Ticker']} ↗</a></span>"
                        f"<span style='color:#b2b5be;font-size:0.83rem'>{_er['Nome']}</span>"
                        f"<span class='earn-date'>{_er['Data']}</span>"
                        f"<span style='color:{_ec};font-family:Courier New;font-size:0.83rem'>{_er['Giorni']}g</span>"
                        f"<span style='color:{_sc};font-size:0.80rem;font-weight:bold'>{_er['Stato']}</span>"
                        f"</div>", unsafe_allow_html=True)
                st.download_button("📥 TXT TradingView",
                    make_tv_txt(pd.DataFrame(_et_res)),
                    f"earnings_{datetime.now().strftime('%Y%m%d')}.txt",
                    "text/plain", key="v43d_et_txt")
            else:
                st.info(f"Nessun earnings nei prossimi {_et_days} giorni.")
    except Exception as _et_ex:
        st.caption(f"⚠️ Earnings tracker: {_et_ex}")

        # ══════════════════════════════════════════════════════════════════════════
    # v43c — COT REPORT: fetch automatico CFTC + grafici posizione netta
    # ══════════════════════════════════════════════════════════════════════════
    try:
        with st.expander("📊 COT Report — Commitment of Traders (CFTC, aggiornato venerdì)", expanded=False):
            _cot_assets = {
                "S&P 500 (E-Mini)":   "13874+",
                "NASDAQ 100":         "20974+",
                "Gold":               "088691",
                "Crude Oil (WTI)":    "067651",
                "EUR/USD":            "099741",
                "USD/JPY (JPY)":      "097741",
                "Bitcoin":            "133741",
                "Argento":            "084691",
                "Rame":               "085692",
                "T-Bond 10Y":         "043602",
            }
            _cot_sel = st.multiselect(
                "Seleziona asset COT", list(_cot_assets.keys()),
                default=["S&P 500 (E-Mini)","Gold","EUR/USD","Bitcoin"],
                key="v43c_cot_sel")

            @st.cache_data(ttl=3600, show_spinner=False)
            def _fetch_cot_v43c(code, asset_name):
                import urllib.request, csv, io as _io
                url = (f"https://www.cftc.gov/dea/newcot/f_year.txt")
                # Usa Quandl/CFTC via pandas-datareader o fallback CSV diretto
                try:
                    import pandas_datareader.data as _pdr
                    from datetime import datetime as _dt
                    df_cot = _pdr.get_data_quandl(
                        f"CFTC/{code}_FO_L_ALL", start="2020-01-01")
                    return df_cot
                except Exception:
                    pass
                # Fallback: CFTC disaggregated CSV annuale
                try:
                    _yr = datetime.now().year
                    _url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{_yr}.zip"
                    import urllib.request, zipfile, io as _zio
                    with urllib.request.urlopen(_url, timeout=10) as r:
                        z = zipfile.ZipFile(_zio.BytesIO(r.read()))
                        fname = [f for f in z.namelist() if f.endswith('.txt')][0]
                        raw = z.read(fname).decode('utf-8', errors='ignore')
                    lines_cot = raw.splitlines()
                    rows = [l.split(',') for l in lines_cot if asset_name.split()[0].upper() in l.upper()]
                    if rows:
                        cols = lines_cot[0].split(',')
                        df_raw = pd.DataFrame(rows, columns=cols[:len(rows[0])])
                        return df_raw
                except Exception:
                    return None

            import plotly.graph_objects as go
            for _ca in _cot_sel:
                _ccode = _cot_assets[_ca]
                st.markdown(f"<div style='color:#50c4e0;font-weight:bold;font-size:0.85rem;"
                            f"margin:8px 0 4px'>📈 {_ca}</div>", unsafe_allow_html=True)
                _cdf = _fetch_cot_v43c(_ccode, _ca)
                if _cdf is not None and not _cdf.empty:
                    try:
                        # Cerca colonne Net position (Commercial, Non-Commercial)
                        _nc_cols = [c for c in _cdf.columns if 'NonComm' in c or 'Non_Comm' in c
                                    or 'NoncommercialLong' in c or 'Noncommercial_Long' in c]
                        _co_cols = [c for c in _cdf.columns if 'CommercialLong' in c
                                    or 'Commercial_Long' in c]
                        _nc_long = next((c for c in _cdf.columns if 'noncomm' in c.lower()
                                        and 'long' in c.lower()), None)
                        _nc_short= next((c for c in _cdf.columns if 'noncomm' in c.lower()
                                        and 'short' in c.lower()), None)
                        _co_long = next((c for c in _cdf.columns if 'commercial' in c.lower()
                                        and 'long' in c.lower()
                                        and 'noncomm' not in c.lower()), None)
                        _co_short= next((c for c in _cdf.columns if 'commercial' in c.lower()
                                        and 'short' in c.lower()
                                        and 'noncomm' not in c.lower()), None)
                        if _nc_long and _nc_short and _co_long and _co_short:
                            _cdf2 = _cdf.copy()
                            _cdf2["Net_NonComm"]    = pd.to_numeric(_cdf2[_nc_long],errors='coerce')                                                      - pd.to_numeric(_cdf2[_nc_short],errors='coerce')
                            _cdf2["Net_Commercial"] = pd.to_numeric(_cdf2[_co_long],errors='coerce')                                                      - pd.to_numeric(_cdf2[_co_short],errors='coerce')
                            _cdf2 = _cdf2.dropna(subset=["Net_NonComm","Net_Commercial"]).tail(52)
                            fig_cot = go.Figure()
                            fig_cot.add_trace(go.Scatter(
                                x=_cdf2.index, y=_cdf2["Net_NonComm"],
                                name="Non-Commercial (Speculatori)",
                                line=dict(color="#00ff88", width=1.5)))
                            fig_cot.add_trace(go.Scatter(
                                x=_cdf2.index, y=_cdf2["Net_Commercial"],
                                name="Commercial (Hedger)",
                                line=dict(color="#ef4444", width=1.5)))
                            fig_cot.add_hline(y=0, line_dash="dot",
                                              line_color="#363a45", line_width=1)
                            fig_cot.update_layout(
                                paper_bgcolor="#131722", plot_bgcolor="#1e222d",
                                font=dict(color="#b2b5be", family="Trebuchet MS", size=11),
                                height=200, margin=dict(l=0,r=0,t=24,b=0),
                                showlegend=True,
                                legend=dict(orientation="h", y=1.1, x=0,
                                            font=dict(size=10, color="#b2b5be")),
                                xaxis=dict(gridcolor="#2a2e39", tickfont=dict(size=9)),
                                yaxis=dict(gridcolor="#2a2e39", tickfont=dict(size=9)),
                                title=dict(text=f"Net Positions — {_ca}",
                                           font=dict(color="#50c4e0", size=11), x=0.01))
                            st.plotly_chart(fig_cot, use_container_width=True,
                                            key=f"v43c_cot_{_ccode}")
                        else:
                            st.caption(f"Colonne Net non trovate per {_ca} — "
                                       f"dati grezzi: {list(_cdf.columns)[:6]}")
                    except Exception as _ce:
                        st.caption(f"Errore grafico COT {_ca}: {_ce}")
                else:
                    # Mostra link diretto CFTC quando fetch fallisce
                    st.markdown(
                        f"<div style='background:#0d1117;border:1px solid #1f2937;border-radius:6px;"
                        f"padding:8px 14px;font-size:0.79rem;color:#6b7280'>"
                        f"⚠️ Fetch CFTC non disponibile per <strong style='color:#b2b5be'>{_ca}</strong>. "
                        f"<a href='https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm' "
                        f"target='_blank' style='color:#50c4e0'>Apri CFTC ↗</a> &nbsp;|&nbsp; "
                        f"<a href='https://www.barchart.com/futures/commitment-of-traders/chart/{_ccode}' "
                        f"target='_blank' style='color:#50c4e0'>Barchart COT ↗</a></div>",
                        unsafe_allow_html=True)

            # Tabella interpretazione COT
            st.markdown("""
<div style='margin-top:12px;font-size:0.78rem;color:#6b7280'>
<b style='color:#50c4e0'>Guida rapida COT:</b><br>
<span style='color:#00ff88'>▲ Non-Commercial NET long</span> = Speculatori rialzisti (trend-following) &nbsp;|&nbsp;
<span style='color:#ef4444'>▲ Commercial NET short</span> = Hedger coperti = segnale contrarian bearish<br>
Dati CFTC pubblicati ogni <b>venerdì 21:30 CET</b>, riferiti al martedì precedente.
</div>""", unsafe_allow_html=True)
    except Exception as _cot_ex:
        st.caption(f"⚠️ COT Report: {_cot_ex}")

        # ══════════════════════════════════════════════════════════════════════════
    # v43c — PATTERN RECOGNITION: candlestick + volume profile
    # ══════════════════════════════════════════════════════════════════════════
    try:
        with st.expander("📈 Pattern Recognition — Candlestick + Volume Profile", expanded=False):
            _pr_c1, _pr_c2 = st.columns([3,1])
            with _pr_c1:
                _pr_tk = st.text_input("Ticker", "AAPL", key="v43c_pr_tk",
                                        placeholder="es: UCG.MI, NVDA, RACE.MI")
            with _pr_c2:
                _pr_period = st.selectbox("Periodo", ["1mo","3mo","6mo","1y"],
                                          index=1, key="v43c_pr_period")
            _pr_go = st.button("🔍 Analizza Pattern", key="v43c_pr_go",
                               use_container_width=False)
            if _pr_go and _pr_tk:
                import yfinance as _ypr
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                with st.spinner(f"Scarico dati {_pr_tk}…"):
                    _pr_df = _ypr.download(_pr_tk.strip().upper(),
                                           period=_pr_period, interval="1d",
                                           auto_adjust=True, progress=False)
                    _pr_df.columns = [c[0] if isinstance(c,tuple) else c for c in _pr_df.columns]
                    _pr_df = _pr_df.dropna()

                if _pr_df.empty:
                    st.warning(f"Nessun dato per {_pr_tk}")
                else:
                    # ── Rilevamento pattern candlestick ──────────────────────
                    O = _pr_df["Open"]; H = _pr_df["High"]
                    L = _pr_df["Low"];  C = _pr_df["Close"]; V = _pr_df["Volume"]
                    body   = (C - O).abs()
                    rng    = H - L
                    upper_wick = H - C.combine(O, max)
                    lower_wick = C.combine(O, min) - L

                    def _is_doji(i):
                        return body.iloc[i] < rng.iloc[i]*0.1
                    def _is_hammer(i):
                        return (lower_wick.iloc[i] > body.iloc[i]*2 and
                                upper_wick.iloc[i] < body.iloc[i]*0.5)
                    def _is_shooting_star(i):
                        return (upper_wick.iloc[i] > body.iloc[i]*2 and
                                lower_wick.iloc[i] < body.iloc[i]*0.5)
                    def _is_engulfing_bull(i):
                        return (i>0 and C.iloc[i-1]<O.iloc[i-1] and
                                C.iloc[i]>O.iloc[i] and
                                C.iloc[i]>O.iloc[i-1] and O.iloc[i]<C.iloc[i-1])
                    def _is_engulfing_bear(i):
                        return (i>0 and C.iloc[i-1]>O.iloc[i-1] and
                                C.iloc[i]<O.iloc[i] and
                                O.iloc[i]>C.iloc[i-1] and C.iloc[i]<O.iloc[i-1])
                    def _is_morning_star(i):
                        return (i>=2 and
                                C.iloc[i-2]<O.iloc[i-2] and
                                body.iloc[i-1]<rng.iloc[i-1]*0.3 and
                                C.iloc[i]>O.iloc[i] and
                                C.iloc[i]>((O.iloc[i-2]+C.iloc[i-2])/2))
                    def _is_evening_star(i):
                        return (i>=2 and
                                C.iloc[i-2]>O.iloc[i-2] and
                                body.iloc[i-1]<rng.iloc[i-1]*0.3 and
                                C.iloc[i]<O.iloc[i] and
                                C.iloc[i]<((O.iloc[i-2]+C.iloc[i-2])/2))

                    n = len(_pr_df)
                    patterns_found = []
                    annot_texts, annot_x, annot_y, annot_colors = [],[],[],[]
                    for idx in range(2, n):
                        pname, pcolor, psymbol = None, None, None
                        if _is_doji(idx):
                            pname,pcolor,psymbol = "Doji","#f0b90b","●"
                        elif _is_hammer(idx):
                            pname,pcolor,psymbol = "Hammer","#00ff88","▲"
                        elif _is_shooting_star(idx):
                            pname,pcolor,psymbol = "Shooting Star","#ef4444","▼"
                        elif _is_engulfing_bull(idx):
                            pname,pcolor,psymbol = "Engulfing Bull","#00c087","▲"
                        elif _is_engulfing_bear(idx):
                            pname,pcolor,psymbol = "Engulfing Bear","#f23645","▼"
                        elif _is_morning_star(idx):
                            pname,pcolor,psymbol = "Morning Star","#26a69a","✦"
                        elif _is_evening_star(idx):
                            pname,pcolor,psymbol = "Evening Star","#ef5350","✦"
                        if pname:
                            patterns_found.append({"Data":str(_pr_df.index[idx])[:10],
                                                   "Pattern":pname,"Prezzo":f"{C.iloc[idx]:.2f}"})
                            annot_texts.append(psymbol)
                            annot_x.append(_pr_df.index[idx])
                            annot_y.append(H.iloc[idx]*1.005)
                            annot_colors.append(pcolor)

                    # ── Volume Profile (POC) ─────────────────────────────────
                    _vp_bins = 20
                    _pr_min, _pr_max = float(L.min()), float(H.max())
                    _vp_edges = [_pr_min + i*(_pr_max-_pr_min)/_vp_bins
                                 for i in range(_vp_bins+1)]
                    _vp_vol   = [0.0]*_vp_bins
                    for j in range(len(_pr_df)):
                        for b in range(_vp_bins):
                            if _vp_edges[b] <= float(C.iloc[j]) < _vp_edges[b+1]:
                                _vp_vol[b] += float(V.iloc[j])
                                break
                    _poc_idx = _vp_vol.index(max(_vp_vol))
                    _poc_price = (_vp_edges[_poc_idx]+_vp_edges[_poc_idx+1])/2

                    # ── Chart: Candlestick + Volume + VP ────────────────────
                    fig_pr = make_subplots(
                        rows=3, cols=2,
                        shared_xaxes=False,
                        column_widths=[0.82,0.18],
                        row_heights=[0.58,0.22,0.20],
                        specs=[[{"rowspan":2},{"rowspan":2,"type":"bar"}],
                               [None,None],
                               [{"colspan":2},None]],
                        vertical_spacing=0.04, horizontal_spacing=0.015)

                    fig_pr.add_trace(go.Candlestick(
                        x=_pr_df.index, open=O, high=H, low=L, close=C,
                        name=_pr_tk.upper(),
                        increasing_line_color="#26a69a",
                        decreasing_line_color="#ef5350"), row=1,col=1)

                    # POC line
                    fig_pr.add_hline(y=_poc_price, line_dash="dash",
                                     line_color="#f0b90b", line_width=1,
                                     annotation_text=f"POC {_poc_price:.2f}",
                                     annotation_font_color="#f0b90b",
                                     annotation_font_size=9, row=1, col=1)

                    # Annotazioni pattern
                    for ax,ay,at,ac in zip(annot_x,annot_y,annot_texts,annot_colors):
                        fig_pr.add_annotation(x=ax,y=ay,text=at,showarrow=False,
                                              font=dict(size=11,color=ac),row=1,col=1)

                    # Volume Profile bar (orizzontale) → verticale sul pannello dx
                    _vp_mids = [(_vp_edges[b]+_vp_edges[b+1])/2 for b in range(_vp_bins)]
                    fig_pr.add_trace(go.Bar(
                        x=_vp_vol, y=_vp_mids, orientation='h',
                        marker_color=["#f0b90b" if b==_poc_idx else "#363a45"
                                      for b in range(_vp_bins)],
                        name="Volume Profile", showlegend=False), row=1,col=2)

                    # Volume bars sotto
                    _vcol = ["#26a69a" if C.iloc[j]>=O.iloc[j] else "#ef5350"
                             for j in range(len(_pr_df))]
                    fig_pr.add_trace(go.Bar(
                        x=_pr_df.index, y=V,
                        marker_color=_vcol, name="Volume",
                        showlegend=False), row=3,col=1)

                    fig_pr.update_layout(
                        paper_bgcolor="#131722", plot_bgcolor="#1e222d",
                        font=dict(color="#b2b5be",family="Trebuchet MS",size=10),
                        height=440, margin=dict(l=0,r=0,t=28,b=0),
                        xaxis_rangeslider_visible=False,
                        showlegend=False,
                        title=dict(text=f"{_pr_tk.upper()} — Candlestick + Volume Profile | POC: {_poc_price:.2f}",
                                   font=dict(color="#50c4e0",size=12),x=0.01))
                    for ax_name in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
                        fig_pr.update_layout(**{ax_name:dict(gridcolor="#2a2e39",
                                                             tickfont=dict(size=9))})
                    st.plotly_chart(fig_pr, use_container_width=True, key="v43c_pr_chart")

                    # Tabella pattern
                    if patterns_found:
                        st.markdown(f"**Pattern rilevati ({len(patterns_found)}):**")
                        _pdf_patt = pd.DataFrame(patterns_found)
                        st.dataframe(_pdf_patt, use_container_width=True, hide_index=True,
                                     column_config={
                                         "Data":st.column_config.TextColumn("Data",width=90),
                                         "Pattern":st.column_config.TextColumn("Pattern",width=130),
                                         "Prezzo":st.column_config.TextColumn("Prezzo",width=80)})
                        st.download_button("📥 TXT TradingView",
                            make_tv_txt(_pdf_patt),
                            f"pattern_{_pr_tk}_{datetime.now().strftime('%Y%m%d')}.txt",
                            "text/plain", key="v43c_pr_txt")
                    else:
                        st.info("Nessun pattern significativo nel periodo selezionato.")
    except Exception as _pr_ex:
        st.caption(f"⚠️ Pattern Recognition: {_pr_ex}")

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

    # ── v41e — MERCATI LIVE GLOBALE con mini sparkline 5gg ─────────────────

    def _mini_sparkline_svg(prices: list, color: str, w: int = 80, h: int = 28) -> str:
        """Genera mini sparkline SVG inline per la card Mercati Live."""
        if not prices or len(prices) < 2:
            return ""
        try:
            import math as _mth_sp
            _vals = [float(v) for v in prices if v is not None and not _mth_sp.isnan(v)]
            if len(_vals) < 2:
                return ""
            _mn, _mx = min(_vals), max(_vals)
            _rng = _mx - _mn if _mx != _mn else 1.0
            _pts = []
            for _i, _v in enumerate(_vals):
                _x = _i / (len(_vals)-1) * (w-4) + 2
                _y = h - 4 - (_v - _mn) / _rng * (h-8)
                _pts.append(f"{_x:.1f},{_y:.1f}")
            _polyline = " ".join(_pts)
            # Area fill sotto la linea
            _area_pts = f"2,{h-2} " + _polyline + f" {w-2},{h-2}"
            _up = _vals[-1] >= _vals[0]
            _fill_color = f"{color}22"
            return (
                f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
                f"style='display:block;margin:3px auto 0'>"
                f"<polygon points='{_area_pts}' fill='{_fill_color}'/>"
                f"<polyline points='{_polyline}' fill='none' "
                f"stroke='{color}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/>"
                f"<circle cx='{_pts[-1].split(chr(44))[0]}' cy='{_pts[-1].split(chr(44))[1]}' "
                f"r='2' fill='{color}'/>"
                f"</svg>"
            )
        except Exception:
            return ""

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
                # Sparkline: ultimi 5 close normalizzati
                _spark = []
                try:
                    _spark = _cl5.tolist()[-5:]
                except Exception:
                    pass
                _results.append({"sym":_sym,"name":_name,"icon":_ico,
                                 "price":_cur,"chg":_chg,"ytd":_ytd,"spark":_spark})
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

                    _spark_svg = _mini_sparkline_svg(_m.get("spark",[]), _c, 76, 26)
                    _live_html += (
                        f"<div style='background:#131722;border:1px solid #2a2e39;"
                        f"border-top:3px solid {_c}88;"
                        f"border-radius:6px;padding:8px 12px 6px;"
                        f"min-width:110px;max-width:165px;flex:1;text-align:center;"
                        f"transition:border-color .2s'>"
                        f"<div style='color:#9ca3af;font-size:0.66rem;white-space:nowrap;"
                        f"margin-bottom:3px;letter-spacing:0.5px'>{_m['icon']} {_m['name']}</div>"
                        f"<div style='color:#e2e8f0;font-family:Courier New;font-size:0.94rem;"
                        f"font-weight:bold;letter-spacing:0.5px'>{_pr}</div>"
                        f"<div style='color:{_c};font-size:0.73rem;font-weight:bold;margin-top:2px'>"
                        f"{_ar} {abs(_m['chg']):.2f}%</div>"
                        f"{_ytd_html}"
                        f"{_spark_svg}"
                        f"</div>"
                    )
                _live_html += "</div></div>"

            _live_html += "</div>"
            st.markdown(_live_html, unsafe_allow_html=True)
    except Exception:
        pass

        # ── v41 — CORRELAZIONI ASSET 30 giorni (inline, funzionante) ─────────
    # ── v41e FEATURE 5 — MAPPA CALORE GLOBALE SVG ─────────────────────────
    with st.expander("🌍 Mappa Calore Globale — Performance indici mondiali v42c", expanded=True):
        try:
            _live_for_map = _fetch_live_markets_v41()
            if _live_for_map:
                _map_data = {m["sym"]: m for m in _live_for_map}
                _map_regions = {
                    "🌎 AMERICAS": [("S&P500","^GSPC"),("NASDAQ","^IXIC"),("DowJones","^DJI"),("Russ2K","^RUT")],
                    "🌍 EUROPA":   [("IT MIB","FTSEMIB.MI"),("FTSE","^FTSE"),("DAX","^GDAXI"),("CAC","^FCHI"),
                                           ("IBEX","^IBEX"),("AEX","^AEX"),("VIX","^VIX"),("BTC","BTC-USD")],
                    "🌏 ASIA/EM":  [("Nikkei","^N225"),("HSeng","^HSI"),("Shanghai","000001.SS"),
                                           ("KOSPI","^KS11"),("Nifty","^NSEI"),("BVSP","^BVSP")],
                }
                _map_macro = [("🥇 Gold","GC=F"),("🛢 Oil","CL=F"),("💵 DXY","DX-Y.NYB"),("📉 TLT","TLT")]
                def _map_card_v41e(lbl, sym):
                    _d=_map_data.get(sym,{}); _c=float(_d.get("chg",0) or 0); _p=float(_d.get("price",0) or 0)
                    _i=min(1.0,abs(_c)/3.0)
                    _bg=f"rgba(0,{int(80+120*_i)},70,.88)" if _c>=0 else f"rgba({int(120+110*_i)},35,35,.88)"
                    _ar,_cl=("▲","#00ff88") if _c>=0 else ("▼","#ef4444")
                    _ps=f"{_p:,.0f}" if _p>10000 else (f"{_p:.1f}" if _p>100 else (f"{_p:.2f}" if _p>0 else "—"))
                    return (f"<div style='background:{_bg};border:1px solid #2a2e39;border-radius:6px;"
                            f"padding:6px 4px;text-align:center;min-width:72px;flex:1;max-width:110px'>"
                            f"<div style='color:#e2e8f0;font-size:.74rem;font-weight:bold'>{lbl}</div>"
                            f"<div style='color:{_cl};font-family:Courier New;font-weight:bold;font-size:.92rem'>{_ar}{abs(_c):.1f}%</div>"
                            f"<div style='color:#9ca3af;font-size:.74rem'>{_ps}</div></div>")
                _rc=st.columns([1,1.5,1])
                for _ci,(_rn,_ra) in enumerate(_map_regions.items()):
                    with _rc[_ci]:
                        st.markdown(f"<div style='color:#50c4e0;font-size:.70rem;font-weight:bold;text-align:center;"
                                    f"letter-spacing:2px;border-bottom:1px solid #2a2e39;padding-bottom:4px;"
                                    f"margin-bottom:6px'>{_rn}</div>",unsafe_allow_html=True)
                        st.markdown("<div style='display:flex;flex-wrap:wrap;gap:4px;justify-content:center'>"
                                    + "".join(_map_card_v41e(l,s) for l,s in _ra) + "</div>",unsafe_allow_html=True)
                st.markdown("<div style='margin-top:8px;display:flex;gap:5px;justify-content:center;flex-wrap:wrap'>"
                            + "".join(_map_card_v41e(l,s) for l,s in _map_macro) + "</div>",unsafe_allow_html=True)
                st.caption("🟢 rialzo · 🔴 ribasso · intensità = % variazione")
            else:
                st.info("Dati mappa non disponibili.")
        except Exception as _map_e:
            st.warning(f"Mappa non disponibile: {str(_map_e)[:80]}")


    # v41e: correlazioni -> tab Settori

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

    # v41: render_home per sparklines/breadth (senza Mercati Live duplicato)
    try:
        from utils.home_tab import render_home
        render_home(df_ep, df_rea)
    except Exception:
        pass

    # ── v43c #4 — EARNINGS CALENDAR (Home) ──────────────────────────────────
    st.markdown("---")
    with st.expander("📅 EARNINGS CALENDAR v43c — Prossimi earnings da Watchlist + Scanner", expanded=False):
        st.markdown("<h4 style='color:#50c4e0;font-size:1.08rem;font-weight:bold;margin-bottom:10px'>📅 Prossimi Earnings — Watchlist &amp; Scanner</h4>", unsafe_allow_html=True)
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
                        f"<b style='font-family:Courier New;color:#00ff88;font-size:1.05rem'>{_tkr_ed}</b>"
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
    with st.expander('📰 NEWS & SENTIMENT v42h — Ultime news con score sentiment · TradingView Italia', expanded=False):
        _render_news_v41(df_ep)
    st.markdown('---')
    # ── v43c: COT REPORT ─────────────────────────────────────────────────
    with st.expander('📊 COT REPORT v42h — Commitment of Traders (CFTC) · Posizionamento istituzionale', expanded=False):
        st.markdown(
            '<div style=\'background:#1e222d;border-left:3px solid #2962ff;border-radius:0 6px 6px 0;padding:10px 16px;margin-bottom:12px\'><span style=\'color:#50c4e0;font-size:0.78rem;font-weight:bold;letter-spacing:1px\'>COS\'\\u00c8 IL COT REPORT</span><br><span style=\'color:#b2b5be;font-size:0.82rem\'>Il <b style="color:#d1d4dc">COT (Commitment of Traders)</b> \\u00e8 un report settimanale pubblicato ogni <b style="color:#f59e0b">venerd\\u00ec alle 21:30 CET</b> dalla CFTC. Fotografa le posizioni <b style="color:#00ff88">LONG</b> e <b style="color:#ef4444">SHORT</b> dei 3 gruppi: <b style="color:#00ff88">Commercial</b> (hedger), <b style="color:#60a5fa">Non-Commercial</b> (hedge fund / speculatori), <b style="color:#6b7280">Non-Reportable</b> (retail). Dati riferiti al <b style="color:#f59e0b">martedi precedente</b> — 3 giorni di ritardo.</span></div>',
            unsafe_allow_html=True)

        st.markdown(
            '<div style=\'background:#131722;border:1px solid #2a2e39;border-radius:8px;padding:12px 16px;margin-bottom:10px\'><div style=\'color:#50c4e0;font-size:0.78rem;font-weight:bold;letter-spacing:1px;margin-bottom:10px\'>\\U0001f4d6 COME LEGGERE IL COT</div><table style=\'width:100%;border-collapse:collapse;font-size:0.81rem;font-family:Trebuchet MS,sans-serif\'><tr style=\'border-bottom:1px solid #2a2e39\'><th style=\'color:#50c4e0;padding:5px 10px;text-align:left\'>Categoria</th><th style=\'color:#50c4e0;padding:5px 10px;text-align:left\'>Chi sono</th><th style=\'color:#50c4e0;padding:5px 10px;text-align:left\'>Interpretazione</th></tr><tr style=\'border-bottom:1px solid #1a2233\'><td style=\'padding:6px 10px;color:#00ff88;font-weight:bold\'>Commercial</td><td style=\'padding:6px 10px;color:#b2b5be\'>Produttori, banche, hedger reali</td><td style=\'padding:6px 10px;color:#b2b5be\'><b style="color:#00ff88">Contrarian</b>: molti short = prezzo vicino ai minimi</td></tr><tr style=\'border-bottom:1px solid #1a2233\'><td style=\'padding:6px 10px;color:#60a5fa;font-weight:bold\'>Non-Commercial</td><td style=\'padding:6px 10px;color:#b2b5be\'>Hedge fund, grandi speculatori</td><td style=\'padding:6px 10px;color:#b2b5be\'><b style="color:#60a5fa">Trend-following</b>: estremi storici = inversione imminente</td></tr><tr><td style=\'padding:6px 10px;color:#6b7280;font-weight:bold\'>Non-Reportable</td><td style=\'padding:6px 10px;color:#b2b5be\'>Piccoli trader retail</td><td style=\'padding:6px 10px;color:#b2b5be\'><b style="color:#ef4444">Contrarian forte</b>: spesso sbagliano top/bottom</td></tr></table></div>',
            unsafe_allow_html=True)

        _cot_links = [
            ('🇺🇸 S&P 500 (ES)', 'https://www.barchart.com/futures/commitment-of-traders/ES*0'),
            ('💻 NASDAQ (NQ)',             'https://www.barchart.com/futures/commitment-of-traders/NQ*0'),
            ('🏆 Dow Jones (YM)',          'https://www.barchart.com/futures/commitment-of-traders/YM*0'),
            ('🥇 Gold (GC)',               'https://www.barchart.com/futures/commitment-of-traders/GC*0'),
            ('🛢 Oil WTI (CL)',            'https://www.barchart.com/futures/commitment-of-traders/CL*0'),
            ('💵 EUR/USD',                 'https://www.barchart.com/futures/commitment-of-traders/EU*0'),
            ('¥ USD/JPY',                       'https://www.barchart.com/futures/commitment-of-traders/JY*0'),
            ('🏦 T-Bond 10Y (TY)',         'https://www.barchart.com/futures/commitment-of-traders/TY*0'),
            ('₿ Bitcoin (BTC)',               'https://www.barchart.com/futures/commitment-of-traders/BTCM*0'),
            ('⚪ Argento (SI)',                'https://www.barchart.com/futures/commitment-of-traders/SI*0'),
            ('🟤 Rame (HG)',               'https://www.barchart.com/futures/commitment-of-traders/HG*0'),
            ('🌾 Grano (ZW)',              'https://www.barchart.com/futures/commitment-of-traders/ZW*0'),
            ('📊 CFTC ufficiale',          'https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm'),
            ('📊 Tradingster charts',      'https://www.tradingster.com/cot'),
        ]
        _cot_style = (
            'display:inline-block;background:#1e222d;border:1px solid #2a2e39;'
            'border-radius:6px;padding:6px 12px;color:#60a5fa;text-decoration:none;'
            'font-size:0.79rem;font-family:Trebuchet MS,sans-serif;white-space:nowrap'
        )
        _cot_html = "<div style='display:flex;flex-wrap:wrap;gap:7px;margin-bottom:10px'>"
        for _lbl, _url in _cot_links:
            _cot_html += f"<a href='{_url}' target='_blank' style='{_cot_style}'>{_lbl}</a>"
        _cot_html += '</div>'
        st.markdown(_cot_html, unsafe_allow_html=True)

        st.markdown(
            "<div style='background:#131722;border:1px solid #2a2e39;border-radius:8px;padding:12px 16px'><div style='color:#50c4e0;font-size:0.78rem;font-weight:bold;letter-spacing:1px;margin-bottom:8px'>\\u26a1 SEGNALI COT DA MONITORARE</div><div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:8px'><div style='background:#1e222d;border-radius:6px;padding:9px 12px'><div style='color:#00ff88;font-size:0.75rem;font-weight:bold;margin-bottom:4px'>\\u2705 SEGNALE BULLISH</div><div style='color:#b2b5be;font-size:0.79rem'>Commercial molto long + Non-Commercial estremo short \\u2192 inversione rialzo imminente</div></div><div style='background:#1e222d;border-radius:6px;padding:9px 12px'><div style='color:#ef4444;font-size:0.75rem;font-weight:bold;margin-bottom:4px'>\\u274c SEGNALE BEARISH</div><div style='color:#b2b5be;font-size:0.79rem'>Non-Commercial molto long (euforia speculativa) \\u2192 spesso anticipa correzioni significative</div></div><div style='background:#1e222d;border-radius:6px;padding:9px 12px'><div style='color:#f59e0b;font-size:0.75rem;font-weight:bold;margin-bottom:4px'>\\u26a0\\ufe0f LIMITE DEL COT</div><div style='color:#b2b5be;font-size:0.79rem'>Il COT \\u00e8 indicatore di <b>timing</b>, non di entrata precisa. I mercati possono restare in estremo per settimane. Usarlo con price action.</div></div><div style='background:#1e222d;border-radius:6px;padding:9px 12px'><div style='color:#60a5fa;font-size:0.75rem;font-weight:bold;margin-bottom:4px'>\\U0001f4c5 QUANDO LEGGERLO</div><div style='color:#b2b5be;font-size:0.79rem'>Ogni <b>venerd\\u00ec dopo le 21:30 CET</b>. Confronta con settimana precedente per il delta posizione netta. Dati riferiti al marted\\u00ec precedente.</div></div></div></div>",
            unsafe_allow_html=True)

    st.markdown('---')

    # ── v43c: COT REPORT — Commitment of Traders ─────────────────────────
    with st.expander('📊 COT REPORT — Commitment of Traders (CFTC) · Posizionamento istituzionale', expanded=False):
        st.markdown("""
<div style='background:#1e222d;border-left:3px solid #2962ff;border-radius:0 6px 6px 0;padding:10px 16px;margin-bottom:12px'>
<span style='color:#50c4e0;font-size:0.78rem;font-weight:bold;letter-spacing:1px'>COS'È IL COT REPORT</span><br>
<span style='color:#b2b5be;font-size:0.82rem'>Il <b style="color:#d1d4dc">COT (Commitment of Traders)</b> è un report settimanale pubblicato ogni <b style="color:#f59e0b">venerdì alle 21:30 CET</b> dalla CFTC (Commodity Futures Trading Commission).
Fotografa le posizioni <b style="color:#00ff88">LONG</b> e <b style="color:#ef4444">SHORT</b> dei 3 grandi gruppi di operatori su futures: <b style="color:#d1d4dc">Commercial</b> (hedger istituzionali), <b style="color:#60a5fa">Non-Commercial</b> (grandi speculatori/fondi) e <b style="color:#6b7280">Non-Reportable</b> (piccoli trader).
I dati si riferiscono al <b style="color:#f59e0b">martedì precedente</b> — 3 giorni di ritardo.</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style='background:#131722;border:1px solid #2a2e39;border-radius:8px;padding:12px 16px;margin-bottom:10px'>
<div style='color:#50c4e0;font-size:0.78rem;font-weight:bold;letter-spacing:1px;margin-bottom:10px'>📖 COME LEGGERE IL COT</div>
<table style='width:100%;border-collapse:collapse;font-size:0.81rem;font-family:Trebuchet MS'>
<tr style='border-bottom:1px solid #2a2e39'>
  <th style='color:#50c4e0;padding:5px 10px;text-align:left'>Categoria</th>
  <th style='color:#50c4e0;padding:5px 10px;text-align:left'>Chi sono</th>
  <th style='color:#50c4e0;padding:5px 10px;text-align:left'>Come interpretarli</th>
</tr>
<tr style='border-bottom:1px solid #1a2233'>
  <td style='padding:6px 10px;color:#00ff88;font-weight:bold'>Commercial</td>
  <td style='padding:6px 10px;color:#b2b5be'>Produttori, banche, hedger</td>
  <td style='padding:6px 10px;color:#b2b5be'>Segnale <b style="color:#00ff88">contrarian</b>: molti short = prezzo vicino ai minimi</td>
</tr>
<tr style='border-bottom:1px solid #1a2233'>
  <td style='padding:6px 10px;color:#60a5fa;font-weight:bold'>Non-Commercial</td>
  <td style='padding:6px 10px;color:#b2b5be'>Hedge fund, grandi speculatori</td>
  <td style='padding:6px 10px;color:#b2b5be'>Segnale <b style="color:#60a5fa">trend-following</b>: seguire quando estremi storici</td>
</tr>
<tr>
  <td style='padding:6px 10px;color:#6b7280;font-weight:bold'>Non-Reportable</td>
  <td style='padding:6px 10px;color:#b2b5be'>Piccoli trader retail</td>
  <td style='padding:6px 10px;color:#b2b5be'>Segnale <b style="color:#ef4444">contrarian forte</b>: spesso sbagliano nei top/bottom</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)

        # Link rapidi COT per i principali asset del tuo scanner
        st.markdown("""
<div style='background:#131722;border:1px solid #2a2e39;border-radius:8px;padding:12px 16px;margin-bottom:10px'>
<div style='color:#50c4e0;font-size:0.78rem;font-weight:bold;letter-spacing:1px;margin-bottom:8px'>🔗 LINK RAPIDI — COT per asset chiave</div>
<div style='display:flex;flex-wrap:wrap;gap:8px'>
""", unsafe_allow_html=True)

        _cot_links = [
            ("🇺🇸 S&P 500 (ES)", "https://www.barchart.com/futures/commitment-of-traders/ES*0"),
            ("💻 NASDAQ (NQ)", "https://www.barchart.com/futures/commitment-of-traders/NQ*0"),
            ("🥇 Gold (GC)", "https://www.barchart.com/futures/commitment-of-traders/GC*0"),
            ("🛢️ Oil WTI (CL)", "https://www.barchart.com/futures/commitment-of-traders/CL*0"),
            ("💵 EUR/USD", "https://www.barchart.com/futures/commitment-of-traders/EU*0"),
            ("¥ USD/JPY", "https://www.barchart.com/futures/commitment-of-traders/JY*0"),
            ("🏦 T-Bond 10Y", "https://www.barchart.com/futures/commitment-of-traders/TY*0"),
            ("₿ Bitcoin (BTC)", "https://www.barchart.com/futures/commitment-of-traders/BTCM*0"),
            ("⚪ Argento (SI)", "https://www.barchart.com/futures/commitment-of-traders/SI*0"),
            ("🟤 Rame (HG)", "https://www.barchart.com/futures/commitment-of-traders/HG*0"),
            ("🌾 Grano (ZW)", "https://www.barchart.com/futures/commitment-of-traders/ZW*0"),
            ("📊 CFTC ufficiale", "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"),
        ]
        _cot_html = ""
        for _lbl, _url in _cot_links:
            _style = "display:inline-block;background:#1e222d;border:1px solid #2a2e39;"
            _style += "border-radius:6px;padding:6px 12px;color:#60a5fa;text-decoration:none;"
            _style += "font-size:0.79rem;font-family:Trebuchet MS;white-space:nowrap"
            _cot_html += (
                f"<a href='{_url}' target='_blank' style='{_style}'>"
                f"{_lbl}</a>"
            )
        st.markdown(_cot_html + "</div></div>", unsafe_allow_html=True)

        st.markdown("""
<div style='background:#131722;border:1px solid #2a2e39;border-radius:8px;padding:12px 16px'>
<div style='color:#50c4e0;font-size:0.78rem;font-weight:bold;letter-spacing:1px;margin-bottom:8px'>⚡ SEGNALI COT DA MONITORARE</div>
<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px'>
  <div style='background:#1e222d;border-radius:6px;padding:8px 12px'>
    <div style='color:#00ff88;font-size:0.75rem;font-weight:bold'>✅ SEGNALE BULLISH</div>
    <div style='color:#b2b5be;font-size:0.79rem;margin-top:4px'>Commercial molto long + Non-Commercial estremo short → inversione al rialzo imminente</div>
  </div>
  <div style='background:#1e222d;border-radius:6px;padding:8px 12px'>
    <div style='color:#ef4444;font-size:0.75rem;font-weight:bold'>❌ SEGNALE BEARISH</div>
    <div style='color:#b2b5be;font-size:0.79rem;margin-top:4px'>Non-Commercial molto long (euforia speculativa) → spesso anticipa correzioni</div>
  </div>
  <div style='background:#1e222d;border-radius:6px;padding:8px 12px'>
    <div style='color:#f59e0b;font-size:0.75rem;font-weight:bold'>⚠️ ATTENZIONE</div>
    <div style='color:#b2b5be;font-size:0.79rem;margin-top:4px'>Il COT è un indicatore di <b>timing</b>, non di entrata precisa. I mercati possono rimanere in estremo per settimane. Usarlo con price action.</div>
  </div>
  <div style='background:#1e222d;border-radius:6px;padding:8px 12px'>
    <div style='color:#60a5fa;font-size:0.75rem;font-weight:bold'>📅 QUANDO USARLO</div>
    <div style='color:#b2b5be;font-size:0.79rem;margin-top:4px'>Leggi ogni venerdì dopo le 21:30 CET. Confronta con settimana precedente per vedere il cambio di posizione netta.</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('---')
    # ── v41e FEATURE 3 — HEATMAP SETTORIALE LIVE ──────────────────────────
    with st.expander("🌡️ Heatmap Settoriale Live — 11 settori × 3 periodi", expanded=True):
        @st.cache_data(ttl=300, show_spinner=False)
        def _sector_heatmap_home_v41e():
            import yfinance as _yh
            _se = {
                "Tech":    "XLK", "Health":  "XLV", "Fin":     "XLF",
                "Energy":  "XLE", "Cons.D":  "XLY", "Cons.S":  "XLP",
                "Ind":     "XLI", "Utils":   "XLU", "Mater":   "XLB",
                "R.Est":   "XLRE","Comm":    "XLC",
            }
            _res = {}
            _tickers_str = " ".join(_se.values())
            try:
                _raw_h = _yh.download(_tickers_str, period="95d", interval="1d",
                                       auto_adjust=True, progress=False, group_by="ticker")
                for _lbl, _etf in _se.items():
                    try:
                        if isinstance(_raw_h.columns, __import__("pandas").MultiIndex):
                            _cl_h = _raw_h[_etf]["Close"].dropna() if _etf in _raw_h.columns.get_level_values(0) else __import__("pandas").Series(dtype=float)
                        else:
                            _cl_h = _raw_h["Close"].dropna()
                        if len(_cl_h) < 5:
                            _res[_lbl] = {"1d":0,"1m":0,"3m":0,"spark":[]}
                            continue
                        _r1d = round((_cl_h.iloc[-1]/_cl_h.iloc[-2]-1)*100,2)  if len(_cl_h)>=2  else 0
                        _r1m = round((_cl_h.iloc[-1]/_cl_h.iloc[-22]-1)*100,2) if len(_cl_h)>=22 else 0
                        _idx3m=min(63,len(_cl_h)-1)
                        _r3m=round((_cl_h.iloc[-1]/_cl_h.iloc[-_idx3m]-1)*100,2) if _idx3m>=2 else 0
                        _spark_h = _cl_h.tolist()[-10:]
                        _res[_lbl] = {"1d":_r1d,"1m":_r1m,"3m":_r3m,"spark":_spark_h}
                    except Exception:
                        _res[_lbl] = {"1d":0,"1m":0,"3m":0,"spark":[]}
            except Exception:
                for _lbl in _se:
                    _res[_lbl] = {"1d":0,"1m":0,"3m":0,"spark":[]}
            return _res

        _ht_period = st.radio("Periodo heatmap", ["1d","1m","3m"], index=0,
                               horizontal=True, key="ht_period_home",
                               label_visibility="collapsed")
        try:
            _ht_data = _sector_heatmap_home_v41e()
            if _ht_data:
                # Calcola scala colori
                _ht_vals = [v[_ht_period] for v in _ht_data.values()]
                _ht_max = max(abs(v) for v in _ht_vals) if _ht_vals else 5
                _ht_max = max(_ht_max, 0.5)

                def _ht_color(val, mx):
                    _t = max(-1, min(1, val/mx))
                    if _t >= 0:
                        _g = int(166 + (255-166)*_t); return f"rgba(0,{_g},136,0.85)"
                    else:
                        _r = int(239 + (255-239)*abs(_t)); return f"rgba({_r},68,68,0.85)"

                # Render heatmap compatta
                _ht_html = (
                    "<div style='display:flex;gap:6px;flex-wrap:wrap;margin-top:4px'>"
                )
                _ht_sorted = sorted(_ht_data.items(), key=lambda x: -x[1][_ht_period])
                for _sec_lbl, _sec_d in _ht_sorted:
                    _sv = _sec_d[_ht_period]
                    _sc = _ht_color(_sv, _ht_max)
                    _ar = "▲" if _sv >= 0 else "▼"
                    _spark_s = _mini_sparkline_svg(_sec_d.get("spark",[]),
                                                    "#00ff88" if _sv>=0 else "#ef4444",
                                                    70, 22)
                    _ht_html += (
                        f"<div style='background:{_sc};border-radius:6px;"
                        f"padding:7px 10px;min-width:82px;flex:1;max-width:110px;"
                        f"text-align:center;cursor:default'>"
                        f"<div style='color:#fff;font-size:0.72rem;font-weight:bold;"
                        f"margin-bottom:2px'>{_sec_lbl}</div>"
                        f"<div style='color:#fff;font-family:Courier New;font-size:0.82rem;"
                        f"font-weight:bold'>{_ar}{abs(_sv):.1f}%</div>"
                        f"{_spark_s}"
                        f"</div>"
                    )
                _ht_html += "</div>"
                st.markdown(_ht_html, unsafe_allow_html=True)
                st.caption(f"ETF settoriali SPDR · aggiornamento ogni 5 min · ordinati per performance {_ht_period}")
            else:
                st.info("Dati settoriali non disponibili — riprova tra qualche secondo.")
        except Exception as _ht_e:
            st.warning(f"Heatmap settoriale: {str(_ht_e)[:80]}")

    # ── v41e: Scanner Summary — segnali rapidi visibili dalla home ──────────
    if not df_ep.empty:
        st.markdown("---")
        st.markdown('<div class="section-pill">⚡ SEGNALI RAPIDI — Top segnali attivi dal tuo scanner</div>',
                    unsafe_allow_html=True)
        _hs_c1, _hs_c2, _hs_c3 = st.columns(3)
        # Top 5 PRO/STRONG per CSS
        with _hs_c1:
            st.markdown("**💪 Top PRO / STRONG**")
            if "Stato_Pro" in df_ep.columns and "CSS" in df_ep.columns:
                _top_pro = (df_ep[df_ep["Stato_Pro"].isin(["PRO","STRONG"])]
                            .sort_values("CSS", ascending=False,
                                         key=lambda s: pd.to_numeric(s, errors="coerce").fillna(0))
                            .head(5))
                for _, _r in _top_pro.iterrows():
                    _sc = "#ffd700" if _r.get("Stato_Pro")=="STRONG" else "#00ff88"
                    _tv = str(_r.get("Ticker","")).replace(".MI","%3AMI")
                    _nome_pr=str(_r.get('Nome',_r.get('Company',_r.get('name','')))).strip()[:22]
                    _nome_lbl=f" <span style='color:#9ca3af;font-size:0.70rem;font-style:italic'>{_nome_pr}</span>" if _nome_pr else ''
                    st.markdown(
                        f"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' "
                        f"style='text-decoration:none'>"
                        f"<span style='font-family:Courier New;color:{_sc};font-weight:bold'>"
                        f"{_r.get('Ticker','')}</span></a>"
                        f"{_nome_lbl}"
                        f"<span style='color:#6b7280;font-size:0.72rem'> · CSS {_r.get('CSS','—')} · {_r.get('Stato_Pro','')}</span>",
                        unsafe_allow_html=True)
            else:
                st.caption("Avvia lo scanner")
        # Top EARLY per Early_Score
        with _hs_c2:
            st.markdown("**📡 Top EARLY**")
            if "Stato_Early" in df_ep.columns and "Early_Score" in df_ep.columns:
                _top_early = (df_ep[df_ep["Stato_Early"]=="EARLY"]
                              .sort_values("Early_Score", ascending=False,
                                            key=lambda s: pd.to_numeric(s, errors="coerce").fillna(0))
                              .head(5))
                for _, _r in _top_early.iterrows():
                    _tv = str(_r.get("Ticker","")).replace(".MI","%3AMI")
                    _nome_ea = str(_r.get('Nome', _r.get('Company', ''))).strip()[:22]
                    _nome_ea_lbl = (f" <span style='color:#9ca3af;font-size:0.70rem;font-style:italic'>{_nome_ea}</span>" if _nome_ea else '')
                    st.markdown(
                        f"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' "
                        f"style='text-decoration:none'>"
                        f"<span style='font-family:Courier New;color:#60a5fa;font-weight:bold'>"
                        f"{_r.get('Ticker','')}</span></a>"
                        f"{_nome_ea_lbl}"
                        f"<span style='color:#6b7280;font-size:0.72rem'> · E:{_r.get('Early_Score','—')} · RSI {_r.get('RSI','—')}</span>",
                        unsafe_allow_html=True)
            else:
                st.caption("Avvia lo scanner")
        # Top HOT per Vol_Ratio
        with _hs_c3:
            st.markdown("**🔥 Top REA-HOT**")
            if not df_rea.empty and "Vol_Ratio" in df_rea.columns:
                _top_hot = (df_rea
                            .sort_values("Vol_Ratio", ascending=False,
                                          key=lambda s: pd.to_numeric(s, errors="coerce").fillna(0))
                            .head(5))
                for _, _r in _top_hot.iterrows():
                    _tv = str(_r.get("Ticker","")).replace(".MI","%3AMI")
                    _vr = _r.get("Vol_Ratio","—")
                    _nome_hot = str(_r.get('Nome', _r.get('Company', ''))).strip()[:22]
                    _nome_hot_lbl = (f" <span style='color:#9ca3af;font-size:0.70rem;font-style:italic'>{_nome_hot}</span>" if _nome_hot else '')
                    st.markdown(
                        f"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' "
                        f"style='text-decoration:none'>"
                        f"<span style='font-family:Courier New;color:#f97316;font-weight:bold'>"
                        f"{_r.get('Ticker','')}</span></a>"
                        f"{_nome_hot_lbl}"
                        f"<span style='color:#6b7280;font-size:0.72rem'> · Vol×{_vr}</span>",
                        unsafe_allow_html=True)
            else:
                st.caption("Avvia lo scanner")

    # ── v43c: Suggerimenti ────────────────────────────────
    with st.expander('💡 Suggerimenti v43c — Novità e roadmap — Novità e roadmap', expanded=False):
        st.markdown("""
**✅ Implementato in v43c:**
- 📱 CSS Responsive: tab scrollabili, colonne fluid, touch 44px
- 🌙 Toggle Light/Dark tema persistente per sessione
- 🔄 Auto-refresh Home ogni N minuti con st.rerun schedulato
- 📊 Sparkline SVG 10d accanto a ogni ticker Top PRO/STRONG
- 🗃️ Export CSV + Excel 1-click dalla barra Home
- 📅 Earnings Tracker compatto con filtro giorni + export CSV
- 🧠 AI Analyst: storico analisi in SQLite (data/ai_history.db)
- 📊 COT Report: fetch automatico CFTC + grafici posizioni nette (Non-Comm vs Commercial)
- 🔔 Web Push Notifications browser (Notification API)
- 🤖 AI Batch multi-ticker: analisi Gemini/Groq + export CSV + salvataggio SQLite
- 📈 Pattern Recognition avanzato: 7 pattern candlestick + Volume Profile + POC
- 📱 CSS Responsive: tab scrollabili, colonne fluid, touch 44px
- 🌙 Toggle Light/Dark tema persistente per sessione
- 🔄 Auto-refresh Home ogni N minuti con st.rerun schedulato
- 📊 Sparkline SVG 10d accanto a ogni ticker Top PRO/STRONG
- 🗃️ Export CSV + Excel 1-click dalla barra Home
- 📅 Earnings Tracker compatto con filtro giorni + export CSV
- 🧠 AI Analyst: storico analisi in SQLite (data/ai_history.db)
- ⬆️ btt-btn: versione definitiva multi-target (`.main`, `.block-container`, `documentElement`, `body`, `window.parent`) + MutationObserver — comparsa garantita
- 🤖 MODULO 2 AI: aggiunte colonne **Vol×** e **Priorità** (auto-calcolata da Stato+CSS+RSI)
   layout 8 colonne identico a ⚡ Momentum Alerts
- 🐛 Fix `applymap→map`: patch compatibilità pandas ≥2.1 per Risk Manager
- 🤖 MODULO 2 AI: nuovo layout stile Momentum Alerts con sorgente multi-tab
   (EARLY / PRO / REA-HOT / CONFLUENCE / Serafini / Finviz / Watchlist / Manuale)
- ⬆️ btt-btn: MutationObserver + retry multiplo per comparsa affidabile
- 📅 Earnings Calendar: contenuto correttamente dentro il menu a tendina
- ⬆️ Bottone Torna su: visibile in basso a destra dopo 200px di scroll
- ⬆️ Bottone **Torna su** fisso in basso a destra (smooth scroll, appare dopo scroll)
- 🔢 Versione aggiornata a **42.0h** in tutti i titoli e label
- 🗺️ Mappa Calore Globale: card HTML responsive a 3 regioni + macro
- 🤖 Tab **Modulo 2 AI** dedicato (PRO / CONFLUENCE / Tutti)
- 📡 Alert Engine: notifiche **Telegram** auto su FIRED + **Email Gmail**
- 💪 Top PRO/STRONG: **Nome azienda** accanto al ticker
- 📑 Tab su 2 righe: CSS flex-wrap + JS MutationObserver
- 📊 Heatmap Settoriale Live nella Home (sostituisce bar chart)
- 🔗 Correlazioni Asset nel tab Settori

**🔜 Idee per v44:**
- 📡 WebSocket feed prezzi in tempo reale (Alpaca, Binance)
- 🔔 Alert push via Telegram bot
- 🤖 AI Analyst: analisi automatica schedulata via APScheduler
- 🧩 Modulo Risk Management: Kelly Criterion, position sizing
- 📊 Sector Rotation avanzata con ETF settoriali
- 📊 COT Report: fetch automatico dati CFTC + grafici posizioni nette
- 🔔 Alert push via browser (Web Push Notifications)
- 🤖 AI batch multi-ticker schedulata
- 📈 Pattern recognition avanzato (candlestick + volume profile)
- 📊 COT Report: fetch automatico dati CFTC via API (grafici posizione netta)
- 🔔 Alert push via browser (Web Push Notifications)
- 📊 Sparkline miniatura accanto al ticker nella Top PRO/STRONG
- 🗃️ Export segnali CSV/Excel con 1 click dalla Home
- 🔄 Auto-refresh Home ogni N minuti con `st.rerun()` schedulato
- 🧠 AI Analyst: storico analisi per ticker in SQLite
- 📱 Layout mobile-first con CSS container queries
- 📅 Earnings tracker compatto con filtro per giorni
- 🌙 Toggle tema persistente per sessione
        """)
    st.markdown("---")

    # ── v41e FEATURE 4 — Portfolio P&L dalla Watchlist ──────────────────────
    try:
        _pnl_positions_home = st.session_state.get("v41_pnl_entries", {})
        if _pnl_positions_home:
            st.markdown('<div class="section-pill">💰 PORTFOLIO P&L — Posizioni attive con prezzi scanner</div>',
                        unsafe_allow_html=True)
            _pnl_price_map = {}
            if not df_ep.empty and "Ticker" in df_ep.columns and "Prezzo" in df_ep.columns:
                for _, _pr_row in df_ep[["Ticker","Prezzo"]].dropna(subset=["Ticker"]).iterrows():
                    try:
                        _pnl_price_map[str(_pr_row["Ticker"])] = float(
                            pd.to_numeric(_pr_row["Prezzo"], errors="coerce") or 0)
                    except Exception:
                        pass

            _total_cost_h = 0.0; _total_val_h = 0.0; _total_pnl_h = 0.0
            _pnl_rows_h = []
            for _pt_h, _pp_h in _pnl_positions_home.items():
                _entry_h = float(_pp_h.get("entry",0))
                _size_h  = int(_pp_h.get("size",1))
                _cur_p_h = _pnl_price_map.get(_pt_h, 0)
                _cost_h  = _entry_h * _size_h
                _val_h   = _cur_p_h * _size_h if _cur_p_h > 0 else _cost_h
                _pnl_v_h = _val_h - _cost_h
                _pnl_p_h = (_cur_p_h/_entry_h - 1)*100 if _entry_h > 0 and _cur_p_h > 0 else 0
                _total_cost_h += _cost_h
                _total_val_h  += _val_h
                _total_pnl_h  += _pnl_v_h
                _pc_h = "#00ff88" if _pnl_v_h >= 0 else "#ef4444"
                _tv_h = _pt_h.replace(".MI","%3AMI")
                _cur_str = f"${_cur_p_h:.2f}" if _cur_p_h > 0 else "—"
                _pnl_rows_h.append(
                    f"<tr style='border-bottom:1px solid #1e222d'>"
                    f"<td style='padding:5px 8px'>"
                    f"<a href='https://it.tradingview.com/chart/?symbol={_tv_h}' target='_blank' "
                    f"style='color:#00ff88;font-family:Courier New;font-weight:bold;"
                    f"text-decoration:none;font-size:0.85rem'>{_pt_h}</a></td>"
                    f"<td style='padding:5px 8px;font-family:Courier New;color:#b2b5be'>${_entry_h:.2f}</td>"
                    f"<td style='padding:5px 8px;font-family:Courier New;color:#d1d4dc'>{_cur_str}</td>"
                    f"<td style='padding:5px 8px;color:#6b7280'>{_size_h}</td>"
                    f"<td style='padding:5px 8px;font-family:Courier New;color:{_pc_h};font-weight:bold'>"
                    f"${_pnl_v_h:+,.0f} ({_pnl_p_h:+.1f}%)</td>"
                    f"</tr>"
                )
            if _pnl_rows_h:
                _tc_h = "#00ff88" if _total_pnl_h >= 0 else "#ef4444"
                _ret_pct_h = (_total_val_h/_total_cost_h - 1)*100 if _total_cost_h > 0 else 0
                _pk1,_pk2,_pk3,_pk4 = st.columns(4)
                _pk1.metric("📦 Posizioni",  len(_pnl_positions_home))
                _pk2.metric("💰 Investito",  f"${_total_cost_h:,.0f}")
                _pk3.metric("💎 Valore",     f"${_total_val_h:,.0f}")
                _pk4.metric("📈 P&L",        f"${_total_pnl_h:+,.0f}  ({_ret_pct_h:+.1f}%)")
                st.markdown(
                    "<div style='overflow-x:auto'>"
                    "<table style='width:100%;border-collapse:collapse;font-size:0.82rem;background:#131722'>"
                    "<tr style='border-bottom:2px solid #2a2e39'>"
                    "<th style='padding:5px 8px;color:#50c4e0;text-align:left'>Ticker</th>"
                    "<th style='padding:5px 8px;color:#50c4e0'>Entry $</th>"
                    "<th style='padding:5px 8px;color:#50c4e0'>Prezzo attuale</th>"
                    "<th style='padding:5px 8px;color:#50c4e0'>Qty</th>"
                    "<th style='padding:5px 8px;color:#50c4e0'>P&L $</th></tr>"
                    + "".join(_pnl_rows_h) +
                    "</table></div>",
                    unsafe_allow_html=True)
                st.caption(
                    f"P&L totale: **${_total_pnl_h:+,.0f}** · "
                    "Prezzi dal tuo ultimo scanner · "
                    "Aggiungi posizioni nel tab **⚖️ Risk Manager → P&L Tracker**"
                )
            st.markdown("---")
    except Exception:
        pass

    st.markdown('<div class="section-pill">🗓️ MACRO CALENDAR v42h — Fed · CPI · NFP · PCE</div>', unsafe_allow_html=True)
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
                    _wl_save_backup()
                    st.success(f"✅ Aggiunti {len(tks)} ticker."); time.sleep(0.5); st.rerun()
                else:
                    st.warning("Seleziona almeno un asset dalla griglia.")
        with c_a2:
            if st.button(f"➕ Tutti ({len(assets)})", key=f"call_{_slug(cat_name)}"):
                tks=[r[0] for r in assets]; nms=[r[1] for r in assets]
                gh_add_to_watchlist(tks, nms, f"Crisis:{cat_name[:18]}", "CrisisMonitor",
                                 "WATCH", st.session_state.current_list_name)
                _wl_save_backup()
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

        import math as _math_r
        _safe_r = []
        for _rrec in df_riepilogo.to_dict(orient='records'):
            _safe_r.append({_k: (None if isinstance(_v, float) and
                (_math_r.isnan(_v) or _math_r.isinf(_v)) else _v)
                for _k, _v in _rrec.items()})
        df_riepilogo = pd.DataFrame(_safe_r)
        for _col_r in df_riepilogo.select_dtypes(include=['object']).columns:
            df_riepilogo[_col_r] = df_riepilogo[_col_r].fillna('—').replace({'nan':'—','None':'—','NaN':'—'})
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
            file_name=f"crisis_{selected_scenario[:25].replace(' ','_')}.txt",
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
        # ── v43c patch: pandas ≥2.1 applymap→map ─────────────────────────
        try:
            import pandas as _pd_patch
            if not hasattr(_pd_patch.io.formats.style.Styler, 'applymap'):
                _pd_patch.io.formats.style.Styler.applymap = _pd_patch.io.formats.style.Styler.map
        except Exception:
            pass
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

    with st.expander("📈 P&L Tracker & Alert Engine v41e", expanded=False):
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

            # ── v41e auto-notifica Telegram su FIRED ───────────────
            _tgtv=(st.secrets.get('TELEGRAM_TOKEN','') if hasattr(st,'secrets') else '') or st.session_state.get('tg_token_pnl','')
            _cgpv=(st.secrets.get('TELEGRAM_CHAT','')  if hasattr(st,'secrets') else '') or st.session_state.get('tg_chat_pnl','')
            if _tgtv and _cgpv and _alerts:
                for _fa in [a for a in _alerts.values() if a.get('fired') and not a.get('notified')]:
                    try:
                        import requests as _rqf
                        _rqf.post(f'https://api.telegram.org/bot{_tgtv}/sendMessage',
                            json={'chat_id':_cgpv,'text':f"🚨 ALERT v41e\n{_fa['tkr']} {_fa['type']} {_fa['val']}"},timeout=5)
                        _fa['notified']=True
                    except Exception: pass

        # ── v41e pannello notifiche ─────────────────────────────
        with st.expander('📡 Canali Notifica v41e', expanded=False):
            _nc1,_nc2=st.columns(2)
            with _nc1:
                st.markdown('**Telegram 🤖**')
                _tgtok=st.text_input('Bot Token',type='password',key='pnl_tg_token',placeholder='123456:ABC-xyz')
                _tgcht=st.text_input('Chat ID',key='pnl_tg_chat',placeholder='-100123456789')
                if _tgtok: st.session_state['tg_token_pnl']=_tgtok
                if _tgcht: st.session_state['tg_chat_pnl']=_tgcht
                if st.button('📨 Report P&L Telegram',key='pnl_tg_send'):
                    try:
                        import requests as _rqt
                        _rqt.post(f'https://api.telegram.org/bot{_tgtok}/sendMessage',
                            json={'chat_id':_tgcht,'text':'💼 Dashboard v41e P&L','parse_mode':'Markdown'},timeout=8)
                        st.success('✅ Inviato!')
                    except Exception as _e: st.error(str(_e))
                if st.button('🔔 Test Telegram',key='pnl_tg_test'):
                    try:
                        import requests as _rqt2
                        _rqt2.post(f'https://api.telegram.org/bot{_tgtok}/sendMessage',
                            json={'chat_id':_tgcht,'text':'✅ Test OK v41e'},timeout=5)
                        st.success('✅ Test inviato!')
                    except Exception as _e2: st.error(str(_e2))
            with _nc2:
                st.markdown('**Email 📧 Gmail**')
                _emto=st.text_input('Destinatario',key='pnl_email_to',placeholder='tuo@email.com')
                _emfrm=st.text_input('Mittente',key='pnl_email_from',placeholder='bot@gmail.com')
                _empwd=st.text_input('App Password',key='pnl_email_pwd',type='password')
                if st.button('📧 Report P&L Email',key='pnl_email_send'):
                    try:
                        import smtplib; from email.mime.text import MIMEText as _MT
                        _m=_MT('Dashboard v41e P&L'); _m['Subject']='P&L v41e'; _m['From']=_emfrm; _m['To']=_emto
                        with smtplib.SMTP_SSL('smtp.gmail.com',465) as _s: _s.login(_emfrm,_empwd); _s.send_message(_m)
                        st.success('✅ Email inviata!')
                    except Exception as _e3: st.error(str(_e3))

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
        _s1, _s2 = st.columns(2)
        with _s1:
            if st.button("💾 Backup locale", key="wl_backup_now",
                         help="Salva watchlist in file JSON locale — persiste tra deploy"):
                _wl_save_backup()
                st.success(f"✅ Backup salvato in {_WL_BACKUP_FILE}")
        with _s2:
            if _GH_SYNC:
                if st.button("☁️ Sync GitHub", key="wl_sync_now",
                             help="Forza upload watchlist → GitHub"):
                    _gh_push(DB_PATH)
                    _wl_save_backup()
                    st.success("✅ Watchlist inviata a GitHub + backup locale!")
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
            tv_txt_btn(df_wl_disp,f"watchlist_{st.session_state.current_list_name}.txt","exp_wl_dl")
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
                        gh_delete_from_watchlist(selected_ids); _wl_save_backup(); st.rerun()

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
            file_name=f"watchlist_{st.session_state.current_list_name}_tradingview.txt",
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
                            _wl_save_backup()
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
    st.markdown("""<style>
.bcd-wrap{width:100%;box-sizing:border-box;overflow-x:hidden;}
.bcd-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px;width:100%;box-sizing:border-box;}
.bcd-card{background:#1e222d;border:1px solid #2a2e39;border-radius:10px;padding:12px 14px;box-sizing:border-box;width:100%;}
.bcd-hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px;}
.bcd-ticker{font-family:Courier New,monospace;font-size:1rem;font-weight:bold;color:#00ff88;letter-spacing:1px;}
.bcd-name{color:#8b949e;font-size:0.75rem;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px;}
.bcd-score-num{font-family:Courier New,monospace;font-weight:bold;font-size:1.05rem;}
.bcd-score-num.sc-a{color:#00ff88;}.bcd-score-num.sc-b{color:#26a69a;}.bcd-score-num.sc-c{color:#f59e0b;}.bcd-score-num.sc-d{color:#ef4444;}
.bcd-badge{display:inline-block;border-radius:9px;padding:1px 7px;font-size:0.70rem;font-weight:bold;margin-top:3px;}
.bcd-badge.g-a{background:rgba(0,255,136,.15);color:#00ff88;border:1px solid #00ff8844;}
.bcd-badge.g-b{background:rgba(38,166,154,.15);color:#26a69a;border:1px solid #26a69a44;}
.bcd-badge.g-c{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid #f59e0b44;}
.bcd-badge.g-d{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid #ef444444;}
.bcd-row{display:flex;justify-content:space-between;align-items:center;margin-top:4px;font-size:0.78rem;}
.bcd-lbl{color:#6b7280;}.bcd-val{font-family:Courier New,monospace;color:#b2b5be;}.bcd-val.up{color:#00ff88;}.bcd-val.dn{color:#ef4444;}.bcd-val.hi{color:#ffd700;}
.bcd-bar-wrap{background:#1a1f2b;border-radius:3px;height:4px;margin-top:5px;overflow:hidden;}
.bcd-bar{height:4px;border-radius:3px;}
.bcd-kpi-row,.hist-kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0 12px 0;}
.bcd-kpi,.hist-kpi{background:#1e222d;border:1px solid #2a2e39;border-radius:8px;padding:8px 12px;text-align:center;}
.bcd-kpi-val,.hist-kpi-val{font-family:Courier New,monospace;font-size:1.15rem;font-weight:bold;color:#2962ff;}
.bcd-kpi-lbl,.hist-kpi-lbl{font-size:0.72rem;color:#6b7280;}
@media(max-width:700px){.bcd-grid{grid-template-columns:1fr;}.bcd-kpi-row,.hist-kpi-row{grid-template-columns:repeat(2,1fr);}.bcd-name{max-width:120px;}}
</style>""", unsafe_allow_html=True)
    st.markdown('<div class="section-pill">💎 BLUE CHIP DIP SCREENER</div>', unsafe_allow_html=True)
    with st.expander("📖 Metodologia BCD", expanded=False):
        st.markdown("""
**Blue Chip Dip** identifica large cap in correzione temporanea su trend rialzista di lungo periodo.

| Campo | Significato |
|---|---|
| **Dip Score** | 0–100 · qualità del dip (RSI basso + sopra EMA200 + liquidità alta) |
| **Dist. EMA200** | % sopra il supporto di lungo periodo |
| **RSI 14** | < 45 = zona dip · < 35 = dip profondo |
| **Vol Ratio** | volume odierno / media 20gg |
| **Grado** | A ≥ 65 · B ≥ 45 · C ≥ 25 · D < 25 |
""")
    c1, c2 = st.columns(2)
    with c1:
        bcd_rsi_max = st.slider("RSI max", 25, 55, 48, 1, key="bcd_rsi_max")
        bcd_min_mcap = st.selectbox("Min MarketCap (B$)", [1,2,5,10,20,50], index=2, key="bcd_mcap")
    with c2:
        bcd_min_score = st.slider("Dip Score min", 0, 80, 30, 5, key="bcd_minscore")
        bcd_top_n = st.selectbox("Top N risultati", [10,20,30,50], index=1, key="bcd_topn")
    bcd_ema200 = st.checkbox("Solo sopra EMA200 (trend rialzista principale)", value=True, key="bcd_ema200")

    def bcd_score(row):
        try:
            rsi = float(row.get("RSI", 50) or 50)
            pr = float(row.get("Prezzo", 1) or 1)
            e200 = float(row.get("EMA200", pr) or pr)
            volr = float(row.get("VolRatio", 1) or 1)
            mcap = float(row.get("MarketCap", 0) or 0)
            dvol = float(row.get("DollarVol_M", 0) or 0)
            if pr <= 0 or e200 <= 0:
                return 0
            rsi_c = max(0, (50 - rsi) / 20 * 40)
            dist = (pr - e200) / e200 * 100
            e200_c = 25 if dist < 5 else 20 if dist < 15 else 10 if dist < 30 else 5
            if dist < 0:
                e200_c = 0
            dvol_c = min(dvol / 200 * 15, 15)
            volr_c = min(max(0, (volr - 0.8) / 2.2 * 10), 10)
            mc_c = 5 if mcap > 20 else 2 if mcap > 5 else 0
            return round(min(rsi_c + e200_c + dvol_c + volr_c + mc_c, 100), 1)
        except Exception:
            return 0

    @st.cache_data(ttl=300, show_spinner=False)
    def bcd_scan(min_mcap_b, rsi_max, ema200_only):
        import yfinance as yf
        universe = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","JNJ","UNH","V","MA","PG","HD","ABBV","MRK","AVGO","CVX","KO","PEP","BAC","COST","WMT","TMO","ABT","MCD","CRM","NEE","ACN","NKE","DHR","TXN","PM","UPS","AMGN","LOW","BMY","IBM","QCOM","RTX","GS","MS","SPGI","CAT","DE","LIN","SYK","ISRG","BKNG","AXP","HON","MMM","GE","BLK","SCHW","T","VZ","DIS","NFLX","ORCL","ADBE","AMD","INTC","AMAT","MU","NOW","SNOW","ENI.MI","ENEL.MI","ISP.MI","UCG.MI","STM.MI","RACE.MI","G.MI","TIT.MI"]
        rows = []
        prog = st.progress(0.0, text="BCD Scan in corso…")
        for i, tk in enumerate(universe):
            prog.progress((i + 1) / len(universe), text=f"Analisi {tk}…")
            try:
                fi = yf.Ticker(tk).fast_info
                mc = getattr(fi, "market_cap", 0) or 0
                if mc < min_mcap_b * 1e9:
                    continue
                raw = yf.download(tk, period="14mo", interval="1d", auto_adjust=True, progress=False)
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                cl = raw["Close"].dropna()
                if len(cl) < 50:
                    continue
                pr = float(cl.iloc[-1])
                e200 = float(cl.rolling(200, min_periods=50).mean().iloc[-1])
                if ema200_only and pr < e200:
                    continue
                d = cl.diff(); g = d.clip(lower=0); l = (-d.clip(upper=0))
                rsi = float((100 - 100 / (1 + g.ewm(com=13).mean() / (l.ewm(com=13).mean() + 1e-9))).iloc[-1])
                if rsi > rsi_max:
                    continue
                e50 = float(cl.ewm(span=50).mean().iloc[-1])
                vol = float(raw["Volume"].iloc[-1]) if "Volume" in raw.columns else 0
                v20 = float(raw["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in raw.columns else 1
                volr = vol / v20 if v20 > 0 else 1
                dvol = pr * vol / 1e6
                if dvol < 5:
                    continue
                atr = float((raw["High"] - raw["Low"]).rolling(14).mean().iloc[-1])
                atrp = atr / pr * 100 if pr > 0 else 2
                hi52 = float(cl.rolling(252, min_periods=20).max().iloc[-1])
                try:
                    nm = yf.Ticker(tk).info.get("longName", tk)[:28]
                except Exception:
                    nm = tk
                ytd = 0
                try:
                    yr = cl.resample("YE").last()
                    if len(yr) > 1:
                        ytd = (pr / float(yr.iloc[-2]) - 1) * 100
                except Exception:
                    pass
                rows.append({"Ticker": tk, "Nome": nm, "Prezzo": round(pr, 2), "RSI": round(rsi, 1), "EMA200": round(e200, 2), "EMA50": round(e50, 2), "Dist_EMA200_%": round((pr / e200 - 1) * 100, 1), "Dist_52wHigh_%": round((pr / hi52 - 1) * 100, 1), "ATRpct": round(atrp, 2), "VolRatio": round(volr, 2), "DollarVol_M": round(dvol, 1), "MarketCap": round(mc / 1e9, 1), "YTD_%": round(ytd, 1)})
            except Exception:
                continue
        prog.empty()
        return rows

    a1, a2, a3 = st.columns([2,1,1])
    with a1:
        run_bcd = st.button("🔍 Avvia BCD Scan", key="bcd_run", type="primary", use_container_width=True)
    with a2:
        clear_bcd = st.button("🗑️ Svuota cache", key="bcd_clear", use_container_width=True)
    with a3:
        reset_bcd = st.button("❌ Reset risultati", key="bcd_reset_res", use_container_width=True)

    if clear_bcd:
        bcd_scan.clear(); st.success("Cache BCD svuotata.")
    if reset_bcd:
        st.session_state["bcd_results"] = []; st.rerun()

    bcd_df = pd.DataFrame(st.session_state.get("bcd_results", []))
    if run_bcd:
        with st.spinner("BCD Scan…"):
            raw_bcd = bcd_scan(bcd_min_mcap, bcd_rsi_max, bcd_ema200)
        if raw_bcd:
            bcd_df = pd.DataFrame(raw_bcd)
            bcd_df["DipScore"] = bcd_df.apply(bcd_score, axis=1)
            bcd_df["Grade"] = bcd_df["DipScore"].apply(lambda v: "A" if v >= 65 else "B" if v >= 45 else "C" if v >= 25 else "D")
            bcd_df = bcd_df.sort_values("DipScore", ascending=False).reset_index(drop=True)
            st.session_state["bcd_results"] = bcd_df.to_dict("records")
        else:
            st.warning("Nessun titolo trovato. Prova ad allargare i filtri.")

    if not bcd_df.empty:
        bcd_view = bcd_df[bcd_df["DipScore"] >= bcd_min_score].head(bcd_top_n)
        gA = int((bcd_view["Grade"] == "A").sum())
        gB = int((bcd_view["Grade"] == "B").sum())
        st.markdown(f"""<div class="bcd-kpi-row"><div class="bcd-kpi"><div class="bcd-kpi-val">{len(bcd_view)}</div><div class="bcd-kpi-lbl">💎 Candidati</div></div><div class="bcd-kpi"><div class="bcd-kpi-val" style="color:#00ff88">{gA}</div><div class="bcd-kpi-lbl">🏆 Grado A</div></div><div class="bcd-kpi"><div class="bcd-kpi-val" style="color:#26a69a">{gB}</div><div class="bcd-kpi-lbl">✅ Grado B</div></div><div class="bcd-kpi"><div class="bcd-kpi-val" style="color:#f59e0b">{bcd_view["RSI"].mean():.1f}</div><div class="bcd-kpi-lbl">📊 RSI medio</div></div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="bcd-grid">', unsafe_allow_html=True)
        for _, rb in bcd_view.iterrows():
            tk = str(rb.get("Ticker", "")); nm = str(rb.get("Nome", tk))[:30]
            pr = rb.get("Prezzo", 0); rsi = rb.get("RSI", 0); ds = rb.get("DipScore", 0); grd = rb.get("Grade", "D")
            dist = rb.get("Dist_EMA200_%", 0); dh = rb.get("Dist_52wHigh_%", 0); volr = rb.get("VolRatio", 1); ytd = rb.get("YTD_%", 0); mc = rb.get("MarketCap", 0); dv = rb.get("DollarVol_M", 0)
            sc_cls = f"sc-{grd.lower()}"; g_cls = f"g-{grd.lower()}"; bc = "#00ff88" if ds >= 65 else "#f59e0b" if ds >= 45 else "#6b7280"
            tv = tk.replace('.MI', '%3AMI').replace('.L', '%3AL')
            ytd_cls = "up" if ytd >= 0 else "dn"; rsi_cls = "up" if rsi < 40 else "dn" if rsi > 60 else ""; vol_cls = "hi" if volr >= 1.2 else ""
            st.markdown(f"""<div class="bcd-card"><div class="bcd-hdr"><div><a href="https://it.tradingview.com/chart/?symbol={tv}" target="_blank" rel="noopener noreferrer" style="text-decoration:none"><span class="bcd-ticker">{tk}</span></a><div class="bcd-name">{nm}</div></div><div style="text-align:right"><span class="bcd-score-num {sc_cls}">{ds:.0f}</span><div><span class="bcd-badge {g_cls}">{grd}</span></div></div></div><div class="bcd-bar-wrap"><div class="bcd-bar" style="width:{min(int(ds),100)}%;background:{bc}"></div></div><div class="bcd-row"><span class="bcd-lbl">Prezzo</span><span class="bcd-val">{pr:.2f}</span></div><div class="bcd-row"><span class="bcd-lbl">RSI 14</span><span class="bcd-val {rsi_cls}">{rsi:.1f}</span></div><div class="bcd-row"><span class="bcd-lbl">Dist. EMA200</span><span class="bcd-val">{dist:+.1f}%</span></div><div class="bcd-row"><span class="bcd-lbl">Dist. 52w High</span><span class="bcd-val dn">{dh:.1f}%</span></div><div class="bcd-row"><span class="bcd-lbl">Vol Ratio</span><span class="bcd-val {vol_cls}">{volr:.2f}x</span></div><div class="bcd-row"><span class="bcd-lbl">DollarVol</span><span class="bcd-val">{dv:.0f}M</span></div><div class="bcd-row"><span class="bcd-lbl">MarketCap</span><span class="bcd-val">{mc:.1f}B</span></div><div class="bcd-row"><span class="bcd-lbl">YTD</span><span class="bcd-val {ytd_cls}">{ytd:+.1f}%</span></div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('---')
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button('📥 TXT TradingView', data=make_tv_txt(bcd_view), file_name=f"BCD_TV_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime='text/plain', key='bcd_tv_exp', use_container_width=True)
        with ex2:
            bx = io.BytesIO()
            with pd.ExcelWriter(bx, engine='xlsxwriter') as bw:
                bcd_view.to_excel(bw, sheet_name='BluChipDip', index=False)
            st.download_button('📊 XLSX', data=bx.getvalue(), file_name=f"BCD_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='bcd_xl_exp', use_container_width=True)
        with st.expander('📋 Tabella completa', expanded=False):
            show_c = [c for c in ["Ticker","Nome","DipScore","Grade","Prezzo","RSI","Dist_EMA200_%","Dist_52wHigh_%","VolRatio","DollarVol_M","MarketCap","YTD_%"] if c in bcd_view.columns]
            st.dataframe(bcd_view[show_c], use_container_width=True)
    else:
        st.info('▶️ Clicca **Avvia BCD Scan** per cercare Blue Chip in dip.')

    st.markdown('<div class="section-pill">📜 STORICO SCANSIONI</div>', unsafe_allow_html=True)
    with st.expander('📖 Guida Storico Scansioni', expanded=False):
        st.markdown("""
Lo **Storico** registra ogni scansione eseguita nel DB locale.

| Colonna | Significato |
|---|---|
| **scanned_at** | Data/ora esecuzione UTC |
| **markets** | Mercati inclusi |
| **n_early/n_pro** | Segnali trovati |
| **elapsed_s** | Durata scan |

**Confronto Snapshot**: seleziona A (baseline) e B (più recente) per vedere ticker entrati/usciti.
""")
    _, col_rst = st.columns([4,1])
    with col_rst:
        if st.button('🗑️ Reset storico', key='rst_hist', type='secondary', use_container_width=True):
            conn = sqlite3.connect(str(DB_PATH)); conn.execute('DELETE FROM scan_history'); conn.commit(); conn.close(); st.success('Storico cancellato!'); st.rerun()
    df_hist = load_scan_history(20)
    if df_hist.empty:
        st.info('📭 Nessuna scansione salvata. Avvia lo scanner dalla sidebar.')
    else:
        if 'elapsed_s' in df_hist.columns:
            df_hist['elapsed_s'] = df_hist['elapsed_s'].apply(lambda x: f"{x:.0f}s" if pd.notna(x) else '—')
        st.markdown(f"""<div class="hist-kpi-row"><div class="bcd-kpi"><div class="bcd-kpi-val">{len(df_hist)}</div><div class="bcd-kpi-lbl">📋 Scan totali</div></div><div class="bcd-kpi"><div class="bcd-kpi-val">{int(df_hist['n_early'].max()) if 'n_early' in df_hist.columns else '—'}</div><div class="bcd-kpi-lbl">📡 Max EARLY</div></div><div class="bcd-kpi"><div class="bcd-kpi-val">{int(df_hist['n_pro'].max()) if 'n_pro' in df_hist.columns else '—'}</div><div class="bcd-kpi-lbl">💪 Max PRO</div></div><div class="bcd-kpi"><div class="bcd-kpi-val">{f"{df_hist['n_tickers'].mean():.0f}" if 'n_tickers' in df_hist.columns else '—'}</div><div class="bcd-kpi-lbl">🔭 Titoli medi</div></div></div>""", unsafe_allow_html=True)
        st.dataframe(df_hist, use_container_width=True)
        st.subheader('🔍 Confronto Snapshot')
        h1, h2 = st.columns(2)
        def slbl(row):
            return f"{str(row.get('scanned_at',''))[:16]} | E:{int(row.get('n_early',0))} P:{int(row.get('n_pro',0))}"
        smap = {row['id']: slbl(row) for _, row in df_hist.iterrows()}
        ids = list(smap.keys())
        with h1:
            id_a = st.selectbox('📅 Scansione A', ids, format_func=lambda i: smap[i], key='sn_a')
        with h2:
            id_b = st.selectbox('📅 Scansione B', ids, format_func=lambda i: smap[i], index=min(1, len(ids)-1), key='sn_b')
        if st.button('🔍 Confronta', use_container_width=True, key='sn_cmp'):
            ea, _ = load_scan_snapshot(id_a); eb, _ = load_scan_snapshot(id_b)
            if ea.empty or eb.empty:
                st.warning('Dati non disponibili per uno dei due snapshot.')
            else:
                ta = set(ea.get('Ticker', pd.Series()).tolist()); tb = set(eb.get('Ticker', pd.Series()).tolist())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('🆕 Nuovi in B', len(tb - ta))
                c2.metric('❌ Usciti da A', len(ta - tb))
                c3.metric('✅ Persistenti', len(ta & tb))
                c4.metric('📊 Overlap %', f"{len(ta & tb) / max(len(ta | tb), 1) * 100:.0f}%")
                r1, r2 = st.columns(2)
                with r1:
                    if tb - ta:
                        st.markdown('**🆕 Nuovi in B:**')
                        st.code('  '.join(sorted(tb - ta)))
                    if ta - tb:
                        st.markdown('**❌ Usciti da A:**')
                        st.code('  '.join(sorted(ta - tb)))
                with r2:
                    if ta & tb:
                        st.markdown(f"**✅ Persistenti ({len(ta & tb)}):**")
                        st.code('  '.join(sorted(ta & tb)))

    st.markdown(BACK_TO_TOP_CSS, unsafe_allow_html=True)
with tab_mom:
    st.markdown("""<style>
.mom-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;width:100%;box-sizing:border-box;}
.mom-card{background:#1e222d;border:1px solid #2a2e39;border-radius:10px;padding:12px 14px;box-sizing:border-box;width:100%;}
.mom-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px;}
.mom-ticker{font-family:Courier New,monospace;font-size:1rem;font-weight:bold;color:#00ff88;}
.mom-name{color:#8b949e;font-size:0.75rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:170px;}
.mom-badge{display:inline-block;border-radius:9px;padding:1px 7px;font-size:0.70rem;font-weight:bold;margin-top:3px;}
.mom-row{display:flex;justify-content:space-between;align-items:center;margin-top:4px;font-size:0.78rem;}
.mom-lbl{color:#6b7280;}.mom-val{font-family:Courier New,monospace;color:#d1d4dc;}
.mom-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0 8px 0;}
.mom-kpi{background:#1e222d;border:1px solid #2a2e39;border-radius:8px;padding:8px 12px;text-align:center;}
.mom-kpi-val{font-family:Courier New,monospace;font-size:1.15rem;font-weight:bold;color:#2962ff;}
.mom-kpi-lbl{font-size:0.72rem;color:#6b7280;}
@media(max-width:900px){.mom-kpis{grid-template-columns:repeat(2,1fr);}}
@media(max-width:700px){.mom-grid{grid-template-columns:1fr;}.mom-kpis{grid-template-columns:1fr 1fr;}.mom-name{max-width:120px;}}
</style>""", unsafe_allow_html=True)
    st.markdown('<div class="section-pill">⚡ MOMENTUM ALERTS v44c</div>', unsafe_allow_html=True)
    st.caption("Rileva breakout, volume spike e pattern tecnici sui tuoi ticker. Layout responsive e adattivo.")
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: ma_breakout = st.checkbox("🚀 Breakout >2%", value=True, key="ma_brk")
    with m2: ma_vol_spike = st.checkbox("📈 Volume >3x", value=True, key="ma_vol")
    with m3: ma_rsi_ob = st.checkbox("🔴 RSI Overbought", value=True, key="ma_rsi_ob")
    with m4: ma_rsi_os = st.checkbox("🔵 RSI Oversold", value=True, key="ma_rsi_os")
    with m5: ma_squeeze = st.checkbox("🔥 Squeeze Fire", value=True, key="ma_sq")
    u1,u2 = st.columns([3,1])
    with u1:
        ma_universe = st.text_input("Ticker da monitorare (o usa quelli dal tuo scanner)", value="SPY,QQQ,NVDA,TSLA,AAPL,AMD,META,GOOGL,MSFT,AMZN", key="ma_universe").strip()
    with u2:
        st.write("")
        if st.checkbox("📊 Usa ticker scanner", key="ma_use_scanner") and not df_ep.empty and "Ticker" in df_ep.columns:
            ma_universe = ",".join(df_ep["Ticker"].dropna().tolist()[:30])
            st.caption(f"{len(df_ep)} ticker dallo scanner")
    if "mom_alerts" not in st.session_state:
        st.session_state.mom_alerts = []
    b1,b2,b3 = st.columns([2,1,1])
    with b1: run_ma = st.button("⚡ Scan Momentum Now", key="ma_scan_run", type="primary", use_container_width=True)
    with b2: ma_min_move = st.number_input("Breakout % min", 1.0, 10.0, 2.0, 0.5, key="ma_min_move")
    with b3: ma_vol_thr = st.number_input("Volume soglia ×", 1.5, 10.0, 3.0, 0.5, key="ma_vol_thr")
    if run_ma:
        import yfinance as yf
        tickers = [t.strip().upper() for t in ma_universe.split(",") if t.strip()][:40]
        new_alerts = []
        prog = st.progress(0.0)
        status = st.empty()
        for i, tk in enumerate(tickers):
            prog.progress((i+1)/len(tickers))
            status.caption(f"Analisi {tk}... ({i+1}/{len(tickers)})")
            try:
                raw = yf.download(tk, period="10d", interval="1d", auto_adjust=True, progress=False)
                raw.columns = [co[0] if isinstance(co, tuple) else co for co in raw.columns]
                if len(raw) < 3:
                    continue
                cl = raw["Close"].dropna(); vol = raw["Volume"].dropna() if "Volume" in raw.columns else None
                hi = raw["High"].dropna(); lo = raw["Low"].dropna()
                price = float(cl.iloc[-1]); prev = float(cl.iloc[-2]); chg_pct = (price/prev - 1)*100
                d = cl.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
                rsi_v = float((100 - 100/(1 + g.ewm(com=13, adjust=False).mean()/(l.ewm(com=13, adjust=False).mean()+1e-10))).iloc[-1])
                atr_v = float(((hi - lo).ewm(com=13, adjust=False).mean()).iloc[-1])
                sma20 = cl.rolling(20).mean().iloc[-1]; std20 = cl.rolling(20).std().iloc[-1]
                squeeze = (abs(price - float(sma20)) < float(std20)*1.5) if not pd.isna(sma20) else False
                vol_ratio = 0.0
                if vol is not None and len(vol) >= 5:
                    vol_ratio = float(vol.iloc[-1]) / float(vol.iloc[:-1].mean()) if float(vol.iloc[:-1].mean()) > 0 else 0
                nome = ""
                if not df_ep.empty and "Ticker" in df_ep.columns and "Nome" in df_ep.columns:
                    nm = df_ep[df_ep["Ticker"] == tk]
                    if not nm.empty:
                        nome = str(nm.iloc[0].get("Nome", ""))[:22]
                css = "—"
                if not df_ep.empty and "Ticker" in df_ep.columns and "CSS" in df_ep.columns:
                    cm = df_ep[df_ep["Ticker"] == tk]
                    if not cm.empty:
                        css = cm.iloc[0].get("CSS", "—")
                def add_alert(tipo, valore, priorita):
                    new_alerts.append({"Ticker": tk, "Nome": nome, "Tipo": tipo, "Valore": valore, "Prezzo": f"${price:.2f}", "RSI": f"{rsi_v:.1f}", "Vol×": f"{vol_ratio:.1f}x", "CSS": css, "Priorità": priorita})
                if ma_breakout and abs(chg_pct) >= ma_min_move:
                    add_alert("🚀 Breakout ▲" if chg_pct > 0 else "💥 Breakdown ▼", f"{chg_pct:+.1f}%", "🔴 ALTA" if abs(chg_pct) >= ma_min_move*2 else "🟡 MEDIA")
                if ma_vol_spike and vol_ratio >= ma_vol_thr:
                    add_alert("📈 Volume Spike", f"{vol_ratio:.1f}×", "🔴 ALTA" if vol_ratio >= ma_vol_thr*1.5 else "🟡 MEDIA")
                if ma_rsi_ob and rsi_v >= 70:
                    add_alert("🔴 RSI Overbought", f"RSI {rsi_v:.1f}", "🟡 MEDIA")
                if ma_rsi_os and rsi_v <= 30:
                    add_alert("🔵 RSI Oversold", f"RSI {rsi_v:.1f}", "🟢 BASSA")
                if ma_squeeze and squeeze:
                    add_alert("🔥 Squeeze Fire", f"ATR {atr_v:.2f}", "🟡 MEDIA")
            except Exception:
                continue
        prog.empty(); status.empty()
        order = {"🔴 ALTA": 0, "🟡 MEDIA": 1, "🟢 BASSA": 2}
        new_alerts.sort(key=lambda x: order.get(x.get("Priorità", ""), 3))
        st.session_state.mom_alerts = new_alerts
        if new_alerts:
            st.success(f"✅ {len(new_alerts)} alert trovati su {len(tickers)} ticker!")
        else:
            st.info("Nessun alert con i parametri attuali. Prova ad abbassare le soglie.")
    if st.session_state.mom_alerts:
        all_alerts = st.session_state.mom_alerts
        high = sum(1 for a in all_alerts if a.get("Priorità", "") == "🔴 ALTA")
        med = sum(1 for a in all_alerts if a.get("Priorità", "") == "🟡 MEDIA")
        low = sum(1 for a in all_alerts if a.get("Priorità", "") == "🟢 BASSA")
        st.markdown(f"""<div class="mom-kpis"><div class="mom-kpi"><div class="mom-kpi-val">{len(all_alerts)}</div><div class="mom-kpi-lbl">⚡ Totale Alert</div></div><div class="mom-kpi"><div class="mom-kpi-val">{high}</div><div class="mom-kpi-lbl">🔴 Alta</div></div><div class="mom-kpi"><div class="mom-kpi-val">{med}</div><div class="mom-kpi-lbl">🟡 Media</div></div><div class="mom-kpi"><div class="mom-kpi-val">{low}</div><div class="mom-kpi-lbl">🟢 Bassa</div></div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="mom-grid">', unsafe_allow_html=True)
        for al in all_alerts[:50]:
            pcolor = "#ef4444" if "ALTA" in al.get("Priorità", "") else "#f59e0b" if "MEDIA" in al.get("Priorità", "") else "#26a69a"
            ttype = al.get("Tipo", "")
            tcolor = "#ef4444" if "Breakdown" in ttype or "Overbought" in ttype else "#f97316" if "Volume" in ttype or "Squeeze" in ttype else "#60a5fa" if "Oversold" in ttype else "#00ff88"
            tv_s = al['Ticker'].replace('.MI', '%3AMI').replace('.L', '%3AL')
            st.markdown(f"""<div class="mom-card"><div class="mom-top"><div><a href="https://it.tradingview.com/chart/?symbol={tv_s}" target="_blank" rel="noopener noreferrer" style="text-decoration:none"><span class="mom-ticker">{al['Ticker']}</span></a><div class="mom-name">{al.get('Nome','')}</div></div><div style="text-align:right"><span class="mom-badge" style="background:{pcolor}22;color:{pcolor};border:1px solid {pcolor}44">{al.get('Priorità','')}</span></div></div><div class="mom-row"><span class="mom-lbl">Tipo</span><span class="mom-val" style="color:{tcolor};font-weight:bold">{ttype}</span></div><div class="mom-row"><span class="mom-lbl">Valore</span><span class="mom-val">{al.get('Valore','')}</span></div><div class="mom-row"><span class="mom-lbl">Prezzo</span><span class="mom-val">{al.get('Prezzo','')}</span></div><div class="mom-row"><span class="mom-lbl">RSI</span><span class="mom-val">{al.get('RSI','')}</span></div><div class="mom-row"><span class="mom-lbl">Vol×</span><span class="mom-val">{al.get('Vol×','')}</span></div><div class="mom-row"><span class="mom-lbl">CSS</span><span class="mom-val">{al.get('CSS','')}</span></div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('---')
        d1, d2 = st.columns([1,1])
        with d1:
            st.download_button('📥 TXT TradingView', data=make_tv_txt(pd.DataFrame(all_alerts)), file_name=f"momentum_alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime='text/plain', key='ma_export_txt', use_container_width=True)
        with d2:
            if st.button('🗑️ Pulisci Alert', key='ma_clear_alerts', use_container_width=True):
                st.session_state.mom_alerts = []
                st.rerun()
    else:
        st.info('Nessun alert attivo. Clicca **⚡ Scan Momentum Now** per avviare la ricerca.')
    with st.expander('📖 Come funziona — Momentum Alerts', expanded=False):
        st.markdown("""
| Alert | Trigger | Priorità |
|---|---|---|
| 🚀 Breakout | Variazione % ≥ soglia | ALTA se 2× soglia |
| 💥 Breakdown | Variazione negativa ≥ soglia | ALTA se 2× soglia |
| 📈 Volume Spike | Volume > N× media 9 giorni | ALTA se 1.5× soglia |
| 🔴 RSI Overbought | RSI ≥ 70 | MEDIA |
| 🔵 RSI Oversold | RSI ≤ 30 | BASSA |
| 🔥 Squeeze Fire | BB strette / ATR basso | MEDIA |
""")
    st.markdown(BACK_TO_TOP_CSS, unsafe_allow_html=True)
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
                _nc1, _nc2, _nc3 = st.columns([1, 0.8, 4])
                _nc1.markdown(f"**{_n['Ticker']}**", unsafe_allow_html=True)
                _nc2.markdown(f"<span style='color:{_sc};font-weight:bold'>{_n['Sentiment']}</span>", unsafe_allow_html=True)
                _nc3.markdown(f"[{_n['Titolo']}]({_n['Link']}) <span style='color:#6b7280;font-size:0.7rem'>{_n['Data']}</span>", unsafe_allow_html=True)
                st.divider()

            # Export
            _ns_df = pd.DataFrame(_news_data)
            _ns_csv = make_tv_txt(_ns_df)
            st.download_button("📥 Export CSV", _ns_csv, "news_sentiment.txt", "text/plain")
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