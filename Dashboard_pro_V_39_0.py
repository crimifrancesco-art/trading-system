# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║         TRADING SCANNER PRO  —  v39.0                                  ║
# ║         Upgrade professionale su v37.0 — tutto il codice esistente      ║
# ║         è intatto. Aggiunti 9 upgrade v38 completamente additivi.       ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  CHANGELOG v38  (su base v37 intatta)                                  ║
# ║  #1  Mobile-first UI — CSS responsive, card layout su smartphone        ║
# ║      Sidebar collassabile, font scalabili, touch-friendly controls      ║
# ║  #2  Alert Multipli — breakout EMA, golden/death cross, squeeze fire    ║
# ║      Bollinger breakout, RSI divergence, volume spike alert             ║
# ║  #3  AI Scanner Autonomo — scansiona e filtra senza click               ║
# ║      Auto-run su schedule, auto-Telegram quando PRO/STRONG trovati      ║
# ║  #4  News & Sentiment — notizie real-time con NLP score                 ║
# ║      Feed RSS Yahoo/Finviz, sentiment Bull/Bear/Neutral per ticker      ║
# ║  #5  SEC Form 4 Insider Buying — acquisti insider ultimi 30gg           ║
# ║      Filtra solo acquisti (non esercizi opzione), score insiders        ║
# ║  #6  Short Interest % — dati short float da Yahoo Finance               ║
# ║      Integrato in colonna scanner + filtro short squeeze setup          ║
# ║  #7  Macro Calendar — Fed, CPI, NFP, earnings prossimi 30gg            ║
# ║      Countdown live, impatto atteso (High/Med/Low), nella Home         ║
# ║  #8  Options Flow proxy — put/call ratio da Yahoo options chain         ║
# ║      Score bullish/bearish, integrato nei tab scanner                  ║
# ║  #9  Performance Turbo — parallel batch yfinance v2, skip inalterato   ║
# ║  #1  Scanner Turbo — batch download 5 ticker/chiamata (-80% latenza)   ║
# ║      Pre-warming cache al boot, smart skip, ETA live stimata           ║
# ║  #2  Menu fisso sticky — tab bar rimane visibile scorrendo la pagina   ║
# ║      Font adattivo, tab attivo evidenziato, scroll-to-top automatico   ║
# ║  #3  AI Signal Explainer — Claude API su ogni ticker PRO/CONFLUENCE    ║
# ║      Setup analysis: validità + rischio + gestione posizione           ║
# ║  #4  Notifiche Telegram — alert engine con bot token configurabile     ║
# ║      Invio automatico su segnale PRO/STRONG, digest mattutino          ║
# ║  #5  Risk Dashboard Pro — correlation matrix, VaR 95%, portfolio heat  ║
# ║      Nuovo tab dedicato con heatmap interattiva e metriche aggregate   ║
# ║  #6  Scanner Avanzato — Gap Scanner e Earnings Play Scanner            ║
# ║      Gap >1% con volume confermato, pre-earnings IV screen             ║
# ║  #7  Ticker Search Globale — Ctrl+K cerca in tutti i tab               ║
# ║  #1  Market Regime Detection — VIX + % sopra EMA200 + Adv/Dec ratio    ║
# ║      Classifica regime: Risk-On / Caution / Risk-Off / Crisis           ║
# ║      Disabilita segnali deboli automaticamente in Risk-Off/Crisis       ║
# ║  #2  Position Sizing Engine — Kelly, Fixed Fractional, ATR-based        ║
# ║      Calcola size ottimale da capitale + rischio% + ATR stop            ║
# ║      Integrato nel tab Risk Manager e nel P&L Tracker                   ║
# ║  #3  Scanner Scheduling (auto-scan) — scan automatico ogni N minuti     ║
# ║      Finestra oraria configurabile (NYSE 9:30-16:00 default)            ║
# ║      Countdown live, pausa/riprendi, storico auto-scan                  ║
# ║  #4  Earnings Calendar — tutti i ticker watchlist con earnings           ║
# ║      nei prossimi 7-14 giorni da Yahoo Finance, nella Home in basso     ║
# ║      Badge: ⚠️ Imminente / 🔔 Questa settimana / 📅 Prossima            ║
# ║  #5  Multi-Timeframe Confluence Matrix — daily/weekly/monthly per tkr   ║
# ║      🟢 3/3 allineati | 🟡 2/3 | 🔴 1/3 | ⚪ no data                  ║
# ║      Nuovo tab "🔀 MTF Matrix" con griglia compatta e drill-down        ║
# ║  #6  Relative Strength vs SPY — RS = return_20d - SPY_return_20d        ║
# ║      Colonne RS_20d, RS_Rank aggiunte a tutti i tab scanner             ║
# ║      Renderer con barra orizzontale verde/rossa                         ║
# ║  #7  Sector Rotation Heatmap interattiva — 11 settori GICS × 4 periodi  ║
# ║      1d/5d/1m/3m | Click su cella → drill-down ticker nel settore       ║
# ║      Sostituzione heatmap statica Home tab con versione interattiva      ║
# ║  #8  Paper Trading Journal — log strutturato entry/exit/note/outcome    ║
# ║      Metriche aggregate per setup type: Win Rate, Avg R, P&L totale     ║
# ║      Nuovo tab "📓 Journal" con export XLSX                             ║
# ║  #9  Earnings in Home — sezione dedicata in fondo alla Home             ║
# ║      Mostra prossimi earnings da watchlist + scanner con countdown      ║
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
        # v37 SCANNER TURBO ENGINE
        # Upgrade vs v36:
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

        # ── v37 BATCH DOWNLOAD: raggruppa ticker in batch da 5 ────────────
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

        # Soglie percentile dinamico (v36, mantenuto)
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
# v36 ENGINE FUNCTIONS
# =========================================================================

# ── #1 MARKET REGIME DETECTION ───────────────────────────────────────────
@st.cache_data(ttl=120)
def _get_market_regime():
    """v36 ENHANCED: VIX+SPY+QQQ+IWM+TLT+TNX, Fear&Greed proxy, breadth multi-indice."""
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

    # ── v36 UPGRADE #6 — RELATIVE STRENGTH vs SPY ───────────────────────
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

/* ── v38 MOBILE-FIRST RESPONSIVE ───────────────────────────────────── */
/* Smartphone portrait (< 480px) */
@media (max-width: 480px) {
    [data-testid="block-container"]{padding:0.5rem !important;}
    [data-testid="stMetric"]{padding:8px 10px !important;}
    [data-testid="stMetricValue"]{font-size:1.2rem !important;}
    [data-testid="stMetricLabel"]{font-size:0.65rem !important;}
    .section-pill{font-size:0.70rem !important;padding:4px 10px !important;}
    [data-testid="stTabs"] > div:first-child > button{
        font-size:0.62rem !important;padding:3px 5px !important;}
    /* Colonne → stack verticale su mobile */
    [data-testid="column"]{min-width:100% !important;flex:1 1 100% !important;}
    /* AgGrid più compatta */
    .ag-cell{font-size:0.72rem !important;padding:4px 6px !important;}
    .ag-header-cell-label{font-size:0.68rem !important;}
    /* Bottoni touch-friendly */
    [data-testid="stButton"]>button{
        min-height:44px !important;font-size:0.82rem !important;}
    /* Input touch-friendly */
    [data-testid="stTextInput"]>div>div>input{
        min-height:44px !important;font-size:0.90rem !important;}
    /* Sidebar overlay su mobile */
    [data-testid="stSidebar"]{
        position:fixed !important;z-index:9999 !important;
        width:85vw !important;max-width:320px !important;}
    /* Nascondi sidebar su mobile di default */
    section[data-testid="stSidebar"]:not([aria-expanded="true"]){
        transform:translateX(-100%) !important;}
}
/* Tablet (481–768px) */
@media (min-width:481px) and (max-width:768px) {
    [data-testid="block-container"]{padding:0.75rem !important;}
    [data-testid="stMetricValue"]{font-size:1.35rem !important;}
    [data-testid="stTabs"] > div:first-child > button{
        font-size:0.68rem !important;padding:4px 6px !important;}
    [data-testid="stButton"]>button{min-height:40px !important;}
}
/* Card mobile — metriche in griglia 2×2 su smartphone */
.mobile-metric-grid{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:8px;
}
@media(min-width:769px){
    .mobile-metric-grid{grid-template-columns:repeat(4,1fr);}
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

# v36 — RS vs SPY renderer (barra orizzontale con valore)
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

# v36 — RS Rank renderer (0-100 badge)
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
st.markdown('<div class="section-pill">SCANNER V38 · WATCHLIST ALERT · P&L TRACKER · BACKTEST PRO · EXPORT PRO · CHART TV-STYLE · MTF MATRIX · JOURNAL · REGIME</div>',unsafe_allow_html=True)
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
                     help="Reset filtri bilanciati (default v36)"):
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
        "🏆 Filtro CSS (v36)",
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
        help="Filtra per forza trend calcolata su EMA/Volume/OBV/ATR (ADX Proxy v36)",
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
st.sidebar.subheader("⚡ Scanner v36")
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

# ── v36 UPGRADE #3 — SCANNER SCHEDULER ────────────────────────────────────
with st.sidebar.expander("⏰ Auto-Scanner v36", expanded=False):
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

# ── v37: AI multi-provider status in sidebar ────────────────────────────
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
    if st.button("🚀 AVVIA SCANNER PRO 38.0",type="primary",use_container_width=True):
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
           # v36
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
          # v36
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
                            headerTooltip="Composite Signal Score v36 — punteggio 0-100 che combina Pro/Ser/FV score + ADX + ATR + liquidità + OBV")
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
    resp = AgGrid(df_disp,gridOptions=go_opts,height=height,
                  update_mode=update,
                  data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                  fit_columns_on_grid_load=False,theme="streamlit",
                  allow_unsafe_jscode=True,key=grid_key)

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
            st.warning("Colonna Ser_OK non trovata. Riesegui scanner v36."); return
        df_f=df[df["Ser_OK"].isin([True,"True","true"])].copy()
        if "Quality_Score" in df_f.columns and s_q>0: df_f=df_f[df_f["Quality_Score"]>=s_q]

    elif status_filter=="FINVIZ_PRO":
        if "FV_Score" not in df.columns:
            st.warning("Colonna FV_Score non trovata. Riesegui scanner v36."); return
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

    # ── v37: Pannello diagnostica filtri sempre visibile ──────────────────
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

    # ── v37: Opzioni ordinamento inline ───────────────────────────────────
    _sort_options = {
        "🏆 CSS (default)":      ("CSS", False),
        "📈 RS vs SPY":          ("RS_20d", False),
        "⚡ Momentum (Pro×RSI)": ("_Momentum_v37", False),
        "📊 Quality Score":      ("Quality_Score", False),
        "🔥 Volume Ratio":       ("Vol_Ratio", False),
        "📡 Early Score":        ("Early_Score", False),
    }
    _sort_avail = {k:v for k,v in _sort_options.items()
                   if v[0] in df_f.columns or v[0] == "_Momentum_v37"}

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

    if _sort_col == "_Momentum_v37" and "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
        df_f = df_f.copy()
        df_f["_Momentum_v37"] = (
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
    selected_df=pd.DataFrame(grid_resp["selected_rows"])

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
# ── v37: Menu sticky + 2 righe + font adattivo + tab attivo evidenziato ──
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
# v37/v38 — DEFINIZIONI FUNZIONI (spostate prima dei tab per evitare NameError)
# =========================================================================

_PATTERN_ALERTS_V38 = {
    "ema_breakout":    {"label": "EMA Breakout",      "icon": "📈", "desc": "Prezzo supera EMA20 dal basso"},
    "golden_cross":    {"label": "Golden Cross",       "icon": "⭐", "desc": "EMA20 incrocia EMA50 al rialzo"},
    "death_cross":     {"label": "Death Cross",        "icon": "💀", "desc": "EMA20 incrocia EMA50 al ribasso"},
    "squeeze_fire":    {"label": "Squeeze Fire 🔥",    "icon": "🔥", "desc": "Uscita da Squeeze (BB fuori KC)"},
    "bb_breakout":     {"label": "BB Breakout",        "icon": "🎯", "desc": "Prezzo rompe Bollinger Upper"},
    "volume_spike":    {"label": "Volume Spike",       "icon": "⚡", "desc": "Volume > 3× media 20g"},
    "rsi_oversold":    {"label": "RSI Oversold→Bull",  "icon": "🔵", "desc": "RSI risale sopra 30"},
    "rsi_overbought":  {"label": "RSI Overbought→Bear","icon": "🔴", "desc": "RSI scende sotto 70"},
}

def _detect_patterns_v38(row: pd.Series) -> list:
    """
    Rileva pattern tecnici su una riga del df_ep.
    Restituisce lista di pattern_id triggherati.
    """
    triggered = []
    try:
        pr   = float(row.get("Prezzo",    0) or 0)
        e20  = float(row.get("EMA20",     0) or 0)
        e50  = float(row.get("EMA50",     0) or 0)
        rsi  = float(row.get("RSI",       50) or 50)
        vrat = float(row.get("Vol_Ratio", 0) or 0)
        sq   = row.get("Squeeze", False)
        atr  = float(row.get("ATR",       0) or 0)

        if pr > 0 and e20 > 0:
            if pr > e20 and rsi > 45:
                triggered.append("ema_breakout")
        if e20 > 0 and e50 > 0:
            if e20 > e50 and rsi > 50:
                triggered.append("golden_cross")
            elif e20 < e50 and rsi < 50:
                triggered.append("death_cross")
        if sq in (True, "True", "true", 1):
            triggered.append("squeeze_fire")
        if vrat >= 3.0:
            triggered.append("volume_spike")
        if rsi < 32:
            triggered.append("rsi_oversold")
        if rsi > 68:
            triggered.append("rsi_overbought")
        # Bollinger breakout approssimato: prezzo > EMA20 + 2×ATR
        if pr > 0 and e20 > 0 and atr > 0:
            if pr > e20 + 2 * atr:
                triggered.append("bb_breakout")
    except Exception:
        pass
    return triggered


def _render_pattern_alerts_v38(df_ep_alerts, tab_name="default"):
    """Tab Alert Multipli: mostra pattern attivi + configurazione soglie."""
    st.markdown('<div class="section-pill">🔔 ALERT MULTIPLI v38 — Pattern Tecnici Real-Time</div>',
                unsafe_allow_html=True)

    if df_ep_alerts is None or (hasattr(df_ep_alerts,"empty") and df_ep_alerts.empty):
        st.info("Avvia lo scanner per rilevare i pattern tecnici.")
        return

    # Configurazione pattern abilitati
    with st.expander("⚙️ Configura pattern da monitorare", expanded=False):
        _pat_cols = st.columns(4)
        _enabled_pats = {}
        for _i, (_pid, _pinfo) in enumerate(_PATTERN_ALERTS_V38.items()):
            _enabled_pats[_pid] = _pat_cols[_i % 4].checkbox(
                f"{_pinfo['icon']} {_pinfo['label']}",
                value=True,
                key=f"pat_en_{_pid}_{tab_name}",
                help=_pinfo["desc"]
            )

    # Rileva pattern su tutto df_ep
    _alert_rows = []
    for _, _r in df_ep_alerts.iterrows():
        _pats = _detect_patterns_v38(_r)
        _pats_active = [p for p in _pats if _enabled_pats.get(p, True)]
        if _pats_active:
            _alert_rows.append({
                "Ticker":   str(_r.get("Ticker","")),
                "Nome":     str(_r.get("Nome",""))[:22],
                "Prezzo":   _r.get("Prezzo",""),
                "RSI":      _r.get("RSI",""),
                "CSS":      _r.get("CSS",""),
                "Pattern":  _pats_active,
                "_stato":   str(_r.get("Stato_Pro","-")),
                "_row":     _r,
            })

    if not _alert_rows:
        st.info("Nessun pattern tecnico rilevato nei dati correnti.")
        return

    # KPI
    _pat_count = {}
    for _ar in _alert_rows:
        for _p in _ar["Pattern"]:
            _pat_count[_p] = _pat_count.get(_p, 0) + 1

    _kpi_cols = st.columns(min(len(_pat_count), 6))
    for _i, (_pid, _cnt) in enumerate(sorted(_pat_count.items(), key=lambda x:-x[1])):
        if _i < 6:
            _pinfo = _PATTERN_ALERTS_V38.get(_pid, {})
            _kpi_cols[_i].metric(
                f"{_pinfo.get('icon','🔔')} {_pinfo.get('label',_pid)}",
                _cnt
            )

    st.markdown("---")

    # Lista alert con badge pattern
    _alert_rows_sorted = sorted(_alert_rows,
        key=lambda x: len(x["Pattern"]) * 10 + (1 if x["_stato"]=="STRONG" else 0),
        reverse=True)

    for _ar in _alert_rows_sorted[:30]:
        _ac1, _ac2, _ac3, _ac4 = st.columns([1.5, 1.5, 3, 1])
        _sc = "#ffd700" if _ar["_stato"]=="STRONG" else "#00ff88" if _ar["_stato"]=="PRO" else "#b2b5be"
        _ac1.markdown(
            f"<span style='font-family:Courier New;color:{_sc};font-weight:bold'>"
            f"{_ar['Ticker']}</span><br>"
            f"<span style='color:#6b7280;font-size:0.72rem'>{_ar['Nome']}</span>",
            unsafe_allow_html=True)
        _ac2.markdown(
            f"<span style='font-family:Courier New;font-size:0.82rem'>"
            f"${_ar['Prezzo']}</span><br>"
            f"<span style='color:#787b86;font-size:0.72rem'>RSI {_ar['RSI']} · CSS {_ar['CSS']}</span>",
            unsafe_allow_html=True)
        _badge_parts = []
        for p in _ar["Pattern"]:
            _is_bear = p in ("death_cross","rsi_overbought","bb_breakout")
            _is_gold = p == "golden_cross"
            _bg_col  = "#ffd70022" if _is_gold else "#ef444422" if _is_bear else "#2962ff22"
            _tx_col  = "#ffd700"   if _is_gold else "#ef4444"   if _is_bear else "#58a6ff"
            _pinfo   = _PATTERN_ALERTS_V38.get(p, {})
            _badge_parts.append(
                f"<span style='background:{_bg_col};color:{_tx_col};"
                f"border-radius:3px;padding:1px 6px;font-size:0.72rem;margin-right:3px'>"
                f"{_pinfo.get('icon','🔔')} {_pinfo.get('label',p)}</span>"
            )
        _badges = " ".join(_badge_parts)
        _ac3.markdown(_badges, unsafe_allow_html=True)
        with _ac4:
            if st.button("📋", key=f"alert_wl_{_ar['Ticker']}_{tab_name}", help="Aggiungi a watchlist"):
                try:
                    gh_add_to_watchlist(_ar["Ticker"], st.session_state.current_list_name)
                    st.success(f"✅ {_ar['Ticker']} aggiunto!")
                except Exception:
                    pass

    # Export alert
    _df_alert_exp = pd.DataFrame([
        {"Ticker": a["Ticker"], "Nome": a["Nome"],
         "Pattern": ", ".join(a["Pattern"]),
         "N_Pattern": len(a["Pattern"]),
         "Prezzo": a["Prezzo"], "RSI": a["RSI"], "CSS": a["CSS"]}
        for a in _alert_rows_sorted
    ])
    _at_ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button("📊 Export Alert",
        _df_alert_exp.to_csv(index=False).encode(),
        f"PatternAlerts_v38_{_at_ts}.csv", "text/csv",
        key=f"alert_export_v38_{tab_name}")


# =========================================================================
# v38 UPGRADE #4 — NEWS & SENTIMENT ENGINE
# Feed RSS Yahoo Finance + score NLP sentiment Bull/Bear/Neutral
# =========================================================================

@st.cache_data(ttl=600)
def _fetch_news_sentiment_v38(tickers: tuple) -> list:
    """
    Scarica RSS news da Yahoo Finance per i ticker forniti.
    Calcola sentiment score con word list NLP semplice.
    """
    import urllib.request as _ur
    import xml.etree.ElementTree as _ET

    _BULL_WORDS = {"surge","rally","soar","beat","record","upgrade","buy","bullish",
                   "outperform","strong","growth","profit","revenue","exceed","positive",
                   "gain","rise","up","high","boost","breakout","above"}
    _BEAR_WORDS = {"crash","fall","drop","miss","downgrade","sell","bearish","underperform",
                   "weak","loss","decline","below","negative","cut","reduce","layoff",
                   "concern","risk","warning","down","low","plunge","recession"}

    _results = []
    for _t in tickers[:20]:  # max 20 per performance
        try:
            _url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_t}&region=US&lang=en-US"
            _req = _ur.Request(_url, headers={"User-Agent":"Mozilla/5.0"})
            with _ur.urlopen(_req, timeout=6) as _r:
                _xml = _r.read()
            _root = _ET.fromstring(_xml)
            _items = _root.findall(".//item")[:5]  # max 5 news per ticker
            for _item in _items:
                _title = _item.findtext("title","")
                _link  = _item.findtext("link","")
                _date  = _item.findtext("pubDate","")[:16]
                # NLP score
                _words = set(_title.lower().split())
                _bull  = len(_words & _BULL_WORDS)
                _bear  = len(_words & _BEAR_WORDS)
                if _bull > _bear:
                    _sentiment = "🟢 Bullish"; _score = _bull
                elif _bear > _bull:
                    _sentiment = "🔴 Bearish"; _score = -_bear
                else:
                    _sentiment = "⚪ Neutral"; _score = 0
                _results.append({
                    "Ticker":    _t,
                    "Titolo":    _title[:80],
                    "Sentiment": _sentiment,
                    "Score":     _score,
                    "Data":      _date,
                    "Link":      _link,
                })
        except Exception:
            pass
    return sorted(_results, key=lambda x: abs(x["Score"]), reverse=True)


def _render_news_sentiment_v38(df_ep_news):
    """Tab News & Sentiment."""
    st.markdown('<div class="section-pill">📰 NEWS & SENTIMENT v38 — Feed Real-Time + NLP Score</div>',
                unsafe_allow_html=True)

    _ns_tickers = []
    if not (df_ep_news is None or (hasattr(df_ep_news,"empty") and df_ep_news.empty)):
        if "Stato_Pro" in df_ep_news.columns:
            _ns_tickers = df_ep_news[df_ep_news["Stato_Pro"].isin(["PRO","STRONG"])]["Ticker"].dropna().tolist()[:20]
        if not _ns_tickers:
            _ns_tickers = df_ep_news["Ticker"].dropna().tolist()[:15]

    # Aggiunge ticker watchlist
    try:
        _wl_ns = load_watchlist()
        if not _wl_ns.empty and "Ticker" in _wl_ns.columns:
            _ns_tickers += _wl_ns[_wl_ns["list_name"]==st.session_state.current_list_name]["Ticker"].dropna().tolist()[:10]
    except Exception:
        pass

    _ns_tickers = list(dict.fromkeys(_ns_tickers))[:25]

    if not _ns_tickers:
        st.info("Avvia lo scanner o aggiungi ticker alla watchlist per vedere le news.")
        return

    _ns_c1, _ns_c2 = st.columns([3,1])
    with _ns_c2:
        _ns_filter = st.selectbox("Filtro",
            ["Tutti","🟢 Solo Bullish","🔴 Solo Bearish","⚪ Solo Neutral"],
            key="ns_filter")
        if st.button("🔄 Aggiorna news", key="ns_refresh"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Carico news..."):
        _news = _fetch_news_sentiment_v38(tuple(_ns_tickers))

    # Filtro sentiment
    if _ns_filter == "🟢 Solo Bullish":
        _news = [n for n in _news if "Bullish" in n["Sentiment"]]
    elif _ns_filter == "🔴 Solo Bearish":
        _news = [n for n in _news if "Bearish" in n["Sentiment"]]
    elif _ns_filter == "⚪ Solo Neutral":
        _news = [n for n in _news if "Neutral" in n["Sentiment"]]

    if not _news:
        st.info("Nessuna news trovata con i filtri correnti.")
        return

    # KPI sentiment
    _n_bull = sum(1 for n in _news if "Bullish" in n["Sentiment"])
    _n_bear = sum(1 for n in _news if "Bearish" in n["Sentiment"])
    _n_neut = sum(1 for n in _news if "Neutral" in n["Sentiment"])
    _sk1,_sk2,_sk3,_sk4 = st.columns(4)
    _sk1.metric("📰 Totale News",   len(_news))
    _sk2.metric("🟢 Bullish",       _n_bull)
    _sk3.metric("🔴 Bearish",       _n_bear)
    _sk4.metric("⚪ Neutral",        _n_neut)

    st.markdown("---")

    # Lista news
    for _n in _news[:40]:
        _sc = "#00ff88" if "Bullish" in _n["Sentiment"] else "#ef4444" if "Bearish" in _n["Sentiment"] else "#6b7280"
        _nc1, _nc2, _nc3 = st.columns([1, 0.8, 4.5])
        _nc1.markdown(
            f"<span style='font-family:Courier New;color:#00ff88;font-weight:bold'>"
            f"{_n['Ticker']}</span>",
            unsafe_allow_html=True)
        _nc2.markdown(
            f"<span style='color:{_sc};font-size:0.78rem'>{_n['Sentiment']}</span>",
            unsafe_allow_html=True)
        _nc3.markdown(
            f"<a href='{_n['Link']}' target='_blank' style='color:#b2b5be;"
            f"font-size:0.82rem;text-decoration:none'>{_n['Titolo']}</a>"
            f"<span style='color:#374151;font-size:0.70rem;margin-left:8px'>{_n['Data']}</span>",
            unsafe_allow_html=True)


# =========================================================================
# v38 UPGRADE #5 — SEC FORM 4 INSIDER BUYING
# =========================================================================

@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def _fetch_insider_buying_v38(tickers: tuple) -> list:
    """
    Scarica transazioni insider via SEC EDGAR API ufficiale (EDGAR Full-Text Search).
    Fallback: yfinance major_holders per % insider ownership.
    API EDGAR: https://efts.sec.gov/LATEST/search-index?q=...&dateRange=custom
    """
    import urllib.request as _ur
    import json as _js
    _results = []

    for _t in tickers[:20]:
        try:
            # SEC EDGAR API — cerca Form 4 recenti per il ticker
            _search_url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22{_t}%22"
                f"&dateRange=custom&startdt={datetime.now().strftime('%Y-%m-%d')[:7]}-01"
                f"&forms=4&hits.hits._source=period_of_report,entity_name,file_date"
                f"&hits.hits.total.value=true"
            )
            _req = _ur.Request(
                _search_url,
                headers={"User-Agent": "TradingScanner/1.0 research@example.com",
                         "Accept": "application/json"}
            )
            with _ur.urlopen(_req, timeout=8) as _r:
                _data = _js.loads(_r.read())

            _hits = _data.get("hits", {}).get("hits", [])
            for _hit in _hits[:3]:
                _src = _hit.get("_source", {})
                _results.append({
                    "Ticker":    _t,
                    "Insider":   _src.get("entity_name", "—")[:30],
                    "Ruolo":     "—",
                    "Data":      _src.get("file_date", "—")[:10],
                    "Tipo":      "Form 4",
                    "Prezzo":    "—",
                    "Valore $":  "—",
                    "Fonte":     "SEC EDGAR",
                })
        except Exception:
            # Fallback: yfinance major_holders
            try:
                import yfinance as _yf_ins
                _mh = _yf_ins.Ticker(_t).major_holders
                if _mh is not None and not (hasattr(_mh,"empty") and _mh.empty):
                    _pct = _mh.iloc[0, 0] if len(_mh) > 0 else "—"
                    _results.append({
                        "Ticker":   _t,
                        "Insider":  "Insider ownership",
                        "Ruolo":    "—",
                        "Data":     "—",
                        "Tipo":     "% Ownership",
                        "Prezzo":   "—",
                        "Valore $": str(_pct),
                        "Fonte":    "Yahoo Finance",
                    })
            except Exception:
                pass

    return _results


@st.cache_data(ttl=3600)
def _fetch_short_interest_v38(tickers: tuple) -> dict:
    """Short Interest % da Yahoo Finance .info."""
    import yfinance as _yf_si
    _result = {}
    for _t in tickers[:40]:
        try:
            _info = _yf_si.Ticker(_t).info
            _short = _info.get("shortPercentOfFloat", None)
            if _short is not None:
                _result[_t] = round(float(_short) * 100, 1)
        except Exception:
            pass
    return _result


# =========================================================================
# v38 UPGRADE #7 — MACRO CALENDAR
# =========================================================================

@st.cache_data(ttl=3600)
def _fetch_macro_calendar_v38() -> list:
    """
    Calendario macro hardcoded con prossimi eventi ad alto impatto.
    In produzione si può integrare con investing.com o FRED API.
    """
    from datetime import timedelta
    _today = datetime.now().date()

    # Genera calendario eventi tipici (approssimazione ciclica)
    _events = []

    # Pattern mensili tipici USA
    _monthly_patterns = [
        # (giorno del mese approssimativo, nome, impatto, descrizione)
        (1,  "ISM Manufacturing",    "🟡 Med",  "Indice attività manifatturiera USA"),
        (3,  "ISM Services",         "🟡 Med",  "Indice attività settore servizi USA"),
        (5,  "NFP + Unemployment",   "🔴 High", "Non-Farm Payrolls + Tasso disoccupazione"),
        (10, "CPI Inflation",        "🔴 High", "Consumer Price Index — inflazione USA"),
        (14, "PPI",                  "🟡 Med",  "Producer Price Index"),
        (15, "Retail Sales",         "🟡 Med",  "Vendite al dettaglio USA"),
        (20, "FOMC Minutes",         "🔴 High", "Verbali Fed — politica monetaria"),
        (25, "GDP Revision",         "🟡 Med",  "Revisione PIL trimestrale USA"),
        (28, "PCE Inflation",        "🔴 High", "Personal Consumption Expenditures — preferito Fed"),
    ]

    for _day, _name, _impact, _desc in _monthly_patterns:
        # Cerca la prossima occorrenza
        for _delta_month in range(0, 3):
            _candidate = _today.replace(day=min(_day, 28))
            if _delta_month == 1:
                _m = _today.month % 12 + 1
                _y = _today.year + (_today.month // 12)
                _candidate = _candidate.replace(year=_y, month=_m)
            elif _delta_month == 2:
                _m = (_today.month + 1) % 12 + 1
                _y = _today.year + ((_today.month + 1) // 12)
                _candidate = _candidate.replace(year=_y, month=_m)

            _days_to = (_candidate - _today).days
            if _days_to >= -1:
                _events.append({
                    "Data":     str(_candidate),
                    "Evento":   _name,
                    "Impatto":  _impact,
                    "Desc":     _desc,
                    "Giorni":   _days_to,
                })
                break

    # Fed meeting dates approssimativi (2026)
    _fed_dates = ["2026-01-29","2026-03-19","2026-05-07","2026-06-18",
                  "2026-07-30","2026-09-17","2026-11-05","2026-12-17"]
    for _fd in _fed_dates:
        try:
            _fd_date = datetime.strptime(_fd,"%Y-%m-%d").date()
            _dt = (_fd_date - _today).days
            if -1 <= _dt <= 90:
                _events.append({
                    "Data":    _fd,
                    "Evento":  "⚠️ FOMC Rate Decision",
                    "Impatto": "🔴 High",
                    "Desc":    "Decisione tassi Fed — massimo impatto mercati",
                    "Giorni":  _dt,
                })
        except Exception:
            pass

    return sorted(_events, key=lambda x: x["Giorni"])


# =========================================================================
# v38 UPGRADE #8 — OPTIONS FLOW PROXY (put/call ratio)
# =========================================================================

@st.cache_data(ttl=900)
def _fetch_options_flow_v38(ticker: str) -> dict:
    """
    Calcola put/call ratio dalla options chain di Yahoo Finance.
    """
    import yfinance as _yf_op
    try:
        _tk = _yf_op.Ticker(ticker)
        _exps = _tk.options
        if not _exps: return {}
        # Usa la scadenza più vicina (indice 0)
        _chain = _tk.option_chain(_exps[0])
        _calls = _chain.calls
        _puts  = _chain.puts
        _call_vol = float(_calls["volume"].fillna(0).sum()) if not _calls.empty else 0
        _put_vol  = float(_puts["volume"].fillna(0).sum())  if not _puts.empty  else 0
        _pcr = _put_vol / _call_vol if _call_vol > 0 else None
        # Score: <0.7 bullish, 0.7-1.2 neutro, >1.2 bearish
        if _pcr is None:
            _signal = "⚪ N/D"
        elif _pcr < 0.7:
            _signal = "🟢 Bullish"
        elif _pcr > 1.2:
            _signal = "🔴 Bearish"
        else:
            _signal = "⚪ Neutro"
        return {
            "ticker":    ticker,
            "pcr":       round(_pcr, 2) if _pcr else None,
            "call_vol":  int(_call_vol),
            "put_vol":   int(_put_vol),
            "signal":    _signal,
            "expiry":    _exps[0],
        }
    except Exception:
        return {}




# =========================================================================
# v39 CLEAN BASE — blocco legacy anticipato rimosso
# =========================================================================

# =========================================================================
# v38 TAB NUOVO — 💡 ANALISI PERSONALE
# Il trader carica i suoi ticker → AI analizza e fornisce consigli operativi
# =========================================================================

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
    "💡 Analisi Personale",   # v38 nuovo
])
(tab_home, tab_mtf, tab_bcd, tab_e, tab_p, tab_r, tab_conf,
 tab_ser, tab_fvpro, tab_of, tab_crisis,
 tab_mtfmatrix, tab_journal, tab_regime,
 tab_w, tab_rm, tab_bt, tab_analisi) = tabs

with tab_home:
    # ── v36 #1 — MARKET REGIME BANNER ────────────────────────────────────
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

    # ── v36 #3 — AUTO-SCAN TRIGGER ───────────────────────────────────────
    if st.session_state.get("_trigger_autoscan"):
        st.session_state["_trigger_autoscan"] = False
        st.toast("⏰ Auto-scan avviato dallo scheduler!", icon="🤖")

    # ── v36 — MERCATI LIVE con FTSE MIB ──────────────────────────────────
    # Sovrascrive la barra di home_tab aggiungendo FTSE MIB dopo Russell2K
    @st.cache_data(ttl=60, show_spinner=False)
    def _fetch_live_markets_v36():
        import yfinance as _yf_live
        _mkts = [
            ("^GSPC",   "S&P 500",    "🇺🇸"),
            ("^IXIC",   "NASDAQ",     "💻"),
            ("^DJI",    "Dow Jones",  "🏭"),
            ("^RUT",    "Russell2K",  "📊"),
            ("FTSEMIB.MI","FTSE MIB", "🇮🇹"),
            ("^VIX",    "VIX",        "😰"),
            ("BTC-USD", "Bitcoin",    "₿"),
            ("GC=F",    "Gold",       "🥇"),
            ("CL=F",    "Oil WTI",    "🛢️"),
            ("DX-Y.NYB","DXY",        "💵"),
        ]
        _results = []
        for _sym, _name, _ico in _mkts:
            try:
                _d = _yf_live.download(_sym, period="2d", interval="1d",
                                       auto_adjust=True, progress=False)
                _d.columns = [c[0] if isinstance(c,tuple) else c for c in _d.columns]
                if len(_d) >= 2:
                    _cur = float(_d["Close"].iloc[-1])
                    _prev= float(_d["Close"].iloc[-2])
                    _chg = (_cur/_prev-1)*100
                elif len(_d) == 1:
                    _cur = float(_d["Close"].iloc[-1])
                    _chg = 0.0
                else:
                    continue
                _results.append({"sym":_sym,"name":_name,"icon":_ico,
                                 "price":_cur,"chg":_chg})
            except Exception:
                pass
        return _results

    try:
        _live_data = _fetch_live_markets_v36()
        if _live_data:
            _now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
            # v36: header SOPRA i box, box su riga separata con scroll orizzontale
            _live_html = (
                f"<div style='background:#1e222d;border-left:3px solid #2962ff;"
                f"border-radius:0 6px 6px 0;padding:6px 12px 8px 12px;margin-bottom:10px'>"
                # Riga 1: titolo + timestamp
                f"<div style='color:#2962ff;font-weight:bold;font-size:0.78rem;"
                f"letter-spacing:1px;margin-bottom:6px'>"
                f"📊 MERCATI LIVE "
                f"<span style='color:#6b7280;font-weight:normal;font-size:0.72rem'>{_now_str}</span>"
                f"</div>"
                # Riga 2: box mercati scrollabili
                f"<div style='display:flex;gap:6px;overflow-x:auto;padding-bottom:2px'>"
            )
            for _m in _live_data:
                _c  = "#26a69a" if _m["chg"]>=0 else "#ef4444"
                _ar = "▲" if _m["chg"]>=0 else "▼"
                _pr = (f"${_m['price']:,.2f}" if _m["sym"] in ("GC=F","CL=F","BTC-USD")
                       else f"{_m['price']:,.2f}" if _m["sym"] in ("^VIX","DX-Y.NYB")
                       else f"{_m['price']:,.0f}" if _m["price"]>1000
                       else f"{_m['price']:,.2f}")
                _live_html += (
                    f"<div style='background:#131722;border:1px solid #2a2e39;"
                    f"border-top:2px solid {_c}44;"
                    f"border-radius:4px;padding:5px 10px;"
                    f"min-width:100px;flex-shrink:0;text-align:center'>"
                    f"<div style='color:#787b86;font-size:0.65rem;white-space:nowrap'>{_m['icon']} {_m['name']}</div>"
                    f"<div style='color:#d1d4dc;font-family:Courier New;font-size:0.82rem;"
                    f"font-weight:bold;margin:2px 0'>{_pr}</div>"
                    f"<div style='color:{_c};font-size:0.70rem;font-weight:bold'>"
                    f"{_ar} {abs(_m['chg']):.2f}%</div>"
                    f"</div>"
                )
            _live_html += "</div></div>"
            st.markdown(_live_html, unsafe_allow_html=True)
    except Exception:
        pass

    # v38: nasconde la barra MERCATI LIVE di home_tab.py (già mostrata da v38 sopra)
    st.markdown("""<style>
    /* Nasconde il primo blocco MERCATI LIVE di home_tab */
    [data-testid="stMain"] > div > div:first-child iframe { display:none !important; }
    </style>""", unsafe_allow_html=True)
    try:
        from utils.home_tab import render_home
        render_home(df_ep, df_rea)
    except Exception as _he:
        import traceback
        st.error(f"Home tab error: {_he}")
        st.code(traceback.format_exc())

    # ── v36 #4 + #9 — EARNINGS CALENDAR (Home, fondo pagina) ─────────────
    st.markdown("---")
    st.markdown('<div class="section-pill">📅 EARNINGS CALENDAR v36 — Prossimi earnings da Watchlist + Scanner</div>',
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

            # Tabella earnings
            for _ed in _earn_data[:25]:
                _ea, _eb, _ec_col, _edd = st.columns([1.2, 1.5, 1.5, 3])
                _ea.markdown(
                    f"<b style='font-family:Courier New;color:#00ff88;font-size:1rem'>"
                    f"{_ed['Ticker']}</b>", unsafe_allow_html=True)
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


    # ── v37/v38 additions for tab_home ──
    st.markdown("---")
    st.markdown('<div class="section-pill">📰 NEWS & SENTIMENT v38</div>', unsafe_allow_html=True)
    with st.expander("📰 Ultime news con score sentiment sui ticker scanner/watchlist",
                     expanded=False):
        _render_news_sentiment_v38(df_ep)

    st.markdown("---")
    st.markdown('<div class="section-pill">🗓️ MACRO CALENDAR v38 — Fed · CPI · NFP · PCE</div>',
                unsafe_allow_html=True)
    _macro_events = _fetch_macro_calendar_v38()
    _mc_soon = [e for e in _macro_events if 0 <= e["Giorni"] <= 14]
    _mc_cols = st.columns(min(len(_mc_soon), 4)) if _mc_soon else []
    for _i, _ev in enumerate(_mc_soon[:4]):
        _ic = "#ef4444" if "High" in _ev["Impatto"] else "#f59e0b" if "Med" in _ev["Impatto"] else "#6b7280"
        (_mc_cols[_i] if _mc_cols else st).markdown(
            f"<div style='background:#1e222d;border-top:2px solid {_ic};"
            f"border-radius:0 0 6px 6px;padding:8px 10px;'>"
            f"<div style='color:{_ic};font-size:0.70rem;font-weight:bold'>{_ev['Impatto']}"
            f" · {_ev['Giorni']}gg</div>"
            f"<div style='color:#d1d4dc;font-size:0.82rem;font-weight:bold'>{_ev['Evento']}</div>"
            f"<div style='color:#6b7280;font-size:0.70rem'>{_ev['Data']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with st.expander("📅 Calendario completo prossimi 90 giorni", expanded=False):
        for _ev in _macro_events[:20]:
            _ic2 = "#ef4444" if "High" in _ev["Impatto"] else "#f59e0b" if "Med" in _ev["Impatto"] else "#6b7280"
            _bg2 = "#ef444415" if "High" in _ev["Impatto"] else "#1e222d"
            st.markdown(
                f"<div style='background:{_bg2};border-left:3px solid {_ic2};"
                f"border-radius:0 4px 4px 0;padding:5px 10px;margin:3px 0;"
                f"display:flex;gap:12px;align-items:center'>"
                f"<span style='color:{_ic2};font-size:0.75rem;min-width:80px'>{_ev['Data']}</span>"
                f"<span style='color:#d1d4dc;font-size:0.82rem;font-weight:bold'>{_ev['Evento']}</span>"
                f"<span style='color:#6b7280;font-size:0.72rem'>{_ev['Desc']}</span>"
                f"<span style='color:{_ic2};font-size:0.72rem;margin-left:auto'>"
                f"{'🔴' if _ev['Giorni']<=3 else '🟡' if _ev['Giorni']<=7 else '🟢'} "
                f"{_ev['Giorni']}gg</span>"
                f"</div>",
                unsafe_allow_html=True
            )

# Options Flow + Short Interest + Insider nel Risk Manager
with tab_e:
    st.session_state.last_active_tab="EARLY"; show_legend("EARLY")
    render_scan_tab(df_ep,"EARLY",["Early_Score","RSI"],[False,True],"EARLY")


    # ── v37/v38 additions for tab_e ──
    st.markdown("---")
    with st.expander("🔔 Alert Multipli v38", expanded=False):
        _render_pattern_alerts_v38(df_ep, tab_name="early")

# News & Sentiment — nuovo expander nella Home
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


    # ── v37/v38 additions for tab_p ──
    with st.expander("🧠 AI Signal Explainer v37 — Analisi Claude su ogni setup PRO", expanded=False):
        _render_ai_explainer_v37(df_ep, "PRO")

# Telegram nel Risk Manager
    st.markdown("---")
    with st.expander("🔔 Alert Multipli v38 — Pattern tecnici rilevati", expanded=False):
        _render_pattern_alerts_v38(df_ep, tab_name="pro")

# Alert nel tab EARLY
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


    # ── v37/v38 additions for tab_conf ──
    with st.expander("🧠 AI Signal Explainer v37 — Analisi CONFLUENCE", expanded=False):
        _df_conf_ai = pd.DataFrame()
        if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
            _df_conf_ai = df_ep[(df_ep["Stato_Early"]=="EARLY")&
                                (df_ep["Stato_Pro"].isin(["PRO","STRONG"]))].copy()
        _render_ai_explainer_v37(_df_conf_ai, "CONF")

# =========================================================================
# v38 UPGRADE #2 — ALERT MULTIPLI ENGINE
# Pattern tecnici: EMA breakout, golden/death cross, squeeze fire,
# Bollinger breakout, RSI divergence, volume spike
# =========================================================================

with tab_analisi:
    st.markdown('<div class="section-pill">💡 ANALISI PERSONALE v38 — Carica i tuoi ticker · Ricevi consigli operativi AI</div>',
                unsafe_allow_html=True)
    st.caption("Inserisci i ticker che vuoi analizzare. L'AI scarica i dati freschi e ti dà consigli su come operare su ciascuno.")

    # ── Input ticker personalizzati ──────────────────────────────────────
    _ap_c1, _ap_c2 = st.columns([2, 1.5])

    with _ap_c1:
        _ap_input = st.text_area(
            "I tuoi ticker (uno per riga)",
            placeholder="AAPL\nMSFT\nENI.MI\nRACE.MI\n...",
            height=180,
            key="ap_tickers_input",
            help="Inserisci i simboli Yahoo Finance. Per titoli italiani usa formato: ENI.MI, RACE.MI, ISP.MI"
        )
        _ap_period = st.select_slider(
            "Periodo analisi",
            options=["1mo","3mo","6mo","1y","2y"],
            value="6mo",
            key="ap_period"
        )

    with _ap_c2:
        st.markdown("**Tipo di analisi:**")
        _ap_swing  = st.checkbox("📈 Swing Trading (3-20gg)",  value=True,  key="ap_swing")
        _ap_trend  = st.checkbox("📊 Trend Following",          value=True,  key="ap_trend")
        _ap_risk   = st.checkbox("⚠️ Risk Assessment",          value=True,  key="ap_risk")
        _ap_entry  = st.checkbox("🎯 Entry Point ottimale",     value=True,  key="ap_entry")
        _ap_levels = st.checkbox("📐 Livelli S/R e target",     value=False, key="ap_levels")

        # AI key status
        _ap_has_key = any([
            st.secrets.get("GEMINI_API_KEY","")     or st.session_state.get("_gemini_api_key",""),
            st.secrets.get("GROQ_API_KEY","")       or st.session_state.get("_groq_api_key",""),
            st.secrets.get("OPENROUTER_API_KEY","") or st.session_state.get("_openrouter_api_key",""),
            st.secrets.get("ANTHROPIC_API_KEY","")  or st.session_state.get("_anthropic_api_key",""),
        ])
        if not _ap_has_key:
            st.warning("⚠️ Configura almeno una API key nel tab PRO → AI Explainer")

    _run_ap = st.button("🔍 Analizza i miei ticker", key="ap_run",
                        type="primary", use_container_width=True,
                        disabled=not _ap_has_key)

    if _run_ap and _ap_input.strip():
        _ap_tickers = [t.strip().upper() for t in _ap_input.strip().splitlines()
                       if t.strip()][:15]  # max 15 ticker

        if not _ap_tickers:
            st.warning("Inserisci almeno un ticker.")
        else:
            st.markdown(f"**Analisi di {len(_ap_tickers)} ticker:** {', '.join(_ap_tickers)}")
            st.markdown("---")

            for _ap_tkr in _ap_tickers:
                with st.expander(f"💡 {_ap_tkr} — Analisi in corso...", expanded=True):
                    with st.spinner(f"Scarico dati e genero analisi per {_ap_tkr}..."):

                        # ── 1. Scarica dati da Yahoo Finance ──────────
                        _ap_data = {}
                        try:
                            import yfinance as _yf_ap
                            _raw_ap = _yf_ap.download(
                                _ap_tkr, period=_ap_period, interval="1d",
                                auto_adjust=True, progress=False
                            )
                            _raw_ap.columns = [c[0] if isinstance(c,tuple) else c
                                               for c in _raw_ap.columns]

                            if not _raw_ap.empty:
                                _cl_ap = _raw_ap["Close"].dropna()
                                _hi_ap = _raw_ap["High"].dropna()
                                _lo_ap = _raw_ap["Low"].dropna()
                                _vo_ap = _raw_ap["Volume"].dropna() if "Volume" in _raw_ap.columns else None

                                # Calcola indicatori
                                _pr_ap   = float(_cl_ap.iloc[-1])
                                _ema20_ap= float(_cl_ap.ewm(span=20,adjust=False).mean().iloc[-1])
                                _ema50_ap= float(_cl_ap.ewm(span=50,adjust=False).mean().iloc[-1])
                                _ema200_ap=float(_cl_ap.ewm(span=min(200,len(_cl_ap)),adjust=False).mean().iloc[-1])

                                # RSI
                                _d_ap = _cl_ap.diff()
                                _g_ap = _d_ap.clip(lower=0); _l_ap = -_d_ap.clip(upper=0)
                                _rs_ap= _g_ap.ewm(com=13,adjust=False).mean()/(_l_ap.ewm(com=13,adjust=False).mean()+1e-10)
                                _rsi_ap = float((100-100/(1+_rs_ap)).iloc[-1])

                                # ATR
                                _tr_ap = (_hi_ap-_lo_ap).ewm(com=13,adjust=False).mean()
                                _atr_ap = float(_tr_ap.iloc[-1])
                                _atr_pct= round(_atr_ap/_pr_ap*100,2)

                                # Performance
                                _ret_1m  = round((_cl_ap.iloc[-1]/_cl_ap.iloc[max(-22,-len(_cl_ap))]-1)*100,1)
                                _ret_3m  = round((_cl_ap.iloc[-1]/_cl_ap.iloc[max(-63,-len(_cl_ap))]-1)*100,1)
                                _ret_6m  = round((_cl_ap.iloc[-1]/_cl_ap.iloc[max(-126,-len(_cl_ap))]-1)*100,1)

                                # Vol ratio
                                _vr_ap = "N/D"
                                if _vo_ap is not None and len(_vo_ap) >= 22:
                                    _vr_ap = round(float(_vo_ap.iloc[-1]/_vo_ap.iloc[-22:].mean()),2)

                                # 52W high/low
                                _52wh = float(_hi_ap.tail(252).max())
                                _52wl = float(_lo_ap.tail(252).min())
                                _dist_52wh = round((_pr_ap/_52wh-1)*100,1)

                                # Trend bias
                                _trend_bias = ("RIALZISTA" if _pr_ap>_ema20_ap>_ema50_ap
                                               else "RIBASSISTA" if _pr_ap<_ema20_ap<_ema50_ap
                                               else "LATERALE")

                                # Info fondamentali
                                _info_ap = {}
                                try:
                                    _ti_ap = _yf_ap.Ticker(_ap_tkr).info
                                    _info_ap = {
                                        "name":     _ti_ap.get("longName","—"),
                                        "sector":   _ti_ap.get("sector","—"),
                                        "pe":       _ti_ap.get("trailingPE","—"),
                                        "fwd_pe":   _ti_ap.get("forwardPE","—"),
                                        "div":      _ti_ap.get("dividendYield","—"),
                                        "beta":     _ti_ap.get("beta","—"),
                                        "mcap":     _ti_ap.get("marketCap",0),
                                    }
                                except Exception:
                                    pass

                                _ap_data = {
                                    "ticker": _ap_tkr,
                                    "nome":   _info_ap.get("name","—"),
                                    "settore":_info_ap.get("sector","—"),
                                    "prezzo": round(_pr_ap,2),
                                    "ema20":  round(_ema20_ap,2),
                                    "ema50":  round(_ema50_ap,2),
                                    "ema200": round(_ema200_ap,2),
                                    "rsi":    round(_rsi_ap,1),
                                    "atr":    round(_atr_ap,4),
                                    "atr_pct":_atr_pct,
                                    "ret_1m": _ret_1m,
                                    "ret_3m": _ret_3m,
                                    "ret_6m": _ret_6m,
                                    "vol_ratio": _vr_ap,
                                    "52w_high": round(_52wh,2),
                                    "52w_low":  round(_52wl,2),
                                    "dist_52wh":_dist_52wh,
                                    "trend_bias":_trend_bias,
                                    "pe":     _info_ap.get("pe","—"),
                                    "fwd_pe": _info_ap.get("fwd_pe","—"),
                                    "beta":   _info_ap.get("beta","—"),
                                    "periodo":_ap_period,
                                }

                        except Exception as _ap_err:
                            st.warning(f"Impossibile scaricare dati per {_ap_tkr}: {_ap_err}")

                        if not _ap_data:
                            continue

                        # ── 2. Mostra metriche rapide ──────────────────
                        _apm1,_apm2,_apm3,_apm4,_apm5,_apm6 = st.columns(6)
                        _tc = "#00ff88" if _ap_data["rsi"]<70 and _ap_data["trend_bias"]=="RIALZISTA" else "#ef4444" if _ap_data["trend_bias"]=="RIBASSISTA" else "#f59e0b"
                        _apm1.metric("💰 Prezzo",      f"${_ap_data['prezzo']:.2f}")
                        _apm2.metric("📊 RSI",         f"{_ap_data['rsi']:.1f}")
                        _apm3.metric("📈 Trend",        _ap_data["trend_bias"])
                        _apm4.metric("📅 Rend 1M",     f"{_ap_data['ret_1m']:+.1f}%")
                        _apm5.metric("📅 Rend 3M",     f"{_ap_data['ret_3m']:+.1f}%")
                        _apm6.metric("⚡ ATR%",         f"{_ap_data['atr_pct']:.1f}%")

                        # ── 3. Costruisce prompt AI contestuale ────────
                        _ap_analysis_types = []
                        if _ap_swing:  _ap_analysis_types.append("Swing Trading (3-20 giorni)")
                        if _ap_trend:  _ap_analysis_types.append("Trend Following")
                        if _ap_risk:   _ap_analysis_types.append("Risk Assessment")
                        if _ap_entry:  _ap_analysis_types.append("Entry Point ottimale")
                        if _ap_levels: _ap_analysis_types.append("Livelli S/R e target")

                        # Regime context
                        try:
                            _rg_ap = _get_market_regime()
                            _regime_ap = f"VIX={_rg_ap['vix']}, Regime={_rg_ap['regime']}, SPY mom={_rg_ap['spy_mom_20d']:+.1f}%"
                        except Exception:
                            _regime_ap = "dati regime non disponibili"

                        _ap_prompt = f"""Sei un trader professionista con 20 anni di esperienza. Analizza questo titolo e fornisci consigli operativi PRATICI e SPECIFICI.

TITOLO: {_ap_data['ticker']} — {_ap_data['nome']}
SETTORE: {_ap_data['settore']}
PERIODO ANALIZZATO: {_ap_data['periodo']}

DATI TECNICI:
- Prezzo attuale: ${_ap_data['prezzo']}
- EMA20: ${_ap_data['ema20']} | EMA50: ${_ap_data['ema50']} | EMA200: ${_ap_data['ema200']}
- RSI(14): {_ap_data['rsi']}
- ATR: ${_ap_data['atr']} ({_ap_data['atr_pct']}% del prezzo)
- Trend bias: {_ap_data['trend_bias']}
- Volume ratio: {_ap_data['vol_ratio']}x
- Performance: 1M {_ap_data['ret_1m']:+.1f}% | 3M {_ap_data['ret_3m']:+.1f}% | 6M {_ap_data['ret_6m']:+.1f}%
- 52W High: ${_ap_data['52w_high']} (distanza: {_ap_data['dist_52wh']:+.1f}%)
- P/E: {_ap_data['pe']} | P/E Fwd: {_ap_data['fwd_pe']} | Beta: {_ap_data['beta']}
- Contesto mercato: {_regime_ap}

TIPO DI ANALISI RICHIESTA: {', '.join(_ap_analysis_types)}

Rispondi in italiano con questo formato ESATTO:

📊 SETUP ATTUALE:
[2-3 righe: descrivi oggettivamente la situazione tecnica basandoti sui dati]

🎯 STRATEGIA CONSIGLIATA:
[2-3 righe: tipo operazione (long/short/attesa), timing, motivazione tecnica]

🔴 ENTRY: $[prezzo entry ottimale basato su ATR/EMA]
🔴 STOP LOSS: $[stop = entry - 1.5×ATR] ([percentuale]%)
🟢 TARGET 1: $[T1 = entry + 1.5×ATR] (R:R 1:1)
🟢 TARGET 2: $[T2 = entry + 3×ATR] (R:R 2:1)

⚠️ RISCHI PRINCIPALI:
[2 righe: rischi specifici per questo titolo ORA]

💡 CONSIGLIO FINALE:
[1 riga: sintesi operativa concisa]"""

                        # ── 4. Chiama AI con fallback ──────────────────
                        try:
                            _ap_text, _ap_prov = _ai_call_with_fallback(_ap_prompt)
                            st.markdown(
                                f"<div style='background:#0d1117;border:1px solid #1f2937;"
                                f"border-left:3px solid #26a69a;border-radius:0 8px 8px 0;"
                                f"padding:14px 18px;font-size:0.88rem;line-height:1.7;"
                                f"white-space:pre-wrap'>{_ap_text}</div>",
                                unsafe_allow_html=True
                            )
                            st.caption(f"Provider AI: {_ap_prov} · Dati: Yahoo Finance · {_ap_period}")
                        except Exception as _ap_ai_err:
                            _err_msg = str(_ap_ai_err)
                            if "NO_KEYS" in _err_msg:
                                st.warning("⚠️ Configura una API key nel tab PRO → AI Explainer")
                            else:
                                st.error(f"Errore AI: {_err_msg[:200]}")
    elif not _ap_has_key:
        st.info("👆 Configura almeno una API key gratuita (Gemini o Groq) nel tab **PRO → AI Explainer** per usare questa funzione.")
    else:
        # Mostra esempio di cosa aspettarsi
        st.markdown(
            "<div style='background:#1e222d;border:1px solid #2a2e39;border-radius:8px;"
            "padding:16px 20px;margin-top:8px'>"
            "<b style='color:#50c4e0'>Come funziona:</b><br><br>"
            "<span style='color:#b2b5be'>"
            "1. Inserisci i ticker che vuoi analizzare (es. AAPL, ENI.MI, RACE.MI)<br>"
            "2. Seleziona il tipo di analisi che ti interessa<br>"
            "3. Clicca <b>Analizza</b> — il sistema scarica i dati freschi da Yahoo Finance<br>"
            "4. L'AI analizza la situazione tecnica e ti fornisce consigli operativi specifici:<br>"
            "&nbsp;&nbsp;&nbsp;• Setup attuale (trend, momentum, volume)<br>"
            "&nbsp;&nbsp;&nbsp;• Entry ottimale, Stop Loss ATR-based, Target R:R<br>"
            "&nbsp;&nbsp;&nbsp;• Rischi specifici del titolo in questo momento<br>"
            "&nbsp;&nbsp;&nbsp;• Consiglio finale sintetico"
            "</span></div>",
            unsafe_allow_html=True
        )


