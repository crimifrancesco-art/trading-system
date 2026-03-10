# -*- coding: utf-8 -*-
"""
ai_analyst.py  —  AI Analyst Module  v30.0  (Google Gemini)
════════════════════════════════════════════════════════════════════
Usa Google Gemini API (gratuita) per brief analitici sui ticker.

SETUP:
  1. aistudio.google.com → Get API Key (gratis, nessuna carta)
  2. Streamlit Cloud → App settings → Secrets:

     [gemini]
     api_key = "AIzaSy..."

MODELLI (tutti gratuiti, 15 req/min):
  gemini-1.5-flash      → default, ottimo bilanciamento
  gemini-1.5-flash-8b   → più veloce
  gemini-2.0-flash-exp  → sperimentale
════════════════════════════════════════════════════════════════════
"""

import json
import urllib.request
import urllib.error
import traceback
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st


# ── Costanti ──────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-1.5-flash-latest"   # free tier, funzionante
GEMINI_MODEL_FB = "gemini-1.5-pro-latest"      # fallback qualità
GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/{model}:generateContent?key={key}"
)
MAX_TOKENS       = 1200
CACHE_KEY_PREFIX = "_ai_analyst_"


# ── Lettura API Key ───────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    """Legge Gemini API key da secrets o env."""
    import os

    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env and env.startswith("AIza"):
        return env

    try:
        s = st.secrets
        for path in (
            ("gemini", "api_key"),
            ("gemini", "GEMINI_API_KEY"),
        ):
            try:
                k = s
                for p in path:
                    k = k[p]
                k = str(k).strip()
                if k.startswith("AIza"):
                    return k
            except Exception:
                pass
        for nome in ("GEMINI_API_KEY", "gemini_api_key", "api_key"):
            try:
                k = str(s[nome]).strip()
                if k.startswith("AIza"):
                    return k
            except Exception:
                pass
    except Exception:
        pass
    return None


def _api_available() -> bool:
    return bool(_get_api_key())


# ── Chiamata REST Gemini ──────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str, model: str = None) -> str:
    """
    Chiama Gemini API, prova modelli in sequenza.
    Su 429 aspetta con backoff prima di riprovare.
    """
    import time
    models = [model or GEMINI_MODEL, "gemini-1.5-flash-latest",
              "gemini-1.5-pro-latest"]
    seen = set()
    models = [m for m in models if not (m in seen or seen.add(m))]

    last_err = None
    for attempt, m in enumerate(models):
        url = GEMINI_BASE_URL.format(model=m, key=api_key.strip())
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": MAX_TOKENS,
                "temperature": 0.4,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {e.code} ({m}): {body[:300]}"
            if e.code in (400, 403):
                try:
                    msg = json.loads(body).get("error", {}).get("message", body[:200])
                except Exception:
                    msg = body[:200]
                raise RuntimeError(f"❌ Gemini Error {e.code}: {msg}")
            if e.code == 429:
                # Rate limit free tier — backoff crescente tra tentativi
                wait = 10 + attempt * 8   # 10s, 18s, 26s
                time.sleep(wait)
                continue
            if e.code == 404:
                continue
            raise RuntimeError(last_err)
        except RuntimeError:
            raise
        except Exception as ex:
            last_err = str(ex)
            continue

    raise RuntimeError(
        "⏳ **Rate limit Gemini (429)** — tutti i modelli occupati.\n\n"
        "Il piano gratuito permette ~15 richieste/minuto. "
        "Aspetta 60 secondi e riprova.\n\n"
        f"Ultimo errore: {last_err}"
    )


# ── Prompt ────────────────────────────────────────────────────────

def _build_prompt(row: pd.Series) -> str:
    def g(k, d=""): return row.get(k, d)
    def last(lst):
        vals = [v for v in (lst or []) if v is not None]
        return round(vals[-1], 2) if vals else "N/D"

    cd   = g("_chart_data", {}) or {}
    e20  = last(cd.get("ema20",  []))
    e50  = last(cd.get("ema50",  []))
    e200 = last(cd.get("ema200", []))
    cl   = last(cd.get("close",  []))
    bbu  = last(cd.get("bb_up",  []))
    bbd  = last(cd.get("bb_dn",  []))

    if cl != "N/D" and e20 != "N/D" and e50 != "N/D":
        if cl > e20 > e50:    trend = "RIALZISTA (prezzo > EMA20 > EMA50)"
        elif cl < e20 < e50:  trend = "RIBASSISTA (prezzo < EMA20 < EMA50)"
        else:                  trend = "LATERALE / MISTO"
    else:
        trend = "N/D"

    return f"""Sei un analista tecnico quantitativo senior. Analizza questo ticker
usando SOLO i dati tecnici forniti. Rispondi ESCLUSIVAMENTE in italiano.
Sii conciso, preciso e operativo. Data: {datetime.now().strftime("%d %b %Y %H:%M")}.

TICKER: {g("Ticker")} | NOME: {g("Nome")} | PREZZO: ${g("Prezzo")} | MCAP: {g("MarketCap_fmt")}

TECNICI:
RSI(14)={g("RSI")} | Vol_Ratio={g("Vol_Ratio")}x | OBV={g("OBV_Trend")}
Squeeze={"🔥 SÌ" if g("Squeeze") else "No"} | Weekly Bull={"✅ Sì" if g("Weekly_Bull") else "No"}
Trend EMA: {trend}
EMA20={e20} | EMA50={e50} | EMA200={e200}
BB Upper={bbu} | BB Lower={bbd}

SCORE:
Quality={g("Quality_Score")}/12 | Early={g("Early_Score")} | Pro={g("Pro_Score")}
Serafini={g("Ser_Score")} | FinViz={g("FV_Score")}
Stato Early={g("Stato_Early")} | Stato Pro={g("Stato_Pro")}

FORMATO RISPOSTA (rispetta questa struttura):

## 🎯 Setup
[1-2 frasi: setup tecnico attuale]

## 📈 Scenario Bullish
**Target 1:** $xxx (+x%) | **Target 2:** $xxx (+x%)
[1 frase: condizione per conferma rialzo]

## 🛡️ Invalidazione
**Stop Loss:** $xxx (-x%)
[1 frase: livello tecnico che invalida il setup]

## ⚡ Catalizzatori
- [punto tecnico 1]
- [punto tecnico 2]
- [punto tecnico 3]

## ⚠️ Rischi
[1-2 frasi: rischi principali]

## 📊 Verdict
**[STRONG BUY / BUY / WATCH / AVOID]** — [1 frase operativa]"""


def _build_portfolio_prompt(df_wl: pd.DataFrame,
                             df_scan: pd.DataFrame) -> str:
    tickers = df_wl.get("Ticker", pd.Series()).tolist()
    liste   = (df_wl["list_name"].value_counts().to_dict()
               if "list_name" in df_wl.columns else {})

    scan_cols = ["Ticker","RSI","Quality_Score","Vol_Ratio",
                 "OBV_Trend","Squeeze","Weekly_Bull","Stato_Early"]
    rows_data = []
    if df_scan is not None and not df_scan.empty and "Ticker" in df_scan.columns:
        avail  = [c for c in scan_cols if c in df_scan.columns]
        merged = df_wl.merge(df_scan[avail], on="Ticker", how="left")
        for _, r in merged.iterrows():
            rows_data.append(
                f"  {str(r.get('Ticker','?')):8s} | "
                f"lista={str(r.get('list_name','?')):10s} | "
                f"RSI={str(r.get('RSI','?')):5s} | "
                f"Q={str(r.get('Quality_Score','?')):4s} | "
                f"Sqz={'🔥' if r.get('Squeeze') else 'No'} | "
                f"W+={'✅' if r.get('Weekly_Bull') else 'No'} | "
                f"OBV={str(r.get('OBV_Trend','?'))}"
            )
    else:
        rows_data = [f"  {t}" for t in tickers]

    return f"""Sei un portfolio manager quantitativo senior. Analizza questo portafoglio.
Data: {datetime.now().strftime("%d %b %Y %H:%M")}. Rispondi in italiano.

PORTAFOGLIO: {len(tickers)} ticker | Liste: {json.dumps(liste, ensure_ascii=False)}

DATI TECNICI:
{"".join(chr(10)+r for r in rows_data)}

Rispondi con questa struttura:

## 📊 Panoramica
[2-3 frasi: stato generale e momentum medio]

## 🔥 Top 3 Momentum
[I 3 ticker con setup migliore]

## ⚠️ Posizioni Deboli
[Ticker da ridurre o uscire]

## ⚖️ Bilanciamento
[Concentrazione, correlazioni, rischi]

## 🎯 Azioni Raccomandate
1. [azione specifica]
2. [azione specifica]
3. [azione specifica]

## 📈 Outlook
**[BULLISH / NEUTRALE / CAUTO]** — [sintesi 1 frase]"""


# ── Cache ─────────────────────────────────────────────────────────

def _cache_key(tkr): return f"{CACHE_KEY_PREFIX}{tkr}"
def _get_cached(tkr): return st.session_state.get(_cache_key(tkr))
def _set_cached(tkr, text): st.session_state[_cache_key(tkr)] = text

def clear_cache(tkr: str = None):
    if tkr:
        st.session_state.pop(_cache_key(tkr), None)
    else:
        for k in list(st.session_state.keys()):
            if k.startswith(CACHE_KEY_PREFIX):
                del st.session_state[k]


# ── UI ticker ─────────────────────────────────────────────────────

def render_ai_analyst(row: pd.Series, key_suffix: str = ""):
    tkr  = row.get("Ticker", "N/D")
    nome = row.get("Nome", "")

    st.markdown(
        f'<div style="background:#1e222d;border-left:3px solid #2962ff;'
        f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:8px">'
        f'<span style="color:#2962ff;font-weight:700;font-size:0.9rem">'
        f'🤖 AI ANALYST</span>'
        f'<span style="color:#787b86;font-size:0.82rem;margin-left:12px">'
        f'Powered by Google Gemini · gratuito</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if not _api_available():
        st.info(
            "🔑 **Gemini API non configurata.**\n\n"
            "1. Vai su **aistudio.google.com** → crea API Key (gratis)\n"
            "2. In Streamlit Cloud → App settings → Secrets:\n\n"
            "```toml\n[gemini]\napi_key = \"AIzaSy...\"\n```"
        )
        try:
            dbg = []
            s = st.secrets
            if "gemini" in s:
                sub = s["gemini"]
                dbg.append(f"✅ sezione [gemini] trovata — chiavi: {list(sub.keys())}")
                for k in sub.keys():
                    v = str(sub[k])
                    dbg.append(f"  · {k} = '{v[:8]}...{v[-4:]}' (len={len(v)})")
            else:
                dbg.append("❌ sezione [gemini] NON trovata")
                dbg.append(f"  chiavi root: {list(s.keys())}")
            with st.expander("🔍 Debug secrets", expanded=True):
                st.code("\n".join(dbg))
        except Exception as de:
            st.caption(f"Secrets non accessibili: {de}")
        return

    cached = _get_cached(tkr)
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        run_btn = st.button(
            f"🧠 Analizza {tkr}",
            key=f"ai_run_{tkr}_{key_suffix}",
            type="primary",
            use_container_width=True,
        )
    with c2:
        regen_btn = False
        if cached:
            regen_btn = st.button(
                "🔄 Rigenera",
                key=f"ai_regen_{tkr}_{key_suffix}",
                use_container_width=True,
            )
    with c3:
        model_sel = st.selectbox(
            "Modello",
            ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"],
            index=0,
            key=f"ai_model_{tkr}_{key_suffix}",
            label_visibility="collapsed",
            help="flash = veloce (15 RPM) | pro = qualità massima",
        )

    if regen_btn:
        clear_cache(tkr)
        st.rerun()

    if run_btn:
        with st.spinner(f"🧠 Gemini analizza **{tkr}** — {nome}..."):
            try:
                result = _call_gemini(_build_prompt(row), _get_api_key(), model_sel)
                _set_cached(tkr, result)
                cached = result
                st.success(f"✅ Analisi completata — {tkr}")
            except RuntimeError as e:
                st.error(str(e))
                return
            except Exception:
                st.error(f"❌ Errore:\n```\n{traceback.format_exc()[-400:]}\n```")
                return

    if cached:
        st.markdown(
            f'<div style="background:#131722;border:1px solid #2a2e39;'
            f'border-radius:6px;padding:18px 22px;margin-top:6px;'
            f'font-size:0.88rem;line-height:1.7;color:#d1d4dc">'
            f'{cached.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption(f"🕐 Cache · {tkr} · {model_sel} · Premi Rigenera per aggiornare")
    else:
        st.markdown(
            '<div style="background:#1e222d;border:1px dashed #363a45;'
            'border-radius:6px;padding:20px;text-align:center;color:#787b86">'
            '🧠 Premi <b>Analizza</b> per il brief AI su questo ticker'
            '</div>',
            unsafe_allow_html=True
        )


# ── UI portafoglio ────────────────────────────────────────────────

def render_portfolio_ai(df_wl: pd.DataFrame,
                        df_scan: pd.DataFrame,
                        key_suffix: str = ""):
    st.markdown(
        '<div style="background:#1e222d;border-left:3px solid #ff9800;'
        'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:8px">'
        '<span style="color:#ff9800;font-weight:700;font-size:0.9rem">'
        '💼 PORTFOLIO INTELLIGENCE</span>'
        '<span style="color:#787b86;font-size:0.82rem;margin-left:12px">'
        'Analisi AI bilanciamento e momentum portafoglio</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if not _api_available():
        st.info("🔑 Configura la Gemini API key per questa funzione.")
        return
    if df_wl is None or df_wl.empty:
        st.info("📭 Watchlist vuota.")
        return

    cache_key = "_ai_portfolio_brief_"
    cached_p  = st.session_state.get(cache_key)
    c1, c2    = st.columns([2, 6])

    with c1:
        run_port = st.button(
            "💼 Analizza Portafoglio",
            key=f"ai_port_{key_suffix}",
            type="primary",
            use_container_width=True,
        )
    with c2:
        if cached_p:
            if st.button("🔄 Rigenera", key=f"ai_port_regen_{key_suffix}"):
                st.session_state.pop(cache_key, None)
                st.rerun()

    if run_port:
        with st.spinner("💼 Gemini analizza il portafoglio..."):
            try:
                result = _call_gemini(
                    _build_portfolio_prompt(df_wl, df_scan), _get_api_key()
                )
                st.session_state[cache_key] = result
                cached_p = result
                st.success("✅ Analisi portafoglio completata")
            except RuntimeError as e:
                st.error(str(e))
                return
            except Exception:
                st.error(f"❌ {traceback.format_exc()[-300:]}")
                return

    if cached_p:
        st.markdown(
            f'<div style="background:#131722;border:1px solid #2a2e39;'
            f'border-radius:6px;padding:18px 22px;font-size:0.88rem;'
            f'line-height:1.7;color:#d1d4dc">'
            f'{cached_p.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        n = df_wl["Ticker"].nunique() if "Ticker" in df_wl.columns else 0
        st.markdown(
            f'<div style="background:#1e222d;border:1px dashed #363a45;'
            f'border-radius:6px;padding:20px;text-align:center;color:#787b86">'
            f'💼 {n} ticker · Premi <b>Analizza Portafoglio</b> per il brief AI'
            f'</div>',
            unsafe_allow_html=True
        )
