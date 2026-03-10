# -*- coding: utf-8 -*-
"""
ai_analyst.py  —  AI Analyst Module  v30.0
════════════════════════════════════════════════════════════════════
Usa Claude API per generare un brief analitico su ogni ticker.
Claude legge i dati tecnici già calcolati dallo scanner e produce
un'analisi strutturata in italiano: setup, target, invalidazione,
rischio, contesto macro.

REQUISITI:
  Streamlit secrets:
    [anthropic]
    api_key = "sk-ant-xxxx"

  Oppure variabile d'ambiente:
    ANTHROPIC_API_KEY = "sk-ant-xxxx"

ARCHITETTURA:
  • Chiamata diretta all'API Anthropic (no SDK — solo urllib)
  • Cache in st.session_state per evitare chiamate ripetute
  • Streaming simulato con st.write_stream per UX fluida
  • Fallback gracile se API non configurata
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


# ── Config ────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1200
CACHE_KEY_PREFIX = "_ai_analyst_"


def _get_api_key() -> Optional[str]:
    """
    Legge la chiave API Anthropic da tutte le possibili fonti.
    Gestisce tutti i formati di Streamlit secrets:
      [anthropic]           ANTHROPIC_API_KEY = "sk-ant-..."
      api_key = "sk-ant-..."

      oppure livello root:
      ANTHROPIC_API_KEY = "sk-ant-..."

    Oppure variabile d'ambiente ANTHROPIC_API_KEY.
    """
    import os

    # 1. Variabile d'ambiente (priorità massima — override esplicito)
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key and env_key.startswith("sk-"):
        return env_key

    # 2. Streamlit secrets — vari formati
    try:
        secrets = st.secrets

        # Formato A: [anthropic] / api_key = "..."
        try:
            key = str(secrets["anthropic"]["api_key"]).strip()
            if key and key.startswith("sk-"): return key
        except Exception:
            pass

        # Formato B: [anthropic] / ANTHROPIC_API_KEY = "..."
        try:
            key = str(secrets["anthropic"]["ANTHROPIC_API_KEY"]).strip()
            if key and key.startswith("sk-"): return key
        except Exception:
            pass

        # Formato C: root level — ANTHROPIC_API_KEY = "..."
        try:
            key = str(secrets["ANTHROPIC_API_KEY"]).strip()
            if key and key.startswith("sk-"): return key
        except Exception:
            pass

        # Formato D: root level — api_key = "..."
        try:
            key = str(secrets["api_key"]).strip()
            if key and key.startswith("sk-"): return key
        except Exception:
            pass

        # Formato E: root level — anthropic_api_key = "..."
        try:
            key = str(secrets["anthropic_api_key"]).strip()
            if key and key.startswith("sk-"): return key
        except Exception:
            pass

    except Exception:
        pass

    return None


def _api_available() -> bool:
    return bool(_get_api_key())


# ── Costruzione prompt ────────────────────────────────────────────

def _build_prompt(row: pd.Series) -> str:
    """
    Costruisce il prompt per Claude a partire dai dati tecnici del ticker.
    """
    tkr   = row.get("Ticker", "")
    nome  = row.get("Nome", "")
    prezzo = row.get("Prezzo", "")
    rsi   = row.get("RSI", "")
    qual  = row.get("Quality_Score", "")
    early = row.get("Early_Score", "")
    pro   = row.get("Pro_Score", "")
    vol_r = row.get("Vol_Ratio", "")
    obv   = row.get("OBV_Trend", "")
    squeeze = row.get("Squeeze", False)
    weekly  = row.get("Weekly_Bull", False)
    stato_e = row.get("Stato_Early", "")
    stato_p = row.get("Stato_Pro", "")
    mcap    = row.get("MarketCap_fmt", "")
    ser_s   = row.get("Ser_Score", "")
    fv_s    = row.get("FV_Score", "")

    # Dati grafico
    cd = row.get("_chart_data", {}) or {}
    closes = cd.get("close", [])
    ema20  = cd.get("ema20",  [])
    ema50  = cd.get("ema50",  [])
    ema200 = cd.get("ema200", [])

    def last(lst):
        vals = [v for v in (lst or []) if v is not None]
        return round(vals[-1], 2) if vals else "N/D"

    ema20_v  = last(ema20)
    ema50_v  = last(ema50)
    ema200_v = last(ema200)
    close_v  = last(closes)

    # Trend EMA
    trend_str = "N/D"
    if close_v != "N/D" and ema20_v != "N/D" and ema50_v != "N/D":
        if close_v > ema20_v > ema50_v:
            trend_str = "RIALZISTA (prezzo > EMA20 > EMA50)"
        elif close_v < ema20_v < ema50_v:
            trend_str = "RIBASSISTA (prezzo < EMA20 < EMA50)"
        else:
            trend_str = "LATERALE / MISTO"

    # Target e stop semplificati da BB
    bb_up = last(cd.get("bb_up", []))
    bb_dn = last(cd.get("bb_dn", []))

    now = datetime.now().strftime("%d %b %Y %H:%M")

    prompt = f"""Sei un analista tecnico quantitativo senior specializzato in trading azionario USA.
Analizza il seguente ticker basandoti ESCLUSIVAMENTE sui dati tecnici forniti.
Rispondi SOLO in italiano. Sii conciso, diretto e operativo. Data analisi: {now}.

═══ DATI TICKER ═══
Ticker:       {tkr}
Nome:         {nome}
Prezzo:       ${prezzo}
Market Cap:   {mcap}

═══ INDICATORI TECNICI ═══
RSI(14):        {rsi}
Vol_Ratio:      {vol_r}x  (volume oggi vs media 20g)
OBV Trend:      {obv}
Squeeze:        {"🔥 SÌ — compressione esplosiva" if squeeze else "No"}
Weekly Bull:    {"✅ Sì — setup multi-timeframe" if weekly else "No"}
Trend EMA:      {trend_str}
EMA 20:         {ema20_v}
EMA 50:         {ema50_v}
EMA 200:        {ema200_v}
BB Upper:       {bb_up}
BB Lower:       {bb_dn}

═══ SCORE SCANNER ═══
Quality Score:  {qual}/12
Early Score:    {early}
Pro Score:      {pro}
Serafini Score: {ser_s}
FinViz Score:   {fv_s}
Stato Early:    {stato_e}
Stato Pro:      {stato_p}

═══ FORMATO RISPOSTA RICHIESTO ═══
Rispondi con questa struttura esatta (usa emoji e markdown):

## 🎯 Setup
[1-2 frasi: descrivi il setup tecnico attuale — cosa sta succedendo]

## 📈 Scenario Bullish
**Target 1:** $xxx (+x%)  |  **Target 2:** $xxx (+x%)
[1 frase: condizione per conferma rialzo]

## 🛡️ Invalidazione
**Stop Loss:** $xxx (-x%)
[1 frase: livello tecnico chiave che invalida il setup]

## ⚡ Catalizzatori
[2-3 bullet con i punti tecnici più forti a favore o contro]

## ⚠️ Rischi
[1-2 frasi: rischi principali da monitorare]

## 📊 Verdict
**[STRONG BUY / BUY / WATCH / AVOID]** — [1 frase di sintesi operativa]

Sii preciso con i prezzi target. Usa i livelli BB/EMA come riferimento."""

    return prompt


# ── Chiamata API ──────────────────────────────────────────────────

def _call_claude(prompt: str, api_key: str) -> str:
    """
    Chiama Claude API via urllib (no dipendenze extra).
    Ritorna il testo della risposta.
    """
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


# ── Cache helpers ─────────────────────────────────────────────────

def _cache_key(tkr: str) -> str:
    return f"{CACHE_KEY_PREFIX}{tkr}"


def _get_cached(tkr: str) -> Optional[str]:
    return st.session_state.get(_cache_key(tkr))


def _set_cached(tkr: str, text: str):
    st.session_state[_cache_key(tkr)] = text


def clear_cache(tkr: str = None):
    """Cancella cache AI per un ticker o per tutti."""
    if tkr:
        st.session_state.pop(_cache_key(tkr), None)
    else:
        for k in list(st.session_state.keys()):
            if k.startswith(CACHE_KEY_PREFIX):
                del st.session_state[k]


# ── UI principale ─────────────────────────────────────────────────

def render_ai_analyst(row: pd.Series, key_suffix: str = ""):
    """
    Renderizza il pannello AI Analyst per un ticker.
    Chiamare dentro show_charts o in un expander.
    """
    tkr = row.get("Ticker", "N/D")
    nome = row.get("Nome", "")

    # Header stile TV
    st.markdown(
        f'<div style="background:#1e222d;border-left:3px solid #2962ff;'
        f'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:8px">'
        f'<span style="color:#2962ff;font-weight:700;font-size:0.9rem">'
        f'🤖 AI ANALYST</span>'
        f'<span style="color:#787b86;font-size:0.82rem;margin-left:12px">'
        f'Powered by Claude · {MODEL}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if not _api_available():
        # Mostra diagnostica per aiutare il setup
        st.info(
            "🔑 **API Anthropic non configurata.** Aggiungi la chiave in "
            "Streamlit Cloud → App settings → Secrets:\n\n"
            "```toml\n[anthropic]\napi_key = \"sk-ant-api03-...\"\n```\n\n"
            "Oppure a livello root:\n```toml\nANTHROPIC_API_KEY = \"sk-ant-api03-...\"\n```"
        )
        # Debug: mostra cosa trova nei secrets (senza esporre la key)
        try:
            _dbg = []
            _s = st.secrets
            if "anthropic" in _s:
                _sub = _s["anthropic"]
                _dbg.append(f"✅ sezione [anthropic] trovata — chiavi: {list(_sub.keys())}")
                for k in _sub.keys():
                    v = str(_sub[k])
                    _dbg.append(f"  · {k} = '{v[:8]}...{v[-4:]}' (len={len(v)})")
            else:
                _dbg.append("❌ sezione [anthropic] NON trovata")
                _dbg.append(f"  chiavi root: {list(_s.keys())}")
            with st.expander("🔍 Debug secrets", expanded=True):
                st.code("\n".join(_dbg))
        except Exception as _de:
            st.caption(f"Secrets non accessibili: {_de}")
        return

    cached = _get_cached(tkr)
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 4])

    with col_btn1:
        run_analysis = st.button(
            f"🧠 Analizza {tkr}",
            key=f"ai_run_{tkr}_{key_suffix}",
            type="primary",
            use_container_width=True,
            help="Chiama Claude API per analisi tecnica completa"
        )
    with col_btn2:
        if cached:
            if st.button(
                "🔄 Rigenera",
                key=f"ai_regen_{tkr}_{key_suffix}",
                use_container_width=True,
                help="Forza nuova analisi (ignora cache)"
            ):
                clear_cache(tkr)
                st.rerun()

    if run_analysis or (cached is None and False):  # auto-run: False per ora
        with st.spinner(f"🧠 Claude sta analizzando **{tkr}** — {nome}..."):
            try:
                api_key = _get_api_key()
                prompt  = _build_prompt(row)
                result  = _call_claude(prompt, api_key)
                _set_cached(tkr, result)
                cached = result
                st.success(f"✅ Analisi completata per **{tkr}**")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                st.error(f"❌ API Error {e.code}: {body[:200]}")
                return
            except Exception:
                st.error(f"❌ Errore: {traceback.format_exc()[-300:]}")
                return

    if cached:
        # Mostra l'analisi in un box styled
        st.markdown(
            f'<div style="background:#131722;border:1px solid #2a2e39;'
            f'border-radius:6px;padding:18px 22px;margin-top:6px;'
            f'font-size:0.88rem;line-height:1.6;color:#d1d4dc">'
            f'{cached.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True
        )
        # Footer con timestamp e bottone copia
        st.caption(
            f"🕐 Analisi in cache · {tkr} · "
            f"Clicca 'Rigenera' per aggiornare"
        )
    else:
        st.markdown(
            '<div style="background:#1e222d;border:1px dashed #363a45;'
            'border-radius:6px;padding:20px;text-align:center;color:#787b86">'
            '🧠 Clicca <b>Analizza</b> per generare il brief AI su questo ticker'
            '</div>',
            unsafe_allow_html=True
        )


# ── Batch analysis per watchlist ──────────────────────────────────

def render_portfolio_ai(df_watchlist: pd.DataFrame,
                        df_scanner: pd.DataFrame,
                        key_suffix: str = ""):
    """
    Analisi AI del portafoglio complessivo.
    Legge tutti i ticker in watchlist + dati scanner e produce
    un brief sul bilanciamento, momentum, e raccomandazioni.
    """
    st.markdown(
        '<div style="background:#1e222d;border-left:3px solid #ff9800;'
        'padding:10px 16px;border-radius:0 6px 6px 0;margin-bottom:8px">'
        '<span style="color:#ff9800;font-weight:700;font-size:0.9rem">'
        '💼 PORTFOLIO INTELLIGENCE</span>'
        '<span style="color:#787b86;font-size:0.82rem;margin-left:12px">'
        'Analisi bilanciamento e momentum portafoglio</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if not _api_available():
        st.info("🔑 Configura la chiave API Anthropic per usare questa funzione.")
        return

    if df_watchlist is None or df_watchlist.empty:
        st.info("📭 Watchlist vuota. Aggiungi ticker per l'analisi portafoglio.")
        return

    cache_key = "_ai_portfolio_brief_"
    cached_portfolio = st.session_state.get(cache_key)

    col1, col2 = st.columns([2, 6])
    with col1:
        run_portfolio = st.button(
            "💼 Analizza Portafoglio",
            key=f"ai_port_{key_suffix}",
            type="primary",
            use_container_width=True
        )
    with col2:
        if cached_portfolio:
            if st.button("🔄 Rigenera", key=f"ai_port_regen_{key_suffix}"):
                st.session_state.pop(cache_key, None)
                st.rerun()

    if run_portfolio:
        with st.spinner("💼 Claude sta analizzando il tuo portafoglio..."):
            try:
                api_key = _get_api_key()
                prompt = _build_portfolio_prompt(df_watchlist, df_scanner)
                result = _call_claude(prompt, api_key)
                st.session_state[cache_key] = result
                cached_portfolio = result
                st.success("✅ Analisi portafoglio completata")
            except Exception:
                st.error(f"❌ Errore: {traceback.format_exc()[-300:]}")
                return

    if cached_portfolio:
        st.markdown(
            f'<div style="background:#131722;border:1px solid #2a2e39;'
            f'border-radius:6px;padding:18px 22px;margin-top:6px;'
            f'font-size:0.88rem;line-height:1.6;color:#d1d4dc">'
            f'{cached_portfolio.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        tickers = df_watchlist["Ticker"].tolist() if "Ticker" in df_watchlist.columns else []
        st.markdown(
            f'<div style="background:#1e222d;border:1px dashed #363a45;'
            f'border-radius:6px;padding:20px;text-align:center;color:#787b86">'
            f'💼 {len(tickers)} ticker in watchlist · '
            f'Clicca <b>Analizza Portafoglio</b> per il brief AI'
            f'</div>',
            unsafe_allow_html=True
        )


def _build_portfolio_prompt(df_wl: pd.DataFrame,
                             df_scan: pd.DataFrame) -> str:
    """Costruisce il prompt per l'analisi portafoglio completa."""
    tickers = df_wl["Ticker"].tolist() if "Ticker" in df_wl.columns else []
    liste   = df_wl["list_name"].value_counts().to_dict() if "list_name" in df_wl.columns else {}
    now = datetime.now().strftime("%d %b %Y %H:%M")

    # Merge con dati scanner
    rows_data = []
    if df_scan is not None and not df_scan.empty and "Ticker" in df_scan.columns:
        merged = df_wl.merge(
            df_scan[["Ticker","RSI","Quality_Score","Vol_Ratio",
                      "OBV_Trend","Squeeze","Weekly_Bull",
                      "Stato_Early","Early_Score","Pro_Score"]
                     if all(c in df_scan.columns for c in ["RSI","Quality_Score"])
                     else ["Ticker"]],
            on="Ticker", how="left"
        )
        for _, r in merged.iterrows():
            rows_data.append(
                f"  {r.get('Ticker','?'):8s} | lista={r.get('list_name','?'):12s} | "
                f"RSI={r.get('RSI','?')!s:5s} | Q={r.get('Quality_Score','?')!s:4s} | "
                f"Vol×={r.get('Vol_Ratio','?')!s:4s} | "
                f"Sqz={'🔥' if r.get('Squeeze') else 'No':3s} | "
                f"W+={'✅' if r.get('Weekly_Bull') else 'No':3s} | "
                f"OBV={r.get('OBV_Trend','?')!s:6s} | "
                f"Stato={r.get('Stato_Early','?')}"
            )
    else:
        rows_data = [f"  {t}" for t in tickers]

    ticker_table = "\n".join(rows_data) if rows_data else "  Nessun dato disponibile"

    prompt = f"""Sei un portfolio manager quantitativo senior. Analizza questo portafoglio azionario
e fornisci raccomandazioni operative concrete. Data analisi: {now}.

═══ COMPOSIZIONE PORTAFOGLIO ═══
Ticker totali: {len(tickers)}
Liste: {json.dumps(liste, ensure_ascii=False)}

═══ DATI TECNICI PER TICKER ═══
Ticker   | Lista        | RSI   | Q    | Vol× | Sqz | W+  | OBV    | Stato
{ticker_table}

═══ ISTRUZIONI ═══
Analizza il portafoglio e rispondi con questa struttura:

## 📊 Panoramica Portafoglio
[2-3 frasi: stato generale del portafoglio, momentum medio, concentrazione]

## 🔥 Top Pick Momentum
[Top 3 ticker con il miglior setup tecnico attuale — spiega perché]

## ⚠️ Posizioni a Rischio
[Ticker con segnali deboli o deterioramento — considera riduzione/uscita]

## ⚖️ Bilanciamento
[Analisi della distribuzione: settori, liquidità, correlazioni evidenti]

## 🎯 Azioni Raccomandate
Lista di 3-5 azioni concrete:
1. [Azione specifica su ticker specifico]
2. ...

## 💡 Opportunità da Aggiungere
[2-3 tipologie di asset o settori che migliorerebbero il bilanciamento]

## 📈 Outlook
**[BULLISH / NEUTRALE / CAUTO]** — [1 frase di sintesi sul portafoglio complessivo]

Sii specifico con i ticker. Usa i dati tecnici forniti come base principale."""

    return prompt
