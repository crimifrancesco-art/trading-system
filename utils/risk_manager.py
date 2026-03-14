# -*- coding: utf-8 -*-
"""
risk_manager.py  — v32.0
========================
Tab "⚖️ Risk Manager" per Trading Scanner PRO.

Funzionalita':
  1. Position Sizing  — quante azioni comprare dato il rischio massimo %
  2. ATR-based Stop   — stop loss / target calcolati sull'ATR del titolo
  3. R:R Calculator   — calcolo Risk:Reward automatico per ogni trade
  4. Trade Plan Grid  — tabella riassuntiva esportabile

Dipendenze: streamlit, pandas, numpy, plotly, urllib (stdlib)
Nessuna libreria esterna aggiuntiva rispetto al progetto.
"""

import json
import urllib.request
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Palette TV ───────────────────────────────────────────────────────────────
_BG     = "#131722"; _PANEL  = "#1e222d"; _BORDER = "#2a2e39"
_GREEN  = "#26a69a"; _RED    = "#ef5350"; _GOLD   = "#ffd700"
_BLUE   = "#2962ff"; _CYAN   = "#50c4e0"; _GRAY   = "#787b86"
_TEXT   = "#d1d4dc"; _ORANGE = "#ff9800"

# ── Fetch ATR live da Yahoo ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_atr(symbol: str, period: int = 14) -> dict:
    """
    Scarica ultimi 60gg daily OHLCV e calcola:
      - Prezzo corrente (close)
      - ATR(14) giornaliero
      - ATR% = ATR/Prezzo * 100
    """
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range=60d")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        q      = result["indicators"]["quote"][0]
        meta   = result["meta"]
        name   = meta.get("longName") or meta.get("shortName") or symbol
        closes = pd.Series(q.get("close", [])).dropna()
        highs  = pd.Series(q.get("high",  [])).dropna()
        lows   = pd.Series(q.get("low",   [])).dropna()
        if len(closes) < period + 2:
            return {"ok": False, "err": "Dati insufficienti"}
        # True Range
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows  - closes.shift(1)).abs(),
        ], axis=1).max(axis=1).dropna()
        atr   = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        price = float(closes.iloc[-1])
        return {
            "ok":    True,
            "name":  name,
            "price": round(price, 4),
            "atr":   round(atr, 4),
            "atr_pct": round(atr / price * 100, 2) if price > 0 else 0,
        }
    except Exception as e:
        return {"ok": False, "err": str(e)}

# ── Calcolo trade plan ───────────────────────────────────────────────────────
def calc_trade_plan(
    ticker:       str,
    price:        float,
    atr:          float,
    capital:      float,
    risk_pct:     float,
    atr_stop_mult:float = 1.5,
    direction:    str   = "LONG",
) -> dict:
    """
    Calcola il piano completo per un singolo trade.

    Returns dict con:
      stop_loss, target1, target2, target3,
      risk_per_share, shares, position_value, risk_dollars,
      rr_t1, rr_t2, rr_t3
    """
    risk_dollars = capital * (risk_pct / 100)

    if direction == "LONG":
        stop_loss = round(price - atr_stop_mult * atr, 4)
        target1   = round(price + 1.0 * atr_stop_mult * atr, 4)  # R:R 1:1
        target2   = round(price + 2.0 * atr_stop_mult * atr, 4)  # R:R 2:1
        target3   = round(price + 3.0 * atr_stop_mult * atr, 4)  # R:R 3:1
    else:  # SHORT
        stop_loss = round(price + atr_stop_mult * atr, 4)
        target1   = round(price - 1.0 * atr_stop_mult * atr, 4)
        target2   = round(price - 2.0 * atr_stop_mult * atr, 4)
        target3   = round(price - 3.0 * atr_stop_mult * atr, 4)

    risk_per_share  = abs(price - stop_loss)
    shares          = int(risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    position_value  = round(shares * price, 2)
    actual_risk     = round(shares * risk_per_share, 2)

    rr_t1 = round(abs(target1 - price) / risk_per_share, 2) if risk_per_share > 0 else 0
    rr_t2 = round(abs(target2 - price) / risk_per_share, 2) if risk_per_share > 0 else 0
    rr_t3 = round(abs(target3 - price) / risk_per_share, 2) if risk_per_share > 0 else 0

    pct_stop  = round((stop_loss - price) / price * 100, 2)
    pct_t1    = round((target1   - price) / price * 100, 2)
    pct_t2    = round((target2   - price) / price * 100, 2)
    pct_t3    = round((target3   - price) / price * 100, 2)

    return {
        "ticker":         ticker,
        "direction":      direction,
        "price":          price,
        "atr":            atr,
        "stop_loss":      stop_loss,
        "target1":        target1,
        "target2":        target2,
        "target3":        target3,
        "risk_per_share": round(risk_per_share, 4),
        "shares":         shares,
        "position_value": position_value,
        "risk_dollars":   actual_risk,
        "pct_capital":    round(position_value / capital * 100, 1) if capital > 0 else 0,
        "rr_t1":          rr_t1,
        "rr_t2":          rr_t2,
        "rr_t3":          rr_t3,
        "pct_stop":       pct_stop,
        "pct_t1":         pct_t1,
        "pct_t2":         pct_t2,
        "pct_t3":         pct_t3,
    }

# ── Chart trade plan ─────────────────────────────────────────────────────────
def _render_trade_chart(plan: dict) -> None:
    """
    Grafico orizzontale con entry, stop, target1/2/3.
    """
    price  = plan["price"]
    sl     = plan["stop_loss"]
    t1     = plan["target1"]
    t2     = plan["target2"]
    t3     = plan["target3"]
    ticker = plan["ticker"]
    is_long = plan["direction"] == "LONG"

    fig = go.Figure()

    # Bande visive
    fig.add_hrect(y0=sl, y1=price,
        fillcolor="rgba(239,83,80,0.12)", line_width=0,
        annotation_text="  RISCHIO", annotation_position="right",
        annotation_font=dict(color=_RED, size=10))
    fig.add_hrect(y0=price, y1=t1,
        fillcolor="rgba(255,152,0,0.10)", line_width=0)
    fig.add_hrect(y0=t1, y1=t2,
        fillcolor="rgba(38,166,154,0.10)", line_width=0)
    fig.add_hrect(y0=t2, y1=t3,
        fillcolor="rgba(41,98,255,0.10)", line_width=0)

    # Linee livelli
    for y, color, label, dash in [
        (sl,    _RED,    f"STOP LOSS  ${sl:.2f}  ({plan['pct_stop']:+.1f}%)",  "dash"),
        (price, _GOLD,   f"ENTRY  ${price:.2f}",                                "solid"),
        (t1,    _ORANGE, f"TARGET 1  ${t1:.2f}  (R:R {plan['rr_t1']:.1f}:1)", "dot"),
        (t2,    _GREEN,  f"TARGET 2  ${t2:.2f}  (R:R {plan['rr_t2']:.1f}:1)", "dot"),
        (t3,    _BLUE,   f"TARGET 3  ${t3:.2f}  (R:R {plan['rr_t3']:.1f}:1)", "dot"),
    ]:
        fig.add_hline(y=y,
            line=dict(color=color, width=1.8, dash=dash),
            annotation_text=f"  {label}",
            annotation_position="right",
            annotation_font=dict(color=color, size=10))

    # Range asse y con margine
    all_prices = [sl, price, t1, t2, t3]
    y_min = min(all_prices) * 0.98
    y_max = max(all_prices) * 1.02

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        height=380,
        margin=dict(l=0, r=220, t=50, b=0),
        font=dict(color=_TEXT, family="Courier New, monospace", size=10),
        title=dict(
            text=f"<b style='color:{_CYAN}'>{ticker}</b>  "
                 f"{'🟢 LONG' if is_long else '🔴 SHORT'}  "
                 f"Piano trade ATR×{plan['atr']:.2f}",
            font=dict(size=13, color=_TEXT), x=0.01),
        yaxis=dict(
            range=[y_min, y_max],
            showgrid=True, gridcolor=_BORDER,
            tickprefix="$", tickfont=dict(size=10)),
        xaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"rm_chart_{ticker}")

# ── Render principale ────────────────────────────────────────────────────────
def render_risk_manager(df_scanner: pd.DataFrame = None) -> None:
    """
    Entry point pubblico. Chiamare con:
        from utils.risk_manager import render_risk_manager
        render_risk_manager(df_scanner=df_ep)
    """
    st.markdown(
        '<div class="section-pill">⚖️ RISK MANAGER — Position Sizing · ATR Stops · R:R</div>',
        unsafe_allow_html=True)

    st.markdown("""
> **Come funziona:** inserisci il tuo capitale e il rischio massimo per trade (%).
> Il sistema calcola automaticamente quante azioni acquistare, dove mettere
> lo stop loss (basato sull'ATR) e quali target puntare (R:R 1:1, 2:1, 3:1).
""")

    # ── Parametri portafoglio ─────────────────────────────────────────────
    st.markdown("### ⚙️ Parametri Portafoglio")
    pc1, pc2, pc3, pc4 = st.columns(4)

    with pc1:
        capital = st.number_input(
            "Capitale ($)",
            min_value=1_000.0, max_value=10_000_000.0,
            value=float(st.session_state.get("rm_capital", 50_000.0)),
            step=1_000.0, format="%.0f",
            help="Capitale totale disponibile per il trading",
            key="rm_capital_input",
        )
        st.session_state["rm_capital"] = capital

    with pc2:
        risk_pct = st.slider(
            "Rischio per trade (%)",
            min_value=0.25, max_value=5.0,
            value=float(st.session_state.get("rm_risk_pct", 1.0)),
            step=0.25,
            help="% del capitale da rischiare per ogni trade. "
                 "1-2% = standard professionale. > 3% = aggressivo.",
            key="rm_risk_pct_input",
        )
        st.session_state["rm_risk_pct"] = risk_pct
        _risk_dollars = capital * risk_pct / 100
        st.caption(f"Rischio $: **${_risk_dollars:,.0f}** per trade")

    with pc3:
        atr_mult = st.slider(
            "ATR multiplier (stop)",
            min_value=0.5, max_value=4.0,
            value=float(st.session_state.get("rm_atr_mult", 1.5)),
            step=0.25,
            help="Stop Loss = Entry ± ATR×moltiplicatore. "
                 "1.5 = standard. 2.0 = ampio. 1.0 = stretto (rischio fakeout).",
            key="rm_atr_mult_input",
        )
        st.session_state["rm_atr_mult"] = atr_mult

    with pc4:
        direction = st.radio(
            "Direzione",
            ["LONG", "SHORT"],
            horizontal=True,
            key="rm_direction",
        )

    st.markdown("---")


    # ── Selezione ticker ──────────────────────────────────────────────────
    st.markdown("### 🎯 Selezione Ticker")

    # Costruisce lista (label, ticker_raw) ordinata A→Z per Nome
    scanner_rows = []
    if df_scanner is not None and not df_scanner.empty and "Ticker" in df_scanner.columns:
        _df_s = df_scanner.dropna(subset=["Ticker"]).copy()
        _sort_col = "Nome" if "Nome" in _df_s.columns else "Ticker"
        _df_s = _df_s.sort_values(_sort_col, key=lambda s: s.str.lower())
        for _, row in _df_s.iterrows():
            t = str(row["Ticker"]).strip()
            if not t:
                continue
            if "Nome" in _df_s.columns and pd.notna(row.get("Nome")):
                nome = str(row["Nome"])[:32].strip()
                _icons = ""
                if row.get("Stato_Pro") == "STRONG": _icons += " ★"
                elif row.get("Stato_Pro") == "PRO":  _icons += " ❖"
                if row.get("Squeeze"):               _icons += " 🔥"
                if row.get("Weekly_Bull"):            _icons += " 📈"
                label = f"{nome}  ({t}){_icons}"
            else:
                label = t
            scanner_rows.append((label, t))

    ts1, ts2 = st.columns([2.5, 1.5])
    with ts1:
        if scanner_rows:
            labels_disp = [r[0] for r in scanner_rows]
            tickers_raw = [r[1] for r in scanner_rows]
            sel_idx = st.selectbox(
                f"Ticker dallo scanner  ({len(scanner_rows)} disponibili — A→Z)",
                options=range(len(labels_disp)),
                format_func=lambda i: labels_disp[i],
                index=0,
                key="rm_ticker_sel",
            )
            ticker_from_scanner = tickers_raw[sel_idx]
            # Mini-info del titolo selezionato
            _row_sel = df_scanner[df_scanner["Ticker"] == ticker_from_scanner]
            if not _row_sel.empty:
                _r = _row_sel.iloc[0]
                _inf = []
                if _r.get("Prezzo"):   _inf.append(f"💲 {_r['Prezzo']}")
                if _r.get("RSI"):      _inf.append(f"RSI {_r['RSI']}")
                if _r.get("Pro_Score"):_inf.append(f"Pro {_r['Pro_Score']}")
                if _r.get("Dollar_Vol"):_inf.append(f"Vol ${float(_r['Dollar_Vol']):.1f}M")
                if _r.get("ATR_pct"):  _inf.append(f"ATR {float(_r['ATR_pct']):.1f}%")
                if _inf:
                    st.caption("  ·  ".join(str(x) for x in _inf))
        else:
            ticker_from_scanner = ""
            st.info("🔍 Esegui lo scanner per popolare la lista. Oppure inserisci il ticker a destra →")

    with ts2:
        manual_ticker = st.text_input(
            "Oppure inserisci manualmente",
            value=st.session_state.get("rm_manual_ticker", ""),
            placeholder="es. AAPL, ENI.MI…",
            key="rm_manual_ticker_input",
            help="Sovrascrive la selezione dalla lista scanner",
        )
        st.session_state["rm_manual_ticker"] = manual_ticker

    # Ticker finale: manuale ha priorita'
    ticker = (manual_ticker.strip().upper()
              if manual_ticker.strip()
              else ticker_from_scanner.strip().upper())

    # Multi-ticker: aggiungi alla lista
    if "rm_trade_plans" not in st.session_state:
        st.session_state["rm_trade_plans"] = []

    st.markdown("")
    ba1, ba2, ba3 = st.columns([1, 1, 3])


    with ba1:
        calc_btn = st.button(
            "📊 Calcola Piano Trade",
            type="primary",
            use_container_width=True,
            key="rm_calc_btn",
        )

    with ba2:
        clear_btn = st.button(
            "🗑️ Svuota lista",
            use_container_width=True,
            key="rm_clear_btn",
        )
        if clear_btn:
            st.session_state["rm_trade_plans"] = []
            st.rerun()

    # ── Calcolo piano ─────────────────────────────────────────────────────
    if calc_btn and ticker:
        with st.spinner(f"Scarico dati {ticker} da Yahoo Finance..."):
            info = _fetch_atr(ticker)

        if not info.get("ok"):
            st.error(f"Errore dati per {ticker}: {info.get('err','sconosciuto')}")
        else:
            # Verifica se prezzo manuale override (da scanner)
            entry_price = info["price"]
            if df_scanner is not None and not df_scanner.empty and "Ticker" in df_scanner.columns:
                match = df_scanner[df_scanner["Ticker"] == ticker]
                if not match.empty and "Prezzo" in match.columns:
                    _sc_price = match.iloc[0].get("Prezzo")
                    if pd.notna(_sc_price) and float(_sc_price) > 0:
                        entry_price = float(_sc_price)

            plan = calc_trade_plan(
                ticker       = ticker,
                price        = entry_price,
                atr          = info["atr"],
                capital      = capital,
                risk_pct     = risk_pct,
                atr_stop_mult= atr_mult,
                direction    = direction,
            )
            plan["name"]    = info.get("name", ticker)
            plan["atr_pct"] = info.get("atr_pct", 0)

            # Aggiorna o aggiunge alla lista
            existing = [p for p in st.session_state["rm_trade_plans"]
                        if p["ticker"] != ticker]
            st.session_state["rm_trade_plans"] = existing + [plan]
            st.rerun()
    elif calc_btn:
        st.warning("Inserisci o seleziona un ticker.")

    # ── Visualizzazione piani trade ───────────────────────────────────────
    plans = st.session_state.get("rm_trade_plans", [])

    if not plans:
        st.info("Nessun piano trade calcolato. Seleziona un ticker e clicca **📊 Calcola Piano Trade**.")
        # Mostra esempio con valori fittizi
        with st.expander("📖 Come funziona il calcolo", expanded=True):
            st.markdown("""
**Esempio — AAPL @ $185, ATR=$3.20, Capitale=$50.000, Rischio 1%:**

| Campo | Valore | Formula |
|---|---|---|
| Rischio $ | $500 | $50.000 × 1% |
| Stop Loss | $180.20 | $185 − 1.5 × $3.20 |
| Target 1 | $189.80 | $185 + 1.5 × $3.20 |
| Target 2 | $194.60 | $185 + 3.0 × $3.20 |
| Target 3 | $199.40 | $185 + 4.5 × $3.20 |
| Rischio/share | $4.80 | Entry − Stop |
| **Azioni** | **104** | **$500 / $4.80** |
| Valore posizione | $19.240 | 104 × $185 |
| % portafoglio | 38.5% | $19.240 / $50.000 |

**Regola operativa:**
- Chiudi 50% posizione a Target 1 → assicuri profitto
- Sposta stop a break-even dopo Target 1
- Lascia correre il rimanente 50% verso Target 2-3 con trailing stop
""")
        return

    # ── Tabella riassuntiva tutti i piani ─────────────────────────────────
    st.markdown("### 📋 Piano Trade Completo")

    # Calcolo utilizzo totale portafoglio
    total_invested  = sum(p["position_value"] for p in plans)
    total_risk      = sum(p["risk_dollars"] for p in plans)
    pct_invested    = total_invested / capital * 100 if capital > 0 else 0
    pct_risk        = total_risk     / capital * 100 if capital > 0 else 0

    # Warning se troppo concentrato
    if pct_invested > 80:
        st.warning(f"⚠️ Esposizione totale: **{pct_invested:.1f}%** del capitale — "
                   f"considera di ridurre il numero di posizioni.")
    if pct_risk > 5:
        st.error(f"🔴 Rischio totale: **{pct_risk:.1f}%** del capitale — "
                 f"supera la soglia di sicurezza (5% max). Riduci i trade.")

    # KPI portafoglio
    kk = st.columns(4)
    def _mkpi(col, label, val, color="#d1d4dc"):
        col.markdown(
            f'<div style="background:#1e222d;border:1px solid #2a2e39;border-radius:6px;'
            f'padding:10px;text-align:center">'
            f'<div style="color:#787b86;font-size:0.72rem">{label}</div>'
            f'<div style="color:{color};font-size:1.25rem;font-weight:bold">{val}</div>'
            f'</div>', unsafe_allow_html=True)

    _mkpi(kk[0], "N° Trade", len(plans), _CYAN)
    _mkpi(kk[1], "Investito", f"${total_invested:,.0f} ({pct_invested:.1f}%)",
          _ORANGE if pct_invested > 60 else _GREEN)
    _mkpi(kk[2], "Rischio Totale",
          f"${total_risk:,.0f} ({pct_risk:.1f}%)",
          _RED if pct_risk > 5 else _GOLD if pct_risk > 3 else _GREEN)
    _mkpi(kk[3], "Capitale Libero",
          f"${capital - total_invested:,.0f} ({100 - pct_invested:.1f}%)", _GRAY)

    st.markdown("")

    # Tabella piani
    rows = []
    for p in plans:
        rows.append({
            "Ticker":     p["ticker"],
            "Nome":       p.get("name", ""),
            "Dir":        p["direction"],
            "Entry":      f"${p['price']:.2f}",
            "ATR%":       f"{p['atr_pct']:.2f}%",
            "Stop Loss":  f"${p['stop_loss']:.2f}",
            "Stop%":      f"{p['pct_stop']:+.1f}%",
            "Target 1":   f"${p['target1']:.2f}",
            "Target 2":   f"${p['target2']:.2f}",
            "Target 3":   f"${p['target3']:.2f}",
            "R:R T2":     f"{p['rr_t2']:.1f}:1",
            "Azioni":     p["shares"],
            "Val. Pos.":  f"${p['position_value']:,.0f}",
            "% Cap.":     f"{p['pct_capital']:.1f}%",
            "Rischio $":  f"${p['risk_dollars']:,.0f}",
        })

    df_plans = pd.DataFrame(rows)

    # Colori condizionali con st.dataframe
    def _color_dir(v):
        return ("color: #00ff88; font-weight: bold" if v == "LONG"
                else "color: #ef4444; font-weight: bold")

    styled = (df_plans.style
              .applymap(_color_dir, subset=["Dir"])
              .set_properties(**{"font-family": "Courier New", "font-size": "13px"}))
    st.dataframe(styled, use_container_width=True, height=min(60 + len(plans) * 40, 350))

    # Export
    exp_csv = df_plans.to_csv(index=False).encode()
    st.download_button(
        "📥 Esporta Trade Plan CSV",
        exp_csv,
        "trade_plan.csv",
        "text/csv",
        key="rm_export_csv",
    )

    st.markdown("---")

    # ── Dettaglio singolo trade con chart ─────────────────────────────────
    st.markdown("### 📊 Dettaglio Grafico Singolo Trade")

    sel_ticker = st.selectbox(
        "Seleziona trade da visualizzare",
        options=[p["ticker"] for p in plans],
        key="rm_detail_sel",
    )

    sel_plan = next((p for p in plans if p["ticker"] == sel_ticker), None)

    if sel_plan:
        # ── Card dettaglio ─────────────────────────────────────────────
        d1, d2, d3, d4 = st.columns(4)

        def _card(col, title, value, sub="", color="#d1d4dc"):
            col.markdown(
                f'<div style="background:#1e222d;border:1px solid #2a2e39;'
                f'border-radius:8px;padding:14px 16px">'
                f'<div style="color:#787b86;font-size:0.72rem;margin-bottom:4px">{title}</div>'
                f'<div style="color:{color};font-size:1.3rem;font-weight:bold;'
                f'font-family:Courier New">{value}</div>'
                f'<div style="color:#6b7280;font-size:0.75rem;margin-top:2px">{sub}</div>'
                f'</div>', unsafe_allow_html=True)

        _card(d1, "Entry Price",    f"${sel_plan['price']:.2f}",
              f"ATR = ${sel_plan['atr']:.2f} ({sel_plan['atr_pct']:.1f}%)", _GOLD)
        _card(d2, "Stop Loss",      f"${sel_plan['stop_loss']:.2f}",
              f"{sel_plan['pct_stop']:+.1f}%  |  ${sel_plan['risk_per_share']:.2f}/share", _RED)
        _card(d3, "Target 2 (pref.)", f"${sel_plan['target2']:.2f}",
              f"R:R {sel_plan['rr_t2']:.1f}:1  |  {sel_plan['pct_t2']:+.1f}%", _GREEN)
        _card(d4, "Position Size",   f"{sel_plan['shares']} azioni",
              f"${sel_plan['position_value']:,.0f}  |  "
              f"Rischio ${sel_plan['risk_dollars']:,.0f}", _CYAN)

        st.markdown("")

        # ── Chart visivo entry/stop/target ─────────────────────────────
        _render_trade_chart(sel_plan)

        # ── Regole operative per questo trade ──────────────────────────
        with st.expander("📋 Regole operative per questo trade", expanded=True):
            is_long = sel_plan["direction"] == "LONG"
            st.markdown(f"""
**{sel_plan['ticker']} — {sel_plan['direction']} @ ${sel_plan['price']:.2f}**

| Livello | Prezzo | Azione |
|---|---|---|
| Entry | **${sel_plan['price']:.2f}** | Limit order — non usare market su titoli < $50M vol |
| Stop Loss | **${sel_plan['stop_loss']:.2f}** | Stop order obbligatorio — **non spostare mai verso il basso** |
| Target 1 | **${sel_plan['target1']:.2f}** | Chiudi **50%** posizione — sposta stop a break-even |
| Target 2 | **${sel_plan['target2']:.2f}** | Chiudi **25%** posizione — obiettivo principale |
| Target 3 | **${sel_plan['target3']:.2f}** | Lascia correre **25%** con trailing stop (1×ATR) |

**Position sizing:**
- Azioni: **{sel_plan['shares']}** × ${sel_plan['price']:.2f} = **${sel_plan['position_value']:,.0f}**
- Rischio: ${sel_plan['risk_dollars']:,.0f} ({risk_pct:.2f}% del capitale)
- % portafoglio: **{sel_plan['pct_capital']:.1f}%**

**Regola tempo:** se il titolo non si muove dopo **5 giorni**, esci al costo.
Il capitale fermo ha un costo opportunita' reale.
""")

    # ── Note sul risk management ──────────────────────────────────────────
    with st.expander("ℹ️ Principi Risk Management", expanded=False):
        st.markdown("""
**Regole fondamentali:**

1. **Max 1-2% per trade** — con 50 trade, anche 10 perdite consecutive non distruggono il conto
2. **Mai spostare lo stop loss verso il basso** — la posizione va contro di te, il mercato ha ragione
3. **No averaging down** — aggiungere a una posizione in perdita moltiplica il rischio
4. **No earnings** — non entrare se l'earnings date e' entro 14 giorni
5. **VIX > 30** — considera di dimezzare il sizing o non operare
6. **Posizione max 5% per titolo** — diversificazione riduce il rischio specifico
7. **Portfolio heat max 10%** — somma di tutti i rischi aperti < 10% del capitale

**Formula Kelly (avanzato):**
```
f* = (p × b - q) / b
dove: p = win rate, b = avg win / avg loss, q = 1 - p
```
Usa Kelly/4 o Kelly/2 per essere conservativo — Kelly pieno e' teoricamente ottimale
ma praticamente troppo volatile per la maggior parte dei trader.
""")
