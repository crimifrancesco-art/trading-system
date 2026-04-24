#!/usr/bin/env python3
"""
Patch script: Dashboard_pro_V_41c.py → Dashboard_pro_V_41d.py
Esegui: python patch_v41d.py
"""
import re

with open("Dashboard_pro_V_41c.py", "r", encoding="utf-8") as f:
    src = f.read()

# ═══════════════════════════════════════════════════════════════════════
# PATCH 1 — CSS TAB SU 2 RIGHE
# ═══════════════════════════════════════════════════════════════════════
CSS_2ROWS = '''
# ── v41d: Tab visibili su 2 righe ──────────────────────────────────────
st.markdown("""
<style>
[data-testid="stTabs"] > div:first-child {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 2px 3px !important;
    max-height: none !important;
    overflow: visible !important;
    border-bottom: 1px solid #2a2e39 !important;
    padding-bottom: 6px !important;
    background: #0e1117 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    white-space: nowrap !important;
    font-size: 0.75rem !important;
    padding: 4px 9px !important;
    border-radius: 4px 4px 0 0 !important;
    margin-bottom: 2px !important;
    border: 1px solid #2a2e3966 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    border-bottom: 2px solid #2962ff !important;
    color: #2962ff !important;
    background: #131722 !important;
}
</style>
""", unsafe_allow_html=True)

'''

src = src.replace(
    "tabs = st.tabs([\n",
    CSS_2ROWS + "tabs = st.tabs([\n",
    1
)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 2 — AGGIUNGI TAB "🤖 Modulo 2 AI" nella lista tab
# ═══════════════════════════════════════════════════════════════════════
src = src.replace(
    '    "🤖 AI Assistant",\n    "🎲 Options Scanner",',
    '    "🤖 AI Assistant",\n    "🤖 Modulo 2 AI",        # v41d — tab dedicato\n    "🎲 Options Scanner",',
    1
)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 3 — AGGIORNA UNPACKING DEI TAB
# ═══════════════════════════════════════════════════════════════════════
src = src.replace(
    """ tab_ai, tab_opts, tab_mom, tab_news,
 tab_analisi, tab_journal,
 tab_w) = tabs""",
    """ tab_ai, tab_ai2, tab_opts, tab_mom, tab_news,
 tab_analisi, tab_journal,
 tab_w) = tabs""",
    1
)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 4 — NUOVO BLOCCO "with tab_ai2"
# Inserisce dopo la chiusura del blocco with tab_ai (cerca marker univoco)
# ═══════════════════════════════════════════════════════════════════════
TAB_AI2_BLOCK = '''

# ── v41d — MODULO 2 AI come TAB autonomo ──────────────────────────────
with tab_ai2:
    st.session_state["last_active_tab"] = "MODULO2_AI"
    st.markdown(
        "<div class='section-pill'>🤖 MODULO 2 — AI ANALYST · Setup · Target · Invalidazione · Rischio</div>",
        unsafe_allow_html=True
    )
    st.caption("Fallback automatico: Gemini (free) → Groq (free) → OpenRouter → Claude · Clicca 🧠 Analizza su ogni ticker")
    _ai2_src_sel = st.radio(
        "Analizza segnali:",
        ["PRO/STRONG (scanner)", "CONFLUENCE", "Tutti scanner"],
        horizontal=True, key="ai2_source_sel"
    )
    if _ai2_src_sel == "PRO/STRONG (scanner)":
        _df_ai2 = df_ep[df_ep["Stato_Pro"].isin(["PRO","STRONG"])].copy() if "Stato_Pro" in df_ep.columns and not df_ep.empty else df_ep.copy()
    elif _ai2_src_sel == "CONFLUENCE":
        if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
            _df_ai2 = df_ep[(df_ep["Stato_Early"]=="EARLY") & (df_ep["Stato_Pro"]=="PRO")].copy()
        else:
            _df_ai2 = df_ep.copy()
    else:
        _df_ai2 = df_ep.copy()
    _render_ai_explainer_v41(_df_ai2, "MOD2")

'''

# Inserisce prima del blocco "with tab_opts:"
src = src.replace(
    "\nwith tab_opts:",
    TAB_AI2_BLOCK + "\nwith tab_opts:",
    1
)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 5 — MAPPA CALORE GLOBALE: sostituisce SVG con HTML a card
# ═══════════════════════════════════════════════════════════════════════
OLD_MAPPA = '''    with st.expander("🌍 Mappa Calore Globale — Performance indici mondiali", expanded=False):
        try:
            _live_for_map = _fetch_live_markets_v41()
            if _live_for_map:
                _map_data = {m["sym"]: m for m in _live_for_map}
                # Regioni geografiche con posizioni SVG (x,y,w,h)
                _regions = [
                    # USA
                    ("🇺🇸 S&P 500",  "^GSPC",      10,  30, 100, 50),
                    ("🇺🇸 NASDAQ",   "^IXIC",      10,  85, 100, 40),
                    ("🇺🇸 DowJones", "^DJI",       10, 130, 100, 35),
                    ("🇺🇸 Russ.2K",  "^RUT",       10, 170, 100, 30),
                    # Europa
                    ("🇮🇹 MIB",      "FTSEMIB.MI", 140, 30,  80, 40),
                    ("🇬🇧 FTSE",     "^FTSE",      225, 30,  80, 40),
                    ("🇩🇪 DAX",      "^GDAXI",     310, 30,  80, 40),
                    ("🇫🇷 CAC",      "^FCHI",      140, 75,  80, 35),
                    ("🇪🇸 IBEX",     "^IBEX",      225, 75,  80, 35),
                    ("🇳🇱 AEX",      "^AEX",       310, 75,  80, 35),
                    # Asia
                    ("🇯🇵 Nikkei",   "^N225",      420, 30, 100, 40),
                    ("🇭🇰 HSeng",    "^HSI",       420, 75, 100, 35),
                    ("🇨🇳 Shangh.",  "000001.SS",  420,115, 100, 35),
                    ("🇰🇷 KOSPI",    "^KS11",      420,155,  55, 30),
                    ("🇮🇳 Nifty",    "^NSEI",      480,155,  40, 30),
                    ("🇧🇷 BVSP",     "^BVSP",      535,155,  55, 30),
                    # Volatilità e altro
                    ("😰 VIX",       "^VIX",       140,120,  80, 30),
                    ("₿ BTC",        "BTC-USD",    225,120,  80, 30),
                    ("🥇 Gold",      "GC=F",       310,120,  80, 30),
                    ("🛢️ Oil",       "CL=F",       140,155,  80, 30),
                    ("💵 DXY",       "DX-Y.NYB",   225,155,  80, 30),
                    ("🏦 TLT",       "TLT",        310,155,  80, 30),
                ]
                _svg_w, _svg_h = 600, 210
                _svg_rects = ""
                for _rlbl, _rsym, _rx, _ry, _rw, _rh in _regions:
                    _md = _map_data.get(_rsym)
                    if not _md:
                        continue
                    _chg_r = _md.get("chg", 0)
                    # Colore: verde = positivo, rosso = negativo, intensità = magnitudine
                    _int = min(1.0, abs(_chg_r) / 3.0)
                    if _chg_r >= 0:
                        _r0,_g0,_b0 = 0,100+int(100*_int),60
                    else:
                        _r0,_g0,_b0 = 120+int(100*_int),40,40
                    _fill = f"rgb({_r0},{_g0},{_b0})"
                    _border = "#00ff88" if _chg_r >= 0 else "#ef4444"
                    _txt_main = _rlbl[:10]
                    _chg_lbl = f"{'+'if _chg_r>=0 else ''}{_chg_r:.1f}%"
                    _pr_lbl  = _fmt_price_v41(_md)[:8]
                    _fsize   = 8 if _rw < 70 else 9
                    _svg_rects += (
                        f"<rect x=\'{_rx}\' y=\'{_ry}\' width=\'{_rw}\' height=\'{_rh}\' "
                        f"rx=\'4\' fill=\'{_fill}\' stroke=\'{_border}\' stroke-width=\'0.8\' opacity=\'0.9\'/>"
                        f"<text x=\'{_rx+_rw//2}\' y=\'{_ry+_rh//2-6}\' text-anchor=\'middle\' "
                        f"font-size=\'{_fsize}\' fill=\'#e2e8f0\' font-family=\'Trebuchet MS\'>{_txt_main}</text>"
                        f"<text x=\'{_rx+_rw//2}\' y=\'{_ry+_rh//2+5}\' text-anchor=\'middle\' "
                        f"font-size=\'{_fsize+1}\' fill=\'white\' font-family=\'Courier New\' font-weight=\'bold\'>"
                        f"{_chg_lbl}</text>"
                        f"<text x=\'{_rx+_rw//2}\' y=\'{_ry+_rh//2+15}\' text-anchor=\'middle\' "
                        f"font-size=\'{_fsize-1}\' fill=\'#9ca3af\' font-family=\'Courier New\'>{_pr_lbl}</text>"
                    )
                _map_svg = (
                    f"<svg width=\'100%\' viewBox=\'0 0 {_svg_w} {_svg_h}\' "
                    f"xmlns=\'http://www.w3.org/2000/svg\' "
                    f"style=\'background:#131722;border-radius:8px;border:1px solid #2a2e39\'>"
                    # Labels regioni
                    f"<text x=\'60\' y=\'18\' text-anchor=\'middle\' font-size=\'10\' fill=\'#50c4e0\' "
                    f"font-family=\'Trebuchet MS\' font-weight=\'bold\'>AMERICAS</text>"
                    f"<text x=\'260\' y=\'18\' text-anchor=\'middle\' font-size=\'10\' fill=\'#50c4e0\' "
                    f"font-family=\'Trebuchet MS\' font-weight=\'bold\'>EUROPA</text>"
                    f"<text x=\'475\' y=\'18\' text-anchor=\'middle\' font-size=\'10\' fill=\'#50c4e0\' "
                    f"font-family=\'Trebuchet MS\' font-weight=\'bold\'>ASIA / EM</text>"
                    f"<line x1=\'125\' y1=\'10\' x2=\'125\' y2=\'{_svg_h-5}\' stroke=\'#2a2e39\' stroke-width=\'1\'/>"
                    f"<line x1=\'415\' y1=\'10\' x2=\'415\' y2=\'{_svg_h-5}\' stroke=\'#2a2e39\' stroke-width=\'1\'/>"
                    f"{_svg_rects}"
                    f"</svg>"
                )
                st.markdown(_map_svg, unsafe_allow_html=True)
                st.caption("Colore: verde = rialzo · rosso = ribasso · intensità proporzionale alla variazione giornaliera %")
        except Exception as _map_e:
            st.info(f"Mappa non disponibile: {str(_map_e)[:60]}")'''

NEW_MAPPA = '''    with st.expander("🌍 Mappa Calore Globale — Performance indici mondiali", expanded=True):
        try:
            _live_for_map = _fetch_live_markets_v41()
            if _live_for_map:
                _map_data = {m["sym"]: m for m in _live_for_map}

                # ── v41d: layout HTML a card, NO SVG raw ──────────────────
                _map_regions = {
                    "🌎 AMERICAS": [
                        ("US S&P 500",  "^GSPC"), ("US NASDAQ",   "^IXIC"),
                        ("US DowJone",  "^DJI"),  ("US Russ.2K",  "^RUT"),
                    ],
                    "🌍 EUROPA": [
                        ("IT MIB",   "FTSEMIB.MI"), ("GB FTSE", "^FTSE"),
                        ("DE DAX",   "^GDAXI"),     ("FR CAC",  "^FCHI"),
                        ("ES IBEX",  "^IBEX"),      ("NL AEX",  "^AEX"),
                        ("😱 VIX",   "^VIX"),       ("₿ BTC",   "BTC-USD"),
                    ],
                    "🌏 ASIA / EM": [
                        ("JP Nikkei",  "^N225"),      ("HK HSeng",   "^HSI"),
                        ("CN Shangh.", "000001.SS"),   ("KR KOSPI",   "^KS11"),
                        ("IN Nifty",   "^NSEI"),       ("BR BVSP",    "^BVSP"),
                    ],
                }
                _map_macro = [
                    ("🥇 Gold", "GC=F"), ("🛢 Oil", "CL=F"),
                    ("💵 DXY",  "DX-Y.NYB"), ("📉 TLT", "TLT"),
                ]

                def _map_card_v41d(lbl, sym):
                    _md2 = _map_data.get(sym, {})
                    _chg2 = _md2.get("chg", 0) or 0
                    _pr2  = _md2.get("price", 0) or 0
                    _inten = min(1.0, abs(_chg2) / 3.0)
                    if _chg2 >= 0:
                        _bg2 = f"rgba(0,{int(80+120*_inten)},70,0.88)"
                        _ar2 = "▲"; _col2 = "#00ff88"
                    else:
                        _bg2 = f"rgba({int(120+110*_inten)},35,35,0.88)"
                        _ar2 = "▼"; _col2 = "#ef4444"
                    if _pr2 > 10000:
                        _pstr = f"{_pr2:,.0f}"
                    elif _pr2 > 100:
                        _pstr = f"{_pr2:,.1f}"
                    elif _pr2 > 0:
                        _pstr = f"{_pr2:.2f}"
                    else:
                        _pstr = "—"
                    return (
                        f"<div style='background:{_bg2};border:1px solid #2a2e39;border-radius:6px;"
                        f"padding:7px 5px;text-align:center;min-width:82px;flex:1;max-width:120px;"
                        f"transition:border-color .2s'>"
                        f"<div style='color:#e2e8f0;font-size:0.64rem;font-family:Trebuchet MS;"
                        f"font-weight:bold;letter-spacing:0.4px;margin-bottom:2px'>{lbl}</div>"
                        f"<div style='color:{_col2};font-family:Courier New;font-weight:bold;"
                        f"font-size:0.88rem'>{_ar2}{abs(_chg2):.1f}%</div>"
                        f"<div style='color:#9ca3af;font-size:0.65rem;font-family:Courier New'>{_pstr}</div>"
                        f"</div>"
                    )

                _reg_cols = st.columns([1, 1.5, 1])
                for _ci, (_rname, _rassets) in enumerate(_map_regions.items()):
                    with _reg_cols[_ci]:
                        st.markdown(
                            f"<div style='color:#50c4e0;font-size:0.72rem;font-weight:bold;"
                            f"letter-spacing:2px;text-align:center;margin-bottom:6px;"
                            f"border-bottom:1px solid #2a2e39;padding-bottom:4px'>{_rname}</div>",
                            unsafe_allow_html=True
                        )
                        _grid_html = "<div style='display:flex;flex-wrap:wrap;gap:5px;justify-content:center'>"
                        for _lbl2, _sym2 in _rassets:
                            _grid_html += _map_card_v41d(_lbl2, _sym2)
                        _grid_html += "</div>"
                        st.markdown(_grid_html, unsafe_allow_html=True)

                # Riga macro
                st.markdown("<div style='margin-top:8px;display:flex;gap:6px;justify-content:center;flex-wrap:wrap'>"
                            + "".join(_map_card_v41d(l, s) for l, s in _map_macro)
                            + "</div>", unsafe_allow_html=True)
                st.caption("🟢 verde = rialzo · 🔴 rosso = ribasso · intensità proporzionale alla variazione giornaliera %")
            else:
                st.info("Dati mappa non disponibili — riprova tra qualche secondo.")
        except Exception as _map_e:
            st.warning(f"Mappa non disponibile: {str(_map_e)[:80]}")'''

src = src.replace(OLD_MAPPA, NEW_MAPPA, 1)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 6 — HEATMAP SETTORIALE: fix 3m (period 95d invece di 3mo)
# ═══════════════════════════════════════════════════════════════════════
src = src.replace(
    '                _raw_h = _yh.download(_tickers_str, period="3mo", interval="1d",\n                                       auto_adjust=True, progress=False, group_by="ticker")',
    '                _raw_h = _yh.download(_tickers_str, period="95d", interval="1d",\n                                       auto_adjust=True, progress=False, group_by="ticker")',
    1
)

# Fix calcolo 3m: usa indice relativo robusto invece di iloc[-63] fisso
src = src.replace(
    '                        _r3m = round((_cl_h.iloc[-1]/_cl_h.iloc[-63]-1)*100,2) if len(_cl_h)>=63 else 0',
    '                        _idx3m = min(63, len(_cl_h)-1)\n                        _r3m = round((_cl_h.iloc[-1]/_cl_h.iloc[-_idx3m]-1)*100,2) if _idx3m >= 2 else 0',
    1
)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 7 — RANKING SETTORI: elimina bar chart, tieni solo tabella testo
# Il bar chart non esiste come plotly separato — la sezione è già solo
# tabella testuale (righe 7833-7849). Nessuna riga plotly da rimuovere.
# Aggiunge però header periodo sopra la tabella per chiarezza.
# ═══════════════════════════════════════════════════════════════════════
src = src.replace(
    '        st.markdown("#### 🏆 Ranking Settori")',
    '        st.markdown("#### 🏆 Ranking Settori — ordinati per periodo selezionato")\n        st.caption("📊 Grafico bar rimosso in v41d — visualizzazione tabella compatta")',
    1
)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 8 — P&L TRACKER: aggiungi notifiche Email/Telegram dopo alert engine
# ═══════════════════════════════════════════════════════════════════════
PNL_NOTIF_BLOCK = '''
            # ── v41d: Notifiche FIRED via Telegram (auto) ─────────────────
            _tgt_pnl = st.session_state.get("tg_token_pnl","") or st.secrets.get("TELEGRAM_TOKEN","")
            _cgp_pnl  = st.session_state.get("tg_chat_pnl","")  or st.secrets.get("TELEGRAM_CHAT","")
            if _tgt_pnl and _cgp_pnl and _alerts:
                _just_fired = [_a for _a in _alerts.values()
                               if _a.get("fired") and not _a.get("notified")]
                for _fa in _just_fired:
                    try:
                        import requests as _reqf
                        _fmsg = (
                            f"🚨 *ALERT FIRED — Dashboard v41d*\\n"
                            f"Ticker: *{_fa['tkr']}*\\n"
                            f"Condizione: {_fa['type']} {_fa['val']}\\n"
                            f"Ora: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                        )
                        _reqf.post(
                            f"https://api.telegram.org/bot{_tgt_pnl}/sendMessage",
                            json={"chat_id": _cgp_pnl, "text": _fmsg, "parse_mode": "Markdown"},
                            timeout=5
                        )
                        _fa["notified"] = True
                    except Exception:
                        pass

        # ── v41d: Pannello canali notifica ────────────────────────────────
        st.markdown("---")
        st.markdown("**📡 Canali di Notifica Alert**")
        _notif_c1, _notif_c2 = st.columns(2)

        with _notif_c1:
            st.markdown("**Telegram 🤖**")
            _tg_tok = st.text_input(
                "Bot Token", type="password",
                value=st.secrets.get("TELEGRAM_TOKEN","") or st.session_state.get("tg_token_pnl",""),
                key="pnl_tg_token", placeholder="123456:ABC-xyz..."
            )
            _tg_cht = st.text_input(
                "Chat ID",
                value=st.secrets.get("TELEGRAM_CHAT","") or st.session_state.get("tg_chat_pnl",""),
                key="pnl_tg_chat", placeholder="-100123456789"
            )
            if _tg_tok: st.session_state["tg_token_pnl"] = _tg_tok
            if _tg_cht: st.session_state["tg_chat_pnl"] = _tg_cht
            _tpnl_total = 0.0
            for _t2, _pos2 in st.session_state.get("v41_pnl_entries",{}).items():
                if not _df_ep_wl.empty and "Ticker" in _df_ep_wl.columns and "Prezzo" in _df_ep_wl.columns:
                    _m2 = _df_ep_wl[_df_ep_wl["Ticker"]==_t2]
                    if not _m2.empty:
                        _tpnl_total += (float(_m2.iloc[0]["Prezzo"]) - _pos2["entry"]) * _pos2["size"]
            if st.button("📨 Invia Report P&L Telegram", key="pnl_tg_send"):
                try:
                    import requests as _rqt
                    _pnl_entries = st.session_state.get("v41_pnl_entries",{})
                    _lines_msg = [f"  • {t}: entry={p['entry']:.2f} size={p['size']}" for t,p in _pnl_entries.items()]
                    _msg_pnl = (
                        f"💼 *Trading Dashboard v41d — Report P&L*\\n"
                        f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\\n"
                        f"Portfolio P&L: *${_tpnl_total:+,.0f}*\\n"
                        f"Posizioni aperte: {len(_pnl_entries)}\\n"
                        + ("\\n".join(_lines_msg) if _lines_msg else "Nessuna posizione")
                    )
                    _rsp = _rqt.post(
                        f"https://api.telegram.org/bot{_tg_tok}/sendMessage",
                        json={"chat_id": _tg_cht, "text": _msg_pnl, "parse_mode": "Markdown"},
                        timeout=8
                    )
                    if _rsp.status_code == 200:
                        st.success("✅ Report P&L inviato su Telegram!")
                    else:
                        st.error(f"Errore {_rsp.status_code}: {_rsp.text[:80]}")
                except Exception as _te:
                    st.error(f"Errore Telegram: {_te}")
            if st.button("🔔 Test Telegram", key="pnl_tg_test"):
                try:
                    import requests as _rqtt
                    _rqtt.post(
                        f"https://api.telegram.org/bot{_tg_tok}/sendMessage",
                        json={"chat_id": _tg_cht,
                              "text": f"✅ Test OK — Dashboard v41d · {datetime.now().strftime('%H:%M')}",
                              "parse_mode": "Markdown"},
                        timeout=6
                    )
                    st.success("✅ Test inviato!")
                except Exception as _te2:
                    st.error(f"Errore: {_te2}")

        with _notif_c2:
            st.markdown("**Email 📧 (Gmail SMTP)**")
            _em_to   = st.text_input("Destinatario", key="pnl_email_to",   placeholder="tuo@email.com")
            _em_from = st.text_input("Mittente Gmail", key="pnl_email_from", placeholder="bot@gmail.com")
            _em_pwd  = st.text_input("App Password", type="password", key="pnl_email_pwd",
                                     help="Genera su myaccount.google.com → Sicurezza → App password")
            _em_subj = st.text_input("Oggetto", key="pnl_email_subj", value="🔔 Alert Dashboard v41d")
            if st.button("📧 Invia Report P&L Email", key="pnl_email_send"):
                try:
                    import smtplib
                    from email.mime.text import MIMEText as _MIMEText
                    _pnl_e = st.session_state.get("v41_pnl_entries",{})
                    _body_e = (
                        f"Trading Dashboard v41d — Report P&L\\n"
                        f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\\n"
                        f"Portfolio P&L: ${_tpnl_total:+,.0f}\\n"
                        f"Posizioni aperte: {len(_pnl_e)}\\n\\n"
                        + "\\n".join(f"  {t}: entry={p['entry']:.2f} size={p['size']}" for t,p in _pnl_e.items())
                    )
                    _alerts_e = st.session_state.get("v41_alerts",{})
                    if _alerts_e:
                        _body_e += "\\n\\nAlert attivi:\\n"
                        for _ak_e, _av_e in _alerts_e.items():
                            _status_e = "🔴 FIRED" if _av_e.get("fired") else "🟡 Attivo"
                            _body_e += f"  {_av_e['tkr']} {_av_e['type']} {_av_e['val']} — {_status_e}\\n"
                    _msg_e = _MIMEText(_body_e)
                    _msg_e["Subject"] = _em_subj
                    _msg_e["From"]    = _em_from
                    _msg_e["To"]      = _em_to
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as _srv_e:
                        _srv_e.login(_em_from, _em_pwd)
                        _srv_e.send_message(_msg_e)
                    st.success("✅ Email inviata!")
                except Exception as _ee:
                    st.error(f"Errore email: {_ee}")
            if st.button("📧 Test Email", key="pnl_email_test"):
                try:
                    import smtplib
                    from email.mime.text import MIMEText as _MIMEText2
                    _msg_t = _MIMEText2(f"Test OK — Dashboard v41d · {datetime.now().strftime('%H:%M')}")
                    _msg_t["Subject"] = "✅ Test " + _em_subj
                    _msg_t["From"] = _em_from; _msg_t["To"] = _em_to
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as _srv_t:
                        _srv_t.login(_em_from, _em_pwd)
                        _srv_t.send_message(_msg_t)
                    st.success("✅ Test email inviato!")
                except Exception as _et:
                    st.error(f"Errore: {_et}")

'''

# Inserisce subito prima della riga "    st.markdown("")" dopo la chiusura dell'expander P&L
src = src.replace(
    '\n    st.markdown("")\n    _wl_col1, _wl_col2 = st.columns([3, 2])',
    PNL_NOTIF_BLOCK + '\n    st.markdown("")\n    _wl_col1, _wl_col2 = st.columns([3, 2])',
    1
)

# ═══════════════════════════════════════════════════════════════════════
# PATCH 9 — MOMENTUM ALERTS: aggiorna header + titolo sezione v41d
# ═══════════════════════════════════════════════════════════════════════
src = src.replace(
    "    st.markdown('<div class=\"section-pill\">⚡ MOMENTUM ALERTS v41c — Alert Pattern Tecnici in Tempo Reale</div>',\n                unsafe_allow_html=True)",
    "    st.markdown('<div class=\"section-pill\">⚡ MOMENTUM ALERTS v41d — Ticker · Nome · Tipo · Valore · RSI · Vol× · CSS · Priorità</div>',\n                unsafe_allow_html=True)",
    1
)

# ── Aggiorna il titolo versione ──────────────────────────────────────────
src = src.replace(
    '"📈 P&L Tracker & Alert Engine v41"',
    '"📈 P&L Tracker & Alert Engine v41d"',
    1
)

# ── Aggiorna versione nel titolo principale se presente ─────────────────
src = src.replace('v41c', 'v41d')
src = src.replace('V_41c', 'V_41d')

# ── Scrivi il file output ───────────────────────────────────────────────
with open("Dashboard_pro_V_41d.py", "w", encoding="utf-8") as f:
    f.write(src)

print("✅ Dashboard_pro_V_41d.py generato con successo!")
print(f"   Dimensione: {len(src):,} caratteri")

# Verifica patch applicate
checks = {
    "CSS 2 righe":       "flex-wrap: wrap !important" in src,
    "Tab Modulo 2 AI":   "tab_ai2" in src,
    "with tab_ai2":      "with tab_ai2:" in src,
    "Mappa HTML card":   "_map_card_v41d" in src,
    "Heatmap 3m fix":    "_idx3m" in src,
    "P&L notifiche":     "pnl_tg_token" in src,
    "Email SMTP":        "smtp.gmail.com" in src,
    "Auto-notifica":     "_just_fired" in src,
    "Mom Alerts v41d":   "MOMENTUM ALERTS v41d" in src,
}
print("\n── Verifica patch ──")
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
