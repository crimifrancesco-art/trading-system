#!/usr/bin/env python3
"""
Patch: Dashboard_pro_V_41c.py → Dashboard_pro_V_41d.py
- NON importa streamlit (solo trasformazioni di testo)
- 13 patch con verifica finale
Esegui: python patch_v41d.py
"""
import sys, os

SRC = "Dashboard_pro_V_41c.py"
DST = "Dashboard_pro_V_41d.py"

if not os.path.exists(SRC):
    print(f"⛔ '{SRC}' non trovato.")
    sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print(f"📂 {SRC}: {len(src):,} caratteri")

# ═══ P1: CSS + JS tab 2 righe ══════════════════════════════════════════
CSS_2ROWS = (
    "\n# \u2500\u2500 v41d: Tab su 2 righe \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "st.markdown(\"\"\"\n"
    "<style>\n"
    "[data-testid=\"stTabs\"] > div:first-child {"
    " display:flex !important; flex-wrap:wrap !important;"
    " gap:2px 3px !important; max-height:none !important;"
    " overflow:visible !important; width:100% !important;"
    " border-bottom:1px solid #2a2e39 !important; padding-bottom:6px !important;}\n"
    "[data-testid=\"stTabs\"] [data-baseweb=\"tab\"] {"
    " white-space:nowrap !important; font-size:0.74rem !important;"
    " padding:4px 8px !important; flex-shrink:0 !important;"
    " margin-bottom:2px !important; max-width:160px !important;"
    " border-radius:4px 4px 0 0 !important; border:1px solid #2a2e3966 !important;}\n"
    "[data-testid=\"stTabs\"] [aria-selected=\"true\"] {"
    " border-bottom:2px solid #2962ff !important;"
    " color:#2962ff !important; background:#131722 !important; font-weight:bold !important;}\n"
    "[data-testid=\"stTabs\"] > div:first-child > div {"
    " overflow:visible !important; max-height:none !important;}\n"
    "[role='tablist'] { display:flex !important; flex-wrap:wrap !important;"
    " overflow:visible !important; max-height:none !important; height:auto !important;}\n"
    "[data-baseweb='tab-list'] { display:flex !important; flex-wrap:wrap !important;"
    " overflow:visible !important; max-height:none !important; height:auto !important;}\n"
    "</style>\n"
    "<script>\n"
    "(function fixTabWrap(){\n"
    "  function apply(){\n"
    "    ['[data-testid=\"stTabs\"] > div:first-child','[role=\"tablist\"]','[data-baseweb=\"tab-list\"]']\n"
    "      .forEach(function(s){document.querySelectorAll(s).forEach(function(el){\n"
    "        el.style.setProperty('display','flex','important');\n"
    "        el.style.setProperty('flex-wrap','wrap','important');\n"
    "        el.style.setProperty('max-height','none','important');\n"
    "        el.style.setProperty('overflow','visible','important');\n"
    "        el.style.setProperty('height','auto','important');\n"
    "        var p=el.parentElement;for(var i=0;i<4;i++){if(p){\n"
    "          p.style.setProperty('overflow','visible','important');\n"
    "          p.style.setProperty('max-height','none','important');\n"
    "          p=p.parentElement;}}\n"
    "      });});\n"
    "    document.querySelectorAll('[data-baseweb=\"tab\"]').forEach(function(t){\n"
    "      t.style.setProperty('white-space','nowrap','important');\n"
    "      t.style.setProperty('flex-shrink','0','important');\n"
    "      t.style.setProperty('font-size','0.74rem','important');\n"
    "    });\n"
    "  }\n"
    "  apply();\n"
    "  [200,600,1200,2500].forEach(function(t){setTimeout(apply,t);});\n"
    "  new MutationObserver(apply).observe(document.body,{childList:true,subtree:true});\n"
    "})();\n"
    "</script>\n"
    "\"\"\", unsafe_allow_html=True)\n\n"
)
n1 = src.count("tabs = st.tabs([\n")
src = src.replace("tabs = st.tabs([\n", CSS_2ROWS + "tabs = st.tabs([\n", 1)
print(f"  P1  CSS+JS 2 righe:        {'OK' if n1 else 'SKIP'}")

# ═══ P2: aggiunge tab Modulo 2 AI ══════════════════════════════════════
OLD_TAB = '    "🤖 AI Assistant",\n    "🎲 Options Scanner",'
NEW_TAB = '    "🤖 AI Assistant",\n    "🤖 Modulo 2 AI",\n    "🎲 Options Scanner",'
n2 = src.count(OLD_TAB); src = src.replace(OLD_TAB, NEW_TAB, 1)
print(f"  P2  tab Modulo 2 AI:       {'OK' if n2 else 'SKIP'}")

# ═══ P3: unpacking tabs ════════════════════════════════════════════════
OLD_UNP = " tab_ai, tab_opts, tab_mom, tab_news,\n tab_analisi, tab_journal,\n tab_w) = tabs"
NEW_UNP = " tab_ai, tab_ai2, tab_opts, tab_mom, tab_news,\n tab_analisi, tab_journal,\n tab_w) = tabs"
n3 = src.count(OLD_UNP); src = src.replace(OLD_UNP, NEW_UNP, 1)
print(f"  P3  unpacking tab_ai2:     {'OK' if n3 else 'SKIP'}")

# ═══ P4: blocco with tab_ai2 ══════════════════════════════════════════
TAB_AI2 = (
    "\n# \u2500\u2500 v41d \u2014 MODULO 2 AI \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "with tab_ai2:\n"
    "    st.session_state[\"last_active_tab\"] = \"MODULO2_AI\"\n"
    "    st.markdown(\"<div class='section-pill'>🤖 MODULO 2 AI ANALYST</div>\", unsafe_allow_html=True)\n"
    "    st.caption(\"Fallback: Gemini \u2192 Groq \u2192 OpenRouter \u2192 Claude\")\n"
    "    _ai2_sel = st.radio(\"Segnali:\", [\"PRO/STRONG\", \"CONFLUENCE\", \"Tutti\"], horizontal=True, key=\"ai2_sel\")\n"
    "    if _ai2_sel == \"PRO/STRONG\":\n"
    "        _df_ai2 = df_ep[df_ep[\"Stato_Pro\"].isin([\"PRO\",\"STRONG\"])].copy() if not df_ep.empty and \"Stato_Pro\" in df_ep.columns else df_ep.copy()\n"
    "    elif _ai2_sel == \"CONFLUENCE\":\n"
    "        _df_ai2 = df_ep[(df_ep.get(\"Stato_Early\",pd.Series(dtype=str))==\"EARLY\") & (df_ep.get(\"Stato_Pro\",pd.Series(dtype=str))==\"PRO\")].copy() if not df_ep.empty else df_ep.copy()\n"
    "    else:\n"
    "        _df_ai2 = df_ep.copy()\n"
    "    _render_ai_explainer_v41(_df_ai2, \"MOD2\")\n\n"
)
n4 = src.count("\nwith tab_opts:"); src = src.replace("\nwith tab_opts:", TAB_AI2 + "\nwith tab_opts:", 1)
print(f"  P4  with tab_ai2:          {'OK' if n4 else 'SKIP'}")

# ═══ P5: Mappa calore HTML card ════════════════════════════════════════
OLD_MAP_MARKER = '    with st.expander("🌍 Mappa Calore Globale — Performance indici mondiali", expanded=False):'
OLD_MAP_END    = '        except Exception as _map_e:\n            st.info(f"Mappa non disponibile: {str(_map_e)[:60]}")'
NEW_MAP = (
    '    with st.expander("🌍 Mappa Calore Globale — Performance indici mondiali v41d", expanded=True):\n'
    '        try:\n'
    '            _live_for_map = _fetch_live_markets_v41()\n'
    '            if _live_for_map:\n'
    '                _map_data = {m["sym"]: m for m in _live_for_map}\n'
    '                _map_regions = {\n'
    '                    "\U0001f30e AMERICAS": [("S&P500","^GSPC"),("NASDAQ","^IXIC"),("DowJones","^DJI"),("Russ2K","^RUT")],\n'
    '                    "\U0001f30d EUROPA":   [("IT MIB","FTSEMIB.MI"),("FTSE","^FTSE"),("DAX","^GDAXI"),("CAC","^FCHI"),\n'
    '                                           ("IBEX","^IBEX"),("AEX","^AEX"),("VIX","^VIX"),("BTC","BTC-USD")],\n'
    '                    "\U0001f30f ASIA/EM":  [("Nikkei","^N225"),("HSeng","^HSI"),("Shanghai","000001.SS"),\n'
    '                                           ("KOSPI","^KS11"),("Nifty","^NSEI"),("BVSP","^BVSP")],\n'
    '                }\n'
    '                _map_macro = [("\U0001f947 Gold","GC=F"),("\U0001f6e2 Oil","CL=F"),("\U0001f4b5 DXY","DX-Y.NYB"),("\U0001f4c9 TLT","TLT")]\n'
    '                def _map_card_v41d(lbl, sym):\n'
    '                    _d=_map_data.get(sym,{}); _c=float(_d.get("chg",0) or 0); _p=float(_d.get("price",0) or 0)\n'
    '                    _i=min(1.0,abs(_c)/3.0)\n'
    '                    _bg=f"rgba(0,{int(80+120*_i)},70,.88)" if _c>=0 else f"rgba({int(120+110*_i)},35,35,.88)"\n'
    '                    _ar,_cl=("▲","#00ff88") if _c>=0 else ("▼","#ef4444")\n'
    '                    _ps=f"{_p:,.0f}" if _p>10000 else (f"{_p:.1f}" if _p>100 else (f"{_p:.2f}" if _p>0 else "\u2014"))\n'
    "                    return (f\"<div style='background:{_bg};border:1px solid #2a2e39;border-radius:6px;\"\n"
    "                            f\"padding:6px 4px;text-align:center;min-width:72px;flex:1;max-width:110px'>\"\n"
    "                            f\"<div style='color:#e2e8f0;font-size:.62rem;font-weight:bold'>{lbl}</div>\"\n"
    "                            f\"<div style='color:{_cl};font-family:Courier New;font-weight:bold;font-size:.84rem'>{_ar}{abs(_c):.1f}%</div>\"\n"
    "                            f\"<div style='color:#9ca3af;font-size:.62rem'>{_ps}</div></div>\")\n"
    '                _rc=st.columns([1,1.5,1])\n'
    '                for _ci,(_rn,_ra) in enumerate(_map_regions.items()):\n'
    '                    with _rc[_ci]:\n'
    "                        st.markdown(f\"<div style='color:#50c4e0;font-size:.70rem;font-weight:bold;text-align:center;\"\n"
    "                                    f\"letter-spacing:2px;border-bottom:1px solid #2a2e39;padding-bottom:4px;\"\n"
    "                                    f\"margin-bottom:6px'>{_rn}</div>\",unsafe_allow_html=True)\n"
    "                        st.markdown(\"<div style='display:flex;flex-wrap:wrap;gap:4px;justify-content:center'>\"\n"
    '                                    + "".join(_map_card_v41d(l,s) for l,s in _ra) + "</div>",unsafe_allow_html=True)\n'
    "                st.markdown(\"<div style='margin-top:8px;display:flex;gap:5px;justify-content:center;flex-wrap:wrap'>\"\n"
    '                            + "".join(_map_card_v41d(l,s) for l,s in _map_macro) + "</div>",unsafe_allow_html=True)\n'
    '                st.caption("\U0001f7e2 rialzo · \U0001f534 ribasso · intensit\u00e0 = % variazione")\n'
    '            else:\n'
    '                st.info("Dati mappa non disponibili.")\n'
    '        except Exception as _map_e:\n'
    '            st.warning(f"Mappa non disponibile: {str(_map_e)[:80]}")'
)
if OLD_MAP_MARKER in src and OLD_MAP_END in src:
    idx_s=src.index(OLD_MAP_MARKER); idx_e=src.index(OLD_MAP_END)+len(OLD_MAP_END)
    src=src[:idx_s]+NEW_MAP+"\n"+src[idx_e:]
    print("  P5  mappa HTML card:       OK")
else:
    print("  P5  mappa HTML card:       SKIP (marker non trovato)")

# ═══ P6: Heatmap 95d + idx3m ══════════════════════════════════════════
n6a=src.count('period="3mo"'); src=src.replace('period="3mo"','period="95d"',1)
n6b=src.count('_cl_h.iloc[-63]-1)*100,2) if len(_cl_h)>=63 else 0')
src=src.replace(
    '_r3m = round((_cl_h.iloc[-1]/_cl_h.iloc[-63]-1)*100,2) if len(_cl_h)>=63 else 0',
    '_idx3m=min(63,len(_cl_h)-1)\n                        _r3m=round((_cl_h.iloc[-1]/_cl_h.iloc[-_idx3m]-1)*100,2) if _idx3m>=2 else 0',1)
print(f"  P6  heatmap 95d+idx3m:     {'OK' if n6a else 'SKIP'}")

# ═══ P7: Ranking settori caption ══════════════════════════════════════
n7=src.count('st.markdown("#### 🏆 Ranking Settori")')
src=src.replace('st.markdown("#### 🏆 Ranking Settori")',
                'st.markdown("#### 🏆 Ranking Settori — ordinati per periodo selezionato")',1)
print(f"  P7  ranking caption:       {'OK' if n7 else 'SKIP'}")

# ═══ P8a: P&L titolo v41d ══════════════════════════════════════════════
n8a=src.count('"📈 P&L Tracker & Alert Engine v41"')
src=src.replace('"📈 P&L Tracker & Alert Engine v41"','"📈 P&L Tracker & Alert Engine v41d"',1)
print(f"  P8a P&L titolo v41d:       {'OK' if n8a else 'SKIP'}")

# ═══ P8b: Pannello notifiche Telegram + Email ══════════════════════════
NOTIF=(
    "\n            # \u2500\u2500 v41d auto-notifica Telegram su FIRED \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "            _tgtv=(st.secrets.get('TELEGRAM_TOKEN','') if hasattr(st,'secrets') else '') or st.session_state.get('tg_token_pnl','')\n"
    "            _cgpv=(st.secrets.get('TELEGRAM_CHAT','')  if hasattr(st,'secrets') else '') or st.session_state.get('tg_chat_pnl','')\n"
    "            if _tgtv and _cgpv and _alerts:\n"
    "                for _fa in [a for a in _alerts.values() if a.get('fired') and not a.get('notified')]:\n"
    "                    try:\n"
    "                        import requests as _rqf\n"
    "                        _rqf.post(f'https://api.telegram.org/bot{_tgtv}/sendMessage',\n"
    "                            json={'chat_id':_cgpv,'text':f\"\U0001f6a8 ALERT v41d\\n{_fa['tkr']} {_fa['type']} {_fa['val']}\"},timeout=5)\n"
    "                        _fa['notified']=True\n"
    "                    except Exception: pass\n"
    "\n        # \u2500\u2500 v41d pannello notifiche \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "        with st.expander('\U0001f4e1 Canali Notifica v41d', expanded=False):\n"
    "            _nc1,_nc2=st.columns(2)\n"
    "            with _nc1:\n"
    "                st.markdown('**Telegram \U0001f916**')\n"
    "                _tgtok=st.text_input('Bot Token',type='password',key='pnl_tg_token',placeholder='123456:ABC-xyz')\n"
    "                _tgcht=st.text_input('Chat ID',key='pnl_tg_chat',placeholder='-100123456789')\n"
    "                if _tgtok: st.session_state['tg_token_pnl']=_tgtok\n"
    "                if _tgcht: st.session_state['tg_chat_pnl']=_tgcht\n"
    "                if st.button('\U0001f4e8 Report P&L Telegram',key='pnl_tg_send'):\n"
    "                    try:\n"
    "                        import requests as _rqt\n"
    "                        _rqt.post(f'https://api.telegram.org/bot{_tgtok}/sendMessage',\n"
    "                            json={'chat_id':_tgcht,'text':'\U0001f4bc Dashboard v41d P&L','parse_mode':'Markdown'},timeout=8)\n"
    "                        st.success('\u2705 Inviato!')\n"
    "                    except Exception as _e: st.error(str(_e))\n"
    "                if st.button('\U0001f514 Test Telegram',key='pnl_tg_test'):\n"
    "                    try:\n"
    "                        import requests as _rqt2\n"
    "                        _rqt2.post(f'https://api.telegram.org/bot{_tgtok}/sendMessage',\n"
    "                            json={'chat_id':_tgcht,'text':'\u2705 Test OK v41d'},timeout=5)\n"
    "                        st.success('\u2705 Test inviato!')\n"
    "                    except Exception as _e2: st.error(str(_e2))\n"
    "            with _nc2:\n"
    "                st.markdown('**Email \U0001f4e7 Gmail**')\n"
    "                _emto=st.text_input('Destinatario',key='pnl_email_to',placeholder='tuo@email.com')\n"
    "                _emfrm=st.text_input('Mittente',key='pnl_email_from',placeholder='bot@gmail.com')\n"
    "                _empwd=st.text_input('App Password',key='pnl_email_pwd',type='password')\n"
    "                if st.button('\U0001f4e7 Report P&L Email',key='pnl_email_send'):\n"
    "                    try:\n"
    "                        import smtplib; from email.mime.text import MIMEText as _MT\n"
    "                        _m=_MT('Dashboard v41d P&L'); _m['Subject']='P&L v41d'; _m['From']=_emfrm; _m['To']=_emto\n"
    "                        with smtplib.SMTP_SSL('smtp.gmail.com',465) as _s: _s.login(_emfrm,_empwd); _s.send_message(_m)\n"
    "                        st.success('\u2705 Email inviata!')\n"
    "                    except Exception as _e3: st.error(str(_e3))\n"
)
MRK_NOTIF='\n    st.markdown("")\n    _wl_col1, _wl_col2 = st.columns([3, 2])'
n8b=src.count(MRK_NOTIF); src=src.replace(MRK_NOTIF, NOTIF+MRK_NOTIF, 1)
print(f"  P8b pannello notifiche:    {'OK' if n8b else 'SKIP'}")

# ═══ P9: Momentum Alerts titolo ═══════════════════════════════════════
n9=src.count('MOMENTUM ALERTS v41c')
src=src.replace('MOMENTUM ALERTS v41c \u2014 Alert Pattern Tecnici in Tempo Reale',
                'MOMENTUM ALERTS v41d \u2014 Ticker \u00b7 Nome \u00b7 Tipo \u00b7 Valore \u00b7 RSI \u00b7 Vol\u00d7 \u00b7 CSS \u00b7 Priorit\u00e0',1)
print(f"  P9  Mom Alerts titolo:     {'OK' if n9 else 'SKIP'}")

# ═══ P10: Correlazioni Home — disabilita duplicato ════════════════════
OLD_CORR='    with st.expander("🔗 Correlazioni Asset — 30 giorni", expanded=False):'
NEW_CORR=(
    '    # v41d: correlazioni spostate sotto Ranking Settori nel tab Settori\n'
    '    with st.expander("🔗 Correlazioni Asset — 30 giorni", expanded=False):\n'
    '        st.info("ℹ️ v41d: le Correlazioni Asset sono ora disponibili nel tab 🏭 Settori, dopo il Ranking Settori.")\n'
    '        if False:  # v41d: disabilitato in Home per evitare duplicato'
)
n10=src.count(OLD_CORR); src=src.replace(OLD_CORR, NEW_CORR, 1)
print(f"  P10 correlazioni home:     {'OK' if n10 else 'SKIP'}")

# ═══ P11: Bar chart settori — disabilita ══════════════════════════════
OLD_BAR='        st.plotly_chart(_fig_sr, use_container_width=True, key="sector_heatmap_v41")'
NEW_BAR=(
    '        # v41d: heatmap gia\u2019 presente nella Home \u2014 bar chart disabilitato\n'
    '        # st.plotly_chart(_fig_sr, use_container_width=True, key="sector_heatmap_v41")  # disabled v41d\n'
    '        st.caption("\U0001f4ca Heatmap settoriale gi\u00e0 disponibile nella Home \u2014 tab \U0001f321\ufe0f Heatmap Settoriale Live.")'
)
n11=src.count(OLD_BAR); src=src.replace(OLD_BAR, NEW_BAR, 1)
print(f"  P11 bar chart disabilitato: {'OK' if n11 else 'SKIP'}")

# ═══ P12: Segnali Rapidi — Nome accanto al ticker Top PRO ════════════
OLD_PRO=(
    "                    st.markdown(\n"
    "                        f\"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' \"\n"
    "                        f\"style='text-decoration:none'>\"\n"
    "                        f\"<span style='font-family:Courier New;color:{_sc};font-weight:bold'>\"\n"
    "                        f\"{_r.get('Ticker','')}</span></a>\"\n"
    "                        f\"<span style='color:#6b7280;font-size:0.72rem'> · CSS {_r.get('CSS','—')} · {_r.get('Stato_Pro','')}</span>\",\n"
    "                        unsafe_allow_html=True)"
)
NEW_PRO=(
    "                    _nome_pr=str(_r.get('Nome',_r.get('Company',_r.get('name','')))).strip()[:22]\n"
    "                    _nome_lbl=f\" <span style='color:#9ca3af;font-size:0.70rem;font-style:italic'>{_nome_pr}</span>\" if _nome_pr else ''\n"
    "                    st.markdown(\n"
    "                        f\"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' \"\n"
    "                        f\"style='text-decoration:none'>\"\n"
    "                        f\"<span style='font-family:Courier New;color:{_sc};font-weight:bold'>\"\n"
    "                        f\"{_r.get('Ticker','')}</span></a>\"\n"
    "                        f\"{_nome_lbl}\"\n"
    "                        f\"<span style='color:#6b7280;font-size:0.72rem'> · CSS {_r.get('CSS','\u2014')} · {_r.get('Stato_Pro','')}</span>\",\n"
    "                        unsafe_allow_html=True)"
)
n12=src.count(OLD_PRO); src=src.replace(OLD_PRO, NEW_PRO, 1)
print(f"  P12 Nome in Top PRO:       {'OK' if n12 else 'SKIP'}")

# ═══ P13: Suggerimenti v41d expander nella Home ════════════════════════
SUGG=(
    "\n    # \u2500\u2500 v41d: Suggerimenti \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "    with st.expander('\U0001f4a1 Suggerimenti v41d \u2014 Novit\u00e0 e roadmap', expanded=False):\n"
    "        st.markdown(\"\"\"\n"
    "**\u2705 Implementato in v41d:**\n"
    "- \U0001f5fa\ufe0f Mappa Calore Globale: card HTML responsive a 3 regioni + macro\n"
    "- \U0001f916 Tab **Modulo 2 AI** dedicato (PRO / CONFLUENCE / Tutti)\n"
    "- \U0001f4e1 Alert Engine: notifiche **Telegram** auto su FIRED + **Email Gmail**\n"
    "- \U0001f4ca Bar chart settori disabilitato (duplicato con Heatmap Live nella Home)\n"
    "- \U0001f517 Correlazioni Asset spostate sotto Ranking Settori nel tab Settori\n"
    "- \U0001f4aa Top PRO/STRONG: **Nome azienda** accanto al ticker\n"
    "- \U0001f4d1 Tab su 2 righe: CSS flex-wrap + JS MutationObserver\n\n"
    "**\U0001f51c Idee per v41e:**\n"
    "- \U0001f514 Alert push via browser (Web Push Notifications)\n"
    "- \U0001f4ca Sparkline miniatura accanto al ticker nella Top PRO/STRONG\n"
    "- \U0001f5c3\ufe0f Export segnali CSV/Excel con 1 click dalla Home\n"
    "- \U0001f504 Auto-refresh Home ogni N minuti con st.rerun() schedulato\n"
    "- \U0001f9e0 AI Analyst: storico analisi per ticker in SQLite\n"
    "- \U0001f4f1 Layout mobile-first con CSS container queries\n"
    "        \"\"\")\n"
)
MRK_SUGG="    st.markdown(\"---\")\n\n    # ── v41c FEATURE 4 — Portfolio P&L"
n13=src.count(MRK_SUGG); src=src.replace(MRK_SUGG, SUGG+MRK_SUGG, 1)
print(f"  P13 Suggerimenti v41d:     {'OK' if n13 else 'SKIP'}")

# ═══ versione globale ══════════════════════════════════════════════════
src=src.replace('v41c','v41d'); src=src.replace('V_41c','V_41d')

# ═══ VERIFICA FINALE ══════════════════════════════════════════════════
checks={
    "flex-wrap wrap":       "flex-wrap:wrap",
    "MutationObserver":     "MutationObserver",
    "tab_ai2 lista":        "Modulo 2 AI",
    "tab_ai2 unpack":       "tab_ai2, tab_opts",
    "with tab_ai2":         "with tab_ai2:",
    "mappa card":           "_map_card_v41d",
    "heatmap 95d":          'period="95d"',
    "heatmap idx3m":        "_idx3m",
    "P&L v41d":             "Alert Engine v41d",
    "tg_token_pnl":         "pnl_tg_token",
    "smtp.gmail.com":       "smtp.gmail.com",
    "correlazioni off":     "v41d: disabilitato in Home",
    "bar chart off":        "v41d: heatmap gia",
    "Nome in Top PRO":      "_nome_lbl",
    "Suggerimenti v41d":    "Suggerimenti v41d",
    "Mom Alerts v41d":      "MOMENTUM ALERTS v41d",
}
failed=[]
print("\n── Verifica ──")
for lbl,marker in checks.items():
    ok=marker in src
    print(f"  {'✅' if ok else '❌'} {lbl}")
    if not ok: failed.append(lbl)

if failed:
    print(f"\n⛔ {len(failed)} patch non applicate: {failed}")
    sys.exit(1)

with open(DST,"w",encoding="utf-8") as f:
    f.write(src)
print(f"\n✅ {DST} scritto: {len(src):,} caratteri")

