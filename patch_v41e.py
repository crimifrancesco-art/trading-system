#!/usr/bin/env python3
"""
patch_v41e.py — Dashboard_pro_V_41d.py → Dashboard_pro_V_41e.py
Fix: STRONG Nome, Correlazioni doppie, NEWS link+Nome,
     Top EARLY Nome, Top REA-HOT Nome, Crisis JSON NaN
NON importa streamlit — solo trasformazioni di testo
"""
import sys, os

SRC = "Dashboard_pro_V_41d.py"
DST = "Dashboard_pro_V_41e.py"

if not os.path.exists(SRC):
    print(f"⛔ '{SRC}' non trovato."); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print(f"📂 {SRC}: {len(src):,} caratteri")

# ═══ P1: STRONG banner — Nome azienda accanto al ticker ════════════════
OLD_STRONG = (
    '                _banner_tickers_html = "  ".join(\n'
    '                    f"<a href=\'https://it.tradingview.com/chart/?symbol={t.replace(\\".MI\\",\\"%3AMI\\")}\'"\n'
    '                    f" target=\'_blank\' style=\'color:#ffd700;font-family:Courier New;"\n'
    '                    f"font-weight:bold;text-decoration:none;font-size:0.88rem\'>{t}</a>"\n'
    '                    for t in _strong_list[:12]\n'
    '                )'
)
NEW_STRONG = (
    '                _df_strong_map = _df_ep_banner.set_index("Ticker")["Nome"].to_dict() if "Nome" in _df_ep_banner.columns else {}\n'
    '                _banner_tickers_html = "  ".join(\n'
    '                    f"<a href=\'https://it.tradingview.com/chart/?symbol={t.replace(\\".MI\\",\\"%3AMI\\")}\'"\n'
    '                    f" target=\'_blank\' style=\'color:#ffd700;font-family:Courier New;"\n'
    '                    f"font-weight:bold;text-decoration:none;font-size:0.88rem\'>{t}</a>"\n'
    '                    f"<span style=\'color:#9ca3af;font-size:0.70rem;font-style:italic\'>"
    '                    f" {str(_df_strong_map.get(t, \'\'))[:18]}</span>"\n'
    '                    for t in _strong_list[:12]\n'
    '                )'
)
# ═══ P2: Correlazioni — rimuovi blocco duplicato con pass ══════════════
OLD_CORR2 = (
    '    # v41d: correlazioni disponibili nel tab Settori\n'
    '    with st.expander("🔗 Correlazioni Asset — 30 giorni", expanded=False):\n'
    '        st.info("\u2139\ufe0f v41d: Correlazioni disponibili nel tab \U0001f3ed Settori \u2192 dopo Ranking Settori.")\n'
    '        pass  # v41d: contenuto originale disabilitato in Home\n'
)
NEW_CORR2 = '    # v41e: correlazioni rimosse dalla Home \u2014 disponibili nel tab Settori\n'
n2 = src.count(OLD_CORR2); src = src.replace(OLD_CORR2, NEW_CORR2, 1)
print(f"  P2  Correlazioni doppia:   {'OK' if n2 else 'SKIP (non presente, OK)'}")

# ═══ P3: NEWS — ticker linkabile TradingView Italia + Nome ════════════
OLD_NEWS = (
    "        _c1.markdown(f\"<span style='font-family:Courier New;color:#00ff88;font-weight:bold'>{n['Ticker']}</span>\",unsafe_allow_html=True)\n"
    "        _c2.markdown(f\"<span style='color:{_sc2};font-size:0.78rem'>{n['Sentiment']}</span>\",unsafe_allow_html=True)\n"
    "        _c3.markdown(f\"<a href='{n['Link']}' target='_blank' style='color:#b2b5be;font-size:0.82rem;text-decoration:none'>{n['Titolo']}</a> <span style='color:#374151;font-size:0.70rem'>{n['Data']}</span>\",unsafe_allow_html=True)"
)
NEW_NEWS = (
    "        _tv_sym_n = str(n['Ticker']).replace('.MI', '%3AMI')\n"
    "        _nome_n = str(n.get('Nome', '')).strip()[:22]\n"
    "        _nome_n_html = f\" <span style='color:#6b7280;font-size:0.70rem;font-style:italic'>{_nome_n}</span>\" if _nome_n else ''\n"
    "        _c1.markdown(\n"
    "            f\"<a href='https://it.tradingview.com/chart/?symbol={_tv_sym_n}' target='_blank' style='text-decoration:none'>\"\n"
    "            f\"<span style='font-family:Courier New;color:#00ff88;font-weight:bold'>{n['Ticker']}</span></a>\"\n"
    "            f\"{_nome_n_html}\", unsafe_allow_html=True)\n"
    "        _c2.markdown(f\"<span style='color:{_sc2};font-size:0.78rem'>{n['Sentiment']}</span>\",unsafe_allow_html=True)\n"
    "        _c3.markdown(f\"<a href='{n['Link']}' target='_blank' style='color:#b2b5be;font-size:0.82rem;text-decoration:none'>{n['Titolo']}</a> <span style='color:#374151;font-size:0.70rem'>{n['Data']}</span>\",unsafe_allow_html=True)"
)
n3 = src.count(OLD_NEWS); src = src.replace(OLD_NEWS, NEW_NEWS, 1)
print(f"  P3  NEWS link+Nome:        {'OK' if n3 else 'SKIP'}")

# ═══ P4: Top EARLY — Nome azienda ══════════════════════════════════════
OLD_EARLY = (
    "                    st.markdown(\n"
    "                        f\"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' \"\n"
    "                        f\"style='text-decoration:none'>\"\n"
    "                        f\"<span style='font-family:Courier New;color:#60a5fa;font-weight:bold'>\"\n"
    "                        f\"{_r.get('Ticker','')}</span></a>\"\n"
    "                        f\"<span style='color:#6b7280;font-size:0.72rem'> \u00b7 E:{_r.get('Early_Score','\u2014')} \u00b7 RSI {_r.get('RSI','\u2014')}</span>\",\n"
    "                        unsafe_allow_html=True)"
)
NEW_EARLY = (
    "                    _nome_ea = str(_r.get('Nome', _r.get('Company', ''))).strip()[:22]\n"
    "                    _nome_ea_lbl = f\" <span style='color:#9ca3af;font-size:0.70rem;font-style:italic'>{_nome_ea}</span>\" if _nome_ea else ''\n"
    "                    st.markdown(\n"
    "                        f\"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' \"\n"
    "                        f\"style='text-decoration:none'>\"\n"
    "                        f\"<span style='font-family:Courier New;color:#60a5fa;font-weight:bold'>\"\n"
    "                        f\"{_r.get('Ticker','')}</span></a>\"\n"
    "                        f\"{_nome_ea_lbl}\"\n"
    "                        f\"<span style='color:#6b7280;font-size:0.72rem'> \u00b7 E:{_r.get('Early_Score','\u2014')} \u00b7 RSI {_r.get('RSI','\u2014')}</span>\",\n"
    "                        unsafe_allow_html=True)"
)
n4 = src.count(OLD_EARLY); src = src.replace(OLD_EARLY, NEW_EARLY, 1)
print(f"  P4  EARLY Nome:            {'OK' if n4 else 'SKIP'}")

# ═══ P5: Top REA-HOT — Nome azienda ════════════════════════════════════
OLD_HOT = (
    "                    st.markdown(\n"
    "                        f\"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' \"\n"
    "                        f\"style='text-decoration:none'>\"\n"
    "                        f\"<span style='font-family:Courier New;color:#f97316;font-weight:bold'>\"\n"
    "                        f\"{_r.get('Ticker','')}</span></a>\"\n"
    "                        f\"<span style='color:#6b7280;font-size:0.72rem'> \u00b7 Vol\u00d7{_vr}</span>\",\n"
    "                        unsafe_allow_html=True)"
)
NEW_HOT = (
    "                    _nome_hot = str(_r.get('Nome', _r.get('Company', ''))).strip()[:22]\n"
    "                    _nome_hot_lbl = f\" <span style='color:#9ca3af;font-size:0.70rem;font-style:italic'>{_nome_hot}</span>\" if _nome_hot else ''\n"
    "                    st.markdown(\n"
    "                        f\"<a href='https://it.tradingview.com/chart/?symbol={_tv}' target='_blank' \"\n"
    "                        f\"style='text-decoration:none'>\"\n"
    "                        f\"<span style='font-family:Courier New;color:#f97316;font-weight:bold'>\"\n"
    "                        f\"{_r.get('Ticker','')}</span></a>\"\n"
    "                        f\"{_nome_hot_lbl}\"\n"
    "                        f\"<span style='color:#6b7280;font-size:0.72rem'> \u00b7 Vol\u00d7{_vr}</span>\",\n"
    "                        unsafe_allow_html=True)"
)
n5 = src.count(OLD_HOT); src = src.replace(OLD_HOT, NEW_HOT, 1)
print(f"  P5  REA-HOT Nome:          {'OK' if n5 else 'SKIP'}")

# ═══ P6: Crisis RIEPILOGO TECNICO — sanifica NaN prima di AgGrid ═══════
OLD_AGGRID_R = (
    '        go_r = gb_r.build()\n'
    '        try:\n'
    '            AgGrid(df_riepilogo,'
)
NEW_AGGRID_R = (
    '        # v41e: sanifica NaN/inf → None prima di AgGrid (fix JSON parse error)\n'
    '        import math as _math_r\n'
    '        _safe_r = []\n'
    '        for _rrec in df_riepilogo.to_dict(orient="records"):\n'
    '            _safe_r.append({_k: (None if isinstance(_v, float) and (_math_r.isnan(_v) or _math_r.isinf(_v)) else _v)\n'
    '                            for _k, _v in _rrec.items()})\n'
    '        df_riepilogo = pd.DataFrame(_safe_r)\n'
    '        for _col_r in df_riepilogo.select_dtypes(include=["object"]).columns:\n'
    '            df_riepilogo[_col_r] = df_riepilogo[_col_r].fillna("\u2014").replace({"nan":"\u2014","None":"\u2014","NaN":"\u2014"})\n'
    '        go_r = gb_r.build()\n'
    '        try:\n'
    '            AgGrid(df_riepilogo,'
)
n6 = src.count(OLD_AGGRID_R); src = src.replace(OLD_AGGRID_R, NEW_AGGRID_R, 1)
print(f"  P6  Crisis JSON NaN:       {'OK' if n6 else 'SKIP'}")

# ═══ versione globale ══════════════════════════════════════════════════
src = src.replace('v41d', 'v41e')
src = src.replace('V_41d', 'V_41e')

# ═══ VERIFICA FINALE ══════════════════════════════════════════════════
checks = {
    "STRONG Nome banner": "_df_strong_map",
    "NEWS link TV":       "_tv_sym_n",
    "EARLY Nome":         "_nome_ea_lbl",
    "HOT Nome":           "_nome_hot_lbl",
    "Crisis NaN fix":     "_safe_r",
}
}
failed = []
print("\n\u2500\u2500 Verifica \u2500\u2500")
for lbl, marker in checks.items():
    ok = marker in src
    print(f"  {'✅' if ok else '❌'} {lbl}")
    if not ok: failed.append(lbl)

if failed:
    print(f"\n⛔ {len(failed)} patch fallite: {failed}")
    sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print(f"\n✅ {DST} scritto: {len(src):,} caratteri")

