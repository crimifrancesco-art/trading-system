#!/usr/bin/env python3
"""patch_v41e.py - Dashboard_pro_V_41d.py -> Dashboard_pro_V_41e.py"""
import sys, os

SRC = "Dashboard_pro_V_41d.py"
DST = "Dashboard_pro_V_41e.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print("FILE " + SRC + ": " + str(len(src)) + " chars")

q  = chr(39)   # apostrofo '
dq = chr(34)   # virgolette "
BS = chr(92)   # backslash

# ══ P1: STRONG banner - Nome (index-based) ══════════════════════════════════
M1 = '_banner_tickers_html = "  ".join('
M2 = "for t in _strong_list[:12]\n                )"
ix1 = src.find(M1)
ix2 = src.find(M2, ix1) + len(M2) if ix1 != -1 else -1
if ix1 != -1 and ix2 > ix1:
    NB = (
        "_df_strong_map = _df_ep_banner.set_index(" + dq + "Ticker" + dq + ")[" + dq + "Nome" + dq + "].to_dict()"
        " if " + dq + "Nome" + dq + " in _df_ep_banner.columns else {}\n"
        "                _banner_tickers_html = " + dq + "  " + dq + ".join(\n"
        "                    f" + dq + "<a href=" + q + "https://it.tradingview.com/chart/?symbol={t.replace(" + q + ".MI" + q + "," + q + "%3AMI" + q + ")}'" + " target=" + q + "_blank" + q + " style=" + q + "color:#ffd700;font-family:Courier New;font-weight:bold;text-decoration:none;font-size:0.88rem" + q + ">{t}</a>" + dq + "\n"
        "                    f" + dq + "<span style=" + q + "color:#9ca3af;font-size:0.70rem;font-style:italic" + q + ">{str(_df_strong_map.get(t," + q + q + "))[:18]}</span>" + dq + "\n"
        "                    for t in _strong_list[:12]\n"
        "                )"
    )
    src = src[:ix1] + NB + src[ix2:]
    print("P1 STRONG Nome: OK")
else:
    print("P1 STRONG Nome: SKIP")

# ══ P2: Correlazioni (opzionale) ════════════════════════════════════════════
OLD2 = (
    "    # v41d: correlazioni disponibili nel tab Settori\n"
    "    with st.expander(" + dq + "\U0001f517 Correlazioni Asset \u2014 30 giorni" + dq + ", expanded=False):\n"
    "        st.info(" + dq + "\u2139\ufe0f v41d: Correlazioni disponibili nel tab \U0001f3ed Settori \u2192 dopo Ranking Settori." + dq + ")\n"
    "        pass  # v41d: contenuto originale disabilitato in Home\n"
)
NEW2 = "    # v41e: correlazioni -> tab Settori\n"
n2 = src.count(OLD2); src = src.replace(OLD2, NEW2, 1)
print("P2 Correlazioni: " + ("OK" if n2 else "SKIP"))

# ══ P3: NEWS ticker link TV + Nome ══════════════════════════════════════════
OLD3 = (
    "        _c1.markdown(f" + dq + "<span style=" + q + "font-family:Courier New;color:#00ff88;font-weight:bold" + q + ">{n[" + q + "Ticker" + q + "]}</span>" + dq + ",unsafe_allow_html=True)\n"
    "        _c2.markdown(f" + dq + "<span style=" + q + "color:{_sc2};font-size:0.78rem" + q + ">{n[" + q + "Sentiment" + q + "]}</span>" + dq + ",unsafe_allow_html=True)\n"
    "        _c3.markdown(f" + dq + "<a href=" + q + "{n[" + q + "Link" + q + "]}" + q + " target=" + q + "_blank" + q + " style=" + q + "color:#b2b5be;font-size:0.82rem;text-decoration:none" + q + ">{n[" + q + "Titolo" + q + "]}</a> <span style=" + q + "color:#374151;font-size:0.70rem" + q + ">{n[" + q + "Data" + q + "]}</span>" + dq + ",unsafe_allow_html=True)"
)
NEW3 = (
    "        _tv_sym_n = str(n[" + q + "Ticker" + q + "]).replace(" + q + ".MI" + q + ", " + q + "%3AMI" + q + ")\n"
    "        _nome_n = str(n.get(" + q + "Nome" + q + ", " + q + q + ")).strip()[:22]\n"
    "        _nome_n_html = (f" + dq + " <span style=" + q + "color:#6b7280;font-size:0.70rem;font-style:italic" + q + ">{_nome_n}</span>" + dq + " if _nome_n else " + q + q + ")\n"
    "        _c1.markdown(\n"
    "            f" + dq + "<a href=" + q + "https://it.tradingview.com/chart/?symbol={_tv_sym_n}" + q + " target=" + q + "_blank" + q + " style=" + q + "text-decoration:none" + q + ">" + dq + "\n"
    "            f" + dq + "<span style=" + q + "font-family:Courier New;color:#00ff88;font-weight:bold" + q + ">{n[" + q + "Ticker" + q + "]}</span></a>" + dq + "\n"
    "            f" + dq + "{_nome_n_html}" + dq + ", unsafe_allow_html=True)\n"
    "        _c2.markdown(f" + dq + "<span style=" + q + "color:{_sc2};font-size:0.78rem" + q + ">{n[" + q + "Sentiment" + q + "]}</span>" + dq + ",unsafe_allow_html=True)\n"
    "        _c3.markdown(f" + dq + "<a href=" + q + "{n[" + q + "Link" + q + "]}" + q + " target=" + q + "_blank" + q + " style=" + q + "color:#b2b5be;font-size:0.82rem;text-decoration:none" + q + ">{n[" + q + "Titolo" + q + "]}</a> <span style=" + q + "color:#374151;font-size:0.70rem" + q + ">{n[" + q + "Data" + q + "]}</span>" + dq + ",unsafe_allow_html=True)"
)
n3 = src.count(OLD3); src = src.replace(OLD3, NEW3, 1)
print("P3 NEWS link+Nome: " + ("OK" if n3 else "SKIP"))

# ══ P4: Top EARLY Nome ══════════════════════════════════════════════════════
OLD4 = (
    "                    st.markdown(\n"
    "                        f" + dq + "<a href=" + q + "https://it.tradingview.com/chart/?symbol={_tv}" + q + " target=" + q + "_blank" + q + " " + dq + "\n"
    "                        f" + dq + "style=" + q + "text-decoration:none" + q + ">" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "font-family:Courier New;color:#60a5fa;font-weight:bold" + q + ">" + dq + "\n"
    "                        f" + dq + "{_r.get(" + q + "Ticker" + q + "," + q + q + ")}</span></a>" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "color:#6b7280;font-size:0.72rem" + q + "> \u00b7 E:{_r.get(" + q + "Early_Score" + q + "," + q + "\u2014" + q + ")} \u00b7 RSI {_r.get(" + q + "RSI" + q + "," + q + "\u2014" + q + ")}</span>" + dq + ",\n"
    "                        unsafe_allow_html=True)"
)
NEW4 = (
    "                    _nome_ea = str(_r.get(" + q + "Nome" + q + ", _r.get(" + q + "Company" + q + ", " + q + q + "))).strip()[:22]\n"
    "                    _nome_ea_lbl = (f" + dq + " <span style=" + q + "color:#9ca3af;font-size:0.70rem;font-style:italic" + q + ">{_nome_ea}</span>" + dq + " if _nome_ea else " + q + q + ")\n"
    "                    st.markdown(\n"
    "                        f" + dq + "<a href=" + q + "https://it.tradingview.com/chart/?symbol={_tv}" + q + " target=" + q + "_blank" + q + " " + dq + "\n"
    "                        f" + dq + "style=" + q + "text-decoration:none" + q + ">" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "font-family:Courier New;color:#60a5fa;font-weight:bold" + q + ">" + dq + "\n"
    "                        f" + dq + "{_r.get(" + q + "Ticker" + q + "," + q + q + ")}</span></a>" + dq + "\n"
    "                        f" + dq + "{_nome_ea_lbl}" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "color:#6b7280;font-size:0.72rem" + q + "> \u00b7 E:{_r.get(" + q + "Early_Score" + q + "," + q + "\u2014" + q + ")} \u00b7 RSI {_r.get(" + q + "RSI" + q + "," + q + "\u2014" + q + ")}</span>" + dq + ",\n"
    "                        unsafe_allow_html=True)"
)
n4 = src.count(OLD4); src = src.replace(OLD4, NEW4, 1)
print("P4 EARLY Nome: " + ("OK" if n4 else "SKIP"))

# ══ P5: Top REA-HOT Nome ════════════════════════════════════════════════════
OLD5 = (
    "                    st.markdown(\n"
    "                        f" + dq + "<a href=" + q + "https://it.tradingview.com/chart/?symbol={_tv}" + q + " target=" + q + "_blank" + q + " " + dq + "\n"
    "                        f" + dq + "style=" + q + "text-decoration:none" + q + ">" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "font-family:Courier New;color:#f97316;font-weight:bold" + q + ">" + dq + "\n"
    "                        f" + dq + "{_r.get(" + q + "Ticker" + q + "," + q + q + ")}</span></a>" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "color:#6b7280;font-size:0.72rem" + q + "> \u00b7 Vol\u00d7{_vr}</span>" + dq + ",\n"
    "                        unsafe_allow_html=True)"
)
NEW5 = (
    "                    _nome_hot = str(_r.get(" + q + "Nome" + q + ", _r.get(" + q + "Company" + q + ", " + q + q + "))).strip()[:22]\n"
    "                    _nome_hot_lbl = (f" + dq + " <span style=" + q + "color:#9ca3af;font-size:0.70rem;font-style:italic" + q + ">{_nome_hot}</span>" + dq + " if _nome_hot else " + q + q + ")\n"
    "                    st.markdown(\n"
    "                        f" + dq + "<a href=" + q + "https://it.tradingview.com/chart/?symbol={_tv}" + q + " target=" + q + "_blank" + q + " " + dq + "\n"
    "                        f" + dq + "style=" + q + "text-decoration:none" + q + ">" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "font-family:Courier New;color:#f97316;font-weight:bold" + q + ">" + dq + "\n"
    "                        f" + dq + "{_r.get(" + q + "Ticker" + q + "," + q + q + ")}</span></a>" + dq + "\n"
    "                        f" + dq + "{_nome_hot_lbl}" + dq + "\n"
    "                        f" + dq + "<span style=" + q + "color:#6b7280;font-size:0.72rem" + q + "> \u00b7 Vol\u00d7{_vr}</span>" + dq + ",\n"
    "                        unsafe_allow_html=True)"
)
n5 = src.count(OLD5); src = src.replace(OLD5, NEW5, 1)
print("P5 REA-HOT Nome: " + ("OK" if n5 else "SKIP"))

# ══ P6: Crisis NaN fix ══════════════════════════════════════════════════════
OLD6 = (
    "        go_r = gb_r.build()\n"
    "        try:\n"
    "            AgGrid(df_riepilogo,"
)
NEW6 = (
    "        import math as _math_r\n"
    "        _safe_r = []\n"
    "        for _rrec in df_riepilogo.to_dict(orient=" + q + "records" + q + "):\n"
    "            _safe_r.append({_k: (None if isinstance(_v, float) and\n"
    "                (_math_r.isnan(_v) or _math_r.isinf(_v)) else _v)\n"
    "                for _k, _v in _rrec.items()})\n"
    "        df_riepilogo = pd.DataFrame(_safe_r)\n"
    "        for _col_r in df_riepilogo.select_dtypes(include=[" + q + "object" + q + "]).columns:\n"
    "            df_riepilogo[_col_r] = df_riepilogo[_col_r].fillna(" + q + "\u2014" + q + ").replace({" + q + "nan" + q + ":" + q + "\u2014" + q + "," + q + "None" + q + ":" + q + "\u2014" + q + "," + q + "NaN" + q + ":" + q + "\u2014" + q + "})\n"
    "        go_r = gb_r.build()\n"
    "        try:\n"
    "            AgGrid(df_riepilogo,"
)
n6 = src.count(OLD6); src = src.replace(OLD6, NEW6, 1)
print("P6 Crisis NaN: " + ("OK" if n6 else "SKIP"))

# ══ versione ════════════════════════════════════════════════════════════════
src = src.replace("v41d", "v41e"); src = src.replace("V_41d", "V_41e")

# ══ verifica (P2 opzionale) ═════════════════════════════════════════════════
checks = {"STRONG Nome": "_df_strong_map", "NEWS TV": "_tv_sym_n",
          "EARLY Nome": "_nome_ea_lbl", "HOT Nome": "_nome_hot_lbl", "Crisis NaN": "_safe_r"}
failed = []
print("\n-- Verifica --")
for lbl, marker in checks.items():
    ok = marker in src
    print("  " + ("OK" if ok else "FAIL") + " " + lbl)
    if not ok: failed.append(lbl)

if failed:
    print("FAILED: " + str(failed)); sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print("\nOK " + DST + " " + str(len(src)) + " chars")

