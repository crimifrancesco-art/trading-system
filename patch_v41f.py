#!/usr/bin/env python3
"""patch_v41f.py - Dashboard_pro_V_41e.py -> Dashboard_pro_V_41f.py
Fix: Mappa Calore Globale responsive (st.columns fisso -> HTML flex puro)
"""
import sys, os

SRC = "Dashboard_pro_V_41e.py"
DST = "Dashboard_pro_V_41f.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print("FILE " + SRC + ": " + str(len(src)) + " chars")

q  = chr(39)
dq = chr(34)

# ══ P1: sostituisce _rc=st.columns([1,1.5,1]) + loop con HTML flex puro ════
OLD1 = (
    "                _rc=st.columns([1,1.5,1])\n"
    "                for _ci,(_rn,_ra) in enumerate(_map_regions.items()):\n"
    "                    with _rc[_ci]:\n"
    "                        st.markdown(f\"<div style='color:#50c4e0;font-size:.70rem;font-weight:bold;text-align:center;\"\n"
    "                                    f\"letter-spacing:2px;border-bottom:1px solid #2a2e39;padding-bottom:4px;\"\n"
    "                                    f\"margin-bottom:6px'>{_rn}</div>\",unsafe_allow_html=True)\n"
    "                        st.markdown(\"<div style='display:flex;flex-wrap:wrap;gap:4px;justify-content:center'>\"\n"
    "                                    + \"\".join(_map_card_v41e(l,s) for l,s in _ra) + \"</div>\",unsafe_allow_html=True)\n"
    "                st.markdown(\"<div style='margin-top:8px;display:flex;gap:5px;justify-content:center;flex-wrap:wrap'>\"\n"
    "                            + \"\".join(_map_card_v41e(l,s) for l,s in _map_macro) + \"</div>\",unsafe_allow_html=True)\n"
    "                st.caption(\"🟢 rialzo · 🔴 ribasso · intensità = % variazione\")"
)
NEW1 = (
    "                # v41f: layout HTML flex puro - nessuna colonna Streamlit fissa\n"
    "                _map_html = (\n"
    "                    \"<div style='display:flex;flex-wrap:wrap;gap:10px;width:100%;box-sizing:border-box'>\"\n"
    "                )\n"
    "                for _rn, _ra in _map_regions.items():\n"
    "                    _map_html += (\n"
    "                        \"<div style='flex:1;min-width:180px;box-sizing:border-box'>\"\n"
    "                        f\"<div style='color:#50c4e0;font-size:.70rem;font-weight:bold;text-align:center;\"\n"
    "                        f\"letter-spacing:2px;border-bottom:1px solid #2a2e39;padding-bottom:4px;\"\n"
    "                        f\"margin-bottom:6px'>{_rn}</div>\"\n"
    "                        \"<div style='display:flex;flex-wrap:wrap;gap:4px;justify-content:center'>\"\n"
    "                        + \"\".join(_map_card_v41e(l,s) for l,s in _ra) +\n"
    "                        \"</div></div>\"\n"
    "                    )\n"
    "                _map_html += \"</div>\"\n"
    "                _map_html += (\n"
    "                    \"<div style='margin-top:8px;display:flex;gap:5px;justify-content:center;flex-wrap:wrap'>\"\n"
    "                    + \"\".join(_map_card_v41e(l,s) for l,s in _map_macro) +\n"
    "                    \"</div>\"\n"
    "                )\n"
    "                st.markdown(_map_html, unsafe_allow_html=True)\n"
    "                st.caption(\"🟢 rialzo · 🔴 ribasso · intensità = % variazione\")"
)
n1 = src.count(OLD1)
src = src.replace(OLD1, NEW1, 1)
print("P1 Mappa flex HTML: " + ("OK" if n1 else "SKIP"))

# ══ versione ════════════════════════════════════════════════════════════════
src = src.replace("v41e", "v41f")
src = src.replace("V_41e", "V_41f")

# ══ verifica ════════════════════════════════════════════════════════════════
checks = {
    "flex puro":    "_map_html",
    "min-width:180": "min-width:180px",
}
failed = []
print("\n-- Verifica --")
for lbl, marker in checks.items():
    ok = marker in src
    print("  " + ("OK" if ok else "FAIL") + " " + lbl)
    if not ok:
        failed.append(lbl)

if failed:
    print("FAILED: " + str(failed)); sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print("\nOK " + DST + " " + str(len(src)) + " chars")

