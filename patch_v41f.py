#!/usr/bin/env python3
"""patch_v41f.py - Dashboard_pro_V_41e.py -> Dashboard_pro_V_41f.py
Fix: SVG Mappa Calore Globale responsiva
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

# ══ P1: SVG mappa — index-based, cerca i 3 marker univoci e sostituisce ════
M_START = "_map_svg = (\n"
M_SVG1  = "f\"<svg width='100%' viewBox='0 0 {_svg_w} {_svg_h}' \"\n"
M_SVG2  = "f\"xmlns='http://www.w3.org/2000/svg' \"\n"
M_SVG3  = "f\"style='background:#131722;border-radius:8px;border:1px solid #2a2e39'>\""

# Trova la sequenza esatta (cerchiamo M_SVG1 dentro il blocco _map_svg)
ix = src.find(M_SVG1)
if ix != -1:
    # trova fine di M_SVG3
    ix3 = src.find(M_SVG3, ix)
    ix3_end = ix3 + len(M_SVG3)
    OLD_SVG_HEADER = src[ix:ix3_end]
    NEW_SVG_HEADER = (
        "f\"<svg width='100%' height='auto' viewBox='0 0 {_svg_w} {_svg_h}' \"\n"
        "                    f\"preserveAspectRatio='xMidYMid meet' \"\n"
        "                    f\"xmlns='http://www.w3.org/2000/svg' \"\n"
        "                    f\"style='display:block;width:100%;max-width:100%;background:#131722;border-radius:8px;border:1px solid #2a2e39'>\""
    )
    src = src[:ix] + NEW_SVG_HEADER + src[ix3_end:]
    print("P1 SVG header responsivo: OK")
else:
    print("P1 SVG header responsivo: SKIP")

# ══ P2: wrap st.markdown(_map_svg) in div overflow-x:auto ══════════════════
OLD2 = "                st.markdown(_map_svg, unsafe_allow_html=True)\n"
NEW2 = (
    "                _map_svg_wrap = (\n"
    "                    " + dq + "<div style=" + q + "width:100%;overflow-x:auto;overflow-y:hidden" + q + ">" + dq + "\n"
    "                    + _map_svg +\n"
    "                    " + dq + "</div>" + dq + "\n"
    "                )\n"
    "                st.markdown(_map_svg_wrap, unsafe_allow_html=True)\n"
)
n2 = src.count(OLD2)
src = src.replace(OLD2, NEW2, 1)
print("P2 SVG div wrapper: " + ("OK" if n2 else "SKIP"))

# ══ versione ════════════════════════════════════════════════════════════════
src = src.replace("v41e", "v41f")
src = src.replace("V_41e", "V_41f")

# ══ verifica ════════════════════════════════════════════════════════════════
checks = {
    "preserveAspectRatio": "preserveAspectRatio",
    "div wrapper":         "_map_svg_wrap",
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

