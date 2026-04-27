#!/usr/bin/env python3
"""patch_v41f.py - Dashboard_pro_V_41e.py -> Dashboard_pro_V_41f.py
Fix: SVG Mappa Calore Globale responsiva (preserveAspectRatio + div wrapper)
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

# ══ P1: SVG mappa — aggiungi preserveAspectRatio + height auto ══════════════
OLD1 = (
    "                _map_svg = (\n"
    "                    f" + dq + "<svg width=" + q + "100%" + q + " viewBox=" + q + "0 0 {_svg_w} {_svg_h}" + q + " " + dq + "\n"
    "                    f" + dq + "xmlns=" + q + "http://www.w3.org/2000/svg" + q + " " + dq + "\n"
    "                    f" + dq + "style=" + q + "background:#131722;border-radius:8px;border:1px solid #2a2e39" + q + ">" + dq + "\n"
)
NEW1 = (
    "                _map_svg = (\n"
    "                    f" + dq + "<svg width=" + q + "100%" + q + " height=" + q + "auto" + q + " viewBox=" + q + "0 0 {_svg_w} {_svg_h}" + q + " " + dq + "\n"
    "                    f" + dq + "preserveAspectRatio=" + q + "xMidYMid meet" + q + " " + dq + "\n"
    "                    f" + dq + "xmlns=" + q + "http://www.w3.org/2000/svg" + q + " " + dq + "\n"
    "                    f" + dq + "style=" + q + "display:block;width:100%;max-width:100%;background:#131722;border-radius:8px;border:1px solid #2a2e39" + q + ">" + dq + "\n"
)
n1 = src.count(OLD1)
src = src.replace(OLD1, NEW1, 1)
print("P1 SVG header responsivo: " + ("OK" if n1 else "SKIP"))

# ══ P2: wrap st.markdown(_map_svg) in div overflow-x:auto ══════════════════
OLD2 = "                st.markdown(_map_svg, unsafe_allow_html=True)\n"
NEW2 = (
    "                _map_svg_wrap = (\n"
    "                    " + dq + "<div style=" + q + "width:100%;overflow-x:auto;overflow-y:hidden;" + q + ">" + dq + "\n"
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
