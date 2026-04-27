#!/usr/bin/env python3
"""patch_v41f.py - Dashboard_pro_V_41e.py -> Dashboard_pro_V_41f.py
Fix: SVG Mappa Calore Globale responsiva (preserveAspectRatio + div wrapper)
Approccio robusto: cerca marker corti e univoci via .find()
"""
import sys, os

SRC = "Dashboard_pro_V_41e.py"
DST = "Dashboard_pro_V_41f.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print("FILE " + SRC + ": " + str(len(src)) + " chars")

# ══ P1: SVG mappa — trova f"<svg ... nel blocco _map_svg e sostituisce ═════
MAP_SVG_DEF = "_map_svg = ("
ix_def = src.find(MAP_SVG_DEF)
if ix_def == -1:
    print("P1 SVG header: SKIP (_map_svg non trovato)")
else:
    # Trova il tag f"<svg subito dopo la definizione
    ix_ftag = src.find('f"<svg', ix_def, ix_def + 2000)
    if ix_ftag == -1:
        ix_ftag = src.find("f'<svg", ix_def, ix_def + 2000)
    if ix_ftag == -1:
        print("P1 SVG header: SKIP (f<svg non trovato)")
    else:
        # Fine del blocco: cerca 2a2e39 (colore bordo) entro 400 chars dal tag
        ix_border = src.find("2a2e39", ix_ftag, ix_ftag + 400)
        if ix_border == -1:
            print("P1 SVG header: SKIP (border color non trovato)")
        else:
            # Vai fino al primo > dopo il border color
            ix_close = src.find(">", ix_border)
            # Includi anche il carattere successivo (virgolette chiusura f-string)
            ix_ftag_end = ix_close + 2  # include > e il " o '
            old_full = src[ix_ftag: ix_ftag_end]
            new_full = (
                "f\"<svg width='100%' height='auto' viewBox='0 0 {_svg_w} {_svg_h}' \"\n"
                "                    f\"preserveAspectRatio='xMidYMid meet' \"\n"
                "                    f\"xmlns='http://www.w3.org/2000/svg' \"\n"
                "                    f\"style='display:block;width:100%;max-width:100%;"
                "background:#131722;border-radius:8px;border:1px solid #2a2e39'>\""
            )
            src = src[:ix_ftag] + new_full + src[ix_ftag_end:]
            print("P1 SVG header responsivo: OK")

# ══ P2: wrap st.markdown(_map_svg) in div overflow ══════════════════════════
MK = "st.markdown(_map_svg, unsafe_allow_html=True)"
ix_mk = src.find(MK)
if ix_mk == -1:
    print("P2 SVG div wrapper: SKIP")
else:
    ix_line_start = src.rfind("\n", 0, ix_mk) + 1
    indent = ""
    for ch in src[ix_line_start:]:
        if ch == " ":
            indent += " "
        else:
            break
    new_mk = (
        indent + "_map_svg_wrap = (\n"
        + indent + "    \"<div style='width:100%;overflow-x:auto;overflow-y:hidden'>\"\n"
        + indent + "    + _map_svg +\n"
        + indent + "    \"</div>\"\n"
        + indent + ")\n"
        + indent + "st.markdown(_map_svg_wrap, unsafe_allow_html=True)"
    )
    src = src[:ix_line_start] + new_mk + src[ix_mk + len(MK):]
    print("P2 SVG div wrapper: OK")

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

