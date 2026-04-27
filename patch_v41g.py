#!/usr/bin/env python3
"""patch_v41g.py  Dashboard_pro_V_41f.py → Dashboard_pro_V_41g.py"""
import sys, os, ast

SRC = "Dashboard_pro_V_41f.py"
DST = "Dashboard_pro_V_41g.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print("FILE " + SRC + ": " + str(len(src)) + " chars\n")

# ── G1: DARK_CSS aggiornato ──────────────────────────────────────────────────
src = src.replace(
    "[data-testid=\"block-container\"]{padding:0.5rem!important}",
    "[data-testid=\"block-container\"]{padding:0.75rem 1.25rem !important;max-width:100%!important}"
)

src = src.replace(
    "    font-family:'Trebuchet MS','Segoe UI',sans-serif !important;}",
    "    font-family:'Trebuchet MS','Segoe UI',sans-serif !important;\n    font-size:14px !important;}"
)

src = src.replace(
    "font-size:1.6rem !important;",
    "font-size:1.75rem !important;"
)
print("G1 DARK_CSS: OK")

# ── G2: font-size scale-up (usa token univoci per evitare doppia sostituzione) 
import re

def scale_font(m):
    val_str = m.group(1)
    try:
        val = float(val_str)
    except:
        return m.group(0)
    if   val <= 0.63: new = 0.76
    elif val <= 0.66: new = 0.78
    elif val <= 0.69: new = 0.80
    elif val <= 0.72: new = 0.83
    elif val <= 0.75: new = 0.85
    elif val < 0.80:  new = 0.87
    elif val < 0.85:  new = 0.93
    elif val < 0.90:  new = 0.96
    elif val < 0.93:  new = 1.00
    elif val < 1.01:  new = 1.10
    else:
        return m.group(0)
    # formatta come originale
    if '.' in val_str and len(val_str.split('.')[0]) == 0:
        return f"font-size:.{str(new).split('.')[1]}rem"
    return f"font-size:{new:.2f}rem"

# Sostituisce solo nelle f-string HTML (solo dentro apici tripli o f-string)
count_before = len(re.findall(r"font-size:(\.?\d+\.?\d*)rem", src))
src = re.sub(r"font-size:(\.?\d+\.?\d*)rem", scale_font, src)
count_after  = len(re.findall(r"font-size:(\.?\d+\.?\d*)rem", src))
print(f"G2 font-size scale: {count_before} trovati, {count_after} dopo")

# ── G3: versione ─────────────────────────────────────────────────────────────
src = src.replace("v41f", "v41g")
src = src.replace("V_41f", "V_41g")
src = src.replace("v32.1", "v32.2")

# ── Verifica ──────────────────────────────────────────────────────────────────
checks = {
    "font-size 14px": "font-size:14px !important",
    "block-container": "max-width:100%!important",
    "v41g": "v41g",
}
failed = []
print("\n-- Verifica --")
for lbl, marker in checks.items():
    ok = marker in src
    print("  " + ("OK" if ok else "FAIL") + " " + lbl)
    if not ok: failed.append(lbl)

try:
    ast.parse(src)
    print("  OK sintassi Python")
except SyntaxError as e:
    print(f"  ERRORE SINTASSI riga {e.lineno}: {e.msg}"); sys.exit(1)

if failed:
    print("FAILED:", failed); sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print(f"\nOK {DST} {len(src)} chars")
