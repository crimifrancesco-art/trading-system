#!/usr/bin/env python3
"""patch_v41g.py  Dashboard_pro_V_41f.py → Dashboard_pro_V_41g.py
Fix leggibilità globale: CSS base + font-size scale-up sicuro.
"""
import sys, os, ast, re

SRC = "Dashboard_pro_V_41f.py"
DST = "Dashboard_pro_V_41g.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print("FILE " + SRC + ": " + str(len(src)) + " chars\n")

# ── G1: DARK_CSS ──────────────────────────────────────────────────────────────
n1 = src.count("[data-testid=\"block-container\"]{padding:0.5rem!important}")
src = src.replace(
    "[data-testid=\"block-container\"]{padding:0.5rem!important}",
    "[data-testid=\"block-container\"]{padding:0.75rem 1.25rem !important;max-width:100%!important}"
)
print(f"G1a block-container: {'OK' if n1 else 'SKIP'}")

n2 = src.count("    font-family:'Trebuchet MS','Segoe UI',sans-serif !important;}")
src = src.replace(
    "    font-family:'Trebuchet MS','Segoe UI',sans-serif !important;}",
    "    font-family:'Trebuchet MS','Segoe UI',sans-serif !important;\n    font-size:14px !important;}"
)
print(f"G1b font-size base 14px: {'OK' if n2 else 'SKIP'}")

src = src.replace("font-size:1.6rem !important;", "font-size:1.75rem !important;")
print("G1c stMetricValue: OK")

# ── G2: font-size scale-up SICURO ─────────────────────────────────────────────
# Strategia: sostituisce solo pattern LETTERALI (numero puro, no {})
# Sostituisce token univoci prima, poi fa il replace finale
# Evita la regex su f-string con espressioni calcolate

FONT_MAP = [
    # Prima i valori più piccoli → evita doppio replace
    # usa token temporanei __FSxx__ per evitare collisioni
    ("font-size:0.62rem", "__FS076__"),
    ("font-size:.62rem",  "__FS076__"),
    ("font-size:0.63rem", "__FS076__"),
    ("font-size:.63rem",  "__FS076__"),
    ("font-size:0.64rem", "__FS077__"),
    ("font-size:.64rem",  "__FS077__"),
    ("font-size:0.65rem", "__FS077__"),
    ("font-size:.65rem",  "__FS077__"),
    ("font-size:0.66rem", "__FS078__"),
    ("font-size:.66rem",  "__FS078__"),
    ("font-size:0.67rem", "__FS078__"),
    ("font-size:.67rem",  "__FS078__"),
    ("font-size:0.68rem", "__FS079__"),
    ("font-size:.68rem",  "__FS079__"),
    ("font-size:0.69rem", "__FS079__"),
    ("font-size:.69rem",  "__FS079__"),
    ("font-size:0.70rem", "__FS082__"),
    ("font-size:.70rem",  "__FS082__"),
    ("font-size:0.71rem", "__FS082__"),
    ("font-size:.71rem",  "__FS082__"),
    ("font-size:0.72rem", "__FS083__"),
    ("font-size:.72rem",  "__FS083__"),
    ("font-size:0.73rem", "__FS084__"),
    ("font-size:.73rem",  "__FS084__"),
    ("font-size:0.74rem", "__FS084__"),
    ("font-size:.74rem",  "__FS084__"),
    ("font-size:0.75rem", "__FS085__"),
    ("font-size:.75rem",  "__FS085__"),
    ("font-size:0.76rem", "__FS087__"),
    ("font-size:.76rem",  "__FS087__"),
    ("font-size:0.77rem", "__FS087__"),
    ("font-size:.77rem",  "__FS087__"),
    ("font-size:0.78rem", "__FS088__"),
    ("font-size:.78rem",  "__FS088__"),
    ("font-size:0.79rem", "__FS089__"),
    ("font-size:.79rem",  "__FS089__"),
    ("font-size:0.80rem", "__FS090__"),
    ("font-size:.80rem",  "__FS090__"),
    ("font-size:0.82rem", "__FS092__"),
    ("font-size:.82rem",  "__FS092__"),
    ("font-size:0.83rem", "__FS093__"),
    ("font-size:.83rem",  "__FS093__"),
    ("font-size:0.84rem", "__FS094__"),
    ("font-size:.84rem",  "__FS094__"),
    ("font-size:0.86rem", "__FS096__"),
    ("font-size:.86rem",  "__FS096__"),
    ("font-size:0.88rem", "__FS097__"),
    ("font-size:.88rem",  "__FS097__"),
    ("font-size:0.90rem", "__FS100__"),
    ("font-size:.90rem",  "__FS100__"),
    ("font-size:0.92rem", "__FS100__"),
    ("font-size:.92rem",  "__FS100__"),
    ("font-size:1.0rem",  "__FS110__"),
    ("font-size:1.00rem", "__FS110__"),
]

TOKEN_FINAL = {
    "__FS076__": "font-size:0.76rem",
    "__FS077__": "font-size:0.77rem",
    "__FS078__": "font-size:0.78rem",
    "__FS079__": "font-size:0.79rem",
    "__FS082__": "font-size:0.82rem",
    "__FS083__": "font-size:0.83rem",
    "__FS084__": "font-size:0.84rem",
    "__FS085__": "font-size:0.85rem",
    "__FS087__": "font-size:0.87rem",
    "__FS088__": "font-size:0.88rem",
    "__FS089__": "font-size:0.89rem",
    "__FS090__": "font-size:0.90rem",
    "__FS092__": "font-size:0.92rem",
    "__FS093__": "font-size:0.93rem",
    "__FS094__": "font-size:0.94rem",
    "__FS096__": "font-size:0.96rem",
    "__FS097__": "font-size:0.97rem",
    "__FS100__": "font-size:1.00rem",
    "__FS110__": "font-size:1.10rem",
}

total = 0
for old, token in FONT_MAP:
    c = src.count(old)
    if c:
        src = src.replace(old, token)
        total += c

for token, final in TOKEN_FINAL.items():
    src = src.replace(token, final)

print(f"G2 font-size scale-up: {total} sostituzioni")

# ── G3: versione ──────────────────────────────────────────────────────────────
src = src.replace("v41f", "v41g")
src = src.replace("V_41f", "V_41g")
src = src.replace("v32.1", "v32.2")

# ── Verifica ──────────────────────────────────────────────────────────────────
checks = {
    "font-size 14px base":  "font-size:14px !important",
    "block-container fix":  "max-width:100%!important",
    "v41g":                 "v41g",
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
    print(f"  ERRORE SINTASSI riga {e.lineno}: {e.msg}")
    lines = src.split('\n')
    for i in range(max(0,e.lineno-3), min(len(lines), e.lineno+2)):
        print(f"    L{i+1}: {lines[i][:120]}")
    sys.exit(1)

if failed:
    print("FAILED:", failed); sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print(f"\nOK {DST} {len(src)} chars  ({total} font-size scalati)")

