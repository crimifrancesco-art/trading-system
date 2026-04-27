#!/usr/bin/env python3
"""patch_v41g.py  Dashboard_pro_V_41f.py → Dashboard_pro_V_41g.py"""
import sys, os

SRC = "Dashboard_pro_V_41f.py"
DST = "Dashboard_pro_V_41g.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print("FILE " + SRC + ": " + str(len(src)) + " chars\n")

# ══ FIX0a: f-string nested quotes riga ai_status (Python 3.12 → 3.11) ═══════
OLD0A = '    f"<span style=\'color:{"#00ff88" if ok else "#374151"}\'>{name.split()[0]}</span>"'
NEW0A = '    ("<span style=\'color:" + ("#00ff88" if ok else "#374151") + "\'>" + name.split()[0] + "</span>")'
n0a = src.count(OLD0A)
src = src.replace(OLD0A, NEW0A, 1)
print(f"FIX0a ai_status fstring: {'OK' if n0a else 'SKIP'} ({n0a})")

# ══ FIX0b: f-string nested quotes riga circle SVG (Python 3.12 → 3.11) ══════
OLD0B = 'f"<circle cx=\'{_pts[-1].split(",")[0]}\' cy=\'{_pts[-1].split(",")[1]}\' "'
NEW0B = ('f"<circle cx=\'{_pts[-1].split(chr(44))[0]}\' cy=\'{_pts[-1].split(chr(44))[1]}\' "')
n0b = src.count(OLD0B)
src = src.replace(OLD0B, NEW0B, 1)
print(f"FIX0b circle fstring: {'OK' if n0b else 'SKIP'} ({n0b})")

# ══ G1: DARK_CSS leggibilità ═════════════════════════════════════════════════
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

# ══ G2: font-size scale-up con token intermedi ═══════════════════════════════
FONT_MAP = [
    ("font-size:0.62rem","__FS076__"),("font-size:.62rem","__FS076__"),
    ("font-size:0.63rem","__FS076__"),("font-size:.63rem","__FS076__"),
    ("font-size:0.65rem","__FS077__"),("font-size:.65rem","__FS077__"),
    ("font-size:0.66rem","__FS078__"),("font-size:.66rem","__FS078__"),
    ("font-size:0.67rem","__FS078__"),("font-size:.67rem","__FS078__"),
    ("font-size:0.68rem","__FS079__"),("font-size:.68rem","__FS079__"),
    ("font-size:0.69rem","__FS079__"),("font-size:.69rem","__FS079__"),
    ("font-size:0.70rem","__FS082__"),("font-size:.70rem","__FS082__"),
    ("font-size:0.71rem","__FS082__"),("font-size:.71rem","__FS082__"),
    ("font-size:0.72rem","__FS083__"),("font-size:.72rem","__FS083__"),
    ("font-size:0.74rem","__FS084__"),("font-size:.74rem","__FS084__"),
    ("font-size:0.75rem","__FS085__"),("font-size:.75rem","__FS085__"),
    ("font-size:0.76rem","__FS087__"),("font-size:.76rem","__FS087__"),
    ("font-size:0.77rem","__FS087__"),("font-size:.77rem","__FS087__"),
    ("font-size:0.78rem","__FS088__"),("font-size:.78rem","__FS088__"),
    ("font-size:0.79rem","__FS089__"),("font-size:.79rem","__FS089__"),
    ("font-size:0.80rem","__FS090__"),("font-size:.80rem","__FS090__"),
    ("font-size:0.82rem","__FS092__"),("font-size:.82rem","__FS092__"),
    ("font-size:0.84rem","__FS094__"),("font-size:.84rem","__FS094__"),
    ("font-size:0.86rem","__FS096__"),("font-size:.86rem","__FS096__"),
    ("font-size:0.88rem","__FS097__"),("font-size:.88rem","__FS097__"),
    ("font-size:0.90rem","__FS100__"),("font-size:.90rem","__FS100__"),
    ("font-size:0.92rem","__FS100__"),("font-size:.92rem","__FS100__"),
    ("font-size:1.0rem", "__FS110__"),("font-size:1.00rem","__FS110__"),
]
TOKEN_FINAL = {
    "__FS076__":"font-size:0.76rem","__FS077__":"font-size:0.77rem",
    "__FS078__":"font-size:0.78rem","__FS079__":"font-size:0.79rem",
    "__FS082__":"font-size:0.82rem","__FS083__":"font-size:0.83rem",
    "__FS084__":"font-size:0.84rem","__FS085__":"font-size:0.85rem",
    "__FS087__":"font-size:0.87rem","__FS088__":"font-size:0.88rem",
    "__FS089__":"font-size:0.89rem","__FS090__":"font-size:0.90rem",
    "__FS092__":"font-size:0.92rem","__FS094__":"font-size:0.94rem",
    "__FS096__":"font-size:0.96rem","__FS097__":"font-size:0.97rem",
    "__FS100__":"font-size:1.00rem","__FS110__":"font-size:1.10rem",
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

# ══ versione ══════════════════════════════════════════════════════════════════
src = src.replace("v41f", "v41g")
src = src.replace("V_41f", "V_41g")
src = src.replace("v32.1", "v32.2")

# ══ verifica ══════════════════════════════════════════════════════════════════
checks = {
    "font-size 14px":  "font-size:14px !important",
    "block-container": "max-width:100%!important",
    "fstr 3.11 fix":   "chr(44)",
    "v41g":            "v41g",
}
failed = []
print("\n-- Verifica --")
for lbl, marker in checks.items():
    ok = marker in src
    print("  " + ("OK" if ok else "FAIL") + " " + lbl)
    if not ok: failed.append(lbl)

try:
    compile(src, DST, "exec")
    print("  OK sintassi (compile)")
except SyntaxError as e:
    print(f"  ERRORE SINTASSI riga {e.lineno}: {e.msg}")
    lines = src.split("\n")
    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
        print(f"    L{i+1}: {lines[i][:120]}")
    sys.exit(1)

if failed:
    print("FAILED:", failed); sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print(f"\nOK {DST} {len(src)} chars  ({total} font-size scalati)")
