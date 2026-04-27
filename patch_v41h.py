#!/usr/bin/env python3
"""patch_v41h.py  Dashboard_pro_V_41g.py → Dashboard_pro_V_41h.py
H4: Elimina blocco correlazioni standalone dalla Home
H5: Chiama patch_home_v322.py per modificare utils/home_tab.py
"""
import sys, os, subprocess

SRC = "Dashboard_pro_V_41g.py"
DST = "Dashboard_pro_V_41h.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print(f"FILE {SRC}: {len(src)} chars\n")

# ══ H4: Elimina blocco correlazioni standalone dalla Home ════════════════════
OLD_CORR_START = "    # v41g: correlazioni -> tab Settori\n"
OLD_CORR_END   = "    # v41: render_home per sparklines/breadth (senza Mercati Live duplicato)\n"

idx_start = src.find(OLD_CORR_START)
idx_end   = src.find(OLD_CORR_END)
if idx_start > 0 and idx_end > idx_start:
    src = src[:idx_start] + "    # v41h: correlazioni rimosse dalla Home (già nel tab Settori)\n" + src[idx_end:]
    print("H4 Elimina correlazioni Home: OK")
else:
    print(f"H4 Elimina correlazioni Home: SKIP (start={idx_start} end={idx_end})")

# ══ versione ══════════════════════════════════════════════════════════════════
src = src.replace("v41g", "v41h")
src = src.replace("V_41g", "V_41h")
src = src.replace("v32.2", "v32.3")

# ══ verifica ══════════════════════════════════════════════════════════════════
checks = {
    "v41h":           "v41h",
    "no corr home":   "home_corr_v41",
}
failed = []
print("\n-- Verifica Dashboard --")
for lbl, marker in checks.items():
    if lbl == "no corr home":
        ok = marker not in src
        print("  " + ("OK" if ok else "FAIL") + f" {lbl} (assente)")
    else:
        ok = marker in src
        print("  " + ("OK" if ok else "FAIL") + f" {lbl}")
    if not ok: failed.append(lbl)

try:
    compile(src, DST, "exec")
    print("  OK sintassi (compile)")
except SyntaxError as e:
    print(f"  ERRORE SINTASSI riga {e.lineno}: {e.msg}")
    sys.exit(1)

if failed:
    print("FAILED:", failed); sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print(f"\nOK {DST} {len(src)} chars")

# ══ H5: patch utils/home_tab.py ══════════════════════════════════════════════
print("\n-- Patch utils/home_tab.py --")
if os.path.exists("patch_home_v322.py"):
    ret = subprocess.run([sys.executable, "patch_home_v322.py"], capture_output=False)
    if ret.returncode != 0:
        print("ERR: patch_home_v322.py fallito"); sys.exit(1)
    # Copia il risultato su utils/home_tab.py
    if os.path.exists("home_tab_v322.py"):
        import shutil
        os.makedirs("utils", exist_ok=True)
        shutil.copy("home_tab_v322.py", "utils/home_tab.py")
        print("home_tab_v322.py copiato → utils/home_tab.py: OK")
    else:
        print("ERR: home_tab_v322.py non generato"); sys.exit(1)
else:
    print("SKIP: patch_home_v322.py non presente (caricalo nel repo)")

