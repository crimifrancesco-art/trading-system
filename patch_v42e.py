from pathlib import Path
import sys, re

SRC_CANDIDATES = [
    Path("Dashboard_pro_V_41e.py"),
    Path("Dashboard_pro_V_41c.py"),
]
DST = Path("Dashboard_pro_V_42e.py")

src_path = next((p for p in SRC_CANDIDATES if p.exists()), None)
if src_path is None:
    raise FileNotFoundError(
        "Nessun file sorgente trovato: Dashboard_pro_V_41e.py o Dashboard_pro_V_41c.py"
    )

src = src_path.read_text(encoding="utf-8")

# Fix 1: link TradingView doppi apici dentro f-string
src = src.replace(
    'https://it.tradingview.com/chart/?symbol={t.replace(".MI","%3AMI")}',
    "https://it.tradingview.com/chart/?symbol={t.replace('.MI','%3AMI')}",
)
src = src.replace(
    'https://it.tradingview.com/chart/?symbol={tk.replace(".MI","%3AMI")}',
    "https://it.tradingview.com/chart/?symbol={tk.replace('.MI','%3AMI')}",
)

# Fix 2: {"#hex" if var else "#hex"} -> {'#hex' if var else '#hex'}
def fix_nested_color(text):
    pattern = r'\{"(#[0-9a-fA-F]{6})" if (\w+) else "(#[0-9a-fA-F]{6})"\}'
    def repl(m):
        return "{'" + m.group(1) + "' if " + m.group(2) + " else '" + m.group(3) + "'}"
    return re.sub(pattern, repl, text)

src = fix_nested_color(src)

# Fix 3: .split(",") dentro f-string -> .split(',')
def fix_split_comma(text):
    return text.replace('.split(",")[', ".split(',')[")

src = fix_split_comma(src)

# Fix 4: aggiorna nome file nel sorgente
src = src.replace("Dashboard_pro_V_41c.py", "Dashboard_pro_V_42e.py")
src = src.replace("Dashboard_pro_V_41e.py", "Dashboard_pro_V_42e.py")

# Controlla sintassi e mostra contesto
try:
    compile(src, str(DST), "exec")
except SyntaxError as e:
    lines = src.splitlines()
    start = max(0, (e.lineno or 1) - 6)
    end   = min(len(lines), (e.lineno or 1) + 6)
    print(f"\nSyntaxError linea {e.lineno}: {e.msg}", file=sys.stderr)
    for i, line in enumerate(lines[start:end], start=start + 1):
        marker = ">>>" if i == e.lineno else "   "
        print(f"{marker} {i:5d}: {line}", file=sys.stderr)
    sys.exit(1)

DST.write_text(src, encoding="utf-8")
print(f"OK: generated {DST} from {src_path.name}")
