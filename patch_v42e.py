from pathlib import Path

SRC_CANDIDATES = [
    Path("Dashboard_pro_V_41e.py"),
    Path("Dashboard_pro_V_41c.py"),
]
DST = Path("Dashboard_pro_V_42e.py")

src_path = next((p for p in SRC_CANDIDATES if p.exists()), None)
if src_path is None:
    raise FileNotFoundError("Nessun file sorgente trovato: Dashboard_pro_V_41e.py o Dashboard_pro_V_41c.py")

src = src_path.read_text(encoding="utf-8")

old = 'https://it.tradingview.com/chart/?symbol={t.replace(".MI","%3AMI")}'
new = "https://it.tradingview.com/chart/?symbol={t.replace('.MI','%3AMI')}"
src = src.replace(old, new)

src = src.replace("Dashboard_pro_V_41c.py", "Dashboard_pro_V_42e.py")
src = src.replace("Dashboard_pro_V_41e.py", "Dashboard_pro_V_42e.py")

compile(src, str(DST), "exec")
DST.write_text(src, encoding="utf-8")

print(f"OK: generated {DST} from {src_path.name}")
