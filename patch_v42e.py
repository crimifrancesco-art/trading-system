from pathlib import Path

SRC = Path("Dashboard_pro_V_41c.py")
DST = Path("Dashboard_pro_V_42e.py")

src = SRC.read_text(encoding="utf-8")
src = src.replace("https://it.tradingview.com/chart/?symbol={t.replace(\".MI\",\"%3AMI\")}", "https://it.tradingview.com/chart/?symbol={t.replace('.MI','%3AMI')}")
src = src.replace("Dashboard_pro_V_41c.py", "Dashboard_pro_V_42e.py")
DST.write_text(src, encoding="utf-8")
compile(src, str(DST), "exec")
print(src[:1000])
print("OK: generated", DST)
