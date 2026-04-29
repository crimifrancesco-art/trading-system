#!/usr/bin/env python3
from pathlib import Path

SRC = "Dashboard_pro_V_41d.py"
DST = "Dashboard_pro_V_42e.py"

original = Path(SRC).read_text(encoding="utf-8")
src = original

src = src.replace('<img src="image-2.jpg"', '<img src="image-2.jpg" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')
src = src.replace('<img src="image-3.jpg"', '<img src="image-3.jpg" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')
src = src.replace('<img src="file:2"', '<img src="file:2" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')
src = src.replace('<img src="file:3"', '<img src="file:3" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')

for old in [
    "Suggerimenti v42c — Novità e roadmap",
    "Suggerimenti v42d — Novità e roadmap",
    "Suggerimenti v41d — Novità e roadmap",
]:
    src = src.replace(old, "Suggerimenti v42e — Novità e roadmap")

for old in [
    "Trading Scanner PRO 41.0d",
    "Trading Scanner PRO 41.0c",
    "Trading Scanner PRO 42.0c",
    "Trading Scanner PRO 42.0d",
]:
    src = src.replace(old, "Trading Scanner PRO 42.0e")

ideas_block = """**🔜 Idee per v42e:**
- 🔔 Alert push via browser (Web Push Notifications)
- 📊 Sparkline miniatura accanto al ticker nella Top PRO/STRONG
- 🗃️ Export segnali CSV/Excel con 1 click dalla Home
- 🔄 Auto-refresh Home ogni N minuti con st.rerun() schedulato
- 🧠 AI Analyst: storico analisi per ticker in SQLite
- 📱 Layout mobile-first con CSS container queries
- 📅 Earnings tracker compatto con filtro per giorni
- 🧠 AI history per ticker con ricerca veloce
- 🌙 Toggle temi persistente per sessione
"""

for old in [
    "**🔜 Idee per v42c:**",
    "**🔜 Idee per v42d:**",
    "**🔜 Idee per v41d:**",
]:
    src = src.replace(old, ideas_block)

if src == original:
    raise SystemExit("Nessuna modifica applicata: controlla che i testi cercati esistano in 41d.")

Path(DST).write_text(src, encoding="utf-8")
compile(src, DST, "exec")
print(f"OK wrote {DST} {len(src)} chars")
