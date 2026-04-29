#!/usr/bin/env python3
from pathlib import Path

SRC = "Dashboard_pro_V_41c.py"
DST = "Dashboard_pro_V_42e.py"

src = Path(SRC).read_text(encoding="utf-8")

src = src.replace('<img src="image-2.jpg"', '<img src="image-2.jpg" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')
src = src.replace('<img src="image-3.jpg"', '<img src="image-3.jpg" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')
src = src.replace('<img src="file:2"', '<img src="file:2" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')
src = src.replace('<img src="file:3"', '<img src="file:3" style="display:block;max-width:70%;height:auto;margin:6px auto 0 auto;"')

src = src.replace("Suggerimenti v42c — Novità e roadmap", "Suggerimenti v42e — Novità e roadmap")
src = src.replace("Suggerimenti v42d — Novità e roadmap", "Suggerimenti v42e — Novità e roadmap")
src = src.replace("Trading Scanner PRO 41.0c", "Trading Scanner PRO 42.0e")
src = src.replace("Trading Scanner PRO 42.0c", "Trading Scanner PRO 42.0e")
src = src.replace("Trading Scanner PRO 42.0d", "Trading Scanner PRO 42.0e")

old_ideas = "**🔜 Idee per v42c:**"
new_ideas = """**🔜 Idee per v42e:**
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
src = src.replace(old_ideas, new_ideas)
src = src.replace("**🔜 Idee per v42d:**", new_ideas)

Path(DST).write_text(src, encoding="utf-8")
compile(src, DST, "exec")
print(f"OK wrote {DST} {len(src)} chars")
