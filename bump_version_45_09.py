"""
Script di versioning: crea Dashboard_pro_V_45_09.py a partire da
Dashboard_pro_V_45_08.py, storicizzando la V45.08 (che resta invariata)
e aggiornando tutti i riferimenti di versione nel nuovo file.

Uso:
    cd /workspaces/trading-system
    source .venv/bin/activate
    python bump_version_45_09.py
"""

import shutil
from pathlib import Path

SRC = Path("Dashboard_pro_V_45_08.py")
DST = Path("Dashboard_pro_V_45_09.py")

if not SRC.exists():
    raise SystemExit(f"ERRORE: {SRC} non trovato. Esegui dalla root del repo.")

shutil.copy(SRC, DST)
text = DST.read_text(encoding="utf-8")

replacements = [
    ('APP_VERSION = "45.08"', 'APP_VERSION = "45.09"'),
    (
        '<div class="section-pill">SCANNER V45.08 · OPPORTUNITY RADAR · WATCHLIST ALERT · P&L TRACKER · BACKTEST PRO · EXPORT PRO · CHART TV-STYLE · MTF MATRIX · JOURNAL · REGIME</div>',
        '<div class="section-pill">SCANNER V45.09 · OPPORTUNITY RADAR · INSIDER RADAR · WATCHLIST ALERT · P&L TRACKER · BACKTEST PRO · EXPORT PRO · CHART TV-STYLE · MTF MATRIX · JOURNAL · REGIME</div>',
    ),
    ('🚀 AVVIA SCANNER PRO 45.08', '🚀 AVVIA SCANNER PRO 45.09'),
    ('Alert_v45_08_{tab_name}_{_at_ts}.csv', 'Alert_v45_09_{tab_name}_{_at_ts}.csv'),
    ('Alert_v45_08_{tab_name}_TradingView_{_at_ts}.txt', 'Alert_v45_09_{tab_name}_TradingView_{_at_ts}.txt'),
    ('trading_scanner_v45_08_home.csv', 'trading_scanner_v45_09_home.csv'),
    ('trading_scanner_v45_08_home.xlsx', 'trading_scanner_v45_09_home.xlsx'),
    ('trading_scanner_v45_08_home_TradingView.txt', 'trading_scanner_v45_09_home_TradingView.txt'),
    (
        "- Eseguire `python -m py_compile Dashboard_pro_V_45_08.py` dopo ogni modifica.",
        "- Eseguire `python -m py_compile Dashboard_pro_V_45_09.py` dopo ogni modifica.",
    ),
    (
        'st.caption("Trading Scanner PRO · Versione V45.08 · Opportunity Radar, Macro Regime e roadmap")',
        'st.caption("Trading Scanner PRO · Versione V45.09 · Opportunity Radar, Insider Radar, Macro Regime e roadmap")',
    ),
    (
        '# ── V45.08: tab Opportunity Radar (strategie selezionabili) ────────────',
        '# ── V45.08: tab Opportunity Radar (strategie selezionabili) ────────────\n# ── V45.09: tab Insider Radar (Form 4 SEC, insider score, cluster) ─────',
    ),
    (
        '<div class="section-pill">💾 EXPORT PRO v45.08 — XLSX Multi-Sheet · TXT TradingView · Timestamp Auto</div>',
        '<div class="section-pill">💾 EXPORT PRO v45.09 — XLSX Multi-Sheet · TXT TradingView · Timestamp Auto</div>',
    ),
    ('TradingScanner_v45_08_Tutti_{_ts}.xlsx', 'TradingScanner_v45_09_Tutti_{_ts}.xlsx'),
    ('TradingScanner_v45_08_Tutti_TradingView_{_ts}.txt', 'TradingScanner_v45_09_Tutti_TradingView_{_ts}.txt'),
    ('TradingScanner_v45_08_{cur_tab}_{_ts}.xlsx', 'TradingScanner_v45_09_{cur_tab}_{_ts}.xlsx'),
    ('TradingScanner_v45_08_{cur_tab}_TradingView_{_ts}.txt', 'TradingScanner_v45_09_{cur_tab}_TradingView_{_ts}.txt'),
]

missing = []
for old, new in replacements:
    if old not in text:
        missing.append(old)
    else:
        text = text.replace(old, new, 1)

if missing:
    DST.unlink(missing_ok=True)
    raise SystemExit(
        "ERRORE: pattern non trovati, file V45.09 NON creato:\n"
        + "\n".join(missing)
    )

DST.write_text(text, encoding="utf-8")
print("OK: Dashboard_pro_V_45_09.py creato e versione aggiornata a 45.09")
