#!/usr/bin/env python3
"""patch_v42a.py  Dashboard_pro_V_41e.py → Dashboard_pro_V_42a.py
NUOVE FEATURE v42a:
  A1 — Export CSV/Excel 1-click dalla Home
  A2 — Auto-refresh Home ogni N minuti
  A4 — AI Analyst storico in SQLite
  A5 — Suggerimenti aggiornati
  A6 — Versione v41e → v42a nelle label
  FIX — f-string nested quotes Python 3.11 compat
"""
import sys, os, re

SRC = "Dashboard_pro_V_41e.py"
DST = "Dashboard_pro_V_42a.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print(f"FILE {SRC}: {len(src)} chars\n")

# ══ FIX0: Python 3.11 compat — f-string nested quotes ════════════════════════
OLD_F1 = (
    '    f"<span style=\'color:{"#00ff88" if ok else "#374151"}\'>'
    '{name.split()[0]}</span>"'
)
NEW_F1 = (
    '    ("<span style=\'color:" + ("#00ff88" if ok else "#374151") + "\'>"'
    ' + name.split()[0] + "</span>")'
)
nf1 = src.count(OLD_F1)
src = src.replace(OLD_F1, NEW_F1, 1)
print(f"FIX0a f-string ai_status: {'OK' if nf1 else 'SKIP'} ({nf1})")

OLD_F2 = 'f"<circle cx=\'{_pts[-1].split(",")[0]}\' cy=\'{_pts[-1].split(",")[1]}\' "'
NEW_F2 = 'f"<circle cx=\'{_pts[-1].split(chr(44))[0]}\' cy=\'{_pts[-1].split(chr(44))[1]}\' "'
nf2 = src.count(OLD_F2)
src = src.replace(OLD_F2, NEW_F2, 1)
print(f"FIX0b f-string circle SVG: {'OK' if nf2 else 'SKIP'} ({nf2})")

# ══ A1+A2: Export CSV + Auto-refresh Home ════════════════════════════════════
OLD_A1 = "    # ── v41e: Suggerimenti ────────────────────────────────\n"
NEW_A1 = """\
    # ── v42a A1 — Export CSV/Excel 1-click dalla Home ───────────────────────
    try:
        _df_ep_home = st.session_state.get("df_ep", None)
        if _df_ep_home is not None and not _df_ep_home.empty:
            _ex_cols = [c for c in ["Ticker","Nome","Prezzo","ProScore","EarlyScore",
                                     "RSI","CSS","StatoPro","StatoEarly","DollarVol",
                                     "QualityScore","ATRpct","VolRatio","Squeeze",
                                     "WeeklyBull"] if c in _df_ep_home.columns]
            if _ex_cols:
                _dfex_out = _df_ep_home[_ex_cols]
                _ex_ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M")
                _ex_c1, _ex_c2, _ex_c3 = st.columns([1, 1, 4])
                with _ex_c1:
                    st.download_button(
                        "⬇️ CSV Segnali",
                        data=_dfex_out.to_csv(index=False).encode("utf-8"),
                        file_name=f"segnali_v42a_{_ex_ts}.csv",
                        mime="text/csv",
                        key="home_export_csv_v42a",
                        help="Esporta segnali PRO/STRONG in CSV"
                    )
                with _ex_c2:
                    try:
                        import io as _io_ex
                        _buf_xl = _io_ex.BytesIO()
                        with __import__("pandas").ExcelWriter(
                                _buf_xl, engine="xlsxwriter") as _xlw:
                            _dfex_out.to_excel(_xlw, index=False,
                                               sheet_name="Segnali_v42a")
                        st.download_button(
                            "⬇️ Excel Segnali",
                            data=_buf_xl.getvalue(),
                            file_name=f"segnali_v42a_{_ex_ts}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="home_export_xlsx_v42a",
                            help="Esporta in Excel (richiede xlsxwriter)"
                        )
                    except Exception:
                        pass
    except Exception:
        pass

    # ── v42a A2 — Auto-refresh Home ogni N minuti ────────────────────────────
    try:
        _ar_c1, _ar_c2 = st.columns([1, 5])
        with _ar_c1:
            _ar_on = st.toggle("🔄 Auto-refresh", value=False,
                               key="home_autorefresh_v42a",
                               help="Aggiorna la Home ogni N minuti")
        if _ar_on:
            with _ar_c2:
                _ar_mins = st.slider("min", 1, 30, 5,
                                     key="home_ar_interval_v42a",
                                     label_visibility="collapsed")
            import time as _art
            _ar_last = st.session_state.get("home_ar_last_v42a", 0)
            _ar_now  = _art.time()
            _ar_rem  = max(0, _ar_mins * 60 - (_ar_now - _ar_last))
            if _ar_now - _ar_last >= _ar_mins * 60:
                st.session_state["home_ar_last_v42a"] = _ar_now
                st.rerun()
            else:
                st.caption(
                    f"🔄 Prossimo refresh tra "
                    f"{int(_ar_rem // 60)}m {int(_ar_rem % 60)}s"
                )
    except Exception:
        pass

    # ── v42a: Suggerimenti ────────────────────────────────
"""
n1 = src.count(OLD_A1)
src = src.replace(OLD_A1, NEW_A1, 1)
print(f"A1+A2 Export CSV + Auto-refresh: {'OK' if n1 else 'SKIP'} ({n1})")

# ══ A4: AI Analyst storico SQLite ════════════════════════════════════════════
OLD_A4 = "# ── v41e — MODULO 2 AI ───────────────────────────────\n"
NEW_A4 = """\
# ── v42a — AI storico SQLite ────────────────────────────────────────────────

def _save_ai_analysis_v42a(ticker: str, analysis: str, model: str = ""):
    \"\"\"Salva analisi AI in SQLite.\"\"\"
    try:
        import sqlite3 as _sq42, datetime as _dt42
        _c42 = _sq42.connect(str(DBPATH))
        _c42.execute(\"\"\"CREATE TABLE IF NOT EXISTS ai_analysis_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, model TEXT, analysis TEXT, created_at TEXT)\"\"\")
        _c42.execute(
            "INSERT INTO ai_analysis_log VALUES (NULL,?,?,?,?)",
            (ticker.upper(), model, analysis[:4000],
             _dt42.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        _c42.commit(); _c42.close()
    except Exception:
        pass

def _load_ai_history_v42a(ticker: str, limit: int = 5) -> list:
    \"\"\"Carica ultime N analisi AI per ticker.\"\"\"
    try:
        import sqlite3 as _sq42
        _c42 = _sq42.connect(str(DBPATH))
        _rows = _c42.execute(
            "SELECT model, analysis, created_at FROM ai_analysis_log "
            "WHERE ticker=? ORDER BY id DESC LIMIT ?",
            (ticker.upper(), limit)).fetchall()
        _c42.close(); return _rows
    except Exception:
        return []

# ── v41e — MODULO 2 AI ───────────────────────────────
"""
n4 = src.count(OLD_A4)
src = src.replace(OLD_A4, NEW_A4, 1)
print(f"A4 AI storico SQLite: {'OK' if n4 else 'SKIP'} ({n4})")

# ══ A5: Suggerimenti aggiornati ═══════════════════════════════════════════════
OLD_A5 = "    with st.expander('💡 Suggerimenti v41e — Novità e roadmap', expanded=False):\n"
NEW_A5 = "    with st.expander('💡 Suggerimenti v42a — Novità e roadmap', expanded=False):\n"
n5a = src.count(OLD_A5); src = src.replace(OLD_A5, NEW_A5, 1)

OLD_A5B = "**✅ Implementato in v41e:**"
NEW_A5B = """\
**✅ Implementato in v42a (da roadmap v41e):**
- 🗃️ **Export CSV/Excel** segnali con 1 click dalla Home
- 🔄 **Auto-refresh Home** ogni N minuti (toggle + slider)
- 🧠 **AI Analyst storico** analisi per ticker in SQLite
- 🔧 Fix compatibilità Python 3.11 (f-string nested quotes)

**✅ Implementato in v41e:**"""
n5b = src.count(OLD_A5B); src = src.replace(OLD_A5B, NEW_A5B, 1)

OLD_A5C = "**🔜 Idee per v41e:**"
NEW_A5C = "**🔜 Idee per v42b:**"
n5c = src.count(OLD_A5C); src = src.replace(OLD_A5C, NEW_A5C, 1)

OLD_A5D = (
    "- 🔔 Alert push via browser (Web Push Notifications)\n"
    "- 📊 Sparkline miniatura accanto al ticker nella Top PRO/STRONG\n"
    "- 🗃️ Export segnali CSV/Excel con 1 click dalla Home\n"
    "- 🔄 Auto-refresh Home ogni N minuti con st.rerun() schedulato\n"
    "- 🧠 AI Analyst: storico analisi per ticker in SQLite\n"
    "- 📱 Layout mobile-first con CSS container queries"
)
NEW_A5D = (
    "- 🔔 Alert push via browser (Web Push Notifications)\n"
    "- 📱 Layout mobile-first con CSS container queries\n"
    "- 🗓️ Earnings sorprese: verde/rosso se EPS actual > estimate\n"
    "- 🔍 Ricerca ticker globale con fuzzy search\n"
    "- 📈 Confronto portafoglio vs benchmark SPY/QQQ nel tempo"
)
src = src.replace(OLD_A5D, NEW_A5D, 1)
print(f"A5 Suggerimenti: {'OK' if n5a and n5b and n5c else 'SKIP'}")

# ══ A6: Versione titolo e label ═══════════════════════════════════════════════
src = src.replace(
    'page_title="Trading Scanner PRO 41.0c"',
    'page_title="Trading Scanner PRO 42.0a"'
)
src = src.replace(
    '# 🧠 Trading Scanner PRO 41.0c',
    '# 🧠 Trading Scanner PRO 42.0a'
)
src = re.sub(r'(MOMENTUM ALERTS )v41e', r'\1v42a', src)
src = re.sub(r'(P&L Tracker & Alert Engine )v41e', r'\1v42a', src)
src = re.sub(r'(NEWS & SENTIMENT )v41e', r'\1v42a', src)
src = re.sub(r'(MACRO CALENDAR )v41e', r'\1v42a', src)
src = re.sub(r'(Mappa Calore Globale — Performance indici mondiali )v41e', r'\1v42a', src)
src = re.sub(r'(Canali Notifica )v41e', r'\1v42a', src)
src = re.sub(r'(Dashboard )v41e', r'\1v42a', src)
print("A6 label versione →v42a: OK")

# ══ VERIFICA ══════════════════════════════════════════════════════════════════
checks = {
    "FIX0a f-string 3.11":   'chr(35)" + ("#00ff88"' if False else
                              '+ ("#00ff88" if ok else "#374151") +',
    "Export CSV button":      "home_export_csv_v42a",
    "Auto-refresh toggle":    "home_autorefresh_v42a",
    "AI storico save func":   "_save_ai_analysis_v42a",
    "AI storico load func":   "_load_ai_history_v42a",
    "Suggerimenti v42a":      "Suggerimenti v42a",
    "TITLE v42a":             "42.0a",
}
failed = []
print("\n-- Verifica --")
for lbl, marker in checks.items():
    ok = marker in src
    print("  " + ("OK" if ok else "FAIL") + f" {lbl}")
    if not ok: failed.append(lbl)

try:
    compile(src, DST, "exec")
    print("  OK sintassi (compile)")
except SyntaxError as e:
    print(f"  ERRORE SINTASSI riga {e.lineno}: {e.msg}")
    lines = src.split("\n")
    for i in range(max(0, e.lineno - 3), min(len(lines), e.lineno + 2)):
        print(f"    L{i+1}: {lines[i][:120]}")
    sys.exit(1)

if failed:
    print("FAILED:", failed); sys.exit(1)

with open(DST, "w", encoding="utf-8") as f:
    f.write(src)
print(f"\nOK {DST} {len(src):,} chars")

