#!/usr/bin/env python3
"""
build_v50_1.py -- Genera Dashboard_pro_V_50_1.py da Dashboard_pro_V_44i.py
Uso: python build_v50_1.py  (eseguire nella stessa cartella del file 44i)
"""
import re
import sys
import py_compile
from pathlib import Path

SRC = Path("Dashboard_pro_V_44i.py")
OUT = Path("Dashboard_pro_V_50_1.py")

V50_UX_BLOCK = (
    "\n\n"
    "# ============================= V50.1 UX LAYER =============================\n"
    "V50_RELEASE_NOTES = {\n"
    "    \"base_version\": \"44i\",\n"
    "    \"release_version\": \"50.1\",\n"
    "    \"focus\": [\n"
    "        \"scanner-first home\",\n"
    "        \"lighter sidebar\",\n"
    "        \"top-N consistency\",\n"
    "        \"clear tab hierarchy\",\n"
    "    ],\n"
    "}\n"
    "\n"
    "\n"
    "def render_v50_workflow_header():\n"
    "    try:\n"
    "        steps = [\"1. Mercati\", \"2. Profilo\", \"3. Parametri\", \"4. Filtri\", \"5. Scan\"]\n"
    "        chips = \"\".join([\n"
    "            f\"<span style='padding:6px 10px;border-radius:999px;background:#12343b;\"\n"
    "            f\"color:#d9fffb;border:1px solid #1f5e69;font-size:0.82rem'>{s}</span>\"\n"
    "            for s in steps\n"
    "        ])\n"
    "        st.markdown(\n"
    "            f\"<div style='display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 12px 0'>{chips}</div>\",\n"
    "            unsafe_allow_html=True,\n"
    "        )\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "\n"
    "def render_v50_focus_box():\n"
    "    try:\n"
    "        markets = []\n"
    "        for key, label in [\n"
    "            (\"mSP500\", \"S&P 500\"), (\"mNasdaq\", \"Nasdaq 100\"), (\"mFTSE\", \"FTSE MIB\"),\n"
    "            (\"mEurostoxx\", \"Eurostoxx 600\"), (\"mDow\", \"Dow Jones\"), (\"mRussell\", \"Russell 2000\"),\n"
    "            (\"mStoxxEmerging\", \"Stoxx Emerging 50\"), (\"mUSSmallCap\", \"US Small Cap 2000\"),\n"
    "        ]:\n"
    "            if bool(st.session_state.get(key, False)):\n"
    "                markets.append(label)\n"
    "        markets_txt = (\n"
    "            \", \".join(markets[:4]) + (f\" +{len(markets)-4}\" if len(markets) > 4 else \"\")\n"
    "            if markets else \"Nessun mercato selezionato\"\n"
    "        )\n"
    "        strong = \"STRONG only\" if bool(\n"
    "            st.session_state.get(\"show_strong_only\", False)\n"
    "        ) else \"Bilanciato\"\n"
    "        topn = int(st.session_state.get(\"top\", 15))\n"
    "        early = float(st.session_state.get(\"min_early_score\", 0))\n"
    "        qual = int(st.session_state.get(\"min_quality\", 0))\n"
    "        pro = float(st.session_state.get(\"min_pro_score\", 0))\n"
    "        st.info(\n"
    "            f\"Workflow attivo -> Mercati: {markets_txt} | Profilo: {strong} | \"\n"
    "            f\"Soglie: Early>={early} . Quality>={qual} . Pro>={pro} | Top N: {topn}\"\n"
    "        )\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
)

REPLACEMENTS = [
    (r'Trading Scanner PRO 44\.0[a-z]?', 'Trading Scanner PRO 50.1'),
    (r'44\.0[a-z]', '50.1'),
    (
        r'st\.sidebar\.title\(\s*["\']⚙️ Configurazione["\']\s*\)',
        'st.sidebar.title("⚙️ Configurazione V50.1")\n'
        'render_v50_workflow_header()\n'
        'render_v50_focus_box()'
    ),
    (r'st\.sidebar\.expander\(\s*["\']Preset Rapidi["\']\s*,\s*expanded=False\)',
     'st.sidebar.expander("2️⃣ Profilo Scanner", expanded=True)'),
    (r'st\.sidebar\.expander\(\s*["\']🎯 Preset Rapidi["\']\s*,\s*expanded=False\)',
     'st.sidebar.expander("2️⃣ Profilo Scanner", expanded=True)'),
    (r'st\.sidebar\.expander\(\s*["\']Mercati["\']\s*,\s*expanded=True\)',
     'st.sidebar.expander("1️⃣ Mercati", expanded=True)'),
    (r'st\.sidebar\.expander\(\s*["\']🌍 Mercati["\']\s*,\s*expanded=True\)',
     'st.sidebar.expander("1️⃣ Mercati", expanded=True)'),
    (r'st\.sidebar\.expander\(\s*["\']Parametri Scanner["\']\s*,\s*expanded=False\)',
     'st.sidebar.expander("3️⃣ Parametri Scanner", expanded=False)'),
    (r'st\.sidebar\.expander\(\s*["\']⚙️ Parametri Scanner["\']\s*,\s*expanded=False\)',
     'st.sidebar.expander("3️⃣ Parametri Scanner", expanded=False)'),
    (r'st\.sidebar\.expander\(\s*["\']Soglie Filtri live["\']\s*,\s*expanded=True\)',
     'st.sidebar.expander("4️⃣ Soglie Filtri Live", expanded=True)'),
    (r'st\.sidebar\.expander\(\s*["\']🎯 Soglie Filtri live["\']\s*,\s*expanded=True\)',
     'st.sidebar.expander("4️⃣ Soglie Filtri Live", expanded=True)'),
    (r'st\.sidebar\.caption\(\s*["\']Nessuna scansione attiva["\']\s*\)',
     'st.sidebar.caption("Nessuna scansione attiva -- configura i 4 step e avvia la scan")'),
    (r'SCANNER V4\.0 . WATCHLIST . ALERT . P/L TRACKER . EXPORT PRO . '
     r'CHART TV-STYLE . MTF MATRIX . JOURNAL . REGIME',
     'SCANNER V50.1 - WORKFLOW GUIDATO - TOP-N RESULTS - WATCHLIST - BACKTEST - EXPORT - REGIME'),
    (r'SCANNER V4\.0 . WATCHLIST . ALERT . P/L TRACKER . BACKTEST PRO . EXPORT PRO . '
     r'CHART TV-STYLE . MTF MATRIX . JOURNAL . REGIME',
     'SCANNER V50.1 - WORKFLOW GUIDATO - TOP-N RESULTS - WATCHLIST - BACKTEST - EXPORT - REGIME'),
]


def main():
    if not SRC.exists():
        print(f"ERRORE: non trovo {SRC} nella cartella corrente.")
        print("Esegui questo script nella stessa cartella del file 44i.")
        sys.exit(1)

    text = SRC.read_text(encoding="utf-8")
    original_len = len(text)

    applied = []
    for pattern, repl in REPLACEMENTS:
        new_text, n = re.subn(pattern, repl, text)
        if n > 0:
            applied.append((pattern[:50], n))
            text = new_text

    anchor = 'st.sidebar.title("⚙️ Configurazione V50.1")'
    if anchor in text and "V50_RELEASE_NOTES" not in text:
        idx = text.find(anchor)
        text = text[:idx] + V50_UX_BLOCK + text[idx:]
        applied.append(("V50_UX_BLOCK injected", 1))

    OUT.write_text(text, encoding="utf-8")

    print(f"OK: creato {OUT} ({len(text)} caratteri, originale {original_len}).")
    print("Sostituzioni applicate:")
    for pat, n in applied:
        print(f"  - {pat!r}: {n} occorrenze")

    if not applied:
        print("ATTENZIONE: nessuna sostituzione ha trovato corrispondenze.")
        print("Il file di output e identico all'originale -- controlla i testi esatti")
        print("nel tuo file 44i (potrebbero differire per spazi/emoji/versione).")

    try:
        py_compile.compile(str(OUT), doraise=True)
        print(f"OK: {OUT} compila correttamente.")
    except py_compile.PyCompileError as e:
        print(f"ERRORE DI COMPILAZIONE in {OUT}:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
