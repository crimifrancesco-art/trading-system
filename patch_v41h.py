#!/usr/bin/env python3
"""patch_v41h.py  Dashboard_pro_V_41g.py → Dashboard_pro_V_41h.py
FIX H1: Sparklines (image2) - height ridotta 255→180
FIX H2: Fear&Greed/Breadth/RSI (image3) - font card ridotti, RSI height 160→120
FIX H3: Ranking settori bar chart (image4) - aggiungi nome settore esteso
FIX H4: Correlazioni Asset Home - ELIMINARE (spostata già in tab Settori)
"""
import sys, os, ast

SRC = "Dashboard_pro_V_41g.py"
DST = "Dashboard_pro_V_41h.py"

if not os.path.exists(SRC):
    print("ERR: " + SRC + " non trovato"); sys.exit(1)

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print("FILE " + SRC + ": " + str(len(src)) + " chars\n")

# ══ H1: Sparklines - riduci height ══════════════════════════════════════════
# nella home_tab.py il fig sparklines ha height=255
n1 = src.count("height=255, paper_bgcolor=BG, plot_bgcolor=PANEL,")
src = src.replace(
    "height=255, paper_bgcolor=BG, plot_bgcolor=PANEL,",
    "height=190, paper_bgcolor=BG, plot_bgcolor=PANEL,"
)
print(f"H1 sparklines height 255→190: {'OK' if n1 else 'SKIP'} ({n1})")

# ══ H2: RSI distribution - riduci height ════════════════════════════════════
n2a = src.count("height=160, margin=dict(l=0, r=0, t=28, b=0),")
src = src.replace(
    "height=160, margin=dict(l=0, r=0, t=28, b=0),",
    "height=130, margin=dict(l=0, r=0, t=22, b=0),"
)
print(f"H2a RSI dist height 160→130: {'OK' if n2a else 'SKIP'} ({n2a})")

# Fear&Greed card: font numero da 2.5rem → 1.8rem
n2b = src.count("font-size:2.6rem;font-weight:800;color:{color};")
src = src.replace(
    "font-size:2.6rem;font-weight:800;color:{color};",
    "font-size:1.9rem;font-weight:800;color:{color};"
)
print(f"H2b Fear&Greed font: {'OK' if n2b else 'SKIP'} ({n2b})")

# Market Breadth: font numero da 2rem → 1.5rem
n2c = src.count("font-size:2.1rem;font-weight:800;font-family:Courier New")
src = src.replace(
    "font-size:2.1rem;font-weight:800;font-family:Courier New",
    "font-size:1.6rem;font-weight:800;font-family:Courier New"
)
print(f"H2c Breadth font: {'OK' if n2c else 'SKIP'} ({n2c})")

# VIX number nella regime bar: da 1.7rem → 1.3rem
n2d = src.count("font-size:1.75rem;font-weight:800;")
src = src.replace(
    "font-size:1.75rem;font-weight:800;",
    "font-size:1.35rem;font-weight:800;"
)
print(f"H2d VIX number font: {'OK' if n2d else 'SKIP'} ({n2d})")

# ══ H3: Ranking settori bar chart - etichette nome esteso ════════════════════
# Attualmente usa solo sdf["label"] (es. "Tech", "Finance")
# Aggiunge mapping nome esteso
OLD_BAR = (
            'fig = go.Figure(go.Bar(\n'
            '                y=sdf["label"], x=sdf["chg"], orientation="h",\n'
            '                marker_color=bar_colors, marker_line_width=0,\n'
            '                hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",\n'
            '            ))'
)
NEW_BAR = (
            '_label_map = {\n'
            '                "Tech":"Technology","Finance":"Financials","Healthcare":"Health Care",\n'
            '                "Energy":"Energy","Industrial":"Industrials","Cons.Discr":"Cons. Discretionary",\n'
            '                "Cons.Stpl":"Cons. Staples","Materials":"Materials","Real Estate":"Real Estate",\n'
            '                "Utilities":"Utilities","Comm.Srv":"Comm. Services","Biotech":"Biotech (XBI)",\n'
            '            }\n'
            '            sdf["label_full"] = sdf["label"].map(_label_map).fillna(sdf["label"])\n'
            '            fig = go.Figure(go.Bar(\n'
            '                y=sdf["label_full"], x=sdf["chg"], orientation="h",\n'
            '                marker_color=bar_colors, marker_line_width=0,\n'
            '                hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",\n'
            '            ))'
)
n3 = src.count(OLD_BAR)
src = src.replace(OLD_BAR, NEW_BAR, 1)
print(f"H3 sector bar label esteso: {'OK' if n3 else 'SKIP'} ({n3})")

# ══ H4: ELIMINA sezione Correlazioni dalla Home ══════════════════════════════
# Blocco: da "# v41g: correlazioni" fino a "# v41: render_home per sparklines"
OLD_CORR = (
    "    # v41g: correlazioni -> tab Settori\n"
    "\n"
    "        @st.cache_data(ttl=3600, show_spinner=False)\n"
    "        def _fetch_corr_v41():\n"
    "            import yfinance as _yc\n"
    "            _corr_syms = {\n"
    "                \"S&P 500\": \"^GSPC\", \"NASDAQ\": \"^IXIC\", \"DAX\": \"^GDAXI\",\n"
    "                \"FTSE MIB\": \"FTSEMIB.MI\", \"Nikkei\": \"^N225\",\n"
    "                \"Bitcoin\": \"BTC-USD\", \"Gold\": \"GC=F\", \"Oil WTI\": \"CL=F\",\n"
    "                \"Silver\": \"SI=F\", \"DXY\": \"DX-Y.NYB\", \"VIX\": \"^VIX\",\n"
    "                \"TLT Bond\": \"TLT\",\n"
    "            }\n"
    "            _raw = _yc.download(\n"
    "                list(_corr_syms.values()), period=\"30d\", interval=\"1d\",\n"
    "                auto_adjust=True, progress=False, group_by=\"ticker\"\n"
    "            )\n"
    "            _closes = {}\n"
    "            for _lab, _sym in _corr_syms.items():\n"
    "                try:\n"
    "                    if isinstance(_raw.columns, pd.MultiIndex):\n"
    "                        _s = _raw[(_sym, \"Close\")].dropna() if (_sym,\"Close\") in _raw.columns else _raw[\"Close\"][_sym].dropna()\n"
    "                    else:\n"
    "                        _s = _raw[\"Close\"].dropna()\n"
    "                    if len(_s) >= 10:\n"
    "                        _closes[_lab] = _s\n"
    "                except Exception:\n"
    "                    pass\n"
    "            if len(_closes) < 2:\n"
    "                return None\n"
    "            _df_c = pd.DataFrame(_closes).pct_change().dropna()\n"
    "            return _df_c.corr().round(2)\n"
    "\n"
    "        try:\n"
    "            _corr_df = _fetch_corr_v41()\n"
    "            if _corr_df is not None and not _corr_df.empty:\n"
    "                import plotly.graph_objects as _go_corr\n"
    "                _labs = list(_corr_df.columns)\n"
    "                _zvals = _corr_df.values.tolist()\n"
    "                _text  = [[f\"{v:.2f}\" for v in row] for row in _zvals]\n"
    "                _fig_corr = _go_corr.Figure(_go_corr.Heatmap(\n"
    "                    z=_zvals, x=_labs, y=_labs, text=_text,\n"
    "                    texttemplate=\"%{text}\",\n"
    "                    colorscale=[\n"
    "                        [0.0,  \"#ef4444\"], [0.4, \"#991b1b\"],\n"
    "                        [0.5,  \"#1e222d\"],\n"
    "                        [0.6,  \"#1a3a2e\"], [1.0, \"#26a69a\"],\n"
    "                    ],\n"
    "                    zmid=0, zmin=-1, zmax=1,\n"
    "                    showscale=True,\n"
    "                    colorbar=dict(\n"
    "                        tickfont=dict(color=\"#787b86\", size=10),\n"
    "                        bgcolor=\"#131722\", bordercolor=\"#2a2e39\",\n"
    "                    )\n"
    "                ))\n"
    "                _fig_corr.update_layout(\n"
    "                    paper_bgcolor=\"#131722\", plot_bgcolor=\"#131722\",\n"
    "                    margin=dict(l=10,r=10,t=10,b=10),\n"
    "                    height=420,\n"
    "                    font=dict(color=\"#d1d4dc\", size=11),\n"
    "                    xaxis=dict(tickfont=dict(size=10), gridcolor=\"#2a2e39\"),\n"
    "                    yaxis=dict(tickfont=dict(size=10), gridcolor=\"#2a2e39\"),\n"
    "                )\n"
    "                st.plotly_chart(_fig_corr, use_container_width=True, key=\"home_corr_v41\")\n"
    "                st.caption(\n"
    "                    \"🟢 +1 = si muovono insieme (rischio correlato) &nbsp;·&nbsp; \"\n"
    "                    \"🔴 −1 = hedge naturale (si muovono opposti) &nbsp;·&nbsp; \"\n"
    "                    \"⬛ 0 = scorrelati. Periodo: 30gg giornaliero.\"\n"
    "                )\n"
    "            else:\n"
    "                st.info(\"Dati correlazione non disponibili — riprova tra qualche secondo.\")\n"
    "        except Exception as _ce:\n"
    "            st.warning(f\"Errore correlazioni: {str(_ce)[:120]}\")\n"
)
NEW_CORR = "    # v41h: correlazioni rimosse dalla Home (già nel tab Settori)\n"
n4 = src.count(OLD_CORR)
src = src.replace(OLD_CORR, NEW_CORR, 1)
print(f"H4 Elimina correlazioni Home: {'OK' if n4 else 'SKIP'} ({n4})")

# ══ versione ══════════════════════════════════════════════════════════════════
src = src.replace("v41g", "v41h")
src = src.replace("V_41g", "V_41h")
src = src.replace("v32.2", "v32.3")

# ══ verifica ══════════════════════════════════════════════════════════════════
checks = {
    "height 190":       "height=190, paper_bgcolor=BG",
    "v41h":             "v41h",
    "no corr home":     "home_corr_v41",   # deve essere ASSENTE
}
failed = []
print("\n-- Verifica --")
for lbl, marker in checks.items():
    if lbl == "no corr home":
        ok = marker not in src
        print("  " + ("OK" if ok else "FAIL") + " " + lbl + " (assente)")
    else:
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
print(f"\nOK {DST} {len(src)} chars")
