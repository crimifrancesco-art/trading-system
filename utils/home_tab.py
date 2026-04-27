#!/usr/bin/env python3
"""patch_home_v322.py  home_tab_v321.py → home_tab_v322.py (utils/home_tab.py)
H1: sparklines height 255→190
H2a: RSI dist height 160→130
H2b: Fear&Greed numero 2.5rem→1.8rem
H2c: Breadth numero 2rem→1.5rem
H2d: VIX numero nel regime bar → più piccolo (1.6rem→1.2rem)
H3: sector bar - label estese + height 320→280
H4: correlazioni expander home_tab - rimuovi (usa st.expander Correlazioni)
"""
import sys, os

# Cerca prima home_tab_v321.py (sandbox), poi utils/home_tab.py (repo)
for candidate in ["home_tab_v321.py", "utils/home_tab.py"]:
    if os.path.exists(candidate):
        SRC = candidate
        break
else:
    print("ERR: home_tab non trovato"); sys.exit(1)

DST = "home_tab_v322.py"
with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()
print(f"FILE {SRC}: {len(src)} chars\n")

# ══ H1: Sparklines height 255 → 190 ══════════════════════════════════════════
n1 = src.count("height=255, paper_bgcolor=BG, plot_bgcolor=PANEL,")
src = src.replace(
    "height=255, paper_bgcolor=BG, plot_bgcolor=PANEL,",
    "height=190, paper_bgcolor=BG, plot_bgcolor=PANEL,"
)
print(f"H1 sparklines height: {'OK' if n1 else 'SKIP'} ({n1})")

# ══ H2a: RSI distribution height 160 → 120 ═══════════════════════════════════
n2a = src.count("height=160, margin=dict(l=0, r=0, t=28, b=0),")
src = src.replace(
    "height=160, margin=dict(l=0, r=0, t=28, b=0),",
    "height=120, margin=dict(l=0, r=0, t=20, b=0),"
)
print(f"H2a RSI dist height: {'OK' if n2a else 'SKIP'} ({n2a})")

# ══ H2b: Fear&Greed numero 2.5rem → 1.7rem ═══════════════════════════════════
n2b = src.count("f'<div style=\"font-size:2.5rem;font-weight:800;color:{color};'")
src = src.replace(
    "f'<div style=\"font-size:2.5rem;font-weight:800;color:{color};'",
    "f'<div style=\"font-size:1.7rem;font-weight:800;color:{color};'"
)
print(f"H2b Fear&Greed font: {'OK' if n2b else 'SKIP'} ({n2b})")

# ══ H2c: Breadth numero 2rem → 1.5rem ════════════════════════════════════════
n2c = src.count("f'<span style=\"color:{color};font-size:2rem;font-weight:800;font-family:Courier New\">'")
src = src.replace(
    "f'<span style=\"color:{color};font-size:2rem;font-weight:800;font-family:Courier New\">'\n",
    "f'<span style=\"color:{color};font-size:1.5rem;font-weight:800;font-family:Courier New\">'\n"
)
print(f"H2c Breadth font: {'OK' if n2c else 'SKIP'} ({n2c})")

# ══ H2d: VIX numero regime bar 1.6rem → 1.2rem ═══════════════════════════════
n2d = src.count("f'<div style=\"color:{color};font-size:1.6rem;font-weight:800;'")
src = src.replace(
    "f'<div style=\"color:{color};font-size:1.6rem;font-weight:800;'",
    "f'<div style=\"color:{color};font-size:1.2rem;font-weight:800;'"
)
print(f"H2d VIX regime font: {'OK' if n2d else 'SKIP'} ({n2d})")

# ══ H3: Sector bar - label estese + height ════════════════════════════════════
OLD_BAR = (
    '            fig = go.Figure(go.Bar(\n'
    '                y=sdf["label"], x=sdf["chg"], orientation="h",\n'
    '                marker_color=bar_colors, marker_line_width=0,\n'
    '                hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",\n'
    '            ))\n'
    '            fig.add_vline(x=0, line=dict(color=BORDER, width=1))\n'
    '            fig.update_layout(\n'
    '                paper_bgcolor=BG, plot_bgcolor=PANEL,\n'
    '                height=320, margin=dict(l=0, r=0, t=10, b=0),\n'
    '                xaxis=dict(gridcolor=BORDER, ticksuffix="%", tickfont=dict(size=9)),\n'
    '                yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),\n'
    '                font=dict(color=TEXT, size=9),\n'
    '            )'
)
NEW_BAR = (
    '            _lmap = {\n'
    '                "Tech":"Technology","Finance":"Financials","Healthcare":"Health Care",\n'
    '                "Energy":"Energy","Industrial":"Industrials","Cons.Discr":"Cons. Discretionary",\n'
    '                "Cons.Stpl":"Cons. Staples","Materials":"Materials","Real Estate":"Real Estate",\n'
    '                "Utilities":"Utilities","Comm.Srv":"Comm. Services","Biotech":"Biotech (XBI)",\n'
    '            }\n'
    '            sdf["label_full"] = sdf["label"].map(_lmap).fillna(sdf["label"])\n'
    '            fig = go.Figure(go.Bar(\n'
    '                y=sdf["label_full"], x=sdf["chg"], orientation="h",\n'
    '                marker_color=bar_colors, marker_line_width=0,\n'
    '                text=[f"{c:+.2f}%" for c in sdf["chg"]],\n'
    '                textposition="outside", textfont=dict(size=9, color=TEXT),\n'
    '                hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",\n'
    '            ))\n'
    '            fig.add_vline(x=0, line=dict(color=BORDER, width=1))\n'
    '            fig.update_layout(\n'
    '                paper_bgcolor=BG, plot_bgcolor=PANEL,\n'
    '                height=280, margin=dict(l=0, r=60, t=10, b=0),\n'
    '                xaxis=dict(gridcolor=BORDER, ticksuffix="%", tickfont=dict(size=9)),\n'
    '                yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),\n'
    '                font=dict(color=TEXT, size=9),\n'
    '            )'
)
n3 = src.count(OLD_BAR)
src = src.replace(OLD_BAR, NEW_BAR, 1)
print(f"H3 sector bar labels: {'OK' if n3 else 'SKIP'} ({n3})")

# ══ H4: Rimuovi expander Correlazioni dalla home_tab (già nel tab Settori) ════
OLD_CORR = (
    '\ndef _render_correlations():\n'
    '    with st.expander("🔗 Correlazioni Asset — 30 giorni", expanded=False):'
)
NEW_CORR = (
    '\ndef _render_correlations():\n'
    '    pass  # v41h: rimossa dalla Home, presente nel tab Settori\n'
    '    return\n'
    '    with st.expander("🔗 Correlazioni Asset — 30 giorni", expanded=False):'
)
n4 = src.count(OLD_CORR)
src = src.replace(OLD_CORR, NEW_CORR, 1)
print(f"H4 disabilita correlazioni: {'OK' if n4 else 'SKIP'} ({n4})")

# ══ versione ══════════════════════════════════════════════════════════════════
src = src.replace("v32.1", "v32.2")
src = src.replace("home_sparklines_v321", "home_sparklines_v322")
src = src.replace("rsi_dist_v321", "rsi_dist_v322")
src = src.replace("sector_bar_v321", "sector_bar_v322")
src = src.replace("corr_matrix_v321", "corr_matrix_v322")
src = src.replace("home_refresh_v321", "home_refresh_v322")

# ══ verifica ══════════════════════════════════════════════════════════════════
checks = {
    "height 190":       "height=190, paper_bgcolor=BG",
    "height 120":       "height=120, margin=dict(l=0, r=0, t=20",
    "label_full":       "label_full",
    "v32.2":            "v32.2",
}
failed = []
print("\n-- Verifica --")
for lbl, marker in checks.items():
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

