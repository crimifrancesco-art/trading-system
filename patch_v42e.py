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

# --- Fix 1: link TradingView doppi apici dentro f-string ---
src = src.replace(
    'https://it.tradingview.com/chart/?symbol={t.replace(".MI","%3AMI")}',
    "https://it.tradingview.com/chart/?symbol={t.replace('.MI','%3AMI')}",
)
src = src.replace(
    'https://it.tradingview.com/chart/?symbol={tk.replace(".MI","%3AMI")}',
    "https://it.tradingview.com/chart/?symbol={tk.replace('.MI','%3AMI')}",
)

# --- Fix 2: {"#hex" if var else "#hex"} -> {'#hex' if var else '#hex'} ---
def fix_nested_color(text):
    pattern = r'\{"(#[0-9a-fA-F]{6})" if (\w+) else "(#[0-9a-fA-F]{6})"\}'
    def repl(m):
        return "{'" + m.group(1) + "' if " + m.group(2) + " else '" + m.group(3) + "'}"
    return re.sub(pattern, repl, text)
src = fix_nested_color(src)

# --- Fix 3: .split(',')[ dentro f-string ---
src = src.replace('.split(",")[', ".split(',')[")

# --- Fix 4: versione titolo page_title ---
for _old in [
    'page_title="Trading Scanner PRO 41.0c"',
    'page_title="Trading Scanner PRO 41.0d"',
    'page_title="Trading Scanner PRO 41.0e"',
]:
    src = src.replace(_old, 'page_title="Trading Scanner PRO 42.0e"')

# --- Fix 5: stringa titolo nel markdown ---
for _old in [
    '"Trading Scanner PRO 41.0c"',
    '"Trading Scanner PRO 41.0d"',
    '"Trading Scanner PRO 41.0e"',
]:
    src = src.replace(_old, '"Trading Scanner PRO 42.0e"')

# --- Fix 6: header H1 inline ---
src = re.sub(
    r'(# [^\n]*Trading Scanner PRO )4[12]\.[0-9][a-zA-Z]',
    r'\g<1>42.0e',
    src,
)

# --- Fix 7: expander Suggerimenti ---
src = src.replace(
    "💡 Suggerimenti v41e — Novità e roadmap",
    "💡 Suggerimenti v42e — Novità e roadmap",
)
src = re.sub(r'Suggerimenti v41[a-z]', "Suggerimenti v42e", src)
src = re.sub(r'Suggerimenti v42[a-d]', "Suggerimenti v42e", src)
src = src.replace("Implementato in v41e:", "Implementato in v42e:")
src = src.replace("Implementato in v41c:", "Implementato in v42e:")
src = src.replace("Idee per v41e:", "Idee per v43:")

# --- Fix 8: nome file nel sorgente ---
src = src.replace("Dashboard_pro_V_41c.py", "Dashboard_pro_V_42e.py")
src = src.replace("Dashboard_pro_V_41e.py", "Dashboard_pro_V_42e.py")
src = src.replace("Dashboard_pro_V_41d.py", "Dashboard_pro_V_42e.py")

# --- Fix 9: bottone Torna su fisso ---
BACK_TO_TOP_CSS = (
    "<style>\n"
    "#btt-btn {\n"
    "    position:fixed; bottom:28px; right:28px; z-index:99999;\n"
    "    background:#2962ff; color:#fff; border:none; border-radius:50%;\n"
    "    width:46px; height:46px; font-size:1.35rem; cursor:pointer;\n"
    "    box-shadow:0 4px 18px rgba(41,98,255,0.50);\n"
    "    display:flex; align-items:center; justify-content:center;\n"
    "    opacity:0; transition:opacity .28s, transform .28s;\n"
    "    transform:translateY(14px); pointer-events:none;\n"
    "}\n"
    "#btt-btn.btt-visible { opacity:1; transform:translateY(0); pointer-events:all; }\n"
    "#btt-btn:hover { background:#1a3fd4; transform:translateY(-2px) scale(1.09); }\n"
    "</style>\n"
    "<button id='btt-btn' title='Torna all\'inizio'\n"
    "  onclick='(window.parent.document.querySelector(\'section.main\')"
    "||window.parent.document.body).scrollTo({top:0,behavior:\'smooth\'})'> &#8679; </button>\n"
    "<script>\n"
    "(function(){\n"
    "  var D=[700,1600,3200];\n"
    "  function a(){\n"
    "    var s=window.parent.document.querySelector('section.main')||window.parent.document.body;\n"
    "    if(!s) return;\n"
    "    s.addEventListener('scroll',function(){\n"
    "      var b=document.getElementById('btt-btn');\n"
    "      if(!b) return;\n"
    "      if(s.scrollTop>200) b.classList.add('btt-visible');\n"
    "      else b.classList.remove('btt-visible');\n"
    "    },{passive:true});\n"
    "  }\n"
    "  D.forEach(function(d){setTimeout(a,d);});\n"
    "})();\n"
    "</script>"
)

DARK_ANCHOR = "st.markdown(DARK_CSS, unsafe_allow_html=True)"
BTT_LINE    = "st.markdown(BACK_TO_TOP_CSS, unsafe_allow_html=True)"

if DARK_ANCHOR in src and "btt-btn" not in src:
    if 'DARK_CSS = """' in src:
        src = src.replace(
            'DARK_CSS = """',
            'BACK_TO_TOP_CSS = (' + '\n' + BACK_TO_TOP_CSS.rstrip() + '\n)\n\nDARK_CSS = """',
            1,
        )
    src = src.replace(DARK_ANCHOR, DARK_ANCHOR + '\n' + BTT_LINE)

# --- Controllo sintassi ---
try:
    compile(src, str(DST), "exec")
except SyntaxError as e:
    _lines = src.splitlines()
    _start = max(0, (e.lineno or 1) - 6)
    _end   = min(len(_lines), (e.lineno or 1) + 6)
    print(f"\nSyntaxError linea {e.lineno}: {e.msg}", file=sys.stderr)
    for _i, _l in enumerate(_lines[_start:_end], start=_start + 1):
        _m = ">>>" if _i == e.lineno else "   "
        print(f"{_m} {_i:5d}: {_l}", file=sys.stderr)
    sys.exit(1)

DST.write_text(src, encoding="utf-8")
print(f"OK: generated {DST} from {src_path.name}")
