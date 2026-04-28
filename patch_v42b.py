#!/usr/bin/env python3
from pathlib import Path
import os, sys, re

SRC = 'Dashboard_pro_V_41e.py'
DST = 'Dashboard_pro_V_42b.py'

if not os.path.exists(SRC):
    print('ERR sorgente non trovato:', SRC)
    sys.exit(1)

src = Path(SRC).read_text(encoding='utf-8')
print(f'FILE {SRC}: {len(src)} chars')

src = src.replace(
    '    f"<span style=\'color:{"#00ff88" if ok else "#374151"}\'>{name.split()[0]}</span>"',
    '    ("<span style=\'color:" + ("#00ff88" if ok else "#374151") + "\'>" + name.split()[0] + "</span>")'
)

src = src.replace(
    'f"<circle cx=\'{_pts[-1].split(",")[0]}\' cy=\'{_pts[-1].split(",")[1]}\' "',
    'f"<circle cx=\'{_pts[-1].split(chr(44))[0]}\' cy=\'{_pts[-1].split(chr(44))[1]}\' "'
)

old = """    st.markdown("---")
    st.markdown('<div class="section-pill">📅 EARNINGS CALENDAR v41 — Prossimi earnings da Watchlist + Scanner</div>',
                unsafe_allow_html=True)
    _earn_tickers = set()
"""
new = """    st.markdown("---")
    with st.expander("📅 EARNINGS CALENDAR v42b — Prossimi earnings da Watchlist + Scanner", expanded=False):
        _earn_tickers = set()
"""
src = src.replace(old, new, 1)

old2 = '''def _render_ai_explainer_v41(df_source, tab_name="PRO"):
    """AI Signal Explainer — multi-provider con fallback automatico."""
    st.markdown(
        '<div class="section-pill">🤖 MODULO 2 — AI ANALYST · Setup · Target · Invalidazione · Rischio</div>',
        unsafe_allow_html=True)
    st.caption("Fallback automatico: Gemini (free) → Groq (free) → OpenRouter → Claude · Clicca 🧠 Analizza su ogni ticker")
'''
new2 = '''def _render_ai_explainer_v41(df_source, tab_name="PRO"):
    """AI Signal Explainer — multi-provider con fallback automatico."""
    st.markdown(
        '<div class="section-pill">🤖 MODULO 2 — AI ANALYST · Setup · Target · Invalidazione · Rischio</div>',
        unsafe_allow_html=True)
    st.caption("Fallback automatico: Gemini (free) → Groq (free) → OpenRouter → Claude · Clicca 🧠 Analizza su ogni ticker")
    st.markdown("<style>.ai2-table{margin-top:6px;border:1px solid #1f2937;border-radius:10px;overflow:hidden;background:#0b1220}.ai2-head,.ai2-row{display:grid;grid-template-columns:1.35fr .68fr .7fr .7fr 1fr;gap:10px;align-items:center}.ai2-head{padding:10px 12px;background:#0b1326;border-bottom:1px solid #1f2937;color:#38bdf8;font-size:.74rem;font-weight:700;letter-spacing:.04em}.ai2-row{padding:12px;border-bottom:1px solid rgba(42,46,57,.75)}.ai2-row:last-child{border-bottom:none}.ai2-ticker{font-family:Courier New,monospace;font-weight:700;color:#00ff88;font-size:1.0rem}.ai2-name{color:#787b86;font-size:.78rem;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}.ai2-badge{display:inline-block;padding:4px 10px;border-radius:8px;font-weight:700;font-size:.76rem;border:1px solid transparent}.ai2-badge.pro{color:#00ff88;background:rgba(0,255,136,.10);border-color:rgba(0,255,136,.25)}.ai2-badge.strong{color:#22c55e;background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.25)}.ai2-mono{font-family:Courier New,monospace;font-weight:700}.ai2-cta{display:block;text-align:center;padding:10px 14px;border-radius:9px;border:1px solid #374151;background:#111827;color:#f9a8d4;font-weight:600}.ai2-cta:hover{border-color:#4b5563;background:#141c2b}@media (max-width: 1100px){.ai2-head,.ai2-row{grid-template-columns:1.2fr .7fr .7fr .7fr .9fr}}</style>", unsafe_allow_html=True)
'''
src = src.replace(old2, new2, 1)

src = src.replace(
'''    for _t in tickers:
        try:
            _info = _yf.Ticker(_t).calendar
            if _info is None:
                continue
''',
'''    _rate_limited = False
    for _t in tickers:
        try:
            _info = _yf.Ticker(_t).calendar
            if _info is None:
                continue
''',1)

src = src.replace(
'''        except Exception:
            continue
    return sorted(_results, key=lambda x: x["Giorni"])
''',
'''        except Exception as _earn_err:
            _msg = str(_earn_err).lower()
            if "too many requests" in _msg or "rate limit" in _msg or "429" in _msg:
                _rate_limited = True
            continue
    if _rate_limited and not _results:
        return [{"Ticker":"RATE_LIMIT","Earnings Date":"—","Giorni":999,
                 "Badge":"⏳ Yahoo rate limited — riprova tra poco","_color":"#f59e0b",
                 "EPS Est":"","Rev Est":""}]
    return sorted(_results, key=lambda x: x["Giorni"])
''',1)

src = src.replace(
    '                except Exception as ae:\n                    st.warning(f"Impossibile scaricare dati: {ae}")\n                    if not apd: continue\n',
    '                except Exception as ae:\n                    _ae = str(ae)\n                    if ("Too Many Requests" in _ae) or ("rate limit" in _ae.lower()) or ("429" in _ae):\n                        st.warning("⏳ Yahoo Finance è temporaneamente rate-limited in 🤖 AI Assistant. Attendi 1-2 minuti e riprova, oppure analizza meno ticker per volta.")\n                    else:\n                        st.warning(f"Impossibile scaricare dati: {_ae}")\n                    if not apd: continue\n',
    1
)

src = src.replace(
    '📅 EARNINGS CALENDAR v41 — Prossimi earnings da Watchlist + Scanner',
    '📅 EARNINGS CALENDAR v42b — Prossimi earnings da Watchlist + Scanner'
)
src = src.replace('🤖 MODULO 2 AI ANALYST', '🤖 MODULO 2 AI ANALYST v42b')
src = src.replace('Trading Scanner PRO 41.0c', 'Trading Scanner PRO 42.0b')
src = re.sub(r'(MOMENTUM ALERTS )v41e', r'\\1v42b', src)
src = re.sub(r'(EARNINGS CALENDAR )v41', r'\\1v42b', src)
src = src.replace('Suggerimenti v41e — Novità e roadmap', 'Suggerimenti v42b — Novità e roadmap')

compile(src, DST, 'exec')
Path(DST).write_text(src, encoding='utf-8')
print(f'OK wrote {DST} {len(src)} chars')
