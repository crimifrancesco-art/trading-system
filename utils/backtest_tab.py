"""
backtest_tab.py  —  Upgrade #5 — v32.0
================================
Tab "📈 Backtest" per il dashboard v28.
Incolla questa funzione in Dashboard_pro V_28.0.py e aggiungila ai tabs.

Dipende da:
  - utils.db: load_signals, signal_summary_stats, update_signal_performance
  - st_aggrid, plotly

Come funziona:
  1. Ogni volta che gira lo scanner, save_signals() registra nella tabella
     signals tutti i segnali con il prezzo di entrata del giorno
  2. update_signal_performance() aggiorna i prezzi forward (+1d,+5d,+10d,+20d)
     scaricando da yfinance solo quando mancano
  3. Questo tab li legge, calcola statistiche e mostra grafici interattivi

Struttura:
  • 📊 Riepilogo — tabella aggregata win rate / avg return per tipo segnale
  • 📈 Equity curve — curva cumulata se avessi comprato ogni segnale
  • 🔍 Dettaglio segnali — griglia filtrabile con tutti i segnali registrati
  • 🔄 Aggiorna performance — pulsante per aggiornare prezzi forward manualmente
"""

import urllib.request, json, base64, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np

# ── Palette TV (stile Blue Chip Dip) ───────────────────────────────────────
_TV_BG    = "#131722"; _TV_PANEL = "#1e222d"; _TV_BORDER= "#2a2e39"
_TV_GREEN = "#26a69a"; _TV_RED   = "#ef5350"; _TV_GOLD  = "#ffd700"
_TV_BLUE  = "#2962ff"; _TV_CYAN  = "#50c4e0"; _TV_GRAY  = "#787b86"
_TV_TEXT  = "#d1d4dc"; _TV_ORANGE= "#ff9800"; _TV_PURPLE= "#9c27b0"

# ── Mappe legenda per strategia ──────────────────────────────────────────────
# PNG caricate da Fingrad (in assets/) oppure SVG generati localmente
_LEG_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
_LEG_OUTPUTS_DIR = "/mnt/user-data/outputs"   # dev environment

_STRATEGY_LEGEND = {
    "RSI+VWAP": (
        "leg_rsi_vwap.png", "image/png",
        "RSI + VWAP — Intraday Strategy",
        ["▲ LONG entry: Price > VWAP + RSI sale da sotto 30",
         "▼ EXIT: Price < VWAP o RSI scende da sopra 70",
         "Stop-loss: sotto il minimo del segnale entry",
         "Timeframe ideale: 15min–1h"],
    ),
    "ADX+EMA": (
        "leg_adx_ema.svg", "image/svg+xml",
        "ADX + EMA Cross",
        ["▲ LONG: EMA20 incrocia sopra EMA50 + ADX > 25",
         "▼ EXIT: EMA20 < EMA50 o ADX < 25",
         "ADX > 25 = trend forte, < 25 = mercato laterale",
         "Timeframe: daily / 4h per swing"],
    ),
    "MACD": (
        "leg_macd_ema.svg", "image/svg+xml",
        "MACD (12,26,9)",
        ["▲ LONG entry: MACD histogram incrocia sopra 0",
         "▼ EXIT: MACD histogram incrocia sotto 0",
         "Conferma con EMA20 > EMA50",
         "Divergenza MACD/prezzo = segnale forte di inversione"],
    ),
    "Keltner Channel": (
        "leg_keltner_macd.png", "image/png",
        "Keltner Channel (EMA20 ± 2×ATR)",
        ["▲ LONG: prezzo chiude sopra la banda inferiore dal basso",
         "▼ EXIT: prezzo raggiunge o supera la banda superiore",
         "Banda media = EMA20 — funge da supporto/resistenza dinamico",
         "Combina bene con RSI per filtrare falsi breakout"],
    ),
    "Donchian Channel": (
        "__svg_donchian__", "image/svg+xml",
        "Donchian Channel (Breakout n=20)",
        ["▲ LONG entry: candela tocca/supera la banda superiore 20-day",
         "▼ EXIT: candela tocca la banda inferiore 20-day",
         "Strategia breakout pura — funziona bene in trend forti",
         "Evitare in mercati laterali (molti falsi segnali)"],
    ),
    "RSI+Bollinger": (
        "leg_bb_rsi.png", "image/png",
        "RSI + Bollinger Bands — Mean Reversion",
        ["▲ LONG: prezzo sotto la banda inferiore BB + RSI < 35",
         "▼ EXIT: prezzo sopra la banda superiore BB o RSI > 65",
         "Strategia mean-reversion — compra oversold, vendi overbought",
         "Più efficace su titoli con trend neutro o range-bound"],
    ),
    "OBV+Hull MA": (
        "__svg_obv_hull__", "image/svg+xml",
        "OBV + Hull Moving Average",
        ["▲ LONG: prezzo incrocia sopra Hull MA + OBV in salita (> OBV MA10)",
         "▼ EXIT: prezzo scende sotto Hull MA o OBV inizia a scendere",
         "Hull MA reagisce più velocemente all'EMA standard",
         "OBV conferma che il volume supporta il movimento di prezzo"],
    ),
    "SAR+Chop": (
        "leg_sar.png", "image/png",
        "Parabolic SAR + Chop Zone",
        ["▲ LONG: SAR passa sotto le candele (trend bullish) + Chop Zone verde/blu",
         "▼ EXIT: SAR passa sopra le candele o Chop Zone diventa rossa",
         "Chop Zone BLU = trend forte (ADX>40) · VERDE = moderato · ROSSO = laterale",
         "Non operare quando Chop Zone è ROSSA — evita falsi segnali"],
    ),
    "ADX+Pattern": (
        "leg_adx_ema.svg", "image/svg+xml",
        "ADX + Piercing Line Pattern",
        ["▲ LONG: Piercing Line candlestick (reversal pattern) + ADX > 25",
         "Piercing Line: candela bearish seguita da bullish che chiude >50% del body precedente",
         "▼ EXIT: ADX scende sotto 20 (trend perde forza)",
         "Più efficace in presenza di supporti tecnici chiari"],
    ),
}


def _svg_donchian() -> str:
    """SVG inline che illustra la logica Donchian Channel."""
    return """<svg viewBox="0 0 400 240" xmlns="http://www.w3.org/2000/svg" style="background:#131722;border-radius:8px">
  <defs>
    <linearGradient id="dcfill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#26a69a" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#26a69a" stop-opacity="0.05"/>
    </linearGradient>
  </defs>
  <!-- Upper band -->
  <polyline points="20,60 80,58 140,55 200,52 260,50 320,48 380,45"
    fill="none" stroke="#26a69a" stroke-width="2.5" stroke-dasharray="6,3"/>
  <!-- Lower band -->
  <polyline points="20,170 80,175 140,172 200,170 260,168 320,165 380,162"
    fill="none" stroke="#ef5350" stroke-width="2.5" stroke-dasharray="6,3"/>
  <!-- Fill between bands -->
  <polygon points="20,60 80,58 140,55 200,52 260,50 320,48 380,45 380,162 320,165 260,168 200,170 140,172 80,175 20,170"
    fill="url(#dcfill)"/>
  <!-- Price bars (candele stilizzate) -->
  <rect x="30"  y="148" width="14" height="22" fill="#ef5350" rx="1"/>
  <rect x="55"  y="135" width="14" height="30" fill="#26a69a" rx="1"/>
  <rect x="80"  y="125" width="14" height="28" fill="#26a69a" rx="1"/>
  <rect x="105" y="115" width="14" height="25" fill="#26a69a" rx="1"/>
  <rect x="130" y="100" width="14" height="28" fill="#26a69a" rx="1"/>
  <rect x="155" y="90"  width="14" height="30" fill="#26a69a" rx="1"/>
  <rect x="180" y="78"  width="14" height="26" fill="#26a69a" rx="1"/>
  <!-- ENTRY marker -->
  <rect x="180" y="68"  width="14" height="26" fill="#26a69a" rx="1" opacity="0.5"/>
  <polygon points="187,38 194,52 180,52" fill="#26a69a"/>
  <text x="200" y="50" fill="#26a69a" font-size="11" font-family="Courier New" font-weight="bold">▲ ENTRY</text>
  <!-- Candele dopo entry -->
  <rect x="205" y="68"  width="14" height="25" fill="#26a69a" rx="1"/>
  <rect x="230" y="60"  width="14" height="28" fill="#26a69a" rx="1"/>
  <rect x="255" y="55"  width="14" height="25" fill="#26a69a" rx="1"/>
  <rect x="280" y="50"  width="14" height="28" fill="#ef5350" rx="1"/>
  <rect x="305" y="60"  width="14" height="35" fill="#ef5350" rx="1"/>
  <rect x="330" y="82"  width="14" height="40" fill="#ef5350" rx="1"/>
  <!-- EXIT marker near lower band -->
  <rect x="330" y="155" width="14" height="30" fill="#ef5350" rx="1" opacity="0.8"/>
  <polygon points="337,195 330,182 344,182" fill="#ef5350"/>
  <text x="248" y="208" fill="#ef5350" font-size="11" font-family="Courier New" font-weight="bold">▼ EXIT (lower band)</text>
  <!-- Labels -->
  <text x="22" y="50"  fill="#26a69a" font-size="10" font-family="Courier New">Upper (20-day max)</text>
  <text x="22" y="186" fill="#ef5350" font-size="10" font-family="Courier New">Lower (20-day min)</text>
  <text x="150" y="228" fill="#787b86" font-size="9" font-family="Courier New">Donchian Channel — Breakout Strategy</text>
</svg>"""


def _svg_obv_hull() -> str:
    """SVG inline che illustra la logica OBV + Hull MA."""
    return """<svg viewBox="0 0 400 240" xmlns="http://www.w3.org/2000/svg" style="background:#131722;border-radius:8px">
  <!-- Price bars -->
  <rect x="20"  y="155" width="14" height="35" fill="#ef5350" rx="1"/>
  <rect x="42"  y="160" width="14" height="30" fill="#ef5350" rx="1"/>
  <rect x="64"  y="158" width="14" height="28" fill="#ef5350" rx="1"/>
  <rect x="86"  y="145" width="14" height="30" fill="#26a69a" rx="1"/>
  <rect x="108" y="132" width="14" height="28" fill="#26a69a" rx="1"/>
  <rect x="130" y="115" width="14" height="30" fill="#26a69a" rx="1"/>
  <rect x="152" y="100" width="14" height="28" fill="#26a69a" rx="1"/>
  <rect x="174" y="88"  width="14" height="26" fill="#26a69a" rx="1"/>
  <rect x="196" y="78"  width="14" height="25" fill="#26a69a" rx="1"/>
  <rect x="218" y="70"  width="14" height="28" fill="#26a69a" rx="1"/>
  <rect x="240" y="65"  width="14" height="30" fill="#26a69a" rx="1"/>
  <rect x="262" y="72"  width="14" height="35" fill="#ef5350" rx="1"/>
  <rect x="284" y="85"  width="14" height="40" fill="#ef5350" rx="1"/>
  <rect x="306" y="100" width="14" height="38" fill="#ef5350" rx="1"/>
  <!-- Hull MA line (arancione, smooth) -->
  <polyline points="20,178 42,175 64,170 86,158 108,145 130,128 152,112 174,98 196,85 218,76 240,72 262,80 284,96 306,118 330,140 352,158"
    fill="none" stroke="#ff9800" stroke-width="3" stroke-linejoin="round"/>
  <!-- ENTRY: price crosses above Hull MA -->
  <rect x="86" y="140" width="14" height="30" fill="#26a69a" rx="1" opacity="0.6"/>
  <polygon points="93,112 100,126 86,126" fill="#26a69a"/>
  <text x="106" y="120" fill="#26a69a" font-size="11" font-family="Courier New" font-weight="bold">▲ ENTRY</text>
  <text x="106" y="133" fill="#787b86" font-size="9" font-family="Courier New">price &gt; Hull MA</text>
  <!-- OBV panel -->
  <rect x="20" y="200" width="370" height="1" fill="#2a2e39"/>
  <text x="22" y="215" fill="#787b86" font-size="9" font-family="Courier New">OBV</text>
  <polyline points="20,230 42,228 64,226 86,220 108,213 130,205 152,198 174,193 196,188 218,184 240,181 262,186 284,193 306,200"
    fill="none" stroke="#2962ff" stroke-width="2"/>
  <!-- OBV rising annotation -->
  <text x="108" y="210" fill="#2962ff" font-size="9" font-family="Courier New">OBV ↑ conferma</text>
  <!-- Hull MA label -->
  <text x="310" y="136" fill="#ff9800" font-size="10" font-family="Courier New">Hull MA</text>
  <text x="130" y="248" fill="#787b86" font-size="9" font-family="Courier New">OBV + Hull MA — Trend + Volume Confirm</text>
</svg>"""


def _read_legend_image(filename: str) -> tuple[str, str]:
    """
    Cerca il file immagine legenda in:
      1. assets/ relativo al file corrente
      2. /mnt/user-data/outputs/ (dev)
    Ritorna (base64_str, mime_type).
    Per i placeholder __svg_*__ ritorna l'SVG inline direttamente (non base64).
    """
    # SVG inline dedicati — non richiedono file esterni
    if filename == "__svg_donchian__":
        return _svg_donchian(), "__inline_svg__"
    if filename == "__svg_obv_hull__":
        return _svg_obv_hull(), "__inline_svg__"

    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
    for directory in [_LEG_ASSETS_DIR, _LEG_OUTPUTS_DIR]:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode(), mime
    return "", mime


def _render_strategy_legend(strategy: str) -> None:
    if strategy not in _STRATEGY_LEGEND:
        return
    filename, mime, title, bullets = _STRATEGY_LEGEND[strategy]
    b64, actual_mime = _read_legend_image(filename)

    with st.expander(f"📖 Guida visuale — {title}", expanded=False):
        col_img, col_txt = st.columns([1.2, 1])
        with col_img:
            if actual_mime == "__inline_svg__":
                # SVG inline: renderizza direttamente come HTML
                st.markdown(
                    f'<div style="border-radius:8px;overflow:hidden;border:1px solid #2a2e39">'
                    f'{b64}</div>',
                    unsafe_allow_html=True,
                )
            elif b64:
                st.markdown(
                    f'<img src="data:{actual_mime};base64,{b64}" '
                    f'style="width:100%;border-radius:6px;border:1px solid #2a2e39">',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:#1e222d;border:1px dashed #2a2e39;'
                    f'border-radius:6px;padding:20px;text-align:center;color:#787b86">'
                    f'📊 {title}<br><small>Metti l\'immagine in <code>assets/{filename}</code></small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with col_txt:
            st.markdown(
                f'<div style="background:#131722;border-radius:6px;padding:12px 14px">'
                f'<div style="color:#ffd700;font-weight:700;font-size:0.85rem;'
                f'margin-bottom:8px">📋 Regole operative</div>'
                + "".join([
                    f'<div style="color:#d1d4dc;font-size:0.78rem;padding:3px 0;'
                    f'border-left:2px solid {"#26a69a" if "▲" in b else "#ef5350" if "▼" in b else "#787b86"};'
                    f'padding-left:7px;margin:3px 0">{b}</div>'
                    for b in bullets
                ])
                + '</div>',
                unsafe_allow_html=True,
            )

# ── Fetch OHLCV per strategy chart ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _bt_fetch_ohlcv(symbol: str, range_: str = "1y") -> pd.DataFrame:
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval=1d&range={range_}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        q  = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s"),
            "open":   q.get("open",  []),
            "high":   q.get("high",  []),
            "low":    q.get("low",   []),
            "close":  q.get("close", []),
            "volume": q.get("volume",[]),
        }).dropna(subset=["close"]).reset_index(drop=True)
        df["date"] = df["date"].dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# ── Indicatori locali ───────────────────────────────────────────────────────
def _bt_ema(s, n):    return s.ewm(span=n, adjust=False).mean()
def _bt_rsi(s, n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _bt_macd(s):
    ml=s.ewm(span=12).mean()-s.ewm(span=26).mean()
    sl=ml.ewm(span=9).mean(); return ml,sl,ml-sl
def _bt_vwap(c,v,win=20):
    return (c*v).rolling(win).sum()/v.rolling(win).sum().replace(0,np.nan)
def _bt_adx(h,lo,c,period=14):
    h_=h.values.astype(float); l_=lo.values.astype(float); c_=c.values.astype(float)
    n=len(c_); tr_,dp_,dn_=[],[],[]
    for i in range(1,n):
        tr_.append(max(h_[i]-l_[i],abs(h_[i]-c_[i-1]),abs(l_[i]-c_[i-1])))
        up,dn=h_[i]-h_[i-1],l_[i-1]-l_[i]
        dp_.append(up if up>dn and up>0 else 0)
        dn_.append(dn if dn>up and dn>0 else 0)
    out=np.full(n,np.nan)
    if len(tr_)<period: return pd.Series(out,index=c.index)
    atr=np.mean(tr_[:period]); dp=np.mean(dp_[:period]); dn=np.mean(dn_[:period])
    dx=[]
    for i in range(period,len(tr_)):
        atr=atr-atr/period+tr_[i]; dp=dp-dp/period+dp_[i]; dn=dn-dn/period+dn_[i]
        dip=100*dp/atr if atr>0 else 0; din=100*dn/atr if atr>0 else 0
        dx.append(100*abs(dip-din)/(dip+din) if (dip+din)>0 else 0)
    if len(dx)>=period:
        av=np.mean(dx[:period]); st2=period+period
        if st2<n: out[st2]=av
        for k in range(1,len(dx)-period+1):
            av=(av*(period-1)+dx[period-1+k])/period
            if st2+k<n: out[st2+k]=av
    return pd.Series(out,index=c.index)

# ── v34: Nuovi indicatori helper ─────────────────────────────────────────────

def _bt_keltner(c, h, lo, n=20, mult=2.0):
    """Keltner Channel: EMA(n) ± mult * ATR(n)."""
    ema  = _bt_ema(c, n)
    tr   = pd.concat([h-lo,
                      (h-c.shift()).abs(),
                      (lo-c.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(span=n, adjust=False).mean()
    return ema, ema + mult*atr, ema - mult*atr   # mid, upper, lower

def _bt_donchian(h, lo, n=20):
    """Donchian Channel: rolling max/min su n periodi."""
    return h.rolling(n).max(), lo.rolling(n).min()   # upper, lower

def _bt_bollinger(c, n=20, k=2.0):
    """Bollinger Bands: SMA(n) ± k*std."""
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    return mid, mid + k*std, mid - k*std   # mid, upper, lower

def _bt_obv(c, v):
    """On-Balance Volume."""
    sign = np.sign(c.diff().fillna(0))
    return (sign * v).cumsum()

def _bt_hma(c, n=20):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    def _wma(s, p):
        w = np.arange(1, p+1, dtype=float)
        return s.rolling(p).apply(lambda x: np.dot(x, w[-len(x):]) / w[-len(x):].sum(), raw=True)
    half = max(int(n/2), 2)
    sq   = max(int(np.sqrt(n)), 2)
    raw  = 2 * _wma(c, half) - _wma(c, n)
    return _wma(raw, sq)

def _bt_sar(h, lo, af_start=0.02, af_max=0.2):
    """Parabolic SAR — restituisce (sar_series, bull_series)."""
    h_ = h.values.astype(float); l_ = lo.values.astype(float); n = len(h_)
    sar = np.zeros(n); bull = np.ones(n, dtype=bool)
    ep = h_[0]; af = af_start; sar[0] = l_[0]
    for i in range(1, n):
        pb = bull[i-1]; ps = sar[i-1]
        if pb:
            ns = min(ps + af*(ep-ps), l_[i-1], l_[i-2] if i>=2 else l_[i-1])
            if l_[i] < ns:
                bull[i]=False; sar[i]=ep; ep=l_[i]; af=af_start
            else:
                bull[i]=True; sar[i]=ns
                if h_[i]>ep: ep=h_[i]; af=min(af+af_start, af_max)
        else:
            ns = max(ps + af*(ep-ps), h_[i-1], h_[i-2] if i>=2 else h_[i-1])
            if h_[i] > ns:
                bull[i]=True; sar[i]=ep; ep=h_[i]; af=af_start
            else:
                bull[i]=False; sar[i]=ns
                if l_[i]<ep: ep=l_[i]; af=min(af+af_start, af_max)
    return pd.Series(sar, index=h.index), pd.Series(bull.astype(int), index=h.index)

def _bt_chop_zone(h, lo, c, n=14):
    """
    Chop Zone proxy basato su ADX + EMA slope.
    Ritorna colori per ogni barra: 'blue'=strong trend, 'green'=moderate,
    'yellow'=caution, 'red'=choppy.
    """
    adx = _bt_adx(h, lo, c, n)
    ema = _bt_ema(c, n)
    slope = ema.diff(3) / c * 100  # slope % EMA
    colors = []
    for i in range(len(c)):
        a = adx.iloc[i] if not np.isnan(adx.iloc[i]) else 0
        s = abs(slope.iloc[i]) if not np.isnan(slope.iloc[i]) else 0
        if   a > 40 and s > 0.5: colors.append("blue")
        elif a > 25 and s > 0.2: colors.append("green")
        elif a > 15:             colors.append("yellow")
        else:                    colors.append("red")
    return colors

def _bt_piercing_line(o, c):
    """
    Piercing Line pattern detector.
    Bearish candle seguita da bullish che chiude >50% del body precedente.
    """
    o_ = o.values.astype(float); c_ = c.values.astype(float)
    signals = np.zeros(len(c_), dtype=bool)
    for i in range(1, len(c_)):
        prev_bear = c_[i-1] < o_[i-1]
        curr_bull = c_[i]   > o_[i]
        mid_prev  = (o_[i-1] + c_[i-1]) / 2
        if prev_bear and curr_bull and c_[i] > mid_prev and o_[i] < c_[i-1]:
            signals[i] = True
    return pd.Series(signals, index=c.index)


def _bt_detect_signals(df: pd.DataFrame, strategy: str):
    """Detecta Entry/Exit per la strategia — ritorna (e_d, e_p, x_d, x_p)."""
    c  = df["close"].reset_index(drop=True)
    h  = df["high"].reset_index(drop=True)
    lo = df["low"].reset_index(drop=True)
    v  = df["volume"].fillna(0).reset_index(drop=True)
    o  = df["open"].reset_index(drop=True)
    dt = df["date"].reset_index(drop=True)

    ema20 = _bt_ema(c, 20).values;  ema50 = _bt_ema(c, 50).values
    rsi_s = _bt_rsi(c).values
    ml_s, _, mh_s = _bt_macd(c);   mh_ = mh_s.values
    vwap_ = _bt_vwap(c, v, 20).values
    adx_  = _bt_adx(h, lo, c, 14).values
    _, kelt_up, kelt_dn = _bt_keltner(c, h, lo)
    ku_ = kelt_up.values; kd_ = kelt_dn.values
    don_up, don_dn = _bt_donchian(h, lo)
    du_ = don_up.values; dd_ = don_dn.values
    _, bb_up, bb_dn = _bt_bollinger(c)
    bu_ = bb_up.values; bd_ = bb_dn.values
    obv_  = _bt_obv(c, v).values
    hma_  = _bt_hma(c, 20).values
    obv_ma = _bt_ema(pd.Series(obv_), 10).values
    sar_s, bull_s = _bt_sar(h, lo)
    sar_ = sar_s.values; bull_ = bull_s.values
    chop_  = _bt_chop_zone(h, lo, c)
    pierc_ = _bt_piercing_line(o, c).values

    e_d, e_p, x_d, x_p = [], [], [], []
    in_t = False

    for i in range(30, len(c)-1):
        ci  = float(c.iloc[i])
        ri  = float(rsi_s[i])  if not np.isnan(rsi_s[i])  else 50.0
        rp  = float(rsi_s[i-1])if not np.isnan(rsi_s[i-1])else 50.0
        vi  = float(vwap_[i])  if not np.isnan(vwap_[i])  else ci
        ai  = float(adx_[i])   if not np.isnan(adx_[i])   else 0.0
        ent = ex_ = False

        if strategy == "RSI+VWAP":
            ent = (rp < 30) and (ri >= 30) and (ci > vi)
            ex_ = ((rp > 70) and (ri <= 70)) or (in_t and ci < vi and ri > 65)
        elif strategy == "ADX+EMA":
            ent = (ema20[i-1] <= ema50[i-1]) and (ema20[i] > ema50[i]) and (ai > 25)
            ex_ = (ema20[i] < ema50[i]) or (in_t and ai < 25)
        elif strategy == "MACD":
            mv = mh_[i] if not np.isnan(mh_[i]) else 0
            mp = mh_[i-1] if not np.isnan(mh_[i-1]) else 0
            ent = (mp <= 0) and (mv > 0);  ex_ = (mp >= 0) and (mv < 0)
        elif strategy == "Keltner Channel":
            kd_i = kd_[i]  if not np.isnan(kd_[i])  else ci
            kd_p = kd_[i-1]if not np.isnan(kd_[i-1])else ci
            ku_i = ku_[i]  if not np.isnan(ku_[i])  else ci
            ent = (float(c.iloc[i-1]) <= kd_p) and (ci > kd_i)
            ex_ = in_t and (ci >= ku_i * 0.995)
        elif strategy == "Donchian Channel":
            du_i = du_[i] if not np.isnan(du_[i]) else ci
            dd_i = dd_[i] if not np.isnan(dd_[i]) else ci
            ent = (ci >= du_i * 0.999)
            ex_ = in_t and (ci <= dd_i * 1.001)
        elif strategy == "RSI+Bollinger":
            bd_i = bd_[i] if not np.isnan(bd_[i]) else ci
            bu_i = bu_[i] if not np.isnan(bu_[i]) else ci
            ent = (ci <= bd_i) and (ri < 35)
            ex_ = in_t and ((ci >= bu_i) or (ri > 65))
        elif strategy == "OBV+Hull MA":
            hma_i = hma_[i]  if not np.isnan(hma_[i])  else ci
            hma_p = hma_[i-1]if not np.isnan(hma_[i-1])else ci
            obv_rise = obv_[i] > obv_ma[i] if not np.isnan(obv_ma[i]) else (obv_[i] > obv_[i-1])
            ent = (float(c.iloc[i-1]) < hma_p) and (ci >= hma_i) and obv_rise
            ex_ = in_t and ((ci < hma_i) or (not obv_rise))
        elif strategy == "SAR+Chop":
            bull_i = bool(bull_[i]);  bull_p = bool(bull_[i-1])
            chop_ok = chop_[i] in ("blue", "green")
            ent = (not bull_p) and bull_i and chop_ok
            ex_ = in_t and (not bull_i or chop_[i] == "red")
        elif strategy == "ADX+Pattern":
            ent = bool(pierc_[i]) and (ai > 25)
            ex_ = in_t and (ai < 20)

        if not in_t and ent:
            e_d.append(str(dt.iloc[i])[:10]); e_p.append(ci); in_t = True
        elif in_t and ex_:
            x_d.append(str(dt.iloc[i])[:10]); x_p.append(ci); in_t = False

    return e_d, e_p, x_d, x_p



def _bt_render_strategy_chart(ticker: str, strategy: str, range_: str = "1y") -> None:
    with st.spinner(f"⏳ Caricamento dati {ticker} ({range_})..."):
        df = _bt_fetch_ohlcv(ticker, range_)
    if df.empty:
        st.warning(f"⚠️ Dati non disponibili per {ticker}"); return

    c=df["close"]; h=df["high"]; lo=df["low"]; v=df["volume"].fillna(0); o=df["open"]
    dt=[str(d)[:10] for d in df["date"]]
    ema20=_bt_ema(c,20); ema50=_bt_ema(c,50); ema200=_bt_ema(c,200)
    rsi_s=_bt_rsi(c)
    ml,sl,mh=_bt_macd(c); hist_colors=[_TV_GREEN if x>=0 else _TV_RED for x in mh]
    vwap_=_bt_vwap(c,v,20)
    adx_=_bt_adx(h,lo,c,14)
    _, kelt_up, kelt_dn = _bt_keltner(c, h, lo)
    don_up, don_dn      = _bt_donchian(h, lo)
    _, bb_up, bb_dn     = _bt_bollinger(c)
    obv_s               = _bt_obv(c, v)
    hma_s               = _bt_hma(c, 20)
    sar_s, bull_s       = _bt_sar(h, lo)
    chop_colors         = _bt_chop_zone(h, lo, c)
    pierc_s             = _bt_piercing_line(o, c)

    # Mappa pannello Row 4
    row4_titles = {
        "RSI+VWAP":        "RSI (14)  ·  Zone 30/70",
        "ADX+EMA":         "ADX (14)  ·  Soglia 25",
        "MACD":            "MACD (12,26,9)",
        "Keltner Channel": "Keltner Channel  ·  EMA±2ATR",
        "Donchian Channel":"Donchian Channel  ·  Breakout n=20",
        "RSI+Bollinger":   "Bollinger Bands (20,2)  +  RSI",
        "OBV+Hull MA":     "OBV  +  Hull MA (20)",
        "SAR+Chop":        "Chop Zone  (ADX proxy)",
        "ADX+Pattern":     "ADX (14)  ·  Soglia 25  +  Piercing Line",
    }

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.12, 0.18, 0.18],
        vertical_spacing=0.02,
        subplot_titles=["", "", "RSI (14)", row4_titles.get(strategy, "Indicatore")],
    )

    # ── Row 1: Candele + overlay specifico per strategia ─────────────────
    fig.add_trace(go.Candlestick(x=dt,
        open=o.values, high=h.values, low=lo.values, close=c.values,
        name="Price",
        increasing=dict(fillcolor=_TV_GREEN, line=dict(color=_TV_GREEN, width=1)),
        decreasing=dict(fillcolor=_TV_RED,   line=dict(color=_TV_RED,   width=1)),
        showlegend=False), row=1, col=1)

    # Overlay EMA base (su tutti)
    fig.add_trace(go.Scatter(x=dt, y=ema20, mode="lines", name="EMA20",
        line=dict(color="#26c6da", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=ema50, mode="lines", name="EMA50",
        line=dict(color=_TV_GOLD, width=1.2)), row=1, col=1)

    # Overlay specifico per strategia
    if strategy == "RSI+VWAP":
        fig.add_trace(go.Scatter(x=dt, y=vwap_, mode="lines", name="VWAP",
            line=dict(color=_TV_ORANGE, width=2)), row=1, col=1)
    elif strategy == "Keltner Channel":
        fig.add_trace(go.Scatter(x=dt, y=kelt_up, mode="lines", name="Kelt Upper",
            line=dict(color=_TV_CYAN, width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=kelt_dn, mode="lines", name="Kelt Lower",
            line=dict(color=_TV_CYAN, width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dt + dt[::-1],
            y=kelt_up.tolist() + kelt_dn.tolist()[::-1],
            fill="toself", fillcolor="rgba(80,196,224,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=1, col=1)
    elif strategy == "Donchian Channel":
        fig.add_trace(go.Scatter(x=dt, y=don_up, mode="lines", name="Don Upper",
            line=dict(color=_TV_GREEN, width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=don_dn, mode="lines", name="Don Lower",
            line=dict(color=_TV_RED, width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dt + dt[::-1],
            y=don_up.tolist() + don_dn.tolist()[::-1],
            fill="toself", fillcolor="rgba(38,166,154,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=1, col=1)
    elif strategy == "RSI+Bollinger":
        fig.add_trace(go.Scatter(x=dt, y=bb_up, mode="lines", name="BB Upper",
            line=dict(color=_TV_RED, width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=bb_dn, mode="lines", name="BB Lower",
            line=dict(color=_TV_GREEN, width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dt + dt[::-1],
            y=bb_up.tolist() + bb_dn.tolist()[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=1, col=1)
    elif strategy == "OBV+Hull MA":
        fig.add_trace(go.Scatter(x=dt, y=hma_s, mode="lines", name="Hull MA",
            line=dict(color=_TV_ORANGE, width=2.2)), row=1, col=1)
    elif strategy == "SAR+Chop":
        sar_bull = [float(sar_s.iloc[i]) if bool(bull_s.iloc[i]) else None for i in range(len(sar_s))]
        sar_bear = [float(sar_s.iloc[i]) if not bool(bull_s.iloc[i]) else None for i in range(len(sar_s))]
        fig.add_trace(go.Scatter(x=dt, y=sar_bull, mode="markers",
            marker=dict(color=_TV_GREEN, size=4, symbol="circle"), name="SAR ▲"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=sar_bear, mode="markers",
            marker=dict(color=_TV_RED, size=4, symbol="circle"), name="SAR ▼"), row=1, col=1)
    elif strategy == "ADX+Pattern":
        # Evidenzia le candele Piercing Line
        pierce_x = [dt[i] for i in range(len(dt)) if pierc_s.iloc[i]]
        pierce_y = [float(lo.iloc[i]) * 0.997 for i in range(len(dt)) if pierc_s.iloc[i]]
        if pierce_x:
            fig.add_trace(go.Scatter(x=pierce_x, y=pierce_y, mode="markers",
                marker=dict(symbol="star", size=12, color=_TV_GOLD,
                            line=dict(color="#fff", width=1)),
                name="Piercing Line"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=ema200, mode="lines", name="EMA200",
            line=dict(color="#7e57c2", width=1.5, dash="dot")), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=dt, y=ema200, mode="lines", name="EMA200",
            line=dict(color="#7e57c2", width=1.5, dash="dot")), row=1, col=1)

    # Entry/Exit markers
    e_d, e_p, x_d, x_p = _bt_detect_signals(df, strategy)
    if e_d:
        fig.add_trace(go.Scatter(x=e_d, y=e_p, mode="markers", name="▲ Entry",
            marker=dict(symbol="triangle-up", size=12, color=_TV_GREEN,
                        line=dict(color="#fff", width=1.5)),
            hovertemplate="<b>▲ ENTRY</b><br>%{x}<br>%{y:.2f}<extra></extra>"),
            row=1, col=1)
    if x_d:
        fig.add_trace(go.Scatter(x=x_d, y=x_p, mode="markers", name="▼ Exit",
            marker=dict(symbol="triangle-down", size=12, color=_TV_RED,
                        line=dict(color="#fff", width=1.5)),
            hovertemplate="<b>▼ EXIT</b><br>%{x}<br>%{y:.2f}<extra></extra>"),
            row=1, col=1)

    # ── Row 2: Volume ────────────────────────────────────────────────────
    vcol = [_TV_GREEN if cl >= op else _TV_RED for cl, op in zip(c, o)]
    fig.add_trace(go.Bar(x=dt, y=v, marker_color=vcol, marker_line_width=0,
        name="Vol", showlegend=False), row=2, col=1)

    # ── Row 3: RSI sempre ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=dt, y=rsi_s, mode="lines",
        line=dict(color=_TV_PURPLE, width=1.8), name="RSI", showlegend=False),
        row=3, col=1)
    fig.add_hrect(y0=70, y1=100, row=3, col=1, fillcolor="rgba(239,83,80,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  row=3, col=1, fillcolor="rgba(38,166,154,0.08)", line_width=0)
    for yv, clr in [(70, _TV_RED), (50, _TV_GRAY), (30, _TV_GREEN)]:
        fig.add_hline(y=yv, row=3, col=1, line=dict(color=clr, width=0.7, dash="dot"))

    # ── Row 4: indicatore dedicato alla strategia ────────────────────────
    if strategy == "ADX+EMA" or strategy == "ADX+Pattern":
        fig.add_trace(go.Scatter(x=dt, y=adx_, mode="lines",
            line=dict(color=_TV_RED, width=2), name="ADX", showlegend=False,
            fill="tozeroy", fillcolor="rgba(239,83,80,0.06)"), row=4, col=1)
        fig.add_hline(y=25, row=4, col=1, line=dict(color=_TV_GOLD, dash="dot", width=1.5))
        fig.add_hrect(y0=25, y1=80, row=4, col=1, fillcolor="rgba(255,152,0,0.07)", line_width=0)
        fig.update_layout(yaxis4=dict(range=[0, 80], tickvals=[0, 25, 50]))

    elif strategy in ("RSI+VWAP", "RSI+Bollinger"):
        fig.add_trace(go.Scatter(x=dt, y=rsi_s, mode="lines",
            line=dict(color=_TV_PURPLE, width=1.8), showlegend=False,
            fill="tozeroy", fillcolor="rgba(156,39,176,0.05)"), row=4, col=1)
        for yv, clr in [(70, _TV_RED), (30, _TV_GREEN)]:
            fig.add_hline(y=yv, row=4, col=1, line=dict(color=clr, width=1.2, dash="dot"))
        fig.add_hrect(y0=70, y1=100, row=4, col=1, fillcolor="rgba(239,83,80,0.12)", line_width=0)
        fig.add_hrect(y0=0,  y1=30,  row=4, col=1, fillcolor="rgba(38,166,154,0.12)", line_width=0)

    elif strategy == "OBV+Hull MA":
        fig.add_trace(go.Scatter(x=dt, y=obv_s, mode="lines",
            line=dict(color=_TV_BLUE, width=1.8), name="OBV", showlegend=False,
            fill="tozeroy", fillcolor="rgba(41,98,255,0.06)"), row=4, col=1)
        obv_ma = _bt_ema(obv_s, 10)
        fig.add_trace(go.Scatter(x=dt, y=obv_ma, mode="lines",
            line=dict(color=_TV_ORANGE, width=1.2, dash="dot"),
            name="OBV MA10", showlegend=False), row=4, col=1)
        fig.add_hline(y=0, row=4, col=1, line=dict(color=_TV_BORDER, width=1))

    elif strategy == "SAR+Chop":
        # Chop Zone: barre colorate
        chop_map = {"blue": "#2979ff", "green": "#26a69a",
                    "yellow": "#ffd700", "red": "#ef5350"}
        chop_vals = adx_.fillna(0).tolist() if hasattr(adx_, 'tolist') else list(adx_)
        bar_cols = [chop_map.get(ch, "#787b86") for ch in chop_colors]
        fig.add_trace(go.Bar(x=dt, y=[30]*len(dt),  # altezza fissa
            marker_color=bar_cols, marker_line_width=0,
            name="Chop Zone", showlegend=False), row=4, col=1)
        # Legenda colori inline
        for col_k, col_v, label in [("blue","#2979ff","Strong"),("green","#26a69a","Moderate"),
                                      ("yellow","#ffd700","Caution"),("red","#ef5350","Choppy")]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(color=col_v, size=10, symbol="square"),
                name=label, showlegend=True), row=4, col=1)

    else:  # MACD / Keltner / Donchian (MACD default)
        fig.add_trace(go.Bar(x=dt, y=mh, marker_color=hist_colors,
            marker_line_width=0, opacity=0.8, name="Hist", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=dt, y=ml, mode="lines",
            line=dict(color=_TV_BLUE, width=1.3), name="MACD", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=dt, y=sl, mode="lines",
            line=dict(color=_TV_ORANGE, width=1.3), name="Signal", showlegend=False), row=4, col=1)
        fig.add_hline(y=0, row=4, col=1, line=dict(color=_TV_BORDER, width=1))

    # ── Layout ───────────────────────────────────────────────────────────
    last_p = float(c.iloc[-1]); first_p = float(c.dropna().iloc[0])
    chg = (last_p/first_p-1)*100; chg_c = _TV_GREEN if chg >= 0 else _TV_RED
    n_e, n_x = len(e_d), len(x_d)
    fig.update_layout(
        title=dict(
            text=(f"<b style='color:{_TV_CYAN}'>{ticker}</b>"
                  f"  <span style='color:{_TV_GRAY}'>{strategy}</span>"
                  f"  <span style='color:{chg_c}'>{'▲' if chg>=0 else '▼'}{abs(chg):.1f}%</span>"
                  f"  <span style='color:{_TV_GRAY};font-size:0.8em'>"
                  f"▲ {n_e} entry · ▼ {n_x} exit</span>"),
            font=dict(size=13, color=_TV_TEXT), x=0.01),
        height=660,
        paper_bgcolor=_TV_BG, plot_bgcolor=_TV_PANEL,
        legend=dict(bgcolor=_TV_PANEL, bordercolor=_TV_BORDER,
                    font=dict(size=9, color=_TV_TEXT),
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=60, b=8),
        font=dict(color=_TV_TEXT, size=10),
        hovermode="x unified",
    )
    for row in [1, 2, 3, 4]:
        n_ = "" if row == 1 else str(row)
        fig.update_layout(**{
            f"xaxis{n_}": dict(showgrid=True, gridcolor=_TV_BORDER, zeroline=False,
                               showticklabels=(row == 4)),
            f"yaxis{n_}": dict(showgrid=True, gridcolor=_TV_BORDER, zeroline=False,
                               tickfont=dict(size=9)),
        })
    fig.update_layout(yaxis3=dict(range=[0, 100], tickvals=[30, 50, 70]))

    k = strategy.replace("+","_").replace(" ","_")
    st.plotly_chart(fig, use_container_width=True, key=f"bt_sc_{ticker}_{k}")


# -- Import db functions ----------------------------------------------------
try:
    from utils.db import (load_signals, signal_summary_stats,
                          update_signal_performance, cache_stats)
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# -- Colori per tipo segnale ------------------------------------------------
SIGNAL_COLORS = {
    "EARLY":      "#60a5fa",
    "PRO":        "#00ff88",
    "HOT":        "#f97316",
    "CONFLUENCE": "#a78bfa",
    "SERAFINI":   "#f59e0b",
    "FINVIZ":     "#38bdf8",
}

PLOTLY_DARK = dict(
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Courier New"),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
)

# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONI STATISTICHE PROFESSIONALI
# ══════════════════════════════════════════════════════════════════════════════

def _calc_sharpe(returns: pd.Series, rf_annual: float = 0.04) -> float:
    """Sharpe Ratio annualizzato (rf = risk-free rate annuale, default 4%)."""
    r = returns.dropna()
    if len(r) < 5: return float("nan")
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess   = r - rf_daily
    std      = excess.std()
    if std == 0: return float("nan")
    return round(float(excess.mean() / std * np.sqrt(252)), 2)

def _calc_sortino(returns: pd.Series, rf_annual: float = 0.04) -> float:
    """Sortino Ratio (penalizza solo il downside)."""
    r = returns.dropna()
    if len(r) < 5: return float("nan")
    rf_daily   = (1 + rf_annual) ** (1/252) - 1
    excess     = r - rf_daily
    downside   = excess[excess < 0]
    down_std   = downside.std()
    if down_std == 0 or len(downside) < 2: return float("nan")
    return round(float(excess.mean() / down_std * np.sqrt(252)), 2)

def _calc_max_drawdown(returns: pd.Series) -> float:
    """Max Drawdown % da picco a valle sulla curva cumulata."""
    r = returns.dropna()
    if len(r) < 3: return float("nan")
    equity  = (1 + r / 100).cumprod()
    peak    = equity.cummax()
    dd      = (equity - peak) / peak * 100
    return round(float(dd.min()), 2)

def _calc_profit_factor(returns: pd.Series) -> float:
    """Profit Factor = somma vincite / |somma perdite|."""
    r = returns.dropna()
    wins  = r[r > 0].sum()
    loss  = abs(r[r < 0].sum())
    if loss == 0: return float("inf") if wins > 0 else float("nan")
    return round(float(wins / loss), 2)

def _calc_win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0: return float("nan")
    return round(float((r > 0).sum() / len(r) * 100), 1)

def _calc_avg_win(returns: pd.Series) -> float:
    r = returns.dropna()
    wins = r[r > 0]
    return round(float(wins.mean()), 2) if len(wins) > 0 else float("nan")

def _calc_avg_loss(returns: pd.Series) -> float:
    r = returns.dropna()
    losses = r[r < 0]
    return round(float(losses.mean()), 2) if len(losses) > 0 else float("nan")

def _calc_max_consec_losses(returns: pd.Series) -> int:
    r = returns.dropna()
    if r.empty: return 0
    max_loss = cur = 0
    for v in r:
        if v < 0: cur += 1; max_loss = max(max_loss, cur)
        else:     cur = 0
    return max_loss

def _build_stats_dict(returns: pd.Series, horizon_label: str) -> dict:
    """Calcola tutte le metriche statistiche per un set di rendimenti."""
    r = returns.dropna()
    return {
        "N segnali":       len(r),
        "Win Rate %":      _calc_win_rate(r),
        "Avg Ret %":       round(float(r.mean()), 2) if len(r) else float("nan"),
        "Avg Win %":       _calc_avg_win(r),
        "Avg Loss %":      _calc_avg_loss(r),
        "Profit Factor":   _calc_profit_factor(r),
        "Max Drawdown %":  _calc_max_drawdown(r),
        "Sharpe":          _calc_sharpe(r),
        "Sortino":         _calc_sortino(r),
        "Max Consec Loss": _calc_max_consec_losses(r),
        "Best Trade %":    round(float(r.max()), 2) if len(r) else float("nan"),
        "Worst Trade %":   round(float(r.min()), 2) if len(r) else float("nan"),
    }

def _render_stats_panel(df_sigs: pd.DataFrame, horizon: str) -> None:
    """
    Pannello statistiche professionali per orizzonte selezionato.
    Mostra metriche per tutti i setup + globale.
    """
    h_label = {"ret_1d": "+1g", "ret_5d": "+5g",
                "ret_10d": "+10g", "ret_20d": "+20g"}.get(horizon, horizon)
    st.markdown(f"### 📐 Statistiche Professionali — Orizzonte {h_label}")

    if horizon not in df_sigs.columns:
        st.warning(f"Colonna {horizon} non disponibile. Aggiorna le performance.")
        return

    # ── Statistiche globali ───────────────────────────────────────────────
    stats_all = _build_stats_dict(df_sigs[horizon], h_label)

    # ── Colori metriche ───────────────────────────────────────────────────
    def _color_sharpe(v):
        if pd.isna(v): return "#6b7280"
        if v >= 1.5:   return "#00ff88"
        if v >= 0.5:   return "#f59e0b"
        return "#ef4444"

    def _color_dd(v):
        if pd.isna(v): return "#6b7280"
        if v >= -5:    return "#00ff88"
        if v >= -15:   return "#f59e0b"
        return "#ef4444"

    def _color_pf(v):
        if pd.isna(v) or v == float("inf"): return "#6b7280"
        if v >= 1.5:   return "#00ff88"
        if v >= 1.0:   return "#f59e0b"
        return "#ef4444"

    # Riga KPI metriche principali
    kc = st.columns(6)
    sharpe_c = _color_sharpe(stats_all["Sharpe"])
    dd_c     = _color_dd(stats_all["Max Drawdown %"])
    pf_c     = _color_pf(stats_all["Profit Factor"])
    wr_c     = "#00ff88" if (stats_all["Win Rate %"] or 0) >= 55 else \
               "#f59e0b" if (stats_all["Win Rate %"] or 0) >= 45 else "#ef4444"

    def _kpi(col, label, value, color="#d1d4dc", suffix=""):
        v_str = f"{value:.2f}{suffix}" if isinstance(value, float) and not pd.isna(value) else str(value) if value is not None else "—"
        col.markdown(
            f'<div style="background:#1e222d;border:1px solid #2a2e39;border-radius:6px;'
            f'padding:10px 14px;text-align:center">'
            f'<div style="color:#787b86;font-size:0.72rem;margin-bottom:4px">{label}</div>'
            f'<div style="color:{color};font-size:1.35rem;font-weight:bold;font-family:Courier New">{v_str}</div>'
            f'</div>', unsafe_allow_html=True)

    _kpi(kc[0], "Sharpe Ratio",    stats_all["Sharpe"],         sharpe_c)
    _kpi(kc[1], "Sortino Ratio",   stats_all["Sortino"],        sharpe_c)
    _kpi(kc[2], "Max Drawdown",    stats_all["Max Drawdown %"], dd_c, "%")
    _kpi(kc[3], "Profit Factor",   stats_all["Profit Factor"],  pf_c)
    _kpi(kc[4], "Win Rate",        stats_all["Win Rate %"],     wr_c, "%")
    _kpi(kc[5], "N Segnali",       stats_all["N segnali"],      "#d1d4dc")

    st.markdown("")

    # Seconda riga: dettaglio trade
    rc = st.columns(4)
    _kpi(rc[0], "Avg Win",         stats_all["Avg Win %"],      "#00ff88", "%")
    _kpi(rc[1], "Avg Loss",        stats_all["Avg Loss %"],     "#ef4444", "%")
    _kpi(rc[2], "Best Trade",      stats_all["Best Trade %"],   "#00ff88", "%")
    _kpi(rc[3], "Max Consec Loss", stats_all["Max Consec Loss"], "#ef4444")

    st.markdown("")

    # ── Tabella breakdown per setup ───────────────────────────────────────
    if "signal_type" in df_sigs.columns:
        st.markdown("**Breakdown per tipo segnale:**")
        rows = []
        signal_types = df_sigs["signal_type"].dropna().unique()
        for stype in sorted(signal_types):
            sub = df_sigs[df_sigs["signal_type"] == stype][horizon]
            s   = _build_stats_dict(sub, h_label)
            rows.append({
                "Setup":           stype,
                "N":               s["N segnali"],
                "Win%":            s["Win Rate %"],
                "Avg%":            s["Avg Ret %"],
                "PF":              s["Profit Factor"],
                "Sharpe":          s["Sharpe"],
                "Max DD%":         s["Max Drawdown %"],
                "Best%":           s["Best Trade %"],
                "Worst%":          s["Worst Trade %"],
                "MaxConsecLoss":   s["Max Consec Loss"],
            })
        df_stats = pd.DataFrame(rows)

        def _color_val(v):
            if pd.isna(v): return "color:#6b7280"
            return f"color:{'#00ff88' if v > 0 else '#ef4444'};font-weight:bold"

        def _color_wr(v):
            if pd.isna(v): return "color:#6b7280"
            if v >= 60:  return "color:#00ff88;font-weight:bold"
            if v >= 50:  return "color:#f59e0b"
            return "color:#ef4444"

        def _color_sharpe_s(v):
            if pd.isna(v): return "color:#6b7280"
            if v >= 1.5:   return "color:#00ff88;font-weight:bold"
            if v >= 0.5:   return "color:#f59e0b"
            return "color:#ef4444"

        styled = (df_stats.style
            .applymap(_color_val,     subset=["Avg%","Best%","Worst%","Max DD%"])
            .applymap(_color_wr,      subset=["Win%"])
            .applymap(_color_sharpe_s,subset=["Sharpe"])
            .format({"Win%": "{:.1f}%", "Avg%": "{:.2f}%",
                     "PF":   "{:.2f}",  "Sharpe": "{:.2f}",
                     "Max DD%": "{:.2f}%",
                     "Best%": "{:.2f}%","Worst%": "{:.2f}%"},
                    na_rep="—"))
        st.dataframe(styled, use_container_width=True, height=260)

    # ── Drawdown chart ────────────────────────────────────────────────────
    with st.expander("📉 Curva Drawdown", expanded=False):
        df_v = df_sigs.dropna(subset=[horizon, "scanned_at"]).copy()
        df_v["scanned_at"] = pd.to_datetime(df_v["scanned_at"])
        df_v = df_v.sort_values("scanned_at")
        if not df_v.empty:
            daily = df_v.groupby(df_v["scanned_at"].dt.date)[horizon].mean()
            equity = (1 + daily / 100).cumprod()
            peak   = equity.cummax()
            dd_curve = (equity - peak) / peak * 100

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_curve.index.astype(str), y=dd_curve.values,
                fill="tozeroy",
                fillcolor="rgba(239,83,80,0.18)",
                line=dict(color=_TV_RED, width=1.5),
                name="Drawdown",
                hovertemplate="%{x}<br>DD: %{y:.2f}%<extra></extra>"
            ))
            fig_dd.add_hline(y=0, line=dict(color="#374151", width=1))
            fig_dd.update_layout(
                **PLOTLY_DARK,
                height=260,
                yaxis=dict(title="Drawdown %", ticksuffix="%"),
                margin=dict(l=0, r=0, t=20, b=0),
                title=dict(text="Max Drawdown nel tempo", font=dict(color=_TV_RED, size=12)),
            )
            st.plotly_chart(fig_dd, use_container_width=True, key="bt_dd_chart")




# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — strategy_chart_widget
# Chiamabile da qualsiasi tab: Dashboard principale, Backtest, Watchlist, ecc.
#
# Parametri:
#   tickers     : lista di str ticker da mostrare nella selectbox
#                 (es. df_ep["Ticker"].tolist())
#                 Se [] → mostra "— libero —" e permette input manuale
#   key_suffix  : stringa univoca per evitare conflitti chiavi Streamlit
#                 (es. "EARLY", "PRO", "bt", "wl")
#   default_ticker : ticker pre-selezionato (opzionale)
# ══════════════════════════════════════════════════════════════════════════════

_SC_RULES = {
    "RSI+VWAP":        ("📊","#e91e63",
                        "RSI incrocia sopra 30 + Prezzo > VWAP",
                        "RSI incrocia sotto 70 o Prezzo < VWAP"),
    "ADX+EMA":         ("📈","#ff9800",
                        "EMA20 incrocia sopra EMA50 + ADX > 25",
                        "EMA20 < EMA50 o ADX < 25"),
    "MACD":            ("⚡","#2962ff",
                        "MACD histogram incrocia sopra 0",
                        "MACD histogram incrocia sotto 0"),
    "Keltner Channel": ("📐","#00bcd4",
                        "Prezzo chiude sopra la banda inferiore Keltner (da sotto)",
                        "Prezzo raggiunge la banda superiore Keltner"),
    "Donchian Channel":("🔲","#26a69a",
                        "Prezzo tocca la banda superiore Donchian (breakout rialzista)",
                        "Prezzo tocca la banda inferiore Donchian"),
    "RSI+Bollinger":   ("📉","#9c27b0",
                        "Prezzo sotto la banda inferiore BB + RSI < 35 (mean reversion long)",
                        "Prezzo sopra la banda superiore BB o RSI > 65"),
    "OBV+Hull MA":     ("🌊","#ff9800",
                        "Prezzo incrocia sopra Hull MA + OBV in salita",
                        "Prezzo scende sotto Hull MA o OBV diverge"),
    "SAR+Chop":        ("🎯","#4caf50",
                        "SAR passa sotto le candele (bull) + Chop Zone non rossa",
                        "SAR passa sopra le candele o Chop Zone diventa rossa"),
    "ADX+Pattern":     ("⭐","#ffd700",
                        "Piercing Line pattern + ADX > 25 (trend in accelerazione)",
                        "ADX scende sotto 20 (trend si indebolisce)"),
}

def strategy_chart_widget(
    tickers: list,
    key_suffix: str = "sc",
    default_ticker: str = "",
    ticker_labels: dict = None,
) -> None:
    """
    ticker_labels: dict opzionale {ticker: "Nome azienda  (TICKER)"}
                   Se fornito, la selectbox mostra il nome invece del solo ticker.
    """
    """
    Widget Strategy Chart riusabile.
    Incollalo in qualsiasi tab con:
        from utils.backtest_tab import strategy_chart_widget
        strategy_chart_widget(df_ep["Ticker"].tolist(), key_suffix="EARLY")
    """
    ks = key_suffix.replace(" ", "_").replace("-", "_")

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{_TV_PANEL};border-left:3px solid {_TV_CYAN};'
        f'padding:8px 14px;border-radius:0 4px 4px 0;margin:12px 0 10px">'
        f'<span style="color:{_TV_CYAN};font-weight:700">📊 STRATEGY CHART</span>'
        f'<span style="color:{_TV_GRAY};font-size:0.78rem;margin-left:10px">'
        f'Candele + indicatori dedicati + segnali Entry ▲ / Exit ▼</span>'
        f'</div>', unsafe_allow_html=True)

    # ── Controlli ────────────────────────────────────────────────────────
    # Colonne: ticker (largo) | strategia | periodo | bottone
    c_tkr, c_str, c_per, c_btn = st.columns([3, 2, 1, 1])

    with c_tkr:
        # Se abbiamo una lista di ticker dal tab → selectbox
        # Altrimenti → input libero (solo per tab Backtest senza dati)
        if tickers:
            # Rimuovi duplicati
            seen = set(); ordered = []
            for t in tickers:
                if t and t not in seen:
                    seen.add(t); ordered.append(t)
            if ticker_labels:
                # Ordina alfabeticamente per nome display (A→Z)
                pairs = sorted(
                    [(t, ticker_labels.get(t, t)) for t in ordered],
                    key=lambda x: x[1].lower()
                )
                opts_raw     = [p[0] for p in pairs]
                opts_display = [p[1] for p in pairs]
                def_idx = opts_raw.index(default_ticker) if default_ticker in opts_raw else 0
                sel_label = st.selectbox(
                    f"Azienda / Ticker  ({len(opts_raw)} — A→Z)",
                    opts_display,
                    index=def_idx,
                    key=f"sc_tkr_{ks}",
                    help=f"{len(opts_raw)} titoli ordinati per nome A→Z"
                )
                sc_ticker = opts_raw[opts_display.index(sel_label)]
            else:
                # Ordina per ticker alfabetico se no labels
                opts = sorted(set(ordered))
                def_idx = opts.index(default_ticker) if default_ticker in opts else 0
                sc_ticker = st.selectbox(
                    f"Ticker  ({len(opts)})",
                    opts,
                    index=def_idx,
                    key=f"sc_tkr_{ks}",
                    help=f"{len(opts)} ticker disponibili in questo tab"
                )
        else:
            sc_ticker = st.text_input(
                "Ticker",
                value=default_ticker or "AAPL",
                key=f"sc_tkr_{ks}",
                placeholder="es. AAPL, ENI.MI",
            ).strip().upper()

    with c_str:
        sc_strategy = st.selectbox(
            "Strategia",
            list(_SC_RULES.keys()),
            key=f"sc_str_{ks}",
        )

    with c_per:
        sc_range = st.selectbox(
            "Periodo",
            ["3mo", "6mo", "1y", "2y"],
            index=2,
            key=f"sc_per_{ks}",
        )

    with c_btn:
        st.write(""); st.write("")
        sc_run = st.button("▶ Mostra", key=f"sc_run_{ks}",
                           use_container_width=True, type="primary")

    # ── Banner Entry / Exit ───────────────────────────────────────────────
    icon, sc_color, e_txt, x_txt = _SC_RULES[sc_strategy]
    b_l, b_r = st.columns(2)
    with b_l:
        st.markdown(
            f'<div style="background:{_TV_PANEL};border:1px solid {_TV_BORDER};'
            f'border-left:4px solid {sc_color};border-radius:6px;'
            f'padding:6px 12px;margin-bottom:8px">'
            f'<span style="color:{_TV_GRAY};font-size:0.65rem">▲ ENTRY {icon}</span><br>'
            f'<span style="color:{_TV_TEXT};font-size:0.8rem;font-weight:600">{e_txt}</span>'
            f'</div>', unsafe_allow_html=True)
    with b_r:
        st.markdown(
            f'<div style="background:{_TV_PANEL};border:1px solid {_TV_BORDER};'
            f'border-left:4px solid {_TV_RED};border-radius:6px;'
            f'padding:6px 12px;margin-bottom:8px">'
            f'<span style="color:{_TV_GRAY};font-size:0.65rem">▼ EXIT</span><br>'
            f'<span style="color:{_TV_TEXT};font-size:0.8rem;font-weight:600">{x_txt}</span>'
            f'</div>', unsafe_allow_html=True)

    # ── Legenda visuale strategia ─────────────────────────────────────────
    _render_strategy_legend(sc_strategy)

    # ── Render grafico ────────────────────────────────────────────────────
    if sc_run and sc_ticker:
        _bt_render_strategy_chart(sc_ticker, sc_strategy, sc_range)
    else:
        st.caption("👆 Seleziona ticker e strategia, poi clicca **▶ Mostra**")


def render_backtest_tab():
    """
    Funzione principale da chiamare dentro il tab backtest.
    Esempio:
        with tab_backtest:
            render_backtest_tab()
    """
    st.markdown('<div class="section-pill">📈 BACKTEST SEGNALI</div>',
                unsafe_allow_html=True)

    if not DB_AVAILABLE:
        st.error("utils.db non disponibile — assicurati che db.py v28 sia installato.")
        return

    # -- Controlli ---------------------------------------------------------
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])

    with col_ctrl1:
        days_back = st.selectbox(
            "📅 Periodo analisi", [7, 14, 30, 60, 90, 180, 365],
            index=2, key="bt_days"
        )

    with col_ctrl2:
        signal_filter = st.selectbox(
            "🔍 Tipo segnale", ["Tutti", "EARLY", "PRO", "HOT",
                                "CONFLUENCE", "SERAFINI", "FINVIZ"],
            key="bt_sig_type"
        )
        sig_type_arg = None if signal_filter == "Tutti" else signal_filter

    with col_ctrl3:
        st.write("")
        if st.button("🔄 Aggiorna performance", key="bt_update",
                     use_container_width=True):
            with st.spinner("Aggiorno prezzi forward... (scarico yfinance)"):
                n = update_signal_performance(max_signals=300)
            st.success(f"✅ Aggiornati {n} segnali.")
            st.rerun()

    # Pulsante Reset Elenco segnali
    _bc1, _bc2 = st.columns([1, 5])
    with _bc1:
        if st.button("🗑️ Reset Elenco Segnali", key="bt_reset_sigs",
                     type="secondary",
                     help="Cancella tutti i segnali registrati dal DB. "
                          "I dati scanner rimangono intatti."):
            try:
                from utils.db import _get_db_path
                import sqlite3 as _sq
                _c = _sq.connect(str(_get_db_path()))
                _c.execute("DELETE FROM signals")
                _c.commit(); _c.close()
                st.success("✅ Elenco segnali cancellato!")
                st.rerun()
            except Exception as _re:
                st.error(f"Errore reset: {_re}")

    # -- Carica dati -------------------------------------------------------
    df_sigs = load_signals(signal_type=sig_type_arg,
                           days_back=days_back, with_perf=True)
    df_summ = signal_summary_stats(days_back=days_back)

    # ── v34 FIX: se signals è vuoto, genera dati demo da scan_history ────
    # Il problema comune: save_signals() non è mai stata chiamata (db vecchio)
    # oppure update_signal_performance() non è stato eseguito → ret_Xd tutti NaN.
    # Soluzione: popoliamo con rendimenti sintetici realistici da Yahoo Finance
    # usando i ticker degli ultimi snapshot salvati nello storico scansioni.
    if df_sigs.empty or df_sigs[["ret_1d","ret_5d","ret_10d","ret_20d"]].notna().sum().sum() == 0:
        st.info(
            "📭 **Nessun dato di performance trovato nel DB.**\n\n"
            "Questo accade quando:\n"
            "- Lo scanner non ha mai chiamato `save_signals()` (db.py vecchio)\n"
            "- `🔄 Aggiorna performance` non è mai stato eseguito\n\n"
            "**Clicca il bottone qui sotto** per generare dati demo realistici "
            "dai tuoi ultimi ticker scansionati, oppure clicca "
            "**🔄 Aggiorna performance** se hai già segnali salvati."
        )

        if st.button("🧪 Genera dati demo (ultimi ticker scansionati)",
                     key="bt_gen_demo", type="primary"):
            with st.spinner("Generazione dati demo da Yahoo Finance…"):
                try:
                    # Recupera ticker dagli snapshot recenti
                    from utils.db import load_scan_history, load_scan_snapshot
                    import urllib.request as _ur, json as _js, time as _tm

                    hist = load_scan_history(5)
                    all_tickers = []
                    for _, row in hist.iterrows():
                        try:
                            ep, _ = load_scan_snapshot(int(row["id"]))
                            if not ep.empty and "Ticker" in ep.columns:
                                all_tickers += ep["Ticker"].dropna().tolist()
                        except Exception:
                            pass

                    # Fallback: ticker di esempio se storico vuoto
                    if not all_tickers:
                        all_tickers = ["AAPL","MSFT","NVDA","GOOGL","AMZN",
                                       "META","TSLA","JPM","V","XOM"]

                    # Dedup e limita a 20 ticker
                    all_tickers = list(dict.fromkeys(all_tickers))[:20]

                    demo_rows = []
                    import sqlite3 as _sq
                    from utils.db import _get_db_path
                    conn = _sq.connect(str(_get_db_path()))

                    # Assicura che la tabella signals esista
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS signals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            scan_id INTEGER, ticker TEXT, nome TEXT,
                            signal_type TEXT, prezzo REAL, rsi REAL,
                            quality_score REAL, pro_score REAL,
                            ser_score REAL, fv_score REAL,
                            squeeze INTEGER, weekly_bull INTEGER,
                            scanned_at TEXT,
                            price_1d REAL, price_5d REAL,
                            price_10d REAL, price_20d REAL,
                            ret_1d REAL, ret_5d REAL,
                            ret_10d REAL, ret_20d REAL
                        )""")

                    # Pulisce demo vecchi
                    conn.execute("DELETE FROM signals WHERE scan_id = -999")

                    sig_types = ["EARLY","PRO","HOT","CONFLUENCE","SERAFINI"]
                    import random, math
                    random.seed(42)

                    for i, tkr in enumerate(all_tickers):
                        try:
                            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{tkr}"
                                   f"?interval=1d&range=30d")
                            req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                            with _ur.urlopen(req, timeout=8) as r:
                                data = _js.loads(r.read())
                            closes = [c for c in
                                data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                                if c is not None]
                            if len(closes) < 22:
                                continue
                            # Simula entry 20 giorni fa e calcola rendimenti reali
                            entry  = closes[-21]
                            p1d    = closes[-20]
                            p5d    = closes[-16] if len(closes) >= 17 else closes[-1]
                            p10d   = closes[-11] if len(closes) >= 12 else closes[-1]
                            p20d   = closes[-1]
                            r1d    = round((p1d  / entry - 1) * 100, 2)
                            r5d    = round((p5d  / entry - 1) * 100, 2)
                            r10d   = round((p10d / entry - 1) * 100, 2)
                            r20d   = round((p20d / entry - 1) * 100, 2)
                            stype  = sig_types[i % len(sig_types)]
                            # Data scansione = ~20 giorni fa
                            from datetime import datetime, timedelta
                            scan_dt = (datetime.now() - timedelta(days=20+i%5)).strftime("%Y-%m-%d %H:%M")
                            conn.execute("""
                                INSERT INTO signals
                                (scan_id,ticker,nome,signal_type,prezzo,rsi,
                                 quality_score,pro_score,ser_score,fv_score,
                                 squeeze,weekly_bull,scanned_at,
                                 price_1d,price_5d,price_10d,price_20d,
                                 ret_1d,ret_5d,ret_10d,ret_20d)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """, (-999, tkr, tkr, stype, round(entry,2),
                                  round(random.uniform(40,70),1),
                                  random.randint(4,10), random.randint(5,9),
                                  random.randint(3,6),  random.randint(3,5),
                                  0, 1, scan_dt,
                                  round(p1d,2), round(p5d,2),
                                  round(p10d,2), round(p20d,2),
                                  r1d, r5d, r10d, r20d))
                            _tm.sleep(0.1)  # throttle Yahoo
                        except Exception:
                            pass

                    conn.commit(); conn.close()
                    st.success(f"✅ Dati demo generati per {len(all_tickers)} ticker. Ricarica la pagina.")
                    st.rerun()
                except Exception as _de:
                    st.error(f"Errore generazione demo: {_de}")

        # ── Strategy Chart sempre visibile anche senza dati performance ──
        st.markdown("---")
        st.markdown("### 📊 Strategy Chart — disponibile sempre")
        strategy_chart_widget(tickers=[], key_suffix="bt_empty")
        return  # ferma solo le statistiche (non ha senso mostrarle senza dati)



    # ══════════════════════════════════════════════════════════════════════
    # 📊 SEZIONE STRATEGY CHART — widget condiviso (nessun ticker pre-caricato)
    # ══════════════════════════════════════════════════════════════════════
    strategy_chart_widget(tickers=[], key_suffix="bt")

    st.markdown("---")

    st.caption(
        f"📊 {len(df_sigs)} segnali negli ultimi {days_back} giorni  "
        f"| {df_sigs['ticker'].nunique()} ticker unici  "
        f"| {df_sigs['ret_20d'].notna().sum()} con performance completa"
    )

    # ---------------------------------------------------------------------
    # 📊 RIEPILOGO  per tipo segnale
    # ---------------------------------------------------------------------
    st.markdown("### 📊 Riepilogo per tipo segnale")

    if not df_summ.empty:
        # Tabella riepilogo con colori
        cols_show = ["Signal", "N", "Avg +1d%", "Win%_ret_1d",
                     "Avg +5d%", "Win%_ret_5d",
                     "Avg +10d%", "Win%_ret_10d",
                     "Avg +20d%", "Win%_ret_20d"]
        cols_show = [c for c in cols_show if c in df_summ.columns]
        df_show   = df_summ[cols_show].copy()

        # Rename per display
        df_show = df_show.rename(columns={
            "Win%_ret_1d":  "Win%+1d", "Win%_ret_5d":  "Win%+5d",
            "Win%_ret_10d": "Win%+10d","Win%_ret_20d": "Win%+20d",
        })

        # Stile Streamlit
        def _color_ret(v):
            if pd.isna(v): return "color: #374151"
            return f"color: {'#00ff88' if v > 0 else '#ef4444'}; font-weight: bold"

        def _color_wr(v):
            if pd.isna(v): return "color: #374151"
            if v >= 60:   return "color: #00ff88; font-weight: bold"
            if v >= 50:   return "color: #f59e0b"
            return "color: #ef4444"

        ret_cols = [c for c in df_show.columns if "Avg +" in c]
        wr_cols  = [c for c in df_show.columns if "Win%" in c]

        styled = (df_show.style
                  .applymap(_color_ret, subset=ret_cols)
                  .applymap(_color_wr,  subset=wr_cols)
                  .format({c: "{:.1f}%" for c in ret_cols + wr_cols},
                          na_rep="—"))
        st.dataframe(styled, use_container_width=True, height=250)

    # ---------------------------------------------------------------------
    # 📈 EQUITY CURVE  — curva cumulata per tipo segnale
    # ---------------------------------------------------------------------
    st.markdown("### 📈 Equity curve cumulata")

    horizon = st.radio(
        "Orizzonte temporale", ["ret_1d", "ret_5d", "ret_10d", "ret_20d"],
        format_func=lambda x: {"ret_1d": "+1 giorno", "ret_5d": "+5 giorni",
                               "ret_10d": "+10 giorni", "ret_20d": "+20 giorni"}[x],
        horizontal=True, key="bt_horizon"
    )

    df_valid = df_sigs.dropna(subset=[horizon, "scanned_at"]).copy()

    if df_valid.empty:
        st.info("Nessun segnale con performance disponibile per questo orizzonte. "
                "Clicca '🔄 Aggiorna performance'.")
    else:
        df_valid["scanned_at"] = pd.to_datetime(df_valid["scanned_at"])
        df_valid = df_valid.sort_values("scanned_at")

        fig_eq = go.Figure()

        # Una curva per ogni tipo segnale selezionato
        types_to_plot = (df_valid["signal_type"].unique().tolist()
                         if signal_filter == "Tutti"
                         else [signal_filter])

        for stype in types_to_plot:
            sub = df_valid[df_valid["signal_type"] == stype].copy()
            if sub.empty: continue

            # Equity: parte da 0, accumula rendimenti medi giornalieri
            daily = (sub.groupby(sub["scanned_at"].dt.date)[horizon]
                       .mean().reset_index())
            daily.columns = ["date", "avg_ret"]
            daily["cumulative"] = (1 + daily["avg_ret"] / 100).cumprod() * 100 - 100

            color = SIGNAL_COLORS.get(stype, "#c9d1d9")
            fig_eq.add_trace(go.Scatter(
                x=daily["date"].astype(str),
                y=daily["cumulative"].round(2),
                mode="lines+markers",
                name=stype,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{stype}</b><br>%{{x}}<br>Cum: %{{y:.1f}}%<extra></extra>"
            ))

        fig_eq.add_hline(y=0, line=dict(color="#374151", width=1, dash="dot"))
        fig_eq.update_layout(
            **PLOTLY_DARK,
            title=dict(text=f"📈 Rendimento cumulato {horizon}",
                       font=dict(color="#00ff88", size=14)),
            height=380,
            yaxis=dict(title="Rendimento cumulato %", ticksuffix="%"),
            xaxis=dict(title="Data segnale"),
            legend=dict(orientation="h", y=1.05, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=50, b=0)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ---------------------------------------------------------------------
    # 📐 STATISTICHE PROFESSIONALI — Sharpe, Sortino, Max DD, Profit Factor
    # ---------------------------------------------------------------------
    _render_stats_panel(df_sigs, horizon)

    # ---------------------------------------------------------------------
    # 🥇 TOP PERFORMER  — migliori e peggiori ticker
    # ---------------------------------------------------------------------
    st.markdown("### 🥇 Top & Bottom performer")

    df_perf_valid = df_sigs.dropna(subset=["ret_20d"]).copy()
    if not df_perf_valid.empty:
        top_col, bot_col = st.columns(2)

        # Aggregazione per ticker
        tkr_stats = (df_perf_valid.groupby("ticker")
                     .agg(avg_ret=("ret_20d", "mean"),
                          n_signals=("id", "count"),
                          best=("ret_20d", "max"),
                          worst=("ret_20d", "min"))
                     .reset_index()
                     .sort_values("avg_ret", ascending=False))

        with top_col:
            st.markdown("**🟢 Top 10 — +20d**")
            top10 = tkr_stats.head(10)[["ticker", "avg_ret", "n_signals"]].copy()
            top10.columns = ["Ticker", "Avg Ret+20d %", "N segnali"]

            def _bg_green(v):
                return f"color: {'#00ff88' if v > 0 else '#ef4444'}; font-weight: bold"

            st.dataframe(
                top10.style.applymap(_bg_green, subset=["Avg Ret+20d %"])
                           .format({"Avg Ret+20d %": "{:.1f}%"}),
                use_container_width=True, height=350
            )

        with bot_col:
            st.markdown("**🔴 Bottom 10 — +20d**")
            bot10 = tkr_stats.tail(10)[::-1][["ticker", "avg_ret", "n_signals"]].copy()
            bot10.columns = ["Ticker", "Avg Ret+20d %", "N segnali"]
            st.dataframe(
                bot10.style.applymap(_bg_green, subset=["Avg Ret+20d %"])
                           .format({"Avg Ret+20d %": "{:.1f}%"}),
                use_container_width=True, height=350
            )

    # ---------------------------------------------------------------------
    # 🔍 GRIGLIA DETTAGLIO
    # ---------------------------------------------------------------------
    st.markdown("### 🔍 Dettaglio segnali registrati")

    disp_cols = ["scanned_at", "ticker", "nome", "signal_type",
                 "prezzo", "rsi", "quality_score", "ser_score", "fv_score",
                 "squeeze", "weekly_bull",
                 "ret_1d", "ret_5d", "ret_10d", "ret_20d"]
    disp_cols = [c for c in disp_cols if c in df_sigs.columns]
    df_disp   = df_sigs[disp_cols].copy()

    # Rename
    df_disp = df_disp.rename(columns={
        "scanned_at": "Data", "ticker": "Ticker", "nome": "Nome",
        "signal_type": "Tipo", "prezzo": "Prezzo",
        "rsi": "RSI", "quality_score": "Quality",
        "ser_score": "Ser", "fv_score": "FV",
        "squeeze": "SQ", "weekly_bull": "W+",
        "ret_1d": "+1d%", "ret_5d": "+5d%",
        "ret_10d": "+10d%", "ret_20d": "+20d%"
    })

    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
        gb = GridOptionsBuilder.from_dataframe(df_disp)
        gb.configure_default_column(sortable=True, resizable=True, filterable=True)
        gb.configure_column("Data",   width=130)
        gb.configure_column("Ticker", width=75, pinned="left")
        gb.configure_column("Nome",   width=160)
        gb.configure_column("Tipo",   width=100)
        gb.configure_column("Prezzo", width=80)
        for rc in ["+1d%", "+5d%", "+10d%", "+20d%"]:
            if rc in df_disp.columns:
                gb.configure_column(rc, width=80,
                    cellStyle={"function":
                        "params.value > 0 ? {'color':'#00ff88','fontWeight':'bold'} : "
                        "params.value < 0 ? {'color':'#ef4444','fontWeight':'bold'} : {}"})
        go_bt = gb.build()
        AgGrid(df_disp, gridOptions=go_bt, height=440,
               update_mode=GridUpdateMode.NO_UPDATE,
               allow_unsafe_jscode=True, theme="streamlit",
               key="bt_detail_grid")
    except Exception:
        # Fallback senza AgGrid
        st.dataframe(df_disp, use_container_width=True, height=440)

    # ---------------------------------------------------------------------
    # 💾 EXPORT + STATS CACHE
    # ---------------------------------------------------------------------
    exp_col, cache_col = st.columns([2, 2])
    with exp_col:
        csv = df_sigs.to_csv(index=False).encode()
        st.download_button("📥 Esporta segnali CSV", csv,
                           f"segnali_backtest_{days_back}g.csv",
                           "text/csv", key="bt_exp_csv")
    with cache_col:
        if st.button("📊 Stats cache", key="bt_cache_stats"):
            cs = cache_stats()
            st.info(
                f"Cache SQLite: **{cs['total_entries']}** entry  "
                f"| 🟢 {cs['fresh']} fresche  "
                f"| ⏰ {cs['stale']} scadute  "
                f"| 💾 {cs['size_mb']} MB"
            )


