import io
import time
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

from utils.formatting import add_formatted_cols, add_links, prepare_display_df
from utils.db import (
    init_db, reset_watchlist_db, add_to_watchlist, load_watchlist,
    DB_PATH, save_scan_history, load_scan_history, load_scan_snapshot,
    delete_from_watchlist, move_watchlist_rows, rename_watchlist,
)
from utils.scanner import load_universe, scan_ticker

# =========================================================================
# EXPORT HELPERS
# =========================================================================

def to_excel_bytes_sheets_dict(sheets_dict: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buffer.getvalue()


def make_tv_csv(df: pd.DataFrame, tab_name: str, ticker_col: str) -> bytes:
    tmp = df[[ticker_col]].copy()
    tmp.insert(0, "Tab", tab_name)
    return tmp.to_csv(index=False).encode("utf-8")


def get_csv_download_button(df: pd.DataFrame, filename: str, key: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# =========================================================================
# JS RENDERERS PER AGGRID
# =========================================================================

name_dblclick_renderer = JsCode("""
class NameDoubleClickRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        this.eGui.innerText = params.value || '';
        const ticker = params.data.Ticker || params.data.ticker;
        if (!ticker) return;
        this.eGui.style.cursor = 'pointer';
        this.eGui.title = 'Doppio click per aprire TradingView';
        this.eGui.ondblclick = function() {
            const symbol = String(ticker).split(".")[0];
            const url = "https://www.tradingview.com/chart/?symbol=" + symbol;
            window.open(url, "_blank");
        }
    }
    getGui() { return this.eGui; }
}
""")

squeeze_renderer = JsCode("""
class SqueeezeRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const val = params.value;
        if (val === true || val === 'True' || val === 'true') {
            this.eGui.innerText = '🔥 SQ';
            this.eGui.style.color = '#f97316';
            this.eGui.style.fontWeight = 'bold';
            this.eGui.title = 'Squeeze attivo: possibile esplosione imminente';
        } else {
            this.eGui.innerText = '-';
            this.eGui.style.color = '#6b7280';
        }
    }
    getGui() { return this.eGui; }
}
""")

rsi_div_renderer = JsCode("""
class RsiDivRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const val = params.value;
        if (val === 'BEARISH') {
            this.eGui.innerText = '⚠️ BEAR';
            this.eGui.style.color = '#ef4444';
            this.eGui.title = 'Divergenza RSI Bearish: prezzo sale, RSI scende';
        } else if (val === 'BULLISH') {
            this.eGui.innerText = '✅ BULL';
            this.eGui.style.color = '#22c55e';
            this.eGui.title = 'Divergenza RSI Bullish: prezzo scende, RSI sale';
        } else {
            this.eGui.innerText = '-';
            this.eGui.style.color = '#6b7280';
        }
    }
    getGui() { return this.eGui; }
}
""")

weekly_renderer = JsCode("""
class WeeklyRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const val = params.value;
        if (val === true || val === 'True' || val === 'true') {
            this.eGui.innerText = '📈 W+';
            this.eGui.style.color = '#22c55e';
            this.eGui.title = 'Trend settimanale rialzista';
        } else if (val === false || val === 'False' || val === 'false') {
            this.eGui.innerText = '📉 W-';
            this.eGui.style.color = '#ef4444';
            this.eGui.title = 'Trend settimanale ribassista';
        } else {
            this.eGui.innerText = '—';
            this.eGui.style.color = '#6b7280';
        }
    }
    getGui() { return this.eGui; }
}
""")

quality_renderer = JsCode("""
class QualityRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const val = Number(params.value || 0);
        this.eGui.innerText = val;
        if (val >= 9) {
            this.eGui.style.color = '#22c55e';
            this.eGui.style.fontWeight = 'bold';
        } else if (val >= 6) {
            this.eGui.style.color = '#f59e0b';
        } else {
            this.eGui.style.color = '#9ca3af';
        }
    }
    getGui() { return this.eGui; }
}
""")

# =========================================================================
# PRESET PARAMETRI (NOVITÀ v21.0)
# =========================================================================

PRESETS = {
    "⚡ Aggressivo":   dict(eh=0.01, prmin=45, prmax=65, rpoc=0.01, top=20),
    "⚖️ Bilanciato":   dict(eh=0.02, prmin=40, prmax=70, rpoc=0.02, top=15),
    "🛡️ Conservativo": dict(eh=0.04, prmin=35, prmax=75, rpoc=0.04, top=10),
}

# =========================================================================
# INIT PAGINA
# =========================================================================

st.set_page_config(page_title="Trading Scanner PRO 21.0", layout="wide", page_icon="🧠")
st.title("🧠 Trading Scanner Versione PRO 21.0")
st.caption(
    "EARLY (continuo) • PRO • REA-HOT • CONFLUENCE • Squeeze • Divergenza RSI • "
    "Multi-TF • Quality Score • Storico Scansioni • Preset"
)

init_db()

if "init_done" not in st.session_state:
    st.session_state.init_done = True
    st.session_state.setdefault("mSP500", True)
    st.session_state.setdefault("mNasdaq", True)
    st.session_state.setdefault("mFTSE", True)
    st.session_state.setdefault("mEurostoxx", False)
    st.session_state.setdefault("mDow", False)
    st.session_state.setdefault("mRussell", False)
    st.session_state.setdefault("mStoxxEmerging", False)
    st.session_state.setdefault("mUSSmallCap", False)
    st.session_state.setdefault("eh", 0.02)
    st.session_state.setdefault("prmin", 40)
    st.session_state.setdefault("prmax", 70)
    st.session_state.setdefault("rpoc", 0.02)
    st.session_state.setdefault("vol_ratio_hot", 1.5)
    st.session_state.setdefault("top", 15)
    st.session_state.setdefault("current_list_name", "DEFAULT")
    st.session_state.setdefault("last_active_tab", "EARLY")

# =========================================================================
# SIDEBAR
# =========================================================================

st.sidebar.title("⚙️ Configurazione")

# --- PRESET rapidi ---
with st.sidebar.expander("🎯 Preset Rapidi", expanded=False):
    st.caption("Imposta automaticamente tutti i parametri scanner.")
    for preset_name, preset_vals in PRESETS.items():
        if st.button(preset_name, use_container_width=True, key=f"preset_{preset_name}"):
            for k, v in preset_vals.items():
                st.session_state[k] = v
            st.success(f"Preset '{preset_name}' applicato!")
            time.sleep(0.5)
            st.rerun()

# --- Mercati ---
with st.sidebar.expander("🌍 Selezione Mercati", expanded=True):
    msp500   = st.checkbox("S&P 500",          st.session_state.mSP500)
    mnasdaq  = st.checkbox("Nasdaq 100",        st.session_state.mNasdaq)
    mftse    = st.checkbox("FTSE MIB",          st.session_state.mFTSE)
    meuro    = st.checkbox("Eurostoxx 600",     st.session_state.mEurostoxx)
    mdow     = st.checkbox("Dow Jones",         st.session_state.mDow)
    mrussell = st.checkbox("Russell 2000",      st.session_state.mRussell)
    mstoxxem = st.checkbox("Stoxx Emerging 50", st.session_state.mStoxxEmerging)
    mussmall = st.checkbox("US Small Cap 2000", st.session_state.mUSSmallCap)

sel = []
if msp500:   sel.append("SP500")
if mnasdaq:  sel.append("Nasdaq")
if mftse:    sel.append("FTSE")
if meuro:    sel.append("Eurostoxx")
if mdow:     sel.append("Dow")
if mrussell: sel.append("Russell")
if mstoxxem: sel.append("StoxxEmerging")
if mussmall: sel.append("USSmallCap")

st.session_state.mSP500        = msp500
st.session_state.mNasdaq       = mnasdaq
st.session_state.mFTSE         = mftse
st.session_state.mEurostoxx    = meuro
st.session_state.mDow          = mdow
st.session_state.mRussell      = mrussell
st.session_state.mStoxxEmerging = mstoxxem
st.session_state.mUSSmallCap   = mussmall

# --- Parametri ---
with st.sidebar.expander("🎛️ Parametri Scanner", expanded=False):
    eh = st.slider(
        "EARLY - Distanza EMA20 %", 0.0, 10.0,
        float(st.session_state.eh * 100), 0.5
    ) / 100
    prmin = st.slider("PRO - RSI minimo",  0, 100, int(st.session_state.prmin), 5)
    prmax = st.slider("PRO - RSI massimo", 0, 100, int(st.session_state.prmax), 5)
    rpoc  = st.slider(
        "REA - Distanza POC %", 0.0, 10.0,
        float(st.session_state.rpoc * 100), 0.5
    ) / 100
    vol_ratio_hot = st.number_input(
        "VolRatio minimo REA-HOT", 0.0, 10.0,
        float(st.session_state.vol_ratio_hot), 0.1
    )
    top = st.number_input("TOP N titoli per tab", 5, 100, int(st.session_state.top), 5)

st.session_state.eh           = eh
st.session_state.prmin        = prmin
st.session_state.prmax        = prmax
st.session_state.rpoc         = rpoc
st.session_state.vol_ratio_hot = vol_ratio_hot
st.session_state.top          = top

# --- Watchlist ---
st.sidebar.divider()
st.sidebar.subheader("📋 Gestione Watchlist")

df_wl_all = load_watchlist()
list_options = sorted(df_wl_all["list_name"].unique()) if not df_wl_all.empty else ["DEFAULT"]
if "DEFAULT" not in list_options:
    list_options.append("DEFAULT")

active_list = st.sidebar.selectbox(
    "Lista Attiva",
    list_options,
    index=list_options.index(st.session_state.current_list_name)
    if st.session_state.current_list_name in list_options else 0,
    key="active_list",
)
st.session_state.current_list_name = active_list

new_list = st.sidebar.text_input("Crea Nuova Watchlist")
if st.sidebar.button("Crea") and new_list.strip():
    st.session_state.current_list_name = new_list.strip()
    st.success(f"Lista '{new_list.strip()}' creata!")
    time.sleep(1)
    st.rerun()

if st.sidebar.button("Reset DB Completo", help="Elimina tutte le watchlist!"):
    reset_watchlist_db()
    st.rerun()

only_watchlist = st.sidebar.checkbox("Mostra solo Watchlist", value=False)

# =========================================================================
# SCANNER
# =========================================================================

if not only_watchlist:
    if st.button("🚀 AVVIA SCANNER PRO 21.0", type="primary", use_container_width=True):
        universe = load_universe(sel)
        if not universe:
            st.warning("Seleziona almeno un mercato!")
        else:
            rep, rrea = [], []
            pb = st.progress(0)
            status = st.empty()
            tot = len(universe)
            for i, tkr in enumerate(universe, 1):
                status.text(f"Analisi {i}/{tot}: {tkr}")
                ep, rea = scan_ticker(tkr, eh, prmin, prmax, rpoc, vol_ratio_hot)
                if ep:
                    rep.append(ep)
                if rea:
                    rrea.append(rea)
                pb.progress(i / tot)

            df_ep_new  = pd.DataFrame(rep)
            df_rea_new = pd.DataFrame(rrea)

            st.session_state.df_ep  = df_ep_new
            st.session_state.df_rea = df_rea_new
            st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")

            # Salva nel DB storico
            save_scan_history(sel, df_ep_new, df_rea_new)

            # Alert HOT
            n_hot = len(df_rea_new) if not df_rea_new.empty else 0
            n_confluence = 0
            if not df_ep_new.empty and "Stato_Early" in df_ep_new.columns and "Stato_Pro" in df_ep_new.columns:
                n_confluence = int(
                    ((df_ep_new["Stato_Early"] == "EARLY") & (df_ep_new["Stato_Pro"] == "PRO")).sum()
                )

            if n_hot >= 5:
                st.toast(f"🔥 {n_hot} segnali REA-HOT trovati!", icon="🔥")
            if n_confluence >= 3:
                st.toast(f"⭐ {n_confluence} segnali CONFLUENCE (EARLY+PRO)!", icon="⭐")

            st.rerun()

df_ep  = st.session_state.get("df_ep",  pd.DataFrame())
df_rea = st.session_state.get("df_rea", pd.DataFrame())

if "last_scan" in st.session_state:
    st.caption(f"⏱️ Ultima scansione: {st.session_state.last_scan}")

# =========================================================================
# LEGENDA
# =========================================================================

def show_legend(title: str):
    with st.expander(f"📖 Legenda {title}", expanded=False):
        if title == "EARLY":
            st.markdown("""
**EARLY** — Titoli vicini alla EMA20 (trend in formazione).
- **Early_Score**: 0–10 continuo (più alto = più vicino alla EMA20). Novità v21.
- **Squeeze** 🔥 SQ: Bande di Bollinger dentro Keltner → esplosione imminente.
- **RSI_Div**: Divergenza RSI/Prezzo (BULL/BEAR).
- **Weekly_Bull**: Trend settimanale allineato rialzista.
""")
        elif title == "PRO":
            st.markdown("""
**PRO** — Segnali di forza con RSI neutrale-rialzista.
- **Pro_Score**: trend + RSI + volume (max 8).
- **Quality_Score**: score composito 0–12 (verde ≥9, arancio ≥6).
""")
        elif title == "REA-HOT":
            st.markdown("""
**REA-HOT** — Volumi anomali vicini al POC (Point Of Control).
- **Vol_Ratio**: volume odierno / media 20gg.
- **Dist_POC_%**: distanza percentuale dal POC del Volume Profile.
""")
        elif title == "CONFLUENCE":
            st.markdown("""
**CONFLUENCE** ⭐ — Titoli che soddisfano contemporaneamente EARLY **e** PRO.
Segnali ad alta probabilità: vicini alla EMA20 con trend + RSI + volumi forti.
Ordinati per **Quality_Score** decrescente.
""")
        elif title == "Regime Momentum":
            st.markdown("""
**Regime Momentum** — PRO ordinati per Momentum = Pro_Score × 10 + RSI.
Ideale per mercati trending con forte spinta direzionale.
""")
        elif title == "Multi-Timeframe":
            st.markdown("""
**Multi-Timeframe** — PRO con trend settimanale rialzista allineato (Weekly_Bull = True).
Filtro anti-contrarian: evita posizioni contro il trend di fondo.
""")
        else:
            st.markdown(f"Segnali scanner per il sistema **{title}**.")


# =========================================================================
# FUNZIONE GENERICA TAB SCANNER
# =========================================================================

def build_aggrid(df_disp: pd.DataFrame, grid_key: str):
    """Costruisce e restituisce la risposta AgGrid con configurazione comune."""
    gb = GridOptionsBuilder.from_dataframe(df_disp)
    gb.configure_default_column(sortable=True, resizable=True, filterable=True, editable=False)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)

    # Larghezze fisse
    col_widths = {
        "Ticker": 90, "Nome": 260,
        "Early_Score": 120, "Pro_Score": 100, "Quality_Score": 120,
        "RSI": 80, "Vol_Ratio": 100, "Squeeze": 100,
        "RSI_Div": 100, "Weekly_Bull": 100,
    }
    for col, w in col_widths.items():
        if col in df_disp.columns:
            gb.configure_column(col, width=w)

    # Renderer personalizzati
    if "Nome" in df_disp.columns:
        gb.configure_column("Nome", cellRenderer=name_dblclick_renderer)
    if "Squeeze" in df_disp.columns:
        gb.configure_column("Squeeze", cellRenderer=squeeze_renderer, headerName="Squeeze")
    if "RSI_Div" in df_disp.columns:
        gb.configure_column("RSI_Div", cellRenderer=rsi_div_renderer, headerName="RSI Div")
    if "Weekly_Bull" in df_disp.columns:
        gb.configure_column("Weekly_Bull", cellRenderer=weekly_renderer, headerName="TF Weekly")
    if "Quality_Score" in df_disp.columns:
        gb.configure_column("Quality_Score", cellRenderer=quality_renderer, headerName="Quality")

    grid_options = gb.build()

    return AgGrid(
        df_disp,
        gridOptions=grid_options,
        height=600,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
        theme="streamlit",
        allow_unsafe_jscode=True,
        key=grid_key,
    )


def render_scan_tab(df: pd.DataFrame, status_filter: str, sort_cols, ascending, title: str):
    st.subheader(f"📊 {title}")
    show_legend(title)

    if df.empty:
        st.info(f"Nessun dato {title}. Esegui lo scanner.")
        return

    # ------------------------------------------------------------------
    # FILTRO BASE
    # ------------------------------------------------------------------
    if status_filter == "EARLY" and "Stato_Early" in df.columns:
        df_f = df[df["Stato_Early"] == "EARLY"].copy()

    elif status_filter == "PRO" and "Stato_Pro" in df.columns:
        df_f = df[df["Stato_Pro"] == "PRO"].copy()

    elif status_filter == "HOT" and "Stato" in df.columns:
        df_f = df[df["Stato"] == "HOT"].copy()

    elif status_filter == "CONFLUENCE":
        if "Stato_Early" in df.columns and "Stato_Pro" in df.columns:
            df_f = df[
                (df["Stato_Early"] == "EARLY") & (df["Stato_Pro"] == "PRO")
            ].copy()
        else:
            df_f = pd.DataFrame()

    elif status_filter == "REGIME":
        stato_pro = df.get("Stato_Pro")
        df_f = df[stato_pro == "PRO"].copy() if stato_pro is not None else df.copy()
        if "Pro_Score" in df_f.columns and "RSI" in df_f.columns:
            df_f["Momentum"] = df_f["Pro_Score"] * 10 + df_f["RSI"]
            sort_cols = ["Momentum"]
            ascending = [False]

    elif status_filter == "MTF":
        stato_pro = df.get("Stato_Pro")
        df_f = df[stato_pro == "PRO"].copy() if stato_pro is not None else df.copy()
        # Filtra per trend settimanale rialzista
        if "Weekly_Bull" in df_f.columns:
            df_f = df_f[df_f["Weekly_Bull"] == True].copy()

    else:
        df_f = df.copy()

    if df_f.empty:
        st.info(f"Nessun segnale {title} trovato.")
        return

    # ------------------------------------------------------------------
    # ORDINAMENTO
    # ------------------------------------------------------------------
    valid_sort = [c for c in sort_cols if c in df_f.columns]
    valid_asc  = ascending[: len(valid_sort)]
    if valid_sort:
        df_f = df_f.sort_values(valid_sort, ascending=valid_asc)
    df_f = df_f.head(int(st.session_state.top))

    # ------------------------------------------------------------------
    # CONTATORI RAPIDI
    # ------------------------------------------------------------------
    m1, m2, m3 = st.columns(3)
    m1.metric("Titoli trovati", len(df_f))
    if "Squeeze" in df_f.columns:
        n_sq = int(df_f["Squeeze"].apply(lambda x: x is True or x == "True" or x == "true").sum())
        m2.metric("🔥 Con Squeeze", n_sq)
    if "Weekly_Bull" in df_f.columns:
        n_wb = int(df_f["Weekly_Bull"].apply(lambda x: x is True or x == "True" or x == "true").sum())
        m3.metric("📈 Weekly Bull", n_wb)

    # ------------------------------------------------------------------
    # FORMATTAZIONE
    # ------------------------------------------------------------------
    df_fmt  = add_formatted_cols(df_f)
    df_disp = prepare_display_df(df_fmt)

    for c in ["Yahoo", "TradingView"]:
        if c in df_disp.columns:
            df_disp = df_disp.drop(columns=[c])

    # Riordina colonne: Ticker, Nome in testa
    cols = list(df_disp.columns)
    base_cols = [c for c in ["Ticker", "Nome"] if c in cols]
    rest = [c for c in cols if c not in base_cols]
    df_disp = df_disp[base_cols + rest]

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------
    c1, c2 = st.columns([1, 2])
    with c1:
        get_csv_download_button(df_f, f"{title.lower().replace(' ', '_')}_export.csv", key=f"exp_{title}")
    with c2:
        st.markdown(
            f"Seleziona righe e clicca **Aggiungi selezionati** alla lista `{st.session_state.current_list_name}`."
        )

    # ------------------------------------------------------------------
    # AGGRID
    # ------------------------------------------------------------------
    grid_response = build_aggrid(df_disp, f"grid_{title}")

    selected_rows = grid_response["selected_rows"]
    selected_df   = pd.DataFrame(selected_rows)

    if st.button(f"➕ Aggiungi selezionati a '{st.session_state.current_list_name}'", key=f"btn_{title}"):
        if not selected_df.empty and "Ticker" in selected_df.columns:
            tickers = selected_df["Ticker"].tolist()
            names   = selected_df["Nome"].tolist() if "Nome" in selected_df.columns else tickers
            add_to_watchlist(tickers, names, title, "Scanner", "LONG", st.session_state.current_list_name)
            st.success(f"✅ Aggiunti {len(tickers)} titoli alla watchlist!")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Nessuna riga selezionata.")


# =========================================================================
# TABS PRINCIPALI
# =========================================================================

tabs = st.tabs([
    "EARLY", "PRO", "REA-HOT",
    "⭐ CONFLUENCE",
    "Regime Momentum", "Multi-Timeframe",
    "Finviz", "📋 Watchlist", "📜 Storico"
])
(
    tab_e, tab_p, tab_r,
    tab_conf,
    tab_regime, tab_mtf,
    tab_finviz, tab_w, tab_hist
) = tabs

# -------------------------------------------------------------------------
with tab_e:
    st.session_state.last_active_tab = "EARLY"
    render_scan_tab(df_ep, "EARLY", ["Early_Score", "RSI"], [False, True], "EARLY")

with tab_p:
    st.session_state.last_active_tab = "PRO"
    render_scan_tab(df_ep, "PRO", ["Quality_Score", "Pro_Score", "RSI"], [False, False, True], "PRO")

with tab_r:
    st.session_state.last_active_tab = "REA-HOT"
    render_scan_tab(df_rea, "HOT", ["Vol_Ratio", "Dist_POC_%"], [False, True], "REA-HOT")

# -------------------------------------------------------------------------
# CONFLUENCE (NOVITÀ v21.0)
# -------------------------------------------------------------------------
with tab_conf:
    st.session_state.last_active_tab = "CONFLUENCE"
    render_scan_tab(
        df_ep, "CONFLUENCE",
        ["Quality_Score", "Early_Score", "Pro_Score"], [False, False, False],
        "⭐ CONFLUENCE"
    )

# -------------------------------------------------------------------------
with tab_regime:
    render_scan_tab(df_ep, "REGIME", ["Pro_Score"], [False], "Regime Momentum")

# -------------------------------------------------------------------------
# MULTI-TIMEFRAME (filtra su Weekly_Bull)
# -------------------------------------------------------------------------
with tab_mtf:
    render_scan_tab(df_ep, "MTF", ["Quality_Score", "Pro_Score"], [False, False], "Multi-Timeframe")

# -------------------------------------------------------------------------
# FINVIZ (MarketCap + Vol_Ratio qualità)
# -------------------------------------------------------------------------
with tab_finviz:
    stato_pro = df_ep.get("Stato_Pro")
    if stato_pro is not None and not df_ep.empty:
        df_finviz = df_ep[stato_pro == "PRO"].copy()
    else:
        df_finviz = df_ep.copy()

    if not df_finviz.empty:
        if "MarketCap" in df_finviz.columns:
            median_mc = df_finviz["MarketCap"].median()
            df_finviz = df_finviz[df_finviz["MarketCap"] >= median_mc]
        if "Vol_Ratio" in df_finviz.columns:
            df_finviz = df_finviz[df_finviz["Vol_Ratio"] > 1.2]

    render_scan_tab(df_finviz, "PRO", ["Quality_Score", "Pro_Score"], [False, False], "Finviz")

# =========================================================================
# TAB WATCHLIST
# =========================================================================

with tab_w:
    st.subheader(f"📋 Watchlist '{st.session_state.current_list_name}'")
    df_w_view = load_watchlist()
    df_w_view = df_w_view[df_w_view["list_name"] == st.session_state.current_list_name]

    if df_w_view.empty:
        st.info("Watchlist vuota.")
    else:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

        with c1:
            get_csv_download_button(
                df_w_view,
                f"watchlist_{st.session_state.current_list_name}.csv",
                key="exp_wl",
            )

        with c2:
            move_target = st.selectbox("Sposta in", list_options, key="move_target")
            ids_to_move = st.multiselect("Seleziona ID da spostare", df_w_view["id"].tolist(), key="ids_move")
            if st.button("Sposta selezionati"):
                if ids_to_move:
                    move_watchlist_rows(ids_to_move, move_target)
                    st.rerun()

        with c3:
            ids_to_del = st.multiselect("Seleziona ID da eliminare", df_w_view["id"].tolist(), key="ids_del")
            if st.button("🗑️ Elimina selezionati"):
                if ids_to_del:
                    delete_from_watchlist(ids_to_del)
                    st.success(f"Eliminati {len(ids_to_del)} record.")
                    time.sleep(0.5)
                    st.rerun()

        with c4:
            new_name_wl = st.text_input("Rinomina lista corrente in:", key="rename_wl")
            if st.button("✏️ Rinomina"):
                if new_name_wl.strip():
                    rename_watchlist(st.session_state.current_list_name, new_name_wl.strip())
                    st.session_state.current_list_name = new_name_wl.strip()
                    st.rerun()

        df_wv = add_links(prepare_display_df(add_formatted_cols(df_w_view)))
        st.write(df_wv.to_html(escape=False, index=False), unsafe_allow_html=True)

    if st.button("🔄 Refresh Watchlist"):
        st.rerun()

# =========================================================================
# TAB STORICO SCANSIONI (NOVITÀ v21.0)
# =========================================================================

with tab_hist:
    st.subheader("📜 Storico Scansioni")
    st.caption("Le ultime 20 scansioni salvate nel database locale.")

    df_hist = load_scan_history(20)

    if df_hist.empty:
        st.info("Nessuna scansione nel database. Esegui lo scanner per iniziare.")
    else:
        st.dataframe(df_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Confronto Snapshot")
        col_a, col_b = st.columns(2)
        with col_a:
            id_a = st.selectbox("Scansione A", df_hist["id"].tolist(), key="snap_a")
        with col_b:
            id_b = st.selectbox("Scansione B", df_hist["id"].tolist(),
                                index=min(1, len(df_hist) - 1), key="snap_b")

        if st.button("Confronta"):
            ep_a, _ = load_scan_snapshot(id_a)
            ep_b, _ = load_scan_snapshot(id_b)

            if ep_a.empty or ep_b.empty:
                st.warning("Dati non disponibili per uno dei due snapshot.")
            else:
                tickers_a = set(ep_a["Ticker"]) if "Ticker" in ep_a.columns else set()
                tickers_b = set(ep_b["Ticker"]) if "Ticker" in ep_b.columns else set()

                nuovi   = tickers_b - tickers_a
                rimossi = tickers_a - tickers_b
                comuni  = tickers_a & tickers_b

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("🆕 Nuovi segnali", len(nuovi))
                mc2.metric("❌ Usciti", len(rimossi))
                mc3.metric("✅ Persistenti", len(comuni))

                if nuovi:
                    st.markdown("**🆕 Nuovi titoli in B non presenti in A:**")
                    st.write(", ".join(sorted(nuovi)))
                if rimossi:
                    st.markdown("**❌ Titoli usciti (in A ma non in B):**")
                    st.write(", ".join(sorted(rimossi)))


# =========================================================================
# EXPORT GLOBALI
# =========================================================================

st.markdown("---")
st.subheader("💾 Export Globali")

df_confluence_exp = pd.DataFrame()
if not df_ep.empty and "Stato_Early" in df_ep.columns and "Stato_Pro" in df_ep.columns:
    df_confluence_exp = df_ep[
        (df_ep["Stato_Early"] == "EARLY") & (df_ep["Stato_Pro"] == "PRO")
    ].copy()

df_w_view_exp = load_watchlist()
df_w_view_exp = df_w_view_exp[
    df_w_view_exp["list_name"] == st.session_state.current_list_name
]

all_tabs_raw = {
    "EARLY":      df_ep,
    "PRO":        df_ep,
    "REA-HOT":    df_rea,
    "CONFLUENCE": df_confluence_exp,
    "Watchlist":  df_w_view_exp,
}

ec1, ec2, ec3, ec4 = st.columns(4)

with ec1:
    xlsx_all = to_excel_bytes_sheets_dict(all_tabs_raw)
    st.download_button(
        label="📊 XLSX Tutti i tab",
        data=xlsx_all,
        file_name="TradingScanner_v21_Tutti.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xlsx_all_tabs",
    )

with ec2:
    tv_rows = []
    for name, df_tab in all_tabs_raw.items():
        if isinstance(df_tab, pd.DataFrame) and not df_tab.empty and "Ticker" in df_tab.columns:
            tmp = df_tab[["Ticker"]].copy()
            tmp.insert(0, "Tab", name)
            tv_rows.append(tmp)
    if tv_rows:
        df_tv_all = pd.concat(tv_rows, ignore_index=True).drop_duplicates(subset=["Ticker"])
        csv_tv_all = df_tv_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📈 CSV TradingView Tutti",
            data=csv_tv_all,
            file_name="TradingScanner_v21_TradingView_Tutti.csv",
            mime="text/csv",
            key="csv_tv_all_tabs",
        )

with ec3:
    current_tab = st.session_state.get("last_active_tab", "EARLY")
    df_current  = all_tabs_raw.get(current_tab, pd.DataFrame())
    xlsx_current = to_excel_bytes_sheets_dict({current_tab: df_current})
    st.download_button(
        label=f"📊 XLSX Tab corrente ({current_tab})",
        data=xlsx_current,
        file_name=f"TradingScanner_v21_{current_tab}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="xlsx_current_tab",
    )

with ec4:
    if isinstance(df_current, pd.DataFrame) and not df_current.empty and "Ticker" in df_current.columns:
        csv_tv_current = make_tv_csv(df_current, current_tab, "Ticker")
        st.download_button(
            label=f"📈 CSV TradingView ({current_tab})",
            data=csv_tv_current,
            file_name=f"TradingScanner_v21_{current_tab}_TV.csv",
            mime="text/csv",
            key="csv_tv_current_tab",
        )
