"""
Script di integrazione: aggiunge la tab "🕵️ Insider Radar" a
Dashboard_pro_V_45_09.py.

Uso:
    cd /workspaces/trading-system
    source .venv/bin/activate
    python add_insider_radar_tab.py
"""

from pathlib import Path

path = Path("Dashboard_pro_V_45_09.py")
text = path.read_text(encoding="utf-8")

# ── 1. Import del modulo (stesso pattern di macro_regime) ───────────────
old_import = '''# ── V45.04: Macro Regime Engine ────────────────────────────────────────────
try:
    from utils.macro_regime import render_macro_regime
    _HAS_MACRO_REGIME = True
except Exception as _macro_import_error:
    _HAS_MACRO_REGIME = False
    render_macro_regime = None'''

new_import = '''# ── V45.04: Macro Regime Engine ────────────────────────────────────────────
try:
    from utils.macro_regime import render_macro_regime
    _HAS_MACRO_REGIME = True
except Exception as _macro_import_error:
    _HAS_MACRO_REGIME = False
    render_macro_regime = None

# ── V45.09: Insider Radar (Form 4 SEC) ──────────────────────────────────────
try:
    from utils.insider_radar import fetch_recent_insider_transactions, compute_insider_score
    _HAS_INSIDER_RADAR = True
except Exception as _insider_import_error:
    _HAS_INSIDER_RADAR = False
    fetch_recent_insider_transactions = None
    compute_insider_score = None'''

# ── 2. Nuovo tab nell'elenco ──────────────────────────────────────────────
old_tabs_list = '''    "🎯 Opportunity Radar",
    "🌡️ Macro Regime",'''
new_tabs_list = '''    "🎯 Opportunity Radar",
    "🕵️ Insider Radar",
    "🌡️ Macro Regime",'''

# ── 3. Nuova variabile nell'unpacking ────────────────────────────────────
old_unpack = '''(tab_home, tab_e, tab_p, tab_r, tab_conf,
 tab_ser, tab_fvpro, tab_radar, tab_macro, tab_crisis,
 tab_regime, tab_mtfmatrix, tab_mtf, tab_bcd, tab_of, tab_rm, tab_bt,
 tab_ai, tab_ai2, tab_opts, tab_mom, tab_news,
 tab_analisi, tab_journal,
 tab_w) = tabs'''
new_unpack = '''(tab_home, tab_e, tab_p, tab_r, tab_conf,
 tab_ser, tab_fvpro, tab_radar, tab_insider, tab_macro, tab_crisis,
 tab_regime, tab_mtfmatrix, tab_mtf, tab_bcd, tab_of, tab_rm, tab_bt,
 tab_ai, tab_ai2, tab_opts, tab_mom, tab_news,
 tab_analisi, tab_journal,
 tab_w) = tabs'''

# ── 4. Blocco with tab_insider: (inserito prima di with tab_macro:) ─────
anchor = '''# ── V45.08: tab Opportunity Radar (strategie selezionabili) ────────────
# ── V45.09: tab Insider Radar (Form 4 SEC, insider score, cluster) ─────'''

insider_tab_block = '''# =========================================================================
# INSIDER RADAR TAB (V45.09)
# =========================================================================
with tab_insider:
    st.markdown('<div class="section-pill">🕵️ INSIDER RADAR — Acquisti CEO, Manager e Insider (Form 4 SEC)</div>',
                unsafe_allow_html=True)
    st.markdown("""
> **Come usare questo tab**: la SEC richiede agli insider (CEO, CFO, director, azionisti >10%)
> di dichiarare ogni transazione entro 2 giorni lavorativi (Form 4). Qui trovi il feed delle
> transazioni più recenti e un **Insider Score** che privilegia acquisti sul mercato, ruoli chiave
> (CEO/CFO) e cluster di più insider che comprano nello stesso periodo.
""")

    if not _HAS_INSIDER_RADAR:
        st.error("Modulo Insider Radar non disponibile.")
        st.caption("Errore import: " + str(_insider_import_error))
    else:
        _ir_c1, _ir_c2, _ir_c3, _ir_c4 = st.columns([2, 2, 2, 2])
        with _ir_c1:
            _ir_max_filings = st.select_slider(
                "Filing da analizzare",
                options=[20, 40, 60, 100, 150],
                value=60,
                key="insider_max_filings",
                help="Numero di filing Form 4 più recenti da scaricare e analizzare"
            )
        with _ir_c2:
            _ir_min_value = st.number_input(
                "Valore minimo transazione ($)",
                min_value=0, value=10000, step=5000,
                key="insider_min_value"
            )
        with _ir_c3:
            _ir_role_filter = st.multiselect(
                "Ruolo",
                options=["CEO/CFO", "Director", "10% Owner", "Altro"],
                default=["CEO/CFO", "Director", "10% Owner", "Altro"],
                key="insider_role_filter"
            )
        with _ir_c4:
            _ir_only_buys = st.checkbox("Solo acquisti (P)", value=True, key="insider_only_buys")

        _ir_run = st.button("🔍 Aggiorna dati Insider", key="insider_scan_btn", type="primary")

        _ir_cache_key = "_insider_df_cache"
        if _ir_run or _ir_cache_key not in st.session_state:
            with st.spinner("Scaricamento Form 4 da SEC EDGAR..."):
                _df_insider_raw = fetch_recent_insider_transactions(max_filings=_ir_max_filings)
            st.session_state[_ir_cache_key] = _df_insider_raw
            st.session_state["_insider_scan_time"] = datetime.now().strftime("%H:%M")

        _df_insider_raw = st.session_state.get(_ir_cache_key, pd.DataFrame())
        _ir_ts = st.session_state.get("_insider_scan_time", "")

        if _df_insider_raw is None or _df_insider_raw.empty:
            st.warning("Nessuna transazione trovata. Premi 'Aggiorna dati Insider' per scaricare i filing più recenti.")
        else:
            st.caption(f"✅ {len(_df_insider_raw)} transazioni analizzate" + (f" · aggiornato alle {_ir_ts}" if _ir_ts else ""))

            _df_view_insider = _df_insider_raw.copy()
            if _ir_only_buys:
                _df_view_insider = _df_view_insider[_df_view_insider["CodiceSEC"] == "P"]
            _df_view_insider = _df_view_insider[_df_view_insider["Valore"] >= _ir_min_value]

            def _ir_role_bucket(r):
                ru = str(r).upper()
                if "CEO" in ru or "CFO" in ru or "CHIEF EXECUTIVE" in ru or "CHIEF FINANCIAL" in ru:
                    return "CEO/CFO"
                if "10% OWNER" in ru:
                    return "10% Owner"
                if "DIRECTOR" in ru:
                    return "Director"
                return "Altro"

            if not _df_view_insider.empty:
                _df_view_insider = _df_view_insider.copy()
                _df_view_insider["RuoloGruppo"] = _df_view_insider["Ruolo"].apply(_ir_role_bucket)
                _df_view_insider = _df_view_insider[_df_view_insider["RuoloGruppo"].isin(_ir_role_filter)]

            st.markdown("### 📋 Transazioni recenti")
            if _df_view_insider.empty:
                st.info("Nessuna transazione corrisponde ai filtri selezionati.")
            else:
                st.dataframe(
                    _df_view_insider[["Ticker", "Issuer", "Insider", "Ruolo", "Tipo",
                                       "Azioni", "Prezzo", "Valore", "DataTransazione", "Is10b5_1"]]
                    .sort_values("Valore", ascending=False),
                    column_config={
                        "Valore": st.column_config.NumberColumn("Valore ($)", format="$%.0f"),
                        "Prezzo": st.column_config.NumberColumn("Prezzo ($)", format="$%.2f"),
                        "Is10b5_1": st.column_config.CheckboxColumn("Piano 10b5-1"),
                    },
                    hide_index=True, use_container_width=True, key="insider_table_raw"
                )

            st.markdown("### 🏆 Insider Score per Ticker")
            _ir_base_for_score = _df_insider_raw[_df_insider_raw["Valore"] >= _ir_min_value]
            _scored = compute_insider_score(_ir_base_for_score)
            if _scored.empty:
                st.info("Nessun acquisto sul mercato sufficientemente rilevante nel campione analizzato.")
            else:
                st.dataframe(
                    _scored,
                    column_config={
                        "Valore_Netto": st.column_config.NumberColumn("Valore Netto ($)", format="$%.0f"),
                        "Insider_Score": st.column_config.ProgressColumn("Insider Score", min_value=0, max_value=100, format="%.0f"),
                        "CEO_CFO_Coinvolto": st.column_config.CheckboxColumn("CEO/CFO"),
                        "Cluster": st.column_config.CheckboxColumn("Cluster ≥2 insider"),
                    },
                    hide_index=True, use_container_width=True, key="insider_score_table"
                )

                _ir_exp_ts = datetime.now().strftime("%Y%m%d_%H%M")
                _ir_ex1, _ir_ex2, _ir_ex3 = st.columns(3)
                with _ir_ex1:
                    st.download_button(
                        "📺 TradingView TXT",
                        data=make_tv_txt(_scored, section="INSIDER_RADAR"),
                        file_name=f"InsiderRadar_{_ir_exp_ts}.txt",
                        mime="text/plain",
                        key="insider_tv_txt",
                        disabled=_scored.empty,
                        help="Formato TradingView: ###INSIDER_RADAR,ticker separati da virgole,"
                    )
                with _ir_ex2:
                    st.download_button(
                        "📊 Export Excel",
                        data=to_excel_bytes({"Insider Score": _scored, "Transazioni": _df_view_insider}),
                        file_name=f"InsiderRadar_{_ir_exp_ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="insider_xlsx_exp"
                    )
                with _ir_ex3:
                    _ir_sel = st.multiselect(
                        "Aggiungi a Watchlist",
                        options=_scored["Ticker"].tolist(),
                        key="insider_wl_select"
                    )
                    if st.button("➕ Aggiungi selezionati", key="insider_wl_add"):
                        if _ir_sel:
                            _ir_names = [
                                _scored.loc[_scored["Ticker"] == t, "Issuer"].values[0]
                                if (_scored["Ticker"] == t).any() else t
                                for t in _ir_sel
                            ]
                            gh_add_to_watchlist(_ir_sel, _ir_names, "InsiderRadar", "InsiderRadar",
                                                 "WATCH", st.session_state.current_list_name)
                            _wl_save_backup()
                            st.success(f"✅ Aggiunti {len(_ir_sel)} ticker."); time.sleep(0.5); st.rerun()
                        else:
                            st.warning("Seleziona almeno un ticker.")


'''

checks = {
    "import macro_regime (ancora)": old_import in text,
    "elenco tabs (Opportunity Radar / Macro Regime)": old_tabs_list in text,
    "unpacking tabs": old_unpack in text,
    "ancora blocco tab_insider": anchor in text,
}
missing = [k for k, v in checks.items() if not v]
if missing:
    raise SystemExit("ERRORE: pattern non trovati, nessuna modifica applicata:\\n" + "\\n".join(missing))

text = text.replace(old_import, new_import, 1)
text = text.replace(old_tabs_list, new_tabs_list, 1)
text = text.replace(old_unpack, new_unpack, 1)
text = text.replace(anchor, anchor + "\\n" + insider_tab_block, 1)

path.write_text(text, encoding="utf-8")
print("OK: tab Insider Radar integrata in Dashboard_pro_V_45_09.py")
