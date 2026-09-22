"""
Script di upgrade: sostituisce il blocco 'with tab_insider:' in
Dashboard_pro_V_45_09.py con una versione che supporta due modalità:
- Feed Rapido (istantaneo, ultime ore)
- Storico Esteso (più giorni, con cache su file locale data/)

Uso:
    cd /workspaces/trading-system
    source .venv/bin/activate
    python upgrade_insider_tab_v2.py
"""

import re
from pathlib import Path

path = Path("Dashboard_pro_V_45_09.py")
text = path.read_text(encoding="utf-8")

start_marker = "# INSIDER RADAR TAB (V45.09)"
end_marker = "with tab_macro:"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
    raise SystemExit(
        "ERRORE: marker non trovati o in ordine sbagliato. "
        "Nessuna modifica applicata. Verifica che add_insider_radar_tab.py "
        "sia già stato eseguito con successo."
    )

# Risali all'inizio della riga del blocco commento "# ====...="
block_start = text.rfind("# =========================================================================", 0, start_idx)
if block_start == -1:
    block_start = start_idx

new_block = '''# =========================================================================
# INSIDER RADAR TAB (V45.09) — Feed Rapido + Storico Esteso
# =========================================================================
with tab_insider:
    st.markdown('<div class="section-pill">🕵️ INSIDER RADAR — Acquisti CEO, Manager e Insider (Form 4 SEC)</div>',
                unsafe_allow_html=True)
    st.markdown("""
> **Come usare questo tab**: la SEC richiede agli insider (CEO, CFO, director, azionisti >10%)
> di dichiarare ogni transazione entro 2 giorni lavorativi (Form 4).
> **Feed Rapido** = ultimi filing in assoluto su tutto EDGAR, istantaneo ma copre solo poche ore
> se il mercato è molto attivo. **Storico Esteso** = analizza N giorni completi tramite l'indice
> giornaliero SEC — più lento (alcuni minuti) ma esaustivo, con cache salvata su disco.
""")

    if not _HAS_INSIDER_RADAR:
        st.error("Modulo Insider Radar non disponibile.")
        st.caption("Errore import: " + str(_insider_import_error))
    else:
        import os as _ir_os
        _IR_CACHE_FILE = "data/insider_radar_extended_cache.csv"

        _ir_mode = st.radio(
            "Modalità di raccolta dati",
            ["⚡ Feed Rapido", "📊 Storico Esteso"],
            key="insider_mode",
            horizontal=True,
        )

        _df_insider_raw = pd.DataFrame()

        if _ir_mode == "⚡ Feed Rapido":
            _ir_c1, _ir_c2 = st.columns(2)
            with _ir_c1:
                _ir_max_filings = st.select_slider(
                    "Filing da analizzare (ultimi in assoluto)",
                    options=[20, 40, 60, 100, 150],
                    value=60,
                    key="insider_max_filings",
                    help="Numero di filing Form 4 più recenti da scaricare e analizzare (istantaneo)."
                )
            with _ir_c2:
                _ir_run = st.button("🔍 Aggiorna Feed Rapido", key="insider_scan_btn", type="primary")

            _ir_cache_key = "_insider_df_cache"
            if _ir_run or _ir_cache_key not in st.session_state:
                with st.spinner("Scaricamento Form 4 da SEC EDGAR (feed rapido)..."):
                    _df_insider_raw = fetch_recent_insider_transactions(max_filings=_ir_max_filings)
                st.session_state[_ir_cache_key] = _df_insider_raw
                st.session_state["_insider_scan_time"] = datetime.now().strftime("%H:%M")

            _df_insider_raw = st.session_state.get(_ir_cache_key, pd.DataFrame())
            _ir_ts = st.session_state.get("_insider_scan_time", "")
            if not _df_insider_raw.empty:
                st.caption(f"✅ {len(_df_insider_raw)} transazioni (feed rapido)" + (f" · aggiornato alle {_ir_ts}" if _ir_ts else ""))

        else:
            _ir_e1, _ir_e2, _ir_e3 = st.columns([1.5, 1.5, 1])
            with _ir_e1:
                _ir_days = st.slider("Giorni lavorativi da analizzare", 1, 10, 3, key="insider_days_back")
            with _ir_e2:
                _ir_max_ext = st.number_input(
                    "Limite massimo filing da scaricare",
                    min_value=50, max_value=1500, value=300, step=50,
                    key="insider_max_ext"
                )
            with _ir_e3:
                _ir_est_min = _ir_max_ext * 0.6 / 60
                st.metric("Tempo stimato", f"~{_ir_est_min:.1f} min")

            _ir_run_ext = st.button("🔍 Avvia Analisi Storica", key="insider_ext_scan_btn", type="primary",
                                      help="Scarica e analizza i filing Form 4 dell'intervallo scelto. Può richiedere alcuni minuti.")

            _ir_ext_cache_key = "_insider_df_cache_extended"

            if _ir_run_ext:
                with st.spinner(f"Analisi di {_ir_max_ext} filing su {_ir_days} giorni lavorativi (~{_ir_est_min:.1f} min)..."):
                    _df_ext = fetch_insider_transactions_extended(days_back=_ir_days, max_filings=_ir_max_ext)
                st.session_state[_ir_ext_cache_key] = _df_ext
                st.session_state["_insider_ext_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                try:
                    _ir_os.makedirs("data", exist_ok=True)
                    _df_ext.to_csv(_IR_CACHE_FILE, index=False)
                except Exception:
                    pass

            _df_insider_raw = st.session_state.get(_ir_ext_cache_key)
            _ir_loaded_from_disk = False
            if _df_insider_raw is None:
                if _ir_os.path.exists(_IR_CACHE_FILE):
                    try:
                        _df_insider_raw = pd.read_csv(_IR_CACHE_FILE)
                        _ir_loaded_from_disk = True
                    except Exception:
                        _df_insider_raw = pd.DataFrame()
                else:
                    _df_insider_raw = pd.DataFrame()

            _ir_ext_ts = st.session_state.get("_insider_ext_scan_time", "")
            if not _df_insider_raw.empty:
                _src_note = "📁 caricato da cache su disco" if _ir_loaded_from_disk else f"aggiornato il {_ir_ext_ts}"
                st.caption(f"✅ {len(_df_insider_raw)} transazioni (storico esteso) · {_src_note}")

        if _df_insider_raw is None or _df_insider_raw.empty:
            st.warning("Nessuna transazione disponibile. Avvia una raccolta dati con i pulsanti sopra.")
        else:
            _ir_c1f, _ir_c2f, _ir_c3f = st.columns(3)
            with _ir_c1f:
                _ir_min_value = st.number_input(
                    "Valore minimo transazione ($)",
                    min_value=0, value=10000, step=5000,
                    key="insider_min_value"
                )
            with _ir_c2f:
                _ir_role_filter = st.multiselect(
                    "Ruolo",
                    options=["CEO/CFO", "Director", "10% Owner", "Altro"],
                    default=["CEO/CFO", "Director", "10% Owner", "Altro"],
                    key="insider_role_filter"
                )
            with _ir_c3f:
                _ir_only_buys = st.checkbox("Solo acquisti (P)", value=True, key="insider_only_buys")

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

            st.markdown("### 📋 Transazioni")
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

text = text[:block_start] + new_block + text[end_idx:]
path.write_text(text, encoding="utf-8")
print("OK: tab Insider Radar aggiornata con modalità Feed Rapido + Storico Esteso.")
