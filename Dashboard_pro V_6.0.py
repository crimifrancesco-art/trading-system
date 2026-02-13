# =============================================================================
# TAB ALERT DESIGNER (VERSIONE ROBUSTA)
# =============================================================================
with tab_alert:
    st.subheader("🔔 Alert Designer per TradingView")
    st.markdown(
        "Costruisci una lista di ticker + commento standard per usarli come alert in TradingView."
    )

    # Costruzione sicura delle sorgenti
    sources = {}

    # EARLY (top)
    if not df_early_all.empty and "Early_Score" in df_early_all.columns:
        sources["EARLY (top)"] = (
            df_early_all.sort_values("Early_Score", ascending=False).head(top)
        )

    # PRO (top)
    if not df_pro_all.empty and "Pro_Score" in df_pro_all.columns:
        sources["PRO (top)"] = (
            df_pro_all.sort_values("Pro_Score", ascending=False).head(top)
        )

    # REA_HOT (top)
    if not df_rea_all.empty and "Rea_Score" in df_rea_all.columns:
        sources["REA_HOT (top)"] = (
            df_rea_all.sort_values("Rea_Score", ascending=False).head(top)
        )

    # MTF ALIGN_LONG (se disponibile)
    if "df_mtf" in locals() and isinstance(df_mtf, pd.DataFrame) and not df_mtf.empty:
        if "Segnale_MTF" in df_mtf.columns and "MTF_Score" in df_mtf.columns:
            mtf_long_src = df_mtf[df_mtf["Segnale_MTF"] == "ALIGN_LONG"]
            if not mtf_long_src.empty:
                sources["MTF ALIGN_LONG"] = (
                    mtf_long_src.sort_values("MTF_Score", ascending=False).head(top)
                )

    if not sources:
        st.caption(
            "Nessun insieme di segnali disponibile al momento per costruire alert "
            "(controlla che ci siano segnali EARLY/PRO/REA/MTF in questa scansione)."
        )
    else:
        sel_sources = st.multiselect(
            "Seleziona insiemi da includere negli alert",
            options=list(sources.keys()),
            default=[list(sources.keys())[0]],
        )
        alert_comment = st.text_input(
            "Commento/Tag standard per alert (es: PRO6_MTF_LONG)",
            "PRO6_SIGNAL",
        )
        prefix = st.text_input("Prefisso script TV (opzionale)", "")

        all_rows = []
        for src in sel_sources:
            df_src = sources.get(src, pd.DataFrame())
            if df_src.empty or "Ticker" not in df_src.columns:
                continue
            for t in df_src["Ticker"].tolist():
                all_rows.append({"symbol": t, "comment": alert_comment, "source": src})

        if all_rows:
            df_alert = pd.DataFrame(all_rows).drop_duplicates(subset=["symbol", "comment"])
            if prefix:
                df_alert["symbol"] = prefix + df_alert["symbol"]

            st.markdown("**Preview alert**")
            st.dataframe(df_alert, use_container_width=True)

            csv_alert = (
                df_alert[["symbol", "comment"]]
                .to_csv(index=False, header=False)
                .encode("utf-8")
            )
            st.download_button(
                "⬇️ CSV Alert (symbol, comment)",
                data=csv_alert,
                file_name=f"alerts_tradingview_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Nessun simbolo utile selezionato per gli alert.")
