# =============================================================================
# 📌 WATCHLIST & NOTE
# =============================================================================
with tab_watch:
    st.subheader("📌 Watchlist & Note (DB persistente)")
    st.markdown(
        "Gestisci la watchlist unificata: aggiunte dai tab, note, cancellazioni, export PDF/XLSX/CSV."
    )

    col_reset, col_refresh = st.columns(2)
    if col_reset.button("🧹 Svuota completamente la Watchlist (reset DB)"):
        reset_watchlist_db()
        st.success("Watchlist e DB azzerati.")
        st.experimental_rerun()

    # nuovo pulsante di refresh manuale
    if col_refresh.button("🔄 Refresh Watchlist"):
        st.experimental_rerun()

    wl_df = load_watchlist()

    if wl_df.empty:
        st.caption("La watchlist è vuota. Aggiungi ticker dagli altri tab.")
    else:
        st.markdown("### Watchlist corrente")
        st.dataframe(wl_df, use_container_width=True)

        st.markdown("### Modifica nota di una riga")
        labels = [
            f"{r['ticker']} – {r.get('name','')} ({r.get('origine','')}) - {r.get('created_at','')}"
            for _, r in wl_df.iterrows()
        ]
        ids = wl_df["id"].astype(str).tolist()
        mapping = dict(zip(labels, ids))

        sel_label = st.selectbox("Seleziona riga", options=labels)
        sel_id = int(mapping[sel_label])
        current_note = wl_df.loc[wl_df["id"] == sel_id, "note"].values[0] or ""
        new_note = st.text_input("Nuova nota", value=current_note, key="wl_edit_note")

        col_upd, col_del = st.columns(2)
        if col_upd.button("💾 Aggiorna nota"):
            update_watchlist_note(sel_id, new_note)
            st.success("Nota aggiornata.")
            st.experimental_rerun()

        st.markdown("### Rimuovi più elementi")
        ids_to_delete = st.multiselect(
            "Seleziona elementi da rimuovere",
            options=wl_df["id"].astype(str).tolist(),
            format_func=lambda x: f"{wl_df.loc[wl_df['id']==int(x),'ticker'].values[0]} – "
                                  f"{wl_df.loc[wl_df['id']==int(x),'name'].values[0]} "
                                  f"({wl_df.loc[wl_df['id']==int(x),'origine'].values[0]})",
        )
        if col_del.button("🗑️ Rimuovi selezionati"):
            delete_from_watchlist(ids_to_delete)
            st.success("Elementi rimossi dalla watchlist.")
            st.experimental_rerun()

        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            wl_df.to_excel(writer, index=False, sheet_name="WATCHLIST")
        xlsx_bytes = out_xlsx.getvalue()

        csv_tickers = wl_df[["ticker"]].drop_duplicates().to_csv(
            index=False, header=False
        ).encode("utf-8")

        pdf_buffer = io.BytesIO()
        try:
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 12)
                    self.cell(0, 10, "Watchlist & Note", 0, 1, "C")
                    self.ln(2)

            pdf = PDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=8)

            pdf.set_font("Arial", "B", 8)
            pdf.cell(30, 6, "Ticker", 1)
            pdf.cell(50, 6, "Nome", 1)
            pdf.cell(25, 6, "Origine", 1)
            pdf.cell(35, 6, "Data", 1)
            pdf.cell(50, 6, "Note", 1)
            pdf.ln()

            pdf.set_font("Arial", size=8)
            for _, row in wl_df.iterrows():
                pdf.cell(30, 6, str(row["ticker"])[:12], 1)
                pdf.cell(50, 6, str(row.get("name",""))[:22], 1)
                pdf.cell(25, 6, str(row.get("origine",""))[:10], 1)
                pdf.cell(35, 6, str(row.get("created_at",""))[:16], 1)
                note_txt = (row["note"] or "")[:30]
                pdf.cell(50, 6, note_txt, 1)
                pdf.ln()

            pdf.output(pdf_buffer)
            pdf_bytes = pdf_buffer.getvalue()
            pdf_ok = True
        except Exception:
            pdf_ok = False
            pdf_bytes = b""

        c1, c2, c3 = st.columns(3)
        if pdf_ok:
            c1.download_button(
                "⬇️ PDF Watchlist",
                data=pdf_bytes,
                file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            c1.caption("PDF non disponibile (errore nella generazione).")

        c2.download_button(
            "⬇️ XLSX Watchlist",
            data=xlsx_bytes,
            file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        c3.download_button(
            "⬇️ CSV Watchlist (solo ticker)",
            data=csv_tickers,
            file_name=f"watchlist_tickers_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
