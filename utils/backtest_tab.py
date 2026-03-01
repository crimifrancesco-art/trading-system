"""
backtest_tab.py  —  Upgrade #4
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

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

# ── Import db functions ────────────────────────────────────────────────────
try:
    from utils.db import (load_signals, signal_summary_stats,
                          update_signal_performance, cache_stats)
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# ── Colori per tipo segnale ────────────────────────────────────────────────
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

    # ── Controlli ─────────────────────────────────────────────────────────
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

    # ── Carica dati ───────────────────────────────────────────────────────
    df_sigs = load_signals(signal_type=sig_type_arg,
                           days_back=days_back, with_perf=True)
    df_summ = signal_summary_stats(days_back=days_back)

    if df_sigs.empty:
        st.info(
            "📭 Nessun segnale registrato ancora.\n\n"
            "**Come iniziare:** esegui lo scanner almeno una volta con il db.py v28 attivo. "
            "I segnali vengono registrati automaticamente ad ogni scansione. "
            "Dopo 1-5 giorni avrai dati sufficienti per il backtest."
        )
        # Mostra istruzioni setup
        with st.expander("🛠️ Setup — come funziona il backtest", expanded=True):
            st.markdown("""
**Flusso automatico:**
1. Ogni volta che clicchi **🚀 AVVIA SCANNER**, i segnali (EARLY/PRO/HOT/ecc.) 
   vengono salvati nel DB con il prezzo di quel momento
2. Il giorno dopo, clicca **🔄 Aggiorna performance** — il sistema scarica 
   i prezzi forward (+1g/+5g/+10g/+20g) e calcola i rendimenti
3. Dopo qualche settimana hai statistiche affidabili

**Colonne chiave:**
- `ret_1d / ret_5d / ret_10d / ret_20d` → rendimento % dal prezzo di entrata
- `Win%` → % di segnali con rendimento positivo
- `Avg Ret` → rendimento medio per tipo di segnale

**Nota:** il backtest è *forward-looking puro* — 
misura cosa sarebbe successo comprando al prezzo del giorno del segnale.
Non è backtesting storico con curve ottimizzate — è più onesto.
""")
        return

    st.caption(
        f"📊 {len(df_sigs)} segnali negli ultimi {days_back} giorni  "
        f"| {df_sigs['ticker'].nunique()} ticker unici  "
        f"| {df_sigs['ret_20d'].notna().sum()} con performance completa"
    )

    # ─────────────────────────────────────────────────────────────────────
    # 📊 RIEPILOGO  per tipo segnale
    # ─────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────
    # 📈 EQUITY CURVE  — curva cumulata per tipo segnale
    # ─────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────
    # 🥇 TOP PERFORMER  — migliori e peggiori ticker
    # ─────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────
    # 🔍 GRIGLIA DETTAGLIO
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Dettaglio segnali registrati")

    disp_cols = ["scanned_at", "ticker", "nome", "signal_type",
                 "entry_price", "rsi", "quality_score", "ser_score", "fv_score",
                 "squeeze", "weekly_bull",
                 "ret_1d", "ret_5d", "ret_10d", "ret_20d"]
    disp_cols = [c for c in disp_cols if c in df_sigs.columns]
    df_disp   = df_sigs[disp_cols].copy()

    # Rename
    df_disp = df_disp.rename(columns={
        "scanned_at": "Data", "ticker": "Ticker", "nome": "Nome",
        "signal_type": "Tipo", "entry_price": "Prezzo",
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

    # ─────────────────────────────────────────────────────────────────────
    # 💾 EXPORT + STATS CACHE
    # ─────────────────────────────────────────────────────────────────────
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
