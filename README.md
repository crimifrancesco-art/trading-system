# Trading System — Dashboard Pro

Automated trading scanner e dashboard di analisi finanziaria (Streamlit).

## Versione corrente

**V45.0** — `Dashboard_pro_V_45_0.py`

## Changelog

### V45.0 (17/08/2026)
- Baseline congelata a partire da V44i.
- Fix: `ModuleNotFoundError` su `xlsxwriter` che bloccava l'export Excel in più sezioni (Rea-Hot, Serafini, Regime, MTF Matrix). Causa: dipendenza mancante in `requirements.txt`. Risolto aggiungendo `xlsxwriter` alle dipendenze.

### V44i e precedenti
- Storico delle versioni 42i → 43a/b/c/d → 44a/b/d/e_final/i mantenuto nel repository come file separati per riferimento e rollback.

### Nota su V50 (rimossa)
- Tentativo di salto diretto a V50 con workflow guidato e pin di `streamlit`/`streamlit-aggrid` (05/08/2026), successivamente rimosso (17/08/2026) per tornare a uno sviluppo incrementale più controllato (44 → 45.x).

## Struttura repository

- `Dashboard_pro_V_XX.py` — versioni della dashboard principale (Streamlit).
- `utils/` — moduli di supporto.
- `data/` — dati statici/cache.
- `assets/` — risorse statiche (immagini, ecc.).
- `.streamlit/` — configurazione Streamlit.
- `.devcontainer/` — configurazione ambiente di sviluppo (Codespaces).
- `requirements.txt` — dipendenze Python.
- `runtime.txt` — versione Python per il deploy.

## Deploy

App distribuita su Streamlit Cloud. Dopo modifiche a `requirements.txt`, verificare sempre in locale con un venv pulito prima del push in produzione.
