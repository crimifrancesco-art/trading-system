# Trading Scanner PRO v45.05 — Intelligence

## Obiettivo
Aggiungere uno storico persistente delle analisi AI e rendere confrontabili i cambiamenti tra analisi successive dello stesso ticker.

## Funzionalità pianificate

- Storico analisi AI in SQLite.
- Confronto tra analisi precedente e attuale.
- Log dei cambi di regime macro.
- Alert quando un ticker passa da PRO a STRONG o da STRONG a PRO.

## Roadmap successiva

### v45.06 — Macro Advanced
- COT automatico CFTC.
- Posizionamento netto e percentile storico.
- Conferma COT + Macro Regime + Trend Strength.
- Dashboard Macro Confirmation Score.

### v45.07 — Notification Layer
- Web Push.
- Telegram migliorato con digest.
- Email giornaliera.
- Alert solo quando cambiano stato, regime o soglia CSS.

## Criteri di completamento v45.05

- Lo storico AI deve essere persistente tra riavvii dell’app.
- Ogni analisi deve conservare ticker, timestamp, provider, prompt e risposta.
- Il confronto deve evidenziare differenze tra analisi consecutive.
- I cambi di regime devono essere registrati con valore precedente e nuovo valore.
- Gli alert di transizione PRO/STRONG devono evitare duplicati per lo stesso evento.

## Verifiche

```bash
python -m py_compile Dashboard_pro_V_45_04.py
```
