# tenring 🎯

**Analisi posturale via computer vision per il tiro a 10 m con pistola ad aria
compressa (ISSF).** Una singola camera (il tuo PC, alla tua sinistra) osserva il
corpo mentre tieni la mira e dà feedback in tempo reale sui difetti di postura —
inclinazione del busto, spalla contratta, testa storta, estensione del braccio,
polso, apertura dei piedi, bilanciamento — più **stabilità** (tremore in fase di
hold) e **ripetibilità** colpo-su-colpo.

> Disciplina statica, una mano, arma **non a fuoco**. Gira in tempo reale su
> MacBook Air **M4**, solo CPU/Neural Engine — **nessuna GPU esterna**.

Non sostituisce SCATT (che traccia la *mira*): tenring guarda il **corpo**, che è
la parte scoperta. Vedi [`docs/THEORY.md`](docs/THEORY.md) per la base tecnica del
tiro a 10 m su cui poggia ogni metrica.

## Cosa misura (ancorato alla letteratura)

| Metrica | Difetto tipico | Rif. teoria |
|---|---|---|
| Inclinazione busto | inclinarsi troppo all'indietro | §2 |
| Spalla arma | spalla alzata/contratta o troppo molle | §3 |
| Estensione braccio | braccio troppo flesso / bloccato | §3 |
| Polso | polso non allineato all'avambraccio | §4 |
| Testa | testa inclinata / ruotata | §5 |
| Apertura piedi | base troppo stretta/larga | §2 |
| Bilanciamento | peso sbilanciato su un piede | §2 |
| Tremore (hold) | oscillazione in fase di mira | §6 |
| Ripetibilità | varianza posturale tra i colpi | §7 |

## Setup (macOS, Apple Silicon)

```bash
cd ~/Documents/tenring

# Ambiente consigliato: Python 3.11/3.12 (MediaPipe non sempre pronto su 3.13)
conda create -n tenring python=3.12 -y
conda activate tenring
pip install -r requirements.txt
pip install -e .          # installa il package `tenring` (necessario per `python -m tenring.*`)
```

> In alternativa a `pip install -e .`, puoi lanciare con `PYTHONPATH=src python -m tenring.app`.

Concedi al terminale l'accesso alla fotocamera:
**Impostazioni di sistema → Privacy e sicurezza → Fotocamera**.

## Uso — semplice

Un comando. Sceglie la webcam del Mac da solo (salta l'iPhone/Continuity Camera),
apre la finestra, premi **Q** per uscire.

```bash
tenring
```

- **Analizza solo quando sei in posizione**: una macchina a stati riconosce il ciclo
  di tiro (in attesa → in posizione → mira/HOLD → colpo) e valuta/registra **solo**
  in fase di mira. Se ti muovi, bevi, o si vede solo il viso, l'analisi va in pausa
  (niente più "postura perfetta" senza senso).
- **Banner grande in alto**: la correzione più importante in quel momento (leggibile
  mentre miri); pannello a sinistra: stato, braccio rilevato, semaforo + cue.
- `q` (o ESC) esce e **apre da solo un report HTML** con verdetto, difetti su cui
  lavorare, grafici (timeline, distribuzione difetti, stabilità) e ripetibilità
  colpo-su-colpo; `s` genera un report al volo.
- Tutto finisce in `sessions/` (`.html` report, `.jsonl` dati per-frame, `.txt` sintesi).

**Calibrazione personale** (opzionale, una volta): mettiti nella tua migliore
posizione di tiro; registra la *tua* postura neutra come riferimento.

```bash
tenring-calibrate
```

### Opzioni (se servono)

| Opzione | Cosa fa |
|---|---|
| `tenring --list-cameras` | elenca le camere per scegliere l'indice giusto |
| `tenring --camera 1` | forza una camera specifica |
| `tenring --no-mirror` | disattiva l'effetto specchio (per camera di profilo) |
| `tenring --complexity 2` | modello più preciso (0 = più veloce) |

> In alternativa ai comandi `tenring` / `tenring-calibrate` puoi usare
> `python -m tenring.app` e `python -m tenring.calibrate`.

## Architettura

```
src/tenring/
  pose/        backend pose estimation astratto (+ MediaPipe BlazePose)
  analysis/    geometry · metrics · rules · phases (hold/tremore) · session
  ui/          overlay OpenCV (scheletro + pannello coaching)
  app.py       loop live   ·   calibrate.py  calibrazione personale
config/
  reference.yaml   soglie ancorate alla letteratura (§ in docs/THEORY.md)
  profile.yaml     (generato) la TUA postura neutra
docs/THEORY.md     base tecnica del tiro a 10 m
```

Il backend di pose è dietro un'interfaccia astratta (`pose/base.py`): si può
sostituire MediaPipe con **RTMPose / YOLO-pose / ViTPose** senza toccare l'analisi.

## Test

```bash
pip install pytest
pytest
```

## Roadmap

- [x] Report HTML di sessione con grafici (timeline, distribuzione difetti,
      stabilità, ripetibilità).
- [x] Banner live con la correzione prioritaria.
- [x] Macchina a stati del ciclo di tiro (analizza solo in posizione; colpo =
      alza → hold → abbassa il braccio).
- [x] Auto-rilevamento braccio che spara + gating visibilità (valuta solo ciò
      che è inquadrato).
- [x] Baseline adattivo di sessione: valuta la deviazione dal TUO assetto in
      questa sessione (mediana mobile), non da soglie assolute inaffidabili da
      webcam. Risolve il "100% fuori tolleranza".
- [x] Tracking e grafici **per-colpo** (tremore + durata hold per colpo,
      ripetibilità colpo-su-colpo). Postura del colpo = istante più fermo dell'hold.
- [x] Stato **RELEASE**: la discesa del braccio dopo lo sparo (ricarica) non è
      più analizzata come mira.
- [x] **Tempi per colpo** (salita+mira / hold / discesa / totale) e **elevazione
      del braccio** (picco sopra il bersaglio vs assestamento ~90°).
- [x] **Report interattivo**: lista colpi a sinistra, dettaglio (tempi di fase,
      elevazione, tremore, postura) a destra; "Panoramica" per i grafici di sessione.
- [ ] Fase 2: **seconda camera** (telefono frontale/posteriore) per quadratura
      spalle e rotazione busto — fusione multi-vista.
- [ ] Rilevamento del colpo più robusto (audio dello scatto / marcatura manuale).
- [ ] Grafici di **trend tra sessioni** (dai `.jsonl` già salvati).
- [ ] Confronto con una posa "gold" personale sovrapposta.

## Stato

MVP — camera del PC soltanto. Pensato per uso personale (test su di me), non ancora
per produzione/vendita.

## Licenza

MIT.
