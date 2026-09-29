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
```

Concedi al terminale l'accesso alla fotocamera:
**Impostazioni di sistema → Privacy e sicurezza → Fotocamera**.

## Uso

1. **Calibrazione** (una volta): mettiti nella tua migliore posizione di tiro; l'app
   registra la *tua* postura neutra come riferimento personale.
   ```bash
   python -m tenring.calibrate --seconds 6 --mirror
   ```
   Salva `config/profile.yaml`.

2. **Sessione live**: PC alla tua sinistra, inquadratura piena dai piedi alla testa.
   ```bash
   python -m tenring.app --mirror
   ```
   - Pannello a sinistra: semaforo verde/giallo/rosso + cue per ogni metrica.
   - `s` salva un resoconto della sessione; `q` esce (salva automaticamente).
   - I resoconti finiscono in `sessions/`.

Opzioni utili: `--camera N`, `--complexity {0,1,2}` (0 = più veloce, 2 = più preciso).

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

- [ ] Fase 2: **seconda camera** (telefono frontale/posteriore) per quadratura
      spalle e rotazione busto — fusione multi-vista.
- [ ] Rilevamento del colpo più robusto (audio dello scatto / marcatura manuale).
- [ ] Grafici di trend tra sessioni.
- [ ] Confronto con una posa "gold" personale sovrapposta.

## Stato

MVP — camera del PC soltanto. Pensato per uso personale (test su di me), non ancora
per produzione/vendita.

## Licenza

MIT.
