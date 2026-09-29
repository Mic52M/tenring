# Teoria del tiro — Pistola ad aria compressa 10 m (ISSF)

> Questo documento è la **base tecnica** dell'analizzatore. Ogni metrica e ogni regola
> nel codice (`config/reference.yaml`, `src/tenring/analysis/`) fa riferimento a un
> punto di questo file. L'obiettivo non è "guardare il corpo" in astratto, ma valutarlo
> secondo i principi consolidati del tiro di precisione a 10 m ad **aria compressa**
> (disciplina statica, una mano, arma **non a fuoco**).

## 0. Contesto della disciplina

- **ISSF 10 m Air Pistol**: da posizione eretta, **non appoggiata**, arma tenuta e
  azionata con **una sola mano**. Bersaglio a 10 m, 10-ring del diametro di 11.5 mm.
- La disciplina è **quasi statica**: il tiratore resta praticamente immobile mentre
  esegue mira e scatto. Questo rende la stima della posa (pose estimation) un problema
  molto più semplice e affidabile che in sport dinamici (golf, corsa).
- Il segreto della prestazione **non è la posizione perfetta di un singolo colpo**, ma
  la **ripetibilità** colpo-su-colpo e la **stabilità** durante la fase di mira. In
  letteratura questo è ciò che separa i tiratori elite dai novizi (minore oscillazione
  del centro di pressione, minore tremore fisiologico).

## 1. I fondamentali (in ordine di priorità)

Secondo la manualistica ISSF e il coaching consolidato, si stabilizzano nell'ordine:

1. **Posizione stabile** (stance) — la base.
2. **Impugnatura costante** (grip) — identica ad ogni colpo.
3. **Rilascio pulito del grilletto** (trigger release) — non rilevabile dalla camera,
   ma le sue conseguenze (strappo) sì.

L'app copre soprattutto **(1)** e in parte **(2)**, più la **stabilità** globale.

## 2. Posizione del corpo (stance) — piano che la camera laterale vede bene

- **Piedi**: circa **larghezza spalle**, orientati ~**45°** rispetto alla linea di tiro
  (per il destrimane, ruotati verso sinistra). Larghezza tra i talloni indicativa
  ~30 cm (12"). Una base **troppo stretta** riduce la stabilità laterale; una
  **troppo larga** irrigidisce.
- **Peso**: distribuito in modo **uniforme** sui due piedi, leggermente verso gli
  avampiedi. Il **baricentro** deve restare **davanti all'anca**, stabile.
- **Bacino/busto**: estendendo il braccio armato, il baricentro si sposta in avanti e
  il corpo compensa con una **leggera inclinazione all'indietro** dell'anca. Questa
  inclinazione deve essere **minima e naturale**.
  - ⚠️ **Errore classico**: **inclinarsi troppo all'indietro** per contrappesare la
    pistola → sposta il baricentro, carica la zona lombare, riduce la ripetibilità.
    → `metric: torso_lean_back` con soglia di allarme.
- **Principio di economia**: la posizione migliore è quella che usa **meno energia
  muscolare** possibile e non affatica un gruppo muscolare più degli altri. Tensione =
  tremore e instabilità.

## 3. Spalla e braccio armato

- La **spalla del braccio armato** deve restare **bassa e rilassata**, così che il
  braccio si estenda in modo naturale. Tensione alla spalla ("spalla dura/alzata") =
  minore stabilità e affaticamento precoce.
  - ⚠️ Errore: **spalla alzata/contratta** (shrug) — → `metric: shoulder_elevation`.
  - ⚠️ Errore opposto: spalla **troppo molle/cadente** con perdita di struttura.
- **Braccio**: **esteso** verso il bersaglio, **fermo ma non ipercontratto/bloccato**.
  Il gomito né iperesteso né flesso in modo instabile. → `metric: arm_extension`.

## 4. Polso e mano

- Il **polso** della mano armata deve restare **dritto e bloccato** (allineato con
  l'avambraccio), identico ad ogni colpo. Un polso che flette introduce errore
  verticale. → `metric: wrist_alignment`.
- **Grip**: fermo ma non strizzato. Troppo stretto → tremore; troppo lasco → movimento
  e pressione sulla spalla. (Difficile da vedere in camera; inferito indirettamente dal
  tremore della mano.)

## 5. Testa e assetto

- La **testa** resta il più possibile **eretta e ferma**, gli occhi allineati alla
  mira. Evitare di **inclinare/piegare la testa** verso l'arma o di ruotarla in modo
  innaturale (tensione al collo). → `metric: head_tilt`.

## 6. Stabilità e tremore (fase di mira / hold)

- Durante la fase di **hold** (arma sollevata e ferma prima dello scatto), il corpo
  oscilla: **deriva lenta** (sway posturale, soprattutto laterale) e **tremore
  fisiologico** più veloce (soprattutto verticale, da spalla/polso).
- Metrica di stabilità: **jitter** (deviazione standard della posizione dei keypoint
  chiave — polso mano armata, naso) nella finestra di hold. Minore jitter = migliore.
  → `analysis/phases.py`.
- L'**oscillazione del centro di pressione** è sempre più bassa negli elite. Proxy dalla
  camera: oscillazione del midpoint anche/spalle. → `metric: body_sway`.

## 7. Ripetibilità colpo-su-colpo

- Marcare la posizione (piedi, assetto) e riprodurla identica ad ogni colpo.
- Metrica: **varianza** delle metriche posturali **tra i colpi** rilevati in una
  sessione. Bassa varianza = posizione consolidata. → `analysis/session.py`.

## 8. Cosa la camera **non** misura (limiti onesti)

- **Punto di mira / rosa di oscillazione del mirino** e **strappo di grilletto**: è il
  dominio dei sistemi ottici tipo SCATT. La CV vede il **corpo**, non la canna/mirino.
- **Camera singola laterale** → ottima sul **piano sagittale** (inclinazione avanti/
  indietro, spalla, testa, estensione braccio, apertura piedi). Debole su **quadratura
  spalle / rotazione busto** (ambiguità di profondità): richiede una seconda camera
  (frontale o posteriore) in una fase 2. MediaPipe fornisce comunque `world_landmarks`
  in 3D (metri, relativi all'anca) che attenuano parzialmente il limite.

## 9. Handedness e geometria camera

- Setup MVP: **PC/camera a sinistra** del tiratore **destrimane**; il tiratore spara con
  la destra estendendo il braccio verso il bersaglio davanti a sé.
- La camera inquadra quindi prevalentemente il **lato sinistro** del corpo; il braccio
  armato (destro) è sul lato lontano e può essere parzialmente occluso. Usiamo i
  `visibility` score dei landmark e, quando serve, i landmark del lato visibile +
  simmetria. Tutto configurabile (`handedness`, `camera_side`).

---

### Fonti (coaching + letteratura, disciplina 10 m)

- ISSF Pistol Rules Book (regolamento 10 m Air Pistol).
- Tandfonline — *A preliminary study of stability in elite and novice 10 meter air
  pistol shooters* (centro di pressione, stabilità elite vs novizi).
- PubMed 16446677 — *Characterization of arm-gun movement during air pistol aiming
  phase* (deriva lenta laterale vs tremore verticale spalla/polso).
- Foresight Shooting / Sport Quantum / Airgun World — guide tecniche stance, baricentro,
  inclinazione all'indietro come errore, spalla rilassata, polso dritto.
- ResearchGate 387210925 — analisi CV del pattern colpi 10 m air pistol.
