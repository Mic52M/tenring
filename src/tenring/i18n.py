"""Tiny i18n layer. Default language is Italian (the author's personal use);
English is available for public artifacts (README screenshots, demos).

Strings are grouped: metric labels, shot-cycle state labels, coaching cues,
and UI strings (HUD + report). Look them up with the helpers at the bottom.
"""
from __future__ import annotations

METRIC = {
    "it": {
        "torso_lean": "Inclinazione busto",
        "shoulder_elevation": "Spalla arma",
        "arm_extension": "Estensione braccio",
        "wrist_alignment": "Polso",
        "head_tilt": "Testa",
        "stance_width": "Apertura piedi",
        "weight_balance": "Bilanciamento",
        "arm_elevation": "Elevazione braccio",
    },
    "en": {
        "torso_lean": "Torso lean",
        "shoulder_elevation": "Shooting shoulder",
        "arm_extension": "Arm extension",
        "wrist_alignment": "Wrist",
        "head_tilt": "Head",
        "stance_width": "Stance width",
        "weight_balance": "Weight balance",
        "arm_elevation": "Arm elevation",
    },
}

STATE = {
    "it": {
        "IDLE": "In attesa (corpo non inquadrato)",
        "READY": "Mettiti in posizione di tiro",
        "AIMING": "In posizione — analisi attiva",
        "HOLD": "HOLD (in mira, fermo)",
        "RELEASE": "Colpo fatto — abbassa / ricarica",
    },
    "en": {
        "IDLE": "Waiting (body not in frame)",
        "READY": "Get into shooting position",
        "AIMING": "In position — analysing",
        "HOLD": "HOLD (on aim, steady)",
        "RELEASE": "Shot taken — lower / reload",
    },
}

CUE = {
    "it": {
        "torso_lean.bad": "Ti stai inclinando troppo all'indietro: raddrizza il busto.",
        "torso_lean.warn": "Leggera inclinazione all'indietro eccessiva.",
        "shoulder_elevation.bad": "Spalla dell'arma alzata e contratta: abbassala e rilassala.",
        "shoulder_elevation.warn": "Spalla un po' tesa, cerca di rilassarla.",
        "shoulder_elevation.drop": "Spalla troppo molle: perdi struttura, ritrova il tono.",
        "arm_extension.bad": "Braccio troppo flesso rispetto al tuo assetto: estendilo verso il bersaglio.",
        "arm_extension.warn": "Estendi meglio il braccio.",
        "wrist_alignment.bad": "Polso fuori dal tuo assetto: bloccalo dritto con l'avambraccio.",
        "wrist_alignment.warn": "Attenzione all'allineamento del polso.",
        "head_tilt.bad": "Testa inclinata rispetto al tuo assetto: tienila eretta, occhi in linea.",
        "head_tilt.warn": "Testa leggermente inclinata.",
        "stance_width.narrow": "Base troppo stretta: allarga i piedi verso la larghezza spalle.",
        "stance_width.wide": "Base troppo larga: irrigidisce, avvicina leggermente i piedi.",
        "stance_width.warn": "Apertura piedi variata rispetto al tuo assetto.",
        "stance_width.bad": "Apertura piedi molto diversa dal tuo assetto: ritrova la base.",
        "weight_balance.bad": "Peso spostato rispetto al tuo assetto: ridistribuisci uniformemente.",
        "weight_balance.warn": "Peso leggermente spostato rispetto al tuo assetto.",
    },
    "en": {
        "torso_lean.bad": "Leaning back too much: straighten your torso.",
        "torso_lean.warn": "Slightly too much backward lean.",
        "shoulder_elevation.bad": "Shooting shoulder raised and tense: lower and relax it.",
        "shoulder_elevation.warn": "Shoulder a little tense, try to relax it.",
        "shoulder_elevation.drop": "Shoulder too loose: you lose structure, regain tone.",
        "arm_extension.bad": "Arm too bent vs your setup: extend it toward the target.",
        "arm_extension.warn": "Extend the arm more.",
        "wrist_alignment.bad": "Wrist off your setup: lock it straight with the forearm.",
        "wrist_alignment.warn": "Watch your wrist alignment.",
        "head_tilt.bad": "Head tilted vs your setup: keep it upright, eyes on line.",
        "head_tilt.warn": "Head slightly tilted.",
        "stance_width.narrow": "Stance too narrow: widen your feet toward shoulder width.",
        "stance_width.wide": "Stance too wide: it stiffens you, bring your feet in a little.",
        "stance_width.warn": "Stance width drifted from your setup.",
        "stance_width.bad": "Stance width far from your setup: find your base again.",
        "weight_balance.bad": "Weight shifted vs your setup: redistribute it evenly.",
        "weight_balance.warn": "Weight slightly shifted vs your setup.",
    },
}

UI = {
    "it": {
        "report_title": "resoconto sessione",
        "subtitle": "Pistola ad aria compressa 10 m",
        "score_sub": "punteggio 0–100 (ripetibilità colpo-su-colpo + stabilità)",
        "stat_shots": "colpi rilevati",
        "stat_frames": "frame in mira",
        "stat_tremor": "tremore medio (hold)",
        "overview": "Panoramica",
        "shot": "Colpo",
        "work_on": "Su cosa lavorare (ripetibilità)",
        "good_rep": "Ottima ripetibilità colpo-su-colpo",
        "low_rep": "{name} — poco ripetibile: varia di ±{std} tra i colpi",
        "need_shots": "Servono almeno {n} colpi per l'analisi di ripetibilità (ne ho rilevati {have}). Esegui più cicli: alza il braccio, mira, spara, abbassa.",
        "not_eval_h": "Non valutato (fuori inquadratura)",
        "not_eval_p": "Non erano ben inquadrate, quindi non giudicate: {items}.",
        "charts": "Grafici di sessione",
        "rep_table": "Ripetibilità colpo-su-colpo",
        "th_metric": "Metrica", "th_mean": "Media", "th_std": "Dev. std (↓ meglio)",
        "th_value": "Valore",
        "no_shots": "Nessun colpo rilevato",
        "verdict_hi": "Molto costante",
        "verdict_mid": "Costante, con margini",
        "verdict_lo": "Ripetibilità da migliorare",
        "verdict_few": "Servono almeno {n} colpi per il verdetto",
        "kpi_total": "tempo totale colpo", "kpi_rise": "salita + mira",
        "kpi_hold": "durata hold", "kpi_descent": "discesa",
        "kpi_tremor": "tremore in hold", "kpi_peak": "picco elevazione braccio",
        "kpi_settle": "elevazione in hold", "kpi_above": "salita sopra il bersaglio",
        "phases": "Fasi del colpo", "ph_rise": "salita+mira", "ph_hold": "hold",
        "ph_descent": "discesa",
        "steady_posture": "Postura all'istante più fermo",
        "ch_shots": "Per colpo: tremore in hold e durata",
        "ch_tremor": "tremore", "ch_hold_s": "hold (s)", "ch_shot_n": "colpo #",
        "ch_rep": "Ripetibilità colpo-su-colpo (dentro la fascia verde = costante)",
        "ch_rep_y": "scarto dal tuo assetto\n(unità di consistenza)",
        "ch_timeline": "Timeline postura (fasi di mira)",
        "ch_time_x": "tempo (s) — solo fasi in mira",
        "ch_stab": "Stabilità in HOLD (tremore, piano frontale) — più basso è meglio",
        "ch_stab_x": "tempo (s)", "ch_stab_y": "tremore polso",
        "ch_dist": "Distribuzione difetti per metrica", "ch_dist_x": "% del tempo",
        "lg_ok": "OK", "lg_warn": "Attenzione", "lg_bad": "Errore",
        "footer": "Generato da tenring · valutazione sulla TUA postura di sessione (deviazione dal tuo assetto), con guardrail dalla teoria del tiro 10 m. Solo fasi in mira; la discesa post-sparo è esclusa. Tremore nel piano frontale. Non misura la mira/mirino: analizza il corpo.",
        "hud_arm": "Braccio arma", "hud_right": "destro", "hud_left": "sinistro",
        "hud_paused": "Analisi in pausa", "hud_inpos": "In posizione",
        "hud_waiting": "In attesa", "hud_tremor": "Tremore", "hud_shots": "Colpi",
        "hud_help": "[Q] esci     [S] salva     {fps} fps",
        "hud_ok": "Postura corretta", "hud_ok_sub": "sei in linea",
    },
    "en": {
        "report_title": "session report",
        "subtitle": "10 m air pistol",
        "score_sub": "score 0–100 (shot-to-shot repeatability + stability)",
        "stat_shots": "shots detected",
        "stat_frames": "frames on aim",
        "stat_tremor": "mean tremor (hold)",
        "overview": "Overview",
        "shot": "Shot",
        "work_on": "What to work on (repeatability)",
        "good_rep": "Excellent shot-to-shot repeatability",
        "low_rep": "{name} — low repeatability: varies by ±{std} across shots",
        "need_shots": "At least {n} shots are needed for repeatability analysis (detected {have}). Run more cycles: raise the arm, aim, fire, lower.",
        "not_eval_h": "Not evaluated (out of frame)",
        "not_eval_p": "These weren't well framed, so they weren't judged: {items}.",
        "charts": "Session charts",
        "rep_table": "Shot-to-shot repeatability",
        "th_metric": "Metric", "th_mean": "Mean", "th_std": "Std dev (↓ better)",
        "th_value": "Value",
        "no_shots": "No shot detected",
        "verdict_hi": "Very consistent",
        "verdict_mid": "Consistent, room to improve",
        "verdict_lo": "Repeatability needs work",
        "verdict_few": "At least {n} shots needed for a verdict",
        "kpi_total": "total shot time", "kpi_rise": "raise + aim",
        "kpi_hold": "hold duration", "kpi_descent": "descent",
        "kpi_tremor": "tremor in hold", "kpi_peak": "peak arm elevation",
        "kpi_settle": "elevation on aim", "kpi_above": "raised above target",
        "phases": "Shot phases", "ph_rise": "raise+aim", "ph_hold": "hold",
        "ph_descent": "descent",
        "steady_posture": "Posture at the steadiest instant",
        "ch_shots": "Per shot: tremor in hold & duration",
        "ch_tremor": "tremor", "ch_hold_s": "hold (s)", "ch_shot_n": "shot #",
        "ch_rep": "Shot-to-shot repeatability (inside green band = consistent)",
        "ch_rep_y": "deviation from your setup\n(consistency units)",
        "ch_timeline": "Posture timeline (aiming phases)",
        "ch_time_x": "time (s) — aiming phases only",
        "ch_stab": "Stability in HOLD (tremor, frontal plane) — lower is better",
        "ch_stab_x": "time (s)", "ch_stab_y": "wrist tremor",
        "ch_dist": "Fault distribution by metric", "ch_dist_x": "% of time",
        "lg_ok": "OK", "lg_warn": "Warning", "lg_bad": "Error",
        "footer": "Generated by tenring · judged against YOUR session posture (deviation from your own setup), with guardrails from 10 m shooting theory. Aiming phases only; the post-shot descent is excluded. Tremor in the frontal plane. It does not measure the sights/aim: it analyses the body.",
        "hud_arm": "Shooting arm", "hud_right": "right", "hud_left": "left",
        "hud_paused": "Analysis paused", "hud_inpos": "In position",
        "hud_waiting": "Waiting", "hud_tremor": "Tremor", "hud_shots": "Shots",
        "hud_help": "[Q] quit     [S] save     {fps} fps",
        "hud_ok": "Posture OK", "hud_ok_sub": "you're on line",
    },
}


def _d(table: dict, lang: str) -> dict:
    return table.get(lang, table["it"])


def metric_label(key: str, lang: str = "it") -> str:
    return _d(METRIC, lang).get(key, key)


def state_label(state_value: str, lang: str = "it") -> str:
    return _d(STATE, lang).get(state_value, state_value)


def cue(key: str, lang: str = "it") -> str:
    return _d(CUE, lang).get(key, "")


def ui(key: str, lang: str = "it", **kw) -> str:
    s = _d(UI, lang).get(key, key)
    return s.format(**kw) if kw else s
