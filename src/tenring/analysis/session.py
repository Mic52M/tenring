"""Session-level aggregation — THEORY §7 (repeatability) & §6 (stability).

Consistency shot-to-shot is the real skill: low variance of posture metrics
across detected shots, plus low average tremor during holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .phases import ShotEvent
from .. import i18n


METRIC_KEYS = ["torso_lean", "shoulder_elevation", "arm_extension",
               "wrist_alignment", "head_tilt", "stance_width", "weight_balance"]


@dataclass
class SessionSummary:
    n_shots: int
    mean_wrist_jitter: float
    mean_sway: float
    consistency: dict           # metric -> std across shots (lower = better)
    means: dict                 # metric -> mean across shots

    def to_text(self, ref: dict, lang: str = "it") -> str:
        head = (f" SESSION REPORT — {self.n_shots} shots" if lang == "en"
                else f" RESOCONTO SESSIONE — {self.n_shots} colpi rilevati")
        lines = ["=" * 52, head, "=" * 52]
        s = ref["stability"]
        jit = self.mean_wrist_jitter
        jflag = "OK" if jit <= s["wrist_jitter_warn"] else (
            "WARN" if jit <= s["wrist_jitter_bad"] else "HIGH")
        tr = "Mean wrist tremor (hold)" if lang == "en" else "Tremore medio polso (hold)"
        sw = "Body sway" if lang == "en" else "Oscillazione corpo (sway)"
        rep = ("Shot-to-shot repeatability (std, lower = better):" if lang == "en"
               else "Ripetibilita' colpo-su-colpo (deviazione std, piu' basso = meglio):")
        mean_w = "mean" if lang == "en" else "media"
        lines.append(f" {tr}: {jit:.4f}  [{jflag}]")
        lines.append(f" {sw}:  {self.mean_sway:.4f}")
        lines.append("-" * 52)
        lines.append(f" {rep}")
        for k in METRIC_KEYS:
            std = self.consistency.get(k, float("nan"))
            mean = self.means.get(k, float("nan"))
            if np.isfinite(std):
                lines.append(f"   {i18n.metric_label(k, lang):<22} {mean_w}={mean:7.2f}  std={std:6.2f}")
        lines.append("=" * 52)
        return "\n".join(lines)


def summarize(shots: list[ShotEvent]) -> Optional[SessionSummary]:
    if not shots:
        return None
    keys = list(METRIC_KEYS)
    consistency, means = {}, {}
    for k in keys:
        vals = [s.metrics_mean.get(k, float("nan")) for s in shots]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            means[k] = float(np.mean(vals))
            consistency[k] = float(np.std(vals)) if len(vals) > 1 else 0.0
        else:
            means[k] = float("nan")
            consistency[k] = float("nan")

    jit = [s.wrist_jitter for s in shots if np.isfinite(s.wrist_jitter)]
    sway = [s.sway for s in shots if np.isfinite(s.sway)]
    return SessionSummary(
        n_shots=len(shots),
        mean_wrist_jitter=float(np.mean(jit)) if jit else float("nan"),
        mean_sway=float(np.mean(sway)) if sway else float("nan"),
        consistency=consistency,
        means=means,
    )
