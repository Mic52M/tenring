"""Rule engine: turn raw metrics into coaching findings (green/yellow/red + cue).

Thresholds come from config/reference.yaml (literature-grounded). When a personal
profile is present (from calibration), metrics that are camera/pose dependent
(torso lean) are evaluated as a DEVIATION from the shooter's own calibrated
neutral, per the user's goal of an app tuned to himself while keeping the
literature guardrails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .metrics import Metrics


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    BAD = "bad"
    NA = "na"      # not measurable this frame


@dataclass
class Finding:
    key: str
    label: str
    value: float
    status: Status
    cue: str = ""

    @property
    def color(self) -> tuple:  # BGR for OpenCV
        return {
            Status.OK: (80, 200, 80),
            Status.WARN: (0, 200, 255),
            Status.BAD: (60, 60, 235),
            Status.NA: (150, 150, 150),
        }[self.status]


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class RuleEngine:
    def __init__(self, ref: dict, profile: Optional[dict] = None) -> None:
        self.ref = ref
        self.profile = profile or {}

    # -- helpers ---------------------------------------------------------------
    def _neutral(self, key: str, default: float) -> float:
        return float(self.profile.get("neutral", {}).get(key, default))

    @staticmethod
    def _band(dev: float, warn: float, bad: float, cue_warn: str, cue_bad: str):
        if dev >= bad:
            return Status.BAD, cue_bad
        if dev >= warn:
            return Status.WARN, cue_warn
        return Status.OK, ""

    # -- main ------------------------------------------------------------------
    def evaluate(self, m: Metrics) -> list[Finding]:
        f: list[Finding] = []
        r = self.ref

        # 1. Torso lean — deviation from calibrated neutral (fallback: ideal).
        c = r["torso_lean_back"]
        if _isnan(m.torso_lean):
            f.append(Finding("torso_lean", "Inclinazione busto", float("nan"), Status.NA))
        else:
            neutral = self._neutral("torso_lean", c["ideal_deg"])
            dev = abs(m.torso_lean - neutral)
            st, cue = self._band(dev, c["warn_deg"], c["bad_deg"], c["cue_warn"], c["cue_bad"])
            f.append(Finding("torso_lean", "Inclinazione busto", m.torso_lean, st, cue))

        # 2. Shoulder elevation — shrug (positive) or drop (negative).
        c = r["shoulder_elevation"]
        if _isnan(m.shoulder_elevation):
            f.append(Finding("shoulder_elevation", "Spalla arma", float("nan"), Status.NA))
        else:
            v = m.shoulder_elevation
            if v <= c["drop_warn_ratio"]:
                st, cue = Status.WARN, c["cue_drop"]
            else:
                st, cue = self._band(v, c["warn_ratio"], c["bad_ratio"], c["cue_warn"], c["cue_bad"])
            f.append(Finding("shoulder_elevation", "Spalla arma", v, st, cue))

        # 3. Arm extension — too flexed (low angle) is the error.
        c = r["arm_extension"]
        if _isnan(m.arm_extension):
            f.append(Finding("arm_extension", "Estensione braccio", float("nan"), Status.NA))
        else:
            v = m.arm_extension
            if v <= c["bad_deg"]:
                st, cue = Status.BAD, c["cue_bad"]
            elif v <= c["warn_deg"]:
                st, cue = Status.WARN, c["cue_warn"]
            else:
                st, cue = Status.OK, ""
            f.append(Finding("arm_extension", "Estensione braccio", v, st, cue))

        # 4. Wrist alignment — deviation below straight line.
        c = r["wrist_alignment"]
        if _isnan(m.wrist_alignment):
            f.append(Finding("wrist_alignment", "Polso", float("nan"), Status.NA))
        else:
            v = m.wrist_alignment
            if v <= c["bad_deg"]:
                st, cue = Status.BAD, c["cue_bad"]
            elif v <= c["warn_deg"]:
                st, cue = Status.WARN, c["cue_warn"]
            else:
                st, cue = Status.OK, ""
            f.append(Finding("wrist_alignment", "Polso", v, st, cue))

        # 5. Head tilt — absolute roll from horizontal.
        c = r["head_tilt"]
        if _isnan(m.head_tilt):
            f.append(Finding("head_tilt", "Testa", float("nan"), Status.NA))
        else:
            st, cue = self._band(abs(m.head_tilt), c["warn_deg"], c["bad_deg"],
                                 c["cue_warn"], c["cue_bad"])
            f.append(Finding("head_tilt", "Testa", m.head_tilt, st, cue))

        # 6. Stance width — two-sided (narrow / wide).
        c = r["stance_width"]
        if _isnan(m.stance_width):
            f.append(Finding("stance_width", "Apertura piedi", float("nan"), Status.NA))
        else:
            v = m.stance_width
            if v <= c["narrow_bad_ratio"]:
                st, cue = Status.BAD, c["cue_narrow"]
            elif v <= c["narrow_warn_ratio"]:
                st, cue = Status.WARN, c["cue_narrow"]
            elif v >= c["wide_bad_ratio"]:
                st, cue = Status.BAD, c["cue_wide"]
            elif v >= c["wide_warn_ratio"]:
                st, cue = Status.WARN, c["cue_wide"]
            else:
                st, cue = Status.OK, ""
            f.append(Finding("stance_width", "Apertura piedi", v, st, cue))

        # 7. Weight balance — absolute offset.
        c = r["weight_balance"]
        if _isnan(m.weight_balance):
            f.append(Finding("weight_balance", "Bilanciamento", float("nan"), Status.NA))
        else:
            st, cue = self._band(abs(m.weight_balance), c["warn_offset"], c["bad_offset"],
                                 c["cue_warn"], c["cue_bad"])
            f.append(Finding("weight_balance", "Bilanciamento", m.weight_balance, st, cue))

        return f
