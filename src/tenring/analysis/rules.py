"""Rule engine: turn raw metrics into coaching findings (green/yellow/red + cue).

Thresholds come from config/reference.yaml (literature-grounded). View-dependent
metrics are judged as a DEVIATION from the shooter's own reference (session
baseline, or a stored calibration), not from absolute angles. Labels and cues are
language-aware via i18n (default Italian).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .metrics import Metrics
from .. import i18n


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
    def __init__(self, ref: dict, profile: Optional[dict] = None,
                 lang: str = "it") -> None:
        self.ref = ref
        self.profile = profile or {}
        self.lang = lang
        self._active_neutral: dict = {}

    # -- helpers ---------------------------------------------------------------
    def _neutral(self, key: str, default: float) -> float:
        return float(self._active_neutral.get(key, default))

    def _has_neutral(self, key: str) -> bool:
        return key in self._active_neutral

    def _lbl(self, key: str) -> str:
        return i18n.metric_label(key, self.lang)

    def _cue(self, key: str) -> str:
        return i18n.cue(key, self.lang)

    def _F(self, key: str, value, status: Status, cue_key: str = "") -> Finding:
        return Finding(key, self._lbl(key), value, status,
                       self._cue(cue_key) if cue_key else "")

    def _dev_bands(self, key: str, value: float, c: dict) -> Finding:
        """Judge |value - neutral| against dev_warn/dev_bad (symmetric)."""
        dev = abs(value - self._neutral(key, 0.0))
        if dev >= c["dev_bad"]:
            return self._F(key, value, Status.BAD, f"{key}.bad")
        if dev >= c["dev_warn"]:
            return self._F(key, value, Status.WARN, f"{key}.warn")
        return self._F(key, value, Status.OK)

    # -- main ------------------------------------------------------------------
    def evaluate(self, m: Metrics, neutral: Optional[dict] = None) -> list[Finding]:
        self._active_neutral = neutral if neutral is not None \
            else self.profile.get("neutral", {})
        f: list[Finding] = []
        r = self.ref

        # 1. Torso lean — deviation from reference (fallback: ideal).
        c = r["torso_lean_back"]
        if _isnan(m.torso_lean):
            f.append(self._F("torso_lean", float("nan"), Status.NA))
        else:
            dev = abs(m.torso_lean - self._neutral("torso_lean", c["ideal_deg"]))
            if dev >= c["bad_deg"]:
                f.append(self._F("torso_lean", m.torso_lean, Status.BAD, "torso_lean.bad"))
            elif dev >= c["warn_deg"]:
                f.append(self._F("torso_lean", m.torso_lean, Status.WARN, "torso_lean.warn"))
            else:
                f.append(self._F("torso_lean", m.torso_lean, Status.OK))

        # 2. Shoulder elevation — shrug (positive) or drop (negative).
        c = r["shoulder_elevation"]
        if _isnan(m.shoulder_elevation):
            f.append(self._F("shoulder_elevation", float("nan"), Status.NA))
        else:
            v = m.shoulder_elevation
            if self._has_neutral("shoulder_elevation"):
                dev = v - self._neutral("shoulder_elevation", 0.0)
                if dev >= c["dev_bad"]:
                    f.append(self._F("shoulder_elevation", v, Status.BAD, "shoulder_elevation.bad"))
                elif dev >= c["dev_warn"]:
                    f.append(self._F("shoulder_elevation", v, Status.WARN, "shoulder_elevation.warn"))
                elif dev <= -c["dev_warn"]:
                    f.append(self._F("shoulder_elevation", v, Status.WARN, "shoulder_elevation.drop"))
                else:
                    f.append(self._F("shoulder_elevation", v, Status.OK))
            elif v <= c["drop_warn_ratio"]:
                f.append(self._F("shoulder_elevation", v, Status.WARN, "shoulder_elevation.drop"))
            elif v >= c["bad_ratio"]:
                f.append(self._F("shoulder_elevation", v, Status.BAD, "shoulder_elevation.bad"))
            elif v >= c["warn_ratio"]:
                f.append(self._F("shoulder_elevation", v, Status.WARN, "shoulder_elevation.warn"))
            else:
                f.append(self._F("shoulder_elevation", v, Status.OK))

        # 3. Arm extension — deviation from reference, else absolute (too flexed).
        c = r["arm_extension"]
        if _isnan(m.arm_extension):
            f.append(self._F("arm_extension", float("nan"), Status.NA))
        elif self._has_neutral("arm_extension"):
            f.append(self._dev_bands("arm_extension", m.arm_extension, c))
        else:
            v = m.arm_extension
            if v <= c["bad_deg"]:
                f.append(self._F("arm_extension", v, Status.BAD, "arm_extension.bad"))
            elif v <= c["warn_deg"]:
                f.append(self._F("arm_extension", v, Status.WARN, "arm_extension.warn"))
            else:
                f.append(self._F("arm_extension", v, Status.OK))

        # 4. Wrist alignment — deviation from reference, else absolute.
        c = r["wrist_alignment"]
        if _isnan(m.wrist_alignment):
            f.append(self._F("wrist_alignment", float("nan"), Status.NA))
        elif self._has_neutral("wrist_alignment"):
            f.append(self._dev_bands("wrist_alignment", m.wrist_alignment, c))
        else:
            v = m.wrist_alignment
            if v <= c["bad_deg"]:
                f.append(self._F("wrist_alignment", v, Status.BAD, "wrist_alignment.bad"))
            elif v <= c["warn_deg"]:
                f.append(self._F("wrist_alignment", v, Status.WARN, "wrist_alignment.warn"))
            else:
                f.append(self._F("wrist_alignment", v, Status.OK))

        # 5. Head tilt — deviation from reference, else absolute roll.
        c = r["head_tilt"]
        if _isnan(m.head_tilt):
            f.append(self._F("head_tilt", float("nan"), Status.NA))
        elif self._has_neutral("head_tilt"):
            f.append(self._dev_bands("head_tilt", m.head_tilt, c))
        else:
            dev = abs(m.head_tilt)
            if dev >= c["bad_deg"]:
                f.append(self._F("head_tilt", m.head_tilt, Status.BAD, "head_tilt.bad"))
            elif dev >= c["warn_deg"]:
                f.append(self._F("head_tilt", m.head_tilt, Status.WARN, "head_tilt.warn"))
            else:
                f.append(self._F("head_tilt", m.head_tilt, Status.OK))

        # 6. Stance width — deviation from reference, else two-sided absolute.
        c = r["stance_width"]
        if _isnan(m.stance_width):
            f.append(self._F("stance_width", float("nan"), Status.NA))
        elif self._has_neutral("stance_width"):
            f.append(self._dev_bands("stance_width", m.stance_width, c))
        else:
            v = m.stance_width
            if v <= c["narrow_bad_ratio"]:
                f.append(self._F("stance_width", v, Status.BAD, "stance_width.narrow"))
            elif v <= c["narrow_warn_ratio"]:
                f.append(self._F("stance_width", v, Status.WARN, "stance_width.narrow"))
            elif v >= c["wide_bad_ratio"]:
                f.append(self._F("stance_width", v, Status.BAD, "stance_width.wide"))
            elif v >= c["wide_warn_ratio"]:
                f.append(self._F("stance_width", v, Status.WARN, "stance_width.wide"))
            else:
                f.append(self._F("stance_width", v, Status.OK))

        # 7. Weight balance — deviation from reference, else absolute offset.
        c = r["weight_balance"]
        if _isnan(m.weight_balance):
            f.append(self._F("weight_balance", float("nan"), Status.NA))
        elif self._has_neutral("weight_balance"):
            f.append(self._dev_bands("weight_balance", m.weight_balance, c))
        else:
            dev = abs(m.weight_balance)
            if dev >= c["bad_offset"]:
                f.append(self._F("weight_balance", m.weight_balance, Status.BAD, "weight_balance.bad"))
            elif dev >= c["warn_offset"]:
                f.append(self._F("weight_balance", m.weight_balance, Status.WARN, "weight_balance.warn"))
            else:
                f.append(self._F("weight_balance", m.weight_balance, Status.OK))

        return f
