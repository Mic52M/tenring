"""Session recorder — accumulate per-frame metrics/findings for the report.

Keeps a compact in-memory log of the session so we can build charts and a
verdict afterwards, and can also dump it to JSONL for later re-analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .metrics import Metrics
from .rules import Finding, Status
from .phases import Stability

_METRIC_KEYS = ["torso_lean", "shoulder_elevation", "arm_extension",
                "wrist_alignment", "head_tilt", "stance_width", "weight_balance"]


@dataclass
class SessionRecorder:
    frames: list = field(default_factory=list)
    _t0: Optional[float] = None

    def add(self, t: float, m: Metrics, findings: list[Finding],
            stability: Optional[Stability]) -> None:
        if m.quality < 0.4:
            return  # skip frames where the body isn't reliably tracked
        if self._t0 is None:
            self._t0 = t
        rec = {"t": round(t - self._t0, 3), "quality": round(m.quality, 3)}
        for k in _METRIC_KEYS:
            v = getattr(m, k)
            rec[k] = None if v != v else round(float(v), 3)  # NaN -> None
        rec["status"] = {f.key: f.status.value for f in findings}
        if stability is not None:
            rec["wrist_jitter"] = (None if stability.wrist_jitter != stability.wrist_jitter
                                   else round(stability.wrist_jitter, 4))
            rec["in_hold"] = bool(stability.in_hold)
        self.frames.append(rec)

    def save_jsonl(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self.frames:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.frames)
