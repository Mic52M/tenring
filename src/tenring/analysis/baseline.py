"""Per-session adaptive baseline for view-dependent posture metrics.

Absolute 3D angles from a single (uncalibrated) webcam drift between sessions —
camera position, distance, clothing all shift them by a fixed offset — even
though WITHIN a session they are very stable. So instead of judging against a
stored calibration (which doesn't transfer), we judge each frame against the
shooter's OWN settled aiming posture in THIS session: a robust running median
over recent aiming frames. This measures what is actually reliable and what the
sport is about — deviation from your own assetto, and repeatability.

A calibration profile, if present, is used only as a warm-start prior until
enough live aiming frames have accumulated.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .metrics import Metrics

# View-dependent metrics whose absolute value we don't trust across sessions.
KEYS = ["torso_lean", "shoulder_elevation", "arm_extension",
        "wrist_alignment", "head_tilt"]


class SessionBaseline:
    def __init__(self, window: int = 150, min_frames: int = 25,
                 prior: Optional[dict] = None) -> None:
        self._buf = {k: deque(maxlen=window) for k in KEYS}
        self.min_frames = min_frames
        self.prior = (prior or {}).copy()

    def update(self, m: Metrics) -> None:
        for k in KEYS:
            v = getattr(m, k)
            if v is not None and np.isfinite(v):
                self._buf[k].append(float(v))

    def get(self, key: str) -> Optional[float]:
        b = self._buf.get(key)
        if b is not None and len(b) >= self.min_frames:
            return float(np.median(b))
        if key in self.prior:                 # warm start from calibration
            return float(self.prior[key])
        return None

    def ready(self, key: str) -> bool:
        b = self._buf.get(key)
        return b is not None and len(b) >= self.min_frames

    def as_dict(self) -> dict:
        out = {}
        for k in KEYS:
            v = self.get(k)
            if v is not None:
                out[k] = v
        return out
