"""Hold-phase detection and stability (tremor/sway) measurement — THEORY §6.

The shot itself (pellet release) is not visible to the camera, so we detect the
*hold*: the armed arm is raised/extended and the wrist becomes still for a
sustained window. Stability during that window (wrist jitter, body sway) is the
metric that separates elite from novice shooters in the literature.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import Metrics


@dataclass
class Stability:
    in_hold: bool = False
    wrist_jitter: float = float("nan")   # std of wrist pos over window / shoulder width
    sway: float = float("nan")           # std of body center over window / shoulder width
    hold_frames: int = 0


@dataclass
class ShotEvent:
    """A completed hold, i.e. one 'shot' worth of posture."""
    t_end: float
    wrist_jitter: float
    sway: float
    duration: float
    metrics_mean: dict = field(default_factory=dict)


class HoldTracker:
    def __init__(self, ref: dict, fps: float = 30.0) -> None:
        p = ref["phases"]
        s = ref["stability"]
        self.raise_min = p["raise_min_arm_deg"]
        self.still_jitter = p["hold_still_jitter"]
        self.hold_min_frames = p["hold_min_frames"]
        self.gap = p["settle_release_gap_sec"]
        self.window = max(1, int(s["hold_window_sec"] * fps))

        self._wrist: deque = deque(maxlen=self.window)
        self._center: deque = deque(maxlen=self.window)
        self._sw: deque = deque(maxlen=self.window)
        self._metric_buf: deque = deque(maxlen=self.window)

        self._hold_frames = 0
        self._was_holding = False
        self._last_hold_t = 0.0
        self.shots: list[ShotEvent] = []

    def update(self, m: Metrics, t: float) -> Stability:
        st = Stability()
        if m.armed_wrist is None or m.body_center is None or not np.isfinite(m.shoulder_width):
            self._reset_hold()
            return st

        self._wrist.append(np.asarray(m.armed_wrist, dtype=float))
        self._center.append(np.asarray(m.body_center, dtype=float))
        self._sw.append(m.shoulder_width)
        self._metric_buf.append(m)

        if len(self._wrist) < max(3, self.window // 2):
            return st  # not enough data yet

        sw = float(np.median(self._sw)) or 1.0
        wrist_arr = np.stack(self._wrist)
        center_arr = np.stack(self._center)
        st.wrist_jitter = float(np.linalg.norm(np.std(wrist_arr, axis=0))) / sw
        st.sway = float(np.linalg.norm(np.std(center_arr, axis=0))) / sw

        raised = np.isfinite(m.arm_extension) and m.arm_extension >= self.raise_min
        still = st.wrist_jitter <= self.still_jitter

        if raised and still:
            self._hold_frames += 1
        else:
            # transition out of a hold -> register a shot if it was long enough
            if self._was_holding and self._hold_frames >= self.hold_min_frames:
                self._register_shot(t, st)
            self._hold_frames = 0

        st.hold_frames = self._hold_frames
        st.in_hold = self._hold_frames >= self.hold_min_frames
        self._was_holding = st.in_hold
        if st.in_hold:
            self._last_hold_t = t
        return st

    def _register_shot(self, t: float, st: Stability) -> None:
        keys = ["torso_lean", "shoulder_elevation", "arm_extension",
                "wrist_alignment", "head_tilt", "stance_width", "weight_balance"]
        means = {}
        for k in keys:
            vals = [getattr(mm, k) for mm in self._metric_buf
                    if np.isfinite(getattr(mm, k))]
            means[k] = float(np.mean(vals)) if vals else float("nan")
        self.shots.append(ShotEvent(
            t_end=t,
            wrist_jitter=st.wrist_jitter,
            sway=st.sway,
            duration=self._hold_frames / max(1, self.window) * self.window,  # frames
            metrics_mean=means,
        ))

    def _reset_hold(self) -> None:
        self._hold_frames = 0
        self._was_holding = False
