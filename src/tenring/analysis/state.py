"""Shooting-cycle state machine — the 'intelligence' about WHEN to analyse.

The app must not average posture over moments the shooter isn't actually aiming
(walking in, drinking water, only the face in frame, reloading). This classifies
each frame into a state and only treats AIMING/HOLD as analysable. A shot is
detected as the natural cycle: raise the arm -> steady hold -> lower the arm.

States:
    IDLE   : no reliable body / torso not in frame          -> not analysed
    READY  : person present & torso framed, but not aiming   -> not analysed
    AIMING : armed arm raised to ~shoulder height & extended -> ANALYSED
    HOLD   : AIMING and the wrist is still                    -> ANALYSED

Replaces the old HoldTracker: it owns both the stability (jitter/sway) buffers
and the shot detection, now gated on real arm raise/lower transitions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .metrics import Metrics
from .phases import ShotEvent


class State(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    AIMING = "AIMING"
    HOLD = "HOLD"
    RELEASE = "RELEASE"   # post-shot: arm coming down to reload -> NOT analysed


_LABEL = {
    State.IDLE: "In attesa (corpo non inquadrato)",
    State.READY: "Mettiti in posizione di tiro",
    State.AIMING: "In posizione — analisi attiva",
    State.HOLD: "HOLD (in mira, fermo)",
    State.RELEASE: "Colpo fatto — abbassa / ricarica",
}


@dataclass
class FrameState:
    state: State = State.IDLE
    # Fields named to be drop-in for overlay/recorder that read a 'stability'.
    in_hold: bool = False
    wrist_jitter: float = float("nan")
    sway: float = float("nan")
    hold_frames: int = 0
    analyze: bool = False   # whether THIS frame is a real aiming frame to record

    @property
    def is_aiming(self) -> bool:
        return self.state in (State.AIMING, State.HOLD)

    @property
    def label(self) -> str:
        return _LABEL[self.state]


class StateMachine:
    def __init__(self, ref: dict, fps: float = 30.0) -> None:
        s = ref["state"]
        p = ref["phases"]
        st = ref["stability"]
        self.aim_raise_min = s["aim_raise_min"]
        self.aim_arm_min = s["aim_arm_min_deg"]
        self.down_raise_max = s["down_raise_max"]
        self.enter_aim = s["enter_aim_frames"]
        self.exit_aim = s["exit_aim_frames"]
        self.hold_still_jitter = p["hold_still_jitter"]
        self.hold_min_frames = p["hold_min_frames"]
        self.window = max(1, int(st["hold_window_sec"] * fps))

        self._wrist: deque = deque(maxlen=self.window)
        self._center: deque = deque(maxlen=self.window)
        self._sw: deque = deque(maxlen=self.window)
        self._metric_buf: deque = deque(maxlen=self.window)

        self.fps = fps if fps > 1 else 30.0
        self.state = State.IDLE
        self._aim_run = 0        # consecutive 'looks like aiming' frames
        self._down_run = 0       # consecutive 'arm down' frames
        self._hold_frames = 0
        # per-attempt (one raise->lower cycle)
        self._attempt_active = False
        self._hold_achieved = False
        self._best_jitter = float("inf")
        self._best_sway = float("nan")
        self._best_metrics: Optional[Metrics] = None
        self._hold_frames_max = 0
        self._t_raise = 0.0
        self._t_hold_start = float("nan")
        self._t_hold_end = float("nan")
        self._peak_elev = float("nan")
        self.shots: list[ShotEvent] = []

    # -- helpers ---------------------------------------------------------------
    def _looks_aiming(self, m: Metrics) -> bool:
        return (m.torso_seen
                and np.isfinite(m.arm_raise) and m.arm_raise >= self.aim_raise_min
                and np.isfinite(m.arm_extension) and m.arm_extension >= self.aim_arm_min)

    def _arm_down(self, m: Metrics) -> bool:
        return np.isfinite(m.arm_raise) and m.arm_raise <= self.down_raise_max

    # -- main ------------------------------------------------------------------
    def update(self, m: Metrics, t: float) -> FrameState:
        fs = FrameState()

        # No reliable torso at all -> IDLE, abandon any attempt.
        if not m.torso_seen or not np.isfinite(m.shoulder_width):
            self._to_idle()
            fs.state = self.state
            return fs

        # stability buffers (only meaningful with a tracked wrist)
        jitter = sway = float("nan")
        if m.armed_wrist is not None and m.body_center is not None:
            self._wrist.append(np.asarray(m.armed_wrist, float))
            self._center.append(np.asarray(m.body_center, float))
            self._sw.append(m.shoulder_width)
            self._metric_buf.append(m)
            if len(self._wrist) >= max(3, self.window // 2):
                sw = float(np.median(self._sw)) or 1.0
                # Use only the image-plane (x, y): with a single camera the depth
                # (z) is noisy, so we measure the tremor we can actually trust.
                # NOTE: this captures vertical + lateral wobble in the camera
                # plane; true depth tremor needs a second camera (fase 2).
                wj = np.std(np.stack(self._wrist)[:, :2], axis=0)
                cj = np.std(np.stack(self._center)[:, :2], axis=0)
                jitter = float(np.linalg.norm(wj)) / sw
                sway = float(np.linalg.norm(cj)) / sw

        aiming = self._looks_aiming(m)
        down = self._arm_down(m)
        self._aim_run = self._aim_run + 1 if aiming else 0
        self._down_run = self._down_run + 1 if down else 0

        # --- state transitions ---
        if self.state in (State.IDLE, State.READY):
            self.state = State.READY
            if self._aim_run >= self.enter_aim:
                self.state = State.AIMING
                self._start_attempt(t)
        elif self.state in (State.AIMING, State.HOLD, State.RELEASE):
            still = np.isfinite(jitter) and jitter <= self.hold_still_jitter
            if aiming:
                # arm genuinely up & extended -> this is a real aiming frame
                self._hold_frames = self._hold_frames + 1 if still else 0
                if self._hold_frames >= self.hold_min_frames:
                    self.state = State.HOLD
                    if not self._hold_achieved:
                        self._t_hold_start = t
                    self._hold_achieved = True
                    self._t_hold_end = t  # extend while still holding
                    self._hold_frames_max = max(self._hold_frames_max, self._hold_frames)
                    if np.isfinite(jitter) and jitter < self._best_jitter:
                        self._best_jitter = jitter
                        self._best_sway = sway
                        self._best_metrics = m
                else:
                    self.state = State.AIMING
            else:
                # arm not up this frame: after a shot this is the descent/reload
                # (RELEASE, not analysed); before any hold it's still rising.
                self._hold_frames = 0
                self.state = State.RELEASE if self._hold_achieved else State.AIMING
            # arm fully lowered -> end of cycle (shot registered if a hold happened)
            if self._down_run >= self.exit_aim:
                self._end_attempt(t)
                self.state = State.READY

        # track the peak arm elevation during the attempt (approach above target)
        if self._attempt_active and np.isfinite(m.arm_elevation):
            if not np.isfinite(self._peak_elev) or m.arm_elevation > self._peak_elev:
                self._peak_elev = m.arm_elevation

        fs.state = self.state
        fs.in_hold = self.state == State.HOLD
        fs.wrist_jitter = jitter
        fs.sway = sway
        fs.hold_frames = self._hold_frames
        # Analyse/record only genuine aiming frames (arm up): excludes the
        # post-shot descent and the reload.
        fs.analyze = (self.state == State.HOLD) or (self.state == State.AIMING and aiming)
        return fs

    # -- attempt bookkeeping ---------------------------------------------------
    def _start_attempt(self, t: float = 0.0) -> None:
        self._attempt_active = True
        self._hold_achieved = False
        self._hold_frames = 0
        self._hold_frames_max = 0
        self._best_jitter = float("inf")
        self._best_sway = float("nan")
        self._best_metrics = None
        self._t_raise = t
        self._t_hold_start = float("nan")
        self._t_hold_end = float("nan")
        self._peak_elev = float("nan")

    def _end_attempt(self, t: float) -> None:
        if self._attempt_active and self._hold_achieved:
            self._register_shot(t)
        self._attempt_active = False
        self._hold_achieved = False
        self._hold_frames = 0

    def _register_shot(self, t: float) -> None:
        keys = ["torso_lean", "shoulder_elevation", "arm_extension",
                "wrist_alignment", "head_tilt", "stance_width", "weight_balance"]
        snap = self._best_metrics
        means = {}
        for k in keys:
            v = getattr(snap, k) if snap is not None else float("nan")
            means[k] = float(v) if (v is not None and np.isfinite(v)) else float("nan")

        ths = self._t_hold_start
        the = self._t_hold_end
        time_to_hold = (ths - self._t_raise) if np.isfinite(ths) else float("nan")
        hold_dur = (the - ths) if (np.isfinite(ths) and np.isfinite(the)) else float("nan")
        descent = (t - the) if np.isfinite(the) else float("nan")
        settle = getattr(snap, "arm_elevation", float("nan")) if snap is not None else float("nan")

        self.shots.append(ShotEvent(
            t_end=t,
            wrist_jitter=(self._best_jitter if self._best_jitter != float("inf")
                          else float("nan")),
            sway=self._best_sway,
            duration=self._hold_frames_max / self.fps,
            metrics_mean=means,
            t_start=self._t_raise,
            time_to_hold=time_to_hold,
            hold_duration=hold_dur,
            descent_time=descent,
            total_time=t - self._t_raise,
            raise_peak=self._peak_elev,
            settle_elev=float(settle) if (settle is not None and np.isfinite(settle)) else float("nan"),
        ))

    def _to_idle(self) -> None:
        # torso lost: if we were mid-hold, don't count a phantom shot.
        self.state = State.IDLE
        self._aim_run = 0
        self._down_run = 0
        self._hold_frames = 0
        self._attempt_active = False
        self._hold_achieved = False
