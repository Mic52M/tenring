"""Shared phase/stability data types — THEORY §6.

The shot itself (pellet release) is not visible to the camera, so we detect the
*hold*: the armed arm is raised/extended and the wrist becomes still for a
sustained window. Stability during that window (wrist jitter, body sway) is the
metric that separates elite from novice shooters in the literature.

The shooting-cycle state machine lives in `state.py`; these are just the record
types it produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stability:
    in_hold: bool = False
    wrist_jitter: float = float("nan")   # std of wrist pos over window / shoulder width
    sway: float = float("nan")           # std of body center over window / shoulder width
    hold_frames: int = 0


@dataclass
class ShotEvent:
    """One completed shot cycle: raise -> hold (shot) -> lower.

    Carries the steadiest-hold posture plus the phase timings and how high the
    arm was raised above the settle, so repeatability can be analysed per shot.
    """
    t_end: float
    wrist_jitter: float
    sway: float
    duration: float                      # hold length (s)
    metrics_mean: dict = field(default_factory=dict)

    # --- per-shot timings (s) ---
    t_start: float = float("nan")        # arm left rest / entered aiming
    time_to_hold: float = float("nan")   # rise+settle time until steady hold
    hold_duration: float = float("nan")  # time held steady
    descent_time: float = float("nan")   # hold end -> arm down (reload)
    total_time: float = float("nan")     # whole cycle

    # --- arm elevation (angle shoulder->wrist vs torso, deg; ~90 at hold) ---
    raise_peak: float = float("nan")     # highest elevation during approach
    settle_elev: float = float("nan")    # elevation at the steady hold
