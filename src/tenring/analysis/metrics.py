"""Per-frame posture metrics for 10 m air pistol, grounded in docs/THEORY.md.

All angles/ratios are computed from MediaPipe *world* landmarks (meters) so they
are invariant to the shooter's distance from the camera. Ratios are normalised
on shoulder width, a stable body-scale reference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..pose.base import KP, PoseResult
from . import geometry as G


@dataclass
class Metrics:
    """One frame's posture readout. NaN where not reliably measurable."""
    torso_lean: float = float("nan")        # signed sagittal lean, deg (THEORY §2)
    shoulder_elevation: float = float("nan")  # armed-shoulder shrug ratio (THEORY §3)
    arm_extension: float = float("nan")     # elbow angle, deg (THEORY §3)
    wrist_alignment: float = float("nan")   # wrist angle, deg (THEORY §4)
    head_tilt: float = float("nan")         # head roll from horizontal, deg (THEORY §5)
    stance_width: float = float("nan")      # ankle gap / shoulder width (THEORY §2)
    weight_balance: float = float("nan")    # hip vs ankle x-offset ratio (THEORY §2)

    # Raw tracking points (world meters) used by phase/stability analysis.
    armed_wrist: Optional[tuple] = None
    body_center: Optional[tuple] = None
    shoulder_width: float = float("nan")
    quality: float = 0.0                    # mean visibility of key landmarks

    def as_dict(self) -> dict:
        return asdict(self)


def _sides(handedness: str):
    """Return (armed, free) landmark index groups for the given handedness."""
    if handedness == "right":
        armed = dict(shoulder=KP.RIGHT_SHOULDER, elbow=KP.RIGHT_ELBOW,
                     wrist=KP.RIGHT_WRIST, index=KP.RIGHT_INDEX)
        free = dict(shoulder=KP.LEFT_SHOULDER)
    else:
        armed = dict(shoulder=KP.LEFT_SHOULDER, elbow=KP.LEFT_ELBOW,
                     wrist=KP.LEFT_WRIST, index=KP.LEFT_INDEX)
        free = dict(shoulder=KP.RIGHT_SHOULDER)
    return armed, free


def compute(pose: PoseResult, handedness: str = "right") -> Metrics:
    if not pose.ok:
        return Metrics()

    w = pose.world_xyz
    img = pose.image_xy
    vis = pose.visibility
    armed, free = _sides(handedness)

    m = Metrics()

    # --- body-scale reference ---
    l_sh, r_sh = w[KP.LEFT_SHOULDER], w[KP.RIGHT_SHOULDER]
    sw = G.dist(l_sh, r_sh)
    m.shoulder_width = sw
    if sw < 1e-6:
        return m

    l_hip, r_hip = w[KP.LEFT_HIP], w[KP.RIGHT_HIP]
    mid_hip = G.midpoint(l_hip, r_hip)
    mid_sh = G.midpoint(l_sh, r_sh)
    m.body_center = tuple(mid_hip)

    # --- 1. Torso lean (sagittal) — THEORY §2 ---
    m.torso_lean = G.signed_lean_sagittal(mid_hip, mid_sh)

    # --- 2. Shoulder elevation (shrug) — THEORY §3 ---
    # y points down: armed shoulder higher => smaller y => positive ratio.
    armed_sh = w[armed["shoulder"]]
    free_sh = w[free["shoulder"]]
    m.shoulder_elevation = float((free_sh[1] - armed_sh[1]) / sw)

    # --- 3. Arm extension (elbow angle) — THEORY §3 ---
    m.arm_extension = G.angle_at(
        w[armed["shoulder"]], w[armed["elbow"]], w[armed["wrist"]]
    )

    # --- 4. Wrist alignment — THEORY §4 ---
    m.wrist_alignment = G.angle_at(
        w[armed["elbow"]], w[armed["wrist"]], w[armed["index"]]
    )

    # --- 5. Head tilt (roll) — THEORY §5 ---
    # Use eyes in image space (frontal plane roll); fall back to ears.
    if vis[KP.LEFT_EYE] > 0.3 and vis[KP.RIGHT_EYE] > 0.3:
        m.head_tilt = abs(G.line_tilt_deg(img[KP.LEFT_EYE], img[KP.RIGHT_EYE]))
    elif vis[KP.LEFT_EAR] > 0.3 and vis[KP.RIGHT_EAR] > 0.3:
        m.head_tilt = abs(G.line_tilt_deg(img[KP.LEFT_EAR], img[KP.RIGHT_EAR]))

    # --- 6. Stance width — THEORY §2 ---
    l_ank, r_ank = w[KP.LEFT_ANKLE], w[KP.RIGHT_ANKLE]
    if vis[KP.LEFT_ANKLE] > 0.3 and vis[KP.RIGHT_ANKLE] > 0.3:
        m.stance_width = G.dist(l_ank, r_ank) / sw

        # --- 7. Weight balance (hip vs ankle horizontal offset) — THEORY §2 ---
        mid_ank = G.midpoint(l_ank, r_ank)
        m.weight_balance = float((mid_hip[0] - mid_ank[0]) / sw)

    # --- tracking point for stability ---
    m.armed_wrist = tuple(w[armed["wrist"]])

    # --- frame quality ---
    key = [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, KP.LEFT_HIP, KP.RIGHT_HIP,
           armed["elbow"], armed["wrist"]]
    m.quality = float(np.mean([vis[k] for k in key]))

    return m
