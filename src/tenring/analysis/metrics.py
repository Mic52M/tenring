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
    armed_side: str = ""                    # 'left'/'right' actually used this frame
    arm_raise: float = float("nan")         # armed-wrist height: 0=at hip, 1=at shoulder
    torso_seen: bool = False                # shoulders+hips reliably in frame

    def as_dict(self) -> dict:
        return asdict(self)


def _seen(pose: PoseResult, idx: int, vmin: float = 0.6, margin: float = 0.02) -> bool:
    """True only if a landmark is confidently detected AND inside the frame.

    MediaPipe extrapolates off-screen joints with a moderate visibility score, so
    visibility alone isn't enough: we also require the normalised image position
    to be within the frame. This is what stops us from judging feet that aren't
    actually in shot.
    """
    if pose.visibility[idx] < vmin:
        return False
    x, y = pose.image_xy[idx]
    return margin <= x <= 1.0 - margin and margin <= y <= 1.0 - margin


def _fold_tilt(raw: float) -> float:
    """Fold a line angle (deg) into [-90, 90], i.e. deviation from horizontal.

    A horizontal line reads ~0 or ~180 depending on point order; both mean 'level'.
    Returns the signed roll magnitude, direction-agnostic to point ordering.
    """
    if raw != raw:  # NaN
        return raw
    return ((raw + 90.0) % 180.0) - 90.0


def _sides(handedness: str):
    """Return (armed, free) landmark index groups for the given handedness."""
    if handedness == "right":
        armed = dict(shoulder=KP.RIGHT_SHOULDER, elbow=KP.RIGHT_ELBOW,
                     wrist=KP.RIGHT_WRIST, index=KP.RIGHT_INDEX, hip=KP.RIGHT_HIP)
        free = dict(shoulder=KP.LEFT_SHOULDER)
    else:
        armed = dict(shoulder=KP.LEFT_SHOULDER, elbow=KP.LEFT_ELBOW,
                     wrist=KP.LEFT_WRIST, index=KP.LEFT_INDEX, hip=KP.LEFT_HIP)
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
    m.armed_side = handedness

    # --- body-scale reference (needs both shoulders) ---
    if not (_seen(pose, KP.LEFT_SHOULDER, 0.5) and _seen(pose, KP.RIGHT_SHOULDER, 0.5)):
        return m  # torso not framed -> nothing reliable this frame
    l_sh, r_sh = w[KP.LEFT_SHOULDER], w[KP.RIGHT_SHOULDER]
    sw = G.dist(l_sh, r_sh)
    m.shoulder_width = sw
    if sw < 1e-6:
        return m

    hips_seen = _seen(pose, KP.LEFT_HIP, 0.5) and _seen(pose, KP.RIGHT_HIP, 0.5)
    m.torso_seen = hips_seen
    l_hip, r_hip = w[KP.LEFT_HIP], w[KP.RIGHT_HIP]
    mid_hip = G.midpoint(l_hip, r_hip)
    mid_sh = G.midpoint(l_sh, r_sh)

    # --- 1. Torso lean (sagittal) — THEORY §2 (needs hips) ---
    if hips_seen:
        m.body_center = tuple(mid_hip)
        m.torso_lean = G.signed_lean_sagittal(mid_hip, mid_sh)

    # --- 2. Shoulder elevation (shrug) — THEORY §3 ---
    # y points down: armed shoulder higher => smaller y => positive ratio.
    armed_sh = w[armed["shoulder"]]
    free_sh = w[free["shoulder"]]
    m.shoulder_elevation = float((free_sh[1] - armed_sh[1]) / sw)

    # --- 3. Arm extension (elbow angle) — THEORY §3 (needs the armed arm) ---
    if _seen(pose, armed["elbow"], 0.4) and _seen(pose, armed["wrist"], 0.4):
        m.arm_extension = G.angle_at(
            w[armed["shoulder"]], w[armed["elbow"]], w[armed["wrist"]]
        )
        # How raised the armed wrist is: 0 = at hip level, 1 = at shoulder level.
        # (y points down: hip_y > shoulder_y.)
        sh_y = w[armed["shoulder"]][1]
        hip_y = w[armed["hip"]][1]
        denom = hip_y - sh_y
        if abs(denom) > 1e-6:
            m.arm_raise = float((hip_y - w[armed["wrist"]][1]) / denom)
        # --- 4. Wrist alignment — THEORY §4 (needs wrist + hand) ---
        if _seen(pose, armed["index"], 0.4):
            m.wrist_alignment = G.angle_at(
                w[armed["elbow"]], w[armed["wrist"]], w[armed["index"]]
            )
        # tracking point for stability
        m.armed_wrist = tuple(w[armed["wrist"]])

    # --- 5. Head tilt (roll) — THEORY §5 ---
    # Fold the raw line angle into [-90, 90] so left/right point ordering
    # (e.g. under --mirror) can't turn a level line into ~180 deg.
    if _seen(pose, KP.LEFT_EYE, 0.4) and _seen(pose, KP.RIGHT_EYE, 0.4):
        m.head_tilt = _fold_tilt(G.line_tilt_deg(img[KP.LEFT_EYE], img[KP.RIGHT_EYE]))
    elif _seen(pose, KP.LEFT_EAR, 0.4) and _seen(pose, KP.RIGHT_EAR, 0.4):
        m.head_tilt = _fold_tilt(G.line_tilt_deg(img[KP.LEFT_EAR], img[KP.RIGHT_EAR]))
    # Reject implausible head roll (>35 deg) as landmark noise, not a real tilt.
    if np.isfinite(m.head_tilt) and abs(m.head_tilt) > 35.0:
        m.head_tilt = float("nan")

    # --- 6+7. Stance width & weight balance — THEORY §2 ---
    # ONLY if both feet are actually in frame (not extrapolated off-screen).
    if hips_seen and _seen(pose, KP.LEFT_ANKLE) and _seen(pose, KP.RIGHT_ANKLE):
        l_ank, r_ank = w[KP.LEFT_ANKLE], w[KP.RIGHT_ANKLE]
        m.stance_width = G.dist(l_ank, r_ank) / sw
        mid_ank = G.midpoint(l_ank, r_ank)
        m.weight_balance = float((mid_hip[0] - mid_ank[0]) / sw)

    # --- frame quality (only landmarks we rely on) ---
    key = [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, armed["elbow"], armed["wrist"]]
    m.quality = float(np.mean([vis[k] for k in key]))

    return m
