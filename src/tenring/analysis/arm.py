"""Automatic detection of the shooting arm.

In 10 m air pistol the shooting arm is the one RAISED and EXTENDED toward the
target; the support arm hangs low. Instead of trusting a fixed handedness config
(fragile, and confused by the mirror effect that swaps MediaPipe's left/right
labels), we detect per frame which arm is actually up and extended, and smooth
the decision over time with hysteresis so it doesn't flicker.

We only need to know *which* landmarks belong to the shooting arm — not whether
they are anatomically 'left' or 'right' — so this is robust to mirroring.
"""
from __future__ import annotations

from ..pose.base import KP, PoseResult
from . import geometry as G


def _raise_score(pose: PoseResult, shoulder, elbow, wrist, hip) -> float:
    """How much this arm looks 'raised & extended toward a target'.

    Higher = more like a shooting arm. Combines: wrist lifted toward shoulder
    height (vs hanging near the hip) and arm extension (elbow near straight).
    Returns -inf if the arm isn't visible enough to judge.
    """
    vis = pose.visibility
    if vis[shoulder] < 0.4 or vis[elbow] < 0.4 or vis[wrist] < 0.4:
        return float("-inf")
    w = pose.world_xyz
    # y points down: wrist high above hip -> (hip.y - wrist.y) large positive.
    lifted = (w[hip][1] - w[wrist][1])
    ext = G.angle_at(w[shoulder], w[elbow], w[wrist]) / 180.0  # 0..1
    if ext != ext:  # NaN
        ext = 0.0
    return lifted * (0.4 + 0.6 * ext)


class ArmSelector:
    """Smoothed decision of the shooting-arm side ('left'/'right')."""

    def __init__(self, default: str = "right", decay: float = 0.85,
                 flip_threshold: float = 0.35) -> None:
        self.state = default
        self._ema = 1.0 if default == "right" else -1.0
        self.decay = decay
        self.flip = flip_threshold

    def update(self, pose: PoseResult) -> str:
        if not pose.ok:
            return self.state
        right = _raise_score(pose, KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW,
                             KP.RIGHT_WRIST, KP.RIGHT_HIP)
        left = _raise_score(pose, KP.LEFT_SHOULDER, KP.LEFT_ELBOW,
                            KP.LEFT_WRIST, KP.LEFT_HIP)
        if right == float("-inf") and left == float("-inf"):
            return self.state
        # Only vote when at least one arm is clearly raised (positive lift).
        if max(right, left) <= 0:
            return self.state
        vote = 1.0 if right >= left else -1.0
        self._ema = self.decay * self._ema + (1.0 - self.decay) * vote
        if self._ema > self.flip:
            self.state = "right"
        elif self._ema < -self.flip:
            self.state = "left"
        return self.state
