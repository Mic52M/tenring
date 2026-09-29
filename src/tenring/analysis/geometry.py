"""Pure geometric helpers. No pose/model knowledge here — just vectors & angles."""
from __future__ import annotations

import numpy as np

VERTICAL = np.array([0.0, 1.0, 0.0])  # world y points DOWN in MediaPipe convention


def _v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(b, dtype=float) - np.asarray(a, dtype=float)


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex b of the triangle a-b-c, in degrees [0..180]."""
    v1, v2 = _v(b, a), _v(b, c)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_to_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the segment a->b from the vertical axis, in degrees [0..180].

    0 = perfectly vertical. Uses the sagittal component so a side camera reads
    forward/backward lean.
    """
    seg = _v(a, b)
    n = np.linalg.norm(seg)
    if n < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(seg, VERTICAL) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def signed_lean_sagittal(hip: np.ndarray, shoulder: np.ndarray) -> float:
    """Signed torso lean in the sagittal (side) plane, degrees.

    Uses world z (depth, toward camera) vs y (down). Positive = leaning BACK
    (shoulders behind hips, away from the target the shooter faces). The sign
    is normalised later using camera_side/handedness.
    """
    dz = shoulder[2] - hip[2]      # depth difference
    dy = hip[1] - shoulder[1]      # vertical span (positive, since y points down)
    if abs(dy) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(dz, dy)))


def line_tilt_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """Tilt of the line p1-p2 from horizontal, in the image/frontal plane, degrees.

    Uses x (right) and y (down). 0 = level.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(dy, dx)))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(_v(a, b)))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0
