"""OpenCV overlay: skeleton + live coaching panel (traffic-light + cues)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..pose.base import KP, PoseResult
from ..analysis.rules import Finding, Status
from ..analysis.phases import Stability

# Minimal skeleton edges (subset that matters for pistol posture).
EDGES = [
    (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER),
    (KP.LEFT_SHOULDER, KP.LEFT_HIP),
    (KP.RIGHT_SHOULDER, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.RIGHT_HIP),
    (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW),
    (KP.RIGHT_ELBOW, KP.RIGHT_WRIST),
    (KP.LEFT_SHOULDER, KP.LEFT_ELBOW),
    (KP.LEFT_ELBOW, KP.LEFT_WRIST),
    (KP.LEFT_HIP, KP.LEFT_KNEE),
    (KP.LEFT_KNEE, KP.LEFT_ANKLE),
    (KP.RIGHT_HIP, KP.RIGHT_KNEE),
    (KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
]


def draw_skeleton(frame: np.ndarray, pose: PoseResult) -> None:
    if not pose.ok:
        return
    h, w = frame.shape[:2]
    pts = (pose.image_xy * np.array([w, h])).astype(int)
    for a, b in EDGES:
        if pose.visibility[a] > 0.3 and pose.visibility[b] > 0.3:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (200, 200, 200), 2)
    for i in range(len(pts)):
        if pose.visibility[i] > 0.3:
            cv2.circle(frame, tuple(pts[i]), 3, (230, 230, 230), -1)


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    sub = frame[y:y + h, x:x + w]
    dark = (sub * 0.35).astype(np.uint8)
    frame[y:y + h, x:x + w] = dark


def draw_panel(
    frame: np.ndarray,
    findings: list[Finding],
    stability: Optional[Stability],
    ref: dict,
    fps: float = 0.0,
    n_shots: int = 0,
    armed_side: str = "",
) -> None:
    h, w = frame.shape[:2]
    pw = 340
    _panel(frame, 0, 0, pw, h)

    y = 30
    cv2.putText(frame, "tenring — 10m air pistol", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    if armed_side:
        label = "destro" if armed_side == "right" else "sinistro"
        cv2.putText(frame, f"Braccio arma: {label}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1, cv2.LINE_AA)
    y += 22

    for fnd in findings:
        cv2.circle(frame, (20, y - 5), 7, fnd.color, -1)
        val = "n/d" if (isinstance(fnd.value, float) and np.isnan(fnd.value)) else f"{fnd.value:6.1f}"
        cv2.putText(frame, f"{fnd.label:<18} {val}", (36, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22
        if fnd.status in (Status.WARN, Status.BAD) and fnd.cue:
            for line in _wrap(fnd.cue, 40):
                cv2.putText(frame, line, (36, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, fnd.color, 1, cv2.LINE_AA)
                y += 18
        y += 4

    # stability block
    y += 6
    cv2.line(frame, (12, y), (pw - 12, y), (90, 90, 90), 1)
    y += 22
    if stability is not None:
        s = ref["stability"]
        jit = stability.wrist_jitter
        col = (80, 200, 80)
        if np.isfinite(jit):
            if jit > s["wrist_jitter_bad"]:
                col = (60, 60, 235)
            elif jit > s["wrist_jitter_warn"]:
                col = (0, 200, 255)
            jtxt = f"{jit:.4f}"
        else:
            jtxt = "n/d"
        state = "HOLD" if stability.in_hold else "..."
        cv2.putText(frame, f"Tremore polso: {jtxt}  [{state}]", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        y += 22

    cv2.putText(frame, f"Colpi: {n_shots}   FPS: {fps:4.1f}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # help footer
    cv2.putText(frame, "[q] esci  [s] salva resoconto", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_banner(frame: np.ndarray, findings: list[Finding],
                stability: Optional[Stability]) -> None:
    """Big glanceable bar at the top: the single most important correction now."""
    worst = None
    for f in findings:
        if f.status == Status.BAD and f.cue:
            worst = f
            break
    if worst is None:
        for f in findings:
            if f.status == Status.WARN and f.cue:
                worst = f
                break

    h, w = frame.shape[:2]
    x0, bar_h = 350, 76
    overlay_img = frame.copy()
    color = worst.color if worst else (80, 200, 80)
    cv2.rectangle(overlay_img, (x0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay_img, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, 0), (x0 + 8, bar_h), color, -1)  # colored edge

    if worst is None:
        cv2.putText(frame, "Postura OK", (x0 + 24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    else:
        lines = _wrap(f"{worst.label}: {worst.cue}", 46)[:2]
        y = 32 if len(lines) > 1 else 46
        for line in lines:
            cv2.putText(frame, line, (x0 + 24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            y += 30

    if stability is not None and stability.in_hold:
        cv2.putText(frame, "HOLD", (w - 110, bar_h + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 120), 2, cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines
