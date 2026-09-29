"""Personal calibration — capture the shooter's own neutral hold posture.

The app is meant to be tuned to *you*: stand in your best, comfortable air-pistol
hold for a few seconds; we record the median of the camera-dependent metrics and
store them as your 'neutral' in config/profile.yaml. Rules then flag deviations
from YOUR baseline, while the literature guardrails in reference.yaml still catch
gross faults.

Usage:
    python -m tenring.calibrate --seconds 6
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from . import config as cfg
from . import capture
from .pose.mediapipe_backend import MediaPipeBackend
from .analysis.metrics import compute


NEUTRAL_KEYS = ["torso_lean", "shoulder_elevation", "arm_extension",
                "wrist_alignment", "head_tilt", "stance_width", "weight_balance"]


def main() -> None:
    ap = argparse.ArgumentParser(description="tenring — calibrazione posturale personale")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--list-cameras", action="store_true",
                    help="elenca le camere disponibili ed esci")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        capture.print_cameras()
        return

    ref = cfg.load_reference()
    handedness = ref["setup"]["handedness"]

    cap = capture.open_camera(args.camera)

    print(f"[calibrate] Mettiti nella tua posizione di tiro ottimale.")
    print(f"[calibrate] Raccolgo per {args.seconds:.0f}s dopo il conto alla rovescia...")

    win = "tenring — calibrazione"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    buffers = {k: [] for k in NEUTRAL_KEYS}
    start = None

    with MediaPipeBackend(model_complexity=2) as backend:
        empty_reads = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > 90:
                    break
                cv2.waitKey(10)
                continue
            empty_reads = 0
            if args.mirror:
                frame = cv2.flip(frame, 1)
            pose = backend.process(frame)
            m = compute(pose, handedness=handedness)

            now = time.time()
            if start is None:
                # 3s countdown before recording
                cv2.putText(frame, "Preparati... in posizione", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
                if _key_or_wait():
                    start = now + 3.0
            elif now < start:
                cd = int(start - now) + 1
                cv2.putText(frame, str(cd), (frame.shape[1] // 2, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 220, 255), 4)
            elif now < start + args.seconds:
                if m.quality > 0.5:  # only record well-detected frames
                    for k in NEUTRAL_KEYS:
                        v = getattr(m, k)
                        if np.isfinite(v):
                            buffers[k].append(v)
                remaining = start + args.seconds - now
                cv2.putText(frame, f"REGISTRO... {remaining:3.1f}s", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 80), 2)
            else:
                break

            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                start = start or 0  # allow abort
                break

    cap.release()
    cv2.destroyAllWindows()

    neutral = {}
    for k in NEUTRAL_KEYS:
        if buffers[k]:
            neutral[k] = round(float(np.median(buffers[k])), 3)
    if not neutral:
        raise SystemExit("[calibrate] Nessun dato raccolto: riprova con più luce/inquadratura piena.")

    profile = cfg.load_profile() or {}
    profile["neutral"] = neutral
    profile["calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = cfg.save_profile(profile)
    print(f"[calibrate] Profilo salvato in {path}")
    for k, v in neutral.items():
        print(f"    {k:<20} {v}")


def _key_or_wait() -> bool:
    # Start recording automatically; press any key handled by caller loop.
    return True


if __name__ == "__main__":
    main()
