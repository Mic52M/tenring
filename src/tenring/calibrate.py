"""Personal calibration — capture the shooter's own neutral hold posture.

The app is meant to be tuned to *you*: stand in your best, comfortable air-pistol
hold for a few seconds; we record the median of the camera-dependent metrics and
store them as your 'neutral' in config/profile.yaml. Rules then flag deviations
from YOUR baseline, while the literature guardrails in reference.yaml still catch
gross faults.

Usage:
    tenring-calibrate
    python -m tenring.calibrate --seconds 6
"""
from __future__ import annotations

import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import time

import cv2
import numpy as np

from . import config as cfg
from . import capture
from .pose.mediapipe_backend import MediaPipeBackend
from .analysis.metrics import compute
from .analysis.arm import ArmSelector
from .analysis.state import StateMachine


NEUTRAL_KEYS = ["torso_lean", "shoulder_elevation", "arm_extension",
                "wrist_alignment", "head_tilt", "stance_width", "weight_balance"]


def main() -> None:
    ap = argparse.ArgumentParser(description="tenring — calibrazione posturale personale")
    ap.add_argument("--camera", type=int, default=-1,
                    help="indice camera (default: auto)")
    ap.add_argument("--list-cameras", action="store_true",
                    help="elenca le camere disponibili ed esci")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                    help="effetto specchio (default: attivo)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = ap.parse_args()

    if args.list_cameras:
        capture.print_cameras()
        return

    ref = cfg.load_reference()
    handedness = ref["setup"]["handedness"]

    cam_index = args.camera if args.camera >= 0 else capture.autodetect()
    cap = capture.open_camera(cam_index)

    print(f"[calibrate] Mettiti nella tua posizione di tiro ottimale.")
    print(f"[calibrate] Raccolgo per {args.seconds:.0f}s dopo il conto alla rovescia...")

    win = "tenring — calibrazione"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    buffers = {k: [] for k in NEUTRAL_KEYS}
    arm = ArmSelector(default=handedness)
    sm = StateMachine(ref, fps=30.0)
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
            now = time.time()
            m = compute(pose, handedness=arm.update(pose))
            fs = sm.update(m, now)

            if start is None:
                # 3s countdown before recording
                cv2.putText(frame, "Mettiti in posizione di tiro (braccio su)", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
                if _key_or_wait():
                    start = now + 3.0
            elif now < start:
                cd = int(start - now) + 1
                cv2.putText(frame, str(cd), (frame.shape[1] // 2, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 220, 255), 4)
            elif now < start + args.seconds:
                # Record ONLY while actually aiming, so the neutral is the real
                # shooting posture (not arm-down at rest).
                if fs.is_aiming and m.quality > 0.5:
                    for k in NEUTRAL_KEYS:
                        v = getattr(m, k)
                        if np.isfinite(v):
                            buffers[k].append(v)
                remaining = start + args.seconds - now
                aim_txt = "IN MIRA — registro" if fs.is_aiming else "ALZA IL BRACCIO E MIRA"
                col = (80, 220, 80) if fs.is_aiming else (0, 200, 255)
                cv2.putText(frame, f"{aim_txt}  {remaining:3.1f}s", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            else:
                break

            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                start = start or 0  # allow abort
                break

    cap.release()
    cv2.destroyAllWindows()

    n_aim = max((len(v) for v in buffers.values()), default=0)
    if n_aim < 10:
        raise SystemExit(
            "[calibrate] Pochi frame in mira raccolti "
            f"({n_aim}). Rifai la calibrazione restando in posizione con il "
            "braccio ALZATO e fermo per tutta la registrazione "
            "(usa --seconds 8 per più tempo).")

    neutral = {}
    for k in NEUTRAL_KEYS:
        if buffers[k]:
            neutral[k] = round(float(np.median(buffers[k])), 3)

    profile = cfg.load_profile() or {}
    profile["neutral"] = neutral
    profile["calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = cfg.save_profile(profile)
    print(f"[calibrate] Profilo salvato in {path}  ({n_aim} frame in mira)")
    for k, v in neutral.items():
        print(f"    {k:<20} {v}")


def _key_or_wait() -> bool:
    # Start recording automatically; press any key handled by caller loop.
    return True


if __name__ == "__main__":
    main()
