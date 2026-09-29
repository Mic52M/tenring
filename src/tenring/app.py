"""tenring — live posture coach for 10 m air pistol (MVP, PC camera only).

Pipeline per frame:
    capture -> pose (MediaPipe) -> metrics -> rules -> hold/stability -> overlay
On exit (or 's'), prints/saves a session summary (consistency + stability).

Usage:
    python -m tenring.app                 # default webcam (index 0)
    python -m tenring.app --camera 1
    python -m tenring.app --mirror        # flip horizontally
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from . import config as cfg
from .pose.mediapipe_backend import MediaPipeBackend
from .analysis.metrics import compute
from .analysis.rules import RuleEngine
from .analysis.phases import HoldTracker
from .analysis.session import summarize
from .ui import overlay


def parse_args():
    ap = argparse.ArgumentParser(description="tenring — 10m air pistol posture coach")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--mirror", action="store_true", help="flip frame horizontally")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="MediaPipe model complexity (0 fast .. 2 accurate)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ref = cfg.load_reference()
    profile = cfg.load_profile()
    handedness = ref["setup"]["handedness"]

    engine = RuleEngine(ref, profile)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Impossibile aprire la camera {args.camera}. "
                         "Su macOS concedi l'accesso camera al terminale in "
                         "Impostazioni > Privacy e sicurezza > Fotocamera.")

    fps_guess = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = HoldTracker(ref, fps=fps_guess if fps_guess > 1 else 30.0)

    if profile:
        print("[tenring] profilo personale caricato (calibrazione attiva).")
    else:
        print("[tenring] nessun profilo: uso i riferimenti da letteratura. "
              "Esegui 'python -m tenring.calibrate' per personalizzare.")

    win = "tenring"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prev = time.time()
    fps = 0.0

    with MediaPipeBackend(model_complexity=args.complexity) as backend:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            pose = backend.process(frame)
            m = compute(pose, handedness=handedness)
            findings = engine.evaluate(m)
            now = time.time()
            stability = tracker.update(m, now)

            overlay.draw_skeleton(frame, pose)
            overlay.draw_panel(frame, findings, stability, ref,
                               fps=fps, n_shots=len(tracker.shots))

            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                _save_summary(tracker, ref)

    cap.release()
    cv2.destroyAllWindows()
    _save_summary(tracker, ref, final=True)


def _save_summary(tracker: HoldTracker, ref: dict, final: bool = False) -> None:
    summary = summarize(tracker.shots)
    if summary is None:
        if final:
            print("[tenring] nessun colpo rilevato in questa sessione.")
        return
    text = summary.to_text(ref)
    print("\n" + text)
    out = Path(cfg._ROOT) / "sessions"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fpath = out / f"session_{stamp}.txt"
    fpath.write_text(text, encoding="utf-8")
    print(f"[tenring] resoconto salvato in {fpath}")


if __name__ == "__main__":
    main()
