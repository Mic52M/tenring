"""tenring — live posture coach for 10 m air pistol (MVP, PC camera only).

Pipeline per frame:
    capture -> pose (MediaPipe) -> metrics -> rules -> hold/stability -> overlay
On exit (or 's'), prints/saves a session summary (consistency + stability).

Uso semplice (sceglie la webcam da solo, premi Q per uscire):
    tenring
    python -m tenring.app
"""
from __future__ import annotations

import os
# Silenzia i log verbosi di MediaPipe/TensorFlow (prima di importare mediapipe).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import time
from pathlib import Path

import cv2

from . import config as cfg
from . import capture
from .pose.mediapipe_backend import MediaPipeBackend
from .analysis.metrics import compute
from .analysis.rules import RuleEngine
from .analysis.state import StateMachine
from .analysis.session import summarize
from .analysis.recorder import SessionRecorder
from .analysis.arm import ArmSelector
from .report import build_report
from .ui import overlay


def parse_args():
    ap = argparse.ArgumentParser(description="tenring — 10m air pistol posture coach")
    ap.add_argument("--camera", type=int, default=-1,
                    help="indice camera (default: auto, sceglie la webcam del Mac)")
    ap.add_argument("--list-cameras", action="store_true",
                    help="elenca le camere disponibili ed esci")
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                    help="effetto specchio (default: attivo)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false",
                    help="disattiva l'effetto specchio (per camera di profilo)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="MediaPipe model complexity (0 fast .. 2 accurate)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        capture.print_cameras()
        return

    ref = cfg.load_reference()
    profile = cfg.load_profile()
    handedness = ref["setup"]["handedness"]

    engine = RuleEngine(ref, profile)

    cam_index = args.camera if args.camera >= 0 else capture.autodetect()
    cap = capture.open_camera(cam_index, args.width, args.height)
    fps_guess = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sm = StateMachine(ref, fps=fps_guess if fps_guess > 1 else 30.0)

    print(f"[tenring] webcam {cam_index} aperta. Premi Q (o ESC) per uscire, "
          "S per salvare il resoconto.")
    if profile:
        print("[tenring] profilo personale caricato (calibrazione attiva).")
    else:
        print("[tenring] nessun profilo: uso i riferimenti da letteratura. "
              "Esegui 'tenring-calibrate' per personalizzare.")

    win = "tenring"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    recorder = SessionRecorder()
    arm = ArmSelector(default=handedness)
    prev = time.time()
    fps = 0.0

    empty_reads = 0
    with MediaPipeBackend(model_complexity=args.complexity) as backend:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > 90:  # ~3s of failures -> give up
                    print("[tenring] la camera ha smesso di fornire immagini.")
                    break
                cv2.waitKey(10)
                continue
            empty_reads = 0
            if args.mirror:
                frame = cv2.flip(frame, 1)

            pose = backend.process(frame)
            armed_side = arm.update(pose)
            m = compute(pose, handedness=armed_side)
            findings = engine.evaluate(m)
            now = time.time()
            fs = sm.update(m, now)

            # Record/report ONLY when actually in the aiming position.
            if fs.is_aiming:
                recorder.add(now, m, findings, fs)

            overlay.draw_skeleton(frame, pose)
            overlay.draw_panel(frame, findings, fs, ref,
                               fps=fps, n_shots=len(sm.shots),
                               armed_side=armed_side,
                               state_label=fs.label, aiming=fs.is_aiming)
            overlay.draw_banner(frame, findings, fs,
                                aiming=fs.is_aiming, state_label=fs.label)

            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                _save_session(sm, recorder, ref)

    cap.release()
    cv2.destroyAllWindows()
    _save_session(sm, recorder, ref, final=True)


def _save_session(sm: StateMachine, recorder: SessionRecorder, ref: dict,
                  final: bool = False) -> None:
    if len(recorder) == 0:
        if final:
            print("[tenring] nessun frame in posizione di tiro: "
                  "niente da analizzare (assicurati di metterti in posizione "
                  "con il braccio alzato, corpo inquadrato).")
        return

    summary = summarize(sm.shots)
    out = Path(cfg._ROOT) / "sessions"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    when = time.strftime("%d/%m/%Y %H:%M")

    # raw per-frame log
    recorder.save_jsonl(out / f"session_{stamp}.jsonl")

    # text summary (if shots detected)
    if summary is not None:
        (out / f"session_{stamp}.txt").write_text(summary.to_text(ref), encoding="utf-8")
        print("\n" + summary.to_text(ref))

    # HTML report + open it
    html_path = out / f"session_{stamp}.html"
    build_report(recorder.frames, sm.shots, summary, ref, html_path, when=when)
    print(f"[tenring] report: {html_path}")
    try:
        import webbrowser
        webbrowser.open(f"file://{html_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
