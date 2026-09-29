"""Camera helpers — robust open + selection on macOS (avoid Continuity Camera).

macOS exposes the iPhone (Continuity Camera) as an extra device, often at index
0, which is why the app may grab the phone instead of the built-in FaceTime cam.
`list_cameras()` probes the indices so you can pick the right one, and
`open_camera()` uses the AVFoundation backend with a short warm-up so the first
few empty frames don't abort the app.
"""
from __future__ import annotations

import sys

import cv2

_IS_MAC = sys.platform == "darwin"
_BACKEND = cv2.CAP_AVFOUNDATION if _IS_MAC else cv2.CAP_ANY


def list_cameras(max_index: int = 6) -> list[dict]:
    """Probe camera indices 0..max_index-1. Returns those that open, with size."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _BACKEND)
        if cap.isOpened():
            ok, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append({"index": i, "width": w, "height": h, "reads": bool(ok)})
        cap.release()
    return found


def print_cameras(max_index: int = 6) -> None:
    cams = list_cameras(max_index)
    if not cams:
        print("Nessuna camera trovata. Concedi l'accesso in Impostazioni > "
              "Privacy e sicurezza > Fotocamera e riprova.")
        return
    print("Camere disponibili (usa --camera N):")
    for c in cams:
        note = "" if c["reads"] else "  (non legge frame)"
        print(f"  [{c['index']}]  {c['width']}x{c['height']}{note}")
    print("Nota: su macOS l'iPhone (Continuity Camera) è spesso l'indice 0. "
          "La webcam integrata del Mac è di solito un altro indice.")


def autodetect() -> int:
    """Pick the best working camera automatically.

    Prefers the HIGHEST index that actually delivers frames: on macOS the iPhone
    (Continuity Camera) sits at index 0, while the built-in Mac webcam is a higher
    index, so this tends to pick the Mac webcam and skip the phone.
    """
    cams = list_cameras()
    readable = [c["index"] for c in cams if c["reads"]]
    if readable:
        return max(readable)
    if cams:
        return cams[0]["index"]
    raise SystemExit(
        "Nessuna webcam trovata.\n"
        "  Concedi l'accesso in Impostazioni > Privacy e sicurezza > Fotocamera, "
        "poi riapri il terminale."
    )


def open_camera(index: int, width: int = 1280, height: int = 720,
                warmup_frames: int = 30) -> cv2.VideoCapture:
    """Open a camera and wait until it delivers real frames.

    Raises SystemExit with a helpful message if it can't be opened/read.
    """
    cap = cv2.VideoCapture(index, _BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(
            f"Impossibile aprire la camera {index}.\n"
            "  - Concedi l'accesso in Impostazioni > Privacy e sicurezza > Fotocamera.\n"
            "  - Elenca le camere con:  python -m tenring.app --list-cameras"
        )
    # Warm-up: Continuity Camera / iPhone can take a few frames to start.
    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
    cap.release()
    raise SystemExit(
        f"La camera {index} si apre ma non fornisce immagini.\n"
        "  - Prova un altro indice:  python -m tenring.app --list-cameras\n"
        "  - Se si attiva l'iPhone e vuoi la webcam del Mac, scegli l'altro indice."
    )
