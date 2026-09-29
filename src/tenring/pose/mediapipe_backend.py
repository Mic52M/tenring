"""MediaPipe Pose Landmarker backend (Tasks API, MediaPipe >= 1.0).

BlazePose via the modern Tasks API: 33 landmarks in normalised image space plus
3D world landmarks in meters. Runs real-time on Apple Silicon CPU; the model file
is downloaded once and cached under models/.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .base import NUM_KP, PoseBackend, PoseResult

_MODELS = {
    0: ("pose_landmarker_lite",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    1: ("pose_landmarker_full",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    2: ("pose_landmarker_heavy",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
}

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def _ensure_model(complexity: int) -> Path:
    name, url = _MODELS[complexity]
    _MODELS_DIR.mkdir(exist_ok=True)
    path = _MODELS_DIR / f"{name}.task"
    if not path.exists():
        print(f"[tenring] scarico il modello {name} (una volta sola)...")
        urllib.request.urlretrieve(url, path)
        print(f"[tenring] modello salvato in {path}")
    return path


class MediaPipeBackend(PoseBackend):
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,  # kept for API compatibility
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "mediapipe non installato. Attiva l'env e installa i requirements:\n"
                "    pip install -r requirements.txt"
            ) from e

        self._mp = mp
        model_path = _ensure_model(model_complexity)

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_image)

        if not res.pose_landmarks:
            return PoseResult.empty()

        lms = res.pose_landmarks[0]
        image_xy = np.zeros((NUM_KP, 2), dtype=np.float32)
        visibility = np.zeros(NUM_KP, dtype=np.float32)
        for i, lm in enumerate(lms):
            image_xy[i] = (lm.x, lm.y)
            visibility[i] = getattr(lm, "visibility", 1.0) or 0.0

        world_xyz = np.zeros((NUM_KP, 3), dtype=np.float32)
        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks[0]):
                world_xyz[i] = (lm.x, lm.y, lm.z)

        return PoseResult(
            ok=True,
            image_xy=image_xy,
            world_xyz=world_xyz,
            visibility=visibility,
        )

    def close(self) -> None:
        self._landmarker.close()
