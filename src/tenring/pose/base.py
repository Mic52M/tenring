"""Abstract pose-estimation backend.

The rest of the app depends only on this interface, never on a specific model.
That lets us start with MediaPipe BlazePose (great on Apple Silicon CPU/ANE,
33 3D keypoints incl. feet) and later swap in RTMPose / YOLO-pose / ViTPose
without touching the analysis code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class KP(IntEnum):
    """MediaPipe Pose landmark indices (33-point topology)."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


NUM_KP = 33


@dataclass
class PoseResult:
    """One frame of pose estimation.

    image_xy:  (33, 2) landmarks in NORMALISED image coords [0..1], origin top-left.
               Used for drawing the overlay.
    world_xyz: (33, 3) landmarks in METERS, roughly centred on the mid-hip, with
               MediaPipe's convention (x right, y down, z toward camera). Used for
               all geometric/angle computations because it is scale- and
               distance-invariant.
    visibility:(33,) per-landmark visibility/confidence in [0..1].
    ok:        True if a person was detected this frame.
    """
    ok: bool
    image_xy: np.ndarray = field(default_factory=lambda: np.zeros((NUM_KP, 2)))
    world_xyz: np.ndarray = field(default_factory=lambda: np.zeros((NUM_KP, 3)))
    visibility: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KP))

    @staticmethod
    def empty() -> "PoseResult":
        return PoseResult(ok=False)


class PoseBackend(ABC):
    """Common interface for any pose model."""

    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        """Run inference on a single BGR frame (OpenCV order)."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional cleanup
        pass

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
