import numpy as np

from tenring.pose.base import KP, NUM_KP, PoseResult
from tenring.analysis.metrics import compute
from tenring.analysis.rules import RuleEngine, Status
from tenring import config as cfg


def _neutral_pose(right_shoulder_y=-0.5):
    """Synthetic right-handed shooter, arm extended forward, upright."""
    w = np.zeros((NUM_KP, 3), dtype=np.float32)
    img = np.zeros((NUM_KP, 2), dtype=np.float32)
    vis = np.ones(NUM_KP, dtype=np.float32)

    w[KP.LEFT_SHOULDER] = (-0.2, -0.5, 0.0)
    w[KP.RIGHT_SHOULDER] = (0.2, right_shoulder_y, 0.0)
    w[KP.LEFT_HIP] = (-0.1, 0.0, 0.0)
    w[KP.RIGHT_HIP] = (0.1, 0.0, 0.0)
    # right arm extended toward target (-z)
    w[KP.RIGHT_ELBOW] = (0.2, -0.5, -0.3)
    w[KP.RIGHT_WRIST] = (0.2, -0.5, -0.6)
    w[KP.RIGHT_INDEX] = (0.2, -0.5, -0.65)
    w[KP.LEFT_ELBOW] = (-0.25, -0.25, 0.0)
    w[KP.LEFT_WRIST] = (-0.25, 0.0, 0.0)
    # legs
    w[KP.LEFT_KNEE] = (-0.15, 0.3, 0.0)
    w[KP.RIGHT_KNEE] = (0.15, 0.3, 0.0)
    w[KP.LEFT_ANKLE] = (-0.2, 0.6, 0.0)
    w[KP.RIGHT_ANKLE] = (0.2, 0.6, 0.0)

    img[KP.LEFT_EYE] = (0.48, 0.20)
    img[KP.RIGHT_EYE] = (0.52, 0.20)  # level
    return PoseResult(ok=True, image_xy=img, world_xyz=w, visibility=vis)


def test_metrics_neutral_reasonable():
    m = compute(_neutral_pose(), handedness="right")
    assert 170 <= m.arm_extension <= 185          # extended
    assert abs(m.shoulder_elevation) < 0.02        # level shoulders
    assert 0.8 <= m.stance_width <= 1.3            # ~shoulder width
    assert abs(m.head_tilt) < 3


def test_rules_flag_shrug():
    ref = cfg.load_reference()
    engine = RuleEngine(ref, profile=None)
    # right shoulder raised (higher => more negative y)
    m = compute(_neutral_pose(right_shoulder_y=-0.62), handedness="right")
    findings = {f.key: f for f in engine.evaluate(m)}
    assert findings["shoulder_elevation"].status == Status.BAD


def test_rules_neutral_mostly_ok():
    ref = cfg.load_reference()
    engine = RuleEngine(ref, profile=None)
    m = compute(_neutral_pose(), handedness="right")
    findings = {f.key: f for f in engine.evaluate(m)}
    for key in ("shoulder_elevation", "arm_extension", "head_tilt", "stance_width"):
        assert findings[key].status in (Status.OK, Status.WARN)
