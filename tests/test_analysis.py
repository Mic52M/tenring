import numpy as np

from tenring.pose.base import KP, NUM_KP, PoseResult
from tenring.analysis.metrics import compute
from tenring.analysis.rules import RuleEngine, Status
from tenring import config as cfg


def _neutral_pose(right_shoulder_y=-0.5):
    """Synthetic right-handed shooter, arm extended forward, upright."""
    w = np.zeros((NUM_KP, 3), dtype=np.float32)
    # All landmarks default to mid-frame so the visibility/in-frame gate passes;
    # specific ones (eyes) are set below for tilt.
    img = np.full((NUM_KP, 2), 0.5, dtype=np.float32)
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


def test_head_tilt_robust_to_mirror_swap():
    # Level eye line but points in reversed (mirrored) order must still read ~0,
    # not ~180. Regression for head_tilt=111 bug seen with --mirror.
    p = _neutral_pose()
    p.image_xy[KP.LEFT_EYE] = (0.52, 0.20)   # swapped x order
    p.image_xy[KP.RIGHT_EYE] = (0.48, 0.20)
    m = compute(p, handedness="right")
    assert abs(m.head_tilt) < 3


def test_rules_neutral_mostly_ok():
    ref = cfg.load_reference()
    engine = RuleEngine(ref, profile=None)
    m = compute(_neutral_pose(), handedness="right")
    findings = {f.key: f for f in engine.evaluate(m)}
    for key in ("shoulder_elevation", "arm_extension", "head_tilt", "stance_width"):
        assert findings[key].status in (Status.OK, Status.WARN)


def test_feet_offscreen_not_evaluated():
    # Ankles out of frame (image y > 1) must NOT be judged, even if MediaPipe
    # reports them with high visibility (regression: feet graded when unseen).
    import math
    p = _neutral_pose()
    p.image_xy[KP.LEFT_ANKLE] = (0.5, 1.15)
    p.image_xy[KP.RIGHT_ANKLE] = (0.5, 1.15)
    m = compute(p, handedness="right")
    assert math.isnan(m.stance_width)
    assert math.isnan(m.weight_balance)


def test_calibrated_neutral_makes_own_posture_ok():
    # A shooter whose webcam-measured arm extension is 150 and shoulder 0.2:
    # in ABSOLUTE terms both would be flagged, but calibrated to HIS neutral they
    # must read OK when he reproduces that posture. (Fixes 100%-out-of-tolerance.)
    ref = cfg.load_reference()
    profile = {"neutral": {"arm_extension": 150.0, "shoulder_elevation": 0.2,
                           "wrist_alignment": 150.0, "head_tilt": 5.0}}
    p = _neutral_pose()
    w = p.world_xyz
    # bend the arm so measured extension ~150 and shoulder raised ~0.2
    w[KP.RIGHT_ELBOW] = (0.2, -0.5, -0.3)
    w[KP.RIGHT_WRIST] = (0.35, -0.42, -0.5)   # ~150 deg elbow
    w[KP.RIGHT_SHOULDER] = (0.2, -0.58, 0.0)  # raised
    m = compute(p, handedness="right")

    without = {f.key: f.status for f in RuleEngine(ref, None).evaluate(m)}
    withcal = {f.key: f.status for f in RuleEngine(ref, profile).evaluate(m)}
    # Without calibration at least one of these trips; with calibration (same
    # posture as neutral) shoulder/arm should be OK.
    assert withcal["shoulder_elevation"] == Status.OK
    assert withcal["arm_extension"] in (Status.OK, Status.WARN)


def test_arm_selector_picks_raised_arm():
    from tenring.analysis.arm import ArmSelector
    # Build a pose where the LEFT wrist is raised to shoulder height and the
    # right hangs at the hip: selector must converge to 'left'.
    p = _neutral_pose()
    w = p.world_xyz
    w[KP.LEFT_SHOULDER] = (-0.2, -0.5, 0.0)
    w[KP.LEFT_ELBOW] = (-0.2, -0.5, -0.3)
    w[KP.LEFT_WRIST] = (-0.2, -0.5, -0.6)   # raised & extended
    w[KP.RIGHT_WRIST] = (0.15, 0.0, 0.0)    # hanging near hip
    sel = ArmSelector(default="right")
    side = "right"
    for _ in range(40):
        side = sel.update(p)
    assert side == "left"
