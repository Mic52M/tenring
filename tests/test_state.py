import copy

from tenring import config as cfg
from tenring.analysis.metrics import Metrics
from tenring.analysis.state import StateMachine, State


def _ref_fast():
    ref = copy.deepcopy(cfg.load_reference())
    ref["stability"]["hold_window_sec"] = 1.0  # shrink window for the test
    return ref


def _m(arm_raise, arm_ext, wrist=(0.2, -0.5, -0.6)):
    return Metrics(
        arm_extension=arm_ext, arm_raise=arm_raise,
        armed_wrist=wrist, body_center=(0.0, 0.0, 0.0),
        shoulder_width=0.4, torso_seen=True, quality=0.9,
    )


def test_not_aiming_when_arm_down():
    sm = StateMachine(_ref_fast(), fps=15)
    fs = None
    for i in range(10):
        fs = sm.update(_m(arm_raise=0.1, arm_ext=90), t=i / 15)
    assert fs.state == State.READY
    assert not fs.is_aiming


def test_idle_when_no_torso():
    sm = StateMachine(_ref_fast(), fps=15)
    fs = sm.update(Metrics(torso_seen=False), t=0.0)
    assert fs.state == State.IDLE
    assert not fs.is_aiming


def test_raise_hold_lower_counts_one_shot():
    sm = StateMachine(_ref_fast(), fps=15)
    t = 0.0
    # arm down
    for _ in range(6):
        sm.update(_m(0.1, 90), t); t += 1 / 15
    # raise & hold still (constant wrist -> zero jitter) for plenty of frames
    for _ in range(50):
        sm.update(_m(0.9, 176), t); t += 1 / 15
    assert sm.state in (State.AIMING, State.HOLD)
    # lower the arm -> shot registered
    for _ in range(8):
        sm.update(_m(0.1, 90), t); t += 1 / 15
    assert len(sm.shots) == 1


def test_raise_lower_without_hold_no_shot():
    sm = StateMachine(_ref_fast(), fps=15)
    t = 0.0
    for _ in range(6):
        sm.update(_m(0.1, 90), t); t += 1 / 15
    # brief raise, not long enough to reach a HOLD, then down
    for _ in range(3):
        sm.update(_m(0.9, 176), t); t += 1 / 15
    for _ in range(8):
        sm.update(_m(0.1, 90), t); t += 1 / 15
    assert len(sm.shots) == 0
