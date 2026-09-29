import numpy as np
from tenring.analysis.metrics import Metrics
from tenring.analysis.baseline import SessionBaseline
from tenring import config as cfg
from tenring.analysis.rules import RuleEngine, Status

def test_baseline_makes_consistent_shooter_ok():
    ref=cfg.load_reference(); eng=RuleEngine(ref,None); base=SessionBaseline(min_frames=25)
    # 60 frames of a stable but 'absolutely weird' wrist angle (158) -> should
    # read OK once the baseline learns it (was 100% bad against absolute).
    last=None
    for i in range(60):
        m=Metrics(wrist_alignment=158.0+np.random.uniform(-1,1),
                  arm_extension=156.0, shoulder_elevation=0.19, torso_lean=-5.0, head_tilt=4.0)
        base.update(m)
        last={f.key:f.status for f in eng.evaluate(m, neutral=base.as_dict())}
    assert last['wrist_alignment']==Status.OK

def test_baseline_warmup_returns_none():
    base=SessionBaseline(min_frames=25)
    base.update(Metrics(wrist_alignment=158.0))
    assert base.get('wrist_alignment') is None  # not enough frames yet
