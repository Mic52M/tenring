import numpy as np

from tenring.analysis import geometry as G


def test_angle_at_right_angle():
    a = np.array([1, 0, 0.0])
    b = np.array([0, 0, 0.0])
    c = np.array([0, 1, 0.0])
    assert abs(G.angle_at(a, b, c) - 90.0) < 1e-6


def test_angle_at_straight():
    a = np.array([-1, 0, 0.0])
    b = np.array([0, 0, 0.0])
    c = np.array([1, 0, 0.0])
    assert abs(G.angle_at(a, b, c) - 180.0) < 1e-6


def test_angle_to_vertical_perfectly_vertical():
    a = np.array([0, 0, 0.0])
    b = np.array([0, 1, 0.0])  # y down
    assert abs(G.angle_to_vertical(a, b)) < 1e-6


def test_signed_lean_sign():
    hip = np.array([0, 0, 0.0])
    shoulder_back = np.array([0, -1.0, 0.3])  # shoulders up (y neg) & behind (z+)
    lean = G.signed_lean_sagittal(hip, shoulder_back)
    assert lean > 0


def test_line_tilt_level():
    p1 = np.array([0.0, 0.5])
    p2 = np.array([0.2, 0.5])
    assert abs(G.line_tilt_deg(p1, p2)) < 1e-6
