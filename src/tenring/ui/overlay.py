"""Premium live HUD overlay.

Skeleton is drawn with OpenCV (anti-aliased lines); all text and panels are
rendered with Pillow using the San Francisco system font, so nothing looks
blurry/aliased like OpenCV's Hershey fonts. One PIL compositing pass per frame.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..pose.base import KP, PoseResult
from ..analysis.rules import Finding, Status
from .. import i18n

# ---------------------------------------------------------------- palette ----
C_PANEL = (15, 18, 24, 210)      # dark glass
C_PANEL2 = (22, 27, 36, 150)
C_STROKE = (255, 255, 255, 22)
C_TEXT = (237, 240, 245)
C_MUTED = (150, 163, 180)
C_ACCENT = (56, 189, 248)        # sky blue
C_OK = (74, 222, 128)
C_WARN = (250, 204, 21)
C_BAD = (248, 113, 113)
C_BONE = (150, 163, 180)

_STATUS_RGB = {
    Status.OK: C_OK, Status.WARN: C_WARN, Status.BAD: C_BAD, Status.NA: C_MUTED,
}

# ---------------------------------------------------------------- fonts ------
_TXT = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"]
_MONO = ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"]
_FONT_CACHE: dict = {}


def _font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, mono)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    paths = _MONO + _TXT if mono else _TXT
    f = None
    for p in paths:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                continue
    if f is None:
        import matplotlib.font_manager as fm
        name = "DejaVu Sans Mono" if mono else "DejaVu Sans"
        f = ImageFont.truetype(fm.findfont(name), size)
    _FONT_CACHE[key] = f
    return f


# ---------------------------------------------------------------- skeleton ---
_EDGES_BODY = [
    (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER), (KP.LEFT_SHOULDER, KP.LEFT_HIP),
    (KP.RIGHT_SHOULDER, KP.RIGHT_HIP), (KP.LEFT_HIP, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.LEFT_KNEE), (KP.LEFT_KNEE, KP.LEFT_ANKLE),
    (KP.RIGHT_HIP, KP.RIGHT_KNEE), (KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
]
_EDGES_ARM = {
    "right": [(KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW), (KP.RIGHT_ELBOW, KP.RIGHT_WRIST)],
    "left": [(KP.LEFT_SHOULDER, KP.LEFT_ELBOW), (KP.LEFT_ELBOW, KP.LEFT_WRIST)],
}


def draw_skeleton(frame: np.ndarray, pose: PoseResult, armed_side: str = "right") -> None:
    if not pose.ok:
        return
    h, w = frame.shape[:2]
    pts = (pose.image_xy * np.array([w, h])).astype(int)
    vis = pose.visibility

    def line(a, b, color, th):
        if vis[a] > 0.3 and vis[b] > 0.3:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, th, cv2.LINE_AA)

    # NB: OpenCV colors are BGR. Scale line widths with resolution.
    s = h / 720.0
    th = max(2, int(round(2 * s)))
    tha = max(3, int(round(3 * s)))
    r = max(3, int(round(3 * s)))
    bone = (180, 163, 150)          # muted grey-blue
    accent = (248, 189, 56)         # sky blue (BGR) = highlight for armed arm
    for a, b in _EDGES_BODY:
        line(a, b, bone, th)
    other = "left" if armed_side == "right" else "right"
    for a, b in _EDGES_ARM[other]:
        line(a, b, bone, th)
    for a, b in _EDGES_ARM[armed_side]:
        line(a, b, accent, tha)
    for i in range(len(pts)):
        if vis[i] > 0.3:
            cv2.circle(frame, tuple(pts[i]), r, (245, 236, 230), -1, cv2.LINE_AA)


# ---------------------------------------------------------------- helpers ----
def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = f"{cur} {wd}".strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def _pill(draw, box, color, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=color)


# ---------------------------------------------------------------- HUD --------
def draw_hud(frame: np.ndarray, findings, fs, ref: dict, fps: float = 0.0,
             n_shots: int = 0, armed_side: str = "", lang: str = "it") -> None:
    """Composite the full premium HUD onto the frame (in place).

    All sizes scale with frame height so the HUD stays crisp at any resolution
    (720p, 1080p, ...) instead of being drawn small and upscaled by the window.
    """
    h, w = frame.shape[:2]
    sc = h / 720.0

    def S(v):  # scale a pixel size
        return int(round(v * sc))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img, "RGBA")

    aiming = getattr(fs, "analyze", False) or getattr(fs, "in_hold", False)
    in_hold = getattr(fs, "in_hold", False)
    state_label = fs.label(lang) if hasattr(fs, "label") else ""
    U = lambda k, **kw: i18n.ui(k, lang, **kw)

    m = S(12)                       # outer margin
    PW = S(320)
    pad = S(20)
    d.rounded_rectangle([m, m, PW, h - m], radius=S(18), fill=C_PANEL)
    d.rounded_rectangle([m, m, PW, h - m], radius=S(18), outline=C_STROKE, width=1)

    x = m + pad
    y = m + pad
    f_brand = _font(S(23))
    d.text((x, y), "TENRING", font=f_brand, fill=C_ACCENT)
    d.text((x + 1, y), "TENRING", font=f_brand, fill=C_ACCENT)  # faux-bold
    tw = d.textlength("TENRING", font=f_brand)
    d.text((x + tw + S(10), y + S(6)), "10m air pistol", font=_font(S(12)), fill=C_MUTED)
    y += S(42)

    st_col = C_OK if in_hold else (C_ACCENT if aiming else C_WARN)
    label = state_label or (U("hud_inpos") if aiming else U("hud_waiting"))
    _pill(d, [x, y, PW - pad, y + S(30)], (st_col[0], st_col[1], st_col[2], 38),
          radius=S(14))
    d.ellipse([x + S(10), y + S(11), x + S(18), y + S(19)], fill=st_col)
    d.text((x + S(26), y + S(7)), label, font=_font(S(14)), fill=C_TEXT)
    y += S(40)

    if armed_side:
        arm_lbl = U("hud_right") if armed_side == "right" else U("hud_left")
        d.text((x, y), f"{U('hud_arm')}: {arm_lbl}", font=_font(S(12)), fill=C_MUTED)
    y += S(24)
    d.line([x, y, PW - pad, y], fill=C_STROKE, width=1)
    y += S(14)

    f_lbl = _font(S(14))
    f_val = _font(S(14), mono=True)
    f_cue = _font(S(12))
    if not aiming:
        d.text((x, y), U("hud_paused"), font=_font(S(13)), fill=C_MUTED)
        y += S(24)
    for fnd in findings:
        if not aiming and fnd.status == Status.NA:
            continue
        col = _STATUS_RGB.get(fnd.status, C_MUTED)
        d.ellipse([x, y + S(4), x + S(9), y + S(13)], fill=col)
        d.text((x + S(18), y), fnd.label, font=f_lbl, fill=C_TEXT)
        val = "—" if (isinstance(fnd.value, float) and np.isnan(fnd.value)) else f"{fnd.value:.1f}"
        vw = d.textlength(val, font=f_val)
        d.text((PW - pad - vw, y + 1), val, font=f_val, fill=C_MUTED)
        y += S(22)
        if aiming and fnd.status in (Status.WARN, Status.BAD) and fnd.cue:
            for ln in _wrap(d, fnd.cue, f_cue, PW - pad - (x + S(18)))[:2]:
                d.text((x + S(18), y), ln, font=f_cue, fill=col)
                y += S(16)
        y += S(4)

    # bottom block
    by = h - m - pad - S(58)
    d.line([x, by - S(10), PW - pad, by - S(10)], fill=C_STROKE, width=1)
    st = ref["stability"]
    jit = getattr(fs, "wrist_jitter", float("nan"))
    if jit == jit:
        jc = C_OK if jit <= st["wrist_jitter_warn"] else (C_WARN if jit <= st["wrist_jitter_bad"] else C_BAD)
        jtxt = f"{jit:.3f}"
    else:
        jc, jtxt = C_MUTED, "—"
    d.text((x, by), U("hud_tremor"), font=_font(S(12)), fill=C_MUTED)
    d.text((x, by + S(15)), jtxt, font=_font(S(20), mono=True), fill=jc)
    d.text((x + S(150), by), U("hud_shots"), font=_font(S(12)), fill=C_MUTED)
    d.text((x + S(150), by + S(15)), str(n_shots), font=_font(S(20), mono=True), fill=C_TEXT)
    d.text((x, h - m - pad + S(6)), U("hud_help", fps=f"{fps:.0f}"),
           font=_font(S(11)), fill=C_MUTED)

    # ---- top banner ----
    bx0, bx1 = PW + S(16), w - S(16)
    if bx1 - bx0 > S(160):
        _banner(d, bx0, m, bx1, m + S(70), findings, aiming, in_hold, state_label, S, U)

    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _banner(d, x0, y0, x1, y1, findings, aiming, in_hold, state_label, S, U):
    if not aiming:
        col = C_ACCENT
        text = state_label or U("hud_waiting")
        sub = None
    else:
        worst = None
        for f in findings:
            if f.status == Status.BAD and f.cue:
                worst = f
                break
        if worst is None:
            for f in findings:
                if f.status == Status.WARN and f.cue:
                    worst = f
                    break
        if worst is None:
            col, text, sub = C_OK, U("hud_ok"), U("hud_ok_sub")
        else:
            col = _STATUS_RGB[worst.status]
            text = worst.cue
            sub = worst.label

    d.rounded_rectangle([x0, y0, x1, y1], radius=S(16), fill=C_PANEL)
    d.rounded_rectangle([x0, y0, x1, y1], radius=S(16),
                        outline=(col[0], col[1], col[2], 120), width=2)
    d.rounded_rectangle([x0, y0, x0 + S(6), y1], radius=S(3), fill=col)

    tx = x0 + S(22)
    maxw = x1 - tx - S(20)
    if sub:
        d.text((tx, y0 + S(12)), sub.upper(), font=_font(S(11)), fill=col)
        lines = _wrap(d, text, _font(S(19)), maxw)[:1]
        d.text((tx, y0 + S(30)), lines[0], font=_font(S(19)), fill=C_TEXT)
    else:
        d.text((tx, y0 + S(24)), text, font=_font(S(21)), fill=C_TEXT)

    if in_hold:
        bw = S(64)
        _pill(d, [x1 - bw - S(12), y0 + S(12), x1 - S(12), y0 + S(36)],
              (C_OK[0], C_OK[1], C_OK[2], 40), radius=S(12))
        d.text((x1 - bw - S(2), y0 + S(16)), "HOLD", font=_font(S(13), mono=True), fill=C_OK)
