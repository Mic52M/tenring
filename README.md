<div align="center">

# tenring

**Real-time posture analysis for 10 m air pistol shooting (ISSF).**

One webcam, no GPU: live feedback while you shoot and a per-shot report afterwards.

![Live HUD](docs/images/hud.png)

</div>

---

## What it is

**tenring** uses computer vision to watch the shooter's *body* — not the sights — and tell
you, shot after shot, where your posture drifts from your own setup and how **repeatable**
you are. In precision shooting the skill isn't a single perfect stance: it's doing the same
thing **every time**.

- Body tracking with [MediaPipe BlazePose](https://developers.google.com/mediapipe) (33 3D keypoints).
- Live traffic-light feedback: torso lean, shoulder, arm, wrist, head, feet, balance, tremor.
- The shot cycle is recognised automatically: *raise the arm, aim, HOLD, fire, lower / reload*.
- Per-shot report: duration of each phase, arm elevation, tremor, repeatability.
- Runs in real time on a **MacBook Air M4**, CPU / Neural Engine only. **No external GPU.**

> It does **not** replace optical trainers like [SCATT](https://www.scatt.com/): those track
> the *sights*. tenring analyses the **body**, which is the uncovered part — they are complementary.

---

## Screenshots

### Session dashboard
Click a shot on the left to see the timing of each phase, peak arm elevation vs the settle on
aim, tremor and the posture at the steadiest instant.

![Dashboard overview](docs/images/dashboard_overview.jpg)

---

## How it works

Per-frame pipeline, ~real-time:

```
webcam -> MediaPipe BlazePose -> geometric metrics -> rules (traffic light)
        -> shot-cycle state machine -> live HUD
                                     -> per-frame log -> HTML report
```

Three choices make the analysis **reliable** with a single webcam:

1. **It detects the shooting arm** — the one raised and extended — instead of trusting a fixed
   setting (robust to right/left-handed shooters and to the mirror effect).
2. **It only judges what it can see** — if the feet aren't in frame, it doesn't grade them
   (no made-up verdicts).
3. **The reference is *your own* session posture**, not textbook absolute angles: 3D angles
   from a frontal webcam drift between sessions, while within a session you are very stable.
   So tenring measures the **deviation from your setup** and your **repeatability** — the
   things that are actually measurable and that matter in shooting.

It also recognises the **shot cycle** with a state machine:

| State | When | Analysed? |
|---|---|---|
| `IDLE` | body not in frame | no |
| `READY` | standing, arm down | no |
| `AIMING` | arm raised and extended | yes |
| `HOLD` | on aim and steady | yes (measures tremor) |
| `RELEASE` | arm coming down after the shot / reload | no |

A **shot** = raise -> steady hold -> lower. The shot's posture is captured at the
**steadiest instant** of the hold.

---

## What it measures

Every metric maps to a section of [`docs/THEORY.md`](docs/THEORY.md), based on ISSF
coaching material and the biomechanics of 10 m air pistol.

| Metric | Typical fault | Ref. |
|---|---|---|
| Torso lean | leaning back too much | §2 |
| Shooting shoulder | shoulder raised/tense or too loose | §3 |
| Arm extension | arm too bent / locked | §3 |
| Wrist | wrist not aligned with the forearm | §4 |
| Head | head tilted / rotated | §5 |
| Stance width | base too narrow/wide | §2 |
| Weight balance | weight shifted onto one foot | §2 |
| Arm elevation | how high you go above the target and how you settle (~90°) | §6b |
| Tremor (hold) | wobble while aiming (frontal plane) | §6 |
| Repeatability | shot-to-shot variance | §7 |

---

## Installation (macOS, Apple Silicon)

```bash
git clone https://github.com/Mic52M/tenring.git
cd tenring

# Recommended: Python 3.11/3.12
conda create -n tenring python=3.12 -y
conda activate tenring
pip install -r requirements.txt
pip install -e .
```

Grant the terminal camera access: **System Settings -> Privacy & Security -> Camera**.

> MediaPipe is pinned to `0.10.21` (the 1.x line has a PoseLandmarker bug on macOS arm64).
> The `.task` model is downloaded automatically on first run.

---

## Usage

Quick start (activates the env and launches fullscreen):

```bash
./run.sh
```

Personal calibration (optional, once — get into your shooting position, arm up):

```bash
./calibra.sh
```

Or, with the `tenring` env active:

```bash
tenring                 # windowed
tenring --fullscreen    # fullscreen, no black bars
tenring --list-cameras  # pick the right webcam (skips the iPhone/Continuity camera)
```

**On-screen keys:** `Q` quit, `S` save a report on the fly.
When the session ends the HTML report opens automatically (in `sessions/`), alongside the raw
`.jsonl` and a `.txt` summary.

| Option | Effect |
|---|---|
| `--camera N` | force a specific webcam |
| `--fullscreen` | fullscreen, no black bars |
| `--no-mirror` | disable the mirror effect (for a side camera) |
| `--width 1280 --height 720` | more fps, less resolution |
| `--complexity {0,1,2}` | pose model: 0 fast ... 2 accurate |

The language of the HUD and report follows `feedback_language` in `config/reference.yaml`
(`it` or `en`).

Recommended MVP setup: **webcam on your left**, full framing **from feet to head**.

---

## Architecture

```
src/tenring/
  pose/        abstract pose-estimation backend (+ MediaPipe BlazePose)
  analysis/    geometry, metrics, rules, state (shot cycle),
               arm (shooting-arm detection), baseline (session reference),
               session, recorder
  ui/          overlay.py  - live HUD (Pillow, San Francisco font)
  report/      html_report.py - interactive per-shot report
  i18n.py      Italian / English strings
  app.py       live loop    ·    calibrate.py  personal calibration
config/
  reference.yaml   thresholds tied to the theory (§ in docs/THEORY.md)
  profile.yaml     (generated) your calibrated neutral
docs/THEORY.md     technical basis of 10 m shooting
```

The pose backend sits behind an abstract interface, so MediaPipe can be swapped for
**RTMPose / YOLO-pose / ViTPose** without touching the analysis.

---

## Roadmap

- [ ] Shot audio for exact shot counting and shot/reload disambiguation.
- [ ] Natural-language coach (LLM) in the report, built from the metrics.
- [ ] Second camera (phase 2) for shoulder squareness and 3D tremor.
- [ ] Cross-session trends (from the saved `.jsonl`).

---

## Tests

```bash
pip install pytest && pytest
```

## License

MIT.

---

<div align="center">
<sub>Personal training project. Not a certified measurement device.</sub>
</div>
