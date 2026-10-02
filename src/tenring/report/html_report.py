"""Build a self-contained HTML session report with charts.

Turns the per-frame log + detected shots into an at-a-glance review: overall
verdict, the top faults to work on, posture timeline, fault distribution,
stability (tremor) and shot-to-shot consistency. Charts are matplotlib PNGs
embedded as base64 so the report is a single portable .html file.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Clean, consistent chart typography to match the dashboard.
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
})

METRIC_LABELS = {
    "torso_lean": "Inclinazione busto",
    "shoulder_elevation": "Spalla arma",
    "arm_extension": "Estensione braccio",
    "wrist_alignment": "Polso",
    "head_tilt": "Testa",
    "stance_width": "Apertura piedi",
    "weight_balance": "Bilanciamento",
}
ANGLE_KEYS = ["torso_lean", "arm_extension", "wrist_alignment", "head_tilt"]

_BG = "#141821"       # matches the card background for seamless charts
_FG = "#e6e6e6"
_GRID = "#232a35"
_OK, _WARN, _BAD = "#4ade80", "#facc15", "#f87171"
_ACCENT = "#38bdf8"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax):
    ax.set_facecolor(_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#3a4150")
    ax.tick_params(colors="#9aa3ad", labelsize=8, length=3)
    ax.yaxis.label.set_color("#9aa3ad")
    ax.xaxis.label.set_color("#9aa3ad")
    ax.title.set_color(_FG)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# --- fault distribution per metric (fraction of frames ok/warn/bad) ----------
def _fault_distribution(records: list) -> dict:
    dist = {}
    for k in METRIC_LABELS:
        counts = {"ok": 0, "warn": 0, "bad": 0}
        for r in records:
            st = r.get("status", {}).get(k)
            if st in counts:
                counts[st] += 1
        total = sum(counts.values())
        if total:
            dist[k] = {s: counts[s] / total for s in counts}
    return dist


def _chart_distribution(dist: dict) -> str:
    keys = [k for k in METRIC_LABELS if k in dist]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), facecolor=_BG)
    y = np.arange(len(keys))
    ok = [dist[k]["ok"] * 100 for k in keys]
    warn = [dist[k]["warn"] * 100 for k in keys]
    bad = [dist[k]["bad"] * 100 for k in keys]
    ax.barh(y, ok, color=_OK, label="OK")
    ax.barh(y, warn, left=ok, color=_WARN, label="Attenzione")
    ax.barh(y, bad, left=np.add(ok, warn), color=_BAD, label="Errore")
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_LABELS[k] for k in keys])
    ax.set_xlabel("% del tempo")
    ax.set_xlim(0, 100)
    ax.set_title("Distribuzione difetti per metrica")
    ax.invert_yaxis()
    _style(ax)
    ax.legend(facecolor=_BG, edgecolor="#3a3f47", labelcolor=_FG, fontsize=7, loc="lower right")
    return _fig_to_b64(fig)


def _chart_timeline(records: list, shots: list) -> str:
    t = [r["t"] for r in records]
    keys = [k for k in ANGLE_KEYS if any(r.get(k) is not None for r in records)]
    if not keys:
        keys = ["torso_lean"]
    shot_ts = [s.t_end for s in shots] if shots else []
    # shots carry absolute time; align to the record timeline origin
    t0_abs = shot_ts and min(shot_ts) or 0
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.2, 1.5 * len(keys)),
                             facecolor=_BG, sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        vals = [r.get(k) if r.get(k) is not None else np.nan for r in records]
        ax.plot(t, vals, color=_ACCENT, linewidth=1.3)
        ax.set_ylabel(METRIC_LABELS[k], fontsize=8)
        _style(ax)
    axes[-1].set_xlabel("tempo (s) — solo fasi in mira")
    axes[0].set_title("Timeline postura (fasi di mira)")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_stability(records: list, ref: dict) -> Optional[str]:
    # Only the HOLD phase is real aiming stability; the rest is the arm still
    # rising/settling and would wildly inflate the tremor.
    pts = [(r["t"], r["wrist_jitter"]) for r in records
           if r.get("wrist_jitter") is not None and r.get("in_hold")]
    if len(pts) < 5:
        return None
    t, jit = zip(*pts)
    s = ref["stability"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), facecolor=_BG)
    ax.plot(t, jit, color="#b07cf0", linewidth=1.1)
    ax.axhline(s["wrist_jitter_warn"], color=_WARN, linestyle="--", linewidth=1)
    ax.axhline(s["wrist_jitter_bad"], color=_BAD, linestyle="--", linewidth=1)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("tremore polso")
    ax.set_title("Stabilità in HOLD (tremore, piano frontale) — più basso è meglio")
    _style(ax)
    return _fig_to_b64(fig)


def _verdict(dist: dict, mean_jitter: float, ref: dict) -> tuple:
    """Return (voto_testo, colore, punteggio_0_100)."""
    if not dist:
        return ("Dati insufficienti", _WARN, 0)
    bad = np.mean([dist[k]["bad"] for k in dist])
    warn = np.mean([dist[k]["warn"] for k in dist])
    score = max(0.0, 100.0 * (1.0 - (bad * 1.0 + warn * 0.4)))
    if score >= 80:
        return ("Ottima postura", _OK, score)
    if score >= 60:
        return ("Buona, con margini", _WARN, score)
    return ("Da migliorare", _BAD, score)


def _top_faults(dist: dict, n: int = 3) -> list:
    ranked = sorted(dist.items(),
                    key=lambda kv: kv[1]["bad"] * 2 + kv[1]["warn"], reverse=True)
    out = []
    for k, d in ranked[:n]:
        if d["bad"] + d["warn"] < 0.05:
            continue
        pct = round((d["bad"] + d["warn"]) * 100)
        out.append((METRIC_LABELS[k], pct))
    return out


# Per-metric "notable" shot-to-shot spread: std at/above this = worth attention.
# Lets us compare degrees vs ratios on one scale (std / notable).
NOTABLE_STD = {
    "torso_lean": 3.0, "shoulder_elevation": 0.05, "arm_extension": 5.0,
    "wrist_alignment": 5.0, "head_tilt": 4.0, "stance_width": 0.10,
    "weight_balance": 0.05,
}


def _chart_shots(shots: list, ref: dict) -> Optional[str]:
    if not shots:
        return None
    idx = list(range(1, len(shots) + 1))
    jit = [s.wrist_jitter for s in shots]
    dur = [s.duration for s in shots]
    s = ref["stability"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.4), facecolor=_BG, sharex=True)
    ax1.bar(idx, jit, color="#b07cf0")
    ax1.axhline(s["wrist_jitter_warn"], color=_WARN, linestyle="--", linewidth=1)
    ax1.axhline(s["wrist_jitter_bad"], color=_BAD, linestyle="--", linewidth=1)
    ax1.set_ylabel("tremore")
    ax1.set_title("Per colpo: tremore in hold e durata")
    _style(ax1)
    ax2.bar(idx, dur, color="#5aa9e6")
    ax2.set_ylabel("hold (s)")
    ax2.set_xlabel("colpo #")
    ax2.set_xticks(idx)
    _style(ax2)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_shot_consistency(shots: list) -> Optional[str]:
    if len(shots) < 2:
        return None
    keys = ["torso_lean", "shoulder_elevation", "arm_extension", "wrist_alignment"]
    idx = list(range(1, len(shots) + 1))
    fig, ax = plt.subplots(figsize=(7.2, 2.9), facecolor=_BG)
    plotted = False
    for k in keys:
        vals = np.array([s.metrics_mean.get(k, np.nan) for s in shots], float)
        if np.all(np.isnan(vals)):
            continue
        mean = np.nanmean(vals)
        # normalise by the metric's notable std so all metrics share one y-scale
        norm = (vals - mean) / NOTABLE_STD.get(k, 1.0)
        ax.plot(idx, norm, marker="o", linewidth=1.2, label=METRIC_LABELS[k])
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.axhline(0, color="#666", linewidth=0.8)
    ax.axhspan(-1, 1, color="#4fc36b", alpha=0.08)  # "consistent" band
    ax.set_xlabel("colpo #")
    ax.set_ylabel("scarto dal tuo assetto\n(unità di consistenza)")
    ax.set_title("Ripetibilità colpo-su-colpo (dentro la fascia verde = costante)")
    ax.set_xticks(idx)
    _style(ax)
    ax.legend(facecolor=_BG, edgecolor="#3a3f47", labelcolor=_FG, fontsize=7)
    return _fig_to_b64(fig)


MIN_SHOTS = 3  # fewer than this is too weak to judge repeatability


def _consistency_faults(summary, n: int = 3) -> list:
    """Rank metrics by shot-to-shot spread relative to their notable std."""
    if not summary or summary.n_shots < MIN_SHOTS:
        return []
    ranked = []
    for k, label in METRIC_LABELS.items():
        std = summary.consistency.get(k, float("nan"))
        if std != std:  # NaN
            continue
        ratio = std / NOTABLE_STD.get(k, 1.0)
        ranked.append((label, std, ratio))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return [(label, std) for label, std, ratio in ranked[:n] if ratio >= 1.0]


def _verdict_consistency(summary, ref: dict) -> tuple:
    """Verdict from repeatability + stability (the reliable signals)."""
    if not summary or summary.n_shots < MIN_SHOTS:
        return (f"Servono almeno {MIN_SHOTS} colpi per il verdetto", _WARN, 0)
    # consistency: 1 = every metric well within its notable std
    ratios = []
    for k in METRIC_LABELS:
        std = summary.consistency.get(k, float("nan"))
        if std == std:
            ratios.append(min(1.0, NOTABLE_STD.get(k, 1.0) / max(std, 1e-6)))
    consistency = float(np.mean(ratios)) if ratios else 0.0
    # stability: from mean tremor vs thresholds
    s = ref["stability"]
    jit = summary.mean_wrist_jitter
    if jit != jit:
        stability = consistency
    else:
        stability = float(np.clip(
            1.0 - (jit - s["wrist_jitter_warn"]) /
            max(s["wrist_jitter_bad"] - s["wrist_jitter_warn"], 1e-6), 0.0, 1.0))
        stability = 1.0 if jit <= s["wrist_jitter_warn"] else stability
    score = 100.0 * (0.6 * consistency + 0.4 * stability)
    if score >= 80:
        return ("Molto costante", _OK, score)
    if score >= 60:
        return ("Costante, con margini", _WARN, score)
    return ("Ripetibilità da migliorare", _BAD, score)


def _fnum(x) -> Optional[float]:
    """NaN/None -> None (JSON null); else rounded float."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if xf != xf else round(xf, 3)


def _shots_json(shots: list) -> list:
    out = []
    for i, s in enumerate(shots):
        out.append({
            "n": i + 1,
            "total": _fnum(s.total_time),
            "rise": _fnum(s.time_to_hold),
            "hold": _fnum(s.hold_duration),
            "descent": _fnum(s.descent_time),
            "tremor": _fnum(s.wrist_jitter),
            "peak": _fnum(s.raise_peak),
            "settle": _fnum(s.settle_elev),
            "metrics": {METRIC_LABELS[k]: _fnum(v)
                        for k, v in s.metrics_mean.items() if k in METRIC_LABELS},
        })
    return out


def build_report(records: list, shots: list, summary, ref: dict,
                 out_path: Path, when: str = "") -> Path:
    dist = _fault_distribution(records)
    mean_jit = summary.mean_wrist_jitter if summary else float("nan")
    verdict, vcolor, score = _verdict_consistency(summary, ref)
    top = _consistency_faults(summary)
    n_shots = summary.n_shots if summary else 0

    # ---- overview (aggregate) charts ----
    charts = []
    for ch in (_chart_shots(shots, ref), _chart_shot_consistency(shots)):
        if ch:
            charts.append(ch)
    if records:
        charts.append(_chart_timeline(records, shots))
    st_chart = _chart_stability(records, ref)
    if st_chart:
        charts.append(st_chart)
    if dist:
        charts.append(_chart_distribution(dist))
    charts_html = "".join(
        f'<div class="card"><img src="data:image/png;base64,{b64}"/></div>'
        for b64 in charts)

    # consistency table
    cons_rows = ""
    if summary:
        for k, label in METRIC_LABELS.items():
            std = summary.consistency.get(k, float("nan"))
            mean = summary.means.get(k, float("nan"))
            if std == std:
                cons_rows += (f"<tr><td>{label}</td><td>{mean:.2f}</td>"
                              f"<td>{std:.2f}</td></tr>")
    cons_rows = cons_rows or '<tr><td colspan="3">Nessun colpo rilevato</td></tr>'

    if summary and summary.n_shots >= MIN_SHOTS:
        top_html = "".join(
            f'<li><b>{name}</b> — poco ripetibile: varia di ±{std:.2f} tra i colpi</li>'
            for name, std in top) or "<li>Ottima ripetibilità colpo-su-colpo 👌</li>"
    else:
        top_html = (f"<li>Servono almeno {MIN_SHOTS} colpi per l'analisi di "
                    f"ripetibilità (ne ho rilevati {n_shots}). Esegui più cicli: "
                    "alza il braccio, mira, spara, abbassa.</li>")

    not_eval = [label for k, label in METRIC_LABELS.items() if k not in dist]
    noteval_html = ""
    if not_eval:
        noteval_html = ('<h2>Non valutato (fuori inquadratura)</h2>'
                        '<p class="muted">Non erano ben inquadrate, quindi non '
                        'giudicate: <b>' + ", ".join(not_eval) + '</b>.</p>')

    overview_html = (
        f'<h2>Su cosa lavorare (ripetibilità)</h2><ul>{top_html}</ul>'
        f'{noteval_html}'
        f'<h2>Grafici di sessione</h2>{charts_html}'
        f'<h2>Ripetibilità colpo-su-colpo</h2>'
        f'<table><tr><th>Metrica</th><th>Media</th><th>Dev. std (↓ meglio)</th></tr>'
        f'{cons_rows}</table>'
    )

    data = {
        "shots": _shots_json(shots),
        "overview": overview_html,
        "tremor_warn": ref["stability"]["wrist_jitter_warn"],
        "tremor_bad": ref["stability"]["wrist_jitter_bad"],
    }
    mean_jit_s = "n/d" if mean_jit != mean_jit else f"{mean_jit:.4f}"

    shot_buttons = "".join(
        f'<button data-k="{i}">Colpo {i + 1}</button>' for i in range(n_shots))

    html = (_HEAD
            + f'<div class="muted">Pistola ad aria compressa 10 m · {when}</div>'
            + f'<div class="hero"><div class="score" style="color:{vcolor}">{int(score)}</div>'
            + f'<div><div class="verdict" style="color:{vcolor}">{verdict}</div>'
            + '<div class="muted">punteggio 0–100 (ripetibilità + stabilità)</div></div></div>'
            + '<div class="grid">'
            + f'<div class="stat"><div class="n">{n_shots}</div><div class="l">colpi rilevati</div></div>'
            + f'<div class="stat"><div class="n">{len(records)}</div><div class="l">frame in mira</div></div>'
            + f'<div class="stat"><div class="n">{mean_jit_s}</div><div class="l">tremore medio (hold)</div></div>'
            + '</div>'
            + '<div class="layout"><aside class="shotlist" id="shotlist">'
            + '<button data-k="overview" class="active">Panoramica</button>'
            + shot_buttons
            + '</aside><main class="detail" id="detail"></main></div>'
            + _FOOT
            + '<script>const DATA=' + json.dumps(data, ensure_ascii=False) + ';'
            + _JS + '</script></body></html>')
    out_path.write_text(html, encoding="utf-8")
    return out_path


_HEAD = """<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tenring — resoconto sessione</title>
<style>
:root{--bg:#0f1216;--fg:#e6e6e6;--muted:#9aa3ad;--card:#171b21;--line:#2a2f37;--accent:#5aa9e6}
*{box-sizing:border-box}
body{margin:0;background:
radial-gradient(1200px 600px at 80% -10%,rgba(56,189,248,.08),transparent 60%),var(--bg);
color:var(--fg);font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',Roboto,sans-serif;
line-height:1.5;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:900px;margin:0 auto;padding:28px 18px 70px}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.01em;font-weight:700}
h1 b{color:var(--accent)}
.muted{color:var(--muted);font-size:13px}
.hero{position:relative;display:flex;gap:20px;align-items:center;overflow:hidden;
background:linear-gradient(180deg,#191e28,#141820);
border:1px solid var(--line);border-radius:16px;padding:20px 22px;margin:18px 0}
.hero::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--accent)}
.score{font-size:46px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.verdict{font-size:21px;font-weight:650}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:12px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 14px}
.stat .n{font-size:23px;font-weight:700;font-variant-numeric:tabular-nums}
.stat .l{color:var(--muted);font-size:12px;margin-top:2px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin:14px 0}
.card img{width:100%;display:block;border-radius:8px}
h2{font-size:14px;margin:20px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
ul{margin:6px 0}li{margin:4px 0}
table{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);
border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase}
tr:last-child td{border-bottom:none}
.layout{display:grid;grid-template-columns:160px 1fr;gap:16px;margin-top:10px}
.shotlist button{display:block;width:100%;text-align:left;margin:0 0 6px;padding:10px 12px;
border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:10px;
cursor:pointer;font-size:14px;transition:.12s}
.shotlist button:hover{border-color:var(--accent)}
.shotlist button.active{border-color:var(--accent);background:#1c2430;box-shadow:inset 3px 0 0 var(--accent)}
.detail{min-height:200px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.kpi{background:linear-gradient(180deg,#191e28,#141820);border:1px solid var(--line);
border-radius:12px;padding:13px 14px}
.kpi .n{font-size:21px;font-weight:750;font-variant-numeric:tabular-nums}
.kpi .l{color:var(--muted);font-size:12px;margin-top:2px}
.phasebar{display:flex;height:26px;border-radius:7px;overflow:hidden;margin:6px 0 2px;border:1px solid var(--line)}
.phasebar span{display:flex;align-items:center;justify-content:center;font-size:11px;color:#0c0f13;font-weight:600;min-width:0;overflow:hidden;white-space:nowrap}
.legend{font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.foot{color:var(--muted);font-size:12px;margin-top:30px}
@media(max-width:640px){.layout{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1><b>tenring</b> — resoconto sessione</h1>"""

_FOOT = """<div class="foot">Generato da tenring · valutazione sulla TUA postura di
sessione (deviazione dal tuo assetto), con guardrail dalla teoria del tiro 10 m
(docs/THEORY.md). Solo fasi in mira; la discesa post-sparo è esclusa. Tremore nel
piano frontale. Non misura la mira/mirino (dominio SCATT): analizza il corpo.</div>"""

_JS = r"""
const C_RISE="#5aa9e6",C_HOLD="#4fc36b",C_DESC="#9aa3ad";
function s1(x){return x==null?"n/d":x.toFixed(1);}
function s2(x){return x==null?"n/d":x.toFixed(2);}
function s3(x){return x==null?"n/d":x.toFixed(3);}
function tremorColor(t){if(t==null)return"var(--muted)";if(t>DATA.tremor_bad)return"#e8503a";if(t>DATA.tremor_warn)return"#f2c14e";return"#4fc36b";}
function phaseBar(sh){
  const r=sh.rise||0,h=sh.hold||0,d=sh.descent||0,tot=(r+h+d)||1;
  const seg=(w,c,lbl)=>`<span style="flex:${w};background:${c}">${w/tot>0.12?lbl:""}</span>`;
  return `<div class="phasebar">${seg(r,C_RISE,"salita+mira")}${seg(h,C_HOLD,"hold")}${seg(d,C_DESC,"discesa")}</div>
  <div class="legend"><span><i style="background:${C_RISE}"></i>salita+mira ${s1(sh.rise)}s</span>
  <span><i style="background:${C_HOLD}"></i>hold ${s1(sh.hold)}s</span>
  <span><i style="background:${C_DESC}"></i>discesa ${s1(sh.descent)}s</span></div>`;
}
function shotHTML(sh){
  let rows="";for(const k in sh.metrics){rows+=`<tr><td>${k}</td><td>${s2(sh.metrics[k])}</td></tr>`;}
  const above=(sh.peak!=null&&sh.settle!=null)?(sh.peak-sh.settle):null;
  return `<h2>Colpo ${sh.n}</h2>
  <div class="kpis">
    <div class="kpi"><div class="n">${s1(sh.total)}s</div><div class="l">tempo totale colpo</div></div>
    <div class="kpi"><div class="n">${s1(sh.rise)}s</div><div class="l">salita + mira</div></div>
    <div class="kpi"><div class="n">${s1(sh.hold)}s</div><div class="l">durata hold</div></div>
    <div class="kpi"><div class="n">${s1(sh.descent)}s</div><div class="l">discesa</div></div>
    <div class="kpi"><div class="n" style="color:${tremorColor(sh.tremor)}">${s3(sh.tremor)}</div><div class="l">tremore in hold</div></div>
    <div class="kpi"><div class="n">${s1(sh.peak)}°</div><div class="l">picco elevazione braccio</div></div>
    <div class="kpi"><div class="n">${s1(sh.settle)}°</div><div class="l">elevazione in hold</div></div>
    <div class="kpi"><div class="n">${above==null?"n/d":"+"+above.toFixed(1)+"°"}</div><div class="l">salita sopra il bersaglio</div></div>
  </div>
  <h2>Fasi del colpo</h2>${phaseBar(sh)}
  <h2>Postura all'istante più fermo</h2>
  <table><tr><th>Metrica</th><th>Valore</th></tr>${rows}</table>`;
}
function render(k){
  const d=document.getElementById("detail");
  if(k==="overview"){d.innerHTML=DATA.overview;}
  else{d.innerHTML=shotHTML(DATA.shots[+k]);}
  document.querySelectorAll("#shotlist button").forEach(b=>b.classList.toggle("active",b.dataset.k===k));
}
document.getElementById("shotlist").addEventListener("click",e=>{
  const b=e.target.closest("button");if(b)render(b.dataset.k);
});
render("overview");
"""
