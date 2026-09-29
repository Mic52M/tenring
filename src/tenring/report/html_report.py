"""Build a self-contained HTML session report with charts.

Turns the per-frame log + detected shots into an at-a-glance review: overall
verdict, the top faults to work on, posture timeline, fault distribution,
stability (tremor) and shot-to-shot consistency. Charts are matplotlib PNGs
embedded as base64 so the report is a single portable .html file.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

_BG = "#0f1216"
_FG = "#e6e6e6"
_OK, _WARN, _BAD = "#4fc36b", "#f2c14e", "#e8503a"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax):
    ax.set_facecolor(_BG)
    for s in ax.spines.values():
        s.set_color("#3a3f47")
    ax.tick_params(colors=_FG, labelsize=8)
    ax.yaxis.label.set_color(_FG)
    ax.xaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    ax.grid(True, color="#242830", linewidth=0.6)


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
        ax.plot(t, vals, color="#5aa9e6", linewidth=1.2)
        ax.set_ylabel(METRIC_LABELS[k], fontsize=8)
        _style(ax)
    axes[-1].set_xlabel("tempo (s) — solo fasi in mira")
    axes[0].set_title("Timeline postura (fasi di mira)")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_stability(records: list, ref: dict) -> Optional[str]:
    pts = [(r["t"], r["wrist_jitter"]) for r in records
           if r.get("wrist_jitter") is not None]
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
    ax.set_title("Stabilità (tremore in mira, piano frontale) — più basso è meglio")
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


def _consistency_faults(summary, n: int = 3) -> list:
    """Rank metrics by shot-to-shot spread relative to their notable std."""
    if not summary or summary.n_shots < 2:
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
    if not summary or summary.n_shots < 2:
        return ("Servono più colpi per il verdetto", _WARN, 0)
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


def build_report(records: list, shots: list, summary, ref: dict,
                 out_path: Path, when: str = "") -> Path:
    dist = _fault_distribution(records)
    mean_jit = summary.mean_wrist_jitter if summary else float("nan")
    verdict, vcolor, score = _verdict_consistency(summary, ref)
    top = _consistency_faults(summary)

    charts = []
    # Per-shot analysis first — the core view for single-shot air pistol.
    sc = _chart_shots(shots, ref)
    if sc:
        charts.append(("", sc))
    scc = _chart_shot_consistency(shots)
    if scc:
        charts.append(("", scc))
    if records:
        charts.append(("", _chart_timeline(records, shots)))
    st_chart = _chart_stability(records, ref)
    if st_chart:
        charts.append(("", st_chart))
    if dist:
        charts.append(("", _chart_distribution(dist)))

    # consistency table
    cons_rows = ""
    if summary:
        for k, label in METRIC_LABELS.items():
            std = summary.consistency.get(k, float("nan"))
            mean = summary.means.get(k, float("nan"))
            if std == std:  # not NaN
                cons_rows += (f"<tr><td>{label}</td><td>{mean:.2f}</td>"
                              f"<td>{std:.2f}</td></tr>")
    n_shots = summary.n_shots if summary else 0

    if summary and summary.n_shots >= 2:
        top_html = "".join(
            f'<li><b>{name}</b> — poco ripetibile: varia di ±{std:.2f} tra i colpi</li>'
            for name, std in top) or "<li>Ottima ripetibilità colpo-su-colpo 👌</li>"
    else:
        top_html = ("<li>Servono almeno 2 colpi per l'analisi di ripetibilità. "
                    "Esegui il ciclo: alza il braccio, mira, spara, abbassa.</li>")

    # Metrics never reliably in frame -> not judged (honest reporting).
    not_eval = [label for k, label in METRIC_LABELS.items() if k not in dist]
    if not_eval:
        noteval_html = ('<h2>Non valutato (fuori inquadratura)</h2>'
                        '<p class="muted">Queste parti non erano ben inquadrate, '
                        'quindi non sono state giudicate: <b>'
                        + ", ".join(not_eval) + '</b>. '
                        'Allarga l\'inquadratura per includerle.</p>')
    else:
        noteval_html = ""

    charts_html = "".join(
        f'<div class="card"><img src="data:image/png;base64,{b64}"/></div>'
        for _, b64 in charts)

    html = _TEMPLATE.format(
        when=when, verdict=verdict, vcolor=vcolor, score=int(score),
        n_frames=len(records), n_shots=n_shots,
        mean_jit=("n/d" if mean_jit != mean_jit else f"{mean_jit:.4f}"),
        top_html=top_html, charts_html=charts_html, noteval_html=noteval_html,
        cons_rows=cons_rows or '<tr><td colspan="3">Nessun colpo rilevato</td></tr>',
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


_TEMPLATE = """<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tenring — resoconto sessione</title>
<style>
:root{{--bg:#0f1216;--fg:#e6e6e6;--muted:#9aa3ad;--card:#171b21;--line:#2a2f37}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.5}}
.wrap{{max-width:860px;margin:0 auto;padding:24px 16px 60px}}
h1{{font-size:22px;margin:0 0 4px}} .muted{{color:var(--muted);font-size:13px}}
.hero{{display:flex;gap:18px;align-items:center;background:var(--card);
border:1px solid var(--line);border-radius:14px;padding:18px;margin:18px 0}}
.score{{font-size:40px;font-weight:700}}
.verdict{{font-size:20px;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:12px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}
.stat .n{{font-size:22px;font-weight:700}} .stat .l{{color:var(--muted);font-size:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:12px;margin:14px 0}} .card img{{width:100%;display:block;border-radius:8px}}
h2{{font-size:15px;margin:22px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
ul{{margin:6px 0}} li{{margin:4px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);
border:1px solid var(--line);border-radius:12px;overflow:hidden}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase}}
tr:last-child td{{border-bottom:none}}
.foot{{color:var(--muted);font-size:12px;margin-top:30px}}
</style></head><body><div class="wrap">
<h1>tenring — resoconto sessione</h1>
<div class="muted">Pistola ad aria compressa 10 m · {when}</div>
<div class="hero">
  <div class="score" style="color:{vcolor}">{score}</div>
  <div><div class="verdict" style="color:{vcolor}">{verdict}</div>
  <div class="muted">punteggio 0–100 (ripetibilità colpo-su-colpo + stabilità)</div></div>
</div>
<div class="grid">
  <div class="stat"><div class="n">{n_shots}</div><div class="l">colpi rilevati</div></div>
  <div class="stat"><div class="n">{n_frames}</div><div class="l">frame in posizione</div></div>
  <div class="stat"><div class="n">{mean_jit}</div><div class="l">tremore medio (hold)</div></div>
</div>
<h2>Su cosa lavorare (ripetibilità)</h2><ul>{top_html}</ul>
{noteval_html}
<h2>Grafici</h2>{charts_html}
<h2>Ripetibilità colpo-su-colpo</h2>
<table><tr><th>Metrica</th><th>Media</th><th>Dev. std (↓ meglio)</th></tr>{cons_rows}</table>
<div class="foot">Generato da tenring · valutazione sulla TUA postura calibrata (se presente),
con guardrail dalla teoria del tiro 10 m (docs/THEORY.md). Solo fasi in mira.
Il tremore è misurato nel piano frontale (con una camera sola non si misura la
profondità). Non misura la mira/mirino (dominio SCATT): analizza il corpo.</div>
</div></body></html>"""
