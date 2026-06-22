#!/usr/bin/env python3
"""make_figures.py — render the two write-up figures from the raw JSONL.

Fig 1: the 4-model DISTANCE curves (reliability) vs context-fill, with the NEAR
       control (capability) overlaid — the headline coordinate-system plot.
Fig 2: lost-in-the-middle — distance success by needle position (start/middle/end)
       at matched fill, for the two models that were position-swept.

Error bars are **run-clustered** bootstrap 95% CIs — NOT needle-level Wilson — so the plots are
no more confident than the analysis (review #6). Same method, B, and seed-family as bootstrap_ci.py:
B = bootstrap_ci.B (5000) resamples; seed = bootstrap_ci.SEED (20260622) for the distance panel,
SEED+1 for the position panel. NOTE: these are statistically equivalent to bootstrap_ci.py but NOT
byte-identical resample draws (the two scripts consume the RNG in a different order) — that is
intentional; we document B/seed rather than thread a shared resample sequence.
Both loaders read an explicit manifest (canonical_manifest.txt / posweep_manifest.txt) and
skip provider=="mock" records, so a stray offline run can't contaminate either figure (#1/#5).
Distance points with no matched near cell (±15% ctx) are ringed (review #5).
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

import matplotlib

from bootstrap_ci import SEED as BOOT_SEED
from bootstrap_ci import _rand, cluster_boot, load_runs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["gpt-3.5-turbo", "gpt-4o-mini", "claude-haiku-4-5-20251001", "claude-sonnet-4-6"]
DISPLAY = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}
COLOR = {
    "gpt-3.5-turbo": "#d62728",
    "gpt-4o-mini": "#ff7f0e",
    "claude-haiku-4-5-20251001": "#1f77b4",
    "claude-sonnet-4-6": "#2ca02c",
}
# (ctx, point, lo, hi, n_runs, n_needles)
Cell = tuple[float, float, float, float, int, int]


def posweep_files() -> list[str]:
    """Manifest-or-glob for the position-sweep files (mirrors analyze_curves.canonical_files)."""
    manifest = Path("posweep_manifest.txt")
    if manifest.exists():
        names = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        return [f for f in names if Path(f).exists()]
    return sorted(glob.glob("posweep_*.jsonl"))


def load_distance() -> dict[str, dict[str, list[Cell]]]:
    """model -> condition -> sorted [(ctx, point, lo95, hi95, n_runs, n_needles)] with
    RUN-CLUSTERED bootstrap CIs (resamples whole runs; needles within a run aren't independent)."""
    cells = load_runs()  # (model, cond, fill) -> [{"outcomes":[...], "ctx":int}, ...]
    rng = _rand(BOOT_SEED)  # B=bootstrap_ci.B (5000), seed=BOOT_SEED; equivalent to bootstrap_ci, not identical draws
    out: dict[str, dict[str, list[Cell]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for key in sorted(cells):  # fixed order => deterministic bootstrap draws
        model, cond, _fill = key
        runs = cells[key]
        pt, lo, hi, nr, nn = cluster_boot(runs, rng)
        ctx = sum(r["ctx"] for r in runs) / len(runs) if runs else 0.0
        out[model][cond].append((ctx, pt, lo, hi, nr, nn))
    for model in out:
        for cond in out[model]:
            out[model][cond].sort()
    return out


def load_positions() -> dict[tuple[str, int], dict[float, Cell]]:
    """(model, fill) -> depth -> run-clustered Cell, from the position sweeps (distance only)."""
    runs: dict[tuple[str, int, float], list[dict]] = collections.defaultdict(list)
    for f in posweep_files():
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("provider") == "mock" or r["condition"] != "distance":
                continue
            key = (r["model"], r["fill_target"], float(r["depth"]))
            runs[key].append({"outcomes": [n["outcome"] for n in r["needles"]],
                              "ctx": r.get("ctx_tokens") or r["fill_target"]})
    rng = _rand(BOOT_SEED + 1)  # position panel: seed=BOOT_SEED+1, same B (5000)
    out: dict[tuple[str, int], dict[float, Cell]] = collections.defaultdict(dict)
    for key in sorted(runs):
        model, fill, depth = key
        pt, lo, hi, nr, nn = cluster_boot(runs[key], rng)
        ctx = sum(r["ctx"] for r in runs[key]) / len(runs[key])
        out[(model, fill)][depth] = (ctx, pt, lo, hi, nr, nn)
    return out


def contour(points: list[tuple[float, float]], thr: float) -> float | None:
    for (c0, s0), (c1, s1) in zip(points, points[1:]):
        if s0 >= thr > s1:
            frac = (s0 - thr) / (s0 - s1) if s0 != s1 else 0.0
            return c0 + frac * (c1 - c0)
    return None


def _matched(c: float, near_ctx: list[float]) -> bool:
    return any(abs(c - nc) <= 0.15 * max(c, 1.0) for nc in near_ctx)


def fig_curves(data: dict[str, dict[str, list[Cell]]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    models = [m for m in ORDER if m in data]
    for m in models:
        dist = data[m].get("distance", [])
        near = data[m].get("near", [])
        if not dist:
            continue
        r90 = contour([(c, pt) for c, pt, *_ in dist], 0.90)
        r90txt = f"R90≈{r90 / 1000:.0f}k" if r90 else "R90>max"
        xs = [c for c, *_ in dist]
        ys = [pt for _c, pt, *_ in dist]
        yerr = [[pt - lo for _c, pt, lo, _hi, _nr, _nn in dist],
                [hi - pt for _c, pt, _lo, hi, _nr, _nn in dist]]
        ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color=COLOR[m], lw=2.2, ms=6,
                    capsize=2.5, elinewidth=1.0,
                    label=f"{DISPLAY[m]}  ({r90txt})  — distance")
        if near:
            ax.plot([c for c, *_ in near], [pt for _c, pt, *_ in near],
                    "--", color=COLOR[m], lw=1.1, alpha=0.45)
            near_ctx = [c for c, *_ in near]
            unx = [c for c, *_ in dist if not _matched(c, near_ctx)]
            uny = [pt for c, pt, *_ in dist if not _matched(c, near_ctx)]
            if unx:
                ax.scatter(unx, uny, facecolors="none", edgecolors=COLOR[m], s=150,
                           linewidths=1.6, zorder=5)
    ax.scatter([], [], facecolors="none", edgecolors="grey", s=150, linewidths=1.6,
               label="○ distance point with NO matched near cell")
    ax.axhline(0.90, color="grey", ls=":", lw=1, alpha=0.7)
    ax.text(1100, 0.905, "R90 threshold (0.90)", color="grey", fontsize=8, va="bottom")
    ax.axhline(0.50, color="grey", ls=":", lw=1, alpha=0.5)
    ax.text(1100, 0.505, "0.50", color="grey", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1000, 1_100_000)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Context fill (measured input tokens, log scale)")
    ax.set_ylabel("Success rate")
    ax.set_title("Reliability vs. context-fill: effective reliable length spans ~54× "
                 "(R90, ordinal)\nsolid = distance (±run-clustered 95% CI)   ·   "
                 "dashed = near control (≥0.97, not 1.00)", fontsize=11.5)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax.grid(True, which="both", ls="-", alpha=0.13)
    fig.tight_layout()
    fig.savefig("fig1_reliability_curves.png", dpi=150)
    print("wrote fig1_reliability_curves.png")


def fig_position(pos: dict[tuple[str, int], dict[float, Cell]]) -> None:
    series = [
        ("claude-haiku-4-5-20251001", 120000, "Haiku 4.5 @120k (77% win)", "#1f77b4"),
        ("claude-haiku-4-5-20251001", 150000, "Haiku 4.5 @150k (96% win)*", "#9ecae1"),
        ("gpt-4o-mini", 55000, "gpt-4o-mini @55k", "#ff7f0e"),
        ("gpt-4o-mini", 85000, "gpt-4o-mini @85k", "#fdae6b"),
    ]
    depths = [0.1, 0.5, 0.9]
    labels = ["Start\n(0.1)", "Middle\n(0.5)", "End\n(0.9)"]
    series = [s for s in series if (s[0], s[1]) in pos]
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 5.6))
    x0 = list(range(len(depths)))
    nan = float("nan")
    for i, (model, fill, lab, col) in enumerate(series):
        cell = pos[(model, fill)]
        cd = [cell.get(d) for d in depths]
        ys = [c[1] if c else nan for c in cd]
        elo = [(c[1] - c[2]) if c else 0.0 for c in cd]
        ehi = [(c[3] - c[1]) if c else 0.0 for c in cd]
        xs = [x + (i - (n - 1) / 2) * width for x in x0]
        bars = ax.bar(xs, ys, width=width * 0.95, color=col, label=lab,
                      yerr=[elo, ehi], capsize=2, error_kw={"elinewidth": 0.8})
        for rect, y in zip(bars, ys):
            if y == y:  # not NaN
                ax.text(rect.get_x() + rect.get_width() / 2, y + 0.05,
                        f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x0)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Distance success rate")
    ax.set_xlabel("Needle position in context")
    ax.set_title("Lost-in-the-middle: at matched fill, the MIDDLE degrades most "
                 "(±run-clustered 95% CI)\n(general across both providers; *Haiku @150k middle "
                 "is window-edge contaminated)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower center", ncol=2)
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig("fig2_lost_in_the_middle.png", dpi=150)
    print("wrote fig2_lost_in_the_middle.png")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    data = load_distance()
    print("DISTANCE cells (model: [(ctx, point, lo, hi, n_runs)...]) — run-clustered:")
    for m in ORDER:
        if m in data and data[m].get("distance"):
            pts = [(round(c), round(pt, 2), round(lo, 2), round(hi, 2), nr)
                   for c, pt, lo, hi, nr, _nn in data[m]["distance"]]
            print(f"  {DISPLAY.get(m, m)}: {pts}")
    fig_curves(data)
    pos = load_positions()
    print("\nPOSITION cells (model, fill): depth -> point [lo,hi]:")
    for (model, fill), cell in sorted(pos.items()):
        row = {d: (round(v[1], 2), round(v[2], 2), round(v[3], 2)) for d, v in sorted(cell.items())}
        print(f"  {DISPLAY.get(model, model)} @{fill}: {row}")
    fig_position(pos)


if __name__ == "__main__":
    main()
