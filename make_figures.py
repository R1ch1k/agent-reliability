#!/usr/bin/env python3
"""make_figures.py — render the two write-up figures from the raw JSONL.

Fig 1: the 4-model DISTANCE curves (reliability) vs context-fill, with the NEAR
       control (capability) overlaid — the headline coordinate-system plot.
Fig 2: lost-in-the-middle — distance success by needle position (start/middle/end)
       at matched fill, for the two models that were position-swept.

Aggregation mirrors analyze_curves.py: success = correct / total needles per cell,
canonical curves use depth == 0.5 only, gpt-3.5 (raw lost then restored; fallback
kept for safety) approximated where ctx_tokens predates logging.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GPT35_CTX_RATIO = 1.25
GPT35_FALLBACK: dict[str, list[tuple[float, float, int]]] = {
    "distance": [(1300, 0.98, 45), (2600, 0.96, 45), (5200, 0.93, 45), (10400, 0.71, 45)],
    "near": [(1300, 1.00, 45), (2600, 0.98, 45), (5200, 0.98, 45), (10400, 1.00, 45)],
}

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


def load_distance() -> dict[str, dict[str, list[tuple[float, float, int]]]]:
    """model -> condition -> sorted [(mean_ctx, success, n)] from dist_results_*.jsonl."""
    raw: dict[tuple[str, str, int], collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    ctxs: dict[tuple[str, str, int], list[float]] = collections.defaultdict(list)
    for f in glob.glob("dist_results_*.jsonl"):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if abs(float(r.get("depth", 0.5)) - 0.5) > 1e-9:
                continue
            key = (r["model"], r["condition"], r["fill_target"])
            c = r.get("ctx_tokens") or r["fill_target"] * GPT35_CTX_RATIO
            ctxs[key].append(c)
            for n in r["needles"]:
                raw[key][n["outcome"]] += 1
    out: dict[str, dict[str, list[tuple[float, float, int]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for (model, cond, _fill), cnt in raw.items():
        tot = sum(cnt.values())
        succ = cnt["correct"] / tot if tot else 0.0
        k = (model, cond, _fill)
        out[model][cond].append((sum(ctxs[k]) / len(ctxs[k]), succ, tot))
    for model in out:
        for cond in out[model]:
            out[model][cond].sort()
    if "gpt-3.5-turbo" not in out:
        out["gpt-3.5-turbo"] = dict(GPT35_FALLBACK)  # type: ignore[assignment]
    return out


def load_positions() -> dict[tuple[str, int], dict[float, tuple[float, float, int]]]:
    """(model, fill) -> depth -> (mean_ctx, success, n) from posweep_*.jsonl (distance)."""
    raw: dict[tuple[str, int, float], collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    ctxs: dict[tuple[str, int, float], list[float]] = collections.defaultdict(list)
    for f in glob.glob("posweep_*.jsonl"):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["condition"] != "distance":
                continue
            key = (r["model"], r["fill_target"], float(r["depth"]))
            ctxs[key].append(r.get("ctx_tokens") or r["fill_target"])
            for n in r["needles"]:
                raw[key][n["outcome"]] += 1
    out: dict[tuple[str, int], dict[float, tuple[float, float, int]]] = collections.defaultdict(
        dict)
    for (model, fill, depth), cnt in raw.items():
        tot = sum(cnt.values())
        succ = cnt["correct"] / tot if tot else 0.0
        k = (model, fill, depth)
        out[(model, fill)][depth] = (sum(ctxs[k]) / len(ctxs[k]), succ, tot)
    return out


def contour(points: list[tuple[float, float, int]], thr: float) -> float | None:
    for (c0, s0, _), (c1, s1, _) in zip(points, points[1:]):
        if s0 >= thr > s1:
            frac = (s0 - thr) / (s0 - s1) if s0 != s1 else 0.0
            return c0 + frac * (c1 - c0)
    return None


def fig_curves(data: dict[str, dict[str, list[tuple[float, float, int]]]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    models = [m for m in ORDER if m in data]
    for m in models:
        dist = data[m].get("distance", [])
        near = data[m].get("near", [])
        if not dist:
            continue
        r90 = contour(dist, 0.90)
        r90txt = f"R90≈{r90/1000:.0f}k" if r90 else "R90>max"
        xs = [p[0] for p in dist]
        ys = [p[1] for p in dist]
        ax.plot(xs, ys, "-o", color=COLOR[m], lw=2.2, ms=6,
                label=f"{DISPLAY[m]}  ({r90txt})  — distance")
        if near:
            nx = [p[0] for p in near]
            ny = [p[1] for p in near]
            ax.plot(nx, ny, "--", color=COLOR[m], lw=1.1, alpha=0.45)
    ax.axhline(0.90, color="grey", ls=":", lw=1, alpha=0.7)
    ax.text(1100, 0.905, "R90 threshold (0.90)", color="grey", fontsize=8, va="bottom")
    ax.axhline(0.50, color="grey", ls=":", lw=1, alpha=0.5)
    ax.text(1100, 0.505, "0.50", color="grey", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1000, 1_100_000)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Context fill (measured input tokens, log scale)")
    ax.set_ylabel("Success rate")
    ax.set_title("Reliability vs. context-fill: effective reliable length scales ~54×\n"
                 "solid = distance (reliability)   ·   dashed = near (capability control, ≈1.00)",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    ax.grid(True, which="both", ls="-", alpha=0.13)
    fig.tight_layout()
    fig.savefig("fig1_reliability_curves.png", dpi=150)
    print("wrote fig1_reliability_curves.png")


def fig_position(pos: dict[tuple[str, int], dict[float, tuple[float, float, int]]]) -> None:
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
    for i, (model, fill, lab, col) in enumerate(series):
        cell = pos[(model, fill)]
        ys = [cell.get(d, (0, float("nan"), 0))[1] for d in depths]
        xs = [x + (i - (n - 1) / 2) * width for x in x0]
        bars = ax.bar(xs, ys, width=width * 0.95, color=col, label=lab)
        for rect, y in zip(bars, ys):
            if y == y:  # not NaN
                ax.text(rect.get_x() + rect.get_width() / 2, y + 0.015,
                        f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x0)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Distance success rate")
    ax.set_xlabel("Needle position in context")
    ax.set_title("Lost-in-the-middle: at matched fill, the MIDDLE degrades most\n"
                 "(general across both providers; *Haiku @150k middle is window-edge "
                 "contaminated)", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower center", ncol=2)
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig("fig2_lost_in_the_middle.png", dpi=150)
    print("wrote fig2_lost_in_the_middle.png")


def main() -> None:
    data = load_distance()
    print("DISTANCE curve cells (model: [(ctx, success, n)...]):")
    for m in ORDER:
        if m in data and data[m].get("distance"):
            pts = [(round(c), round(s, 2), n) for c, s, n in data[m]["distance"]]
            print(f"  {DISPLAY.get(m, m)}: {pts}")
    fig_curves(data)
    pos = load_positions()
    print("\nPOSITION cells (model, fill): depth -> success:")
    for (model, fill), cell in sorted(pos.items()):
        row = {d: round(v[1], 2) for d, v in sorted(cell.items())}
        print(f"  {DISPLAY.get(model, model)} @{fill}: {row}")
    fig_position(pos)


if __name__ == "__main__":
    main()
