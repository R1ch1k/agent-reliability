#!/usr/bin/env python3
"""bootstrap_ci.py — run-CLUSTERED bootstrap CIs + severity breakdown for the distance probe.

WHY (review findings #5 / #6 / #9): needle outcomes within a run share ONE conversation and
ONE manual, so they are not independent — a needle-level interval (e.g. Wilson on needle count)
is too tight. This resamples whole RUNS (clusters) with replacement, and reports for every cell:
the point success, a 95% bootstrap CI, and BOTH the run count and the needle count. It also
prints the near-control minima (with clustered CI) and the distance-condition severity mix, so
the quantitative claims in the write-up are reproducible from the committed data + this script
rather than from an un-committed subagent analysis.

Deterministic (seeded), pure stdlib. Reads the same canonical manifest as analyze_curves.py.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from analyze_curves import canonical_files

B = 5000
SEED = 20260622
ORDER = ["gpt-3.5-turbo", "gpt-4o-mini", "claude-haiku-4-5-20251001", "claude-sonnet-4-6"]
# severity buckets; legacy data uses flat "wrong", post-#8 data uses fabricated/unclassified_wrong
WRONGISH = ("wrong", "fabricated", "unclassified_wrong")


def _rand(seed: int):
    """Tiny deterministic LCG -> [0,1); avoids any global random-state coupling."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def load_runs() -> dict[tuple[str, str, int], list[dict]]:
    """(model, condition, fill_target) -> list of runs; each run = {outcomes:[...], ctx:int}."""
    cells: dict[tuple[str, str, int], list[dict]] = collections.defaultdict(list)
    for f in canonical_files():
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("provider") == "mock":
                continue
            if abs(float(r.get("depth", 0.5)) - 0.5) > 1e-9:
                continue
            outs = [n["outcome"] for n in r["needles"]]
            cells[(r["model"], r["condition"], r["fill_target"])].append(
                {"outcomes": outs, "ctx": r.get("ctx_tokens") or 0}
            )
    return cells


def cluster_boot(runs: list[dict], rng) -> tuple[float, float, float, int, int]:
    """Return (point_success, lo95, hi95, n_runs, n_needles), resampling whole runs."""
    n = len(runs)
    n_needles = sum(len(r["outcomes"]) for r in runs)
    cor = sum(o == "correct" for r in runs for o in r["outcomes"])
    point = cor / n_needles if n_needles else 0.0
    if n <= 1:
        return point, point, point, n, n_needles
    reps: list[float] = []
    for _ in range(B):
        idx = [int(next(rng) * n) % n for _ in range(n)]
        samp = [runs[i] for i in idx]
        c = sum(o == "correct" for r in samp for o in r["outcomes"])
        t = sum(len(r["outcomes"]) for r in samp)
        reps.append(c / t if t else 0.0)
    reps.sort()
    return point, reps[int(0.025 * B)], reps[min(B - 1, int(0.975 * B))], n, n_needles


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    cells = load_runs()
    rng = _rand(SEED)
    models = [m for m in ORDER if any(k[0] == m for k in cells)]

    print("=" * 92)
    print("RUN-CLUSTERED BOOTSTRAP — success with 95% CI by resampling whole runs (not needles)")
    print(f"B={B} resamples, seed={SEED}. Columns: success [lo, hi]  (n_runs / n_needles)")
    print("=" * 92)
    for m in models:
        print(f"\n{m}")
        for cond in ("distance", "near"):
            fills = sorted({k[2] for k in cells if k[0] == m and k[1] == cond})
            for fill in fills:
                runs = cells[(m, cond, fill)]
                pt, lo, hi, nr, nn = cluster_boot(runs, rng)
                ctx = sum(r["ctx"] for r in runs) / len(runs) if runs else 0
                print(f"  {cond:8s} ctx≈{ctx:>8,.0f}: {pt:.3f} [{lo:.2f}, {hi:.2f}]"
                      f"  ({nr} runs / {nn} needles)")

    print("\n" + "=" * 92)
    print("NEAR CONTROL — minimum cell per model, run-clustered CI (NOT pinned at 1.00)")
    print("=" * 92)
    for m in models:
        near_cells = [(k[2], cells[k]) for k in cells if k[0] == m and k[1] == "near"]
        if not near_cells:
            continue
        scored = []
        for fill, runs in near_cells:
            pt, lo, hi, nr, nn = cluster_boot(runs, rng)
            ctx = sum(r["ctx"] for r in runs) / len(runs) if runs else 0
            scored.append((pt, lo, hi, nr, nn, ctx))
        pt, lo, hi, nr, nn, ctx = min(scored, key=lambda x: x[0])
        print(f"  {m:>26}: near MIN={pt:.3f} [{lo:.2f}, {hi:.2f}] "
              f"at ctx≈{ctx:,.0f} ({nr} runs / {nn} needles)")

    print("\n" + "=" * 92)
    print("DISTANCE SEVERITY MIX — per fill (correct/distractor/fabricated/unclass-wrong/abstain)")
    print("legacy runs use a flat 'wrong'; post-#8 runs split it. Read severity at the relevant ctx.")
    print("=" * 92)
    for m in models:
        fills = sorted({k[2] for k in cells if k[0] == m and k[1] == "distance"})
        if not fills:
            continue
        print(f"\n{m}")
        for fill in fills:
            runs = cells[(m, "distance", fill)]
            outs = [o for r in runs for o in r["outcomes"]]
            tot = len(outs) or 1
            ctx = sum(r["ctx"] for r in runs) / len(runs) if runs else 0
            cor = outs.count("correct") / tot
            dis = outs.count("distractor") / tot
            fab = outs.count("fabricated") / tot
            unc = (outs.count("unclassified_wrong") + outs.count("wrong")) / tot
            abst = outs.count("abstained") / tot
            print(f"  ctx≈{ctx:>8,.0f}: correct={cor:.2f} distractor={dis:.2f} "
                  f"fabricated={fab:.2f} wrong*={unc:.2f} abstain={abst:.2f}  (n={tot})")
    print("\n* 'wrong*' = legacy flat 'wrong' (pre-#8 data, not separable) + new unclassified_wrong.")


if __name__ == "__main__":
    main()
