#!/usr/bin/env python3
"""invariant_power_sim.py — why N=3 can't license a cross-model 'law' (review #9).

Makes the write-up's invariant-hunt power claim reproducible from a committed script instead of
an un-committed subagent analysis. Two parts, both pure stdlib + deterministic:

  (1) OBSERVED fits of the R90 ladder: the geometric ('constant ratio per rung') fit on the first
      THREE models, and what happens when the 4th (Sonnet) is added out-of-sample.
  (2) NULL: how often a RANDOM increasing 3-sequence fits a straight line with R^2 >= 0.73 — i.e.
      how little a high R^2 means at N=3. If the null rate is high, the observed fit is not evidence.

Conclusion the script supports: a striking 3-point fit is expected by chance, and the geometric
ladder did NOT survive the out-of-sample 4th rung -> keep the DESCRIPTIVE coordinate system, not a
predictive law. (Matches the qualitative conclusion already in the docs; numbers are now derivable.)
"""
from __future__ import annotations

import math
import sys

# committed R90 first-crossing contours (tokens) and advertised windows
R90 = {"gpt-3.5-turbo": 7321, "gpt-4o-mini": 30683,
       "claude-haiku-4-5-20251001": 125668, "claude-sonnet-4-6": 395206}
LADDER = [7321, 30683, 125668, 395206]          # in capability order
WINDOWS = [16000, 128000, 200000, 1000000]
TRIALS = 200000
SEED = 20260622


def _lcg(seed: int):
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def r2_line(xs: list[float], ys: list[float]) -> float:
    """R^2 of an ordinary least-squares line y = a + b x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 1.0
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / syy


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    print("=" * 78)
    print("OBSERVED — R90 ladder as a GEOMETRIC progression (constant ratio per rung)")
    print("=" * 78)
    ratios = [LADDER[i + 1] / LADDER[i] for i in range(len(LADDER) - 1)]
    print(f"  rung ratios: {[f'{r:.2f}x' for r in ratios]}  (geometric law needs these ~equal)")
    idx3 = [0.0, 1.0, 2.0]
    log3 = [math.log(v) for v in LADDER[:3]]
    idx4 = [0.0, 1.0, 2.0, 3.0]
    log4 = [math.log(v) for v in LADDER]
    print(f"  geometric fit R^2 on FIRST 3 models : {r2_line(idx3, log3):.3f}")
    print(f"  geometric fit R^2 on ALL 4 models   : {r2_line(idx4, log4):.3f}  "
          "(Sonnet added out-of-sample)")
    # predicted vs actual Sonnet from the first-3 geometric fit
    mx = sum(idx3) / 3
    my = sum(log3) / 3
    b = sum((x - mx) * (y - my) for x, y in zip(idx3, log3)) / sum((x - mx) ** 2 for x in idx3)
    a = my - b * mx
    pred_sonnet = math.exp(a + b * 3)
    print(f"  3-rung fit predicts Sonnet R90 ≈ {pred_sonnet:,.0f}; actual = {LADDER[3]:,.0f} "
          f"({LADDER[3] / pred_sonnet - 1:+.0%}) -> law NOT confirmed out-of-sample")

    print("\n" + "=" * 78)
    print("OBSERVED — R90 proportional to advertised window?")
    print("=" * 78)
    fr = [R90[m] / w for m, w in zip(R90, WINDOWS)]
    print(f"  R90 / window: {[f'{x:.2%}' for x in fr]}  (a 'law' needs these ~equal; they span "
          f"{max(fr) / min(fr):.1f}x)")

    print("\n" + "=" * 78)
    print("NULL — random INCREASING 3-sequences: how often does a line fit with R^2 >= 0.73?")
    print(f"(TRIALS={TRIALS:,}, seed={SEED}) — high rate => a good 3-point fit means little")
    print("=" * 78)
    rng = _lcg(SEED)
    thr = 0.73
    hit_lin = hit_log = 0
    idx = [0.0, 1.0, 2.0]
    for _ in range(TRIALS):
        tri = sorted(next(rng) for _ in range(3))               # increasing 3-sequence in (0,1)
        if r2_line(idx, tri) >= thr:
            hit_lin += 1
        tlog = sorted(math.log(1e3 + next(rng) * 1e6) for _ in range(3))  # increasing in log space
        if r2_line(idx, tlog) >= thr:
            hit_log += 1
    print(f"  linear law  (index vs value):     R^2>={thr} in {hit_lin / TRIALS:.0%} of random triples")
    print(f"  geometric law (index vs log val): R^2>={thr} in {hit_log / TRIALS:.0%} of random triples")
    print("\nConclusion: a high 3-point R^2 is the NULL, not a signal; and the geometric ladder broke")
    print("on the out-of-sample 4th rung. -> report a DESCRIPTIVE coordinate system, not a law (#9).")


if __name__ == "__main__":
    main()
