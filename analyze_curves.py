#!/usr/bin/env python3
"""
analyze_curves.py — fit the reliability coordinate system across models (Design §17).

Reads all dist_results_*.jsonl, and for each model's DISTANCE curve:
  * aggregates success vs context-fill (ctx_tokens; gpt-3.5 predates that field, so its
    ctx is APPROXIMATED from fill_target × a measured ratio — flagged in output);
  * fits a 2-parameter descending logistic  success(c) = 1 / (1 + (c/c50)^beta)
    by weighted (by n) grid-search — pure stdlib, no scipy/numpy dependency;
  * reports the model-free contour R90 (ctx where success first crosses 0.90) as the
    robust, fit-independent comparison point;
  * runs a data-collapse test: rescale x -> c/R90 and tabulate success at common
    multiples, to see whether the curves are one universal shape or need two params.

This is DESCRIPTION (place measured models on shared axes), not prediction. c50 for
models that never cross 0.50 in-window (gpt-3.5, Haiku) is an extrapolation — lean on
R90 + beta for those.
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

# gpt-3.5-turbo runs predate the ctx_tokens field; approximate its per-call fill from
# the target using the ratio observed on models that DO log it (~1.25-1.3x).
GPT35_CTX_RATIO = 1.25

# gpt-3.5-turbo RAW run-records were lost to an overly-broad cleanup (20 Jun 2026);
# its aggregate is preserved in Brief Findings. Reconstructed here so the ladder is
# complete; ctx approximated as fill_target x ~1.3 (its run predated ctx_tokens logging).
# Re-run (~$3) to restore clean raw data. (ctx, success, n_needles)
GPT35_FALLBACK: dict[str, list[tuple[float, float, int]]] = {
    "distance": [(1300, 0.98, 45), (2600, 0.96, 45), (5200, 0.93, 45), (10400, 0.71, 45)],
    "near": [(1300, 1.00, 45), (2600, 0.98, 45), (5200, 0.98, 45), (10400, 1.00, 45)],
}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval. NOTE: n here is needle-count, so this is OPTIMISTIC (needles within
    a run are correlated — review #6); the run-clustered interval is in bootstrap_ci.py."""
    if n <= 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def canonical_files() -> list[str]:
    """The canonical run files. Prefer an explicit manifest over a bare glob so a stray mock or
    scratch file can never be ingested into the real curves (review finding #1)."""
    manifest = Path("data/canonical_manifest.txt")
    if manifest.exists():
        names = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        # Run files live next to the manifest (data/); resolve relative to it so bare
        # filenames in the manifest stay portable regardless of the working directory.
        return [str(manifest.parent / n) for n in names if (manifest.parent / n).exists()]
    return sorted(glob.glob("data/dist_results_*.jsonl"))


def load() -> dict[str, dict[str, list[tuple[float, float, int]]]]:
    """model -> condition -> sorted list of (ctx, success_rate, n_needles)."""
    raw: dict[tuple[str, str, int], collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    ctxs: dict[tuple[str, str, int], list[float]] = collections.defaultdict(list)
    for f in canonical_files():
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("provider") == "mock":
                continue  # never ingest mock records (defense beyond the filename split)
            if abs(float(r.get("depth", 0.5)) - 0.5) > 1e-9:
                continue  # position-sweep arms (depth != 0.5) excluded from canonical curves
            key = (r["model"], r["condition"], r["fill_target"])
            c = r.get("ctx_tokens") or 0
            if not c:  # pre-ctx_tokens record (gpt-3.5): approximate
                c = r["fill_target"] * GPT35_CTX_RATIO
            ctxs[key].append(c)
            for n in r["needles"]:
                raw[key][n["outcome"]] += 1
    out: dict[str, dict[str, list[tuple[float, float, int]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for (model, cond, _fill), cnt in raw.items():
        tot = sum(cnt.values())
        succ = cnt["correct"] / tot if tot else 0.0
        mean_ctx = sum(ctxs[(model, cond, _fill)]) / len(ctxs[(model, cond, _fill)])
        out[model][cond].append((mean_ctx, succ, tot))
    for model in out:
        for cond in out[model]:
            out[model][cond].sort()
    return out


def fit_logistic(points: list[tuple[float, float, int]]) -> tuple[float, float, float]:
    """Weighted grid-search fit of success(c)=1/(1+(c/c50)^beta). Returns (c50, beta, sse)."""
    best = (float("nan"), float("nan"), float("inf"))
    cs = [p[0] for p in points]
    lo, hi = min(cs) * 0.3, max(cs) * 6
    # log-spaced c50 grid
    c50_grid = [lo * (hi / lo) ** (i / 60) for i in range(61)]
    beta_grid = [0.3 + 0.1 * i for i in range(0, 110)]  # 0.3 .. 11.2
    for c50 in c50_grid:
        for beta in beta_grid:
            sse = 0.0
            for c, s, n in points:
                pred = 1.0 / (1.0 + (c / c50) ** beta)
                sse += n * (s - pred) ** 2
            if sse < best[2]:
                best = (c50, beta, sse)
    return best


def contour(points: list[tuple[float, float, int]], thr: float) -> float | None:
    """ctx where success first crosses BELOW thr (linear interp). None if never crossed."""
    for (c0, s0, _), (c1, s1, _) in zip(points, points[1:]):
        if s0 >= thr > s1:
            frac = (s0 - thr) / (s0 - s1) if s0 != s1 else 0.0
            return c0 + frac * (c1 - c0)
    return None


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    data = load()
    if "gpt-3.5-turbo" not in data:  # raw data lost; use logged-summary fallback
        data["gpt-3.5-turbo"] = dict(GPT35_FALLBACK)  # type: ignore[assignment]
        print("NOTE: gpt-3.5-turbo from LOGGED SUMMARY (raw data lost; re-run to restore).\n")
    order = ["gpt-3.5-turbo", "gpt-4o-mini", "claude-haiku-4-5-20251001",
             "claude-sonnet-4-6", "open-model"]
    models = [m for m in order if m in data] + [m for m in data if m not in order]

    print("=" * 78)
    print("PER-MODEL DISTANCE CURVE FITS (success = 1/(1+(ctx/c50)^beta))")
    print("R90 = raw FIRST downward crossing of 0.90 (linear-interp), NOT a fitted parameter;")
    print("only c50/beta come from the logistic fit. R90 is noise-sensitive on non-monotonic")
    print("curves (esp. Sonnet, which bounces back above 0.90 after its first dip).")
    print("=" * 78)
    print(f"{'model':>26} {'R90(1st×)':>9} {'c50(fit)':>9} {'beta':>6} "
          f"{'shape':>9} {'pts':>4}")
    fits: dict[str, tuple[float, float, float | None]] = {}
    for m in models:
        pts = data[m].get("distance", [])
        if len(pts) < 3:
            continue
        c50, beta, _ = fit_logistic(pts)
        r90 = contour(pts, 0.90)
        fits[m] = (c50, beta, r90)
        if r90 is None:
            # Never crosses 0.90 in-window -> the logistic fit is unconstrained on the downside
            # (c50/beta just rail to the grid edges), so it is meaningless. Report the fit as n/a
            # and bound R90 by the DEEPEST ctx actually tested (">Nk") — NOT ">max": a flat curve
            # tested to 212k cannot be ranked against a model whose knee sits beyond that.
            maxc = max(p[0] for p in pts)
            print(f"{m:>26} {('>' + format(maxc, ',.0f')):>9} {'n/a':>9} {'n/a':>6} {'flat':>9} {len(pts):>4}")
            continue
        shape = "sharp" if beta >= 3 else ("gradual" if beta >= 1.2 else "v.gradual")
        print(f"{m:>26} {r90:>9,.0f} {c50:>9,.0f} {beta:>6.2f} {shape:>9} {len(pts):>4}")

    print("\n" + "=" * 78)
    print("LADDER — does the knee move with capability? (R90 = effective reliable length)")
    print("R90 is a DESCRIPTIVE first-crossing contour: the cross-model ORDERING is robust, the")
    print("precise value is noise-sensitive (esp. Sonnet). Read the ladder as ordinal, not exact.")
    print("=" * 78)
    r90s: list[tuple[str, float]] = [
        (m, r90v) for m in models if m in fits and (r90v := fits[m][2]) is not None
    ]
    for i, (m, r) in enumerate(r90s):
        rel = f"  ({r / r90s[0][1]:.1f}× the weakest)" if i else "  (baseline)"
        print(f"  {m:>26}: R90 ≈ {r:>8,.0f} tok{rel}")

    print("\n" + "=" * 78)
    print("COLLAPSE TEST — rescale x -> ctx/R90; if curves are ONE shape, success matches")
    print("at equal multiples. Differing values => steepness (beta) is model-specific.")
    print("=" * 78)
    print(f"{'model':>26} {'beta':>6} | success at ctx ≈ {'1.0×R90':>8} {'1.3×R90':>8} {'1.6×R90':>8}")
    for m in models:
        if m not in fits or not fits[m][2]:
            continue
        c50, beta, r90 = fits[m]
        row = []
        for mult in (1.0, 1.3, 1.6):
            pred = 1.0 / (1.0 + ((mult * r90) / c50) ** beta)  # type: ignore[operator]
            row.append(f"{pred:>8.2f}")
        print(f"{m:>26} {beta:>6.2f} | {'':>17} {row[0]} {row[1]} {row[2]}")
    print("\n(If the 1.3× / 1.6× columns differ across models, the decay SHAPE differs —")
    print(" i.e. a single-parameter rescaling does NOT collapse them: two params needed.)")

    print("\n" + "=" * 78)
    print("NEAR control — must stay AT/NEAR ceiling (it is NOT exactly 1.00; report min + CI)")
    print("=" * 78)
    for m in models:
        near = data[m].get("near", [])
        if not near:
            continue
        lo = min(near, key=lambda x: x[1])
        hi_ctx = max(near, key=lambda x: x[0])
        ci = wilson(lo[1], lo[2])
        print(f"  {m:>26}: near MIN={lo[1]:.3f} (n={lo[2]} needles, 95% CI "
              f"[{ci[0]:.2f},{ci[1]:.2f}]) at ctx≈{lo[0]:,.0f}; deepest ctx≈{hi_ctx[0]:,.0f} "
              f"near={hi_ctx[1]:.2f}")
        # matched-control check: every distance fill should have a near cell at ~the same ctx
        near_ctx = [c for c, _, _ in near]
        unmatched = [c for c, _, _ in data[m].get("distance", [])
                     if all(abs(c - nc) > 0.15 * max(c, 1.0) for nc in near_ctx)]
        if unmatched:
            print(f"      ⚠ {len(unmatched)} distance fill(s) lack a matched near cell (±15% ctx): "
                  f"{[f'{x:,.0f}' for x in unmatched]}")
    print("(CI is needle-level = OPTIMISTIC; needles within a run are correlated — see "
          "bootstrap_ci.py for the run-clustered interval. Review findings #5/#6.)")


if __name__ == "__main__":
    main()
