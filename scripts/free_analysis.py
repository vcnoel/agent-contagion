"""Escalation policies at equal budget, paired per task against policy=none.

The metric is per-account success (`correct / n`); `all_pass` is reported too
but a 0.5B model free-running four accounts rarely clears it, so the paired
per-account delta is what separates the policies. One episode is one task, so
the bootstrap resamples episodes.
"""
import collections
import glob
import json
import sys

import numpy as np

def boot(vals, n=4000):
    """Each cell gets its own fixed seed, so an interval does not depend on
    which files were analysed before it."""
    rng = np.random.default_rng(0)
    v = np.asarray(vals, float)
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    draws = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/free_*.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    pair = f"{rows[0]['small']} -> {rows[0]['large']}"
    base = {r["seed"]: r["correct"] / r["n"] for r in rows if r["policy"] == "none"}
    print(f"\n{pair}   (baseline per-account success = {np.mean(list(base.values())):.3f})")
    print(f"  {'policy':8s} {'budget':>6} {'esc.frac':>9} {'acct.success':>13} "
          f"{'delta vs none':>14} {'95% CI':>20}")
    combos = sorted({(r["policy"], r["budget"]) for r in rows if r["policy"] != "none"})
    for policy, budget in combos:
        sub = [r for r in rows if r["policy"] == policy and r["budget"] == budget]
        esc = sum(r["escalated"] for r in sub) / max(sum(r["steps"] for r in sub), 1)
        acc = [r["correct"] / r["n"] for r in sub]
        deltas = [r["correct"] / r["n"] - base[r["seed"]] for r in sub if r["seed"] in base]
        m, lo, hi = boot(deltas)
        print(f"  {policy:8s} {budget:>6.3f} {esc:>9.3f} {np.mean(acc):>13.3f} "
              f"{m:>+14.4f} [{lo:+.4f}, {hi:+.4f}]")
