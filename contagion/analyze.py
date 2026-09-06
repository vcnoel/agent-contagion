"""Estimate the contagion coefficient and the step reproduction number.

Two things this file is careful about:

1. *Pairing.* The clean baseline for step s is the same task's step s. Comparing
   condition means directly would confound the effect with step composition,
   since an injection at t only ever scores steps s > t.
2. *Clustering.* Observations within one task share a prompt, an account set and
   a random draw, so they are not independent. Every interval here is a cluster
   bootstrap that resamples whole tasks.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def index(rows):
    clean, inj = {}, collections.defaultdict(dict)
    for r in rows:
        if r["cond"] == "clean":
            clean[(r["seed"], r["s"])] = r
        else:
            inj[r["cond"]][(r["seed"], r["t"], r["s"])] = r
    return clean, inj


def cluster_boot(by_seed: dict, n=4000, seed=0):
    """by_seed: seed -> list of per-observation values. Returns (mean, lo, hi)."""
    seeds = [s for s, v in by_seed.items() if len(v)]
    if not seeds:
        return (float("nan"),) * 3
    flat = np.concatenate([by_seed[s] for s in seeds])
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        draws[i] = np.concatenate([by_seed[s] for s in pick]).mean()
    return float(flat.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def deltas_by_seed(clean, inj_cond, relation=None, kinds=None):
    d = collections.defaultdict(list)
    for (seed, t, s), r in inj_cond.items():
        if relation and r["relation"] != relation:
            continue
        if kinds and r["kind"] not in kinds:
            continue
        c = clean.get((seed, s))
        if c is not None:
            d[seed].append(c["ok_faithful"] - r["ok_faithful"])
    return {k: np.asarray(v, dtype=float) for k, v in d.items()}


def r0_by_seed(clean, inj_cond, t, relation="indep"):
    acc = collections.defaultdict(float)
    seen = set()
    for (seed, tt, s), r in inj_cond.items():
        if tt != t or (relation and r["relation"] != relation):
            continue
        c = clean.get((seed, s))
        if c is not None:
            acc[seed] += c["ok_faithful"] - r["ok_faithful"]
            seen.add(seed)
    return {s: np.asarray([acc[s]]) for s in seen}


def report(path):
    rows = load(path)
    model = rows[0]["model"]
    clean, inj = index(rows)
    seeds = sorted({r["seed"] for r in rows})
    print(f"\n{'=' * 72}\n{model}   ({len(seeds)} tasks, {len(rows)} evaluation points)\n{'=' * 72}")

    base = collections.defaultdict(list)
    for (_, _), r in clean.items():
        base[r["kind"]].append(r["ok_faithful"])
    print("clean accuracy by kind:  " + "   ".join(
        f"{k}={np.mean(v):.3f}" for k, v in sorted(base.items())) +
        f"   overall={np.mean([x for v in base.values() for x in v]):.3f}")

    print("\ncontagion coefficient lambda (positive = injection raises later error)")
    for cond in ("err", "placebo"):
        if cond not in inj:
            continue
        for rel in ("indep", "dep"):
            m, lo, hi = cluster_boot(deltas_by_seed(clean, inj[cond], rel))
            print(f"  {cond:8s} {rel:5s}  lambda = {m:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

    print("\nlambda on independent steps, by step kind (err condition)")
    for kind in ("read", "rate", "compute", "submit"):
        m, lo, hi = cluster_boot(deltas_by_seed(clean, inj["err"], "indep", {kind}))
        if m == m:
            print(f"  {kind:8s}  lambda = {m:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

    print("\nR0(t): extra downstream failures on logically independent steps")
    ts = sorted({t for (_, t, _) in inj["err"]})
    for t in ts:
        m, lo, hi = cluster_boot(r0_by_seed(clean, inj["err"], t), seed=t)
        if m == m:
            print(f"  t={t:2d}   R0 = {m:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    for p in ap.parse_args().paths:
        report(p)


if __name__ == "__main__":
    main()


KINDS = ["read", "rate", "compute", "submit"]


def injected_kind(t: int) -> str:
    """Steps repeat read/rate/compute/submit per account, so t fixes the kind."""
    return KINDS[t % 4]


def superspreader(path):
    """Is R0 a property of position, of what kind of step was corrupted, or both?"""
    rows = load(path)
    if any(r.get("jitter") or r.get("period", 4) != 4 for r in rows):
        raise SystemExit(f"{path}: injected_kind() assumes period 4 and no jitter; "
                         "this file was produced with a different layout")
    clean, inj = index(rows)
    model = rows[0]["model"]
    horizon = max(r["s"] for r in rows) + 1
    ts = sorted({t for (_, t, _) in inj["err"]})

    print(f"\n--- {model}: R0 decomposed ---")
    print(f"{'t':>3} {'kind':>8} {'remaining':>10} {'R0':>8} {'per-remaining':>14}")
    per_kind = collections.defaultdict(list)
    per_kind_norm = collections.defaultdict(list)
    for t in ts:
        by_seed = r0_by_seed(clean, inj["err"], t)
        vals = np.concatenate(list(by_seed.values())) if by_seed else np.array([])
        if not len(vals):
            continue
        rem = horizon - 1 - t
        k = injected_kind(t)
        norm = vals.mean() / rem if rem else float("nan")
        per_kind[k].append(vals.mean())
        per_kind_norm[k].append(norm)
        print(f"{t:>3} {k:>8} {rem:>10} {vals.mean():>8.3f} {norm:>14.4f}")

    print(f"\n  mean R0 by corrupted step kind:")
    for k in KINDS:
        if per_kind[k]:
            print(f"    {k:8s}  R0={np.mean(per_kind[k]):+.3f}   "
                  f"lambda per remaining step={np.mean(per_kind_norm[k]):+.4f}")

    # Does R0 scale with the number of steps left to infect, within a kind?
    print("\n  R0 ~ remaining, slope within each corrupted-step kind:")
    for k in KINDS:
        pts = [(horizon - 1 - t, np.concatenate(list(r0_by_seed(clean, inj["err"], t).values())).mean())
               for t in ts if injected_kind(t) == k and r0_by_seed(clean, inj["err"], t)]
        if len(pts) >= 3:
            x = np.array([p[0] for p in pts], dtype=float)
            y = np.array([p[1] for p in pts], dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            print(f"    {k:8s}  slope={slope:+.4f}/step  intercept={intercept:+.3f}  r={r:+.3f}")
