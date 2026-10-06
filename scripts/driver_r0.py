"""The escalation drivers' R0 decompositions by corrupted step kind.

R0(t) sums the contagion increment over the independent steps after t, so it
is normalised here by the number of independent steps remaining after t,
which excludes the remaining dependent steps of the injected account. A
denominator that counts all remaining steps would penalise kinds injected
early in an account (the read) relative to the submission, which has no
dependent steps after it. Reports raw R0 and R0 per independent remaining
step, each averaged within kind and cluster bootstrapped over tasks.

Usage: python scripts/driver_r0.py results/forced_qwen-0.5b.jsonl ...
"""
import collections
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import KINDS, index, injected_kind, load, r0_by_seed  # noqa: E402


def boot(vals, n=4000):
    rng = np.random.default_rng(0)
    vals = np.asarray(vals, float)
    draws = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)])
    return vals.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


for path in sys.argv[1:]:
    rows = load(path)
    if any(r.get("jitter") or r.get("period", 4) != 4 for r in rows):
        raise SystemExit(f"{path}: injected_kind() assumes period 4 and no jitter")
    clean, inj = index(rows)
    model = rows[0]["model"]
    indep_after = collections.defaultdict(set)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] == "indep":
            indep_after[t].add(s)
    raw = collections.defaultdict(dict)
    norm = collections.defaultdict(dict)
    for t in sorted(indep_after):
        kind = injected_kind(t)
        for seed, v in r0_by_seed(clean, inj["err"], t).items():
            raw[kind].setdefault(seed, []).append(float(v[0]))
            norm[kind].setdefault(seed, []).append(float(v[0]) / len(indep_after[t]))
    print(f"\n{model}: R0 by corrupted kind")
    print(f"  {'kind':8s} {'raw R0':>24s} {'R0 per independent remaining step':>36s}")
    for kind in KINDS:
        a = boot([np.mean(v) for v in raw[kind].values()])
        b = boot([np.mean(v) for v in norm[kind].values()])
        print(f"  {kind:8s} {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]"
              f"          {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]")
