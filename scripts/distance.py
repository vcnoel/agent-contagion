"""Free probe: does contagion decay with distance from the injected step?

Pure re-analysis of the runs already on disk. If lambda is flat in (s - t), the
effect is a persistent state change; if it decays, it is a local perturbation.
"""
import collections, sys
import numpy as np
sys.path.insert(0, ".")
from contagion.analyze import load, index, cluster_boot   # noqa: E402

for path in sys.argv[1:]:
    rows = load(path); clean, inj = index(rows)
    print(f"\n{rows[0]['model']}: lambda by distance (independent steps only)")
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep":
            continue
        c = clean.get((seed, s))
        if c is not None:
            buckets[s - t][seed].append(c["ok_faithful"] - r["ok_faithful"])
    for d in sorted(buckets):
        by_seed = {k: np.asarray(v, float) for k, v in buckets[d].items()}
        n = sum(len(v) for v in by_seed.values())
        if n < 30:
            continue
        m, lo, hi = cluster_boot(by_seed, n=1500, seed=d)
        bar = "#" * max(0, int(round(m * 200)))
        print(f"  d={d:2d}  lambda={m:+.4f} [{lo:+.4f},{hi:+.4f}]  n={n:4d}  {bar}")
