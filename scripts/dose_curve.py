"""k-shot injection strength, paired against a clean baseline at the same step."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, ".")
from contagion.analyze import cluster_boot

for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/dose2_*.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    m = rows[0]["model"]
    by = {}
    for r in rows:
        by[(r["seed"], r["offset"], r["k"], r["arm"])] = r["ok"]
    print(f"\n{m}: failure rate at step offset+4k, poisoned minus clean")
    print(f"   {'k':>2} {'clean':>7} {'poisoned':>9} {'lambda_k':>9} {'95% CI':>20}")
    for k in sorted({r["k"] for r in rows if r["k"] > 0}):
        d = collections.defaultdict(list)
        c_all, p_all = [], []
        for (seed, off, kk, arm), ok in by.items():
            if kk != k or arm != "poisoned":
                continue
            c = by.get((seed, off, k, "clean"))
            if c is None:
                continue
            d[seed].append(c - ok); c_all.append(c); p_all.append(ok)
        mean, lo, hi = cluster_boot({s: np.asarray(v, float) for s, v in d.items()}, n=2000, seed=k)
        print(f"   {k:>2} {np.mean(c_all):>7.3f} {np.mean(p_all):>9.3f} "
              f"{mean:>+9.4f} [{lo:+.4f},{hi:+.4f}]")
