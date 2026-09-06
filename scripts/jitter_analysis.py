"""Period vs structural role.

With filler steps inserted, the analogous step is no longer a fixed distance
away. If contagion follows the *role* wherever it lands, the effect is
structural; if it stays at distance 4, it is positional.
"""
import collections, sys
import numpy as np
sys.path.insert(0, ".")
from contagion.analyze import load, index, cluster_boot

for path in sys.argv[1:]:
    rows = load(path); clean, inj = index(rows); m = rows[0]["model"]
    same, diff = collections.defaultdict(list), collections.defaultdict(list)
    bydist = collections.defaultdict(list)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep" or not r.get("t_kind"):
            continue
        if r["t_kind"] == "note" or r["kind"] == "note":
            continue                       # filler steps carry no injectable error
        c = clean.get((seed, s))
        if c is None:
            continue
        delta = c["ok_faithful"] - r["ok_faithful"]
        (same if r["t_kind"] == r["kind"] else diff)[seed].append(delta)
        bydist[s - t].append(delta)
    a = cluster_boot({k: np.asarray(v, float) for k, v in same.items()}, n=1500)
    b = cluster_boot({k: np.asarray(v, float) for k, v in diff.items()}, n=1500)
    print(f"\n{m}  (jitter: analogous steps no longer a fixed distance apart)")
    print(f"  same role      lambda = {a[0]:+.4f}  [{a[1]:+.4f},{a[2]:+.4f}]")
    print(f"  different role lambda = {b[0]:+.4f}  [{b[1]:+.4f},{b[2]:+.4f}]")
    print(f"  ratio = {(a[0]/b[0] if b[0] > 0.0005 else float('inf')):.1f}x")
    print("  by raw distance:  " + "  ".join(
        f"d{d}={np.mean(bydist[d]):+.2f}" for d in sorted(bydist) if len(bydist[d]) >= 15))
