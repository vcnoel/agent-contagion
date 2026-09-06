"""Strong placebo (placebo2) lambda, overall and same role."""
import collections
import glob
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import cluster_boot, index, load  # noqa: E402

KINDS = ["read", "rate", "compute", "submit"]

for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/p2strong_*.jsonl")):
    rows = load(path)
    clean, inj = index(rows)
    model = rows[0]["model"]
    for rel in ("indep", "dep"):
        d = collections.defaultdict(list)
        for (seed, t, s), r in inj["placebo2"].items():
            if r["relation"] != rel:
                continue
            c = clean.get((seed, s))
            if c is not None:
                d[seed].append(c["ok_faithful"] - r["ok_faithful"])
        m, lo, hi = cluster_boot({k: np.asarray(v, float) for k, v in d.items()}, n=2000)
        print("{} placebo2 {}: lambda={:+.4f} [{:+.4f},{:+.4f}]".format(
            model, rel, m, lo, hi))
    d = collections.defaultdict(list)
    for (seed, t, s), r in inj["placebo2"].items():
        if r["relation"] != "indep" or r["kind"] != KINDS[t % 4]:
            continue
        c = clean.get((seed, s))
        if c is not None:
            d[seed].append(c["ok_faithful"] - r["ok_faithful"])
    m, lo, hi = cluster_boot({k: np.asarray(v, float) for k, v in d.items()}, n=2000)
    print("{} placebo2 same role: lambda={:+.4f} [{:+.4f},{:+.4f}]".format(
        model, m, lo, hi))
