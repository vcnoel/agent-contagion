"""Lambda on independent steps excluding read injections, per model."""
import collections
import glob
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import cluster_boot, index, load  # noqa: E402

for path in sorted(glob.glob("results/forced_*.jsonl")):
    rows = load(path)
    clean, inj = index(rows)
    d = collections.defaultdict(list)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep" or t % 4 == 0:
            continue
        c = clean.get((seed, s))
        if c is not None:
            d[seed].append(c["ok_faithful"] - r["ok_faithful"])
    m, lo, hi = cluster_boot({k: np.asarray(v, float) for k, v in d.items()}, n=2000)
    print("{:12s} noread lambda={:+.4f} [{:+.4f},{:+.4f}]".format(
        rows[0]["model"], m, lo, hi))
