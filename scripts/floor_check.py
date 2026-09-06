"""Lambda restricted to steps the model gets right when clean, which removes
the floor effect a weak clean baseline imposes on paired deltas."""
import collections
import glob
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import cluster_boot, index, load  # noqa: E402

for pattern in sys.argv[1:]:
    for path in sorted(glob.glob(pattern)):
        rows = load(path)
        clean, inj = index(rows)
        d = collections.defaultdict(list)
        for (seed, t, s), r in inj["err"].items():
            if r["relation"] != "indep":
                continue
            c = clean.get((seed, s))
            if c is None or c["ok_faithful"] != 1:
                continue
            d[seed].append(1 - r["ok_faithful"])
        m, lo, hi = cluster_boot({k: np.asarray(v, float) for k, v in d.items()}, n=2000)
        bad = [r["gen"] for r in rows
               if r["cond"] == "clean" and r["kind"] == "read"
               and r["ok_faithful"] == 0][:3]
        name = path.split("\\")[-1].split("/")[-1]
        print("{:28s} floor controlled lambda={:+.4f} [{:+.4f},{:+.4f}]".format(
            name, m, lo, hi))
        for g in bad:
            print("    failed clean read: {!r}".format(g[:70]))
