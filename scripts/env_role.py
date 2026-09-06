"""Same role against different role lambda for any forced results file that
records t_kind (env2, env3, and any rerun of env1)."""
import collections
import glob
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import cluster_boot, index, load  # noqa: E402

for pattern in sys.argv[1:] or ["results/forced2_*.jsonl"]:
    for path in sorted(glob.glob(pattern)):
        rows = load(path)
        clean, inj = index(rows)
        out = {}
        for same in (True, False):
            d = collections.defaultdict(list)
            KINDS = ["read", "rate", "compute", "submit"]
            for (seed, t, s), r in inj["err"].items():
                if r["relation"] != "indep":
                    continue
                tk = r.get("t_kind") or KINDS[t % 4]
                if (r["kind"] == tk) != same:
                    continue
                c = clean.get((seed, s))
                if c is not None:
                    d[seed].append(c["ok_faithful"] - r["ok_faithful"])
            out[same] = cluster_boot(
                {k: np.asarray(v, float) for k, v in d.items()}, n=2000)
        m1, lo1, hi1 = out[True]
        m0, lo0, hi0 = out[False]
        ratio = m1 / m0 if m0 > 0 else float("inf")
        print("{:28s} same={:+.4f} [{:+.4f},{:+.4f}]  diff={:+.4f} "
              "[{:+.4f},{:+.4f}]  ratio={:.1f}x".format(
                  path.split("/")[-1].split("\\")[-1],
                  m1, lo1, hi1, m0, lo0, hi0, ratio))
