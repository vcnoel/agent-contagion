"""Closed loop contagion: paired per task deltas on non injected accounts."""
import collections
import glob
import json
import sys

import numpy as np

def boot(vals, n=4000):
    """Each interval gets its own fixed seed, so it does not depend on which
    files or arms were analysed before it."""
    rng = np.random.default_rng(0)
    v = np.asarray(vals, float)
    draws = np.array([rng.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n)])
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/closed_*.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    model = rows[0]["model"]
    by = collections.defaultdict(dict)
    for r in rows:
        other = [v for k, v in r["per_account"].items() if k != r["inj_aid"]]
        by[r["seed"]][r["arm"]] = (np.mean(other), r["per_account"][r["inj_aid"]])
    print("\n{}  ({} tasks, injection at lockstep step {})".format(
        model, len(by), rows[0]["i_inj"]))
    clean_o = np.mean([v["clean"][0] for v in by.values()])
    clean_i = np.mean([v["clean"][1] for v in by.values()])
    print("  clean: other accounts={:.3f}  injected account={:.3f}".format(
        clean_o, clean_i))
    for arm in ("err", "placebo"):
        d_other = [v["clean"][0] - v[arm][0] for v in by.values() if arm in v]
        d_inj = [v["clean"][1] - v[arm][1] for v in by.values() if arm in v]
        m, lo, hi = boot(d_other)
        mi, loi, hii = boot(d_inj)
        print("  {:8s} drop other={:+.3f} [{:+.3f},{:+.3f}]   "
              "drop injected={:+.3f} [{:+.3f},{:+.3f}]".format(
                  arm, m, lo, hi, mi, loi, hii))
