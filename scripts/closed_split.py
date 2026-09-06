"""Does closed loop 'improvement' come from the wrongly read account?

The err injection replaces one step with read_account(other), so the account
named in the corrupted call is itself in the scored set. This splits the
non injected accounts into the one the corrupted call named and the rest,
which are untouched by any surface mention.
"""
import collections
import glob
import json
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.env import make_task  # noqa: E402

rng = np.random.default_rng(0)


def boot(vals, n=4000):
    v = np.asarray(vals, float)
    if not len(v):
        return (float("nan"),) * 3
    draws = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/closed_*.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    model, i_inj = rows[0]["model"], rows[0]["i_inj"]
    by = collections.defaultdict(dict)
    named = {}
    for r in rows:
        seed = r["seed"]
        if seed not in named:
            task = make_task(4, seed=seed)
            canon = task.steps[i_inj]
            rr = random.Random(1000 * seed + i_inj)
            wrong = rr.choice([a for a in task.accounts if a.aid != canon.account])
            named[seed] = wrong.aid
        by[seed][r["arm"]] = r["per_account"]

    print("\n{}  (injection at step {}, names another account)".format(model, i_inj))
    for label, pick in (("named by the corrupted call", "named"),
                        ("never mentioned", "rest")):
        d = []
        for seed, arms in by.items():
            if "err" not in arms or "clean" not in arms:
                continue
            aid_named = named[seed]
            inj = [r["inj_aid"] for r in rows if r["seed"] == seed][0]
            keys = ([aid_named] if pick == "named"
                    else [k for k in arms["clean"] if k not in (inj, aid_named)])
            if not keys:
                continue
            d.append(np.mean([arms["clean"][k] for k in keys])
                     - np.mean([arms["err"][k] for k in keys]))
        m, lo, hi = boot(d)
        print("  {:28s} drop = {:+.3f} [{:+.3f}, {:+.3f}]  (n={})".format(
            label, m, lo, hi, len(d)))
