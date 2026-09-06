"""Lambda under value based scoring instead of exact string match.

A call counts as correct if the tool name matches, string arguments match
exactly, and numeric arguments match by value (so 1.5, 1.50 and 01.5 are all
the same answer). This tests whether the headline effect is an artifact of
rigid surface scoring."""
import collections
import glob
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import cluster_boot, index, load  # noqa: E402

CALL = re.compile(r"^(?:call\s*)?([a-z_]+)\s*\((.*)\)\s*$", re.I | re.S)


def parse(call):
    c = call.strip().split("\n")[0].strip()
    m = CALL.match(c)
    if not m:
        return None
    name, argstr = m.group(1).lower(), m.group(2)
    args = []
    for a in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", argstr):
        a = a.strip().strip("\"'").strip()
        try:
            args.append(("n", round(float(a), 4)))
        except ValueError:
            args.append(("s", a))
    return name, tuple(args)


def ok(gen, expected):
    p = parse(gen)
    return p is not None and p == parse(expected)


for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/forced_*.jsonl")):
    rows = load(path)
    # The faithful target must be rebuilt from the task, since the file only
    # carries the generation.
    import random

    from contagion.env import make_task
    from contagion.inject import build

    cache = {}
    def target(r):
        key = (r["seed"], r["cond"], r["t"])
        if key not in cache:
            task = make_task(4, seed=r["seed"])
            k = r["seed"]  # seed == seed0 + k with seed0 = 0
            steps = (build(task, "clean", None, random.Random(0)) if r["cond"] == "clean"
                     else build(task, r["cond"], r["t"], random.Random(1000 * k + r["t"])))
            cache[key] = steps
        return cache[key][r["s"]].call

    for r in rows:
        r["ok_faithful"] = int(ok(r["gen"], target(r)))

    clean, inj = index(rows)
    d = collections.defaultdict(list)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep":
            continue
        c = clean.get((seed, s))
        if c is not None:
            d[seed].append(c["ok_faithful"] - r["ok_faithful"])
    m, lo, hi = cluster_boot({k: np.asarray(v, float) for k, v in d.items()}, n=2000)
    base = np.mean([r["ok_faithful"] for r in rows if r["cond"] == "clean"])
    print("{:12s} semantic: clean acc={:.3f}  lambda={:+.4f} [{:+.4f},{:+.4f}]".format(
        rows[0]["model"], base, m, lo, hi))
