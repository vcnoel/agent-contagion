"""Is contagion just a function of how much of the prefix the injection changed?

A `read` injection rewrites the whole downstream block of its account; a `submit`
injection changes one number. If R0 tracks the number of altered tokens rather
than what the corrupted step *authors*, the superspreader ordering is an artefact.
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.env import make_task            # noqa: E402
from contagion.inject import build             # noqa: E402
from contagion.trajectory import messages_upto  # noqa: E402
from contagion.analyze import index, load, r0_by_seed, injected_kind  # noqa: E402

import random

path = sys.argv[1]
rows = load(path)
if any(r.get("jitter") or r.get("period", 4) != 4 for r in rows):
    raise SystemExit(f"{path}: injected_kind() assumes period 4 and no jitter")
clean_idx, inj = index(rows)
seeds = sorted({r["seed"] for r in rows})
horizon = max(r["s"] for r in rows) + 1

def prefix_text(task, steps, s):
    return "\n".join(m["content"] for m in messages_upto(task, steps, s))

# altered characters in the prefix seen at the *first* independent step after t
diff_by_t = collections.defaultdict(list)
for seed in seeds[:10]:
    task = make_task(4, seed=seed)
    clean_steps = build(task, "clean", None, random.Random(0))
    for t in range(horizon - 1):
        err_steps = build(task, "err", t, random.Random(1000 * (seed - seeds[0]) + t))
        s = min(horizon - 1, ((t // 4) + 1) * 4)   # first step of the next account
        a, b = prefix_text(task, clean_steps, s), prefix_text(task, err_steps, s)
        diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        diff_by_t[t].append(diff)

print(f"{'t':>3} {'kind':>8} {'altered_chars':>14} {'R0':>8}")
xs, ys, kinds = [], [], []
for t in sorted(diff_by_t):
    by_seed = r0_by_seed(clean_idx, inj["err"], t)
    if not by_seed:
        continue
    r0 = np.concatenate(list(by_seed.values())).mean()
    d = float(np.mean(diff_by_t[t]))
    xs.append(d); ys.append(r0); kinds.append(injected_kind(t))
    print(f"{t:>3} {injected_kind(t):>8} {d:>14.1f} {r0:>8.3f}")

xs, ys = np.array(xs), np.array(ys)
print(f"\ncorr(altered_chars, R0) = {np.corrcoef(xs, ys)[0,1]:+.3f}")
rem = np.array([horizon - 1 - t for t in sorted(diff_by_t) if r0_by_seed(clean_idx, inj['err'], t)], float)
print(f"corr(remaining_steps, R0) = {np.corrcoef(rem, ys)[0,1]:+.3f}")
# partial: regress R0 on remaining, correlate residual with diff size
res = ys - np.polyval(np.polyfit(rem, ys, 1), rem)
print(f"corr(altered_chars, R0 | remaining) = {np.corrcoef(xs, res)[0,1]:+.3f}")
for k in ["read", "rate", "compute", "submit"]:
    m = [i for i, kk in enumerate(kinds) if kk == k]
    if m:
        print(f"  {k:8s} mean altered_chars={xs[m].mean():7.1f}  mean R0={ys[m].mean():+.3f}")
