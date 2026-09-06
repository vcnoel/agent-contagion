"""Positional desync on dispatch steps.

When an injection makes the model fail a later *read* step -- the step that
dispatches the next account -- what does it read instead? Each failed indep
read-step generation is classified by where its account argument points:

  revisit   an account whose subtask is already complete at step s
  skip      an account that comes later than the one due at step s
  other     unparseable, unknown id, right account with the wrong call shape

`revisit` and `skip` are both losses of place, not of arithmetic: the model
knows how to read, it no longer knows where it is.
"""
import collections
import re
import sys

sys.path.insert(0, ".")
from contagion.analyze import load, index          # noqa: E402
from contagion.env import make_task                # noqa: E402

AID = re.compile(r'AC-\d{4}')

for path in sys.argv[1:]:
    rows = load(path)
    model = rows[0]["model"]
    _, inj = index(rows)
    tasks = {}
    tally = collections.Counter()
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep" or r["kind"] != "read" or r["ok_faithful"] == 1:
            continue
        task = tasks.setdefault(seed, make_task(4, seed=seed))
        order = [a.aid for a in task.accounts]
        due = task.steps[s].account
        got = AID.findall(r["gen"])
        if not got:
            tally["other"] += 1
            continue
        aid = got[0]
        if aid not in order:
            tally["other"] += 1
        elif order.index(aid) < order.index(due):
            tally["revisit"] += 1
        elif order.index(aid) > order.index(due):
            tally["skip"] += 1
        else:
            tally["other"] += 1     # right account, wrong call shape
    n = sum(tally.values())
    if not n:
        print(f"{model}: no failed independent read steps")
        continue
    parts = "  ".join(f"{k}={v} ({100*v/n:.0f}%)" for k, v in tally.most_common())
    print(f"{model:11s} n={n:>4}   {parts}")
