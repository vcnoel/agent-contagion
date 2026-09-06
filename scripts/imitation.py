"""Is the later failure a *reproduction of the injected error's operator*?

For each failure at s = t + P (P = subtask period), we ask what relation the
emitted call bears to the injected one:

  literal    the emitted argument is the injected argument, copied verbatim
  operator   the emitted value is the injected *transformation* re-applied to
             the new operands (injected 10x, emitted 10x of the correct value)
  other      a failure unrelated to the injection

A high `operator` rate means the model induced a rule from its own mistake.
"""
import collections, json, random, re, sys
import numpy as np
sys.path.insert(0, ".")
from contagion.analyze import load, index
from contagion.env import make_task
from contagion.inject import build

NUM = re.compile(r'-?\d+\.?\d*')


def nums(s):
    return [float(x) for x in NUM.findall(s)]


def strs(s):
    return re.findall(r'"([^"]*)"', s)


def classify(kind, canon_t, inj_t, canon_s, got_s):
    gs, cs = strs(got_s), strs(canon_s)
    gi, ci = strs(inj_t), strs(canon_t)
    if kind in ("read", "rate"):
        if gs and gi and gs[0] == gi[0] and gs[0] != (cs[0] if cs else None):
            return "literal"
        return "other"
    # numeric kinds: compare the ratio applied
    ni, nc = nums(inj_t), nums(canon_t)
    ng, ncs = nums(got_s), nums(canon_s)
    if kind == "compute":
        ni, nc, ng, ncs = ni[:1], nc[:1], ng[:1], ncs[:1]
    else:                       # submit: value is the last number
        ni, nc, ng, ncs = ni[-1:], nc[-1:], ng[-1:], ncs[-1:]
    if not (ni and nc and ng and ncs) or nc[0] == 0 or ncs[0] == 0:
        return "other"
    if abs(ng[0] - ni[0]) < 0.02:
        return "literal"
    r_inj, r_got = ni[0] / nc[0], ng[0] / ncs[0]
    if r_inj != 0 and abs(r_got - r_inj) / abs(r_inj) < 0.02:
        return "operator"
    return "other"


for path in sys.argv[1:]:
    rows = load(path); clean, inj = index(rows)
    model = rows[0]["model"]
    tally = collections.defaultdict(collections.Counter)
    for (seed, t, s), r in inj["err"].items():
        if s - t != 4 or r["relation"] != "indep" or r["ok_faithful"] == 1:
            continue
        c = clean.get((seed, s))
        if c is None or c["ok_faithful"] != 1:
            continue        # only count steps the model gets right when clean
        task = make_task(4, seed=seed)
        steps = build(task, "err", t, random.Random(1000 * seed + t))
        lab = classify(steps[t].kind, task.steps[t].call, steps[t].call,
                       task.steps[s].call, "CALL " + r["gen"])
        tally[steps[t].kind][lab] += 1

    print(f"\n{model}: mechanism of the d=4 failures")
    tot = collections.Counter()
    for kind in ("read", "rate", "compute", "submit"):
        c = tally[kind]; n = sum(c.values())
        tot.update(c)
        if n:
            print(f"  {kind:8s} n={n:3d}   " + "  ".join(
                f"{k}={c[k]/n:.0%}" for k in ("literal", "operator", "other") if c[k]))
    n = sum(tot.values())
    print(f"  {'ALL':8s} n={n:3d}   " + "  ".join(
        f"{k}={tot[k]/n:.0%}" for k in ("literal", "operator", "other")))
    print(f"  reproduced the injected error's form: {(tot['literal']+tot['operator'])/n:.1%}")
