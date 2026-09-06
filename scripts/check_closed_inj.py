"""Confirm the closed loop injection at a compute step names no other account."""
import random
import sys

sys.path.insert(0, ".")
from contagion.env import make_task  # noqa: E402
from contagion.inject import _placebo2_form, build  # noqa: E402

for seed in (0, 1, 2):
    task = make_task(4, seed=seed)
    i_inj = 6
    canon = task.steps[i_inj]
    corrupted = build(task, "err", i_inj, random.Random(1000 * seed + i_inj))
    call = corrupted[i_inj].call
    others = [a.aid for a in task.accounts if a.aid != canon.account]
    names_other = any(a in call for a in others)
    print("seed {}  kind={}  inj={!r}".format(seed, canon.kind, call))
    print("   canonical={!r}".format(canon.call))
    print("   placebo={!r}".format(_placebo2_form(canon.call)))
    print("   names another account: {}".format(names_other))
