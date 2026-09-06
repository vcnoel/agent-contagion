"""Injection operators.

Three conditions, all of which keep the trajectory the same *length* so that
prefix length is never confounded with condition:

  clean    canonical trajectory
  placebo  step t's action is rewritten in an equivalent surface form
           (semantics identical, canonical pattern broken) -- controls for
           "any deviation from the template degrades what follows"
  err      step t's action carries a wrong argument, and the environment
           returns the result for the action actually issued. Downstream steps
           of the *same* account then propagate that error faithfully, which is
           what a real agent does.
"""
from __future__ import annotations

import json
import random

from .env import RATES, Account, Step, Task, _fmt_call

Condition = str  # "clean" | "placebo" | "err"


def _account_of(task: Task, aid: str) -> Account:
    return next(a for a in task.accounts if a.aid == aid)


def _placebo_form(call: str) -> str:
    """Semantics-preserving surface rewrite: widen the argument separator."""
    return call.replace("(", "( ", 1)[::-1].replace(")", " )"[::-1], 1)[::-1]


def _placebo2_form(call: str) -> str:
    """Strong semantics-preserving rewrite: single quotes, padded separators,
    and trailing zeros on every number. The value of every argument is
    unchanged, the surface is heavily off template."""
    import re

    c = call.replace('"', "'").replace("(", "( ", 1)
    c = c[::-1].replace(")", " )"[::-1], 1)[::-1]
    c = c.replace(",", " , ")

    def pad(m):
        s = m.group(0)
        return s + ("00" if "." in s else ".00")

    # Pad only outside quoted strings, so ids like 'AC-1234' stay untouched.
    parts = re.split(r"('[^']*')", c)
    return "".join(p if p.startswith("'") else
                   re.sub(r"-?\d+(?:\.\d+)?", pad, p) for p in parts)


def build(task: Task, condition: Condition, t: int | None, rng: random.Random) -> list[Step]:
    """Return the trajectory under `condition` with the injection at step `t`."""
    steps = [Step(s.idx, s.account, s.kind, s.call, s.result) for s in task.steps]
    if condition == "clean" or t is None:
        return steps

    st = steps[t]
    if condition == "placebo":
        st.call = _placebo_form(st.call)
        return steps

    if condition == "placebo2":
        st.call = _placebo2_form(st.call)
        return steps

    if condition != "err":
        raise ValueError(condition)

    acc = _account_of(task, st.account)
    obs_balance, obs_tier = acc.balance, acc.tier

    if st.kind == "read":
        wrong = rng.choice([a for a in task.accounts if a.aid != acc.aid])
        st.call = _fmt_call("read_account", wrong.aid)
        st.result = json.dumps({"balance": wrong.balance, "tier": wrong.tier})
        obs_balance, obs_tier = wrong.balance, wrong.tier

    obs_rate = RATES[obs_tier]
    if st.kind == "rate":
        wrong_tier = rng.choice([x for x in RATES if x != obs_tier])
        st.call = _fmt_call("get_rate", wrong_tier)
        st.result = json.dumps({"rate": RATES[wrong_tier]})
        obs_rate = RATES[wrong_tier]

    # With no compute step in the subtask the submitted value is the rate itself.
    has_compute = any(x.kind == "compute" for x in steps if x.account == acc.aid)
    obs_product = round(obs_balance * obs_rate, 2) if has_compute else obs_rate
    if st.kind == "compute":
        wrong_balance = round(obs_balance * rng.choice([0.1, 0.5, 2.0, 10.0]), 2)
        st.call = _fmt_call("compute", wrong_balance, obs_rate)
        obs_product = round(wrong_balance * obs_rate, 2)
        st.result = json.dumps({"product": obs_product})

    if st.kind == "submit":
        st.call = _fmt_call("submit", acc.aid, round(obs_product * rng.choice([0.5, 2.0, 10.0]), 2))

    # Faithful downstream propagation within the same account.
    for s in steps[t + 1:]:
        if s.account != acc.aid:
            break
        if s.kind == "rate":
            s.call = _fmt_call("get_rate", obs_tier)
            s.result = json.dumps({"rate": obs_rate})
        elif s.kind == "compute":
            s.call = _fmt_call("compute", obs_balance, obs_rate)
            s.result = json.dumps({"product": obs_product})
        elif s.kind == "submit":
            s.call = _fmt_call("submit", acc.aid, obs_product)
    return steps
