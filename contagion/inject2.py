"""Injection operators for the warehouse environment.

Same contract as inject.build: conditions keep trajectory length identical,
the error condition corrupts one step with a tool result consistent with the
corrupted call, and the rest of that item's steps propagate it faithfully.
"""
from __future__ import annotations

import json
import random

from .env import Step, _fmt_call
from .env2 import Task2
from .inject import _placebo2_form, _placebo_form


def build2(task: Task2, condition: str, t: int | None, rng: random.Random) -> list[Step]:
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

    item = next(i for i in task.items if i.sku == st.account)
    obs_stock, obs_target = item.stock, item.target

    if st.kind == "check":
        wrong = rng.choice([i for i in task.items if i.sku != item.sku])
        st.call = _fmt_call("check_stock", wrong.sku)
        st.result = json.dumps({"stock": wrong.stock, "target": wrong.target})
        obs_stock, obs_target = wrong.stock, wrong.target

    obs_qty = obs_target - obs_stock
    if st.kind == "order":
        wrong_stock = max(0, obs_stock + rng.choice([-25, 25, 50, obs_stock]))
        st.call = _fmt_call("compute_order", obs_target, wrong_stock)
        obs_qty = obs_target - wrong_stock
        st.result = json.dumps({"quantity": obs_qty})

    if st.kind == "place":
        st.call = _fmt_call("place_order", item.sku,
                            obs_qty * rng.choice([2, 5, 10]))

    for s in steps[t + 1:]:
        if s.account != item.sku:
            break
        if s.kind == "order":
            s.call = _fmt_call("compute_order", obs_target, obs_stock)
            s.result = json.dumps({"quantity": obs_qty})
        elif s.kind == "place":
            s.call = _fmt_call("place_order", item.sku, obs_qty)
    return steps
