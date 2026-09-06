"""Teacher forced contagion measurement on the semi structured environment."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .env import RATES, Step, _fmt_call
from .env3 import SYSTEM3, _read_result, make_task3, user_prompt3
from .inject import _placebo2_form, _placebo_form
from .models import load
from .run_forced import _apply_template, generate
from .trajectory import normalize


def build3(task, condition, t, rng):
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

    acc = next(a for a in task.accounts if a.aid == st.account)
    obs_balance, obs_tier = acc.balance, acc.tier
    obs_shape = acc.shape

    if st.kind == "read":
        wrong = rng.choice([a for a in task.accounts if a.aid != acc.aid])
        st.call = _fmt_call("read_account", wrong.aid)
        st.result = _read_result(wrong)
        obs_balance, obs_tier = wrong.balance, wrong.tier
        obs_shape = wrong.shape

    obs_rate = RATES[obs_tier]
    if st.kind == "rate":
        wrong_tier = rng.choice([x for x in RATES if x != obs_tier])
        st.call = _fmt_call("get_rate", wrong_tier)
        st.result = json.dumps({"rate": RATES[wrong_tier]})
        obs_rate = RATES[wrong_tier]

    obs_product = round(obs_balance * obs_rate, 2)
    if st.kind == "compute":
        wrong_balance = round(obs_balance * rng.choice([0.1, 0.5, 2.0, 10.0]), 2)
        st.call = _fmt_call("compute", wrong_balance, obs_rate)
        obs_product = round(wrong_balance * obs_rate, 2)
        st.result = json.dumps({"product": obs_product})

    if st.kind == "submit":
        st.call = _fmt_call("submit", acc.aid,
                            round(obs_product * rng.choice([0.5, 2.0, 10.0]), 2))

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


def _msgs(task, steps, s):
    m = [{"role": "system", "content": SYSTEM3},
         {"role": "user", "content": user_prompt3(task)}]
    for st in steps[:s]:
        m.append({"role": "assistant", "content": st.call})
        m.append({"role": "user", "content": "RESULT " + st.result})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-accounts", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=90_000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--conditions", default="err,placebo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    items = []
    for k in range(args.n_tasks):
        task = make_task3(args.n_accounts, seed=args.seed0 + k)
        T = task.horizon
        clean = build3(task, "clean", None, random.Random(0))
        for s in range(T):
            items.append(dict(task=task, steps=clean, cond="clean", t=-1, s=s))
        for t in range(T - 1):
            for cond in args.conditions.split(","):
                st = build3(task, cond, t, random.Random(1000 * k + t))
                for s in range(t + 1, T):
                    items.append(dict(task=task, steps=st, cond=cond, t=t, s=s,
                                      inj_account=task.steps[t].account,
                                      t_kind=task.steps[t].kind))
    print("{} evaluation points".format(len(items)))

    prompts = [_apply_template(tok, _msgs(it["task"], it["steps"], it["s"]))
               for it in items]
    gens = generate(tok, model, prompts, args.batch_size)

    out = Path(args.out or "results/forced3_{}.jsonl".format(args.model))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it, g in zip(items, gens):
            task, steps, s = it["task"], it["steps"], it["s"]
            dep = it["cond"] != "clean" and steps[s].account == it.get("inj_account")
            f.write(json.dumps({
                "model": args.model, "seed": task.seed, "cond": it["cond"],
                "t": it["t"], "s": s, "kind": steps[s].kind,
                "t_kind": it.get("t_kind", ""), "env": "semistructured",
                "s_account": steps[s].account,
                "relation": "dep" if dep else "indep",
                "ok_faithful": int(normalize(g) == normalize(steps[s].call)),
                "ok_canonical": int(normalize(g) == normalize(task.steps[s].call)),
                "gen": g.strip()[:120],
            }) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
