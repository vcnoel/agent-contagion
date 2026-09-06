"""Teacher forced contagion measurement on the warehouse environment."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .env2 import SYSTEM2, make_task2, user_prompt2
from .inject2 import build2
from .models import load
from .run_forced import _apply_template, generate
from .trajectory import normalize


def _msgs(task, steps, s):
    m = [{"role": "system", "content": SYSTEM2},
         {"role": "user", "content": user_prompt2(task)}]
    for st in steps[:s]:
        m.append({"role": "assistant", "content": st.call})
        m.append({"role": "user", "content": "RESULT " + st.result})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-items", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--conditions", default="err,placebo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    items = []
    for k in range(args.n_tasks):
        task = make_task2(args.n_items, seed=args.seed0 + k)
        T = task.horizon
        clean = build2(task, "clean", None, random.Random(0))
        for s in range(T):
            items.append(dict(task=task, steps=clean, cond="clean", t=-1, s=s))
        for t in range(T - 1):
            for cond in args.conditions.split(","):
                st = build2(task, cond, t, random.Random(1000 * k + t))
                for s in range(t + 1, T):
                    items.append(dict(task=task, steps=st, cond=cond, t=t, s=s,
                                      inj_account=task.steps[t].account,
                                      t_kind=task.steps[t].kind))
    print("{} evaluation points".format(len(items)))

    prompts = [_apply_template(tok, _msgs(it["task"], it["steps"], it["s"]))
               for it in items]
    gens = generate(tok, model, prompts, args.batch_size)

    out = Path(args.out or "results/forced2_{}.jsonl".format(args.model))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it, g in zip(items, gens):
            task, steps, s = it["task"], it["steps"], it["s"]
            dep = it["cond"] != "clean" and steps[s].account == it.get("inj_account")
            f.write(json.dumps({
                "model": args.model, "seed": task.seed, "cond": it["cond"],
                "t": it["t"], "s": s, "kind": steps[s].kind,
                "t_kind": it.get("t_kind", ""), "env": "warehouse",
                "s_account": steps[s].account,
                "relation": "dep" if dep else "indep",
                "ok_faithful": int(normalize(g) == normalize(steps[s].call)),
                "ok_canonical": int(normalize(g) == normalize(task.steps[s].call)),
                "gen": g.strip()[:120],
            }) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
