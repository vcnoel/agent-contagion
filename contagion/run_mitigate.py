"""Does anything short of deleting the bad step actually help?

The err, flag and retry arms contain the same injected error at step t and
differ in what the context looks like afterwards. The redact arm contains no
error at all:

  err      the wrong action stands, and the rest of its subtask propagates it
  flag     as err, but the tool result carries an explicit error marker
           (the "detect and warn" guardrail)
  retry    the wrong action stands and is immediately followed by the correct
           action and result, then the subtask proceeds correctly. Built from
           the clean prefix, so it lacks err's propagated dependent steps, and
           it is one step longer than the other arms.
  redact   the clean prefix with step t replaced by a call-shaped placeholder
           (context compaction). Measures the cost of the edit itself.
  clean    no error at all (the paired baseline)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .env import SYSTEM, Step, make_task, user_prompt
from .inject import build
from .models import load
from .run_forced import _apply_template, generate
from .trajectory import normalize

ARMS = ["err", "flag", "retry", "redact"]


def _msgs(task, ctx: list[Step]) -> list[dict]:
    m = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_prompt(task)}]
    for st in ctx:
        m.append({"role": "assistant", "content": st.call})
        m.append({"role": "user", "content": "RESULT " + st.result})
    return m


def context_for(task, clean, errd, arm: str, t: int, s: int) -> list[Step]:
    """The steps visible to the model when it is asked to produce step s."""
    if arm == "err":
        return errd[:s]
    if arm == "flag":
        ctx = [Step(x.idx, x.account, x.kind, x.call, x.result) for x in errd[:s]]
        payload = json.loads(ctx[t].result)
        payload["error"] = "value inconsistent with the account record"
        ctx[t].result = json.dumps(payload)
        return ctx
    if arm == "retry":
        # wrong action, then the correct one, then the subtask done correctly
        return clean[:t] + [errd[t]] + clean[t:s]
    if arm == "redact":
        ctx = clean[:s]
        ctx = [Step(x.idx, x.account, x.kind, x.call, x.result) for x in ctx]
        ctx[t].call = "CALL [redacted]"
        ctx[t].result = json.dumps({"status": "step removed from history"})
        return ctx
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=25)
    ap.add_argument("--n-accounts", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    items = []
    for i in range(args.n_tasks):
        task = make_task(args.n_accounts, seed=20_000 + i)
        T = task.horizon
        clean = build(task, "clean", None, random.Random(0))
        for s in range(T):
            items.append(dict(task=task, ctx=clean[:s], arm="clean", t=-1, s=s))
        for t in range(T - 1):
            errd = build(task, "err", t, random.Random(31 * i + t))
            for arm in ARMS:
                for s in range(t + 1, T):
                    if clean[s].account == clean[t].account:
                        continue                      # keep only independent steps
                    items.append(dict(task=task, ctx=context_for(task, clean, errd, arm, t, s),
                                      arm=arm, t=t, s=s))

    prompts = [_apply_template(tok, _msgs(it["task"], it["ctx"])) for it in items]
    print("{} evaluation points".format(len(prompts)))
    gens = generate(tok, model, prompts, args.batch_size)

    out = Path(args.out or "results/mitigate_{}.jsonl".format(args.model))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it, g in zip(items, gens):
            task, s = it["task"], it["s"]
            canonical = normalize(task.steps[s].call)
            f.write(json.dumps({
                "model": args.model, "seed": task.seed, "arm": it["arm"],
                "t": it["t"], "s": s, "kind": task.steps[s].kind,
                "t_kind": task.steps[it["t"]].kind if it["t"] >= 0 else "",
                "same_role": int(it["t"] >= 0 and (s % 4) == (it["t"] % 4)),
                "ok": int(normalize(g) == canonical), "gen": g.strip()[:100]}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
