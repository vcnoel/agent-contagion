"""Closed loop contagion: inject into a free running rollout.

The teacher forced design measures susceptibility with the model's own
mistakes held out. This experiment closes the loop: the model free runs the
whole task, and in the injected arm its own emitted action at one lockstep
step is replaced by a corrupted call whose tool result is consistent, after
which the model continues unconstrained. Scoring is per account correctness
of what the model actually submitted, so compounding is included.

Arms, on identical tasks:
  clean    pure free run
  err      at step i_inj the action is replaced by a read of a wrong account
  placebo  at step i_inj the action is replaced by the canonical call in a
           strongly off template surface form (same semantics)

Contagion in the closed loop is the paired drop in correctness on accounts
other than the injected one.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .env import make_task
from .inject import _placebo2_form, build
from .models import load
from .run_free import Episode, apply_template, batch_step
from .runtime import parse


def rollout(tok, model, tasks, arm: str, i_inj: int, seed: int, slack=4):
    eps = [Episode(t) for t in tasks]
    horizon = tasks[0].horizon
    for i in range(horizon + slack):
        live = [e for e in eps if not e.done]
        if not live:
            break
        prompts = [apply_template(tok, e.msgs) for e in live]
        texts, _ = batch_step(tok, model, prompts)

        for e, text in zip(live, texts):
            e.steps += 1
            if arm != "clean" and i == i_inj:
                canon = e.task.steps[i_inj]
                if arm == "err":
                    # Kind appropriate corruption, the same operator the
                    # teacher forced design uses. At a compute step this is a
                    # wrong operand, which names no other account, so the
                    # scored accounts are never mentioned in the injection.
                    corrupted = build(e.task, "err", i_inj,
                                      random.Random(1000 * e.task.seed + i_inj))
                    call = corrupted[i_inj].call
                else:
                    # The surface is off template but the meaning is the
                    # canonical one, so the placebo carries no error payload.
                    call = _placebo2_form(canon.call)
                result = e.ledger.call(*parse(call if arm == "err" else canon.call))
                e.observe(call, result)
                continue
            p = parse(text)
            if p is None:
                e.observe(text, json.dumps({"error": "no tool call found"}))
            else:
                e.observe(text.strip().splitlines()[0], e.ledger.call(*p))
            if len(e.ledger.submissions) >= len(e.task.accounts):
                e.done = True
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-accounts", type=int, default=4)
    ap.add_argument("--i-inj", type=int, default=6,
                    help="lockstep index of the injected step, default the "
                         "second account's compute step, whose corruption "
                         "names no other account")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    tasks = [make_task(args.n_accounts, seed=args.seed0 + k)
             for k in range(args.n_tasks)]
    inj_aid = {t.seed: t.steps[args.i_inj].account for t in tasks}

    out = Path(args.out or "results/closed_{}.jsonl".format(args.model))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for arm in ("clean", "err", "placebo"):
            eps = rollout(tok, model, tasks, arm, args.i_inj, args.seed0)
            ok_other = ok_inj = n_other = 0
            for e in eps:
                per = {a.aid: int(abs(e.ledger.submissions.get(
                    a.aid, float("nan")) - a.interest) < 0.011)
                    for a in e.task.accounts}
                aid = inj_aid[e.task.seed]
                f.write(json.dumps({
                    "model": args.model, "seed": e.task.seed, "arm": arm,
                    "i_inj": args.i_inj, "inj_aid": aid, "steps": e.steps,
                    "per_account": per,
                    "calls": [m["content"][:80] for m in e.msgs
                              if m["role"] == "assistant"]}) + "\n")
                ok_inj += per[aid]
                for k, v in per.items():
                    if k != aid:
                        ok_other += v
                        n_other += 1
            print("{:8s} other accounts={:.3f}  injected account={:.3f}".format(
                arm, ok_other / max(n_other, 1), ok_inj / len(eps)), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
