"""Teacher forced contagion in the free text environment.

The injection corrupts the VALUE a previous answer asserts, in a sentence
that is otherwise ordinary. The placebo restates the correct value in a
different but equally ordinary wording, so form and content are separated
without a template to deviate from.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .env4 import (SYSTEM4, Step, answer_ok, expected_value, make_task4,
                   question_for, user_prompt4)
from .models import load
from .run_forced import _apply_template, generate

# Rewordings that keep the value and change everything else around it.
PARAPHRASE = {
    "charge": ["{ref} is recorded at {val} euros.",
               "That one, {ref}, stands at {val} euros.",
               "For {ref} the figure is {val} euros."],
    "label":  ["{ref} files under {val}.",
               "That would be {val}, for {ref}.",
               "For {ref} the label reads {val}."],
}


def build4(task, condition, t, rng):
    steps = [Step(s.idx, s.account, s.kind, s.call, s.result) for s in task.steps]
    if condition == "clean" or t is None:
        return steps

    st = steps[t]
    rec = next(r for r in task.records if r.ref == st.account)
    if condition == "placebo":
        val = rec.label if st.kind == "label" else "{:.2f}".format(rec.charge)
        st.call = rng.choice(PARAPHRASE[st.kind]).format(ref=st.account, val=val)
        return steps
    if condition != "err":
        raise ValueError(condition)

    # A wrong value, stated with the same confidence as a right one.
    if st.kind == "charge":
        # A wrong lookup. Nothing has to be induced to answer a later charge,
        # so this arm is the control on the label arm below.
        wrong = round(rec.charge * rng.choice([0.1, 0.5, 2.0, 10.0]), 2)
        st.call = "The charge on {} is {:.2f} euros.".format(st.account, wrong)
    else:
        # A wrong CONVENTION, not a wrong value: the two fields are correct
        # and in the wrong order. A model that induces the rule from its own
        # context should reapply the reversal at the next label.
        st.call = "The file label for {} is {}/{}.".format(
            st.account, rec.num, rec.vendor)
    return steps


def _msgs(task, steps, s):
    """Strict user/assistant alternation.

    The record list used to be a user turn of its own, which put two user
    turns in a row before the first question. Qwen's chat template tolerates
    that; Gemma's refuses outright, so the environment silently could not run
    on a whole family. The list now rides on the first question instead.
    """
    m = [{"role": "system", "content": SYSTEM4},
         {"role": "user",
          "content": user_prompt4(task) + "\n\n" + question_for(task, 0)}]
    for k in range(s):
        if k:
            m.append({"role": "user", "content": question_for(task, k)})
        m.append({"role": "assistant", "content": steps[k].call})
    if s:
        m.append({"role": "user", "content": question_for(task, s)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-records", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=70_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--conditions", default="err,placebo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    items = []
    for k in range(args.n_tasks):
        task = make_task4(args.n_records, seed=args.seed0 + k)
        T = task.horizon
        clean = build4(task, "clean", None, random.Random(0))
        for s in range(T):
            items.append(dict(task=task, steps=clean, cond="clean", t=-1, s=s))
        for t in range(T - 1):
            for cond in args.conditions.split(","):
                st = build4(task, cond, t, random.Random(1000 * k + t))
                for s in range(t + 1, T):
                    items.append(dict(task=task, steps=st, cond=cond, t=t, s=s,
                                      inj_account=task.steps[t].account,
                                      t_kind=task.steps[t].kind))
    print("{} evaluation points".format(len(items)))

    prompts = [_apply_template(tok, _msgs(it["task"], it["steps"], it["s"]))
               for it in items]
    gens = generate(tok, model, prompts, args.batch_size, max_new_tokens=40)

    out = Path(args.out or "results/forced4_{}.jsonl".format(args.model))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it, g in zip(items, gens):
            task, steps, s = it["task"], it["steps"], it["s"]
            dep = it["cond"] != "clean" and steps[s].account == it.get("inj_account")
            # Both turns are lookups, so the faithful answer is always the
            # record's own value: nothing downstream inherits the injection.
            faithful = expected_value(task, s)
            f.write(json.dumps({
                "model": args.model, "seed": task.seed, "cond": it["cond"],
                "t": it["t"], "s": s, "kind": steps[s].kind,
                "t_kind": it.get("t_kind", ""), "env": "freetext",
                "s_account": steps[s].account,
                "relation": "dep" if dep else "indep",
                "ok_faithful": answer_ok(g, faithful, steps[s].kind),
                "ok_canonical": answer_ok(g, expected_value(task, s), steps[s].kind),
                "gen": g.strip()[:120],
            }) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
