"""E1: teacher-forced per-step contagion measurement.

For a task instance and an injection step t, we ask the model for its action at
each later step s given a prefix that is byte-identical across conditions except
for the injection. Because the prefix is forced, the model's own subsequent
mistakes cannot compound -- what is measured at step s is the effect of the
injection alone. That is the estimand:

    lambda(t, s) = P(err at s | injected at t) - P(err at s | clean)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .env import make_task
from .inject import build
from .models import load
from .trajectory import messages_upto, normalize


def _apply_template(tok, msgs):
    # enable_thinking=False keeps hybrid reasoning tokenizers (Qwen3) from
    # opening a think block, which would break exact match scoring. Templates
    # without the kwarg raise TypeError and take the plain path.
    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True)
    try:
        return render(msgs)
    except Exception:
        merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
        merged += msgs[2:]
        return render(merged)


@torch.no_grad()
def generate(tok, model, prompts: list[str], batch_size: int, max_new_tokens: int = 24,
             temperature: float = 0.0):
    out = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="gen", leave=False):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(do_sample=False) if temperature <= 0 else dict(
            do_sample=True, temperature=temperature, top_p=0.95)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **kw)
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def build_items(n_tasks: int, n_accounts: int, seed0: int, jitter: int = 0,
                period: int = 4, conditions=("err", "placebo")):
    """Every (condition, t, s) evaluation point, plus the shared clean baseline."""
    items = []
    for k in range(n_tasks):
        task = make_task(n_accounts, seed=seed0 + k, jitter=jitter, period=period)
        T = task.horizon
        clean = build(task, "clean", None, random.Random(0))
        for s in range(T):
            items.append(dict(task=task, steps=clean, cond="clean", t=-1, s=s))
        for t in range(T - 1):
            for cond in conditions:
                st = build(task, cond, t, random.Random(1000 * k + t))
                inj_account = task.steps[t].account
                for s in range(t + 1, T):
                    items.append(dict(task=task, steps=st, cond=cond, t=t, s=s,
                                      inj_account=inj_account,
                                      t_kind=task.steps[t].kind))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--n-accounts", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--period", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--conditions", default="err,placebo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    items = build_items(args.n_tasks, args.n_accounts, args.seed0, args.jitter,
                        args.period, tuple(args.conditions.split(",")))
    print(f"{len(items)} evaluation points")

    prompts = [_apply_template(tok, messages_upto(it["task"], it["steps"], it["s"])) for it in items]
    gens = generate(tok, model, prompts, args.batch_size, temperature=args.temperature)

    out_path = Path(args.out or f"results/forced_{args.model}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it, g in zip(items, gens):
            task, steps, s = it["task"], it["steps"], it["s"]
            faithful = normalize(steps[s].call)
            canonical = normalize(task.steps[s].call)
            got = normalize(g)
            dep = it["cond"] != "clean" and steps[s].account == it.get("inj_account")
            f.write(json.dumps({
                "model": args.model, "seed": task.seed, "cond": it["cond"],
                "t": it["t"], "s": s, "kind": steps[s].kind,
                "t_kind": it.get("t_kind", ""), "jitter": args.jitter,
                "period": args.period, "temperature": args.temperature,
                "s_account": steps[s].account,
                "relation": "dep" if dep else "indep",
                "ok_faithful": int(got == faithful),
                "ok_canonical": int(got == canonical),
                "gen": g.strip()[:120],
            }) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
