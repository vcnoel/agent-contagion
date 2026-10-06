"""E3: free-running rollouts under a fixed escalation budget.

A small model drives the trajectory. At each step an escalation policy may hand
that single step to a large model instead. Policies are compared at *equal
budget* -- the same expected fraction of steps escalated -- so the only thing
that differs is where the budget is spent.

Episodes are advanced in lockstep so that every step is one batched generation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .env import SYSTEM, make_task, user_prompt
from .models import load
from .runtime import Ledger, parse


class Episode:
    def __init__(self, task):
        self.task = task
        self.ledger = Ledger(task)
        self.msgs = [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user_prompt(task)}]
        self.escalated = 0
        self.steps = 0
        self.done = False

    def observe(self, call_text: str, result: str):
        self.msgs.append({"role": "assistant", "content": call_text.strip()})
        self.msgs.append({"role": "user", "content": f"RESULT {result}"})


def apply_template(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
        return tok.apply_chat_template(merged + msgs[2:], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def batch_step(tok, model, prompts, max_new_tokens=24, chunk=8):
    """Returns (texts, mean_token_logprobs).

    Generation is chunked to bound peak VRAM and sustained power draw: with two
    models resident on one 16GB laptop GPU, full-width batches have twice caused
    a hard reboot mid-run.
    """
    texts, confs = [], []
    for i in range(0, len(prompts), chunk):
        sub = prompts[i:i + chunk]
        enc = tok(sub, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             return_dict_in_generate=True, output_scores=True)
        new = out.sequences[:, enc["input_ids"].shape[1]:]
        logps = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in out.scores], dim=1)
        picked = logps.gather(2, new.unsqueeze(-1)).squeeze(-1)
        mask = (new != tok.pad_token_id).float()
        conf = ((picked * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist()
        texts.extend(tok.decode(r, skip_special_tokens=True) for r in new)
        confs.extend(conf)
    return texts, confs


KIND_INDEX = {"read": 0, "rate": 1, "compute": 2, "submit": 3}


def should_escalate(policy, i, horizon, budget, conf, threshold, rng,
                    target_idx=0):
    if policy == "none" or budget <= 0:
        return False
    if policy == "random":
        return rng.random() < budget
    if policy == "front":
        return i < math.ceil(budget * horizon)
    if policy == "back":
        return i >= horizon - math.ceil(budget * horizon)
    if policy == "conf":
        return conf is not None and conf < threshold
    # R0-informed: spend the budget on dispatch steps (period-4 read positions),
    # optionally also on submit positions. Nominal fractions 1/4 and 1/2.
    if policy == "dispatch":
        return i % 4 == 0
    if policy == "dispatch_submit":
        return i % 4 in (0, 3)
    # Matched: protect the positions of one step kind, passed in by name. These
    # positions equal the kind only while the driver stays in step with the
    # canonical cycle.
    if policy == "matched":
        return i % 4 == target_idx
    raise ValueError(policy)


def rollout(tok_s, small, tok_l, large, tasks, policy, budget, threshold, seed,
            slack=4, target_idx=0):
    """Returns (episodes, all small-model confidences seen during the run).

    The confidences from a policy="none" rollout are the calibration set for the
    `conf` policy: its threshold is chosen as the `budget`-quantile of those
    values, so that `conf` escalates the same expected fraction of steps as the
    positional policies and the comparison is at equal budget.
    """
    eps = [Episode(t) for t in tasks]
    horizon = tasks[0].horizon
    rng = random.Random(seed)
    all_confs = []
    for i in range(horizon + slack):
        live = [e for e in eps if not e.done]
        if not live:
            break
        prompts = [apply_template(tok_s, e.msgs) for e in live]
        texts, confs = batch_step(tok_s, small, prompts)
        all_confs.extend(confs)

        esc_idx = [j for j, e in enumerate(live)
                   if should_escalate(policy, i, horizon, budget, confs[j],
                                      threshold, rng, target_idx)]
        if esc_idx and large is not None:
            l_prompts = [apply_template(tok_l, live[j].msgs) for j in esc_idx]
            l_texts, _ = batch_step(tok_l, large, l_prompts)
            for j, t in zip(esc_idx, l_texts):
                texts[j] = t
                live[j].escalated += 1

        for e, text, c in zip(live, texts, confs):
            p = parse(text)
            e.steps += 1
            if p is None:
                e.observe(text, json.dumps({"error": "no tool call found"}))
            else:
                e.observe(text.strip().splitlines()[0], e.ledger.call(*p))
            if len(e.ledger.submissions) >= len(e.task.accounts):
                e.done = True
    return eps, all_confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", default=None)
    ap.add_argument("--policies", default="none,random,front,back,conf")
    ap.add_argument("--budgets", default="0.25")
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-accounts", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--target-kind", default="read",
                    choices=list(KIND_INDEX),
                    help="step kind the `matched` policy protects, taken from "
                         "this driver's own R0 decomposition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok_s, small = load(args.small)
    tok_l, large = (load(args.large) if args.large else (None, None))
    tasks = [make_task(args.n_accounts, seed=args.seed0 + k) for k in range(args.n_tasks)]

    policies = args.policies.split(",")
    # `conf` needs a calibration set of confidences; a policy="none" rollout
    # provides it. Run "none" first whether or not it was asked for.
    if "conf" in policies and "none" not in policies:
        policies = ["none"] + policies
    pilot_confs = None

    out = Path(args.out or f"results/free_{args.small}_{args.large}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for policy in policies:
            budgets = [0.0] if policy == "none" else [float(b) for b in args.budgets.split(",")]
            for budget in budgets:
                threshold = None
                if policy == "conf":
                    threshold = float(np.quantile(pilot_confs, budget))
                eps, confs = rollout(tok_s, small, tok_l, large, tasks, policy, budget,
                                     threshold, seed=args.seed0,
                                     target_idx=KIND_INDEX[args.target_kind])
                if policy == "none":
                    pilot_confs = confs
                for e in eps:
                    sc = e.ledger.score()
                    f.write(json.dumps({
                        "small": args.small, "large": args.large, "policy": policy,
                        "budget": budget, "threshold": threshold,
                        "target_kind": args.target_kind if policy == "matched" else "",
                        "seed": e.task.seed, "steps": e.steps,
                        "escalated": e.escalated, **sc}) + "\n")
                agg = [e.ledger.score()["all_pass"] for e in eps]
                esc = sum(e.escalated for e in eps) / max(sum(e.steps for e in eps), 1)
                print(f"{policy:8s} budget={budget:.2f}  all_pass={sum(agg)}/{len(agg)}"
                      f"  actual_escalation={esc:.3f}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
