"""k-shot poisoning curve.

If an erroneous action works as an in-context demonstration, then *repeating* it
at successive analogous steps should behave like adding shots to a prompt: the
failure rate at the next analogous step should rise with k. This injects the same
error operator at steps t, t+P, ..., t+P(k-1) and scores step t+P*k.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

from .env import make_task
from .inject import build
from .models import load
from .run_forced import _apply_template, generate
from .trajectory import messages_upto, normalize

PERIOD = 4


class _ShadowTask:
    """Lets `build` operate on an already-modified trajectory."""
    def __init__(self, task, steps):
        self.accounts, self.steps, self.seed = task.accounts, steps, task.seed

    @property
    def horizon(self):
        return len(self.steps)


def build_k(task, offset: int, k: int, seed: int, consistent: bool = False):
    """Apply the err operator at offset, offset+P, ... offset+P(k-1), cumulatively.

    `consistent` reuses one RNG draw for every injection, so all k demonstrations
    exhibit the *same* wrong rule. Without it each injection draws its own
    operator, and the k demonstrations merely agree that something is wrong. The
    contrast separates rule induction from sheer volume of corrupted context.
    """
    steps = None
    for j in range(k):
        base = task if steps is None else _ShadowTask(task, steps)
        steps = build(base, "err", offset + j * PERIOD,
                      random.Random(seed if consistent else seed + j))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--n-accounts", type=int, default=5)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--consistent", action="store_true")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load(args.model)
    items = []
    for i in range(args.n_tasks):
        task = make_task(args.n_accounts, seed=10_000 + i)
        for offset in range(PERIOD):
            for k in range(0, args.max_k + 1):
                s = offset + PERIOD * k
                if s >= task.horizon:
                    continue
                clean = build(task, "clean", None, random.Random(0))
                # Paired clean baseline at the *same* step, so that k is never
                # confounded with how far into the trajectory the scored step sits.
                items.append(dict(task=task, steps=clean, k=k, offset=offset, s=s, arm="clean"))
                if k > 0:
                    items.append(dict(task=task, steps=build_k(task, offset, k, seed=7 * i + offset, consistent=args.consistent),
                                      k=k, offset=offset, s=s, arm="poisoned"))

    prompts = [_apply_template(tok, messages_upto(it["task"], it["steps"], it["s"])) for it in items]
    print(f"{len(prompts)} evaluation points")
    gens = generate(tok, model, prompts, args.batch_size)

    out = Path(args.out or f"results/dose_{args.model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it, g in zip(items, gens):
            canonical = normalize(it["task"].steps[it["s"]].call)
            f.write(json.dumps({
                "model": args.model, "seed": it["task"].seed, "k": it["k"],
                "offset": it["offset"], "s": it["s"],
                "kind": it["task"].steps[it["s"]].kind, "arm": it["arm"],
                "consistent": int(args.consistent),
                "ok": int(normalize(g) == canonical), "gen": g.strip()[:100]}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
