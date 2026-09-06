"""Second environment: warehouse restock.

Same causal skeleton as the ledger audit, different everything else: domain,
tool names, argument arity, value types, and subtask period (three steps per
item instead of four). For each item in a stated order the agent must check
the stock, compute the order quantity against a target it is given, and place
the order. Items are mutually independent, so an error injected into one item
is logically irrelevant to every step of another.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .env import Step, _fmt_call


@dataclass
class Item:
    sku: str
    stock: int
    target: int

    @property
    def order(self) -> int:
        return self.target - self.stock


@dataclass
class Task2:
    items: list[Item]
    steps: list[Step] = field(default_factory=list)
    seed: int = 0

    @property
    def horizon(self) -> int:
        return len(self.steps)

    # analyze.py and run_forced address groups through `account`; Step reuses
    # that field name for the sku, so the machinery carries over unchanged.
    @property
    def accounts(self):
        return self.items


def make_task2(n_items: int = 5, seed: int = 0) -> Task2:
    rng = random.Random(seed)
    items: list[Item] = []
    used: set[str] = set()
    for _ in range(n_items):
        while True:
            sku = "SKU-{}".format(rng.randint(100, 999))
            if sku not in used:
                used.add(sku)
                break
        stock = rng.randint(3, 60)
        items.append(Item(sku=sku, stock=stock, target=stock + rng.randint(15, 120)))

    steps: list[Step] = []
    for it in items:
        i = len(steps)
        steps.append(Step(i, it.sku, "check",
                          _fmt_call("check_stock", it.sku),
                          json.dumps({"stock": it.stock, "target": it.target})))
        steps.append(Step(i + 1, it.sku, "order",
                          _fmt_call("compute_order", it.target, it.stock),
                          json.dumps({"quantity": it.order})))
        steps.append(Step(i + 2, it.sku, "place",
                          _fmt_call("place_order", it.sku, it.order),
                          json.dumps({"status": "ordered"})))
    return Task2(items=items, steps=steps, seed=seed)


SYSTEM2 = """You are a warehouse restock agent. You issue exactly one tool call per turn.

Tools:
  check_stock(sku)          -> {"stock": int, "target": int}
  compute_order(target, stock) -> {"quantity": int}
  place_order(sku, quantity)   -> {"status": string}

Procedure, repeated for each SKU in the order given:
  1. check_stock(sku)
  2. compute_order(target, stock)   using the values returned by check_stock
  3. place_order(sku, quantity)     using the quantity you were given

Never do arithmetic yourself. Always copy values from the tool results above.

Reply with the tool call and nothing else, in the form:
CALL name(arg1, arg2)
String arguments are double-quoted. Numbers are bare."""


def user_prompt2(task: Task2) -> str:
    skus = ", ".join(i.sku for i in task.items)
    return "Restock these SKUs in order: {}\nBegin.".format(skus)
