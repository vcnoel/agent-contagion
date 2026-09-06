"""Ledger-audit environment.

A trajectory is a flat list of assistant actions, each followed by a tool result.
Accounts are mutually independent: the three steps belonging to account A carry no
information relevant to account B. That independence is the whole point -- it is
what lets an injected error at one step be *logically irrelevant* to a later step,
so that any measured degradation there is contagion and not dependence.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Literal

TIERS = ["A", "B", "C", "D"]
RATES = {"A": 0.045, "B": 0.032, "C": 0.021, "D": 0.014}

StepKind = Literal["read", "rate", "compute", "submit", "note"]


@dataclass
class Step:
    idx: int
    account: str          # group id; steps of different accounts are independent
    kind: StepKind
    call: str             # canonical correct action at this step
    result: str           # tool result the environment returns for `call`


@dataclass
class Account:
    aid: str
    balance: float
    tier: str

    @property
    def rate(self) -> float:
        return RATES[self.tier]

    @property
    def interest(self) -> float:
        return round(self.balance * self.rate, 2)


@dataclass
class Task:
    accounts: list[Account]
    steps: list[Step] = field(default_factory=list)
    seed: int = 0

    @property
    def horizon(self) -> int:
        return len(self.steps)


def _fmt_call(name: str, *args) -> str:
    inner = ", ".join(json.dumps(a) for a in args)
    return f"CALL {name}({inner})"


def make_task(n_accounts: int = 4, seed: int = 0, jitter: int = 0, period: int = 4) -> Task:
    """`jitter` inserts filler steps so that analogous steps are no longer a
    fixed distance apart. It exists to separate "the period is 4" from "the step
    is structurally analogous". jitter=0 leaves the task byte-identical and draws
    no extra randomness, so seeded tasks stay comparable across runs."""
    rng = random.Random(seed)
    accounts: list[Account] = []
    used: set[str] = set()
    for _ in range(n_accounts):
        while True:
            aid = f"AC-{rng.randint(1000, 9999)}"
            if aid not in used:
                used.add(aid)
                break
        accounts.append(
            Account(aid=aid, balance=round(rng.uniform(500, 9500), 2), tier=rng.choice(TIERS))
        )

    # `period` sets how many steps one subtask takes, so that "the period is 4"
    # can be separated from "the number 4". period=4 is byte-identical to before.
    steps: list[Step] = []
    for acc in accounts:
        i = len(steps)
        blk = [Step(i, acc.aid, "read",
                    _fmt_call("read_account", acc.aid),
                    json.dumps({"balance": acc.balance, "tier": acc.tier})),
               Step(i + 1, acc.aid, "rate",
                    _fmt_call("get_rate", acc.tier),
                    json.dumps({"rate": acc.rate}))]
        if period >= 4:
            blk.append(Step(i + 2, acc.aid, "compute",
                            _fmt_call("compute", acc.balance, acc.rate),
                            json.dumps({"product": acc.interest})))
        if period >= 5:
            blk.append(Step(i + 3, acc.aid, "note",
                            _fmt_call("log_note", acc.aid),
                            json.dumps({"status": "noted"})))
        value = acc.interest if period >= 4 else acc.rate
        blk.append(Step(i + len(blk), acc.aid, "submit",
                        _fmt_call("submit", acc.aid, value),
                        json.dumps({"status": "recorded"})))
        steps.extend(blk)

    if jitter:
        out: list[Step] = []
        for acc in accounts:
            blk = [s for s in steps if s.account == acc.aid]
            out.extend(blk)
            for _ in range(rng.randint(0, jitter)):
                out.append(Step(0, acc.aid, "note",
                                _fmt_call("log_note", acc.aid),
                                json.dumps({"status": "noted"})))
        for j, st in enumerate(out):
            st.idx = j
        steps = out
    return Task(accounts=accounts, steps=steps, seed=seed)


SYSTEM = """You are a ledger audit agent. You issue exactly one tool call per turn.

Tools:
  read_account(account_id) -> {"balance": float, "tier": string}
  get_rate(tier)           -> {"rate": float}
  compute(x, y)            -> {"product": float}
  submit(account_id, interest) -> {"status": string}
  log_note(account_id)     -> {"status": string}

Procedure, repeated for each account in the order given:
  1. read_account(account_id)
  2. get_rate(tier)             using the tier returned by read_account
  3. compute(balance, rate)     using the balance and rate you were given
  4. submit(account_id, interest)  using the product you were given
(If a step is not available as a tool, skip it and use the value you were given.)

Never do arithmetic yourself. Always copy values from the tool results above.

Reply with the tool call and nothing else, in the form:
CALL name(arg1, arg2)
String arguments are double-quoted. Numbers are bare."""


def user_prompt(task: Task) -> str:
    ids = ", ".join(a.aid for a in task.accounts)
    return f"Audit these accounts in order: {ids}\nBegin."
