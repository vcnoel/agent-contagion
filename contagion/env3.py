"""Third environment: semi structured ledger audit, no fixed period.

Accounts come in three shapes. A full account needs read, rate, compute,
submit. An account whose read already returns the rate skips the rate step.
An account whose read already returns the interest skips everything but the
submit. Shapes are mixed randomly per task, so the analogous step is at no
fixed distance and the trajectory has no cycle, while subtask independence,
the property the causal design needs, is unchanged.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .env import RATES, TIERS, Step, _fmt_call


@dataclass
class Acct3:
    aid: str
    balance: float
    tier: str
    shape: str          # "full" | "ratefree" | "precomputed"

    @property
    def rate(self) -> float:
        return RATES[self.tier]

    @property
    def interest(self) -> float:
        return round(self.balance * self.rate, 2)


@dataclass
class Task3:
    accounts: list
    steps: list = field(default_factory=list)
    seed: int = 0

    @property
    def horizon(self) -> int:
        return len(self.steps)


def _read_result(acc: Acct3) -> str:
    if acc.shape == "full":
        return json.dumps({"balance": acc.balance, "tier": acc.tier})
    if acc.shape == "ratefree":
        return json.dumps({"balance": acc.balance, "tier": acc.tier,
                           "rate": acc.rate})
    return json.dumps({"balance": acc.balance, "tier": acc.tier,
                       "interest": acc.interest})


def make_task3(n_accounts: int = 5, seed: int = 0) -> Task3:
    rng = random.Random(seed)
    accounts, used = [], set()
    shapes = ["full", "ratefree", "precomputed"]
    for j in range(n_accounts):
        while True:
            aid = "AC-{}".format(rng.randint(1000, 9999))
            if aid not in used:
                used.add(aid)
                break
        accounts.append(Acct3(
            aid=aid, balance=round(rng.uniform(500, 9500), 2),
            tier=rng.choice(TIERS),
            shape=shapes[j % 3] if j < 3 else rng.choice(shapes)))
    rng.shuffle(accounts)

    steps: list[Step] = []
    for acc in accounts:
        i = len(steps)
        steps.append(Step(i, acc.aid, "read",
                          _fmt_call("read_account", acc.aid),
                          _read_result(acc)))
        if acc.shape == "full":
            steps.append(Step(len(steps), acc.aid, "rate",
                              _fmt_call("get_rate", acc.tier),
                              json.dumps({"rate": acc.rate})))
        if acc.shape in ("full", "ratefree"):
            steps.append(Step(len(steps), acc.aid, "compute",
                              _fmt_call("compute", acc.balance, acc.rate),
                              json.dumps({"product": acc.interest})))
        steps.append(Step(len(steps), acc.aid, "submit",
                          _fmt_call("submit", acc.aid, acc.interest),
                          json.dumps({"status": "recorded"})))
    return Task3(accounts=accounts, steps=steps, seed=seed)


SYSTEM3 = """You are a ledger audit agent. You issue exactly one tool call per turn.

Tools:
  read_account(account_id) -> {"balance": float, "tier": string, ...}
  get_rate(tier)           -> {"rate": float}
  compute(x, y)            -> {"product": float}
  submit(account_id, interest) -> {"status": string}

Procedure, repeated for each account in the order given:
  1. read_account(account_id)
  2. get_rate(tier)     ONLY if the read result did not include a rate
  3. compute(balance, rate)  ONLY if the read result did not include the interest
  4. submit(account_id, interest)

Skip any step whose value the read result already gave you. Never do
arithmetic yourself. Always copy values from the tool results above.

Reply with the tool call and nothing else, in the form:
CALL name(arg1, arg2)
String arguments are double-quoted. Numbers are bare."""


def user_prompt3(task: Task3) -> str:
    ids = ", ".join(a.aid for a in task.accounts)
    return "Audit these accounts in order: {}\nBegin.".format(ids)
