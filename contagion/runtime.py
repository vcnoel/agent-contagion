"""Executable ledger: runs whatever call the agent actually emits."""
from __future__ import annotations

import json
import re

from .env import RATES, Task

CALL_RE = re.compile(r'CALL\s+([a-z_]+)\s*\((.*?)\)\s*$', re.I | re.S)


def parse(text: str):
    """First CALL line of a generation -> (name, args) or None."""
    for line in text.strip().splitlines():
        m = CALL_RE.match(line.strip())
        if not m:
            continue
        name, raw = m.group(1).lower(), m.group(2).strip()
        try:
            args = json.loads(f"[{raw}]") if raw else []
        except json.JSONDecodeError:
            return name, None
        return name, args
    return None


class Ledger:
    def __init__(self, task: Task):
        self.task = task
        self.by_id = {a.aid: a for a in task.accounts}
        self.submissions: dict[str, float] = {}

    def call(self, name: str, args) -> str:
        if args is None:
            return json.dumps({"error": "could not parse arguments"})
        try:
            if name == "read_account":
                acc = self.by_id.get(args[0])
                if acc is None:
                    return json.dumps({"error": f"no account {args[0]}"})
                return json.dumps({"balance": acc.balance, "tier": acc.tier})
            if name == "get_rate":
                r = RATES.get(str(args[0]).upper())
                if r is None:
                    return json.dumps({"error": f"no tier {args[0]}"})
                return json.dumps({"rate": r})
            if name == "compute":
                return json.dumps({"product": round(float(args[0]) * float(args[1]), 2)})
            if name == "submit":
                self.submissions[str(args[0])] = float(args[1])
                return json.dumps({"status": "recorded"})
        except (IndexError, TypeError, ValueError):
            return json.dumps({"error": "bad arguments"})
        return json.dumps({"error": f"no tool {name}"})

    def score(self) -> dict:
        correct = sum(
            1 for a in self.task.accounts
            if abs(self.submissions.get(a.aid, float("nan")) - a.interest) < 0.011
        )
        return {"correct": correct, "n": len(self.task.accounts),
                "all_pass": int(correct == len(self.task.accounts))}
