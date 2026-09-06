from __future__ import annotations

from .env import SYSTEM, Step, Task, user_prompt


def messages_upto(task: Task, steps: list[Step], s: int) -> list[dict]:
    """Chat messages containing steps [0, s), ready for the model to emit step s."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(task)}]
    for st in steps[:s]:
        msgs.append({"role": "assistant", "content": st.call})
        msgs.append({"role": "user", "content": f"RESULT {st.result}"})
    return msgs


def normalize(call: str) -> str:
    """Whitespace- and quote-insensitive normal form of a CALL string."""
    c = call.strip().split("\n")[0].strip()
    if c.lower().startswith("call"):
        c = c[4:].strip()
    c = c.replace("'", '"')
    return "".join(c.split())
