"""Fourth environment: free text record review, no tool call syntax at all.

The first three environments share a shape a reviewer can fairly call one
environment in three costumes: a rigid CALL template, arguments in
parentheses, exact match scoring. This one drops all of it. The agent reads
records and answers in ordinary sentences, and an answer is correct when it
states the right value, however it is worded.

Two turns per record, deliberately of different kinds:

  label   apply a stated convention (vendor, slash, order number). The value
          is nowhere in the prompt, so the model must induce and reapply the
          rule. This is where the mechanism of section 3.3 should bite.
  charge  read a number that is printed in the prompt. Nothing has to be
          induced, so a wrong earlier answer teaches nothing.

The two kinds sit in the same task, the same context, and the same
conversation, which makes the comparison between them a within task
dissociation rather than a comparison across experiments.

Two earlier drafts are recorded here because both were wrong in instructive
ways. The first asked the model to apply a percentage discount: clean
accuracy fell to 0.10 on Qwen2.5 1.5B, so the task measured arithmetic, not
contagion. The second made both turns pure lookups: clean accuracy reached
1.000 and lambda was exactly zero, which is the right answer to the wrong
question, since a lookup has no rule to induce.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from .env import Step


@dataclass
class Record:
    vendor: str
    num: int
    charge: float

    @property
    def ref(self) -> str:
        return "{}-{}".format(self.vendor[:3].upper(), self.num % 1000)

    @property
    def label(self) -> str:
        return "{}/{}".format(self.vendor, self.num)


@dataclass
class Task4:
    records: list
    steps: list = field(default_factory=list)
    seed: int = 0

    @property
    def horizon(self) -> int:
        return len(self.steps)

    @property
    def accounts(self):
        return self.records


VENDORS = ["Aldridge", "Bellamy", "Cortez", "Duverger", "Ekstrom",
           "Farrow", "Guerrero", "Halloway"]


def make_task4(n_records: int = 5, seed: int = 0) -> Task4:
    rng = random.Random(seed)
    records, used = [], set()
    for _ in range(n_records):
        while True:
            v = rng.choice(VENDORS)
            n = rng.randint(10_000, 99_999)
            if (v, n % 1000) not in used:
                used.add((v, n % 1000))
                break
        records.append(Record(vendor=v, num=n,
                              charge=round(rng.uniform(120, 9800), 2)))

    steps: list[Step] = []
    for r in records:
        i = len(steps)
        steps.append(Step(i, r.ref, "label",
                          "The file label for {} is {}.".format(r.ref, r.label),
                          "Noted."))
        steps.append(Step(i + 1, r.ref, "charge",
                          "The charge on {} is {:.2f} euros.".format(r.ref, r.charge),
                          "Noted."))
    return Task4(records=records, steps=steps, seed=seed)


SYSTEM4 = """You are reviewing supplier records. Answer each question in one short sentence.

A file label is written as the vendor name, then a slash, then the order number.

State the value plainly. Do not add commentary and do not explain."""


def user_prompt4(task: Task4) -> str:
    lines = ["Records under review:", ""]
    for r in task.records:
        lines.append("  {} - vendor {}, order number {}, charge {:.2f} euros"
                     .format(r.ref, r.vendor, r.num, r.charge))
    lines.append("")
    lines.append("I will ask about them one at a time.")
    return "\n".join(lines)


def question_for(task: Task4, s: int) -> str:
    st = task.steps[s]
    if st.kind == "label":
        return "What is the file label for {}?".format(st.account)
    return "What is the charge on {}?".format(st.account)


NUM = re.compile("\\d[\\d   ,.]*\\d|\\d")


def _to_float(raw):
    """Parse a number in any convention a model might use. Scoring decides
    lambda, so a right answer written 4 520,00 must not be read as 452000.
    Whichever of comma or dot comes last, with one or two digits after it,
    is the decimal separator; every other separator groups thousands."""
    s = raw
    for ch in (" ", " ", " "):
        s = s.replace(ch, "")
    dec = max(s.rfind(","), s.rfind("."))
    if dec != -1 and len(s) - dec - 1 in (1, 2):
        s = s[:dec].replace(",", "").replace(".", "") + "." + s[dec + 1:]
    else:
        s = s.replace(",", "").replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def stated_value(text):
    """The charge an answer asserts. Reference ids end in digits, so a number
    attached to a hyphen belongs to an id and is never the answer."""
    for m in NUM.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == "-":
            continue
        v = _to_float(m.group(0))
        if v is not None:
            return v
    return None


LABEL = re.compile(r"([A-Za-z]+)\s*/\s*(\d+)")


def stated_label(text):
    """The label an answer asserts, normalised. None when the answer states
    nothing of the form name slash number."""
    m = LABEL.search(text)
    return "{}/{}".format(m.group(1).lower(), m.group(2)) if m else None


def answer_ok(gen: str, expected, kind: str, tol: float = 0.02) -> int:
    """Correct when the answer states the right value. The sentence may be
    phrased any way at all, which is the point of this environment."""
    first = gen.strip().split("\n")[0]
    if kind == "label":
        got = stated_label(first)
        return int(got is not None and got == str(expected).lower())
    v = stated_value(first)
    return int(v is not None and abs(v - float(expected)) <= tol)


def expected_value(task: Task4, s: int):
    st = task.steps[s]
    rec = next(r for r in task.records if r.ref == st.account)
    return rec.label if st.kind == "label" else rec.charge
