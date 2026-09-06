"""Smoke test the free text environment before it costs GPU time."""
import random
import sys

sys.path.insert(0, ".")
from contagion.env4 import (answer_ok, expected_value, make_task4,  # noqa: E402
                            question_for, stated_value)
from contagion.run_forced4 import _msgs, build4  # noqa: E402

t = make_task4(5, seed=1)
print("horizon", t.horizon, "kinds", [s.kind for s in t.steps][:6])
clean = build4(t, "clean", None, random.Random(0))
err = build4(t, "err", 2, random.Random(7))
plc = build4(t, "placebo", 2, random.Random(7))

print("\nclean@2 :", clean[2].call)
print("err@2   :", err[2].call)
print("placebo@2:", plc[2].call)
print("propagated@3:", err[3].call)
print("clean@3     :", clean[3].call)

others = [s for s in t.steps if s.account != t.steps[2].account]
print("\ninjection names another record:",
      any(o.account in err[2].call for o in others))
print("independent steps unchanged:",
      all(err[i].call == clean[i].call for i in range(t.horizon)
          if t.steps[i].account != t.steps[2].account))

print("\nscoring, wording invariance:")
for phrasing in ["The charge is 4520.00 euros.",
                 "It comes to 4 520,00 euros",
                 "4520 euros.",
                 "Roughly 4519.99 euros, I think.",
                 "The charge is 9999.00 euros."]:
    print("   ok={}  <- {!r}".format(answer_ok(phrasing, 4520.00), phrasing))

msgs = _msgs(t, clean, 3)
print("\nprompt turns:", len(msgs))
print("last user turn:", msgs[-1]["content"])
print("prior assistant turn:", msgs[-2]["content"])
