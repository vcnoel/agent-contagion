"""Check the camera-ready review's data claims against the result files."""
import collections
import glob
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import load, index  # noqa: E402

def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

print("F2: clean per-step accuracy by kind, escalation drivers")
for m in ["qwen-0.5b", "qwen3.5-0.8b"]:
    acc = collections.defaultdict(list)
    for r in rows(f"results/forced_{m}.jsonl"):
        if r["cond"] == "clean":
            acc[r["kind"]].append(r["ok_faithful"])
    print(f"  {m:13s}", "  ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(acc.items())))

print("\nF5: floor-controlled lambda vs clean accuracy, preregistered nine")
NINE = ["smol-360m", "qwen-0.5b", "llama-1b", "olmo-1b", "qwen-1.5b",
        "smol-1.7b", "gemma-2b", "qwen-3b", "llama-3b"]
accs, floors = [], []
for m in NINE:
    rs = load(f"results/forced_{m}.jsonl")
    clean, inj = index(rs)
    q = np.mean([r["ok_faithful"] for r in clean.values()])
    d = [c["ok_faithful"] - r["ok_faithful"] for (seed, t, s), r in inj["err"].items()
         if r["relation"] == "indep" and (c := clean.get((seed, s))) and c["ok_faithful"] == 1]
    accs.append(q); floors.append(np.mean(d))
    print(f"  {m:11s} clean={q:.3f}  floor lambda={np.mean(d):+.4f}")
r = np.corrcoef(accs, floors)[0, 1]
# two-sided permutation p for n=9
rng = np.random.default_rng(0)
perm = np.mean([abs(np.corrcoef(accs, rng.permutation(floors))[0, 1]) >= abs(r)
                for _ in range(20000)])
print(f"  corr(clean acc, floor lambda) = {r:+.3f}   permutation p = {perm:.3f}")

print("\nF6: placeholder copies among same-role redaction failures on clean-correct steps")
for p in sorted(glob.glob("results/mitigate_*.jsonl")):
    rs = rows(p)
    clean = {(r["seed"], r["s"]): r["ok"] for r in rs if r["arm"] == "clean"}
    for role in (1, 0):
        f = [r for r in rs if r["arm"] == "redact" and r["same_role"] == role
             and clean.get((r["seed"], r["s"])) == 1 and r["ok"] == 0]
        cp = sum("[redacted]" in r["gen"] for r in f)
        print(f"  {rs[0]['model']:11s} same_role={role}  {cp}/{len(f)} copy the placeholder")

print("\nF12: Llama pair policies actually run")
for p in sorted(glob.glob("results/free_llama-1b_gemma-2b*.jsonl")):
    print(" ", p, dict(collections.Counter((r["policy"], r["budget"]) for r in rows(p))))
