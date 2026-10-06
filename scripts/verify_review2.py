"""Check the second camera-ready review's data claims."""
import collections
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from contagion.analyze import KINDS, index, injected_kind, load, r0_by_seed  # noqa: E402

print("1: driver R0 per remaining step, all-remaining vs independent-remaining denominators")
for m in ["qwen-0.5b", "qwen3.5-0.8b"]:
    rows = load(f"results/forced_{m}.jsonl")
    clean, inj = index(rows)
    horizon = max(r["s"] for r in rows) + 1
    indep_rem = collections.defaultdict(set)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] == "indep":
            indep_rem[t].add(s)
    allrem = collections.defaultdict(list)
    indrem = collections.defaultdict(list)
    raw = collections.defaultdict(list)
    for t in sorted(indep_rem):
        k = injected_kind(t)
        v = np.mean([float(x[0]) for x in r0_by_seed(clean, inj["err"], t).values()])
        raw[k].append(v)
        allrem[k].append(v / (horizon - 1 - t))
        indrem[k].append(v / len(indep_rem[t]))
    print(f"  {m}")
    for k in KINDS:
        print(f"    {k:8s} raw R0={np.mean(raw[k]):.3f}  /all remaining={np.mean(allrem[k]):.4f}"
              f"  /indep remaining={np.mean(indrem[k]):.4f}")

print("\n3: competence correlation robustness, nine models")
NINE = ["smol-360m", "qwen-0.5b", "llama-1b", "olmo-1b", "qwen-1.5b",
        "smol-1.7b", "gemma-2b", "qwen-3b", "llama-3b"]
acc, lam = [], []
for m in NINE:
    rows = load(f"results/forced_{m}.jsonl")
    clean, inj = index(rows)
    acc.append(np.mean([r["ok_faithful"] for r in clean.values()]))
    d = [clean[(seed, s)]["ok_faithful"] - r["ok_faithful"]
         for (seed, t, s), r in inj["err"].items()
         if r["relation"] == "indep" and (seed, s) in clean]
    lam.append(np.mean(d))
acc, lam = np.array(acc), np.array(lam)
print(f"  pearson all nine = {np.corrcoef(acc, lam)[0, 1]:+.3f}")
keep = np.array([m != "qwen-0.5b" for m in NINE])
print(f"  pearson without qwen-0.5b = {np.corrcoef(acc[keep], lam[keep])[0, 1]:+.3f}")
rho, p = stats.spearmanr(acc, lam)
print(f"  spearman all nine = {rho:+.3f}  p = {p:.3f}")

print("\n2: Qwen2.5 ladder same-role lambda")
for m in ["qwen-0.5b", "qwen-1.5b", "qwen-3b"]:
    rows = load(f"results/forced_{m}.jsonl")
    clean, inj = index(rows)
    d = [clean[(seed, s)]["ok_faithful"] - r["ok_faithful"]
         for (seed, t, s), r in inj["err"].items()
         if r["relation"] == "indep" and r["kind"] == injected_kind(t) and (seed, s) in clean]
    print(f"  {m:10s} same-role lambda = {np.mean(d):+.3f}")

print("\n3b: floor-controlled lambda, rank correlation with clean accuracy")
fl = []
for m in NINE:
    rows = load(f"results/forced_{m}.jsonl")
    clean, inj = index(rows)
    d = [clean[(seed, s)]["ok_faithful"] - r["ok_faithful"]
         for (seed, t, s), r in inj["err"].items()
         if r["relation"] == "indep" and (seed, s) in clean and clean[(seed, s)]["ok_faithful"] == 1]
    fl.append(np.mean(d))
fl = np.array(fl)
rho, p = stats.spearmanr(acc, fl)
print(f"  spearman floor = {rho:+.3f}  p = {p:.3f};  pearson without qwen-0.5b = "
      f"{np.corrcoef(acc[keep], fl[keep])[0, 1]:+.3f}")
