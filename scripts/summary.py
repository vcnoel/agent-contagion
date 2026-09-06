"""Cross-model summary: is contagion a function of competence?"""
import collections, glob, sys
import numpy as np
sys.path.insert(0, ".")
from contagion.analyze import load, index, cluster_boot, deltas_by_seed

PARAMS = {"smol-360m":0.36,"qwen-0.5b":0.49,"llama-1b":1.24,"olmo-1b":1.48,
          "smol-1.7b":1.71,"gemma-2b":2.61,"qwen-1.5b":1.54,"llama-3b":3.21,"qwen-3b":3.09}

rows_all = []
print(f"{'model':11s} {'B':>5} {'clean q':>8} {'lambda':>9} {'95% CI':>20} {'placebo':>9} {'d=4 spike':>10}")
for path in sorted(glob.glob("results/forced_*.jsonl")):
    rows = load(path); clean, inj = index(rows); m = rows[0]["model"]
    q = np.mean([r["ok_faithful"] for r in clean.values()])
    lam, lo, hi = cluster_boot(deltas_by_seed(clean, inj["err"], "indep"))
    pla, _, _ = cluster_boot(deltas_by_seed(clean, inj["placebo"], "indep"))
    # periodicity: lambda at d=4 vs mean lambda at d in {1,2,3,5,6}
    at, off = collections.defaultdict(list), collections.defaultdict(list)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep": continue
        c = clean.get((seed, s))
        if c is None: continue
        d = s - t
        (at if d == 4 else off)[seed].append(c["ok_faithful"] - r["ok_faithful"]) \
            if d in (1,2,3,4,5,6) else None
    a = np.mean(np.concatenate([np.array(v) for v in at.values()])) if at else float("nan")
    o = np.mean(np.concatenate([np.array(v) for v in off.values()])) if off else float("nan")
    print(f"{m:11s} {PARAMS.get(m,0):5.2f} {q:8.3f} {lam:+9.4f} [{lo:+.4f},{hi:+.4f}] {pla:+9.4f} "
          f"{a:.3f} vs {o:.3f}")
    rows_all.append((m, PARAMS.get(m, 0), q, lam))

q = np.array([r[2] for r in rows_all]); lam = np.array([r[3] for r in rows_all])
p = np.log10(np.array([r[1] for r in rows_all]))
print(f"\nacross {len(rows_all)} models, 4 families:")
print(f"  lambda mean={lam.mean():.4f}  sd={lam.std(ddof=1):.4f}  range=[{lam.min():.4f}, {lam.max():.4f}]")
print(f"  clean accuracy range = [{q.min():.3f}, {q.max():.3f}]")
print(f"  corr(clean accuracy, lambda) = {np.corrcoef(q, lam)[0,1]:+.3f}")
print(f"  corr(log10 params,   lambda) = {np.corrcoef(p, lam)[0,1]:+.3f}")
print(f"  corr(log10 params, clean acc) = {np.corrcoef(p, q)[0,1]:+.3f}")
