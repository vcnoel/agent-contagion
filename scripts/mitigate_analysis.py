"""Mitigation arms, paired against the clean baseline at the same step."""
import collections
import glob
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from contagion.analyze import cluster_boot  # noqa: E402

ARMS = ["err", "flag", "retry", "redact"]
LABEL = {"err": "error left in place", "flag": "error flagged in the result",
         "retry": "corrected immediately after", "redact": "error redacted"}

for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/mitigate_*.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    model = rows[0]["model"]
    clean = {(r["seed"], r["s"]): r["ok"] for r in rows if r["arm"] == "clean"}
    print("\n{}".format(model))
    print("  {:32s} {:>9} {:>22}".format("arm", "lambda", "95% CI"))
    for arm in ARMS:
        for same_only in (True, False):
            d = collections.defaultdict(list)
            for r in rows:
                if r["arm"] != arm:
                    continue
                if same_only and not r["same_role"]:
                    continue
                c = clean.get((r["seed"], r["s"]))
                if c is not None:
                    d[r["seed"]].append(c - r["ok"])
            m, lo, hi = cluster_boot({k: np.asarray(v, float) for k, v in d.items()}, n=2000)
            tag = LABEL[arm] + (" [same role]" if same_only else " [all indep]")
            print("  {:32s} {:+9.4f} [{:+.4f}, {:+.4f}]".format(tag, m, lo, hi))
