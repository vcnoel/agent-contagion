"""GPU energy per run, integrated from a 1 Hz nvidia-smi power log.

Usage: python scripts/energy.py logs_power.csv

Runs appear in the log as contiguous segments of sustained utilisation; gaps
(downloads, idle) separate them. Each segment is reported with its duration,
mean draw, energy, and -- for E1 runs, which are 7,680 evaluation points --
joules per evaluation point.
"""
import sys
from datetime import datetime

import numpy as np

rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        continue
    try:
        t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
        rows.append((t, float(parts[1]), float(parts[2]), float(parts[3])))
    except ValueError:
        continue

# contiguous high-utilisation segments, allowing brief dips
segs, cur, low = [], [], 0
for r in rows:
    if r[2] >= 30:
        cur.append(r); low = 0
    elif cur:
        low += 1
        cur.append(r)
        if low > 20:                      # 20 s of idle ends a segment
            segs.append(cur[:-low]); cur, low = [], 0
if cur:
    segs.append(cur[: len(cur) - low if low else len(cur)])

print(f"{'segment':>7} {'start':>9} {'min':>6} {'mean W':>7} {'peak mem MB':>12} "
      f"{'Wh':>7} {'J/point (7680)':>15}")
for i, s in enumerate(segs):
    dur = (s[-1][0] - s[0][0]).total_seconds()
    if dur < 180:
        continue
    p = np.array([x[1] for x in s])
    joules = float(np.trapezoid(p, dx=dur / max(len(p) - 1, 1)))
    print(f"{i:>7} {s[0][0].strftime('%H:%M:%S'):>9} {dur/60:>6.1f} {p.mean():>7.1f} "
          f"{max(x[3] for x in s):>12.0f} {joules/3600:>7.2f} {joules/7680:>15.2f}")
