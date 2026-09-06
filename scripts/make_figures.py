# -*- coding: utf-8 -*-
"""Paper figures, all read from results/*.jsonl (the JSONL files are the record).

fig1_competence.pdf -- contagion against per-step competence and against scale.
fig2_role.pdf       -- (a) lambda by distance, all models; (b) same-role against
                       different-role with the equality diagonal.
fig3_induction.pdf  -- repeating one wrong rule against k different wrong rules.
fig4_jitter.pdf     -- displacing the analogue moves the effect with it.
"""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contagion.analyze import load, index, cluster_boot, deltas_by_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

NAMES = {"smol-360m": "SmolLM2 360M", "qwen-0.5b": "Qwen2.5 0.5B",
         "llama-1b": "Llama 3.2 1B", "olmo-1b": "OLMo 2 1B",
         "qwen-1.5b": "Qwen2.5 1.5B", "smol-1.7b": "SmolLM2 1.7B",
         "gemma-2b": "Gemma 2 2B", "qwen-3b": "Qwen2.5 3B", "llama-3b": "Llama 3.2 3B"}
PARAMS = {"smol-360m": 0.36, "qwen-0.5b": 0.49, "llama-1b": 1.24, "olmo-1b": 1.48,
          "qwen-1.5b": 1.54, "smol-1.7b": 1.71, "gemma-2b": 2.61, "qwen-3b": 3.09,
          "llama-3b": 3.21}
FAM = {"qwen": "#1f5fa8", "llama": "#c44e52", "smol": "#dd8452",
       "gemma": "#8172b3", "olmo": "#55a868"}
BLUE, RED = "#1f5fa8", "#c44e52"
ORDER = sorted(PARAMS, key=PARAMS.get)


def fam(m):
    return FAM[m.split("-")[0]]


def role_split(path, jitter=False):
    rows = load(path)
    clean, inj = index(rows)
    same, diff = collections.defaultdict(list), collections.defaultdict(list)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep":
            continue
        if jitter:
            tk, sk = r.get("t_kind"), r["kind"]
            if not tk or tk == "note" or sk == "note":
                continue
            is_same = tk == sk
        else:
            is_same = (s % 4) == (t % 4)
        c = clean.get((seed, s))
        if c is not None:
            (same if is_same else diff)[seed].append(c["ok_faithful"] - r["ok_faithful"])

    def to(d):
        return {k: np.asarray(v, float) for k, v in d.items()}

    return cluster_boot(to(same), n=2000), cluster_boot(to(diff), n=2000)


def dist_profile(path, jitter=False, min_n=60):
    rows = load(path)
    clean, inj = index(rows)
    b = collections.defaultdict(list)
    for (seed, t, s), r in inj["err"].items():
        if r["relation"] != "indep":
            continue
        if jitter and (r.get("t_kind") == "note" or r["kind"] == "note"):
            continue
        c = clean.get((seed, s))
        if c is not None:
            b[s - t].append(c["ok_faithful"] - r["ok_faithful"])
    ds = sorted(d for d in b if len(b[d]) >= min_n)
    return np.array(ds), np.array([np.mean(b[d]) for d in ds])


DMAX = 14


def lambda_matrix(models, fn, jitter=False, dmax=DMAX, min_n=60):
    """rows = models, cols = distance 1..dmax, NaN where too few observations."""
    M = np.full((len(models), dmax), np.nan)
    for i, m in enumerate(models):
        d, v = dist_profile(str(ROOT / "results" / fn.format(m)), jitter=jitter,
                            min_n=min_n)
        for dd, vv in zip(d, v):
            if 1 <= dd <= dmax:
                M[i, dd - 1] = vv
    return M


def heat(ax, M, models, vmax, label=None, xlabel=False, ylabel=True):
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("0.93")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax, interpolation="nearest",
                   extent=[0.5, M.shape[1] + 0.5, len(models) - 0.5, -0.5])
    if label:
        ax.text(0.995, 0.94, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.35, lw=0))
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([NAMES[m] for m in models] if ylabel else [])
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    if xlabel:
        ax.set_xlabel("distance from the injected step")
    else:
        ax.set_xticklabels([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


# ---------- fig 1: contagion does not track competence ----------
pts = []
for path in sorted(glob.glob(str(ROOT / "results" / "forced_*.jsonl"))):
    rows = load(path)
    clean, inj = index(rows)
    m = rows[0]["model"]
    # The figures show the preregistered ladder only. Later additions (the
    # 7B and 9B extensions, the Qwen3 and Qwen3.5 families) are reported in
    # the appendix scaling table instead.
    if m not in PARAMS:
        continue
    q = float(np.mean([r["ok_faithful"] for r in clean.values()]))
    lam, lo, hi = cluster_boot(deltas_by_seed(clean, inj["err"], "indep"), n=2000)
    pts.append((m, q, lam, lo, hi))
pts.sort(key=lambda p: PARAMS[p[0]])
lams = np.array([p[2] for p in pts])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, xs, xlab in [(axes[0], [p[1] for p in pts], "clean per-step accuracy"),
                     (axes[1], [PARAMS[p[0]] for p in pts], "parameters (B)")]:
    ax.axhline(lams.mean(), color="gray", lw=0.5, ls=":", zorder=0)
    for (m, q, lam, lo, hi), x in zip(pts, xs):
        ax.errorbar(x, lam, yerr=[[lam - lo], [hi - lam]], fmt="o", ms=4.2,
                    color=fam(m), ecolor=fam(m), elinewidth=0.9, capsize=0, zorder=3)
    ax.set_xlabel(xlab)
    ax.set_ylim(0, max(p[4] for p in pts) * 1.15)

r = np.corrcoef([p[1] for p in pts], lams)[0, 1]
axes[0].set_ylabel(r"contagion $\lambda$")
axes[0].set_title(r"competence  ($r = {:+.3f}$)".format(r), loc="left")
axes[1].set_xscale("log")
axes[1].set_title("scale", loc="left")
axes[1].tick_params(labelleft=False)
axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, color=c, label=k.capitalize())
                        for k, c in FAM.items()],
               frameon=False, loc="upper right", handletextpad=0.3,
               borderaxespad=0.2, labelspacing=0.22)
fig.tight_layout()
fig.savefig(FIG / "fig1_competence.pdf")
fig.savefig(FIG / "fig1_competence.png")
print("wrote fig1_competence.pdf")

# ---------- fig 2: the payload lands on the analogous step ----------
from matplotlib import gridspec  # noqa: E402

M = lambda_matrix(ORDER, "forced_{}.jsonl")
fig = plt.figure(figsize=(7.4, 2.9))
outer = gridspec.GridSpec(1, 4, width_ratios=[1.45, 0.045, 0.10, 1.0], wspace=0.30)
axh = fig.add_subplot(outer[0])
im = heat(axh, M, ORDER, vmax=np.nanmax(M), xlabel=True)
axh.set_title("contagion by distance, at subtask period 4", loc="left")
for x in (4, 8, 12):
    axh.axvline(x, color="white", lw=0.5, ls=":", alpha=0.55)

cax = fig.add_subplot(outer[1])
cb = fig.colorbar(im, cax=cax)
cax.set_title(r"$\lambda$", fontsize=8, pad=4)
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.4)

ax = fig.add_subplot(outer[3])
res = [(m, *role_split(str(ROOT / "results" / "forced_{}.jsonl".format(m)))) for m in ORDER]
lim = (1.5e-3, 0.7)
ax.plot(lim, lim, color="gray", lw=0.7, ls="--", zorder=0)
for m, (a, alo, ahi), (b, blo, bhi) in res:
    ax.errorbar(max(b, lim[0]), a, xerr=[[max(b - blo, 0)], [bhi - b]],
                yerr=[[a - alo], [ahi - a]], fmt="o", ms=4.2, color=fam(m),
                ecolor=fam(m), elinewidth=0.9, capsize=0, zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel(r"$\lambda$, different role")
ax.set_ylabel(r"$\lambda$, same role")
ax.set_title("every model sits above equality", loc="left")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, color=c, label=k.capitalize())
                   for k, c in FAM.items()],
          frameon=False, loc="lower right", handletextpad=0.3, labelspacing=0.22,
          borderaxespad=0.2)
fig.savefig(FIG / "fig2_role.pdf")
fig.savefig(FIG / "fig2_role.png")
print("wrote fig2_role.pdf")


# ---------- fig 3: rule induction ----------
def kcurve(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by = {(r["seed"], r["offset"], r["k"], r["arm"]): r["ok"] for r in rows}
    ks, ms, los, his = [], [], [], []
    for k in sorted({r["k"] for r in rows if r["k"] > 0}):
        d = collections.defaultdict(list)
        for (seed, off, kk, arm), ok in by.items():
            if kk == k and arm == "poisoned":
                c = by.get((seed, off, k, "clean"))
                if c is not None:
                    d[seed].append(c - ok)
        mm, lo, hi = cluster_boot({s: np.asarray(v, float) for s, v in d.items()},
                                  n=2000, seed=k)
        ks.append(k)
        ms.append(mm)
        los.append(lo)
        his.append(hi)
    return np.array(ks), np.array(ms), np.array(los), np.array(his)


models3 = ["qwen-1.5b", "qwen-3b", "llama-1b"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
for ax, m in zip(axes, models3):
    for fn, col, lab in [("dose3c_{}.jsonl", RED, "one wrong rule, repeated"),
                         ("dose2_{}.jsonl", BLUE, r"$k$ different wrong rules")]:
        k, mm, lo, hi = kcurve(str(ROOT / "results" / fn.format(m)))
        ax.fill_between(k, lo, hi, color=col, alpha=0.22, lw=0)
        ax.plot(k, mm, lw=1.4, color=col, label=lab)
    ax.set_title(NAMES[m], loc="left")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel(r"$k$ erroneous demonstrations")
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel(r"$\lambda_k$ at next analogous step")
axes[0].legend(frameon=False, loc="lower right", handletextpad=0.5, labelspacing=0.22)
fig.tight_layout()
fig.savefig(FIG / "fig3_induction.pdf")
fig.savefig(FIG / "fig3_induction.png")
print("wrote fig3_induction.pdf")

# ---------- fig 4: role, not distance ----------
jit_models = [m for m in ORDER if (ROOT / "results" / "jitter_{}.jsonl".format(m)).exists()]
if jit_models:
    Mf = lambda_matrix(jit_models, "forced_{}.jsonl")
    Mj = lambda_matrix(jit_models, "jitter_{}.jsonl", jitter=True, min_n=25)
    vmax = float(np.nanmax([np.nanmax(Mf), np.nanmax(Mj)]))

    fig = plt.figure(figsize=(7.8, 2.9))
    outer = gridspec.GridSpec(1, 4, width_ratios=[1.5, 0.05, 0.18, 0.9], wspace=0.32)
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.14)
    for r_, (mat, lab) in enumerate([(Mf, "fixed period"), (Mj, "filler steps inserted")]):
        ax = fig.add_subplot(inner[r_])
        im = heat(ax, mat, jit_models, vmax=vmax, label=lab, xlabel=(r_ == 1))
        ax.axvline(4, color="white", lw=0.5, ls=":", alpha=0.55)
        if r_ == 0:
            ax.set_title("displacing the analogue moves the effect with it", loc="left")

    cax = fig.add_subplot(outer[1])
    cb = fig.colorbar(im, cax=cax)
    cax.set_title(r"$\lambda$", fontsize=8, pad=4)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)

    ax = fig.add_subplot(outer[3])
    lim = (1e-3, 0.8)
    ax.plot(lim, lim, color="gray", lw=0.7, ls="--", zorder=0)
    for m in jit_models:
        for fn, jitflag, mk in [("forced_{}.jsonl", False, "o"), ("jitter_{}.jsonl", True, "s")]:
            (a, alo, ahi), (b, blo, bhi) = role_split(
                str(ROOT / "results" / fn.format(m)), jitter=jitflag)
            ax.errorbar(max(b, lim[0]), a, xerr=[[max(b - blo, 0)], [bhi - b]],
                        yerr=[[a - alo], [ahi - a]], fmt=mk, ms=4.2, color=fam(m),
                        ecolor=fam(m), elinewidth=0.9, capsize=0,
                        mfc=fam(m) if not jitflag else "white", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"$\lambda$, different role")
    ax.set_ylabel(r"$\lambda$, same role")
    ax.set_title("selectivity is unchanged", loc="left")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, color="0.35",
                                  label="fixed period"),
                       plt.Line2D([], [], marker="s", ls="", ms=4, color="0.35",
                                  mfc="white", label="filler steps")],
              frameon=False, loc="lower right", handletextpad=0.3, labelspacing=0.22,
              borderaxespad=0.2)
    fig.savefig(FIG / "fig4_jitter.pdf")
    fig.savefig(FIG / "fig4_jitter.png")
    print("wrote fig4_jitter.pdf")
