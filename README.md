# An Error in the Context Is a Demonstration

Code, tasks, and result files for a controlled measurement of **error
contagion** in small-model agent trajectories. One error is injected into a
teacher-forced tool-use trajectory; only later steps that are logically
independent of the corrupted one are scored. Any degradation there is the
model treating the error as an in-context demonstration, not task dependence.

The contagion coefficient is

    lambda(t, s) = P(error at s | error injected at t) - P(error at s | clean)

and its sum over the remaining horizon, `R0(t)`, is the expected number of
additional failures one failure at step `t` causes.

## Findings, in brief

- **Contagion does not track competence.** Across nine instruction-tuned
  models from five families (0.36B-3.2B), clean per-step accuracy spans
  0.648-1.000 while lambda sits at 0.069 +/- 0.016; the correlation is
  -0.006. A placebo (same step, semantics-preserving rewrite) is null.
- **The payload lands on the structurally analogous step** (same-role to
  different-role ratios 4x-77x), survives filler-step jitter and period
  changes, and grows with repeated demonstrations of one wrong rule but not
  with an equal number of different wrong rules: rule induction, not a
  corrupted context.
- **Detection is not mitigation.** Flagging the error in its tool result
  changes nothing downstream; visible correction and in-place redaction each
  leave damage of their own.
- **Escalation has a where.** At equal budget, no placement policy transfers
  across model pairs, but the step kind carrying a driver's R0 under teacher
  forcing predicts which single kind is worth protecting in free-running
  rollouts.

The paper is under double-blind review; it will be linked here on
acceptance.

## Layout

    contagion/env.py          ledger environment, tasks, canonical trajectories
    contagion/env2-4.py       warehouse, semi-structured, and free-text environments
    contagion/inject.py       clean / placebo / err injection operators
    contagion/runtime.py      executable ledger for free-running rollouts
    contagion/run_forced.py   E1: teacher-forced per-step measurement
    contagion/run_dose.py     E2: k-shot poisoning, one rule vs k different rules
    contagion/run_mitigate.py E2b: flag / retry / redact arms
    contagion/run_free.py     E3: escalation policies at equal budget
    contagion/run_closed.py   closed-loop injection into free-running trajectories
    contagion/analyze.py      paired, cluster-bootstrapped estimation
    scripts/*.py              per-experiment analyses; every reported number
    scripts/*.sh, *.ps1       the exact run configurations
    results/*.jsonl           the complete measurement record

## Reproducing

Environment: `pip install -r requirements.txt`. Every experiment runs on one
16 GB consumer GPU, under an hour per model.

Rerun a measurement (models are keyed in `contagion/models.py`):

    python -m contagion.run_forced --model qwen-1.5b --n-tasks 30 --n-accounts 4

Recompute any reported number from the shipped record, no GPU needed:

    python -m contagion.analyze results/forced_qwen-1.5b.jsonl
    python scripts/summary.py            # cross-model table
    python scripts/jitter_analysis.py results/jitter_*.jsonl
    python scripts/dose_curve.py         # rule-induction dose response
    python scripts/mitigate_analysis.py  # flag / retry / redact arms
    python scripts/free_analysis.py      # escalation at equal budget
    python scripts/make_figures.py       # paper figures

Tasks are seeded and conditions are deterministic given the seed, so
`results/*.jsonl` regenerate byte-comparably on the same hardware and
versions.

## Preregistration

Predictions were registered before their data existed and are reported
whether they held or failed. `PREREGISTRATION_2.md` registers the
out-of-sample generation predictions (P5a-P5f); `FREEZE_2.txt` holds its
SHA-256 at registration time together with a manifest of every result file
then on disk.
