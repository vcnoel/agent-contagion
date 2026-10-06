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

- **Contagion across competence.** Across nine instruction-tuned models from
  five families (0.36B to 3.2B), clean per-step accuracy spans 0.648 to
  1.000 while the unconditional lambda sits at 0.069 (standard deviation
  0.016 across models), with a correlation of -0.006. A placebo that rewrites
  the same step without changing its meaning is null for seven of the nine.
  An estimator restricted to steps a model gets right when clean correlates
  with clean accuracy at -0.66, so on steps they can do, more competent
  models are less susceptible. Past 3.2B lambda falls with scale in the three
  families extended upward, and across newer generations in two of three.
- **Structural role.** The effect concentrates on later steps with the same
  structural role as the corrupted one (same-role to different-role ratios of
  4x to 77x), follows that role under filler-step jitter and period changes,
  and grows with repetitions of one wrong rule more than with as many
  different wrong rules, which the paper reads as in-context rule induction.
- **Mitigation.** Flagging the error in its tool result leaves same-role
  contagion unchanged. An immediate visible correction cuts it by 29 to 41
  percent. Replacing the step with a call-shaped redaction placeholder
  removes the error, and most same-role failures in that arm copy the
  placeholder verbatim (52 to 79 percent), so the marker is itself imitated.
- **Escalation.** At equal budget no placement policy transfers across model
  pairs. The best single step kind to protect, identified after the fact,
  differs by pair: the dispatch with a 1.5B rescuer, rate with a 3B rescuer
  of the same driver, the submission on the Qwen3.5 pair. The drivers' own
  teacher-forced decompositions (`scripts/driver_r0.py`) rank the dispatch
  first for both drivers, so they name the best kind on one pair of three,
  and per-step clean error names it on two.

Valentin Noël. *An Error in the Context Is a Demonstration.* SLM-Agents:
1st Workshop on Small Language Models for Agentic Systems, NeurIPS 2026
(poster).

```bibtex
@inproceedings{noel2026error,
  title     = {An Error in the Context Is a Demonstration},
  author    = {No{\"e}l, Valentin},
  booktitle = {SLM-Agents: 1st Workshop on Small Language Models for Agentic
               Systems, NeurIPS},
  year      = {2026}
}
```

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
    python scripts/dose_curve.py "results/dose3c_*.jsonl"  # one rule repeated
    python scripts/dose_curve.py "results/dose2_*.jsonl"   # k different rules
    python scripts/mitigate_analysis.py  # flag / retry / redact arms
    python scripts/free_analysis.py      # escalation policies at equal budget
    python scripts/free_analysis.py "results/matched_*.jsonl"  # one step kind protected
    python scripts/driver_r0.py results/forced_qwen-0.5b.jsonl \
        results/forced_qwen3.5-0.8b.jsonl  # escalation drivers' decompositions
    python scripts/energy.py results/power_log.csv  # GPU energy per run
    python scripts/make_figures.py       # paper figures

Tasks are seeded and conditions are deterministic given the seed, so
`results/*.jsonl` regenerate byte-comparably on the same hardware and
versions.

## Predictions

Every prediction is reported beside its outcome, whether it held or failed.
The first four (P1-P4) were written in the repository during the study and
carry no timestamp that predates the data. `PREREGISTRATION_2.md` registers
the out-of-sample generation predictions (P5a-P5f) before their runs, and
`FREEZE_2.txt` holds its SHA-256 at registration time together with a
manifest of every result file then on disk.
