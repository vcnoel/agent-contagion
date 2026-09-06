# Preregistration 2: out-of-sample generation predictions

Date: 2026-09-04. Status at time of writing: `results/forced_qwen3.5-4b.jsonl`
may exist on disk but has not been analyzed, plotted, read, or summarized by
anyone or anything. `google/gemma-4-E2B-it` has never been downloaded to this
machine. These predictions are derived from the post-hoc generation trend of
Table `tab:scale` (Qwen2.5 0.5/1.5/3/7B, Qwen3 0.6/1.7/4B, Qwen3.5 0.8/2B) and
are registered so that the trend stops being a postdiction.

Protocol: identical to E1 (`contagion.run_forced`, 30 tasks, 4 accounts,
greedy decoding, cluster bootstrap over tasks). Estimator: unconditional
lambda on independent steps, plus the floor-controlled variant. Thinking
disabled via chat template. Any deviation (quantisation, loader change for the
multimodal Gemma 4) will be recorded next to the result.

## Predictions

**P5a (Qwen3.5-4B, within-family decline).** lambda(indep) is positive with a
95% CI excluding zero, and its point estimate is below Qwen3.5-2B's +0.031.
Predicted interval: [+0.010, +0.030].

**P5b (Qwen3.5-4B, placebo).** Placebo lambda(indep) in [-0.005, +0.005].

**P5c (Qwen3.5-4B, selectivity sharpens with generation).** Same-role lambda
at least 10x different-role lambda; different-role leakage below +0.010
(Qwen3.5-2B measured zero different-role leakage, the sharpest in the paper).

**P5d (Gemma-4-E2B, cross-family generation effect).** lambda(indep) positive
with a 95% CI excluding zero, and below Gemma-2-2B's +0.065. Predicted
interval: [+0.015, +0.050]. Placebo in [-0.005, +0.005].

**P5e (mechanism, both models).** Among failed independent steps at the
analogous position, operator imitation remains the modal failure on
value-producing steps (scripts/imitation.py), and read-step failures remain
predominantly positional (scripts/desync.py revisit+skip > unparseable) --
i.e. newer models keep failing *cleanly*, by rule induction, not by format
collapse.

## Falsification

The generation story of the camera-ready is wrong if: P5a fails high (no
within-family decline in the 2026 generation), P5d fails (a newer 2B-class
model at or above its 2024 counterpart, or a null), or P5c fails (selectivity
does not hold). A null lambda for Gemma-4-E2B would instead echo the Gemma-2-9B
cell and be reported as a family property, not hidden.

SHA-256 of this file at registration is recorded in FREEZE_2.txt alongside
the sha of every results file present on disk at this moment.

## Amendment 1 (2026-09-04, later the same day, before any SmolLM3 download)

**P5f (SmolLM3-3B, third family in the generation contrast).**
`HuggingFaceTB/SmolLM3-3B` (2025 generation, one size only; thinking disabled)
has not been downloaded or run at the time of this amendment. Predictions
under the identical E1 protocol: lambda(indep) positive with a 95% CI
excluding zero and below SmolLM2-1.7B's +0.065; predicted interval
[+0.020, +0.055]. Placebo in [-0.005, +0.005]. Same-role selectivity at
least 4x. The generation story is strengthened if the decline replicates in
a third family, and bounded if it does not. FREEZE_2.txt gains the amended
file's hash as a second dated line; the original line remains valid for the
pre-amendment text.
