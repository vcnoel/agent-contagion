#!/bin/bash
# Everything still missing, restricted to models < 3B. Batch sizes are kept
# small because other jobs share the GPU.
PY=~/miniconda3/envs/gemma_spectral/python.exe
run () { echo "=== $1 $(date +%H:%M:%S)"; shift; $PY "$@" 2>&1 | tr '\r' '\n' | grep -Ev "it/s|Loading" | tail -6; }

# E3: free-running escalation at equal budget (P4)
run "free qwen-0.5b -> qwen-1.5b" -m contagion.run_free \
    --small qwen-0.5b --large qwen-1.5b --n-tasks 30 --n-accounts 4 \
    --policies none,random,front,back,conf --budgets 0.125,0.25,0.5 \
    --out results/free_qwen-0.5b_qwen-1.5b.jsonl
run "free llama-1b -> gemma-2b" -m contagion.run_free \
    --small llama-1b --large gemma-2b --n-tasks 30 --n-accounts 4 \
    --policies none,random,front,back,conf --budgets 0.125,0.25,0.5 \
    --out results/free_llama-1b_gemma-2b.jsonl

# Mitigation arms (flag / retry / redact)
run "mitigate qwen-1.5b" -m contagion.run_mitigate --model qwen-1.5b \
    --n-tasks 25 --n-accounts 4 --batch-size 12 --out results/mitigate_qwen-1.5b.jsonl
run "mitigate gemma-2b" -m contagion.run_mitigate --model gemma-2b \
    --n-tasks 25 --n-accounts 4 --batch-size 12 --out results/mitigate_gemma-2b.jsonl

# Robustness: sampled decoding and period 5, small models only
run "T0.7 qwen-1.5b" -m contagion.run_forced --model qwen-1.5b \
    --n-tasks 30 --n-accounts 4 --temperature 0.7 --batch-size 12 \
    --out results/t07_qwen-1.5b.jsonl
run "period5 qwen-1.5b" -m contagion.run_forced --model qwen-1.5b \
    --n-tasks 20 --n-accounts 3 --period 5 --batch-size 12 \
    --out results/p5_qwen-1.5b.jsonl
run "period3 gemma-2b" -m contagion.run_forced --model gemma-2b \
    --n-tasks 25 --n-accounts 5 --period 3 --batch-size 12 \
    --out results/p3_gemma-2b.jsonl
echo "=== small_queue done $(date +%H:%M:%S)"
