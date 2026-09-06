#!/bin/bash
PY=~/miniconda3/envs/gemma_spectral/python.exe
$PY -m contagion.run_free --small llama-1b --large gemma-2b --n-tasks 30 --n-accounts 4 \
    --policies none,front,back,conf --budgets 0.125,0.25,0.5 \
    --out results/free_llama-1b_gemma-2b.part2.jsonl 2>&1 | tr '\r' '\n' | grep -Ev "it/s|Loading" | tail -8
echo "=== pair2_part2 done $(date +%H:%M:%S)"
