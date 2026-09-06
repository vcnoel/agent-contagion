#!/bin/bash
PY=~/miniconda3/envs/gemma_spectral/python.exe
run () { echo "=== $1 $(date +%H:%M:%S)"; shift; $PY -m contagion.run_forced "$@" 2>&1 | tr '\r' '\n' | grep -Ev "it/s|Loading" | tail -2; }
# subtask period 3 and 5: does the effect track the task's structure or the number 4?
run "period3 qwen-1.5b" --model qwen-1.5b --n-tasks 25 --n-accounts 5 --period 3 --batch-size 24 --out results/p3_qwen-1.5b.jsonl
run "period3 qwen-3b"   --model qwen-3b   --n-tasks 25 --n-accounts 5 --period 3 --batch-size 16 --out results/p3_qwen-3b.jsonl
run "period5 qwen-1.5b" --model qwen-1.5b --n-tasks 20 --n-accounts 3 --period 5 --batch-size 24 --out results/p5_qwen-1.5b.jsonl
run "period5 qwen-3b"   --model qwen-3b   --n-tasks 20 --n-accounts 3 --period 5 --batch-size 16 --out results/p5_qwen-3b.jsonl
# sampled decoding: not an artefact of greedy
run "T0.7 qwen-1.5b" --model qwen-1.5b --n-tasks 30 --n-accounts 4 --temperature 0.7 --batch-size 24 --out results/t07_qwen-1.5b.jsonl
run "T0.7 qwen-3b"   --model qwen-3b   --n-tasks 30 --n-accounts 4 --temperature 0.7 --batch-size 16 --out results/t07_qwen-3b.jsonl
echo "=== robust done $(date +%H:%M:%S)"
