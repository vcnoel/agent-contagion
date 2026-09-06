#!/bin/bash
PY=~/miniconda3/envs/gemma_spectral/python.exe
for spec in "qwen-3b 12" "gemma-2b 12" "llama-3b 12"; do
  set -- $spec
  echo "=== jitter $1 $(date +%H:%M:%S)"
  $PY -m contagion.run_forced --model "$1" --n-tasks 15 --n-accounts 3 --jitter 2 \
      --batch-size "$2" --out "results/jitter_$1.jsonl" 2>&1 | tr '\r' '\n' | grep -Ev "it/s|Loading" | tail -2
  echo "--- $1 done $(date +%H:%M:%S)"
done
echo "=== jitter2 all done $(date +%H:%M:%S)"
