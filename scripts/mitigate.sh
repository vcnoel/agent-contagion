#!/bin/bash
PY=~/miniconda3/envs/gemma_spectral/python.exe
for spec in "qwen-1.5b 24" "qwen-3b 16" "gemma-2b 16"; do
  set -- $spec
  echo "=== mitigate $1 $(date +%H:%M:%S)"
  $PY -m contagion.run_mitigate --model "$1" --n-tasks 25 --n-accounts 4 \
      --batch-size "$2" --out "results/mitigate_$1.jsonl" 2>&1 | tr '\r' '\n' | grep -Ev "it/s|Loading" | tail -2
done
echo "=== mitigate done $(date +%H:%M:%S)"
