#!/bin/bash
PY=~/miniconda3/envs/gemma_spectral/python.exe
for spec in "qwen-1.5b 24" "qwen-3b 16" "llama-1b 24" "gemma-2b 16"; do
  set -- $spec
  echo "=== dose $1 $(date +%H:%M:%S)"
  $PY -m contagion.run_dose --model "$1" --n-tasks 40 --n-accounts 6 --max-k 4 \
      --batch-size "$2" --out "results/dose2_$1.jsonl" 2>&1 | tr '\r' '\n' | grep -Ev "it/s|Loading" | tail -2
done
echo "=== dose2 done $(date +%H:%M:%S)"
