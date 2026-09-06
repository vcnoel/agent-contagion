#!/bin/bash
PY=~/miniconda3/envs/gemma_spectral/python.exe
for spec in "qwen-0.5b 24" "qwen-3b 16" "qwen-7b 8"; do
  set -- $spec
  echo "=== $1 (batch $2) $(date +%H:%M:%S)"
  $PY -m contagion.run_forced --model "$1" --n-tasks 30 --n-accounts 4 \
      --batch-size "$2" --out "results/forced_$1.jsonl" 2>&1 | grep -Ev "it/s\]|it/s|Loading" | tail -3
done
echo "=== ladder done $(date +%H:%M:%S)"
