#!/bin/bash
PY=~/miniconda3/envs/gemma_spectral/python.exe
for spec in "smol-360m 24" "llama-1b 24" "olmo-1b 24" "smol-1.7b 16" "gemma-2b 16" "llama-3b 12"; do
  set -- $spec
  echo "=== $1 (batch $2) $(date +%H:%M:%S)" 
  timeout 3600 $PY -m contagion.run_forced --model "$1" --n-tasks 30 --n-accounts 4 \
      --batch-size "$2" --out "results/forced_$1.jsonl" 2>&1 | tr '\r' '\n' | grep -Ev "it/s|Loading" | tail -2
  echo "--- $1 exit=$? $(date +%H:%M:%S)"
done
echo "=== ladder2 done $(date +%H:%M:%S)"
