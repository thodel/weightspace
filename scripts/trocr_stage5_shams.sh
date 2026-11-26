#!/usr/bin/env bash
# the six sham-tested finals of the preregistration (seed 1 of each subject x setting), discovery half only; card 0 only (waits inside the reader)
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"; export TMPDIR=$HOME/atr-cache/tmp CUDA_VISIBLE_DEVICES=0
AC=$HOME/atr-cache/trocr; LOG=$AC/s5_queue.log
for s in A B; do for cfg in small-lr1e-5 small-lr5e-5 big-lr1e-5; do
  kind=${cfg%%-*}; N=c4$kind-$s-${cfg#*-}-s1
  PYTHONPATH=. $AC/venv/bin/python scripts/trocr_stage5_shams.py $s $N $AC/stage5/$N 2>&1 | grep -E "delta|Error|Traceback" | tee -a $LOG
done; done
echo "finished shams $(date +%H:%M:%S)" | tee -a $LOG
