#!/usr/bin/env bash
# Runs scripts/trocr_stage4_confirm.py FINAL_DIR on card 0; a pass refused by wsp.card.ready ("no pass:") waits 5 minutes and resumes (rows are appended per read).
cd "$(dirname "$0")/.."
export TMPDIR=$HOME/atr-cache/tmp PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
PY=$HOME/atr-cache/trocr/venv/bin/python
L=runs/trocr-stage4-confirm
for i in $(seq 1 400); do
  $PY scripts/trocr_stage4_confirm.py "$1" >> $L/chain.log 2> $L/last_err.log && { echo "confirmation done" >> $L/chain.log; exit 0; }
  if grep -q "no pass:" $L/last_err.log; then echo "card not ready, waiting" >> $L/chain.log; sleep 300; else echo FAILED >> $L/chain.log; tail -8 $L/last_err.log >> $L/chain.log; exit 1; fi
done
