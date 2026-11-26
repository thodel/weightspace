#!/usr/bin/env bash
# Stage 5, C4 queue (docs/TROCR_STAGE5_PREREG.md section 3): train one final, read it on practice and on the discovery half. Card 0 only, waits for wsp.card.ready.
#   scripts/trocr_stage5_c4.sh small|big     (runs every missing (subject, lr, seed) in the fixed order; a finished reading is skipped on rerun)
set -uo pipefail
KIND=${1:?small|big}
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"
export TMPDIR=$HOME/atr-cache/tmp CUDA_VISIBLE_DEVICES=0
AC=$HOME/atr-cache/trocr; OUT=$AC/stage5; LOG=$AC/s5_queue.log
ready() { until ~/Repo/weightspace/.venv/bin/python -c "from wsp.card import ready; ok,m=ready(); raise SystemExit(0 if ok else 3)"; do sleep 120; done; }
disk() { free=$(df --output=avail -BG "$HOME/atr-cache" | tail -1 | tr -dc 0-9); [ "$free" -ge 8 ] || { echo "STOP disk ${free} GB" | tee -a $LOG; exit 4; }; }
has() { grep -q "\"name\": \"$1\", \"what\": \"$2\"" docs/trocr/stage5_reads.jsonl 2>/dev/null; }
if [ "$KIND" = small ]; then MAN=$R/data/practice/trocr-blb-train-1200-seen.jsonl; EP=5; CFG="1e-5 5e-5"; else MAN=$R/data/practice/trocr-blb-train-all.jsonl; EP=1; CFG="1e-5"; fi
for lr in $CFG; do for s in A B; do for seed in 1 2 3; do
  case $s in A) M=dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2;; B) M=dh-unibe__trocr-kurrent;; esac
  NAME=c4${KIND}-$s-lr$lr-s$seed; D=$OUT/$NAME
  if [ ! -f $D/stage5_train.json ]; then
    disk; ready; echo "train $NAME start $(date +%H:%M:%S)" | tee -a $LOG
    ( cd ~/Repo/training-atr-models/engines && PYTHONPATH=$HOME/Repo/training-atr-models/src:. ~/Repo/training-atr-models/.venvs/trocr-train/bin/python $R/scripts/trocr_stage5_train.py --base-model $AC/$M --manifest $MAN --output-dir $D --seed $seed --lrate $lr --epochs $EP ) > $AC/s5_$NAME.train.log 2>&1 || { echo "FAILED train $NAME" | tee -a $LOG; exit 5; }
    rm -rf $D/trainer_tmp
    echo "train $NAME done $(date +%H:%M:%S) $(grep -o '"seconds_total": [0-9.]*' $D/stage5_train.json)" | tee -a $LOG
  fi
  for what in practice discovery; do
    has $NAME $what && continue
    ready; PYTHONPATH=. $AC/venv/bin/python scripts/trocr_stage5_read.py $s $NAME $what --dir $D 2>&1 | tail -1 | tee -a $LOG
  done
done; done; done
echo "finished c4 $KIND $(date +%H:%M:%S)" | tee -a $LOG
