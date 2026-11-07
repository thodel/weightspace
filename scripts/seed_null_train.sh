#!/usr/bin/env bash
# One seed of the seed-variance null: ketos train, the trainer's hertwig-v2 recipe, fixed epoch count.
# usage: seed_null_train.sh SEED EPOCHS DEVICE(cpu|cuda:0) [OUTDIR]
# GPU: card 0 only; starts only if >= 12 GB free there; samples own+card memory; writes wall time.
set -euo pipefail
SEED=$1; EPOCHS=$2; DEV=$3
W=$HOME/atr-cache/wsp/seed-null
OUT=${4:-$W/seed$SEED}
mkdir -p "$OUT"; export TMPDIR=$HOME/atr-cache/tmp
if [ "$DEV" = cpu ]; then export CUDA_VISIBLE_DEVICES=""; else
  export CUDA_VISIBLE_DEVICES=0
  FREE=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "card 0 free before start: $FREE MiB" | tee "$OUT/gpu_precheck.txt"
  if [ "$FREE" -lt 12288 ]; then echo "less than 12 GB free: not starting"; exit 3; fi
  DEV=cuda:0
  ( while true; do nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/gpu_mem.log"; sleep 30; done ) &
  SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null' EXIT
fi
START=$(date +%s)
$HOME/Repo/weightspace/.venv/bin/ketos --device "$DEV" --threads 4 --workers 4 --seed "$SEED" train \
  --format-type binary --training-data "$W/train_bin.lst" --evaluation-data "$W/val_bin.lst" \
  --output "$OUT" --weights-format coreml --batch-size 16 --schedule cosine --lrate 0.0005 \
  --quit fixed --epochs "$EPOCHS" --min-epochs "$EPOCHS" \
  --load $HOME/Repo/weightspace/models/kraken/kraken-cursive_inzigkofen.mlmodel \
  --resize union --normalization NFD --normalize-whitespace --augment > "$OUT/train.log" 2>&1
echo "wall_seconds $(( $(date +%s) - START ))" > "$OUT/wall.txt"
