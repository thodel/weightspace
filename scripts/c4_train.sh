#!/usr/bin/env bash
# One C4 run: ketos train from combined-v1 on the corrected data, 5 fixed epochs, GPU card 0 only (>= 12 GB free), see docs/C4_RETRAIN.md.
# usage: c4_train.sh ARM LRATE SEED
set -euo pipefail
ARM=$1; LR=$2; SEED=$3
W=$HOME/atr-cache/wsp/c4; OUT=$W/${ARM}_seed$SEED
mkdir -p "$OUT"; export TMPDIR=$HOME/atr-cache/tmp CUDA_VISIBLE_DEVICES=0
FREE=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
echo "card 0 free before start: $FREE MiB" | tee "$OUT/gpu_precheck.txt"
if [ "$FREE" -lt 12288 ]; then echo "less than 12 GB free: not starting"; exit 3; fi
( while true; do nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/gpu_mem.log"; sleep 15; done ) &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null' EXIT
START=$(date +%s)
$HOME/Repo/weightspace/.venv/bin/ketos --device cuda:0 --threads 4 --workers 4 --seed "$SEED" train \
  --format-type binary --training-data "$W/train_bin.lst" --evaluation-data "$W/val_bin.lst" \
  --output "$OUT" --weights-format coreml --batch-size 16 --schedule cosine --lrate "$LR" \
  --quit fixed --epochs 5 --min-epochs 5 \
  --load $HOME/Repo/weightspace/models/kraken/kraken-bern-combined-v1.mlmodel \
  --resize union --normalization NFD --normalize-whitespace --augment > "$OUT/train.log" 2>&1
echo "wall_seconds $(( $(date +%s) - START ))" > "$OUT/wall.txt"
