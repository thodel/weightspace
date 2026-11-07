#!/usr/bin/env bash
# One seed of the seed-variance null: ketos train on CPU, the trainer's hertwig-v2 recipe with a fixed epoch count.
# usage: seed_null_train.sh SEED EPOCHS [OUTDIR]
set -euo pipefail
SEED=$1; EPOCHS=$2
W=$HOME/atr-cache/wsp/seed-null
OUT=${3:-$W/seed$SEED}
mkdir -p "$OUT"; export TMPDIR=$HOME/atr-cache/tmp CUDA_VISIBLE_DEVICES=""
exec $HOME/Repo/weightspace/.venv/bin/ketos --device cpu --threads 12 --workers 4 --seed "$SEED" train \
  --format-type binary --training-data "$W/train_bin.lst" --evaluation-data "$W/val_bin.lst" \
  --output "$OUT" --weights-format coreml --batch-size 16 --schedule cosine --lrate 0.0005 \
  --quit fixed --epochs "$EPOCHS" \
  --load $HOME/Repo/weightspace/models/kraken/kraken-cursive_inzigkofen.mlmodel \
  --resize union --normalization NFD --normalize-whitespace --augment
