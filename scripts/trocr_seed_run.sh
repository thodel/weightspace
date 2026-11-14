#!/usr/bin/env bash
# One seeded run of the recorded recipe of job 20261026T064540Z (docs/TROCR_NULLS_PREREG.md). Usage: scripts/trocr_seed_run.sh SEED
# Reads the job's compiled manifests (read only), writes only ~/atr-cache/trocr/seed-null/sSEED. Card 0 only, and only when it is ready.
set -euo pipefail
S=$1
JOB=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs/20261026T064540Z-trocr-german-kurrent-xvi-xix-from-kurrent-v2
OUT=$HOME/atr-cache/trocr/seed-null/s$S
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT" "$HOME/atr-cache/tmp"
export TMPDIR=$HOME/atr-cache/tmp CUDA_VISIBLE_DEVICES=0
cd "$REPO" && /home/tobias/Repo/weightspace/.venv/bin/python -c "from wsp.card import ready; ok,m=ready(); print(m); raise SystemExit(0 if ok else 3)"
free_gb=$(df --output=avail -BG "$HOME/atr-cache" | tail -1 | tr -dc 0-9); [ "$free_gb" -ge 3 ] || { echo "disk under 3 GB"; exit 4; }
export PYTHONPATH=/home/tobias/Repo/training-atr-models/src:/home/tobias/Repo/training-atr-models/engines
cd /home/tobias/Repo/training-atr-models/engines
echo "start seed $S $(date -Is)"
/home/tobias/Repo/training-atr-models/.venvs/trocr-train/bin/python -m trocr_train_svc.train_trocr \
  --train-manifest "$JOB/data/train.jsonl" --val-manifest "$JOB/data/val.jsonl" --data-root "$JOB" \
  --output-dir "$OUT" --base-model dh-unibe/trocr-kurrent-XVI-XVII --seed "$S" --device cuda:0 \
  --max-new-tokens 256 --beam-size 4 --length-penalty 1.0 --no-repeat-ngram-size 3 --epochs 3 --batch-size 8 \
  --accumulate-grad-batches 1 --lrate 5e-05 --lr-scheduler cosine --warmup-ratio 0.1 --weight-decay 0.0 \
  --max-grad-norm 1.0 --optim adamw_torch --workers 8 --precision bf16 --gradient-checkpointing
echo "done seed $S $(date -Is)"
