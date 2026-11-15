#!/usr/bin/env bash
# Seeds given as arguments, one after the other (docs/TROCR_NULLS_PREREG.md amendment 14.11.2026); stops at the first failure.
# Deletes each run's epoch checkpoints (optimizer states) after it; the final model stays in ~/atr-cache/trocr/seed-null/sN.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
for S in "$@"; do
  "$D/trocr_seed_run.sh" "$S" > "$HOME/atr-cache/trocr/seed-null/run_s$S.log" 2>&1 || { echo "FAILED seed $S $(date -Is)"; exit 1; }
  rm -rf "$HOME/atr-cache/trocr/seed-null/s$S"/checkpoint-*
  echo "done seed $S $(date -Is)"
done
echo "chain finished $(date -Is)"
