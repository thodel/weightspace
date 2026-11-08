#!/usr/bin/env bash
# seeds in sequence, one run at a time, card released between runs
cd "$(dirname "$0")/.."
for s in "$@"; do
  until scripts/seed_null_train.sh "$s" 91 cuda:0; do rc=$?; [ $rc = 3 ] || { echo "seed $s failed rc=$rc"; exit $rc; }; echo "waiting for card 0"; sleep 300; done
  echo "seed $s done"
done
