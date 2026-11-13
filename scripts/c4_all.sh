#!/usr/bin/env bash
# arm A (lr 1e-5) seeds 1-3, then arm B (lr 5e-5) seeds 1-3; one at a time; waits (never squeezes) when card 0 lacks 12 GB
cd "$(dirname "$0")/.."
run() { until scripts/c4_train.sh "$1" "$2" "$3"; do rc=$?; [ $rc = 3 ] || { echo "$1 seed $3 failed rc=$rc"; exit $rc; }; echo "waiting for card 0"; sleep 300; done; echo "$1 seed $3 done"; }
for s in 1 2 3; do run A 1e-5 $s; done
for s in 1 2 3; do run B 5e-5 $s; done
