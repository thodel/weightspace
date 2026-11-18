#!/usr/bin/env bash
# One programmer session on card 0: scripts/trocr_programmer_one.sh PLANTED_ID ARM  (log in ~/atr-cache/trocr/prog_<id>.<arm>.log)
cd "$(dirname "$0")/.."
export TMPDIR=$HOME/atr-cache/tmp PYTHONPATH=.
echo "start $1 $2 $(date -Is)"
$HOME/atr-cache/trocr/venv/bin/python scripts/run_trocr_programmer.py "$1" "$2" > "$HOME/atr-cache/trocr/prog_$1.$2.log" 2>&1 || { echo "FAILED $1 $2"; tail -5 "$HOME/atr-cache/trocr/prog_$1.$2.log"; exit 1; }
grep -a "^done" "$HOME/atr-cache/trocr/prog_$1.$2.log"
RUN=$(ls -d runs/*-trocr-planted-$1-$2-b* | tail -1)
rm -rf "$RUN/work.trocr" "$RUN/current.trocr"
