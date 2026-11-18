#!/usr/bin/env bash
# Arms (c) and (d) on the planted TrOCR models, one session at a time on card 0 (docs/TROCR_STAGE3_PREREG.md). Practice level; no held-out here.
# A session that ends unchanged keeps no copy of the model (its digest is the corrupted model's).
cd "$(dirname "$0")/.."
export TMPDIR=$HOME/atr-cache/tmp PYTHONPATH=.
PY=$HOME/atr-cache/trocr/venv/bin/python
for ID in "$@"; do
  for ARM in c d; do
    echo "start $ID $ARM $(date -Is)"
    $PY scripts/run_trocr_planted.py "$ID" "$ARM" > "$HOME/atr-cache/trocr/planted_$ID.$ARM.log" 2>&1 || { echo "FAILED $ID $ARM"; tail -3 "$HOME/atr-cache/trocr/planted_$ID.$ARM.log"; exit 1; }
    RUN=$(ls -d runs/*-trocr-planted-$ID-$ARM-b* | tail -1)
    $PY - "$RUN" <<'P'
import json, shutil, sys
from pathlib import Path
r = Path(sys.argv[1]); s = json.loads((r / "summary.json").read_text())
if s["final_digest"] == s["start_digest"]:
    for d in ("final.trocr", "current.trocr"):
        shutil.rmtree(r / d, ignore_errors=True)
for d in ("work.trocr",):
    shutil.rmtree(r / d, ignore_errors=True)
print("done", r.name, "passes", s["used"], "kept", s["kept"], "CER", round(s["baseline"]["cer"], 4), "->", round(s["last_pass"]["cer"], 4), "(last pass, not the kept state)")
P
  done
done
echo "baselines finished $(date -Is)"
