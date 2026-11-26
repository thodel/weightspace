#!/usr/bin/env bash
# Stage 5, sibling average (A) and ceiling comparators (B) (docs/TROCR_STAGE5_PREREG.md section 3): practice readings of every variant, then the discovery half
# for all sibling variants and for the best alpha (lowest practice CER, ties to the smaller alpha) per ceiling comparator. Card 0 only.
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"
export TMPDIR=$HOME/atr-cache/tmp CUDA_VISIBLE_DEVICES=0
AC=$HOME/atr-cache/trocr; LOG=$AC/s5_queue.log; PY="env PYTHONPATH=. $AC/venv/bin/python scripts/trocr_stage5_read.py"
ready() { until ~/Repo/weightspace/.venv/bin/python -c "from wsp.card import ready; ok,m=ready(); raise SystemExit(0 if ok else 3)"; do sleep 120; done; }
has() { grep -q "\"name\": \"$1\", \"what\": \"$2\"" docs/trocr/stage5_reads.jsonl 2>/dev/null; }
rd() { has "$2" "$3" && return 0; ready; $PY $1 "$2" "$3" --variant "$4" 2>&1 | tail -1 | tee -a $LOG; }
V=()
for s in s1 s2 s3 s4 s5 s42; do V+=("A sibpair-$s sibpair:$s:0.6"); done
V+=("A sibmean sibmean:0.6" "A sibavg7 sibavg7")
for k in A D; do for a in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do V+=("B ceil-$k-a$a toward:$k:$a"); done; done
for v in "${V[@]}"; do set -- $v; rd $1 $2 practice $3; done
BEST=$(python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("docs/trocr/stage5_reads.jsonl")]
for k in "AD":
    c=[r for r in rows if r["what"]=="practice" and r["name"].startswith(f"ceil-{k}-a")]
    b=min(c,key=lambda r:(r["cer"],float(r["name"].split("-a")[1])))
    print(b["name"], b["source"])
PY
)
for v in "${V[@]}"; do set -- $v; case $2 in ceil-*) echo "$BEST" | grep -q "^$2 " || continue;; esac; rd $1 $2 discovery $3; done
echo "best alpha per ceiling comparator: $BEST" | tee -a $LOG
echo "finished variants $(date +%H:%M:%S)" | tee -a $LOG
