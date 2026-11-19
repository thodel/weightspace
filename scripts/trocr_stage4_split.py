#!/usr/bin/env python3
"""Fix the discovery/confirmation split of the held-out objects (docs/TROCR_STAGE4_PREREG.md) before any stage-4 read.

    python scripts/trocr_stage4_split.py            # writes data/heldout/blb-eval-hand.split.json

Uses only the document ids and line counts of data/heldout/blb-eval-hand.jsonl (no model output). Seed 20261119, balanced by line count."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp import stage4

SEED = 20261119
SRC = Path("data/heldout/blb-eval-hand.jsonl")
OUT = Path("data/heldout/blb-eval-hand.split.json")

rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
sp = stage4.split_objects(rows, SEED)
sp.update({"source": str(SRC), "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(), "n_lines": len(rows),
           "rule": "objects (document ids) in random.Random(seed) shuffled order, each to the half with fewer lines so far (ties to discovery)"})
body = json.dumps(sp, indent=1, ensure_ascii=False)
OUT.write_text(body)
sp["sha256_of_file"] = hashlib.sha256(body.encode()).hexdigest()
print(json.dumps({k: sp[k] for k in ("seed", "lines", "n_objects", "source_sha256", "sha256_of_file")}, indent=1))
