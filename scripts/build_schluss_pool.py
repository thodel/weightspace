"""Bern A V 1447 (Schlussredaktion) as out-of-domain material for kraken-bern-hertwig-v2 (docs/MODEL_FAMILY.md,
08.11.2026): another scribe of the same disputation records, which hertwig-v2 (trained on A V 1443 only) never saw.

    python scripts/build_schluss_pool.py

The 118 pages of ``A_V_1447_Schlussredaktion_train`` (24.10 = 25.10 export) are split by page, before any reading:
held-out candidates are the pages none of whose lines occurs in any existing list under data/practice or
data/heldout (those pages were read before, by other studies); ``random.Random(20261108)`` draws a quarter of all
118 pages from them as the held-out side; every other page is the practice pool. The 8 pages of
``A_V_1447_Schlussredaktion_test`` (bern-eval) are not used here. Writes the same files as build_ood_pools.py.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from wsp.data import load, page_lines, sha256_file  # noqa: E402

SRC = Path.home() / "atr-cache/wsp/hf/bern-disputation-1528-rev20261025/data/train/A_V_1447_Schlussredaktion_train/20261025_170131-0000.parquet"
OUT = Path.home() / "atr-cache" / "wsp" / "ood" / "schluss"
PROJECT = "A_V_1447_Schlussredaktion_train"

t = pq.read_table(SRC)
names = [Path(f).stem for f in t.column("filename").to_pylist()]
used = set()
for d in ("practice", "heldout"):
    for lst in (REPO / "data" / d).glob("*.jsonl"):
        for r in load(lst):
            k = r["key"].split("/")
            if len(k) >= 3 and k[0] == PROJECT:
                used.add(k[1])
free = sorted(n for n in names if n not in used)
k = round(0.25 * len(names))
held = set(random.Random(20261108).sample(free, min(k, len(free))))
OUT.mkdir(parents=True, exist_ok=True)
rec = {"source": "schluss", "dir": str(SRC), "sha256_source": sha256_file(SRC), "pages": len(names),
       "pages_in_earlier_lists": sorted(used), "split": "random.Random(20261108).sample(pages in no earlier list, round(0.25 * 118)) -> held-out"}
for side, pick in (("pool", lambda n: n not in held), ("heldout", lambda n: n in held)):
    idx = sorted((i for i, n in enumerate(names) if pick(n)), key=lambda i: names[i])
    tab = t.take(pa.array(idx))
    p = OUT / f"{side}.parquet"; pq.write_table(tab, p)
    rec[side] = {"parquet": str(p), "sha256": sha256_file(p), "pages": len(idx), "page_names": [names[i] for i in idx],
                 "lines": sum(len(page_lines(t.column("xml_content")[i].as_py())) for i in idx)}
(REPO / "data" / "ood").mkdir(parents=True, exist_ok=True)
(REPO / "data" / "ood" / "schluss.split.json").write_text(json.dumps(rec, indent=1))
print({s: (rec[s]["pages"], rec[s]["lines"]) for s in ("pool", "heldout")}, "pages in earlier lists:", len(used))
