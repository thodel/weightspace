#!/usr/bin/env python3
"""Stage 5 shams (docs/TROCR_STAGE5_PREREG.md section 8): five shams of the net change (final - subject) of a C4 final, same tensors, same Frobenius norm per tensor,
seed 20261119 (wsp.stage4.sham_deltas), each read once on the DISCOVERY half only. Appends to docs/trocr/stage5_reads.jsonl with name ``sham<i>:<final name>``.
    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/trocr_stage5_shams.py SUBJECT FINAL_NAME FINAL_DIR"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from wsp import metrics, stage4
import scripts.trocr_stage5_read as R
from wsp.trocr_subject import TrocrSubject

subject, name, fdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
rows = R.lines("discovery")
clusters = [metrics.page_of(r["key"]) for r in rows]
base = R.baseline(subject, "discovery")
sub = TrocrSubject(R.MODELS[subject], device="cuda:0", batch=8)
orig = {n: v.copy() for n, v in sub.tensors().items()}
final = TrocrSubject(fdir, device="cpu", check_card=False).tensors()
delta = stage4.edit_delta(orig, final)
done = {r["name"] for r in map(json.loads, R.OUT.read_text().splitlines()) if r["what"] == "discovery"}
for i, d in enumerate(stage4.sham_deltas(delta, 20261119, 5)):
    nm = f"sham{i}:{name}"
    if nm in done:
        continue
    for n in d:
        sub.set_tensor(n, (orig[n].astype("float64") + d[n]).astype("float32"))
    t0 = time.time()
    while True:
        try:
            preds = sub.forward(rows)
            break
        except RuntimeError as e:
            if "no pass" not in str(e):
                raise
            print("card not ready:", e, flush=True)
            time.sleep(120)
    for n in d:
        sub.set_tensor(n, orig[n])
    ed = metrics.per_line_edits([p.text for p in preds], [r["text"] for r in rows])
    b = metrics.bootstrap_delta(base, ed, clusters=clusters, resamples=2000)
    rec = {"subject": subject, "name": nm, "what": "discovery", "source": f"sham of {name}", "lines": len(rows), "cer": sum(e for e, _ in ed) / sum(c for _, c in ed),
           "delta_cer": b["delta_cer"], "ci95_cluster": b["ci95_cluster"], "seconds": round(time.time() - t0, 1), "norm": float(sum((x ** 2).sum() for x in d.values()) ** .5), "edits": ed}
    with R.OUT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(subject, nm, "delta", round(100 * b["delta_cer"], 2), [round(100 * x, 2) for x in b["ci95_cluster"]], flush=True)
