"""Read a page list with the trainer's pass engine (transformers, greedy, cap 3,072) under wsp/hf_pass.py.

    python scripts/a4_read.py --adapter DIR|none --pages draw.jsonl --data-root DIR --out raw.jsonl [--mode peft]

Refuses to start unless 14 GB are free on card 0 and no trainer job is waiting (wsp.card). One process, one base load.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from wsp import card, vlm_reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--pages", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="peft")
a = ap.parse_args()
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
adapter = None if a.adapter == "none" else a.adapter
argv = vlm_reference.command(adapter, Path(a.pages), Path(a.data_root), out.with_suffix(".report.json"), 3072)
out.with_suffix(".argv.json").write_text(json.dumps(argv))
jobs = [{"adapter": adapter, "mode": a.mode, "pages": a.pages, "data_root": a.data_root, "out": str(out)}]
out.with_suffix(".jobs.json").write_text(json.dumps(jobs))
ok, why = card.ready(14000)
if not ok:
    sys.exit(f"card 0 not ready: {why}")
T = vlm_reference.TRAINER
env = {**os.environ, "PYTHONPATH": f"{T}/src:{T}/engines", "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
with open(out.with_suffix(".log"), "w") as log:
    r = subprocess.run([str(vlm_reference.PY), str(REPO / "wsp/hf_pass.py"), "--argv", str(out.with_suffix(".argv.json")),
                        "--jobs", str(out.with_suffix(".jobs.json"))], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(T))
sys.exit(r.returncode)
