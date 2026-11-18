#!/usr/bin/env python3
"""Run one non-LLM arm of the TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md): new session, preregistration stub, sandbox, optional held-out.

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/run_trocr_planted.py PLANTED_ID c|d [--heldout]

Needs the planted models (``python -m wsp.trocr_plant make``), ``python -m wsp.card --watch`` running outside the sandbox (a pass inside the
network-cut sandbox reads the card's state from that file), and card 0 with at least 12 GB free. The program is the file under runs/trocr-planted/arms/,
copied unchanged into the run folder.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from wsp import run as R
from wsp.data import sha256_file
from wsp.trocr_plant import OUT

REPO = Path(__file__).resolve().parent.parent
PRACTICE = REPO / "data/practice/trocr-blb-train-300-seen.jsonl"
HELD = "blb-eval-hand"
BUDGET = {"c": 10, "d": 10}
MIN_EFFECT = 0.005


def main(pid, arm, heldout):
    plant = json.loads((OUT / "plant.json").read_text())[pid]
    prog_src = REPO / "runs/trocr-planted/arms" / f"program_{arm}.py"
    run_dir = R.new(str(OUT / pid), str(PRACTICE), BUDGET[arm], run_id=f"trocr-planted-{pid}-{arm}", device="cuda:0",
                    min_effect=MIN_EFFECT, cluster_rule=True)
    meta = json.loads((run_dir / "session.json").read_text())
    shutil.copyfile(prog_src, run_dir / "program.py")
    (run_dir / "preregistration.md").write_text(f"""# Pre-registration stub, {run_dir.name}

Planted-defect control, TrOCR, arm ({arm}) (docs/TROCR_STAGE3_PREREG.md, committed before this run). Program: `runs/trocr-planted/arms/program_{arm}.py`, copied unchanged (sha256 `{sha256_file(prog_src)}`).
Subject: planted `{pid}` = {plant['what']} (defect `{plant['defect']}`, size `{plant['size']}`), tensor digest `{meta['subject_digest']}`, original `{plant['original_digest']}`.
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `{meta['practice_sha256']}`. Held-out: `data/heldout/{HELD}.jsonl`. Budget {BUDGET[arm]} passes, min effect {MIN_EFFECT}, page-clustered interval, generation settings {json.dumps(meta['generation'])}.
""")
    cmd = R.sandbox_command(run_dir)
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    (run_dir / "sandbox.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-4000:])
    print(run_dir.name, "exit", r.returncode)
    if r.returncode:
        print(r.stderr[-1500:])
        return 1
    if heldout:
        R.heldout(str(run_dir), HELD)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], "--heldout" in sys.argv))
