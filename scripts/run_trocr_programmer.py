#!/usr/bin/env python3
"""Run one LLM-programmer session of the TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, section 3).

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/run_trocr_programmer.py PLANTED_ID ARM [--no-heldout]
ARM: ad (a, Sonnet 5.5, with the donor), an (a, Sonnet 5.5, without the donor), f (a2, Fable 5.2, with the donor), b (blind, Sonnet 5.5, with the donor, budget 0).

The program and the hypotheses file are those the programmer wrote under runs/trocr-planted/<sNN>/<arm>/ (sNN is an opaque label, map in runs/trocr-planted/map.json,
written when the packs were built); they are copied unchanged. The preregistration is written, from the hypotheses file and the digests, before the sandbox
starts. The held-out read (blb-eval-hand) is outside the sandbox and happens once per frozen model (cached by digest).
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
MIN_EFFECT = 0.005
BUDGET = {"ad": 10, "an": 10, "f": 10, "b": 0}
WHO = {"ad": "Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor",
       "an": "Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **without** the donor",
       "f": "Claude Fable 5.2 (`claude-fable-5-2`), arm (a2), with the parent donor",
       "b": "Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (b), blind (budget 0), with the parent donor"}
HYP = {"b": "hypotheses.md"}


def main(pid, arm, heldout=True):
    lab = {v: k for k, v in json.loads((REPO / "runs/trocr-planted/map.json").read_text()).items()}[pid]
    adir = REPO / "runs/trocr-planted" / lab / arm
    prog = adir / "program.py"
    hyp = adir / HYP.get(arm, "preregistration_hypotheses.md")
    plant = json.loads((OUT / "plant.json").read_text())[pid]
    run_dir = R.new(str(OUT / pid), str(PRACTICE), BUDGET[arm], run_id=f"trocr-planted-{pid}-{arm}", device="cuda:0",
                    min_effect=MIN_EFFECT, cluster_rule=True)
    meta = json.loads((run_dir / "session.json").read_text())
    shutil.copyfile(prog, run_dir / "program.py")
    pm = json.loads((REPO / f"data/heldout/{HELD}.meta.json").read_text())
    (run_dir / "preregistration.md").write_text(f"""# Pre-registration, {run_dir.name}

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `{arm}`: {WHO[arm]}.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/{lab}/{'pack_nodonor' if arm == 'an' else 'pack'}/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/{lab}/{arm}/program.py`, copied unchanged (sha256 `{sha256_file(prog)}`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `{pid}` = {plant['what']} (tensor digest `{meta['subject_digest']}`; original `{plant['original_digest']}`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `{meta['practice_sha256']}`. Held-out: `data/heldout/{HELD}.jsonl`, {pm['n']} lines, sha256 `{pm['sha256']}`.
Budget {BUDGET[arm]}, min effect {MIN_EFFECT}, page-clustered interval, generation settings {json.dumps(meta['generation'])}.

## Hypotheses (the programmer's file `{hyp.name}`)

{hyp.read_text()}
""")
    r = subprocess.run(["bash", "-c", R.sandbox_command(run_dir)], capture_output=True, text=True)
    (run_dir / "sandbox.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-4000:])
    print(run_dir.name, "exit", r.returncode, flush=True)
    if r.returncode:
        print(r.stderr[-1500:])
        return 1
    if heldout:
        R.heldout(str(run_dir), HELD)
    s = json.loads((run_dir / "summary.json").read_text())
    print("done", run_dir.name, "passes", s["used"], "kept", s["kept"], "CER", round(s["baseline"]["cer"] if s["baseline"] else float("nan"), 4), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], "--no-heldout" not in sys.argv))
