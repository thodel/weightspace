"""The trainer's own scorer on a page draw: the reference every vLLM pass is held to.

Builds the ``evaluate_qlora`` command line from the v2 job's recorded
parameters with the trainer's ``vlm_cmd.evaluate_cmd`` and runs it in the
trainer's ``vlm-train-tf5`` venv on card 0. Greedy, transformers, the held-out
cap of 3,072 (HELDOUT_BOUNDARY.md, #341). This is gate G1's reference and the
path of the one held-out score (docs/PHASE5_PLAN.md §3, §7 decision 3).

    python -m wsp.vlm_reference --adapter <peft dir> --pages <draw>.jsonl \\
        --data-root <set dir> --report ~/atr-cache/wsp/passes/<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

#: The account's home, from where this repo lies (~/Repo/weightspace), not $HOME or the
#: password database: the sandbox sets HOME to the run folder and ``unshare -U`` maps the
#: uid to nobody, whose home is /nonexistent (wsp.run.sandbox_command).
HOME = Path(__file__).resolve().parents[3]
TRAINER = HOME / "Repo/training-atr-models"
PY = TRAINER / ".venvs/vlm-train-tf5/bin/python"
JOB = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/archive/train-manifests/"
           "20261029T133027Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v2/job.json")

_ARGV = """
import json, sys
from atr_training.contracts import VlmTrainParams
from atr_training.vlm_cmd import evaluate_cmd
job = json.load(open(sys.argv[1]))
p = VlmTrainParams(**job["request"]["params"]).model_copy(update={"max_new_tokens": int(sys.argv[6]),
                                                                  "eval_samples": int(sys.argv[7])})
adapter = sys.argv[2]
cmd = evaluate_cmd(sys.executable, params=p, base_model=job["request"]["base_model"],
                   adapter_dir=adapter or "-", val_jsonl=sys.argv[3], data_root=sys.argv[4], report=sys.argv[5])
if not adapter:
    i = cmd.index("--adapter"); cmd[i:i + 2] = ["--no-adapter"]
print(json.dumps(cmd))
"""


def child_env() -> dict:
    """The trainer's venv with the account's HF cache and temp dir, whatever HOME the caller has."""
    return {**os.environ, "HOME": str(HOME), "HF_HUB_CACHE": str(HOME / ".cache/huggingface/hub"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TMPDIR": str(HOME / "atr-cache/tmp"),
            "PYTHONPATH": f"{TRAINER}/src:{TRAINER}/engines", "CUDA_VISIBLE_DEVICES": "0"}


def command(adapter: str | None, pages: Path, data_root: Path, report: Path, max_new_tokens: int) -> list[str]:
    n = sum(1 for l in pages.read_text(encoding="utf-8").splitlines() if l.strip())
    env = child_env()
    out = subprocess.run([str(PY), "-c", _ARGV, str(JOB), adapter or "", str(pages), str(data_root),
                          str(report), str(max_new_tokens), str(n)],
                         capture_output=True, text=True, env=env, check=True)
    return json.loads(out.stdout)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.vlm_reference")
    ap.add_argument("--adapter", help="omitted = the base alone (--no-adapter)")
    ap.add_argument("--pages", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cmd = command(a.adapter, Path(a.pages), Path(a.data_root), Path(a.report), a.max_new_tokens)
    print(" ".join(cmd), flush=True)
    if a.dry_run:
        return 0
    return subprocess.run(cmd, env=child_env(), cwd=str(TRAINER)).returncode


if __name__ == "__main__":
    sys.exit(main())
