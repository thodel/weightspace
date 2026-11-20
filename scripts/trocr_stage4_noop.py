#!/usr/bin/env python3
"""Harness check of stage 4 (docs/TROCR_STAGE4_PREREG.md, section "Harness check"): a no-op session on a subject, under the file allow-list, with and without the donor.

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/trocr_stage4_noop.py {A|B} {donor|nodonor} [--measure]

The program does nothing: it lists which weight files, held-out lines and other sessions' files it can open (``wsp.landlock.openable`` on candidates
found outside the sandbox), optionally takes the C0 pass, and freezes. The script asserts that only the subject (and the donor, in ``donor`` mode) and the
session's own folder are visible, and that no held-out line and no other run folder can be read. Writes runs/<id>/visible.json."""
import json
import subprocess
import sys
from pathlib import Path

from wsp import run as R
from wsp import landlock

REPO = Path(__file__).resolve().parent.parent
AC = Path.home() / "atr-cache/trocr"
SUBJ = {"A": (AC / "dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2", AC / "dh-unibe__trocr-kurrent-XVI-XVII"),
        "B": (AC / "dh-unibe__trocr-kurrent", AC / "microsoft__trocr-base-handwritten")}
PRACTICE = REPO / "data/practice/trocr-blb-train-1200-seen.jsonl"
PROGRAM = '''import json
from wsp.run import open_session
from wsp import landlock
s = open_session(".")
c = json.load(open("candidates.json"))
vis = {"weights": landlock.openable(c["weights"]), "other": landlock.openable(c["other"])}
json.dump(vis, open("visible.json", "w"), indent=1)
MEASURE
s.freeze()
'''


def main(which, mode, measure=False):
    subject, donor = SUBJ[which]
    run_dir = R.new(str(subject), str(PRACTICE), 1 if measure else 0, run_id=f"trocr-stage4-noop-{which}-{mode}", device="cuda:0" if measure else "cpu",
                    min_effect=0.005, cluster_rule=True, restrict_files=True, donors=[str(donor)] if mode == "donor" else [])
    weights = subprocess.run(["find", str(Path.home() / "atr-cache"), str(REPO / "runs"), "-maxdepth", "6", "-name", "*.safetensors"],
                             capture_output=True, text=True).stdout.split()
    other = [str(p) for p in (REPO / "data/heldout").glob("*.jsonl")] + [str(p) for p in sorted((REPO / "runs").glob("*trocr-planted*/session.json"))[:5]] \
        + [str(REPO / "docs/TROCR_STAGE4_PLAN.md")]
    (run_dir / "candidates.json").write_text(json.dumps({"weights": weights, "other": other}))
    (run_dir / "program.py").write_text(PROGRAM.replace("MEASURE", 's.measure("C0")' if measure else "pass"))
    r = subprocess.run(["bash", "-c", R.sandbox_command(run_dir)], capture_output=True, text=True)
    (run_dir / "sandbox.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-3000:])
    print(run_dir.name, "exit", r.returncode)
    if r.returncode:
        print(r.stderr[-1500:])
        return 1
    vis = json.loads((run_dir / "visible.json").read_text())
    own = {str(p) for p in weights if p.startswith(str(run_dir))}
    allowed = {str(p) for p in subject.glob("*.safetensors")} | own | ({str(p) for p in donor.glob("*.safetensors")} if mode == "donor" else set())
    extra = sorted(set(vis["weights"]) - allowed)
    missing = sorted({str(p) for p in subject.glob("*.safetensors")} - set(vis["weights"]))
    print("weight files visible:", len(vis["weights"]), "| of the", len(weights), "found outside")
    print("beyond subject/donor/own folder:", extra, "| subject missing:", missing, "| other files readable:", vis["other"])
    ok = not extra and not missing and not vis["other"] and (mode == "nodonor") != any(str(p) in vis["weights"] for p in donor.glob("*.safetensors"))
    print("CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], "--measure" in sys.argv))
