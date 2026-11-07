"""Run one arm of the planted-defect control: new session, preregistration, sandbox, held-out (docs/CONTROLS_REPORT.md).

    .venv/bin/python scripts/run_planted.py k1-zero-row-e a|b|c|d

The preregistration is written, from the arm's hypotheses file and the digests, before the sandbox starts; the program is the
file under runs/planted/<cid>/<arm>/ (a, b) or runs/planted/arms/ (c, d), copied unchanged.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from wsp import run as R
from wsp.data import sha256_file
from wsp.plant import OUT, SUBJECT, PARENT, CORRUPTIONS

REPO = Path(__file__).resolve().parent.parent
PRACTICE = REPO / "data/practice/blb-train-300.jsonl"
HELD = "blb-eval-hand"
PROGRAMMER = {   # (a), (b): Claude Sonnet 5.5 subagents as below; for the unrelated-donor and k9-k11 variants as the 'a'/'b' entries
    "a": "Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.",
    "b": "Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.",
    "c": "none: a fixed hand-written list of repairs (runs/planted/arms/program_c.py; program_c_u.py with the donor swapped for the unrelated-donor variants), written by the session's author (Claude Sonnet 5.5) before any pass, identical for every model.",
    "d": "none: plain coordinate search over output biases (runs/planted/arms/program_d.py), identical for every model.",
}
BUDGET = {"a": 10, "b": 0, "c": 10, "d": 10}


def main(cid, arm):
    unrelated = cid.endswith("-u")             # the donor is kraken-bern-hertwig-v2, not the parent
    base = cid[:-2] if unrelated else cid
    what = CORRUPTIONS[base][0]
    subj = OUT / f"{base}.mlmodel"
    plant = json.loads((OUT / "plant.json").read_text())[base]
    arm_dir = REPO / "runs/planted" / cid / arm if arm in "ab" else REPO / "runs/planted/arms"
    prog = arm_dir / "program.py" if arm in "ab" else arm_dir / (f"program_{arm}_u.py" if unrelated and arm == "c" else f"program_{arm}.py")
    hyp_file = arm_dir / ("preregistration_hypotheses.md" if arm == "a" else "hypotheses.md" if arm == "b" else "none")
    hyps = hyp_file.read_text() if hyp_file.exists() else f"(the rules are in the header of `{prog.relative_to(REPO)}`)"
    run_dir = R.new(str(subj), str(PRACTICE), BUDGET[arm], run_id=f"planted-{cid}-{arm}")
    meta = json.loads((run_dir / "session.json").read_text())
    pm = json.loads((REPO / "data/heldout/blb-eval-hand.meta.json").read_text())
    pre = f"""# Pre-registration — {run_dir.name}

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm ({arm}). Written before the first pass; the program is `{prog.relative_to(REPO)}`, copied unchanged.

1. **Subject:** `{subj.name}` (a corrupted copy of `kraken-german-xix-blb-v1`: {what}; `runs/planted/plant.json`) — tensor digest `{meta['subject_digest']}`, file sha256 `{meta['subject_file_sha256']}`; the uncorrupted model has tensor digest `{plant['original_digest']}`.
   Donor: {'`kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm' if unrelated else 'parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds'}.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `{meta['practice_sha256']}`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, {pm['n']} lines, {len(pm['documents'])} objects, sha256 `{pm['sha256']}`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** {BUDGET[arm]} passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. {'Budget 0: edits are accepted unmeasured (`keep_unmeasured`).' if arm == 'b' else ''}
5. **Hypotheses (predicted direction and magnitude):**

{hyps}

6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** {PROGRAMMER[arm]}
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
"""
    (run_dir / "preregistration.md").write_text(pre)
    shutil.copy(prog, run_dir / "program.py")
    print("preregistration written; starting sandbox", flush=True)
    cmd = R.sandbox_command(run_dir)
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    (run_dir / "stdout.log").write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
    print(out.stdout[-1500:], out.stderr[-1500:], flush=True)
    if not (run_dir / "final.mlmodel").exists():
        print("NO final.mlmodel: the program did not freeze", flush=True)
        return run_dir
    R.heldout(str(run_dir), HELD)
    return run_dir


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
