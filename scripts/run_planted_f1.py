"""Fable 5.1 rerun of the LLM arms (a), (b) of the planted-defect control (docs/CONTROLS_REPORT.md §11).

    .venv/bin/python scripts/run_planted_f1.py k1-zero-row-e a|b

Same subject, practice set, budget, keep rule, preregistration template and held-out scoring as scripts/run_planted.py;
the program is the one a Claude Fable 5.1 subagent wrote under runs/planted-f1/<sNN>/<arm>/ (sNN is an opaque label so
that the path does not name the defect; the map is runs/planted-f1/map.json, written after all programs were in).
Run id: planted-<cid>-<arm>-f1 (as runs/20261109-val-*-f1-b10-1).

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
_COMMON = ("Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, "
           "tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/{s}/pack/, an opaque folder name) and "
           "runs/planted-f1/PROGRAMMER_BRIEF{u}.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); "
           "not told which defect was planted; never shown the Sonnet runs, their programs or results. ")
PROGRAMMER = {"a": _COMMON + "Wrote its program in advance, did not run it.",
              "b": _COMMON + "Budget 0."}
OPERATOR = ("Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does "
            "the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.")
BUDGET = {"a": 10, "b": 0}
import os
_MAPF = Path(os.environ.get("WSP_F1_MAP", REPO / "runs/planted-f1/map.json"))
MAP = json.loads(_MAPF.read_text()) if _MAPF.exists() else {}


def main(cid, arm):
    unrelated = cid.endswith("-u")             # the donor is kraken-bern-hertwig-v2, not the parent
    base = cid[:-2] if unrelated else cid
    what = CORRUPTIONS[base][0]
    subj = OUT / f"{base}.mlmodel"
    plant = json.loads((OUT / "plant.json").read_text())[base]
    s = {v: k for k, v in MAP.items()}[cid]
    arm_dir = REPO / "runs/planted-f1" / s / arm
    prog = arm_dir / "program.py"
    hyp_file = arm_dir / ("preregistration_hypotheses.md" if arm == "a" else "hypotheses.md")
    hyps = hyp_file.read_text() if hyp_file.exists() else f"(the rules are in the header of `{prog.relative_to(REPO)}`)"
    run_dir = R.new(str(subj), str(PRACTICE), BUDGET[arm], run_id=f"planted-{cid}-{arm}-f1")
    meta = json.loads((run_dir / "session.json").read_text())
    pm = json.loads((REPO / "data/heldout/blb-eval-hand.meta.json").read_text())
    pre = f"""# Pre-registration — {run_dir.name}

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm ({arm}), **Fable 5.1 rerun** of `runs/20261107-planted-{cid}-{arm}-b{BUDGET[arm]}-1` (Sonnet 5.5). Written before the first pass; the program is `{prog.relative_to(REPO)}`, copied unchanged.

1. **Subject:** `{subj.name}` (a corrupted copy of `kraken-german-xix-blb-v1`: {what}; `runs/planted/plant.json`) — tensor digest `{meta['subject_digest']}`, file sha256 `{meta['subject_file_sha256']}`; the uncorrupted model has tensor digest `{plant['original_digest']}`.
   Donor: {'`kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm' if unrelated else 'parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds'}.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `{meta['practice_sha256']}`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, {pm['n']} lines, {len(pm['documents'])} objects, sha256 `{pm['sha256']}`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** {BUDGET[arm]} passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. {'Budget 0: edits are accepted unmeasured (`keep_unmeasured`).' if arm == 'b' else ''}
5. **Hypotheses (predicted direction and magnitude):**

{hyps}

6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** {PROGRAMMER[arm].format(s=s, u='_UNRELATED' if unrelated else '')} {OPERATOR}
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
