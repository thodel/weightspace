"""Item 3: a session on a validation-based practice set (docs/CONTROLS_REPORT.md §3).

    .venv/bin/python scripts/run_validation.py xix-blb-v1|bern-combined-v1 a1|a2|a3|replay
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from wsp import run as R
from wsp.data import load, sha256_file
from wsp.metrics import score

REPO = Path(__file__).resolve().parent.parent
M = Path.home() / "Repo/weightspace/models/kraken"
CFG = {
    "xix-blb-v1": dict(subject="kraken-german-xix-blb-v1", practice="blb-val-300", held="blb-eval-hand-b",
                       replay="runs/validation/arms/program_blb_replay.py", donors="`kraken-german-xix-v1` (parent; trained on no BLB data: `seen_by` is empty for every line)",
                       source="eval-side objects of lassberg-blb-gt (the trainer's validation set for this model), 7 whole objects drawn from blb-eval-hand; the training side has no validation part (PHASE note in CONTROLS_REPORT §3)",
                       audit="data/practice/overlap-blb-val-300.json"),
    "bern-combined-v1": dict(subject="kraken-bern-combined-v1", practice="bern-val", held="bern-eval",
                             replay="runs/validation/arms/program_combined_replay.py",
                             donors="`kraken-bern-hertwig-v2` (trained on all 6 Hertwig val pages: 212 lines seen) and `kraken-bern-schlussredaktion-v1` (trained on 8 of the 11 Schlussredaktion val pages: 216 lines seen); donor rule applied through `rule_mask`",
                             source="the 17 pages the trainer reserved for validation in the job of kraken-bern-combined-v1 (pages_val.lst; partition 0.9, page level), 6 Hertwig pages from the 22.10 export (correct images), 11 Schlussredaktion pages",
                             audit="data/practice/overlap-bern-val.json"),
}
PROG = {"replay": "Claude Fable 5.1 (claude-fable-5-1) wrote the original program on 06.11.2026; replayed here unchanged (bern: only the donor-rule masks added)."}
for k in ("a1", "a2", "a3"):
    PROG[k] = "Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/validation/PROGRAMMER_BRIEF.md; wrote the program in advance, did not run it."


def main(model, att):
    c = CFG[model]
    subj = M / f"{c['subject']}.mlmodel"
    prac = REPO / f"data/practice/{c['practice']}.jsonl"
    held = REPO / f"data/heldout/{c['held']}.jsonl"
    pdir = REPO / "runs/validation" / model / att
    prog = Path(c["replay"]) if att == "replay" else pdir / "program.py"
    hyp = (pdir / "preregistration_hypotheses.md").read_text() if att != "replay" else f"(the 8 hypotheses H1–H8 of the original preregistration `runs/20261106-{'german-xix-blb' if model=='xix-blb-v1' else 'bern-combined'}-v1-b10-1/preregistration.md` / the header of `{prog}`)"
    run_dir = R.new(str(subj), str(prac), 10, run_id=f"val-{model}-{att}")
    meta = json.loads((run_dir / "session.json").read_text())
    pm = json.loads(held.with_suffix(".meta.json").read_text()); pr = json.loads(prac.with_suffix(".meta.json").read_text())
    (run_dir / "preregistration.md").write_text(f"""# Pre-registration — {run_dir.name}

Validation-practice control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 3). Written before the first pass; program `{prog}` copied unchanged.

1. **Subject:** `{subj.name}` — tensor digest `{meta['subject_digest']}`, file sha256 `{meta['subject_file_sha256']}`. Donors: {c['donors']}.
2. **Practice set:** `{prac.relative_to(REPO)}`, {pr['n']} lines, sha256 `{meta['practice_sha256']}`. Source: {c['source']}. **Not training data of the subject** (audit `{c['audit']}`: no shared object/page with the subject's training pages, none with the held-out set).
3. **Held-out set:** `{held.relative_to(REPO)}`, {pm['n']} lines, sha256 `{pm['sha256']}`; audited against the practice set at object/page level in the same file.
4. **Budget:** 10 passes (pass 1 = C0). Keep rule PROTOCOL.md §3 (interval below 0, `min_effect` 0) plus the proposed donor rule (§2 item 7): donor edits are judged on lines unseen by that donor (`rule_mask`).
5. **Hypotheses (predicted direction and magnitude):**

{hyp}

6. **Controls:** C0/C1 on practice and held-out before the session (docs/CONTROLS_REPORT.md §3); other attempts on the same model and set.
7. **Programmer:** {PROG[att]}
""")
    shutil.copy(prog, run_dir / "program.py")
    out = subprocess.run(["bash", "-c", R.sandbox_command(run_dir)], capture_output=True, text=True)
    (run_dir / "stdout.log").write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
    print(out.stdout[-1200:], out.stderr[-1200:], flush=True)
    if (run_dir / "final.mlmodel").exists():
        R.heldout(str(run_dir), c["held"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
