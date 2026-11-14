"""Fable 5.1 rerun of the LLM arms (a), (b) of the adapter planted-defect control (stage A3; arms (a), (b) existed for p3 only).

    PYTHONPATH=. .venv/bin/python scripts/run_planted_adapter_f1.py p3-alpha2 a|b [q01|q02]

The third argument is the pack folder: q01 = first attempt (11.11.), q02 = second attempt with fresh programmers (same pack,
title `q02`; the programmers were told not to read q01/).

As scripts/run_planted_adapter.py (same subject, practice pass, budget, keep rule, score set, outcome classes), except: the program
is the one a Claude Fable 5.1 subagent wrote under runs/planted-adapter-f1/q01/<arm>/ (q01 = p3-alpha2; the pack's title, which
named the defect in the Sonnet runs, was replaced by `q01`), run id planted-adapter-<cid>-<arm>-f1, and the pass server's spool
is ~/atr-cache/wsp/spool/a3-f1 (the a3 spool holds a STOP file from 07.11.).

The practice pass is the 32 pages of data/practice/kurrent-val-32.jsonl; the frozen adapter is scored once on the 64 pages of
data/practice/kurrent-val-holdout-64.jsonl (validation side, disjoint pages; the test side stays closed). Passes go through a running
wsp.vllm_server (spool ~/atr-cache/wsp/spool/a3): fast, and not run-to-run deterministic (docs/ADAPTER_REPORT.md, A0).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from wsp import plant_adapter as P
from wsp import run as R
from wsp.data import sha256_file

REPO = Path(__file__).resolve().parent.parent
PRACTICE = REPO / "data/practice/kurrent-val-32.jsonl"
SCORESET = REPO / "data/practice/kurrent-val-holdout-64.jsonl"
SIDE = "/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-val"
CLEAN = Path.home() / "atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2"
BUDGET = {"a": 6, "b": 0, "c": 4}
_C = ("Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash "
      "(py_compile only); read only runs/planted-adapter-f1/q01/pack/pack.md and runs/planted-adapter-f1/PROGRAMMER_BRIEF.md (the Sonnet brief with the model id "
      "and the read-restriction changed); not told which defect was planted; never shown the Sonnet runs or results. Operator: a Claude Opus 5.5 session that knows "
      "the defect and the Sonnet results and wrote no hypothesis or edit. ")
PROGRAMMER = {"a": _C + "Wrote its program in advance, did not run it.", "b": _C + "Budget 0."}
SPOOL = Path.home() / "atr-cache/wsp/spool/a3-f1"


def main(cid, arm, pack="q01"):
    plant = json.loads((P.OUT / "plant.json").read_text())
    subj = P.OUT / cid
    assert cid == "p3-alpha2"
    arm_dir = REPO / "runs/planted-adapter-f1" / pack / arm
    prog = arm_dir / "program.py"
    hyp_file = arm_dir / ("preregistration_hypotheses.md" if arm == "a" else "hypotheses.md")
    hyps = hyp_file.read_text() if hyp_file.exists() else f"(the rules are in the header of `{prog.relative_to(REPO)}`)"
    run_dir = R.new(str(subj), str(PRACTICE), BUDGET[arm], run_id=f"planted-adapter-{cid}-{arm}-f1", data_root=SIDE,
                    spool=str(SPOOL))
    meta = json.loads((run_dir / "session.json").read_text())
    pre = f"""# Pre-registration — {run_dir.name}

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm ({arm}), **Fable 5.1 rerun** (pack `{pack}`, attempt {1 if pack == 'q01' else 2}) of `runs/20261107-planted-adapter-{cid}-{arm}-b{BUDGET[arm]}-1` (Sonnet 5.5). Written before the first pass; the program is `{prog.relative_to(REPO)}`, copied unchanged.

1. **Subject:** `{cid}`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {json.dumps(plant[cid])}); tensor digest `{meta['subject_digest']}`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `{plant['source_digest'][:16]}…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `{meta['practice_sha256']}`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `{sha256_file(SCORESET)}`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** {BUDGET[arm]} passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

{hyps}

6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** {PROGRAMMER[arm].replace("q01", pack)}
"""
    (run_dir / "preregistration.md").write_text(pre)
    shutil.copy(prog, run_dir / "program.py")
    print("preregistration written; starting sandbox", flush=True)
    out = subprocess.run(["bash", "-c", R.sandbox_command(run_dir)], capture_output=True, text=True)
    (run_dir / "stdout.log").write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
    print(out.stdout[-1500:], out.stderr[-1500:], flush=True)
    final = run_dir / "final.adapter"
    if not final.exists():
        print("NO final.adapter: the program did not freeze", flush=True)
        return run_dir
    res = P.score(final, SCORESET, spool=SPOOL)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["heldout"] = {"set": SCORESET.stem, "metrics": res["metrics"], "per_page": res["per_page"], "keys": res["keys"],
                          "final_digest": res["digest"], "engine": "vllm batched (not repeatable)"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"score set: CER {res['metrics']['cer']:.4f} truncated {res['metrics']['truncated_cer']:.4f} at cap {res['metrics']['at_cap']}")
    return run_dir


if __name__ == "__main__":
    main(*sys.argv[1:4])
