#!/usr/bin/env python3
"""Run one stage-4 programmer session on subject A and read its frozen model on the DISCOVERY half (docs/TROCR_STAGE4_PREREG.md sections 5, 6; amendment 3).

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/run_trocr_stage4_session.py ARM [--no-read]
ARM: p1 (Sonnet 5.5, donor), p2 (Fable 5.2, donor), p3s (Sonnet 5.5, no donor), p3f (Fable 5.2, no donor), b (Sonnet 5.5, blind, budget 0, donor),
     r (replay, fresh session of the family of the better of p1/p2, with the donor).
The program and hypotheses are those the programmer wrote under runs/trocr-stage4/A/<arm>/, copied unchanged. The session runs under the Landlock allow-list
(donor files only in the arms that have the donor). The discovery read (17 objects of blb-eval-hand, never the confirmation half) is outside the sandbox,
once per frozen digest; a frozen model equal to the subject needs no read (delta 0)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trocr_stage4_discovery as DS
import trocr_stage4_grid as G
from wsp import data, metrics, run as R
from wsp.data import sha256_file
from wsp.trocr_subject import TrocrSubject

REPO = G.REPO
SUBJ = os.environ.get("STAGE4_SUBJECT", "A")                # "A" (from-kurrent-v2, donor XVI-XVII) or "B" (trocr-kurrent, donor IAM base)
ADIR = REPO / f"runs/trocr-stage4/{SUBJ}"
SUBJECT, DONOR = G.MODELS[SUBJ], G.MODELS[{"A": "D", "B": "I"}[SUBJ]]
BUDGET = {"p1": 10, "p2": 10, "p3s": 10, "p3f": 10, "b": 0, "r": 10}
DONOR_ARMS = {"p1", "p2", "b", "r"}
WHO = {"p1": "Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm P1, with the donor", "p2": "Claude Fable 5.2 (`claude-fable-5-2`), arm P2, with the donor",
       "p3s": "Claude Sonnet 5.5, arm P3, donor absent", "p3f": "Claude Fable 5.2, arm P3, donor absent",
       "b": "Claude Sonnet 5.5, arm B, blind (budget 0), with the donor", "r": "arm R, replay, with the donor"}
DISC_OUT = REPO / f"docs/trocr/stage4_{SUBJ}_sessions.jsonl"


def discovery_read(run_dir: Path, arm: str):
    summ = json.loads((run_dir / "summary.json").read_text())
    final = run_dir / "final.trocr"
    split = json.loads(DS.SPLIT.read_text())
    disc = set(split["objects"]["discovery"])
    rows = [r for r in data.load(str(DS.HELD)) if r["doc"] in disc]
    assert len(rows) == split["lines"]["discovery"]
    clusters = [metrics.page_of(r["key"]) for r in rows]
    base = next(r for r in DS.jl(DS.DISC) if r["subject"] == SUBJ and r["what"] == "baseline")
    bedits = [tuple(x) for x in base["edits"]]
    sub = TrocrSubject(final if final.is_dir() else SUBJECT, device="cuda:0", batch=8)
    same = sub.digest() == TrocrSubject(SUBJECT, device="cpu", check_card=False).digest()
    if same:
        rec = {"arm": arm, "run": run_dir.name, "kept": summ["kept"], "frozen_equals_subject": True, "discovery_cer": base["cer"], "delta_cer": 0.0, "ci95_cluster": [0.0, 0.0]}
    else:
        ed = G.edits(G.read(sub, rows), rows)
        b = metrics.bootstrap_delta(bedits, ed, clusters=clusters, resamples=2000)
        rec = {"arm": arm, "run": run_dir.name, "kept": summ["kept"], "frozen_equals_subject": False, "discovery_cer": sum(e for e, _ in ed) / sum(c for _, c in ed),
               "delta_cer": b["delta_cer"], "ci95_cluster": b["ci95_cluster"], "edits": ed}
    with DISC_OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(arm, "discovery delta", round(100 * rec["delta_cer"], 2), [round(100 * x, 2) for x in rec["ci95_cluster"]], flush=True)


def main(arm, read=True):
    if BUDGET[arm]:                                   # the card gate runs here, outside the allow-list (amendment 3)
        from wsp import card
        import time
        while True:
            ok, why = card.ready()
            if ok:
                break
            print("card not ready:", why, flush=True)
            time.sleep(120)
    adir = ADIR / arm
    prog, hyp = adir / "program.py", adir / "hypotheses.md"
    run_dir = R.new(str(SUBJECT), str(G.PRACTICE), BUDGET[arm], run_id=f"trocr-stage4-{SUBJ}-{arm}", device="cuda:0", min_effect=0.005, cluster_rule=True,
                    restrict_files=True, donors=[str(DONOR)] if arm in DONOR_ARMS else [])
    meta = json.loads((run_dir / "session.json").read_text())
    shutil.copyfile(prog, run_dir / "program.py")
    (run_dir / "preregistration.md").write_text(f"""# Pre-registration, {run_dir.name}

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject {SUBJ} (`{Path(SUBJECT).name}`), programmer {WHO[arm]}. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/{SUBJ}/{'pack' if arm in DONOR_ARMS else 'pack_nodonor'}/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/{SUBJ}/{arm}/program.py`, copied unchanged (sha256 `{sha256_file(prog)}`); hypotheses below as written, before any pass.
Subject digest `{meta['subject_digest']}`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `{meta['practice_sha256']}`. Budget {BUDGET[arm]}, min effect 0.005,
page-clustered interval, generation settings {json.dumps(meta['generation'])}. Landlock allow-list: `landlock.json`; donor files {'visible' if arm in DONOR_ARMS else 'absent'}.

## Hypotheses (`hypotheses.md`)

{hyp.read_text()}
""")
    r = subprocess.run(["bash", "-c", R.sandbox_command(run_dir)], capture_output=True, text=True)
    (run_dir / "sandbox.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-4000:])
    print(run_dir.name, "exit", r.returncode, flush=True)
    if r.returncode:
        print(r.stderr[-1500:])
    if read and (run_dir / "summary.json").exists():
        discovery_read(run_dir, arm)
    s = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    print("done", run_dir.name, "passes", s.get("used"), "kept", s.get("kept"), flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--no-read" not in sys.argv))
