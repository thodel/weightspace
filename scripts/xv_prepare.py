"""Set up a cross-vendor replicate run folder from a Fable replicate (docs/XVENDOR_PLAN.md §4).

Run it on a checkout of main, at the moment the session starts (the run id carries the date):

    .venv/bin/python scripts/xv_prepare.py --experiment kurrent --provider gemini --model gemini-4-argon
    .venv/bin/python scripts/xv_prepare.py --experiment bern    --provider openai --model <pinned gpt id> --dry-run

What it does, all inside the new run folder:
1. ``wsp.run new`` with the template run's subject, practice list (from this checkout), budget,
   seed and data root;
2. the template's ``PROGRAMMER_BRIEF.md`` with run name and checkout path replaced, plus the
   errata the Fable programmer found (kurrent only, plan §3.4);
3. the template's ``pack/``, if it has one;
4. ``preregistration.md``: the header, §1-4 and §6 from the template, §5 left empty for the
   programmer, §7 written anew; it names no earlier session's edit or result;
5. ``programmer.json``: provider, pinned model, and the files the brief allows (enforced by
   ``wsp.programmer``).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "kurrent": {
        "template": "runs/20261110-kurrent-page-v2-f1-b10-1",
        "stem": "kurrent-page-v2",
        "readable": [
            "docs/readings/adapter_kurrent-page-v2.md", "docs/readings/adapter_kurrent-page-v2.json",
            "docs/baselines/c0-kurrent-val-v2.json",
            "data/practice/kurrent-val-64.jsonl", "data/practice/kurrent-val-64.meta.json",
            "wsp/ops.py", "wsp/session.py", "wsp/run.py", "wsp/lora_subject.py", "wsp/pages.py", "wsp/metrics.py",
        ],
        # Found by the Fable programmer in round 0 (runs/20261110-kurrent-page-v2-f1-b10-1/notes.md).
        "errata": [
            ("direction=u, axis=0)", "direction=u, axis=1)"),
            ("`practice_per_page`\n  maps each key to [reference length, hypothesis length].",
             "`practice_per_page`\n  maps each key to [edits, reference characters]."),
        ],
    },
    "bern": {
        "template": "runs/20261111-bern-combined-v1-f1-b10-1",
        "stem": "bern-combined-v1",
        "readable": [
            "{run}/pack/*",
            "data/practice/bern-train-300.jsonl", "data/practice/bern-train-300.meta.json",
            "wsp/ops.py", "wsp/session.py", "wsp/run.py", "wsp/kraken_subject.py", "wsp/subject.py",
            "wsp/align.py", "wsp/metrics.py", "wsp/data.py",
        ],
        "errata": [],
    },
}

TAG = {"gemini": "g1", "openai": "o1"}

SECTION7 = """7. **Programmer and operator:**
   - **Programmer:** {label} (`{model}`), called through the {vendor} API by `wsp.programmer`
     with the provider's default reasoning settings. It has the tools `read_file`, `list_dir`,
     `write_file`, `run_python` (CPU-only reading in the sandbox, on a throw-away copy of this
     folder) and `finish_round`. The tool layer lets it read only the files the brief lists and
     write only section 5 of this file, `program_NN.py` and `notes.md`.
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists. It does
       not read any other run, report or plan.
     - **Rounds:** it writes `program_NN.py` in rounds. The operator runs each one in the
       sandbox and gives it the stdout; the conversation carries over between rounds.
     - **Record:** `programmer/transcript.jsonl` (every call and tool result),
       `programmer/usage.json` (tokens), `programmer/rounds.json`.
   - **Operator:** a Claude session. It set up the run, runs the programs, scores held-out once,
     runs the controls and writes the comparison. It writes no hypotheses and no edits. What it
     knew before the first pass is written in `docs/XVENDOR_PLAN.md` §6, which the programmer
     does not read. The brief and this file name no edit and no result of an earlier session.
"""

LABEL = {"gemini": ("Google", "Gemini"), "openai": ("OpenAI", "GPT")}


def sections(text: str) -> tuple[str, dict[int, str]]:
    """Split a pre-registration into its header and its numbered top-level sections."""
    parts = re.split(r"(?m)^(?=(\d)\. \*\*)", text)
    head, secs = parts[0], {}
    i = 1
    while i < len(parts):
        num, body = int(parts[i]), parts[i + 1]
        secs[num] = body
        i += 2
    return head, secs


def build_prereg(template: str, old_name: str, new_name: str, old_root: str, label: str, model: str,
                 vendor: str, deviation: str = "") -> str:
    head, secs = sections(template)
    fix = lambda t: t.replace(old_name, new_name).replace(old_root, str(ROOT))  # noqa: E731
    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    head = re.sub(r"(?m)^# .*$", f"# Pre-registration — {new_name}", fix(head))
    head = re.sub(r"\d\d\.\d\d\.20\d\d", today, head, count=1)
    head = head.replace("with a different\nprogrammer", f"with a different\nprogrammer ({label})")
    if deviation:
        head = head.rstrip("\n") + f"\n\n**Deviation from the template:** {deviation}\n\n"
    out = [head]
    for n in (1, 2, 3, 4):
        out.append(fix(secs[n]))
    out.append("5. **Hypotheses** (written by the programmer before the first pass):\n\n"
               "   _To be written by the programmer._\n\n")
    out.append(fix(secs[6]))
    out.append(SECTION7.format(label=label, model=model, vendor=vendor))
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    ap.add_argument("--provider", choices=TAG, required=True)
    ap.add_argument("--model", required=True, help="pinned model id, as `wsp.programmer check` printed it")
    ap.add_argument("--label", help="human name of the model, e.g. 'Gemini 4 Argon'")
    ap.add_argument("--attempt", type=int, default=1, help="1 for g1/o1, 2 for g2/o2, ...")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    exp = EXPERIMENTS[a.experiment]
    tdir = ROOT / exp["template"]
    meta = json.loads((tdir / "session.json").read_text())
    old_root = meta["practice"].split("/data/practice/")[0]
    practice = ROOT / "data/practice" / Path(meta["practice"]).name
    tag = TAG[a.provider][0] + str(a.attempt)
    run_id = f"{exp['stem']}-{tag}"
    vendor, family = LABEL[a.provider]
    label = a.label or f"{family} ({a.model})"
    brief = (tdir / "PROGRAMMER_BRIEF.md").read_text()
    missing = [e[0] for e in exp["errata"] if e[0] not in brief]
    if missing:
        print("errata text not found in the template brief:", missing, file=sys.stderr)
        return 1
    prereg_t = (tdir / "preregistration.md").read_text()
    if sorted(sections(prereg_t)[1]) != [1, 2, 3, 4, 5, 6, 7]:
        print("template pre-registration does not have sections 1-7", file=sys.stderr)
        return 1

    plan = {"run_id": run_id, "subject": meta["subject"], "practice": str(practice),
            "practice_sha256 (template)": meta["practice_sha256"], "budget": meta["budget"],
            "seed": meta["seed"], "data_root": meta.get("data_root"), "template": exp["template"],
            "provider": a.provider, "model": a.model, "errata": len(exp["errata"])}
    if a.dry_run:
        print(json.dumps(plan, indent=1))
        return 0

    sys.path.insert(0, str(ROOT))
    from wsp.run import new
    run_dir = new(meta["subject"], str(practice), meta["budget"], run_id, meta["seed"], meta.get("data_root"))
    new_meta = json.loads((run_dir / "session.json").read_text())
    if new_meta["practice_sha256"] != meta["practice_sha256"] or new_meta["subject_digest"] != meta["subject_digest"]:
        shutil.rmtree(run_dir)
        print("practice list or subject differs from the template run; run folder removed", file=sys.stderr)
        return 1

    old_name, new_name = tdir.name, run_dir.name
    for old, new_t in exp["errata"]:
        brief = brief.replace(old, new_t)
    brief = brief.replace(old_name, new_name).replace(old_root, str(ROOT))
    (run_dir / "PROGRAMMER_BRIEF.md").write_text(brief)
    if (tdir / "pack").is_dir():
        shutil.copytree(tdir / "pack", run_dir / "pack")
    deviation = ""
    if exp["errata"]:
        deviation = (f"the brief carries {len(exp['errata'])} corrections that the Fable programmer found in the "
                     f"template brief ({exp['template']}/notes.md, round 0); docs/XVENDOR_PLAN.md §3.4.")
        new_meta["deviations"] = new_meta.get("deviations", []) + ["brief errata corrected: " + deviation]
        (run_dir / "session.json").write_text(json.dumps(new_meta, indent=1))
    (run_dir / "preregistration.md").write_text(
        build_prereg(prereg_t, old_name, new_name, old_root, label, a.model, vendor, deviation))
    (run_dir / "notes.md").write_text(f"# Notes — {new_name}\n\n")
    (run_dir / "programmer.json").write_text(json.dumps({
        "provider": a.provider, "model": a.model, "label": label, "root": str(ROOT),
        "readable": [r for r in exp["readable"]], "options": {},
        "template": exp["template"], "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1))
    print(json.dumps({**plan, "run_dir": str(run_dir.relative_to(ROOT))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
