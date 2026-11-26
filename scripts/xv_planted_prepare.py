"""E3 (docs/XVENDOR_PLAN.md §2): set up the planted-defect cells for a cross-vendor programmer.

    .venv/bin/python scripts/xv_planted_prepare.py --provider gemini --model gemini-4-argon --label "Gemini 4 Argon"
    .venv/bin/python scripts/xv_planted_prepare.py --provider openai --model gpt-6.2-astra --label "GPT-6.2 Astra"

Writes runs/planted-xv/<tag>/ (tag g1 or o1), mirroring the Fable rerun (runs/planted-f1, runs/planted-adapter-f1):
- the three briefs, word for word except the title, the folder paths and the model-id line;
- the 14 packs s01..s14 and the adapter pack q01, copied unchanged (the same opaque labels; the map
  runs/planted-f1/map.json is not readable through the tool layer);
- one folder per cell, sNN/a, sNN/b, q01/a, q01/b, each with a programmer.json: the brief and its own pack
  readable, only the arm's two files writable, tools read_file/list_dir/write_file (no run_python: the
  Fable programmers had py_compile only, and write_file compiles), one round with a fixed task.
Each cell is a fresh conversation, as each Fable cell was a fresh subagent. Run a cell with
    .venv/bin/python -m wsp.programmer round --run runs/planted-xv/<tag>/s06/a --provider gemini --round 0
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F1 = ROOT / "runs/planted-f1"
AF1 = ROOT / "runs/planted-adapter-f1"
TAG = {"gemini": "g1", "openai": "o1"}
FILES = {"a": ("preregistration_hypotheses.md", "program.py"), "b": ("hypotheses.md", "program.py")}
API_KRAKEN = ["wsp/ops.py", "wsp/session.py", "wsp/run.py", "wsp/kraken_subject.py", "wsp/align.py"]
API_LORA = ["wsp/ops.py", "wsp/session.py", "wsp/run.py", "wsp/lora_subject.py"]

SYSTEM = """You are the programmer of a weight-space session. You work only through the tools you are given.
Your instructions are in the brief your task names; read it first and follow it exactly.
Tool equivalents of what the brief says:
- "read" a file: read_file / list_dir (paths relative to the repository root; only the files the brief lists are open to you);
- "write" a file: write_file (whole file; only the two files your task names, in your folder). write_file runs
  `python -m py_compile` on .py files and reports any error, so you need no shell.
You have no passes: you do not run the program or load any model.
When both files are written, call finish_round. Do not stop without calling it."""


def brief(text: str, old_dir: str, new_dir: str) -> str:
    text = text.replace(", Fable rerun, 11.11.2026)", f", cross-vendor rerun, {datetime.now(timezone.utc):%d.%m.%Y})")
    text = text.replace(old_dir, new_dir)
    return text.replace("(claude-fable-5-1 unless you know otherwise)", "(as your task gives it)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=TAG, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    tag = TAG[a.provider]
    out = ROOT / "runs/planted-xv" / tag
    if out.exists():
        print(f"{out} exists; not overwritten", file=sys.stderr)
        return 1
    rel = out.relative_to(ROOT).as_posix()
    out.mkdir(parents=True)
    for name in ("PROGRAMMER_BRIEF.md", "PROGRAMMER_BRIEF_UNRELATED.md"):
        b = brief((F1 / name).read_text(), "runs/planted-f1/", f"{rel}/")
        assert "fable" not in b.lower(), name
        (out / name).write_text(b)
    b = brief((AF1 / "PROGRAMMER_BRIEF.md").read_text(), "runs/planted-adapter-f1/", f"{rel}/")
    assert "fable" not in b.lower()
    (out / "PROGRAMMER_BRIEF_ADAPTER.md").write_text(b)

    mp = json.loads((F1 / "map.json").read_text())
    cells = []
    for s in sorted(mp):
        shutil.copytree(F1 / s / "pack", out / s / "pack")
        br = "PROGRAMMER_BRIEF_UNRELATED.md" if mp[s].endswith("-u") else "PROGRAMMER_BRIEF.md"
        cells += [(s, arm, br, API_KRAKEN) for arm in "ab"]
    shutil.copytree(AF1 / "q01" / "pack", out / "q01" / "pack")
    cells += [("q01", arm, "PROGRAMMER_BRIEF_ADAPTER.md", API_LORA) for arm in "ab"]

    for s, arm, br, api in cells:
        d = out / s / arm
        d.mkdir(parents=True)
        f1, f2 = FILES[arm]
        task = (f"You are the programmer of pack {s}, task ({arm}). Your brief is {rel}/{br}; your pack is {rel}/{s}/pack/. "
                f"Read the brief first and do task ({arm}) as it describes: write {rel}/{s}/{arm}/{f1} and then "
                f"{rel}/{s}/{arm}/{f2}, nothing else. Your model id is `{a.model}` ({a.label}, called through the "
                f"{'Google' if a.provider == 'gemini' else 'OpenAI'} API by the operator's tool layer, provider default "
                f"reasoning settings); your tools are read_file, list_dir, write_file (with py_compile) and finish_round.")
        (d / "programmer.json").write_text(json.dumps({
            "provider": a.provider, "model": a.model, "label": a.label, "root": str(ROOT),
            "readable": [f"{rel}/{br}", f"{rel}/{s}/pack/*"] + api,
            "writable": [f1.replace(".", r"\."), f2.replace(".", r"\.")],
            "tools": ["read_file", "list_dir", "write_file"],
            "system": SYSTEM, "task": task, "program": "program.py", "options": {},
            "template": f"{'runs/planted-adapter-f1/q01' if s == 'q01' else 'runs/planted-f1/' + s}/{arm}",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
    print(json.dumps({"dir": rel, "cells": len(cells)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
