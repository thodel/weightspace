"""The session runner: new / freeze / heldout / export, and the programmer's entry point.

    python -m wsp.run new --subject models/kraken/<id>.mlmodel --practice data/practice/<set>.jsonl --budget 25
    python -m wsp.run freeze runs/<session>
    python -m wsp.run heldout runs/<session> --set <name>
    python -m wsp.run export runs/<session> --as <id>

Inside the sandbox the programmer does::

    from wsp.run import open_session
    s = open_session(".")          # cwd is the run folder
    s.hypothesis(...); s.edit(...); s.measure(); s.keep() / s.revert(); s.freeze()

``open_session`` loads the subject from the run's ``session.json`` and, when
``current.mlmodel`` exists, restores the edited weights into it (the start
tensors stay those of the original artefact, so the diff is against the
pre-registered digest). ``keep()`` writes ``current.mlmodel``; ``freeze()``
writes ``final.mlmodel`` and the summary.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from wsp.data import load, sha256_file
from wsp.session import Session

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
HELDOUT = REPO / "data" / "heldout"
KURRENT = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1")
#: Adapter held-out sets: frozen trainer draws, scored by the trainer's own evaluate_qlora (PHASE5_PLAN §7.3).
ADAPTER_HELDOUT = {"kurrent-test": KURRENT / "vlm-pages-test"}

PREREGISTRATION = """# Pre-registration — {name}

Filled in **before the first pass** (PROTOCOL.md §2). Empty fields are a deviation.

1. **Subject:** `{subject}` — tensor digest `{digest}`, file sha256 `{file_sha}`.
   Donors (parent/siblings) and their digests:
2. **Practice set:** `{practice}` — {lines} lines, list sha256 `{practice_sha}`.
   Source and split side: (train side only) — selection rule: see `data/practice/{practice_stem}.meta.json`.
3. **Held-out set:** identity (document ids / projects), size, sha256:
   Overlap audit result (`data/practice/overlap.json`):
4. **Budget:** {budget} passes.
5. **Hypotheses** (each with predicted direction and magnitude of CER change, and the edit class):
   - H1:
   - H2:
   - H3:
6. **Controls to run:** C0 no-op, C1 parent, C2 average, C3 α-sweep, C4 retrain, C5 sham, C6 blind — which, and when.
"""


def _is_adapter(path: str | Path) -> bool:
    return Path(path).is_dir() and (Path(path) / "adapter_config.json").is_file()


def _ext(subject) -> str:
    """What a saved state of this subject is called: a CoreML file or a peft directory."""
    from wsp.lora_subject import LoraSubject
    return ".adapter" if isinstance(subject, LoraSubject) else ".mlmodel"


def _subject(path: str | Path, device: str = "cpu", meta: dict | None = None, run_dir: Path | None = None):
    """A kraken file, or a peft directory (phase 5). An adapter gets its forward pass only
    inside a session: the trainer's evaluate_qlora on the session's practice pages."""
    p = Path(path)
    if p.suffix == ".mlmodel":
        from wsp.kraken_subject import KrakenSubject
        return KrakenSubject(p, device=device)
    if _is_adapter(p):
        from wsp.lora_subject import LoraSubject, reference_runner
        fwd = None
        if meta and run_dir:
            fwd = reference_runner(meta["practice"], meta["data_root"], run_dir / "passes")
        return LoraSubject(p, forward=fwd)
    raise ValueError(f"no subject for {p}")


def _file_sha(path: Path) -> str:
    """sha256 of a file, or of an adapter's weights file."""
    return sha256_file(path / "adapter_model.safetensors" if path.is_dir() else path)


def _unshare_works() -> bool:
    try:
        return subprocess.run(["unshare", "-Un", "true"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def sandbox_command(run_dir: Path, script: str = "program.py") -> str:
    venv_py = REPO / ".venv" / "bin" / "python"
    env = ["env", "-i", f"PATH=/usr/bin:/bin", f"HOME={run_dir}", f"TMPDIR={Path.home() / 'atr-cache' / 'tmp'}",
           "CUDA_VISIBLE_DEVICES=0", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "no_proxy=*"]
    cmd = env + (["unshare", "-Un"] if _unshare_works() else []) + [str(venv_py), "-I", script]
    return "cd " + str(run_dir) + " && " + " ".join(cmd)


def new(subject: str, practice: str, budget: int, run_id: str | None = None, seed: int = 20261106,
        data_root: str | None = None) -> Path:
    subj_path, prac_path = Path(subject), Path(practice)
    s = _subject(subj_path)
    items = load(prac_path)
    stem = run_id or subj_path.stem.replace("kraken-", "")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    n = 1
    while (RUNS / f"{date}-{stem}-b{budget}-{n}").exists():
        n += 1
    run_dir = RUNS / f"{date}-{stem}-b{budget}-{n}"
    run_dir.mkdir(parents=True)
    meta = {"name": run_dir.name, "subject": str(subj_path.resolve()), "subject_digest": s.digest(),
            "subject_file_sha256": _file_sha(subj_path), "practice": str(prac_path.resolve()),
            "practice_sha256": sha256_file(prac_path), "practice_lines": len(items), "budget": budget, "seed": seed,
            "created": datetime.now(timezone.utc).isoformat(), "unshare": _unshare_works(),
            "deviations": [] if _unshare_works() else ["unshare -Un not available: network cut by offline variables only"]}
    if _is_adapter(subj_path):
        if not data_root:
            raise ValueError("an adapter session needs --data-root (where the practice pages' images resolve)")
        meta.update({"unit": "page", "data_root": str(Path(data_root).resolve())})
    (run_dir / "session.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "preregistration.md").write_text(PREREGISTRATION.format(
        name=run_dir.name, subject=subj_path, digest=meta["subject_digest"], file_sha=meta["subject_file_sha256"],
        practice=prac_path, practice_stem=prac_path.stem, lines=len(items), practice_sha=meta["practice_sha256"], budget=budget))
    (run_dir / "notes.md").write_text(f"# Notes — {run_dir.name}\n\n")
    (run_dir / "program.py").write_text(
        '"""The programmer\'s script. Runs inside the sandbox; only the wsp API and numpy."""\n'
        "from wsp.run import open_session\n\ns = open_session('.')\nprint(s.subject.describe()['parameters'], 'parameters;', s.budget_left, 'passes left')\n")
    (run_dir / "env.txt").write_text(subprocess.run([str(REPO / ".venv/bin/pip"), "freeze"], capture_output=True, text=True).stdout)
    print(run_dir)
    print("sandbox:", sandbox_command(run_dir))
    return run_dir


class RunSession(Session):
    """A Session that persists the edited weights next to its log."""

    def keep(self, reason: str = "") -> bool:
        """The harness's keep rule decides (Session.keep); only a kept state is persisted.
        Until 06.11.2026 this wrapper returned None, so a programmer reading the return value
        saw every keep as refused (session 20261106-bern-combined-v1-b10-1, deviations)."""
        kept = super().keep(reason)
        if kept:
            self.subject.save(self.run_dir / f"current{_ext(self.subject)}")
        return kept

    def freeze(self) -> dict:
        summary = super().freeze()
        final = self.subject.save(self.run_dir / f"final{_ext(self.subject)}")
        summary["final_file_sha256"] = _file_sha(final)
        meta = json.loads((self.run_dir / "session.json").read_text())
        summary["deviations"] = meta.get("deviations", []) + summary.get("deviations", [])
        summary["subject"] = meta["subject"]; summary["name"] = meta["name"]
        summary["sandbox"] = sandbox_command(self.run_dir)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary


def open_session(run_dir: str | Path = ".", device: str = "cpu") -> RunSession:
    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / "session.json").read_text())
    s = _subject(meta["subject"], device=device, meta=meta, run_dir=run_dir)
    if s.digest() != meta["subject_digest"]:
        raise RuntimeError("subject digest differs from session.json; the artefact changed")
    items = load(meta["practice"])
    sess = RunSession(s, items, run_dir, budget=meta["budget"], seed=meta["seed"], practice_id=meta["practice_sha256"],
                      unit=meta.get("unit", "line"))
    cur = run_dir / f"current{_ext(s)}"
    if cur.exists():
        edited = _subject(cur, device=device)
        for name, value in edited.tensors().items():
            if not (value == s.tensors()[name]).all():
                s.set_tensor(name, value)
        sess.note(f"state restored from current.mlmodel (digest {edited.digest()[:12]})")
    return sess


def freeze(run_dir: str) -> dict:
    sess = open_session(run_dir)
    return sess.freeze()


def heldout_adapter(run_dir: Path, set_name: str) -> dict:
    """Score a frozen adapter once on a frozen trainer draw with the trainer's own scorer
    (transformers, greedy, cap 3,072), and with ``wsp.pages`` on the same readings."""
    from wsp import pages
    from wsp import vlm_reference
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    final = run_dir / "final.adapter"
    if not final.is_dir():
        raise RuntimeError("freeze the session first (no final.adapter)")
    side = ADAPTER_HELDOUT[set_name]
    report = run_dir / "readings" / f"heldout_{set_name}.json"
    report.parent.mkdir(exist_ok=True)
    t = time.time()
    if vlm_reference.main(["--adapter", str(final), "--pages", str(side / "pages.jsonl"),
                           "--data-root", str(side), "--report", str(report)]) != 0:
        raise RuntimeError("evaluate_qlora failed; nothing recorded")
    trainer = json.loads(report.read_text())
    raw = pages.read_reference(report.with_suffix(".raw.jsonl"))
    rows = load(side / "pages.jsonl")
    keys = [Path(r["image"]).stem for r in rows]
    own = pages.score([raw[k]["text"] for k in keys], [r["text"] for r in rows], [raw[k]["at_cap"] for k in keys])
    result = {"set": set_name, "set_sha256": sha256_file(side / "pages.jsonl"), "pages": len(rows),
              "trainer": {k: trainer.get(k) for k in ("cer", "wer", "truncated_cer", "length_ratio", "samples", "truncated_at_cap")},
              "wsp": {k: v for k, v in own.items() if k != "per_page"}, "per_page": own["per_page"],
              "final_sha256": _file_sha(final), "when": datetime.now(timezone.utc).isoformat(),
              "wall_s": round(time.time() - t, 1)}
    if summary.get("heldout") is None:
        summary["heldout"] = result
    else:
        summary.setdefault("heldout_repeats", []).append(result)
        summary.setdefault("deviations", []).append(f"second held-out evaluation on {set_name}: not a result")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"held-out {set_name}: trainer CER {result['trainer']['cer']}, truncated {result['trainer']['truncated_cer']}, "
          f"at cap {result['trainer']['truncated_at_cap']} of {len(rows)}")
    return result


def heldout(run_dir: str, set_name: str) -> dict:
    """Score the frozen artefact once on a held-out set, from outside the sandbox."""
    from wsp.metrics import score, per_line_edits
    run_dir = Path(run_dir).resolve()
    if set_name in ADAPTER_HELDOUT:
        return heldout_adapter(run_dir, set_name)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    final = run_dir / "final.mlmodel"
    if not final.exists():
        raise RuntimeError("freeze the session first (no final.mlmodel)")
    set_path = HELDOUT / f"{set_name}.jsonl"
    if not set_path.exists():
        raise FileNotFoundError(f"{set_path}: held-out sets are built outside phase 0 (none exist yet)")
    items = load(set_path)
    s = _subject(final)
    t = time.time()
    hyps = [p.text for p in s.forward(items)]; refs = [x["text"] for x in items]
    m = score(hyps, refs)
    result = {"set": set_name, "set_sha256": sha256_file(set_path), "lines": len(items), "metrics": m.as_dict(),
              "per_line": per_line_edits(hyps, refs), "final_digest": s.digest(), "when": datetime.now(timezone.utc).isoformat(),
              "wall_s": round(time.time() - t, 1)}
    if summary.get("heldout") is None:
        summary["heldout"] = result
    else:
        summary.setdefault("heldout_repeats", []).append(result)
        summary.setdefault("deviations", []).append(f"second held-out evaluation on {set_name}: not a result")
    summary_path.write_text(json.dumps(summary, indent=2))
    (run_dir / "readings").mkdir(exist_ok=True)
    (run_dir / "readings" / f"heldout_{set_name}.jsonl").write_text(
        "".join(json.dumps({"key": i["key"], "hyp": h, "ref": r}, ensure_ascii=False) + "\n" for i, h, r in zip(items, hyps, refs)))
    print(f"held-out {set_name}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed} ({len(items)} lines)")
    return result


def export(run_dir: str, model_id: str) -> Path:
    """Write the frozen .mlmodel and a registry record in the trainer's trained/<id>/ layout.

    Nothing is copied to the share and nothing is registered: that is phase 4.
    """
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text())
    meta = json.loads((run_dir / "session.json").read_text())
    final = run_dir / "final.mlmodel"
    if not final.exists():
        raise RuntimeError("freeze the session first")
    out = run_dir / "export" / model_id
    out.mkdir(parents=True, exist_ok=True)
    weights = out / f"{model_id}.mlmodel"
    shutil.copy2(final, weights)
    s = _subject(weights)
    heldout_m = (summary.get("heldout") or {}).get("metrics") or {}
    record = {                                  # fields as trained/<id>/metadata.json of the trainer
        "model_id": model_id, "job_id": meta["name"], "engine": "kraken",
        "code": {"weightspace": _git_rev()},
        "created": datetime.now(timezone.utc).isoformat(),
        "weights": weights.name, "source_weights": str(final),
        "weights_digest": {"files": {weights.name: sha256_file(weights)}, "tensors": s.digest()},
        "metrics": {"cer": heldout_m.get("cer"), "wer": heldout_m.get("wer"), "length_ratio": heldout_m.get("length_ratio"),
                    "measured_on": (summary.get("heldout") or {}).get("set"), "samples": (summary.get("heldout") or {}).get("lines")},
        "request": {"engine": "kraken", "model_id": model_id, "base_model": meta["subject"],
                    "method": "weight-space programming (no gradient)", "budget": summary["budget"], "passes_used": summary["used"],
                    "practice": meta["practice"], "practice_sha256": meta["practice_sha256"],
                    "edits": summary["edits"], "kept": summary["kept"], "diff": str(run_dir / "diff.json")},
        "progress": {"val_accuracy": None},
    }
    (out / "metadata.json").write_text(json.dumps(record, indent=2))
    registry = {"id": model_id, "engine": "kraken", "enabled": False, "base_model": Path(meta["subject"]).stem,
                "task": "htr", "level": "line", "local_path": str(weights), "weights_digest": record["weights_digest"],
                "notes": f"weight-space edit of {Path(meta['subject']).stem}; session {meta['name']}; NOT registered (phase 4 needs an OK)"}
    (out / f"{model_id}.yaml").write_text(_yaml(registry))
    print(out)
    return out


def _git_rev() -> dict:
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return {"commit": rev, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _yaml(d: dict, indent: int = 0) -> str:
    out = []
    for k, v in d.items():
        pad = "  " * indent
        if isinstance(v, dict):
            out.append(f"{pad}{k}:"); out.append(_yaml(v, indent + 1))
        elif isinstance(v, bool):
            out.append(f"{pad}{k}: {'true' if v else 'false'}")
        elif v is None:
            out.append(f"{pad}{k}: null")
        else:
            out.append(f"{pad}{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, str) else v}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new"); n.add_argument("--subject", required=True); n.add_argument("--practice", required=True)
    n.add_argument("--budget", type=int, required=True); n.add_argument("--id"); n.add_argument("--seed", type=int, default=20261106)
    n.add_argument("--data-root", help="adapters: where the practice pages' images resolve")
    f = sub.add_parser("freeze"); f.add_argument("run")
    h = sub.add_parser("heldout"); h.add_argument("run"); h.add_argument("--set", required=True)
    e = sub.add_parser("export"); e.add_argument("run"); e.add_argument("--as", dest="model_id", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "new":
        new(a.subject, a.practice, a.budget, a.id, a.seed, a.data_root)
    elif a.cmd == "freeze":
        s = freeze(a.run); print(json.dumps({k: s[k] for k in ("name", "used", "edits", "kept", "final_digest")}, indent=1))
    elif a.cmd == "heldout":
        heldout(a.run, a.set)
    elif a.cmd == "export":
        export(a.run, a.model_id)


if __name__ == "__main__":
    main()
