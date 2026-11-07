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
import shlex
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
SCORES = Path.home() / "atr-cache" / "wsp" / "scores"

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


def _subject(path: str | Path, device: str = "cpu"):
    p = Path(path)
    if p.suffix == ".mlmodel":
        from wsp.kraken_subject import KrakenSubject
        return KrakenSubject(p, device=device)
    raise ValueError(f"no subject for {p.suffix} yet (adapters are phase 5)")


def _unshare_works() -> bool:
    try:
        return subprocess.run(["unshare", "-Un", "true"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def sandbox_command(run_dir: Path, script: str = "program.py") -> str:
    venv_py = Path.home() / "Repo" / "weightspace" / ".venv" / "bin" / "python"
    env = ["env", "-i", f"PATH=/usr/bin:/bin", f"HOME={run_dir}", f"TMPDIR={Path.home() / 'atr-cache' / 'tmp'}",
           "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "no_proxy=*"]
    # -I ignores PYTHONPATH; the shim puts this checkout's wsp first so a worktree runs its own harness
    shim = f"import sys,runpy; sys.path.insert(0, {str(REPO)!r}); runpy.run_path({script!r}, run_name='__main__')"
    cmd = env + (["unshare", "-Un"] if _unshare_works() else []) + [str(venv_py), "-I", "-c", shlex.quote(shim)]
    return "cd " + str(run_dir) + " && " + " ".join(cmd)


def new(subject: str, practice: str, budget: int, run_id: str | None = None, seed: int = 20261106) -> Path:
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
            "subject_file_sha256": sha256_file(subj_path), "practice": str(prac_path.resolve()),
            "practice_sha256": sha256_file(prac_path), "practice_lines": len(items), "budget": budget, "seed": seed,
            "created": datetime.now(timezone.utc).isoformat(), "unshare": _unshare_works(),
            "deviations": [] if _unshare_works() else ["unshare -Un not available: network cut by offline variables only"]}
    (run_dir / "session.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "preregistration.md").write_text(PREREGISTRATION.format(
        name=run_dir.name, subject=subj_path, digest=meta["subject_digest"], file_sha=meta["subject_file_sha256"],
        practice=prac_path, practice_stem=prac_path.stem, lines=len(items), practice_sha=meta["practice_sha256"], budget=budget))
    (run_dir / "notes.md").write_text(f"# Notes — {run_dir.name}\n\n")
    (run_dir / "program.py").write_text(
        '"""The programmer\'s script. Runs inside the sandbox; only the wsp API and numpy."""\n'
        "from wsp.run import open_session\n\ns = open_session('.')\nprint(s.subject.describe()['parameters'], 'parameters;', s.budget_left, 'passes left')\n")
    (run_dir / "env.txt").write_text(subprocess.run([str(Path.home() / "Repo" / "weightspace" / ".venv" / "bin" / "pip"), "freeze"], capture_output=True, text=True).stdout)
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
            self.subject.save(self.run_dir / "current.mlmodel")
        return kept

    def freeze(self) -> dict:
        summary = super().freeze()
        final = self.subject.save(self.run_dir / "final.mlmodel")
        summary["final_file_sha256"] = sha256_file(final)
        meta = json.loads((self.run_dir / "session.json").read_text())
        summary["deviations"] = meta.get("deviations", []) + summary.get("deviations", [])
        summary["subject"] = meta["subject"]; summary["name"] = meta["name"]
        summary["sandbox"] = sandbox_command(self.run_dir)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary


def open_session(run_dir: str | Path = ".", device: str = "cpu") -> RunSession:
    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / "session.json").read_text())
    s = _subject(meta["subject"], device=device)
    if s.digest() != meta["subject_digest"]:
        raise RuntimeError("subject digest differs from session.json; the artefact changed")
    items = load(meta["practice"])
    sess = RunSession(s, items, run_dir, budget=meta["budget"], seed=meta["seed"], practice_id=meta["practice_sha256"])
    cur = run_dir / "current.mlmodel"
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


def heldout(run_dir: str, set_name: str) -> dict:
    """Score the frozen artefact once on a held-out set, from outside the sandbox."""
    from wsp.metrics import score, per_line_edits
    run_dir = Path(run_dir).resolve()
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
    set_sha = sha256_file(set_path)
    # a score is a function of (weights, list): reuse it when the same digest was scored on the same list
    # (corrupted-model sessions that end unchanged; docs/CONTROLS_REPORT.md). Not a second evaluation.
    cache = SCORES / f"heldout-{set_name}-{set_sha[:12]}-{s.digest()[:16]}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else None
    if cached:
        hyps = cached["hyps"]; refs = [x["text"] for x in items]
        assert cached["keys"] == [x["key"] for x in items]
    else:
        hyps = [p.text for p in s.forward(items)]; refs = [x["text"] for x in items]
    m = score(hyps, refs)
    if not cached:
        SCORES.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"digest": s.digest(), "set": set_name, "set_sha256": set_sha, "keys": [x["key"] for x in items],
                                     "hyps": hyps, "metrics": m.as_dict()}, ensure_ascii=False))
    result = {"set": set_name, "set_sha256": set_sha, "scored_from_cache": bool(cached), "lines": len(items), "metrics": m.as_dict(),
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
    f = sub.add_parser("freeze"); f.add_argument("run")
    h = sub.add_parser("heldout"); h.add_argument("run"); h.add_argument("--set", required=True)
    e = sub.add_parser("export"); e.add_argument("run"); e.add_argument("--as", dest="model_id", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "new":
        new(a.subject, a.practice, a.budget, a.id, a.seed)
    elif a.cmd == "freeze":
        s = freeze(a.run); print(json.dumps({k: s[k] for k in ("name", "used", "edits", "kept", "final_digest")}, indent=1))
    elif a.cmd == "heldout":
        heldout(a.run, a.set)
    elif a.cmd == "export":
        export(a.run, a.model_id)


if __name__ == "__main__":
    main()
