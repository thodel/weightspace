"""Re-run a finished session exactly as it ran, and check that it did.

    python -m wsp.replay runs/<session> --out ~/atr-cache/wsp/replay/<session>

The session's ``program.py``, ``session.json`` and ``preregistration.md`` are copied
into a fresh folder and the program is executed again with the current harness.
Edits are logged only as shapes and norms, so the program itself has to recompute
them; the *decisions* are taken from the original ``log.jsonl`` (``replay.json``,
read by ``wsp.run.open_session``), because a session run under an earlier keep rule
would otherwise diverge at its first keep. Everything the current harness adds to a
pass (since 07.11.2026: the block table of ``wsp.blocks``) is computed afresh.

The check, written to ``replay_check.json``: the tensor digest of every pass equals
the original's, every decision matches, the final digest matches, and every pass
reads exactly the same text on every line. If all hold, the replay *is* the session,
and the original held-out readings are the readings of the replayed final state.

Run it from the repository root (the program imports ``wsp`` from there), ideally under
``unshare -Un`` like the sandbox; the program is the session's own, already audited code.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
import traceback
from pathlib import Path


def _events(log: Path) -> list[dict]:
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def _readings(d: Path) -> dict[str, dict[str, str]]:
    out = {}
    for f in sorted(d.glob("pass_*.jsonl")):
        out[f.stem] = {json.loads(l)["key"]: json.loads(l)["text"] for l in f.read_text().splitlines() if l.strip()}
    return out


def replay(src: Path, out: Path) -> dict:
    src, out = src.resolve(), out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("session.json", "program.py", "preregistration.md"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    orig = _events(src / "log.jsonl")
    decisions = [{"kept": bool(e["kept"]), "reason": e.get("reason"), "rule": e.get("rule")}
                 for e in orig if e["kind"] == "decision"]
    (out / "replay.json").write_text(json.dumps({"source": str(src), "decisions": decisions}, indent=1))

    cwd, error = os.getcwd(), None
    os.chdir(out)
    try:
        runpy.run_path(str(out / "program.py"), run_name="__main__")
    except SystemExit:
        pass
    except Exception:
        error = traceback.format_exc()
    finally:
        os.chdir(cwd)
    if error is None and not (out / "summary.json").exists() and (src / "summary.json").exists():
        from wsp.run import freeze            # the original was frozen from outside (wsp.run freeze)
        try:
            freeze(str(out))
        except Exception:
            error = traceback.format_exc()

    new = _events(out / "log.jsonl") if (out / "log.jsonl").exists() else []
    passes_o = [e["digest"] for e in orig if e["kind"] == "pass"]
    passes_n = [e["digest"] for e in new if e["kind"] == "pass"]
    kept_o = [d["kept"] for d in decisions]
    kept_n = [bool(e["kept"]) for e in new if e["kind"] == "decision"]
    ro, rn = _readings(src / "readings"), _readings(out / "readings")
    text_diff = {p: sum(1 for k, t in ro[p].items() if rn.get(p, {}).get(k) != t) for p in ro}
    summary_o = json.loads((src / "summary.json").read_text()) if (src / "summary.json").exists() else {}
    summary_n = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    check = {
        "source": str(src), "replay": str(out), "error": error,
        "passes": {"original": len(passes_o), "replay": len(passes_n), "digests_equal": passes_o == passes_n},
        "decisions": {"original": kept_o, "replay": kept_n, "equal": kept_o == kept_n},
        "final_digest": {"original": summary_o.get("final_digest"), "replay": summary_n.get("final_digest"),
                         "equal": summary_o.get("final_digest") == summary_n.get("final_digest")},
        "readings": {"passes_compared": len(ro), "lines_differing_per_pass": text_diff,
                     "identical": bool(ro) and all(v == 0 for v in text_diff.values()) and set(ro) == set(rn)},
    }
    check["exact"] = (error is None and check["passes"]["digests_equal"] and check["decisions"]["equal"]
                      and check["final_digest"]["equal"] and check["readings"]["identical"])
    (out / "replay_check.json").write_text(json.dumps(check, indent=1))
    return check


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.replay")
    ap.add_argument("run"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    c = replay(Path(a.run), Path(a.out))
    print(json.dumps({k: c[k] for k in ("exact", "passes", "decisions", "final_digest")}, indent=1))
    if c["error"]:
        print(c["error"], file=sys.stderr)
    print("readings identical:", c["readings"]["identical"], c["readings"]["lines_differing_per_pass"])
    sys.exit(0 if c["exact"] else 1)


if __name__ == "__main__":
    main()
