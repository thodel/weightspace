"""BRIEF_PHASE0 §7 smoke test: a budget-2 session whose only edit is a no-op.

Needs WSP_MODEL and WSP_PRACTICE like test_roundtrip.py. Creates a real run
folder under runs/ (so the artefacts can be inspected) named
``<date>-smoke-<model>-b2-<n>``; it is small (log, diff, summary, the
re-saved .mlmodel) and the readings are git-ignored.
"""
import json
import os
from pathlib import Path

import pytest

kraken = pytest.importorskip("kraken")

from wsp.run import new, open_session, RUNS  # noqa: E402


def test_noop_session_budget_2():
    model = os.environ.get("WSP_MODEL") or pytest.skip("WSP_MODEL not set")
    practice = os.environ.get("WSP_PRACTICE") or pytest.skip("WSP_PRACTICE not set")
    run_dir = new(model, practice, budget=2, run_id="smoke-" + Path(model).stem.replace("kraken-", ""))
    s = open_session(run_dir)
    base = s.measure("C0 baseline")
    assert base.lines == s._log[0]["practice_lines"]
    out_bias = next(n for n in s.subject.tensors() if n.startswith("O_") and n.endswith("lin.bias"))
    h = s.hypothesis("scaling the output bias by 1.0 changes nothing", "CER unchanged, 0 lines differ", "output bias")
    s.edit("scale", [out_bias], hypothesis=h, factor=1.0)
    after = s.measure("after no-op")
    assert after.cer == base.cer and after.collapsed == base.collapsed
    s.keep("no-op kept to exercise the path")
    summary = s.freeze()
    assert summary["used"] == 2 and summary["edits"] == 1 and summary["kept"] == 1
    assert summary["start_digest"] == summary["final_digest"]
    assert json.loads((run_dir / "diff.json").read_text()) == {}
    for key in ("budget", "used", "practice_id", "start_digest", "final_digest", "baseline", "last_pass",
                "changed_tensors", "heldout", "deviations", "final_file_sha256", "sandbox"):
        assert key in summary, key
    assert summary["changed_tensors"] == {}
    assert (run_dir / "final.mlmodel").exists() and (run_dir / "log.jsonl").exists()
    kinds = [json.loads(l)["kind"] for l in (run_dir / "log.jsonl").read_text().splitlines()]
    assert kinds == ["start", "pass", "hypothesis", "edit", "pass", "decision", "freeze"]
    # a second open sees the budget as spent and refuses a third pass
    s2 = open_session(run_dir)
    assert s2.budget_left == 0
