import json

import numpy as np
import pytest

from wsp.session import Session, BudgetExhausted, SessionClosed
from wsp.subject import ArraySubject

REFS = ["ein kurzer satz", "noch ein satz", "und der dritte"]


def toy_forward(tensors, practice):
    """A toy recogniser: emits the reference, but swaps 'a'->'o' while gain[0] < 1."""
    out = []
    for item in practice:
        t = item["text"]
        if tensors["gain"][0] < 1.0:
            t = t.replace("a", "o")
        out.append(t)
    return out


def make(tmp_path, budget=3):
    subj = ArraySubject({"gain": np.array([0.5]), "w": np.zeros((2, 2))}, forward=toy_forward)
    practice = [{"key": f"l{i}", "text": r} for i, r in enumerate(REFS)]
    return subj, Session(subj, practice, tmp_path / "run", budget=budget)


def test_edit_measure_keep_improves(tmp_path):
    subj, s = make(tmp_path)
    base = s.measure("baseline")
    assert base.cer > 0
    h = s.hypothesis("gain below 1 corrupts a->o", "CER to 0 when gain >= 1", "scaling")
    s.edit("scale", ["gain"], hypothesis=h, factor=2.0)
    m = s.measure("after scale")
    assert m.cer == 0
    s.keep("cer fell")
    summary = s.freeze()
    assert summary["kept"] == 1 and summary["used"] == 2
    assert "gain" in summary["changed_tensors"] and "w" not in summary["changed_tensors"]
    assert (tmp_path / "run" / "diff.json").exists()
    log = [json.loads(l) for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in log] == ["start", "pass", "hypothesis", "edit", "pass", "decision", "freeze"]


def test_revert_restores_digest(tmp_path):
    subj, s = make(tmp_path)
    d0 = subj.digest()
    s.edit("scale", ["gain"], factor=3.0)
    assert subj.digest() != d0
    s.revert("no better")
    assert subj.digest() == d0


def test_budget_enforced_and_persisted(tmp_path):
    subj, s = make(tmp_path, budget=1)
    s.measure()
    with pytest.raises(BudgetExhausted):
        s.measure()
    # a restarted session in the same folder remembers the count
    subj2, s2 = make(tmp_path, budget=1)
    with pytest.raises(BudgetExhausted):
        s2.measure()


def test_no_second_edit_before_decision_and_closed_after_freeze(tmp_path):
    subj, s = make(tmp_path)
    s.edit("scale", ["gain"], factor=2.0)
    with pytest.raises(RuntimeError):
        s.edit("scale", ["gain"], factor=2.0)
    s.keep()
    s.freeze()
    with pytest.raises(SessionClosed):
        s.note("too late")
