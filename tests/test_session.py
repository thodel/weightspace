import json

import numpy as np
import pytest

from wsp.session import Session, BudgetExhausted, SessionClosed
from wsp.subject import ArraySubject

REFS = ["ein kurzer satz", "noch ein satz", "und das dritte"]   # every line holds an "a": the toy edit moves every line


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
    assert s.keep() is False            # no pass since the edit: refused, undone
    s.freeze()
    with pytest.raises(SessionClosed):
        s.note("too late")


def test_keep_refused_when_interval_contains_zero(tmp_path):
    """The rule of PROTOCOL §3 (06.11.2026): a keep needs the paired 95 % interval of
    CER(after) − CER(last kept) to exclude 0; a noise-sized edit is refused and undone."""
    subj, s = make(tmp_path, budget=4)
    d0 = subj.digest()
    s.measure("baseline")
    s.edit("scale", ["w"], factor=2.0)   # w is zeros: the output does not change at all
    s.measure("after no-op")
    assert s.keep("looks fine") is False
    assert subj.digest() == d0           # the refused edit was undone
    log = [json.loads(l) for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    dec = [e for e in log if e["kind"] == "decision"][-1]
    assert dec["kept"] is False and dec["reason"] == "interval contains 0"
    assert dec["vs_kept"]["resamples"] == 2000 and dec["vs_kept"]["ci95"][0] <= 0 <= dec["vs_kept"]["ci95"][1]
    # a real improvement passes the same rule, and the kept state moves with it
    s.edit("scale", ["gain"], factor=2.0)
    s.measure("after scale")
    assert s.keep("cer fell") is True
    assert s._vs_kept["ci95"][1] < 0
    summary = s.freeze()
    assert summary["kept"] == 1 and summary["edits"] == 2 and summary["keep_rule"]["min_effect"] == 0.0


def test_keep_refused_when_cer_rises(tmp_path):
    subj, s = make(tmp_path, budget=3)
    subj.set_tensor("gain", np.array([2.0]))     # start from the clean state
    s.measure("baseline")
    s.edit("scale", ["gain"], factor=0.25)       # back below 1: a->o again, CER rises
    s.measure("worse")
    assert s.keep() is False
    dec = [json.loads(l) for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()][-1]
    assert dec["reason"] == "CER rose" and subj.tensors()["gain"][0] == 2.0


def test_edit_per_tensor_arguments_and_revert(tmp_path):
    import numpy as np
    from wsp.session import Session
    from wsp.subject import ArraySubject
    t = {"a": np.zeros((2, 3)), "b": np.zeros(4)}
    s = ArraySubject(t, forward=lambda ten, items: ["x" for _ in items])
    sess = Session(s, [{"key": "k", "text": "x"}], tmp_path, budget=3)
    sess.measure("C0")
    sess.edit("transplant", ["a", "b"], per_tensor={"a": {"donor": np.ones((2, 3))}, "b": {"donor": np.full(4, 2.0)}})
    assert (s.tensors()["a"] == 1).all() and (s.tensors()["b"] == 2).all()
    sess.revert("test")
    assert (s.tensors()["a"] == 0).all() and (s.tensors()["b"] == 0).all()


class CapSubject(ArraySubject):
    """Pages: the edit that fixes the text also makes every page run to the cap."""

    def forward(self, practice):
        from wsp.subject import Prediction
        texts = toy_forward(self._t, practice)
        cap = bool(self._t["gain"][0] >= 1.0)
        return [Prediction(key=i["key"], text=t, extra={"at_cap": cap}) for i, t in zip(practice, texts)]


def test_page_keep_refused_when_pages_at_cap_rise(tmp_path):
    subj = CapSubject({"gain": np.array([0.5]), "w": np.zeros((2, 2))})
    practice = [{"key": f"p{i}", "text": r} for i, r in enumerate(REFS)]
    s = Session(subj, practice, tmp_path / "run", budget=3, unit="page")
    s.measure("baseline")
    h = s.hypothesis("gain fixes a->o", "CER to 0, cap unchanged", "scaling")
    s.edit("scale", ["gain"], hypothesis=h, factor=2.0)
    m = s.measure("after scale")
    assert m.cer == 0
    assert s.keep_allowed() == (False, "pages at the cap rose (0 -> 3)")
    assert s.keep("cer fell") is False
    assert subj.tensors()["gain"][0] == 0.5
    events = [json.loads(l) for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert events[-2]["page"]["at_cap"] == 3


def test_a_pass_that_never_ran_is_not_counted(tmp_path):
    subj, s = make(tmp_path, budget=2)

    def refuse(practice):
        raise RuntimeError("pass not started: card 0 full")
    subj.forward = refuse
    with pytest.raises(RuntimeError):
        s.measure("refused")
    assert s.budget_left == 2
    assert not (tmp_path / "run" / "budget.json").exists()


def test_reopened_session_compares_against_the_kept_state(tmp_path):
    subj, s = make(tmp_path, budget=4)
    s.measure("baseline")
    h = s.hypothesis("gain fixes a->o", "CER to 0", "scaling")
    s.edit("scale", ["gain"], hypothesis=h, factor=2.0)
    s.measure("after scale")
    assert s.keep("cer fell")
    # a new process: the subject carries the kept weights, the session is built again
    again = Session(subj, s.practice, tmp_path / "run", budget=4)
    h = again.hypothesis("no-op", "no change", "none")
    again.edit("scale", ["w"], hypothesis=h, factor=1.0)
    again.measure("no-op")
    assert again._vs_kept["delta_cer"] == 0
    assert again._baseline_lines != again._kept_lines       # the baseline stayed the first pass
