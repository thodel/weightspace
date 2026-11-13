"""Controls harness: budget-0 keep, donor-rule mask, validation audit, planted corruptions (engine-free)."""
import json

import numpy as np
import pytest

from wsp import plant
from wsp.audit import audit_validation
from wsp.session import Session
from wsp.subject import ArraySubject

REFS = ["ein kurzer satz", "noch ein satz", "und das dritte", "ach ja", "mal sehen"]


def toy_forward(tensors, practice):
    return [it["text"].replace("a", "o") if tensors["gain"][0] < 1.0 else it["text"] for it in practice]


def make(tmp_path, budget, seen=None):
    subj = ArraySubject({"gain": np.array([0.5])}, forward=toy_forward)
    practice = [{"key": f"l{i}", "text": r, "seen_by": (seen or {}).get(i, [])} for i, r in enumerate(REFS)]
    return Session(subj, practice, tmp_path / "run", budget=budget)


def test_keep_unmeasured_only_at_budget_zero(tmp_path):
    s = make(tmp_path / "a", 0)
    s.edit("scale", ["gain"], factor=2.0)
    s.keep_unmeasured("blind")
    assert s.subject.tensors()["gain"][0] == 1.0 and s.freeze()["kept"] == 1
    s2 = make(tmp_path / "b", 3)
    s2.edit("scale", ["gain"], factor=2.0)
    with pytest.raises(RuntimeError):
        s2.keep_unmeasured("no")


def test_unseen_by_and_rule_mask(tmp_path):
    s = make(tmp_path, 3, seen={0: ["donor"], 1: ["donor"]})
    assert s.unseen_by("donor") == [False, False, True, True, True]
    assert s.unseen_by("other") == [True] * 5
    s.measure("C0")
    s.edit("scale", ["gain"], factor=2.0)
    s.measure("after", rule_mask=s.unseen_by("donor"))
    assert s._vs_kept["rule_lines"] == 3
    assert s.keep("cer fell")
    with pytest.raises(ValueError):
        s2 = make(tmp_path / "x", 3); s2.measure("C0"); s2.measure("bad", rule_mask=[True])


def test_audit_validation(tmp_path):
    job = tmp_path / "job"; (job / "data" / "pages").mkdir(parents=True)
    (job / "data" / "pages_train.lst").write_text("/x/pages/000000_K2911-14_0001.xml\n/x/pages/000001_K2911-15_0002.xml\n")
    rows = lambda docs: "".join(json.dumps({"key": f"p/{d}_0001/l1", "doc": d, "text": "x"}) + "\n" for d in docs)
    prac = tmp_path / "prac.jsonl"; prac.write_text(rows(["K2911,39"]))
    held = tmp_path / "held.jsonl"; held.write_text(rows(["K2911,46"]))
    assert audit_validation(prac, held, job, "object")["clean"]
    bad = tmp_path / "bad.jsonl"; bad.write_text(rows(["K2911,14"]))
    assert audit_validation(bad, held, job, "object")["findings"][0]["kind"] == "practice_unit_in_training_pages"
    assert audit_validation(held, held, job, "object")["findings"][0]["kind"] == "practice_unit_in_heldout"


def test_planted_corruptions_are_what_they_say():
    cm = {" ": 1, "e": 2, "n": 3}
    t = {plant.W: np.arange(12, dtype=np.float32).reshape(4, 3) + 1, plant.B: np.arange(4, dtype=np.float32) + 1}
    u = {k: v.copy() for k, v in t.items()}; plant.zero_row(u, cm, "e")
    assert (u[plant.W][2] == 0).all() and u[plant.B][2] == 0 and (u[plant.W][3] == t[plant.W][3]).all()
    u = {k: v.copy() for k, v in t.items()}; plant.negate_row(u, cm, "n")
    assert (u[plant.W][3] == -t[plant.W][3]).all()
    u = {k: v.copy() for k, v in t.items()}; plant.swap_rows(u, cm, "e", "n")
    assert (u[plant.W][2] == t[plant.W][3]).all() and u[plant.B][3] == t[plant.B][2]
    u = {k: v.copy() for k, v in t.items()}; plant.blank_bias(u, cm, 2.0)
    assert u[plant.B][0] == t[plant.B][0] + 2
    u = {k: v.copy() for k, v in t.items()}; plant.noise_block(u, cm, names=(plant.W,), rel_norm=0.25)
    assert np.linalg.norm(u[plant.W] - t[plant.W]) == pytest.approx(0.25 * np.linalg.norm(t[plant.W]), rel=1e-4)


def test_cluster_bootstrap_is_wider_when_lines_are_correlated():
    from wsp.metrics import bootstrap_delta
    import random
    rng = random.Random(1)
    # 20 pages x 10 lines; the edit helps on whole pages (page effect), so lines are strongly correlated
    before, after, pages = [], [], []
    for p in range(20):
        eff = rng.choice([-3, 0, 3])
        for _ in range(10):
            before.append((5, 40)); after.append((max(0, 5 + eff + rng.choice([-1, 0, 1])), 40)); pages.append(p)
    d = bootstrap_delta(before, after, resamples=400, seed=3, clusters=pages)
    assert d["clusters"] == 20
    w_line = d["ci95"][1] - d["ci95"][0]; w_cl = d["ci95_cluster"][1] - d["ci95_cluster"][0]
    assert w_cl > 1.5 * w_line
    assert "ci95_cluster" not in bootstrap_delta(before, after, resamples=50, seed=3)
