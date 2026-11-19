"""Stage-4 harness pieces (docs/TROCR_STAGE4_PLAN.md section 7): the revert-after-keep refusal, shams, margins, interpolation search, object split, and the
file allow-list of a donor-free sandbox."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from wsp import landlock, stage4


def _t(seed=0):
    r = np.random.default_rng(seed)
    return {"a": r.standard_normal((6, 4)).astype(np.float32), "b": r.standard_normal(5).astype(np.float32), "c": r.standard_normal((3, 3)).astype(np.float32)}


def test_sham_has_the_norm_of_the_real_change_on_the_same_tensors_and_is_seeded():
    before = _t(); after = {k: v.copy() for k, v in before.items()}
    after["a"] = after["a"] * 1.3; after["b"] = after["b"] + 0.5
    d = stage4.edit_delta(before, after)
    assert set(d) == {"a", "b"}
    shams = stage4.sham_deltas(d, seed=7, k=5)
    assert len(shams) == 5 and all(set(s) == {"a", "b"} for s in shams)
    for s in shams:
        for n in d:
            assert abs(np.linalg.norm(s[n]) - np.linalg.norm(d[n])) < 1e-9 * max(1, np.linalg.norm(d[n]))
            assert s[n].shape == d[n].shape
    assert not np.allclose(shams[0]["a"], shams[1]["a"])
    again = stage4.sham_deltas(d, seed=7, k=5)
    assert all(np.array_equal(x["a"], y["a"]) for x, y in zip(shams, again))
    assert not np.array_equal(stage4.sham_deltas(d, seed=8, k=1)[0]["a"], shams[0]["a"])


def test_margin_states_are_the_edit_at_other_sizes():
    before = _t(); after = {k: v.copy() for k, v in before.items()}
    after["a"] = after["a"] * 2.0
    m = stage4.margin_states(before, after, (0.5, 1.0, 2.0))
    assert np.allclose(m[1.0]["a"], after["a"], atol=1e-6)
    assert np.allclose(m[0.5]["a"], before["a"] * 1.5, atol=1e-6) and np.allclose(m[2.0]["a"], before["a"] * 3.0, atol=1e-5)
    assert set(m[0.5]) == {"a"}                                     # only the changed tensor


def test_interpolation_and_line_search():
    s, d = _t(1), _t(2)
    h = stage4.interpolate_all(s, d, 0.25)
    assert np.allclose(h["a"], 0.75 * s["a"] + 0.25 * d["a"], atol=1e-6)
    assert np.array_equal(stage4.interpolate_all(s, d, 0.0)["b"], s["b"])
    res = stage4.line_search(lambda a: abs(a - 0.4) + 0.2, baseline_cer=0.30)
    assert res["alpha"] == 0.4 and res["passes"] == 10 and len(res["grid"]) == 9
    assert stage4.line_search(lambda a: 0.5, baseline_cer=0.3)["alpha"] is None


def test_object_split_is_disjoint_balanced_and_fixed_by_seed():
    rows = [{"doc": f"D{i % 9}"} for i in range(90 + 7)]
    a = stage4.split_objects(rows, seed=20261119)
    assert set(a["objects"]["discovery"]).isdisjoint(a["objects"]["confirmation"])
    assert sum(a["n_objects"].values()) == 9 and sum(a["lines"].values()) == len(rows)
    assert abs(a["lines"]["discovery"] - a["lines"]["confirmation"]) <= 20
    assert a == stage4.split_objects(rows, seed=20261119)
    assert stage4.half_of("D3", a) in ("discovery", "confirmation") and stage4.half_of("nope", a) is None


class _FakeSubject:                                               # enough of a Subject for Session: no model
    def __init__(self):
        self.t = {"w": np.ones(4, np.float32)}

    def tensors(self): return self.t
    def digest(self): return "d"
    def describe(self): return {}
    def apply(self, op, names, **kw):
        for n in names:
            self.t[n] = self.t[n] * kw["factor"]
    def set_tensor(self, n, v): self.t[n] = v


def test_revert_or_keep_with_no_pending_edit_is_a_logged_refusal_not_a_crash(tmp_path):
    from wsp.session import Session
    s = Session(_FakeSubject(), [{"key": "k_p1_l1", "text": "a"}], tmp_path, budget=3)
    h = s.hypothesis("x", "y")
    s.edit("scale", ["w"], hypothesis=h, factor=2.0)
    s.revert("first")                                              # decides the edit
    s.revert("again")                                              # p01's crash: nothing pending -> refusal
    assert s.keep("late") is False
    ev = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [e["call"] for e in ev if e["kind"] == "refusal"] == ["revert", "keep"]
    assert float(s.subject.t["w"][0]) == 1.0                       # the state was not touched by the refusals
    assert [e["kept"] for e in ev if e["kind"] == "decision"] == [False]


def test_landlock_hides_the_donor_and_everything_else(tmp_path):
    if landlock.abi_version() < 1:
        pytest.skip("Landlock not available")
    donor, subj, run, other = tmp_path / "donor", tmp_path / "subject", tmp_path / "run", tmp_path / "other_run"
    for d in (donor, subj, run, other):
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"x" * 16)
    code = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
        from wsp import landlock
        landlock.restrict(read=["/usr", "/lib", "/lib64", "/etc", "/proc", {str(Path(sys.executable).resolve().parent.parent)!r}, {str(subj)!r}], write=[{str(run)!r}])
        seen = landlock.openable({[str(d / "model.safetensors") for d in (donor, subj, run, other)]!r})
        try:
            (open({str(other / 'w.txt')!r}, "w")).write("x"); wrote_other = True
        except OSError:
            wrote_other = False
        open({str(run / 'w.txt')!r}, "w").write("ok")
        print(json.dumps({{"seen": seen, "wrote_other": wrote_other}}))
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["seen"] == sorted([str(subj / "model.safetensors"), str(run / "model.safetensors")])   # its subject and its own folder, no donor, no other run
    assert out["wrote_other"] is False and (run / "w.txt").read_text() == "ok"
