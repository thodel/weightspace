"""Engine-free checks of wsp.alpha_overlay: interpolation and restore, edit kinds, the sweep check, figures."""
import numpy as np
import pytest

from wsp import alpha_overlay as ao


class Toy:
    def __init__(self):
        self.t = {"L_16.w": np.array([2.0, 4.0])}

    def tensors(self):
        return self.t

    def set_tensor(self, n, v):
        self.t[n] = np.asarray(v, dtype=float)


def test_interpolated_sets_and_restores():
    c = Toy()
    par = {"L_16.w": np.array([0.0, 0.0])}
    with ao.interpolated(c, par, 0.5):
        assert list(c.t["L_16.w"]) == [1.0, 2.0]
    with ao.interpolated(c, par, 1.5):
        assert list(c.t["L_16.w"]) == [3.0, 6.0]
    assert list(c.t["L_16.w"]) == [2.0, 4.0]
    with pytest.raises(RuntimeError):
        with ao.interpolated(c, par, 0.0):
            raise RuntimeError
    assert list(c.t["L_16.w"]) == [2.0, 4.0]              # restored after an error too


def test_edit_counts_by_kind():
    e = ao.edit_counts(["ab", "a bx", "ac"], ["a b", "a b", "ab"])
    assert e == {"del_letter": 0, "del_space": 1, "ins_letter": 1, "ins_space": 0, "sub": 1}


def test_check_against_sweep():
    summary = {"L_16@0.5": {"cer": 0.08767}, "child": {"cer": 0.06}}
    sweep = {"rows": [{"block": "L_16", "alpha": 0.5, "metrics": {"cer": 0.08767}},
                      {"block": "C_0", "alpha": 1.5, "metrics": {"cer": 0.07}}]}
    assert ao.check(summary, sweep) == [{"tag": "L_16@0.5", "here": 0.08767, "committed": 0.08767, "equal": True}]


def _data(T=10):
    st = lambda status: {"hyp": "x", "status": status, "p_blank": list(np.linspace(0, 1, T)), "chars": []}
    states = {"child": st("ooo")}
    summary = {"child": {"cer": 0.05, "edits": dict.fromkeys(["del_letter", "del_space", "ins_letter", "ins_space", "sub"], 1)}}
    for b in ("L_12", "L_16"):
        for a in ao.ALPHAS:
            states[f"{b}@{a:g}"] = st("odo" if a == 0 else "ooo")
            summary[f"{b}@{a:g}"] = {"cer": 0.06, "edits": {**summary["child"]["edits"], "del_space": 3}}
    return states, summary


def test_figures_and_pick(tmp_path):
    pytest.importorskip("matplotlib")
    from wsp.viz import render
    states, summary = _data()
    data = {"summary": summary, "blocks": ["L_12", "L_16"], "alphas": ao.ALPHAS, "child": "c", "parent": "p", "practice": "s"}
    assert len(render(ao.plot_summary, data, tmp_path, "sum", "svg")) == 2
    d = {"key": "k", "child": "c", "parent": "p", "blocks": ["L_12", "L_16"],
         "tags": [("0", "0"), ("0.5", "0.5"), ("1", "child"), ("1.5", "1.5")], "ref": "a b",
         "runs": [(1, 1), (4, 5), (8, 8)], "image": np.ones((120, 80)), "px_per_frame": 8, "states": states}
    assert all(p.stat().st_size > 5_000 for p in render(ao.plot_line, d, tmp_path, "line", "png"))
    lines = [{"key": "long", "ref": "x" * 20, "runs": [(0, 0)] * 20,
              "states": {"child": {"hyp": "x" * 20, "status": "o" * 20}, "L_16@0.5": {"hyp": "x" * 18, "status": "o" * 18 + "dd"}}},
             {"key": "short", "ref": "xx", "runs": [(0, 0)] * 2,
              "states": {"child": {"hyp": "xx", "status": "oo"}, "L_16@0.5": {"hyp": "", "status": "dd"}}}]
    assert ao.pick_lines({"lines": lines}, 5) == ["long"]
