import numpy as np
import pytest

from wsp import stage5


def _ts(v, extra=0.0):
    return {"a": np.full((2, 3), v, dtype=np.float32), "b": np.arange(4, dtype=np.float32) + extra}


def test_average_uniform_and_weighted():
    out, rep = stage5.average([_ts(1.0), _ts(3.0)])
    assert np.allclose(out["a"], 2.0) and rep["weight_sum"] == 1.0 and rep["tensors"] == 2
    out, rep = stage5.average([_ts(1.0), _ts(3.0)], [0.25, 0.75])
    assert np.allclose(out["a"], 2.5)
    out, _ = stage5.average([_ts(1.0)] * 7)
    assert np.array_equal(out["a"], _ts(1.0)["a"])            # the average of identical models is the model, bit for bit


def test_average_rejects_bad_weights_and_shapes():
    with pytest.raises(ValueError):
        stage5.average([_ts(1.0), _ts(2.0)], [0.5, 0.6])
    with pytest.raises(ValueError):
        stage5.average([_ts(1.0), _ts(2.0)], [1.5, -0.5])
    bad = _ts(2.0)
    bad["a"] = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        stage5.average([_ts(1.0), bad])
    missing = {"a": _ts(2.0)["a"]}
    with pytest.raises(ValueError, match="names"):
        stage5.average([_ts(1.0), missing])
    nan = _ts(2.0)
    nan["a"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        stage5.average([_ts(1.0), nan])


def test_vocab_check():
    with pytest.raises(ValueError, match="vocabulary"):
        stage5.check_compatible([_ts(1), _ts(1)], [{"x": 0}, {"x": 1}])


def test_toward_matches_interpolation():
    s, o = _ts(1.0), _ts(3.0)
    t = stage5.toward(s, o, 0.6)
    assert np.allclose(t["a"], 1.0 + 0.6 * 2.0)
    assert np.array_equal(stage5.toward(s, o, 0.0)["a"], s["a"])
    assert np.allclose(stage5.toward(s, o, 1.0)["a"], o["a"])
    # average of two with weight 0.4/0.6 equals the interpolation at alpha 0.6 (the equal-mixing-weight comparison)
    avg, _ = stage5.average([s, o], [0.4, 0.6])
    assert np.allclose(avg["a"], t["a"], atol=1e-6)


def test_seed_mean_edits():
    m = stage5.seed_mean_edits([[(2, 10), (0, 5)], [(4, 10), (1, 5)]])
    assert m == [(3.0, 10), (0.5, 5)]
    with pytest.raises(ValueError):
        stage5.seed_mean_edits([[(2, 10)], [(4, 11)]])
    with pytest.raises(ValueError):
        stage5.seed_mean_edits([[(2, 10)], [(4, 10), (1, 5)]])


def test_classes():
    c = stage5.classify
    assert c(0.02, [0.01, 0.03], 0.05, [0.04, 0.06]) == "harms"
    assert c(-0.05, [-0.06, -0.04], -0.01, [-0.02, -0.005]) == "beats"
    assert c(-0.01, [-0.02, -0.005], 0.02, [0.01, 0.03]) == "below"
    assert c(0.0, [-0.001, 0.001], 0.03, [0.02, 0.04]) == "below"           # no gain at all is below arm I
    assert c(-0.03, [-0.04, -0.02], 0.0, [-0.01, 0.01]) == "matches"
    assert c(-0.003, [-0.01, 0.004], 0.0, [-0.01, 0.01]) == "unresolved"
    assert c(-0.03, [-0.04, 0.001], 0.0, [-0.01, 0.01]) == "unresolved"     # own interval reaches 0: not a gain
    with pytest.raises(ValueError):
        c(float("nan"), [0, 0], 0, [0, 0])


def test_select_other():
    cs = [{"name": "c4-small-lr1e-5-s2", "delta_cer": -0.012, "ci95_cluster": [-0.016, -0.008]},
          {"name": "sib-all7", "delta_cer": -0.02, "ci95_cluster": [-0.03, -0.01]},
          {"name": "c4-lr5e-5", "delta_cer": -0.03, "ci95_cluster": [-0.05, 0.001]}]      # reaches 0: not eligible
    assert stage5.select_other(cs)["name"] == "sib-all7"
    assert stage5.select_other([cs[2]]) is None
    assert stage5.select_other([]) is None
    tie = [{"name": "b", "delta_cer": -0.02, "ci95_cluster": [-0.03, -0.01]}, {"name": "a", "delta_cer": -0.02, "ci95_cluster": [-0.03, -0.01]}]
    assert stage5.select_other(tie)["name"] == "a"
