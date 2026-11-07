"""Engine-free checks of wsp.layerview: frame map, CTC alignment, ridge probes, summaries and both figures."""
import numpy as np
import pytest

from wsp import layerview as lv


def _logp(path, classes=4, sharp=8.0):
    """Log-posteriors whose best path is ``path`` (one label per frame)."""
    x = np.zeros((len(path), classes))
    x[np.arange(len(path)), path] = sharp
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def test_to_frames_and_input_frames():
    act = np.arange(20, dtype=float).reshape(10, 2)          # 10 columns, 2 features
    f = lv.to_frames(act, 3)                                # 3 columns per frame, the 10th dropped
    assert f.shape == (3, 2) and f[0, 0] == pytest.approx(2.0)
    img = np.arange(2 * 8, dtype=float).reshape(2, 8)       # height 2, width 8
    x = lv.input_frames(img, 2)
    assert x.shape == (2, 8) and list(x[0, :2]) == [0.0, 8.0]   # column-major per frame


def test_forced_align_follows_the_posteriors_and_keeps_doubles_apart():
    labels = [1, 1, 2]                                      # a double letter needs a blank between
    lp = _logp([0, 1, 1, 0, 1, 0, 2, 2, 0])
    runs = lv.forced_align(lp, labels)
    assert runs == [(1, 2), (4, 4), (6, 7)]
    assert lv.forced_align(lp[:2], labels) is None          # too few frames
    # a character the model does not emit still gets frames, where its posterior is least bad
    lp2 = _logp([0, 1, 0, 0, 0, 2, 0])
    lp2[3, 3] = np.log(0.3)
    r = lv.forced_align(lp2, [1, 3, 2])
    assert r[0][0] >= 1 and r[0][1] < r[1][0] and r[1][1] < r[2][0]


def test_greedy_and_char_status():
    lp = _logp([0, 1, 1, 0, 1, 2, 0])
    assert lv.greedy(lp) == [(1, 1, 2), (1, 4, 4), (2, 5, 5)]
    assert lv.char_status("ab c", "ac") == ["ok", "del", "del", "ok"]
    assert lv.char_status("abc", "axc") == ["ok", "sub", "ok"]


def test_ridge_total_equals_one_pass():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)); y = (X[:, 0] > 0).astype(int)
    one = lv.Ridge(5, 2); one.add(X, y)
    parts = [lv.Ridge(5, 2) for _ in range(2)]; parts[0].add(X[:120], y[:120]); parts[1].add(X[120:], y[120:])
    W1, c1 = one.solve(1e-3); W2, c2 = lv.Ridge.total(parts).solve(1e-3)
    assert np.allclose(W1, W2) and np.allclose(c1, c2)
    assert ((X @ W1 + c1).argmax(1) == y).mean() > 0.95


def test_fold_docs_keeps_documents_whole_and_is_deterministic():
    items = [{"key": f"d{i % 7}/l{i}", "doc": f"d{i % 7}"} for i in range(70)]
    f1, f2 = lv.fold_docs(items), lv.fold_docs(list(reversed(items)))
    assert f1 == f2
    by_doc = {}
    for it in items:
        by_doc.setdefault(it["doc"], set()).add(f1[it["key"]])
    assert all(len(v) == 1 for v in by_doc.values())
    assert set(f1.values()) == set(range(lv.FOLDS))


def _row(ref, status, ident, emit):
    full = lambda v: {layer: v.get(layer, "0" * len(ref)) for layer in lv.LAYERS + ["model"]}
    return {"key": "x", "ref": ref, "hyp": "", "status": status, "runs": [(i, i) for i in range(len(ref))],
            "identity": full(ident), "emission": full(emit),
            "frame": {layer: {"identity_frame_acc": (0.5, 10), "emission_balanced_acc": (0.6, 20)} for layer in lv.LAYERS}}


def test_summarise_and_lost_at():
    # 'a' kept; ' ' deleted but both probes still have it at L_16; 'b' deleted and never identified
    ln = _row("a b", ["ok", "del", "del"], {"L_14": "110", "L_16": "110", "O_18": "100", "model": "100"},
              {"L_14": "110", "L_16": "111", "O_18": "101", "model": "100"})
    s = lv.summarise([ln])
    assert s["deleted_lost_at"] == {"L_16": 1, "never": 1} and s["deleted_space_lost_at"] == {"L_16": 1}
    assert s["layers"]["L_16"]["space_del"]["emission"] == 1.0 and s["layers"]["O_18"]["char_del"]["identity"] == 0.0
    assert s["layers"]["L_14"]["identity_frame_acc"] == 0.5


def test_figures_render_without_a_model(tmp_path):
    pytest.importorskip("matplotlib")
    from wsp.viz import render
    ln = _row("ab c", ["ok", "ok", "del", "sub"], {"L_16": "1110", "O_18": "1100"}, {"L_16": "1111", "O_18": "1101"})
    data = {**lv.summarise([ln]), "model": "toy", "set": "toy-set", "meta": {"lines_scored": 1}}
    assert len(render(lv.plot_probe, data, tmp_path, "probe", "svg")) == 2
    T = 12
    d = {"key": "toy/line", "model": "toy", "image": np.random.default_rng(1).random((120, T * 8)), "px_per_frame": 8,
         "frames": T, "conv": {b: np.random.default_rng(2).random((4, T)) for b in lv.BLOCKS[:4]},
         "lstm": {b: (np.ones(T), np.ones(T) * 2) for b in lv.BLOCKS[4:7]}, "blank": np.full(T, 0.8),
         "top": np.full(T, 0.2), "ref": "ab c", "hyp": "abc", "status": ["ok", "ok", "del", "ok"],
         "runs": [(1, 1), (3, 4), (6, 6), (9, 9)], "hyp_chars": [("a", 1, 1), ("b", 3, 4), ("c", 9, 9)],
         "identity": ln["identity"], "emission": ln["emission"]}
    paths = render(lv.plot_decision, d, tmp_path, "line", "png")
    assert all(p.stat().st_size > 5_000 for p in paths)
