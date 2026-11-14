"""wsp.atlas_select on toy data: frame labels, AUROC against sklearn, the page bootstrap, probes, profiles, the card page."""
import json

import numpy as np
import pytest

from wsp import atlas_select as S


def _lines():
    # two lines; frames: runs of one or two frames, margins at both ends
    a = {"key": "p/a/l1", "page": "p/a", "split": 0, "ref": "ab cd", "frames": 14,
         "runs": [[1, 2], [3, 3], [5, 5], [7, 8], [10, 10]]}
    b = {"key": "p/b/l1", "page": "p/b", "split": 1, "ref": "Ea, b", "frames": 12,
         "runs": [[2, 2], [3, 4], [5, 5], [7, 7], [9, 9]]}
    return [a, b]


def test_frame_table_labels_runs_margins_and_positions():
    ft = S.frame_table(_lines())
    assert len(ft["line"]) == 26
    first = ft["line"] == 0
    assert list(ft["char"][first][:4]) == ["", "a", "a", "b"]
    assert list(ft["margin"][first][[0, 11, 12, 13]]) == [1, 2, 2, 2]
    assert ft["wpos"][first][1] == "word-initial" and ft["wpos"][first][3] == "word-final"
    assert ft["prev"][first][3] == "a" and ft["next"][first][3] == " "
    sec = ft["line"] == 1
    assert "word-internal" in ft["wpos"][sec][3]                           # 'a' in "Ea,"
    assert set(ft["shuf"][ft["run"]]) <= set("abcdE, ")


def test_shuffle_keeps_spaces_and_characters():
    rng = np.random.default_rng(0)
    for ref in ["ab cd ef", "x", "  a  "]:
        s = S.shuffled(ref, rng)
        assert sorted(s) == sorted(ref) and [i for i, c in enumerate(s) if c == " "] == [i for i, c in enumerate(ref) if c == " "]


def test_char_classes():
    assert S.char_class(" ") == ["space"]
    assert set(S.char_class("B")) == {"upper", "ascender"}
    assert S.char_class(",") == ["punct"] and S.char_class("7") == ["digit"]
    assert "descender" in S.char_class("g") and "round" in S.char_class("o") and "ascender" in S.char_class("ſ")


def test_auroc_matches_sklearn_and_respects_the_universe():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 3)); X[:, 2] = 0.0                            # unit 2 is constant (a dead filter)
    pos = rng.random(n) < 0.3
    X[pos, 0] += 1.0
    uni = rng.random(n) < 0.8
    rows = rng.random(n) < 0.6
    A = S.auroc(X, pos[None], uni[None], rows)
    m = uni & rows
    assert A[0, 0] == pytest.approx(roc_auc_score(pos[m], X[m, 0]))
    assert A[0, 1] == pytest.approx(roc_auc_score(pos[m], X[m, 1]))
    assert A[0, 2] == 0.5 and S.sel(A[0, 2]) == 0


def test_page_auc_point_equals_auroc_and_interval_contains_it():
    rng = np.random.default_rng(2)
    n = 600
    pages = rng.integers(0, 12, n)
    pos = rng.random(n) < 0.4
    x = rng.normal(size=n) + 0.8 * pos
    r = S.page_auc(x, pos, pages, resamples=400)
    assert r["auroc"] == pytest.approx(S.auroc(x[:, None], pos[None], np.ones((1, n), bool))[0, 0])
    assert r["s_ci95"][0] <= r["s"] <= r["s_ci95"][1]


def test_sparse_probe_finds_a_separable_code_and_bootstrap_is_paired():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 3000
    y = rng.choice(list("abc"), n)
    X = rng.normal(size=(n, 20)).astype(np.float32)
    for k, c in enumerate("abc"):
        X[y == c, k] += 3.0
    page = rng.integers(0, 10, n)
    split = (page % 2).astype(int)
    tune = (page % 4) == 2
    r = S.probe(X, y, split, page, tune)
    assert r["bal"] > 0.9 and r["units_used"] <= 20
    noise = dict(r, pred=rng.choice(list("abc"), len(r["y"])))
    b1, b2 = S.bal_boot([r, noise], resamples=300)
    assert np.all(b1 - b2 > 0.3)


def test_probe_subsample_keeps_frames_and_labels_together(monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(S, "MAX_TRAIN", 300)               # fewer than the selection frames: the subsample is drawn
    rng = np.random.default_rng(4)
    n = 3000
    y = rng.choice(list("abc"), n)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    for k, c in enumerate("abc"):
        X[y == c, k] += 3.0
    page = rng.integers(0, 10, n)
    r = S.probe(X, y, (page % 2).astype(int), page, (page % 4) == 2)
    assert r["bal"] > 0.9


def test_corr_is_per_column_and_nan_for_constant_units():
    a = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    b = np.array([[2.0, 1.0], [4.0, 5.0], [6.0, 2.0]])
    C = S._corr(a, b)
    assert C[0, 0] == pytest.approx(1.0) and np.isnan(C[1, 1])


def test_block_columns_and_names():
    names = S.unit_names()
    cols = S.block_cols(names)
    assert len(names) == 1392 and len(cols["C_6"]) == 64 and len(cols["L_14 rev"]) == 200
    assert names[cols["L_14 rev"][0]] == "L_14 rev 0" and names[cols["L_14 fwd"][-1]] == "L_14 fwd 199"


def test_card_page_is_self_contained_and_cannot_close_its_script():
    d = {"model": "m", "set": "s", "lines": 1, "gate2": {"passes": False},
         "cards": [{"unit": "C_0 ch 0", "dead": False, "top": [["char:</script>", 0.9]], "best_selection": "char:a",
                    "auroc": 0.9, "random": 0.5, "blank": 0.5, "shuffle": 0.5, "means": [0.1], "line": [0.0] * 20}]}
    page = S.render_cards(d)
    assert page.count("</script>") == 2 and "http" not in page.split("<script type=application/json")[0].lower()
    blob = page.split("id=data>")[1].split("</script>")[0]
    assert json.loads(blob)["cards"][0]["top"][0][0] == "char:</script>"
