import numpy as np

from wsp.family import _gram, _stats, body_spec, cka_from_grams, lineages


def test_cka_is_invariant_to_unit_order_and_scale():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    perm = rng.permutation(16)
    assert abs(cka_from_grams(_gram(X), _gram(X)) - 1) < 1e-12
    assert abs(cka_from_grams(_gram(X), _gram(3.0 * X[:, perm])) - 1) < 1e-9
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))            # any orthogonal mix of units
    assert abs(cka_from_grams(_gram(X), _gram(X @ Q)) - 1) < 1e-9
    Y = rng.normal(size=(200, 24))
    assert cka_from_grams(_gram(X), _gram(Y)) < 0.3            # unrelated features, different width


def test_stats_ignores_structural_zeros_and_counts_identity():
    x = np.array([0.0, 0.0, 1.0, 2.0, 3.0]); y = np.array([0.0, 0.0, 1.0, 2.5, 3.0])
    s = _stats([x], [y])
    assert s["structural_zero"] == 2 and abs(s["identical"] - 2 / 3) < 1e-12
    assert s["cosine"] > 0.99


def test_body_spec_ignores_alphabet_size():
    a = "[1,120,0,1 Cr{C_0}3,13,32 Lbx{L_16}200 Do{Do_17} O{O_18}1c297]"
    b = "[1,120,0,1 Cr{C_0}3,13,32 Lbx{L_16}200 Do{Do_17} O{O_18}1c135]"
    assert body_spec(a) == body_spec(b)


def test_lineages_join_on_body_cosine():
    w = {"models": {"a": {}, "b": {}, "c": {}, "d": {}},
         "pairs": {"a|b": {"body": {"cosine": 0.9}}, "b|c": {"body": {"cosine": 0.75}},
                   "a|c": {"body": {"cosine": 0.7}}, "a|d": {"body": {"cosine": 0.03}},
                   "b|d": {"body": {"cosine": 0.04}}, "c|d": {"body": {"cosine": 0.02}}}}
    assert lineages(w) == [["a", "b", "c"], ["d"]]
