import numpy as np

from wsp import diff, align


def test_compare_and_sparse_roundtrip(tmp_path):
    a = {"w": np.zeros((3, 2), dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    b = {"w": a["w"].copy(), "b": a["b"].copy()}
    b["w"][2, 1] = 5.0
    c = diff.compare(a, b)
    assert c["w"]["changed"] == 1 and c["w"]["top_rows"][0][0] == 2
    assert c["b"]["changed"] == 0
    sd = diff.sparse_diff(a, b)
    assert list(sd) == ["w"] and sd["w"]["index"] == [5]
    p = diff.save_sparse_diff(sd, tmp_path / "d.json")
    back = diff.apply_sparse_diff(a, diff.load_sparse_diff(p))
    assert np.array_equal(back["w"], b["w"])


def test_codec_alignment_by_character():
    ca = {"a": 1, "b": 2, "c": 3}
    cb = {"b": 1, "a": 2, "z": 3}
    m = align.codec_row_map(ca, cb)
    assert m == {1: 2, 2: 1}
    wb = np.array([[0.0], [10.0], [20.0], [30.0]])
    g = align.gather_rows(wb, m, n_rows_a=4)
    assert g[1, 0] == 20.0 and g[2, 0] == 10.0 and g[3, 0] == 0.0
