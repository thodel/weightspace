"""The defect operations of wsp.trocr_plant on plain arrays (no model, no download)."""
import numpy as np

from wsp import trocr_plant as P


def _t():
    rng = np.random.default_rng(0)
    return {"dec.embed.tokens": rng.standard_normal((40, 8)).astype(np.float32),
            "dec.06.cross.o.weight": rng.standard_normal((8, 128)).astype(np.float32),
            "dec.11.norm_ffn.weight": np.ones(8, np.float32), "dec.11.norm_ffn.bias": np.zeros(8, np.float32),
            "dec.embed.norm.weight": np.ones(8, np.float32) * 2, "dec.embed.norm.bias": np.zeros(8, np.float32)}


CTX = {"freq": [5, 7, 9, 11, 13, 15]}


def test_token_row_defects_touch_only_their_rows():
    t = _t(); o = {k: v.copy() for k, v in t.items()}
    P.zero_rows(t, CTX, k=2)
    assert not t["dec.embed.tokens"][[5, 7]].any() and np.array_equal(t["dec.embed.tokens"][[0, 1, 9]], o["dec.embed.tokens"][[0, 1, 9]])
    t = _t(); P.negate_rows(t, CTX, k=3)
    assert np.array_equal(t["dec.embed.tokens"][[5, 7, 9]], -o["dec.embed.tokens"][[5, 7, 9]])
    t = _t(); P.swap_rows(t, CTX, a=0, b=1)
    assert np.array_equal(t["dec.embed.tokens"][5], o["dec.embed.tokens"][7]) and np.array_equal(t["dec.embed.tokens"][7], o["dec.embed.tokens"][5])


def test_zero_head_zeroes_one_slice_of_64_columns():
    t = _t(); o = t["dec.06.cross.o.weight"].copy()
    P.zero_head(t, CTX, layer=6, head=1, kind="cross")
    w = t["dec.06.cross.o.weight"]
    assert not w[:, 64:128].any() and np.array_equal(w[:, :64], o[:, :64])


def test_noise_has_the_stated_relative_norm_and_is_reproducible():
    t = _t(); o = t["dec.06.cross.o.weight"].copy()
    P.noise(t, CTX, names=("dec.06.cross.o.weight",), rel_norm=0.25, seed=3)
    d = np.linalg.norm((t["dec.06.cross.o.weight"] - o).astype(np.float64)) / np.linalg.norm(o.astype(np.float64))
    assert abs(d - 0.25) < 1e-3
    t2 = _t(); P.noise(t2, CTX, names=("dec.06.cross.o.weight",), rel_norm=0.25, seed=3)
    assert np.array_equal(t["dec.06.cross.o.weight"], t2["dec.06.cross.o.weight"])


def test_shift_and_scale_and_the_registry_is_consistent():
    t = _t(); P.shift(t, CTX, name="dec.embed.norm.bias", rel_norm=1.0, seed=1)
    assert abs(np.linalg.norm(t["dec.embed.norm.bias"]) - np.linalg.norm(t["dec.embed.norm.weight"])) < 1e-4
    t = _t(); P.scale(t, CTX, names=("dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"), factor=0.5)
    assert np.allclose(t["dec.11.norm_ffn.weight"], 0.5)
    for d, spec in P.DEFECTS.items():
        assert spec["sizes"] and spec["what"] and spec["scope"] and spec["donor"] and callable(spec["fn"]), d
        for sz, kw in spec["sizes"].items():
            names = kw.get("names", ())
            assert all(n.startswith(("enc.", "dec.")) for n in names), (d, sz)
