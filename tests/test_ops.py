import numpy as np
import pytest

from wsp import ops


def test_scale_add_interpolate():
    w = np.ones((2, 3), dtype=np.float32)
    assert ops.apply("scale", w, factor=2.0).sum() == 12
    assert ops.apply("add", w, delta=np.full((2, 3), -1.0)).sum() == 0
    o = ops.apply("interpolate", w, other=np.zeros((2, 3)), alpha=0.25)
    assert np.allclose(o, 0.75)


def test_project_out_removes_direction():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 5)).astype(np.float32)
    d = rng.normal(size=5)
    out = ops.apply("project_out", w, direction=d, axis=0, index=[1, 2])
    d = d / np.linalg.norm(d)
    assert np.allclose(out[[1, 2]] @ d, 0, atol=1e-5)
    assert np.allclose(out[0], w[0])


def test_project_out_axis1():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(6, 4)).astype(np.float32)
    d = rng.normal(size=6)
    out = ops.apply("project_out", w, direction=d, axis=1)
    assert np.allclose((d / np.linalg.norm(d)) @ out, 0, atol=1e-5)


def test_drop_padding_rows_keeps_centre():
    w = np.arange(2 * 3 * 255 * 1, dtype=np.float32).reshape(2, 3, 255, 1)
    out = ops.apply("drop_padding_rows", w, keep=127, axis=2)
    assert out.shape == (2, 3, 1, 1)
    assert np.array_equal(out[:, :, 0, 0], w[:, :, 127, 0])


def test_shape_mismatch_refused():
    with pytest.raises(ValueError):
        ops.apply("transplant", np.ones((2, 2)), donor=np.ones((3, 2)))
    with pytest.raises(KeyError):
        ops.apply("nope", np.ones(2))
