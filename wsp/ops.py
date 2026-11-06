"""Edit operations: pure functions on numpy arrays, replayable from their arguments.

Every op takes the tensor (and donors as arrays) and returns a new array of
the same dtype. Names are the registry keys the Session logs.
"""
from __future__ import annotations

import numpy as np

REGISTRY: dict[str, callable] = {}


def op(fn):
    REGISTRY[fn.__name__] = fn
    return fn


@op
def scale(w: np.ndarray, factor: float) -> np.ndarray:
    return (w * factor).astype(w.dtype)


@op
def add(w: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if delta.shape != w.shape:
        raise ValueError(f"delta {delta.shape} does not match {w.shape}")
    return (w + delta).astype(w.dtype)


@op
def set_rows(w: np.ndarray, index: list[int], values: np.ndarray) -> np.ndarray:
    out = w.copy()
    out[np.asarray(index)] = values
    return out


@op
def set_cols(w: np.ndarray, index: list[int], values: np.ndarray) -> np.ndarray:
    out = w.copy()
    out[:, np.asarray(index)] = values
    return out


@op
def interpolate(w: np.ndarray, other: np.ndarray, alpha: float) -> np.ndarray:
    """(1-alpha)*w + alpha*other. alpha=1 is a transplant, alpha>1 extrapolates."""
    if other.shape != w.shape:
        raise ValueError(f"other {other.shape} does not match {w.shape}")
    return ((1.0 - alpha) * w + alpha * other).astype(w.dtype)


@op
def transplant(w: np.ndarray, donor: np.ndarray) -> np.ndarray:
    if donor.shape != w.shape:
        raise ValueError(f"donor {donor.shape} does not match {w.shape}")
    return donor.astype(w.dtype).copy()


@op
def project_out(w: np.ndarray, direction: np.ndarray, axis: int = 0,
                index: list[int] | None = None) -> np.ndarray:
    """Remove the component along a unit ``direction`` from selected rows/columns.

    ``axis=0``: each selected row ``w[i, :]`` loses its projection onto the
    direction (direction has length ``w.shape[1]``). ``axis=1``: each selected
    column ``w[:, j]`` does (direction has length ``w.shape[0]``). This is the
    Exp-8 edit of the post — subtract one direction from N output columns.
    """
    d = np.asarray(direction, dtype=np.float64)
    nrm = np.linalg.norm(d)
    if nrm == 0:
        raise ValueError("direction has zero norm")
    d = d / nrm
    out = w.astype(np.float64).copy()
    if axis == 0:
        if d.shape[0] != w.shape[1]:
            raise ValueError("direction length must equal the number of columns")
        rows = np.arange(w.shape[0]) if index is None else np.asarray(index)
        out[rows] -= np.outer(out[rows] @ d, d)
    elif axis == 1:
        if d.shape[0] != w.shape[0]:
            raise ValueError("direction length must equal the number of rows")
        cols = np.arange(w.shape[1]) if index is None else np.asarray(index)
        out[:, cols] -= np.outer(d, d @ out[:, cols])
    else:
        raise ValueError("axis must be 0 or 1")
    return out.astype(w.dtype)


@op
def drop_padding_rows(w: np.ndarray, keep: int, axis: int = 2) -> np.ndarray:
    """Keep one kernel row of a conv weight (out, in, kH, kW) and drop the rest.

    For kraken+'s ``Cr255,1,85,1,1`` on a one-row feature map only the centre
    row (index 127) ever sees data; the other 254 see zero padding and carry
    gradient-free weights. Keeping that row is function-preserving once the
    layer's padding is set to 0 in the VGSL spec (``Cr1,1,85,1,1``).
    """
    return np.take(w, [keep], axis=axis).copy()


def apply(name: str, w: np.ndarray, **kwargs) -> np.ndarray:
    if name not in REGISTRY:
        raise KeyError(f"unknown op {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](w, **kwargs)
