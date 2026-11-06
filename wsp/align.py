"""Alignment of rows between models whose codecs or module names differ.

Index-wise merging of two kraken output layers is wrong as soon as one codec
grew (``resize: union``); rows are paired by the character they emit.
"""
from __future__ import annotations

import numpy as np


def codec_row_map(codec_a: dict[str, int], codec_b: dict[str, int]) -> dict[int, int]:
    """row index in a -> row index in b, for every character both codecs share."""
    return {ia: codec_b[ch] for ch, ia in codec_a.items() if ch in codec_b}


def gather_rows(w_b: np.ndarray, row_map: dict[int, int], n_rows_a: int,
                fill: np.ndarray | None = None) -> np.ndarray:
    """Build an array in a's row order from b's rows; rows a has and b lacks take ``fill`` (or zeros)."""
    out = np.zeros((n_rows_a,) + w_b.shape[1:], dtype=w_b.dtype) if fill is None else fill.copy()
    for ia, ib in row_map.items():
        out[ia] = w_b[ib]
    return out


def common_names(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    """Tensor names present in both with equal shapes — the only ones averaging may touch."""
    return [n for n in a if n in b and a[n].shape == b[n].shape]
