"""Tensor-level comparison of two subjects, and sparse diffs for publication."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray], row_top: int = 10) -> dict:
    """Per-tensor L2 of the difference, relative L2, cosine, and the rows that moved most."""
    out: dict[str, dict] = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            out[name] = {"only_in": "a" if name in a else "b"}
            continue
        x, y = a[name].astype(np.float64), b[name].astype(np.float64)
        if x.shape != y.shape:
            out[name] = {"shape_a": list(x.shape), "shape_b": list(y.shape)}
            continue
        d = y - x
        l2 = float(np.linalg.norm(d))
        nx = float(np.linalg.norm(x))
        cos = float((x * y).sum() / (nx * np.linalg.norm(y))) if nx and np.linalg.norm(y) else float("nan")
        entry = {"shape": list(x.shape), "l2": l2, "rel_l2": l2 / nx if nx else float("nan"),
                 "cosine": cos, "changed": int((d != 0).sum()), "numel": int(d.size)}
        if d.ndim >= 2 and d.shape[0] > 1:
            rows = np.linalg.norm(d.reshape(d.shape[0], -1), axis=1)
            top = np.argsort(rows)[::-1][:row_top]
            entry["top_rows"] = [[int(i), float(rows[i])] for i in top if rows[i] > 0]
        out[name] = entry
    return out


def sparse_diff(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict[str, dict]:
    """Indices and new values of every changed element, per tensor (shape-equal only)."""
    diff: dict[str, dict] = {}
    for name, x in before.items():
        y = after.get(name)
        if y is None or y.shape != x.shape:
            continue
        mask = y != x
        if mask.any():
            idx = np.flatnonzero(mask)
            diff[name] = {"shape": list(x.shape), "index": idx.tolist(),
                          "value": y.reshape(-1)[idx].tolist(), "dtype": str(y.dtype)}
    return diff


def apply_sparse_diff(tensors: dict[str, np.ndarray], diff: dict[str, dict]) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in tensors.items()}
    for name, d in diff.items():
        flat = out[name].reshape(-1)
        flat[np.asarray(d["index"])] = np.asarray(d["value"], dtype=flat.dtype)
        out[name] = flat.reshape(out[name].shape)
    return out


def save_sparse_diff(diff: dict, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(diff))
    return path


def load_sparse_diff(path: Path) -> dict:
    return json.loads(Path(path).read_text())
