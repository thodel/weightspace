"""Stage-4 comparators of the TrOCR track (docs/TROCR_STAGE4_PLAN.md sections 4 to 7); pure functions on arrays and line lists, no model.

* sham generator (arm S, amendment A5): for a kept edit, ``k`` random edits of the same tensors with the same Frobenius norm of the change.
* margin states (arm M): a kept edit at other sizes (half, double) along its own direction.
* interpolation line search (arm I): one parameter, the global interpolation toward the donor, on a fixed grid; the best grid point is the comparator.
* object split: the held-out objects (documents) split into a discovery half and a confirmation half by seed, balanced by line count.
Nothing here reads a held-out model output; the split uses the document ids and line counts of the held-out list only.
"""
from __future__ import annotations

import random
import zlib
from typing import Callable, Iterable, Sequence

import numpy as np

SHAM_COUNT = 5
MARGIN_FACTORS = (0.5, 2.0)
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)          # with the baseline pass this is the budget of 10 passes


def edit_delta(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The change of every tensor that differs between two states."""
    return {n: (after[n].astype(np.float64) - before[n].astype(np.float64)) for n in before if not np.array_equal(before[n], after[n])}


def _rng(seed: int, i: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, i, zlib.crc32(name.encode())])


def sham_deltas(deltas: dict[str, np.ndarray], seed: int, k: int = SHAM_COUNT) -> list[dict[str, np.ndarray]]:
    """``k`` shams of an edit: for each changed tensor an i.i.d. Gaussian change of the same shape, rescaled to the same Frobenius norm as the real change."""
    out = []
    for i in range(k):
        d = {}
        for n, real in deltas.items():
            r = _rng(seed, i, n).standard_normal(real.shape)
            nr = np.linalg.norm(r)
            d[n] = r * (np.linalg.norm(real) / nr) if nr > 0 else r
        out.append(d)
    return out


def apply_deltas(base: dict[str, np.ndarray], deltas: dict[str, np.ndarray], scale: float = 1.0) -> dict[str, np.ndarray]:
    """``base + scale * delta`` for the tensors of ``deltas`` (float32, as the model holds them)."""
    return {n: (base[n].astype(np.float64) + scale * d).astype(base[n].dtype) for n, d in deltas.items()}


def margin_states(before: dict[str, np.ndarray], after: dict[str, np.ndarray], factors: Sequence[float] = MARGIN_FACTORS) -> dict[float, dict[str, np.ndarray]]:
    """The kept edit at ``factor`` times its size, for each factor (1.0 is the edit itself)."""
    d = edit_delta(before, after)
    return {f: apply_deltas(before, d, f) for f in factors}


def interpolate_all(subject: dict[str, np.ndarray], donor: dict[str, np.ndarray], alpha: float, names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """(1 - alpha) * subject + alpha * donor for the named tensors (all shared names by default)."""
    names = list(names) if names is not None else [n for n in subject if n in donor and subject[n].shape == donor[n].shape]
    return {n: ((1 - alpha) * subject[n].astype(np.float64) + alpha * donor[n].astype(np.float64)).astype(subject[n].dtype) for n in names}


def line_search(score: Callable[[float], float], baseline_cer: float, alphas: Sequence[float] = ALPHAS) -> dict:
    """Arm I: the CER of each grid point (``score(alpha)`` is one counted pass on the practice set); the best point is the one with the lowest CER, the
    smaller alpha on a tie; ``alpha`` is None when no point beats the baseline. The harness's keep rule is applied to the best point by the caller."""
    rows = [{"alpha": a, "cer": float(score(a))} for a in alphas]
    best = min(rows, key=lambda r: (r["cer"], r["alpha"]))
    return {"baseline_cer": baseline_cer, "grid": rows, "best": best, "alpha": best["alpha"] if best["cer"] < baseline_cer else None, "passes": 1 + len(rows)}


def split_objects(rows: Sequence[dict], seed: int, halves: tuple[str, str] = ("discovery", "confirmation")) -> dict:
    """Split the held-out objects (``row["doc"]``) into two document-disjoint halves, fixed by ``seed``: objects in seeded random order, each to the half
    with fewer lines so far (ties to the first). Reads document ids and line counts only."""
    count: dict[str, int] = {}
    for r in rows:
        count[r["doc"]] = count.get(r["doc"], 0) + 1
    docs = sorted(count)
    random.Random(seed).shuffle(docs)
    parts: dict[str, list[str]] = {h: [] for h in halves}
    lines = {h: 0 for h in halves}
    for d in docs:
        h = halves[0] if lines[halves[0]] <= lines[halves[1]] else halves[1]
        parts[h].append(d)
        lines[h] += count[d]
    return {"seed": seed, "objects": {h: sorted(v) for h, v in parts.items()}, "lines": lines, "n_objects": {h: len(v) for h, v in parts.items()}}


def half_of(doc: str, split: dict) -> str | None:
    for h, docs in split["objects"].items():
        if doc in docs:
            return h
    return None
