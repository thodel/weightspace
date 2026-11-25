"""Stage 5 helpers (docs/TROCR_STAGE5_PREREG.md): averaging of same-recipe siblings by canonical tensor name, interpolation toward an average,
the class of a comparator against arm I, and the selection rule for the confirmation read. Pure numpy; no model is loaded here.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

MIN_EFFECT = 0.005          # 0.5 CER points, as the keep rule of stage 4 and R16's noise bound


def check_compatible(tensor_sets: Sequence[dict[str, np.ndarray]], vocabs: Sequence[dict] | None = None, labels: Sequence[str] | None = None) -> None:
    """Every set has the same tensor names and shapes; with ``vocabs`` also the same tokenizer vocabulary. Raises ValueError naming the first mismatch."""
    labels = list(labels) if labels else [f"model{i}" for i in range(len(tensor_sets))]
    ref = tensor_sets[0]
    for lab, ts in zip(labels[1:], tensor_sets[1:]):
        if set(ts) != set(ref):
            diff = sorted(set(ts) ^ set(ref))
            raise ValueError(f"{lab}: tensor names differ from {labels[0]}: {diff[:4]} ({len(diff)} in all)")
        for n in ref:
            if ts[n].shape != ref[n].shape:
                raise ValueError(f"{lab}: {n} has shape {ts[n].shape}, {labels[0]} has {ref[n].shape}")
    if vocabs is not None:
        for lab, v in zip(labels[1:], vocabs[1:]):
            if v != vocabs[0]:
                raise ValueError(f"{lab}: tokenizer vocabulary differs from {labels[0]}")


def average(tensor_sets: Sequence[dict[str, np.ndarray]], weights: Sequence[float] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Weighted average by name (float64 accumulation, float32 result). Weights default to uniform, must be positive and sum to 1 (to 1e-9).
    Returns (tensors, report) with the weights used, their sum and the number of tensors; a non-finite result raises."""
    n = len(tensor_sets)
    if n == 0:
        raise ValueError("nothing to average")
    w = [1.0 / n] * n if weights is None else [float(x) for x in weights]
    if len(w) != n or any(x <= 0 for x in w) or abs(sum(w) - 1.0) > 1e-9:
        raise ValueError(f"weights {w} are not {n} positive numbers summing to 1")
    check_compatible(tensor_sets)
    out = {}
    for name in tensor_sets[0]:
        acc = np.zeros(tensor_sets[0][name].shape, dtype=np.float64)
        for wi, ts in zip(w, tensor_sets):
            acc += wi * ts[name].astype(np.float64)
        if not np.isfinite(acc).all():
            raise ValueError(f"{name}: non-finite after averaging")
        out[name] = acc.astype(np.float32)
    return out, {"n": n, "weights": w, "weight_sum": sum(w), "tensors": len(out)}


def toward(subject: dict[str, np.ndarray], other: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    """``subject + alpha * (other - subject)`` on every tensor of ``subject``; ``other`` must carry the same names and shapes (alpha 1 gives ``other``)."""
    check_compatible([subject, other])
    return {n: (subject[n].astype(np.float64) + alpha * (other[n].astype(np.float64) - subject[n].astype(np.float64))).astype(np.float32) for n in subject}


def seed_mean_edits(edit_lists: Sequence[Sequence[Sequence[int]]]) -> list[tuple[float, int]]:
    """Per line the mean edit count over seeds (reference length is the same in every list); the arm-level reading of a comparator trained with several seeds."""
    k = len(edit_lists)
    n = len(edit_lists[0])
    if any(len(e) != n for e in edit_lists):
        raise ValueError("the seeds read different numbers of lines")
    out = []
    for i in range(n):
        lens = {e[i][1] for e in edit_lists}
        if len(lens) != 1:
            raise ValueError(f"line {i}: reference lengths differ between seeds")
        out.append((sum(e[i][0] for e in edit_lists) / k, lens.pop()))
    return out


def classify(own_delta: float, own_ci: Sequence[float], diff_delta: float, diff_ci: Sequence[float], min_effect: float = MIN_EFFECT) -> str:
    """Class of a comparator against arm I. ``own_*``: CER(comparator) - CER(subject) and its page-clustered 95 % interval; ``diff_*``: CER(comparator) - CER(arm I),
    paired on the same lines. Order of the tests:
      harms       own interval entirely above +min_effect;
      beats       diff interval entirely below 0;
      below       diff interval entirely above 0 (arm I is better; includes smaller gains and no gain);
      matches     diff interval contains 0, own delta below -min_effect and own interval entirely below 0;
      unresolved  diff interval contains 0 and the comparator is not itself a gain."""
    for x in (own_delta, diff_delta, *own_ci, *diff_ci):
        if not math.isfinite(x):
            raise ValueError("non-finite delta or interval")
    if own_ci[0] > min_effect:
        return "harms"
    if diff_ci[1] < 0:
        return "beats"
    if diff_ci[0] > 0:
        return "below"
    if own_delta < -min_effect and own_ci[1] < 0:
        return "matches"
    return "unresolved"


def select_other(candidates: Sequence[dict], min_effect: float = MIN_EFFECT) -> dict | None:
    """The one model besides arm I that is read on the confirmation half (one call per subject). ``candidates``: dicts with ``name``, ``delta_cer`` and ``ci95_cluster``
    (discovery half, against the subject). Eligible: the interval lies entirely below ``-min_effect`` (a gain by the keep rule). Chosen: the most negative delta;
    ties by name. None if no candidate is eligible (then only arm I is read)."""
    ok = [c for c in candidates if c["ci95_cluster"][1] < -min_effect]
    if not ok:
        return None
    return sorted(ok, key=lambda c: (c["delta_cer"], c["name"]))[0]
