"""Page scoring for the VLM subjects (phase 5).

The line metrics of ``wsp.metrics`` with what a page model adds: pages that
ran to the generation cap, ``truncated_cer`` (the hypothesis cut to the
reference length, as the trainer's ``textmetrics`` defines it: reading without
stopping), ``length_ratio``, and a page-scale loop test. Text is normalised the
same way (NFD, whitespace flattened). Pages are the bootstrap unit.

The pure-Python ``metrics.levenshtein`` is quadratic in interpreted code; a
5,000-character page takes tens of seconds. ``levenshtein`` here is the same
recurrence with each row done in numpy: the insertion chain
``cur[j] = min(cur[j], cur[j-1] + 1)`` is ``j + cummin(cur[k] - k)``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from wsp.metrics import normalise, bootstrap_delta


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    ca = np.frombuffer(a.encode("utf-32-le"), dtype=np.uint32)
    cb = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    ar = np.arange(len(cb) + 1, dtype=np.int64)
    prev = ar.copy()
    for i, x in enumerate(ca, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (cb != x))
        cur = np.minimum.accumulate(cur - ar) + ar
        prev = cur
    return int(prev[-1])


def page_loop(hyp: str, ref: str, n: int = 24, extra: int = 3) -> bool:
    """A repetition loop on a page: some n-character stretch occurs ``extra`` or
    more times more often in the hypothesis than in the reference. Ordinary
    repeated words and formulae occur in the reference too and cancel out."""
    if len(hyp) < n * (extra + 1):
        return False
    grams = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
    gram, count = grams.most_common(1)[0]
    return count - ref.count(gram) >= extra


def page_edits(hyps: Sequence[str], refs: Sequence[str]) -> list[tuple[int, int]]:
    """(edits, reference characters) per page, normalised."""
    out = []
    for h, r in zip(hyps, refs):
        h, r = normalise(h), normalise(r)
        out.append((levenshtein(r, h), len(r)))
    return out


def score(hyps: Sequence[str], refs: Sequence[str], at_cap: Sequence[bool]) -> dict:
    nh = [normalise(h) for h in hyps]
    nr = [normalise(r) for r in refs]
    edits = [(levenshtein(r, h), len(r)) for h, r in zip(nh, nr)]
    trunc = [levenshtein(r, h[:len(r)]) for h, r in zip(nh, nr)]
    chars = sum(c for _, c in edits)
    return {"pages": len(refs), "chars": chars,
            "cer": sum(e for e, _ in edits) / chars,
            "truncated_cer": sum(trunc) / chars,
            "length_ratio": sum(len(h) for h in nh) / chars,
            "at_cap": int(sum(bool(x) for x in at_cap)),
            "loops": int(sum(page_loop(h, r) for h, r in zip(nh, nr))),
            "per_page": edits}


def read_vllm(path: Path) -> dict[str, dict]:
    """``wsp.vllm_forward`` readings by page key: text, at_cap, tokens."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = {"text": d["text"], "at_cap": d["finish_reason"] == "length", "tokens": d["tokens"]}
    return out


def read_reference(path: Path) -> dict[str, dict]:
    """``evaluate_qlora``'s ``.raw.jsonl`` by page key (the image's stem)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[Path(d["image"]).stem] = {"text": d["raw"], "at_cap": bool(d["at_cap"])}
    return out


def agreement(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Gate G1: two engines on the same pages. CER of b against a as reference,
    exact matches, and whether the same pages ran to the cap."""
    keys = sorted(set(a) & set(b))
    e = [(levenshtein(normalise(a[k]["text"]), normalise(b[k]["text"])), len(normalise(a[k]["text"])))
         for k in keys]
    cap_a = {k for k in keys if a[k]["at_cap"]}
    cap_b = {k for k in keys if b[k]["at_cap"]}
    return {"pages": len(keys), "exact": sum(normalise(a[k]["text"]) == normalise(b[k]["text"]) for k in keys),
            "cer_between": sum(x for x, _ in e) / max(1, sum(c for _, c in e)),
            "worst": sorted(((round(x / max(c, 1), 4), k) for (x, c), k in zip(e, keys)), reverse=True)[:5],
            "at_cap_a": sorted(cap_a), "at_cap_b": sorted(cap_b), "at_cap_same": cap_a == cap_b}


def compare(before: dict, after: dict, resamples: int = 2000) -> dict:
    """Paired page bootstrap of CER(after) − CER(before) on ``score`` outputs."""
    return bootstrap_delta(before["per_page"], after["per_page"], resamples=resamples)
