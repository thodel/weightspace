"""Scoring that does not depend on any engine.

Corpus CER/WER the way ``ketos test`` and the trainer count (NFD, whitespace
flattened), collapse detection, confusions, and a paired per-line bootstrap.
"""
from __future__ import annotations

import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

BOOTSTRAP_SEED = 20261106


def normalise(text: str) -> str:
    """NFD, whitespace flattened to single spaces, stripped."""
    return " ".join(unicodedata.normalize("NFD", text).split())


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance over any sequences (characters or words)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align(a: str, b: str) -> list[tuple[str, str]]:
    """Character alignment of reference ``a`` and hypothesis ``b``.

    Returns pairs (ref_char, hyp_char) with '' for insertions/deletions.
    Standard backtrace of the edit-distance matrix; fine for line length.
    """
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((a[i - 1], "")); i -= 1
        else:
            pairs.append(("", b[j - 1])); j -= 1
    pairs.reverse()
    return pairs


def trigrams(s: str, n: int = 3) -> Counter:
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def degenerate_repeat(hyp: str, ref: str, ngram: int = 3, repeats: int = 6) -> bool:
    """A repetition loop: the hypothesis's most frequent n-gram occurs ``repeats`` times or
    more **and** at least twice as often as in the reference, plus two. Correct long lines
    repeat " . ", " , " or "en " six times and more; only the excess over the reference
    marks a loop (docs/PHASE4_REPORT.md §1)."""
    g = trigrams(hyp, ngram)
    if not g:
        return False
    top, n = g.most_common(1)[0]
    return n >= repeats and n >= 2 * trigrams(ref, ngram).get(top, 0) + 2


def is_collapsed(hyp: str, ref: str, ratio: tuple[float, float] = (0.5, 2.0),
                 ngram: int = 3, repeats: int = 6, legacy: bool = False) -> bool:
    """A degenerate line: length far off the reference, or a repetition loop.

    Both shapes were seen in the pipeline: CTC outputs bounded by the image
    width that still emit one character over and over, and VLM loops that
    run to the token ceiling.

    Since 07.11.2026 the loop test is ``degenerate_repeat`` (PROTOCOL.md §5). ``legacy=True``
    gives the test every count before that date used: any n-gram occurring ``repeats``
    times, which also fires on correct long lines (142 of the 428 lines counted as
    collapsed for xix-blb-v1 on blb-eval).
    """
    h, r = normalise(hyp), normalise(ref)
    if r:
        q = len(h) / len(r)
        if q < ratio[0] or q > ratio[1]:
            return True
    elif h:
        return len(h) > 20
    if legacy:
        if len(h) >= ngram * repeats:
            grams = trigrams(h, ngram)
            if grams and grams.most_common(1)[0][1] >= repeats:
                return True
        return False
    return degenerate_repeat(h, r, ngram, repeats)


@dataclass
class Metrics:
    lines: int
    cer: float
    wer: float
    collapsed: int
    length_ratio: float
    char_edits: int
    ref_chars: int
    confusions: list[tuple[str, str, int]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["confusions"] = [list(c) for c in self.confusions]
        return d


def score(hyps: Sequence[str], refs: Sequence[str], top_confusions: int = 10,
          extra: dict | None = None) -> Metrics:
    """Corpus-level metrics over paired lines. An empty hypothesis counts as empty."""
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses for {len(refs)} references")
    char_edits = ref_chars = word_edits = ref_words = hyp_chars = 0
    collapsed = 0
    conf: Counter = Counter()
    for h, r in zip(hyps, refs):
        h, r = normalise(h), normalise(r)
        char_edits += levenshtein(r, h)
        ref_chars += len(r)
        hyp_chars += len(h)
        word_edits += levenshtein(r.split(), h.split())
        ref_words += len(r.split())
        if is_collapsed(h, r):
            collapsed += 1
        for a, b in align(r, h):
            if a != b:
                conf[(a, b)] += 1
    return Metrics(
        lines=len(refs),
        cer=char_edits / ref_chars if ref_chars else float("nan"),
        wer=word_edits / ref_words if ref_words else float("nan"),
        collapsed=collapsed,
        length_ratio=hyp_chars / ref_chars if ref_chars else float("nan"),
        char_edits=char_edits,
        ref_chars=ref_chars,
        confusions=[(a, b, n) for (a, b), n in conf.most_common(top_confusions)],
        extra=dict(extra or {}),
    )


def per_line_edits(hyps: Sequence[str], refs: Sequence[str]) -> list[tuple[int, int]]:
    """(edits, ref_chars) per line, the unit the bootstrap resamples."""
    return [(levenshtein(normalise(r), normalise(h)), len(normalise(r))) for h, r in zip(hyps, refs)]


def page_of(key: str) -> str:
    """The page (the cluster of a line): ``project/page`` of a key ``project/page/line``; a key without a page is its own cluster."""
    parts = key.split("/")
    return "/".join(parts[:2]) if len(parts) >= 3 else key


def bootstrap_delta(before: Sequence[tuple[int, int]], after: Sequence[tuple[int, int]],
                    resamples: int = 2000, seed: int = BOOTSTRAP_SEED, clusters: Sequence | None = None) -> dict:
    """Paired per-line bootstrap of corpus CER(after) − CER(before).

    Lines are resampled together, so the interval is about the *same* lines
    read by two models. Returns the point estimate and a 95 % interval (``ci95``, line level).

    With ``clusters`` (one id per line, e.g. the page or object the line is cut from) the result also
    carries ``ci95_cluster``: a two-stage cluster bootstrap that resamples the clusters with replacement
    and then the lines within each drawn cluster (PROTOCOL §5, brief item 4). Lines of one page are
    correlated, so the cluster interval is the honest one; ``ci95`` is kept unchanged so that every
    decision of the keep rule and every earlier number stays reproducible.
    """
    if len(before) != len(after):
        raise ValueError("paired bootstrap needs the same lines on both sides")
    n = len(before)
    rng = random.Random(seed)

    def cer(rows: Iterable[tuple[int, int]]) -> float:
        e = c = 0
        for ed, rc in rows:
            e += ed; c += rc
        return e / c if c else float("nan")

    point = cer(after) - cer(before)
    deltas = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(cer(after[i] for i in idx) - cer(before[i] for i in idx))
    deltas.sort()
    lo = deltas[int(0.025 * resamples)]
    hi = deltas[min(resamples - 1, int(0.975 * resamples))]
    out = {"delta_cer": point, "ci95": [lo, hi], "resamples": resamples, "seed": seed, "lines": n}
    if clusters is not None:
        if len(clusters) != n:
            raise ValueError("one cluster id per line")
        import numpy as np
        groups: dict = {}
        for i, c in enumerate(clusters):
            groups.setdefault(c, []).append(i)
        arrs = [np.asarray(v) for v in groups.values()]
        b = np.asarray(before, dtype=np.float64); a = np.asarray(after, dtype=np.float64)
        g = np.random.default_rng(seed)
        ds = []
        for _ in range(resamples):
            e0 = e1 = c0 = c1 = 0.0
            for k in g.integers(0, len(arrs), len(arrs)):
                idx = arrs[k]
                j = idx[g.integers(0, len(idx), len(idx))]
                e0 += b[j, 0].sum(); c0 += b[j, 1].sum(); e1 += a[j, 0].sum(); c1 += a[j, 1].sum()
            ds.append(e1 / c1 - e0 / c0)
        ds.sort()
        out.update({"ci95_cluster": [ds[int(0.025 * resamples)], ds[min(resamples - 1, int(0.975 * resamples))]],
                    "clusters": len(arrs)})
    return out
