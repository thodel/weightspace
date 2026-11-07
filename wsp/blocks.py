"""Block statistics: where on the character inventory the errors fall.

The idea is CERberus's (W. Haverals, https://github.com/WHaverals/CERberus, MIT):
align reference and hypothesis character by character and report, per Unicode
block, how many reference characters there are and how many were read wrong.
Three things are done differently, because this harness compares models by
paired differences and needs the per-block numbers to add up:

1. **A partition, not a selection.** CERberus counts a character only if it falls
   in one of the ranges the user ticked (``a-z``, ``A-Z``, digits, part of ASCII
   punctuation, a MUFI list); everything else (spaces, ``ä`` in NFC, combining
   marks, ``ſ``, ``ß``) silently drops out. Here every code point lands in exactly
   one block: ASCII is split the CERberus way (space, lowercase, uppercase,
   digits, punctuation/symbols), everything else by its Unicode block, with a
   fallback bucket so nothing is lost.
2. **Every edit is charged once.** Substitutions and deletions are charged to the
   reference character's block (as in CERberus); insertions, which CERberus does
   not count at all, are charged to the inserted character's block. So
   ``sum(errors over blocks) == Levenshtein distance`` and the blocks' error shares
   ``errors_b / ref_chars`` add up to the corpus CER exactly.
3. **Per line, for the bootstrap.** CERberus aligns one long text; here every line
   is aligned on its own (the unit the paired bootstrap resamples), so a block's
   change between two models gets an interval like the corpus CER does.

The text is scored after ``wsp.metrics.normalise`` (NFD, whitespace flattened),
the same string the CER is computed on. Under NFD ``ä`` is ``a`` + U+0308, so
umlauts and the superscript-e of older German show up in *Combining Diacritical
Marks*, the block that tells "letter read, mark lost" apart from "letter wrong".

The MUFI overlay reproduces CERberus's MUFI list as an extra, overlapping group
(reported separately; it is not part of the partition).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from wsp.metrics import BOOTSTRAP_SEED, align, normalise

# Unicode blocks (Unicode 15), the ones that occur in HTR ground truth of Latin-script
# material plus their neighbours. Anything not listed falls into "Other (U+xx80 range)".
_UNICODE_BLOCKS: list[tuple[int, int, str]] = [
    (0x0080, 0x00FF, "Latin-1 Supplement"),
    (0x0100, 0x017F, "Latin Extended-A"),
    (0x0180, 0x024F, "Latin Extended-B"),
    (0x0250, 0x02AF, "IPA Extensions"),
    (0x02B0, 0x02FF, "Spacing Modifier Letters"),
    (0x0300, 0x036F, "Combining Diacritical Marks"),
    (0x0370, 0x03FF, "Greek and Coptic"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x1AB0, 0x1AFF, "Combining Diacritical Marks Extended"),
    (0x1D00, 0x1D7F, "Phonetic Extensions"),
    (0x1D80, 0x1DBF, "Phonetic Extensions Supplement"),
    (0x1DC0, 0x1DFF, "Combining Diacritical Marks Supplement"),
    (0x1E00, 0x1EFF, "Latin Extended Additional"),
    (0x1F00, 0x1FFF, "Greek Extended"),
    (0x2000, 0x206F, "General Punctuation"),
    (0x2070, 0x209F, "Superscripts and Subscripts"),
    (0x20A0, 0x20CF, "Currency Symbols"),
    (0x20D0, 0x20FF, "Combining Diacritical Marks for Symbols"),
    (0x2100, 0x214F, "Letterlike Symbols"),
    (0x2150, 0x218F, "Number Forms"),
    (0x2190, 0x21FF, "Arrows"),
    (0x2200, 0x22FF, "Mathematical Operators"),
    (0x2300, 0x23FF, "Miscellaneous Technical"),
    (0x25A0, 0x25FF, "Geometric Shapes"),
    (0x2600, 0x26FF, "Miscellaneous Symbols"),
    (0x2700, 0x27BF, "Dingbats"),
    (0x2C60, 0x2C7F, "Latin Extended-C"),
    (0x2E00, 0x2E7F, "Supplemental Punctuation"),
    (0xA720, 0xA7FF, "Latin Extended-D"),
    (0xAB30, 0xAB6F, "Latin Extended-E"),
    (0xE000, 0xF8FF, "Private Use Area"),
    (0xFB00, 0xFB4F, "Alphabetic Presentation Forms"),
    (0xFE20, 0xFE2F, "Combining Half Marks"),
    (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms"),
]

SPACE, LOWER, UPPER, DIGITS, ASCII_PUNCT = ("Space", "Latin lowercase a-z", "Latin uppercase A-Z",
                                            "Digits 0-9", "ASCII punctuation and symbols")

#: CERberus's MUFI list (app.py, default_unicode_ranges['MUFI Glyphs']), kept verbatim as an overlay.
MUFI_CERBERUS: list[tuple[int, int]] = [
    (0xf0, 0xf0), (0xf8, 0xf8), (0x100, 0x101), (0x112, 0x113), (0x127, 0x127), (0x12b, 0x12b), (0x14c, 0x14d),
    (0x16b, 0x16b), (0x1b6, 0x1b6), (0x223, 0x223), (0x233, 0x233), (0x2b3, 0x2b3), (0x2bc, 0x2bc), (0x2e2, 0x2e2),
    (0x304, 0x304), (0x3c9, 0x3c9), (0x1d43, 0x1d43), (0x1d48, 0x1d49), (0x1d50, 0x1d50), (0x1d52, 0x1d52),
    (0x1d57, 0x1d58), (0x1d5b, 0x1d5b), (0x1d9c, 0x1d9c), (0x1da6, 0x1da6), (0x1dd1, 0x1dd1), (0x1de3, 0x1de3),
    (0x1e9c, 0x1e9c), (0x2071, 0x2071), (0x207f, 0x207f), (0x2184, 0x2184), (0xa750, 0xa753), (0xa759, 0xa759),
    (0xa75c, 0xa75f), (0xa76b, 0xa76b), (0xa76e, 0xa770), (0xa78f, 0xa78f), (0xe154, 0xe154), (0xe1dc, 0xe1dc),
    (0xe554, 0xe554), (0xe5b8, 0xe5b8), (0xe5dc, 0xe5dc), (0xe665, 0xe665), (0xe681, 0xe681), (0xe74d, 0xe74d),
    (0xe8e5, 0xe8e5), (0xeed7, 0xeed7)]
MUFI = "MUFI (CERberus list, overlay)"

_cache: dict[str, str] = {}


def block_of(ch: str) -> str:
    """The block a single code point belongs to; total over all of Unicode."""
    b = _cache.get(ch)
    if b is not None:
        return b
    cp = ord(ch)
    if cp < 0x80:
        if ch.isspace():
            b = SPACE
        elif "a" <= ch <= "z":
            b = LOWER
        elif "A" <= ch <= "Z":
            b = UPPER
        elif "0" <= ch <= "9":
            b = DIGITS
        else:
            b = ASCII_PUNCT
    else:
        b = next((name for lo, hi, name in _UNICODE_BLOCKS if lo <= cp <= hi), None)
        if b is None:
            start = cp & ~0x7F
            b = f"Other (U+{start:04X}-U+{start + 0x7F:04X})"
    _cache[ch] = b
    return b


def in_mufi(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in MUFI_CERBERUS)


def _zero() -> dict:
    return {"ref": 0, "correct": 0, "sub": 0, "del": 0, "ins": 0}


def line_counts(ref: str, hyp: str, normalised: bool = False) -> dict[str, dict]:
    """Per block: reference characters, correct, substituted, deleted, and insertions charged to it."""
    if not normalised:
        ref, hyp = normalise(ref), normalise(hyp)
    out: dict[str, dict] = defaultdict(_zero)
    for a, b in align(ref, hyp):
        if a:
            groups = [block_of(a)] + ([MUFI] if in_mufi(a) else [])
            for g in groups:
                out[g]["ref"] += 1
                if a == b:
                    out[g]["correct"] += 1
                elif b:
                    out[g]["sub"] += 1
                else:
                    out[g]["del"] += 1
        else:
            for g in [block_of(b)] + ([MUFI] if in_mufi(b) else []):
                out[g]["ins"] += 1
    return dict(out)


def errors(c: dict) -> int:
    return c["sub"] + c["del"] + c["ins"]


def aggregate(per_line: Iterable[dict[str, dict]]) -> dict[str, dict]:
    tot: dict[str, dict] = defaultdict(_zero)
    for line in per_line:
        for g, c in line.items():
            for k, v in c.items():
                tot[g][k] += v
    return dict(tot)


def table(per_line: Sequence[dict[str, dict]]) -> dict:
    """Corpus-level block table: for each block its counts, its own CER (errors / ref chars of the
    block) and its share of the corpus CER (errors / all ref chars); the shares of the partition
    add up to the corpus CER."""
    tot = aggregate(per_line)
    ref_all = sum(c["ref"] for g, c in tot.items() if g != MUFI)
    edits_all = sum(errors(c) for g, c in tot.items() if g != MUFI)
    rows = {}
    for g, c in sorted(tot.items(), key=lambda kv: -errors(kv[1])):
        rows[g] = {**c, "errors": errors(c),
                   "block_cer": errors(c) / c["ref"] if c["ref"] else None,
                   "share_of_cer": errors(c) / ref_all if ref_all else None,
                   "overlay": g == MUFI}
    return {"ref_chars": ref_all, "edits": edits_all, "cer": edits_all / ref_all if ref_all else None, "blocks": rows}


def per_line_counts(hyps: Sequence[str], refs: Sequence[str]) -> list[dict[str, dict]]:
    return [line_counts(r, h) for h, r in zip(hyps, refs)]


def _matrix(per_line: Sequence[dict[str, dict]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    E = np.zeros((len(per_line), len(names))); N = np.zeros((len(per_line), len(names)))
    idx = {g: i for i, g in enumerate(names)}
    for r, line in enumerate(per_line):
        for g, c in line.items():
            if g in idx:
                E[r, idx[g]] = errors(c); N[r, idx[g]] = c["ref"]
    return E, N


def bootstrap_blocks(before: Sequence[dict[str, dict]], after: Sequence[dict[str, dict]],
                     resamples: int = 2000, seed: int = BOOTSTRAP_SEED) -> dict[str, dict]:
    """Paired per-line bootstrap, per block, of two quantities:

    * ``delta_share``: change of the block's share of the corpus CER (errors_b / all ref chars),
      in CER units, so the blocks' deltas add up to the corpus CER delta;
    * ``delta_block_cer``: change of the block's own error rate (errors_b / ref chars of b).

    The same resampled lines serve every block (one index draw per resample), as in
    ``wsp.metrics.bootstrap_delta``; with the same seed the corpus interval is reproduced."""
    if len(before) != len(after):
        raise ValueError("paired bootstrap needs the same lines on both sides")
    names = sorted({g for line in list(before) + list(after) for g in line})
    Eb, Nb = _matrix(before, names); Ea, _ = _matrix(after, names)
    n = len(before)
    all_ref = Nb[:, [i for i, g in enumerate(names) if g != MUFI]].sum(1)
    rng = np.random.default_rng(seed)

    def stats(rows):
        eb, ea, nb, ar = Eb[rows].sum(0), Ea[rows].sum(0), Nb[rows].sum(0), all_ref[rows].sum()
        share = (ea - eb) / ar if ar else np.full(len(names), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            own = np.where(nb > 0, (ea - eb) / nb, np.nan)
        return share, own

    point_share, point_own = stats(np.arange(n))
    S = np.empty((resamples, len(names))); O = np.empty((resamples, len(names)))
    for k in range(resamples):
        rows = rng.integers(0, n, n)
        S[k], O[k] = stats(rows)
    out = {}
    for i, g in enumerate(names):
        s = np.sort(S[:, i]); o = O[:, i][~np.isnan(O[:, i])]; o.sort()
        lo, hi = int(0.025 * resamples), min(resamples - 1, int(0.975 * resamples))
        out[g] = {"delta_share": float(point_share[i]), "ci95_share": [float(s[lo]), float(s[hi])],
                  "delta_block_cer": None if np.isnan(point_own[i]) else float(point_own[i]),
                  "ci95_block_cer": ([float(o[int(0.025 * len(o))]), float(o[min(len(o) - 1, int(0.975 * len(o)))])]
                                     if len(o) else None),
                  "errors_before": int(Eb[:, i].sum()), "errors_after": int(Ea[:, i].sum()),
                  "ref": int(Nb[:, i].sum()), "excludes_0": bool(s[lo] > 0 or s[hi] < 0),
                  "overlay": g == MUFI}
    return out
