"""Collapsed lines: which are model failures and which are data defects.

    python -m wsp.collapse --set data/heldout/blb-eval.jsonl \
        --readings docs/baselines/readings/blb-eval/kraken-german-xix-blb-v1.jsonl \
        [--siblings <readings.jsonl> ...] --out docs/collapse/blb-eval__xix-blb-v1.json

Until 07.11.2026 ``wsp.metrics.is_collapsed`` flagged a line when its hypothesis is less than half or more
than twice the reference length, *or* when one character trigram occurs six times or
more. The second test fires on correct long lines: normalised punctuation (" . ", " , ")
and common endings ("en ") repeat that often in ordinary text. ``degenerate_repeat``
is the stricter test, now the default of ``is_collapsed``: the most frequent trigram must also occur at least
twice as often as in the reference, plus two.

Every flagged line is put in exactly one class, in this order:

- ``detector_false_positive``: length ratio within [0.5, 2] and no degenerate repeat;
- ``vertical``: the crop is taller than wide (envelope addresses, marginal notes);
- ``degenerate_box``: the crop is under 8 px high;
- ``tiny_reference``: the reference has 1 to 3 characters;
- ``box_holds_more``: the hypothesis is more than twice the reference;
- ``repetition_loop``: a degenerate repeat within normal length;
- ``model_specific``: too short, and a sibling model reads the line at normal length
  with less than half the reference length in edits;
- ``short_all_models``: too short for every model given (crop/text mismatch, illegible,
  struck through, or a hand no model reads). Without siblings these stay here.

Only ``model_specific`` lines are weight-space targets; the rest are data or metric defects.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from wsp.metrics import degenerate_repeat, is_collapsed, levenshtein, normalise  # noqa: F401 (re-exported)

ORDER = ["detector_false_positive", "vertical", "degenerate_box", "tiny_reference", "box_holds_more",
         "repetition_loop", "model_specific", "short_all_models"]


def classify(ref: str, hyp: str, crop_w: int, crop_h: int, siblings: list[str] = ()) -> str | None:
    """None if the line is not flagged by the *legacy* ``is_collapsed`` test; else its class
    (see module doc). The legacy test is used here on purpose: this module exists to say
    which of the lines counted as collapsed before 07.11.2026 were what."""
    r, h = normalise(ref), normalise(hyp)
    if not is_collapsed(h, r, legacy=True):
        return None
    q = len(h) / max(len(r), 1)
    if 0.5 <= q <= 2 and not degenerate_repeat(h, r):
        return "detector_false_positive"
    if crop_h > crop_w:
        return "vertical"
    if crop_h < 8:
        return "degenerate_box"
    if len(r) <= 3:
        return "tiny_reference"
    if q > 2:
        return "box_holds_more"
    if q >= 0.5:
        return "repetition_loop"
    for s in siblings:
        o = normalise(s)
        if len(o) >= 0.5 * len(r) and levenshtein(r, o) < 0.5 * len(r):
            return "model_specific"
    return "short_all_models"


def profile(set_path: Path, readings: Path, siblings: list[Path] = ()) -> dict:
    items = [json.loads(l) for l in set_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    hyp = {json.loads(l)["key"]: json.loads(l)["hyp"] for l in readings.read_text(encoding="utf-8").splitlines() if l.strip()}
    sib = [{json.loads(l)["key"]: json.loads(l)["hyp"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
           for p in siblings]
    lines, edits_total, ref_total = [], 0, 0
    for it in items:
        k = it["key"]; r, h = normalise(it["text"]), normalise(hyp[k])
        e = levenshtein(r, h); edits_total += e; ref_total += len(r)
        w, ht = Image.open(it["image"]).size
        c = classify(it["text"], hyp[k], w, ht, [s[k] for s in sib if k in s])
        lines.append({"key": k, "doc": it.get("doc"), "class": c, "edits": e, "ref_chars": len(r),
                      "hyp_chars": len(h), "crop": [w, ht]})
    by = {}
    for c in ORDER:
        sel = [x for x in lines if x["class"] == c]
        if sel:
            by[c] = {"lines": len(sel), "edits": sum(x["edits"] for x in sel),
                     "share_of_edits": sum(x["edits"] for x in sel) / edits_total,
                     "documents": Counter(x["doc"] for x in sel).most_common(5)}
    flagged = [x for x in lines if x["class"]]
    # data defects leave; model failures (model_specific, repetition_loop) and false positives stay
    real = [x for x in lines if x["class"] in (None, "detector_false_positive", "model_specific", "repetition_loop")]
    return {"set": set_path.stem, "readings": str(readings), "siblings": [str(p) for p in siblings],
            "lines": len(lines), "cer": edits_total / ref_total, "flagged": len(flagged),
            "collapsed_strict": sum(1 for x in flagged if x["class"] != "detector_false_positive"),
            "cer_without_flagged_data_defects": sum(x["edits"] for x in real) / max(1, sum(x["ref_chars"] for x in real)),
            "classes": by, "per_line": [x for x in lines if x["class"]]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.collapse")
    ap.add_argument("--set", required=True); ap.add_argument("--readings", required=True)
    ap.add_argument("--siblings", nargs="*", default=[]); ap.add_argument("--out")
    a = ap.parse_args(argv)
    p = profile(Path(a.set), Path(a.readings), [Path(s) for s in a.siblings])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(p, indent=1, ensure_ascii=False))
    print(f"{p['set']}: {p['lines']} lines, CER {p['cer']:.4f}, flagged {p['flagged']}, strictly collapsed {p['collapsed_strict']}")
    for c, d in p["classes"].items():
        print(f"  {d['lines']:4d} lines {d['share_of_edits']:6.1%} of edits  {c}")


if __name__ == "__main__":
    main()
