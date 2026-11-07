"""blb-eval-hand: the handwriting part of blb-eval, chosen by data properties only (docs/PHASE5_REPORT.md).

    .venv/bin/python scripts/build_blb_eval_hand.py

A subset of data/heldout/blb-eval.jsonl: same keys, same crops, same image hashes, so every reading
of blb-eval can be re-aggregated on it exactly. No rule looks at any model's output: choosing test
lines by how a model reads them would bias every number measured on them.

Object and page rules (decided by looking at the page images, 07.11.2026):
  printed_book    K2913,17   Die Regesten des Stiftes Kreuzlingen (1853), printed; the reference is the
                             BLB's OCR of the print, not a hand correction
  typescript      K3123,57   typewritten essay (1955)
  printed_form    K2911,129  printed government form (Colmar, an XI) with handwritten entries
  music           K2912,66   song in music notation, words under the staves
  envelope_page   K2915-3_0010  outside of a folded letter, addresses in several orientations
Line rules (geometry and reference only):
  vertical        crop taller than wide
  thin_box        crop under 8 px high
  tall_box        box higher than 2.5 x the page's median box height (lines of 8 px and more): two lines
  short_ref       normalised reference of 1-3 characters
"""
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from wsp.data import page_lines, sha256_file
from wsp.metrics import normalise

D = "/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt"
SRC = Path("data/heldout/blb-eval.jsonl")
OUT = Path("data/heldout/blb-eval-hand.jsonl")
OBJECTS = {"K2913,17": "printed_book", "K3123,57": "typescript", "K2911,129": "printed_form", "K2912,66": "music"}
PAGES = {"manuskripte_eval/K2915-3_0010": "envelope_page"}


def main():
    items = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    boxes, med = {}, {}
    for page in sorted({tuple(i["key"].split("/")[:2]) for i in items}):
        proj, stem = page
        ls = page_lines(open(f"{D}/{proj}/page/{stem}.xml", encoding="utf-8").read())
        hs = {}
        for l in ls:
            ys = [p[1] for p in l["boundary"]]
            hs[l["id"]] = max(ys) - min(ys)
        boxes[page] = hs
        big = [h for h in hs.values() if h >= 8]
        med[page] = st.median(big) if big else None
    kept, why = [], {}
    for i in items:
        proj, stem, lid = i["key"].split("/")
        page = (proj, stem)
        w, h = Image.open(i["image"]).size
        rule = (OBJECTS.get(i["doc"]) or PAGES.get(f"{proj}/{stem}")
                or ("vertical" if h > w else None)
                or ("thin_box" if h < 8 else None)
                or ("tall_box" if med[page] and boxes[page][lid] > 2.5 * med[page] else None)
                or ("short_ref" if len(normalise(i["text"])) <= 3 else None))
        if rule:
            why[i["key"]] = rule
        else:
            kept.append(i)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    counts = Counter(why.values())
    meta = {"set": OUT.stem, "source": str(SRC), "source_sha256": sha256_file(SRC), "n": len(kept),
            "candidates": len(items), "sha256": sha256_file(OUT),
            "documents": sorted({r["doc"] for r in kept}),
            "excluded": {"by_rule": dict(counts), "objects": OBJECTS, "pages": PAGES,
                         "lines": why},
            "selection": "blb-eval minus the object, page and line rules of scripts/build_blb_eval_hand.py; "
                         "no rule uses a model's output"}
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(OUT, meta["sha256"], f"{len(kept)} of {len(items)} lines, {len(meta['documents'])} documents")
    for r, n in counts.most_common():
        print(f"  {n:5d} excluded: {r}")


if __name__ == "__main__":
    main()
