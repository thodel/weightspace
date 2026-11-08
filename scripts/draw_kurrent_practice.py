"""Draw the phase-5 practice set from the registered val side of german-kurrent-xvi-xix-v1.

One page per document, chosen at random (seed 20261107), plus two more from each
of the two documents that hold 160 pages each (two thirds of the side), so no
single hand dominates and every hand is present. Drawn from the references and
the document key alone: no model's output enters the draw (docs/PHASE5_PLAN.md
§7, decision 1). The test side is never read.

    python scripts/draw_kurrent_practice.py
"""
import collections
import hashlib
import json
import random
from pathlib import Path

SEED = 20261107
SIDE = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-val")
RENDER = Path.home() / "atr-cache/wsp/render/kurrent-val/images"
OUT = Path("data/practice/kurrent-val-64.jsonl")


def doc_of(stem: str) -> str:
    return stem.split("_")[1]          # <counter>_<Transkribus id or work name>_…, as make_split.py keys it


rows = [json.loads(l) for l in (SIDE / "pages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by_doc = collections.defaultdict(list)
for r in rows:
    by_doc[doc_of(Path(r["image"]).stem)].append(r)
rng = random.Random(SEED)
pick = []
for doc in sorted(by_doc):
    pages = sorted(by_doc[doc], key=lambda r: r["image"])
    n = 3 if len(pages) >= 160 else 1
    pick += [(doc, r) for r in rng.sample(pages, n)]
out = []
for doc, r in pick:
    stem = Path(r["image"]).stem
    out.append({"key": stem, "text": r["text"], "image": str(RENDER / f"{stem}.png"), "doc": doc,
                "source": r["image"], "image_sha256": hashlib.sha256((SIDE / r["image"]).read_bytes()).hexdigest()})
body = "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
OUT.write_text(body, encoding="utf-8")
lens = sorted(len(o["text"]) for o in out)
all_lens = sorted(len(r["text"]) for r in rows)
meta = {"set": OUT.stem, "source": str(SIDE), "side": "val (registered, german-kurrent-xvi-xix-v1)",
        "draw_md5_of_side": hashlib.md5((SIDE / "pages.jsonl").read_bytes()).hexdigest(),
        "n": len(out), "seed": SEED, "candidates": len(rows), "documents": sorted(by_doc),
        "rule": "one page per document, three from each 160-page document",
        "ref_chars_median": lens[len(lens) // 2], "ref_chars_median_side": all_lens[len(all_lens) // 2],
        "ref_chars_max": lens[-1], "sha256": hashlib.sha256(body.encode()).hexdigest()}
OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
(Path.home() / "atr-cache/wsp/render/kurrent-val-64.keys").write_text("".join(o["key"] + "\n" for o in out))
print(json.dumps({k: v for k, v in meta.items() if k != "documents"}, indent=1))
