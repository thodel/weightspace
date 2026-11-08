"""A3's score set: 64 registered-val pages that are not in the practice draw.

Stands in for the held-out set in the planted-defect control (stage A3) so that the test side
stays closed until the one score of the real session (A4). Same side, same rules as
draw_kurrent_practice.py: drawn from references and document keys alone, seed 20261107, one
page per document first (a document with nothing left is skipped), the rest from the two
large documents. No model output enters the draw.
"""
import collections, hashlib, json, random
from pathlib import Path

SEED = 20261107
SIDE = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-val")
RENDER = Path.home() / "atr-cache/wsp/render/kurrent-val/images"
used = {json.loads(l)["key"] for l in Path("data/practice/kurrent-val-64.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
rows = [json.loads(l) for l in (SIDE / "pages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    k = Path(r["image"]).stem
    if k not in used:
        by[k.split("_")[1]].append(r)
rng = random.Random(SEED + 1)
pick = []
for d in sorted(by):
    pick.append(rng.choice(sorted(by[d], key=lambda r: r["image"])))
rest = [r for d in sorted(by) for r in sorted(by[d], key=lambda r: r["image"]) if r not in pick]
pick += rng.sample(rest, max(0, 64 - len(pick)))
pick = pick[:64] if len(pick) > 64 else pick
out = [{"key": Path(r["image"]).stem, "text": r["text"], "image": str(RENDER / (Path(r["image"]).stem + ".png")),
        "doc": Path(r["image"]).stem.split("_")[1], "source": r["image"]} for r in pick]
body = "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
Path("data/practice/kurrent-val-holdout-64.jsonl").write_text(body, encoding="utf-8")
meta = {"set": "kurrent-val-holdout-64", "n": len(out), "seed": SEED + 1, "docs": len({o['doc'] for o in out}),
        "disjoint_from": "kurrent-val-64 (pages)", "role": "A3 score set only; never practice; test side untouched",
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "ref_chars_median": sorted(len(o["text"]) for o in out)[len(out)//2]}
Path("data/practice/kurrent-val-holdout-64.meta.json").write_text(json.dumps(meta, indent=2))
print(meta)
