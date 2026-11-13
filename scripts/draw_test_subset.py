"""The held-out set of stage A4: a document-stratified subset of the trainer's frozen test draw.

Two pages from each of the 70 test documents of german-kurrent-xvi-xix-v1 (one if a document has one), seed 20261109, chosen from
page file names and document keys alone. The frozen draw has 733 pages, which cost about seven hours per reading with the
transformers scorer; the subset is read once for the unedited adapter and once for the frozen one. The trainer's own full-draw numbers
for v2 (CER 0.4181, truncated_cer 0.2502, 49 of 733 at the cap) are quoted beside it, never mixed with it.
The reference texts are copied into the draw file (the scorer needs them) but not printed or read by the programmer: the file lives
outside the sandbox's reach (the held-out tree is not mounted in the session).
"""
import collections, hashlib, json, random
from pathlib import Path

SEED = 20261109
SIDE = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-test")
OUT = Path.home() / "atr-cache/wsp/heldout/kurrent-test-134"
OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (SIDE / "pages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[Path(r["image"]).stem.split("_")[1]].append(r)
rng = random.Random(SEED)
pick = []
for d in sorted(by):
    pick += rng.sample(sorted(by[d], key=lambda r: r["image"]), min(2, len(by[d])))
body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pick)
(OUT / "pages.jsonl").write_text(body, encoding="utf-8")
meta = {"n": len(pick), "documents": len(by), "seed": SEED, "source": str(SIDE),
        "source_pages_sha256": hashlib.sha256((SIDE / "pages.jsonl").read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(body.encode()).hexdigest()}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print(meta)
