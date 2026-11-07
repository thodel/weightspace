"""Seed-null data: the 22.10.2026 Hertwig train project split by page into train and validation.

Writes the PAGE XML and image of every page under ~/atr-cache/wsp/seed-null/pages/{train,val}/,
the line lists docs/seed-null/{train,val}.jsonl (key, text, doc; for wsp.audit) and a manifest.
The split is page-level, seeded, and fixed for every seed of the null. Compile with
`ketos compile -f page` afterwards (scripts/seed_null_run.sh).
"""
import hashlib, json, random, re, sys
from pathlib import Path
import pyarrow.parquet as pq

PARQUET = Path.home() / "atr-cache/wsp/hf/bern-disputation-1528-rev20261022/data/train/A_V_1443_Hertwig_train/20261022_114935-0000.parquet"
OUT = Path.home() / "atr-cache/wsp/seed-null"
SPLIT_SEED, N_VAL = 42, 5            # trainer: partition 0.9, seed 42, page-level; 5 of 50 pages

sys.path.insert(0, ".")
from wsp.data import page_lines, page_doc_id, sha256_file


def main():
    t = pq.read_table(PARQUET)
    names = t.column("filename").to_pylist()
    order = sorted(range(len(names)), key=lambda i: names[i])
    val_idx = set(random.Random(SPLIT_SEED).sample(order, N_VAL))
    man = {"parquet": str(PARQUET), "parquet_sha256": sha256_file(PARQUET), "split_seed": SPLIT_SEED,
           "val_pages": [], "train_pages": []}
    rows = {"train": [], "val": []}
    for i in order:
        split = "val" if i in val_idx else "train"
        xml = t.column("xml_content")[i].as_py()
        img = t.column("image")[i].as_py()["bytes"]
        stem = Path(names[i]).stem
        d = OUT / "pages" / split
        d.mkdir(parents=True, exist_ok=True)
        m = re.search(r'imageFilename="([^"]+)"', xml)
        (d / stem).with_suffix(".xml").write_text(xml, encoding="utf-8")
        (d / (m.group(1) if m else names[i])).write_bytes(img)
        man[f"{split}_pages"].append(stem)
        doc = page_doc_id(xml, stem)
        for ln in page_lines(xml):
            rows[split].append({"key": f"A_V_1443_Hertwig_train/{stem}/{ln['id']}", "text": ln["text"], "doc": doc})
    for split, r in rows.items():
        p = Path("docs/seed-null") / f"{split}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r), encoding="utf-8")
        man[f"{split}_lines"] = len(r)
        man[f"{split}_sha256"] = sha256_file(p)
    Path("docs/seed-null/split.json").write_text(json.dumps(man, indent=1))
    print({k: (len(v) if isinstance(v, list) else v) for k, v in man.items()})


main()
