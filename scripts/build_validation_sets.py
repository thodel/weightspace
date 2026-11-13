"""Validation-based practice sets that are not training data (docs/CONTROLS_REPORT.md, item 3).

    .venv/bin/python scripts/build_validation_sets.py

bern-val: the 17 pages the trainer reserved for validation in the job of kraken-bern-combined-v1
(`pages_val.lst` of job 20261025T160407Z, partition 0.9, page level), every line, Hertwig pages from the 22.10 export
(correct images), Schlussredaktion pages from the 25.10 export. Each line carries `seen_by`: the models of the family
whose own training pages include that page (hertwig-v2 trained on every page of the Hertwig project but its own 5 validation
pages; schlussredaktion-v1 likewise for its 12).

blb-val: the trainer's job for kraken-german-xix-blb-v1 used the 157 `_eval` pages as its validation set and trained on every
line of the four train projects, so there is no validation side apart from the eval side. blb-val-300 is carved from the eval
side at *object* level: a seeded draw of whole objects out of blb-eval-hand until 450 lines, 300 lines sampled from them;
blb-eval-hand-b is blb-eval-hand without those objects (the held-out set for sessions that use blb-val-300).
"""
import json
import random
from pathlib import Path

from wsp.data import ParquetSource, cut_line, sha256_file, page_nr, page_doc_id, page_lines, load, PRACTICE_ROOT
import io, re
from PIL import Image

H = Path.home() / "atr-cache/wsp/hf"
HERTWIG = H / "bern-disputation-1528-rev20261022/data/train/A_V_1443_Hertwig_train/20261022_114935-0000.parquet"
SCHLUSS = H / "bern-disputation-1528-rev20261025/data/train/A_V_1447_Schlussredaktion_train/20261025_170131-0000.parquet"
VAL = {"13922499": {5, 16, 20, 31, 34, 38}, "13922579": {16, 17, 20, 31, 34, 94, 99, 112, 137, 143, 566}}   # pages_val.lst of the combined job
HV2_VAL = {4, 11, 21, 23, 50}                                                                               # hertwig-v1/v2 job (same artefact)
SCH_VAL = {7, 20, 21, 24, 36, 42, 46, 103, 113, 121, 143, 566}                                              # schlussredaktion-v1 job


SKIPPED = []


def bern():
    rows, pages = [], []
    out = PRACTICE_ROOT / "bern-val"
    out.mkdir(parents=True, exist_ok=True)
    for path in (HERTWIG, SCHLUSS):
        src = ParquetSource(str(path))
        order = sorted(range(len(src.filenames)), key=lambda j: (src.projects[j], src.filenames[j]))
        for i in order:
            xml = src.xml[i]
            doc, nr = page_doc_id(xml, ""), int(page_nr(xml) or -1)
            if nr not in VAL.get(doc, set()):
                continue
            pages.append((doc, nr))
            im = Image.open(io.BytesIO(src.table.column("image")[i].as_py()["bytes"]))
            for ln in page_lines(xml):
                try:
                    crop = cut_line(im, ln)
                except ValueError as e:      # degenerate geometry: kraken refuses, as for sample_lines
                    SKIPPED.append({"page": f"{doc}/{nr}", "line": ln["id"], "reason": str(e)}); continue
                key = f"{src.projects[i]}/{Path(src.filenames[i]).stem}/{ln['id']}"
                f = out / (re.sub(r"[^A-Za-z0-9._-]+", "_", key) + ".png")
                crop.save(f)
                seen = []
                if doc == "13922499":
                    seen = ["kraken-bern-hertwig-v1", "kraken-bern-hertwig-v2"] if nr not in HV2_VAL else []
                else:
                    seen = ["kraken-bern-schlussredaktion-v1"] if nr not in SCH_VAL else []
                rows.append({"key": key, "text": ln["text"], "image": str(f), "doc": doc, "page": nr,
                             "image_sha256": sha256_file(f), "seen_by": seen})
    return rows, sorted(set(pages))


def write(name, rows, root, meta_extra):
    lst = root / f"{name}.jsonl"
    lst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {"set": name, "n": len(rows), "sha256": sha256_file(lst), "candidates": len(rows), "seed": 20261107,
            "documents": sorted({r["doc"] for r in rows if r.get("doc")}), **meta_extra}
    (root / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def blb():
    hand = load("data/heldout/blb-eval-hand.jsonl")
    docs = sorted({r["doc"] for r in hand})
    rng = random.Random(20261107)
    rng.shuffle(docs)
    chosen, n = [], 0
    for d in docs:
        if n >= 450:
            break
        chosen.append(d); n += sum(1 for r in hand if r["doc"] == d)
    pool = [r for r in hand if r["doc"] in chosen]
    prac = random.Random(20261107).sample(pool, 300)
    prac = [{**r, "seen_by": []} for r in prac]
    rest = [r for r in hand if r["doc"] not in chosen]
    return prac, rest, sorted(chosen)


if __name__ == "__main__":
    P, H_ = Path("data/practice"), Path("data/heldout")
    rows, pages = bern()
    m = write("bern-val", rows, P, {"source": f"parquet:{HERTWIG},{SCHLUSS}", "selection": "every line of the 17 validation pages of the combined-v1 job (pages_val.lst)",
                                  "pages": [f"{d}/{n}" for d, n in pages], "skipped": SKIPPED})
    print("bern-val", m["n"], m["sha256"][:12])
    prac, rest, chosen = blb()
    m = write("blb-val-300", prac, P, {"source": "data/heldout/blb-eval-hand.jsonl (eval objects)", "objects": chosen,
              "selection": "whole objects drawn with random.Random(20261107) until 450 hand lines, then 300 lines sampled; the eval side is the trainer's validation set (see script header)"})
    print("blb-val-300", m["n"], m["sha256"][:12], chosen)
    m = write("blb-eval-hand-b", rest, H_, {"source": "data/heldout/blb-eval-hand.jsonl minus the objects of blb-val-300", "excluded_objects": chosen})
    print("blb-eval-hand-b", m["n"], m["sha256"][:12])
