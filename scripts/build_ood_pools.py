"""Out-of-domain sources for RQ3 at 15-30 % CER (docs/MODEL_FAMILY.md, 08.11.2026): split each source
into a practice pool and a held-out pool by document, *before* any model reads a line of it.

    python scripts/build_ood_pools.py [<source> ...]

For every source in SOURCES: read the HF ``datasets`` arrow cache (or parquet shards) page by page, keep
the projects the rule keeps (training/validation duplicates and other-language sub-collections out), group
the pages into documents (the Transkribus ``docId``, or the project where one volume is one project),
and draw the held-out documents with ``random.Random(20261108)`` (a quarter of the documents, at least
one). Nothing here looks at a model or a reading: the split is a property of the data only.

Writes ``~/atr-cache/wsp/ood/<source>/{pool,heldout}.parquet`` (the page rows, image bytes included, in
the ``dh-unibe/image-text_*`` layout that ``wsp.data``'s ``parquet:`` source reads) and
``data/ood/<source>.split.json`` (documents per side, page and line counts, parquet sha256).
Practice and confirmation lines are then sampled from ``pool.parquet`` only (``wsp.data build``), the
held-out set from ``heldout.parquet`` only (``scripts/build_ood_heldout.py``).
"""
from __future__ import annotations

import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from wsp.data import page_doc_id, page_lines, sha256_file  # noqa: E402

HUB = "/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub/datasets"
OUT = Path.home() / "atr-cache" / "wsp" / "ood"
SPLIT_SEED = 20261108
HELDOUT_SHARE = 0.25
#: lines kept per side: the side's pages in a seeded random order, taken until the budget is reached (so the
#: copies stay small; practice and confirmation need ~600 lines, the held-out set ~1,000-2,000)
MAX_LINES = {"pool": 6000, "heldout": 2000}

# name -> dir, keep-project regex, drop-project regex, document unit, note
SOURCES = {
    # 16th-century Swiss chancery and letter hands: outside the Bern lineage's training
    # (catmus_medieval: CATMuS Medieval, fro/la/es/it; cursive_inzigkofen: Johannes Jaeck, 15th c., Berlin mss.;
    #  bern-hertwig-v2: Bern A V 1443 only)
    # (St. Gallen Missiven, dh-unibe/image-text_sg-missiven, would fit but carries no line geometry: page-level text only)
    "rats-richtebuecher": (f"{HUB}/dh-unibe___image-text_rats-und-richtebuecher_xv-xvi", r"^Rats-undRichteb", None, "project",
                           "Rats- und Richtebücher XV-XVI; one microfilm volume per project; TRAINING_VALIDATION_SET_*, Test_* and "
                           "escript_test* repeat pages of the volumes and are dropped"),
    "bullinger": (f"{HUB}/dh-unibe___image-text_bullinger-autoren", None, r"^TRAINING_|TEST", "docid",
                  "Bullinger correspondence, letters by author; TRAINING_* projects repeat pages and are dropped"),
    "kf-post-1500": (f"{HUB}/dh-unibe___image-text_koenigsfelden-charters-post-1500", None, r"^TRAINING_|^FRAD", "project",
                     "Königsfelden charters; TRAINING_* repeat pages, FRAD068 (Colmar) is another archive and dropped; one charter per project"),
    "aaeb": (f"{HUB}/dh-unibe___image-text_aaeb-xiv-xvii", None, r"^TRAINING_|test", "docid",
             "Archives de l'ancien Évêché de Bâle XIV-XVII; TRAINING_* repeat pages"),
    # 19th/20th-century German hands: outside the Kurrent lineage's training
    # (german-xix-v1: zh-regierungsratsprotokolle MM_1_001-MM_2_110, kurrent-xix TRAIN_*, parlamentsdienste 1849,
    #  nr-sr-vereinigte-bundesversammlung; german-xix-blb-v1: lassberg-blb-gt; bohemian_19th: Bohemian 19th-c. records)
    "federal-minutes": ("/mnt/wbkolleg_dh_1/tei/datasets/dh-unibe__image-text_federal-minutes-testset", None, None, "docid",
                        "Swiss Federal Council minutes 1848-1903 (german-xix-v1's benchmark; never trained on)"),
    "basel-grundbuch": (f"{HUB}/dh-unibe___image-text_historisches-grundbuch-basel_xix-xx_train", None, None, "docid-volume",
                        "Historisches Grundbuch Basel XIX-XX; one page per project HGB_<a>_<vol>_<b>_<page>; document = Transkribus docId, "
                        "or HGB_<a>_<vol>_<b> where the page has none"),
    "german-20th": (f"{HUB}/dh-unibe___image-text_german-20th-century", r"^TRAIN_|^TRAINING_TESTSET", r"duplicated|\+$|_M2$", "project",
                    "German 20th-century hands (CITlab collections); duplicated/'+'/M2 re-exports of the same pages dropped"),
}


def read_pages(d: str):
    files = sorted(glob.glob(d + "/**/*.arrow", recursive=True)) or sorted(glob.glob(d + "/**/*.parquet", recursive=True))
    for f in files:
        if f.endswith(".parquet"):
            t = pq.read_table(f)
        else:
            with pa.memory_map(f) as src:
                try:
                    t = pa.ipc.open_stream(src).read_all()
                except pa.ArrowInvalid:
                    t = pa.ipc.open_file(src).read_all()
        t = t.select(["image", "xml_content", "filename", "project_name"])
        yield f, t


def doc_of(unit: str, project: str, xml: str, filename: str) -> str:
    if unit == "docid":
        return page_doc_id(xml, project)
    if unit == "project":
        return project
    if unit == "volume":
        return re.sub(r"_\d+$", "", project)
    if unit == "docid-volume":
        return page_doc_id(xml, re.sub(r"_\d+$", "", project))
    raise ValueError(unit)


def build(name: str) -> dict:
    d, keep, drop, unit, note = SOURCES[name]
    rows, seen = [], set()
    for f, t in read_pages(d):
        proj = t.column("project_name").to_pylist(); fn = t.column("filename").to_pylist(); xml = t.column("xml_content").to_pylist()
        for i in range(t.num_rows):
            if keep and not re.search(keep, proj[i]):
                continue
            if drop and re.search(drop, proj[i]):
                continue
            key = (proj[i], fn[i])
            if key in seen:
                continue
            seen.add(key)
            n = len(page_lines(xml[i]))
            if not n:
                continue
            rows.append({"file": f, "i": i, "project": proj[i], "filename": fn[i],
                         "doc": doc_of(unit, proj[i], xml[i], fn[i]), "lines": n})
    docs = sorted({r["doc"] for r in rows})
    k = max(1, round(HELDOUT_SHARE * len(docs)))
    held = set(random.Random(SPLIT_SEED).sample(docs, k))
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for side, pick in (("pool", lambda r: r["doc"] not in held), ("heldout", lambda r: r["doc"] in held)):
        sel = sorted((r for r in rows if pick(r)), key=lambda r: (r["project"], r["filename"]))
        side_docs, side_pages = sorted({r["doc"] for r in sel}), len(sel)
        if sum(r["lines"] for r in sel) > MAX_LINES[side]:
            shuffled, kept, n = random.Random(SPLIT_SEED).sample(sel, len(sel)), [], 0
            for r in shuffled:
                if n >= MAX_LINES[side]:
                    break
                kept.append(r); n += r["lines"]
            sel = sorted(kept, key=lambda r: (r["project"], r["filename"]))
        by_file = defaultdict(list)
        for r in sel:
            by_file[r["file"]].append(r["i"])
        parts = []
        for f, t in read_pages(d):
            if f in by_file:
                parts.append(t.take(pa.array(sorted(by_file[f]))))
        tab = pa.concat_tables(parts)
        # one fixed order: (project, filename)
        order = sorted(range(tab.num_rows), key=lambda j: (tab.column("project_name")[j].as_py(), tab.column("filename")[j].as_py()))
        tab = tab.take(pa.array(order))
        p = out / f"{side}.parquet"
        pq.write_table(tab, p)
        tables[side] = {"parquet": str(p), "sha256": sha256_file(p), "pages": tab.num_rows,
                        "lines": sum(r["lines"] for r in sel), "documents": sorted({r["doc"] for r in sel}),
                        "side_documents": side_docs, "side_pages": side_pages,
                        "page_sample": (f"pages in random.Random({SPLIT_SEED}).sample order until {MAX_LINES[side]} lines"
                                        if tab.num_rows < side_pages else "all")}
    rec = {"source": name, "dir": d, "keep_projects": keep, "drop_projects": drop, "document_unit": unit, "note": note,
           "split": f"random.Random({SPLIT_SEED}).sample(sorted documents, max(1, round({HELDOUT_SHARE} * n))) -> held-out",
           "documents": len(docs), **tables}
    (REPO / "data" / "ood").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "ood" / f"{name}.split.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


if __name__ == "__main__":
    for n in sys.argv[1:] or SOURCES:
        r = build(n)
        print(f"{n:20s} docs {r['documents']:4d}  pool {r['pool']['pages']:5d} p / {r['pool']['lines']:6d} l"
              f"  heldout {r['heldout']['pages']:5d} p / {r['heldout']['lines']:6d} l ({len(r['heldout']['documents'])} docs)", flush=True)
