"""C4 data: training and validation pages of the combined-v1 job, Hertwig from the corrected 22.10 export.

Reads pages_train.lst / pages_val.lst of job 20261025T160407Z-kraken-bern-combined-v1 (read-only, on the share) for (docId, pageNr),
takes the PAGE XML and image of each from the parquet shards (Hertwig 22.10, Schlussredaktion 25.10), writes them under
~/atr-cache/wsp/c4/pages/{train,val}/{hertwig,schluss}/ and the line lists docs/c4/{train,val}.jsonl and docs/c4/split.json.
"""
import collections, json, re, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, ".")
from wsp.data import page_lines, page_doc_id, page_nr, sha256_file

J = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs/20261025T160407Z-kraken-bern-combined-v1/data")
H = Path.home() / "atr-cache/wsp/hf"
SRC = {"13922499": ("hertwig", H / "bern-disputation-1528-rev20261022/data/train/A_V_1443_Hertwig_train/20261022_114935-0000.parquet", "A_V_1443_Hertwig_train"),
       "13922579": ("schluss", H / "bern-disputation-1528-rev20261025/data/train/A_V_1447_Schlussredaktion_train/20261025_170131-0000.parquet", "A_V_1447_Schlussredaktion_train")}
OUT = Path.home() / "atr-cache/wsp/c4"


def wanted(name):
    out = set()
    for l in (J / f"pages_{name}.lst").read_text().split():
        x = Path(l).read_text(encoding="utf-8")
        out.add((re.search(r'docId="(\d+)"', x).group(1), int(re.search(r'pageNr="(\d+)"', x).group(1))))
    return out


def main():
    want = {"train": wanted("train"), "val": wanted("val")}
    man = {"job": "20261025T160407Z-kraken-bern-combined-v1", "sources": {}}
    rows = {"train": [], "val": []}; pages = collections.defaultdict(list)
    for doc, (short, path, proj) in SRC.items():
        t = pq.read_table(path)
        man["sources"][short] = {"parquet": str(path), "sha256": sha256_file(path)}
        names = t.column("filename").to_pylist()
        for i in sorted(range(len(names)), key=lambda j: names[j]):
            xml = t.column("xml_content")[i].as_py()
            key = (page_doc_id(xml, ""), int(page_nr(xml) or -1))
            for split in ("train", "val"):
                if key in want[split]:
                    stem = Path(names[i]).stem
                    d = OUT / "pages" / split / short; d.mkdir(parents=True, exist_ok=True)
                    m = re.search(r'imageFilename="([^"]+)"', xml)
                    (d / f"{stem}.xml").write_text(xml, encoding="utf-8")
                    (d / (m.group(1) if m else names[i])).write_bytes(t.column("image")[i].as_py()["bytes"])
                    pages[f"{split}_{short}"].append(key[1])
                    for ln in page_lines(xml):
                        rows[split].append({"key": f"{proj}/{stem}/{ln['id']}", "text": ln["text"], "doc": doc})
    Path("docs/c4").mkdir(exist_ok=True)
    for split, r in rows.items():
        p = Path("docs/c4") / f"{split}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r), encoding="utf-8")
        man[f"{split}_lines"] = len(r); man[f"{split}_sha256"] = sha256_file(p)
    man["pages"] = {k: sorted(v) for k, v in pages.items()}
    man["page_counts"] = {k: len(v) for k, v in pages.items()}
    for split in ("train", "val"):
        assert sum(len(v) for k, v in pages.items() if k.startswith(split)) == len(want[split]), split
    Path("docs/c4/split.json").write_text(json.dumps(man, indent=1))
    print({k: v for k, v in man.items() if k not in ("pages", "sources")})


main()
