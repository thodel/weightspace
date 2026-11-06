"""Practice sets: deterministic line samples from the train side, as line crops on local disk.

``sample_lines(source, n, seed, set_name)`` writes
``data/practice/<set_name>.jsonl`` with one ``{"key", "text", "image", "doc", "image_sha256"}``
per line, the crops under ``~/atr-cache/wsp/practice/<set_name>/``, and the
list's sha256 next to it. Sources:

- ``arrow:<file.arrow>`` — a kraken binary dataset (``ketos compile``); the
  ``lines`` column holds the line image bytes and the text. No document ids
  are stored in an arrow, so ``doc`` is ``null`` and the audit has to argue
  from the arrow's provenance.
- ``pagexml:<dir>[,<dir>...]`` — a project tree with ``<dir>/page/*.xml`` and
  ``<dir>/*.jpg`` (the ``lassberg-blb-gt`` layout). Lines are cut from the
  baseline and boundary polygons with kraken's ``extract_polygons``, as
  ``rpred`` does on the serving path. ``doc`` is the shelfmark.
- ``parquet:<file.parquet>`` — a ``dh-unibe/image-text_*`` page shard with
  ``xml_content`` (Transkribus PAGE) and ``image`` bytes. ``doc`` is the
  Transkribus ``docId``.

Selection rule, every source: enumerate every candidate line in a fixed order
(pages sorted by name, lines in document order; arrow rows by index), then
``random.Random(seed).sample(candidates, n)``. Lines without a baseline or
without text are not candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PRACTICE_ROOT = Path.home() / "atr-cache" / "wsp" / "practice"
LIST_ROOT = Path("data/practice")
HELDOUT_ROOT = Path.home() / "atr-cache" / "wsp" / "heldout"
HELDOUT_LIST_ROOT = Path("data/heldout")
SEED = 20261106
PAGE_NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"


@dataclass
class Candidate:
    key: str
    text: str
    doc: str | None
    page: str            # image id within its source
    line_id: str | None  # PAGE TextLine id; None for arrow rows
    index: int | None    # arrow row index


# -- PAGE XML -----------------------------------------------------------------

def _points(s: str) -> list[tuple[int, int]]:
    return [(int(float(x)), int(float(y))) for x, y in (p.split(",") for p in s.split())]


def page_lines(xml_text: str) -> list[dict]:
    """Every TextLine with a baseline, a boundary and text, in document order."""
    root = ET.fromstring(xml_text)
    out = []
    for tl in root.iter(PAGE_NS + "TextLine"):
        bl = tl.find(PAGE_NS + "Baseline")
        co = tl.find(PAGE_NS + "Coords")
        uni = tl.find(f"{PAGE_NS}TextEquiv/{PAGE_NS}Unicode")
        text = (uni.text or "").strip() if uni is not None else ""
        if bl is None or co is None or not text or not bl.get("points") or not co.get("points"):
            continue
        out.append({"id": tl.get("id"), "text": text,
                    "baseline": _points(bl.get("points")), "boundary": _points(co.get("points"))})
    return out


def page_doc_id(xml_text: str, fallback: str) -> str:
    """Transkribus docId, else the BLB shelfmark from the Comments JSON, else the fallback."""
    m = re.search(r'<TranskribusMetadata[^>]*docId="(\d+)"', xml_text)
    if m:
        return m.group(1)
    m = re.search(r"<Comments>(\{.*?\})</Comments>", xml_text, re.S)
    if m:
        try:
            return json.loads(m.group(1)).get("shelfmark", fallback).replace(" ", "")
        except json.JSONDecodeError:
            pass
    return fallback


def page_nr(xml_text: str) -> str | None:
    m = re.search(r'<TranskribusMetadata[^>]*pageNr="(\d+)"', xml_text)
    return m.group(1) if m else None


def cut_line(image, line: dict):
    """One line crop the way rpred cuts it (kraken.lib.segmentation.extract_polygons)."""
    from kraken.containers import BaselineLine, Segmentation
    from kraken.lib.segmentation import extract_polygons

    seg = Segmentation(type="baselines", imagename="page", text_direction="horizontal-lr",
                       script_detection=False,
                       lines=[BaselineLine(id=line["id"], baseline=line["baseline"], boundary=line["boundary"])])
    crop, _ = next(extract_polygons(image, seg))
    return crop


# -- sources -------------------------------------------------------------------

class ArrowSource:
    def __init__(self, path: str):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        self.path = Path(path)
        self._mm = pa.memory_map(str(self.path), "r")
        self.reader = ipc.open_file(self._mm)
        self.table = self.reader.read_all()
        self.lines = self.table.column("lines")

    def candidates(self) -> list[Candidate]:
        texts = self.lines.field("text") if hasattr(self.lines, "field") else None
        out = []
        n = self.table.num_rows
        stem = self.path.stem
        for i in range(n):
            t = self.lines[i].as_py()["text"] if texts is None else texts[i].as_py()
            t = (t or "").strip()
            if t:
                out.append(Candidate(key=f"{stem}:{i}", text=t, doc=None, page=stem, line_id=None, index=i))
        return out

    def image(self, c: Candidate):
        from PIL import Image
        return Image.open(io.BytesIO(self.lines[c.index].as_py()["im"]))

    def describe(self) -> dict:
        return {"kind": "arrow", "path": str(self.path), "rows": self.table.num_rows,
                "sha256": sha256_file(self.path)}


class PageXMLSource:
    def __init__(self, dirs: Iterable[str]):
        self.dirs = [Path(d) for d in dirs]
        self._xml: dict[str, str] = {}

    def _pages(self):
        for d in self.dirs:
            for xml in sorted((d / "page").glob("*.xml")):
                yield d.name, xml

    def candidates(self) -> list[Candidate]:
        out = []
        for project, xml in self._pages():
            text = xml.read_text(encoding="utf-8")
            doc = page_doc_id(text, xml.stem.split("_")[0])
            for ln in page_lines(text):
                out.append(Candidate(key=f"{project}/{xml.stem}/{ln['id']}", text=ln["text"], doc=doc,
                                     page=f"{project}/{xml.stem}", line_id=ln["id"], index=None))
        return out

    def image(self, c: Candidate):
        from PIL import Image
        project, stem = c.page.split("/", 1)
        d = next(p for p in self.dirs if p.name == project)
        xml_text = (d / "page" / f"{stem}.xml").read_text(encoding="utf-8")
        line = next(l for l in page_lines(xml_text) if l["id"] == c.line_id)
        m = re.search(r'imageFilename="([^"]+)"', xml_text)
        im = Image.open(d / (m.group(1) if m else f"{stem}.jpg"))
        return cut_line(im, line)

    def describe(self) -> dict:
        return {"kind": "pagexml", "dirs": [str(d) for d in self.dirs],
                "pages": sum(1 for _ in self._pages())}


class ParquetSource:
    def __init__(self, path: str):
        import pyarrow.parquet as pq
        self.path = Path(path)
        self.table = pq.read_table(self.path)
        self.filenames = self.table.column("filename").to_pylist()
        self.xml = self.table.column("xml_content").to_pylist()
        self.projects = self.table.column("project_name").to_pylist()

    def candidates(self) -> list[Candidate]:
        out = []
        for i in sorted(range(len(self.filenames)), key=lambda j: self.filenames[j]):
            text = self.xml[i]
            doc = page_doc_id(text, Path(self.filenames[i]).stem)
            page = page_nr(text) or Path(self.filenames[i]).stem
            for ln in page_lines(text):
                out.append(Candidate(key=f"{self.projects[i]}/{Path(self.filenames[i]).stem}/{ln['id']}",
                                     text=ln["text"], doc=doc, page=str(i), line_id=ln["id"], index=None))
        return out

    def image(self, c: Candidate):
        from PIL import Image
        i = int(c.page)
        line = next(l for l in page_lines(self.xml[i]) if l["id"] == c.line_id)
        im = Image.open(io.BytesIO(self.table.column("image")[i].as_py()["bytes"]))
        return cut_line(im, line)

    def describe(self) -> dict:
        return {"kind": "parquet", "path": str(self.path), "pages": len(self.filenames),
                "projects": sorted(set(self.projects)), "sha256": sha256_file(self.path)}


def open_source(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "arrow":
        return ArrowSource(rest)
    if kind == "pagexml":
        return PageXMLSource(rest.split(","))
    if kind == "parquet":
        return ParquetSource(rest)
    raise ValueError(f"unknown source {spec!r}; use arrow:, pagexml: or parquet:")


# -- the sampler ---------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_lines(source: str, n: int, seed: int, set_name: str,
                 list_root: Path = LIST_ROOT, image_root: Path = PRACTICE_ROOT) -> Path:
    """``n == 0`` takes every candidate line in source order (a held-out set is
    the whole eval side, never a sample; PROTOCOL.md §2.3 wants its identity)."""
    src = open_source(source)
    cands = src.candidates()
    if n and len(cands) < n:
        raise ValueError(f"{source} has {len(cands)} candidate lines, fewer than {n}")
    chosen = random.Random(seed).sample(cands, n) if n else list(cands)
    out_dir = image_root / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    list_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in chosen:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", c.key)
        img = out_dir / f"{safe}.png"
        if not img.exists():
            src.image(c).save(img)
        rows.append({"key": c.key, "text": c.text, "image": str(img), "doc": c.doc,
                     "image_sha256": sha256_file(img)})   # the list hash then identifies the pixels too
    lst = list_root / f"{set_name}.jsonl"
    lst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    digest = sha256_file(lst)
    meta = {"set": set_name, "source": source, "source_info": src.describe(), "n": len(rows), "seed": seed,
            "candidates": len(cands), "sha256": digest,
            "documents": sorted({r["doc"] for r in rows if r["doc"]}),
            "selection": ("random.Random(seed).sample over all candidate lines in source order" if n
                          else "every candidate line in source order (no sampling)")}
    (list_root / f"{set_name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return lst


def load(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


SOURCE_NOTES = {
    "bern-hertwig-train-300": "HF `dh-unibe/image-text_bern-disputation-1528`, project `A_V_1443_Hertwig_train`, "
                              "revision `28a16d55` (export of 22.10.2026 — the one hertwig-v1/v2 were trained on; "
                              "the 25.10 export pairs 44 of 50 pages with the wrong image), lines cut from the "
                              "Transkribus polygons with `extract_polygons`. Eval pages 8/17/26/39/48/55 are not in this project.",
    "bern-hertwig-train-300-export1025": "same lines (same keys, same seed) from the **25.10.2026** export (`59ec6c64`), "
                                         "kept only to document the image mismatch; not a practice set for sessions.",
    "blb-train-300": "`/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/{lassberg,korrespondenten,unbestimmt,manuskripte}` "
                     "(the four *train* projects; `_eval` projects and `drucke` excluded), PAGE XML rectangles with "
                     "bottom-edge baselines, cut with `extract_polygons`. Document = BLB shelfmark.",
    "medieval-val-300": "`~/atr-cache/arrows/german_val.arrow` (the 150 *validation* documents of `german-medieval-v1`, "
                        "compiled 29.09.2026 on asteraix; sha256 in the meta file). Rows carry no document id. "
                        "Not the test side (`german_test.arrow` untouched).",
    "medieval-val-300-archive": "the same 300 row indices drawn from the *archived* idhefix copy of the validation arrow "
                                "(`archive/idhefix-2026-09/arrows/german_val.arrow`, a different compile of the same pages); "
                                "a diagnostic set only, built to check that the asteraix and idhefix arrows agree.",
    "medieval-train-300": "`~/atr-cache/arrows/sweep_train.arrow` (train side of `german-medieval-v1`, 236,908 lines, "
                          "compiled 30.09.2026). Rows carry no document id; held-out documents were removed at compile time "
                          "by the trainer's held-out boundary.",
}


def write_readme(list_root: Path = LIST_ROOT) -> Path:
    lines = ["# Practice sets (train side only)", "",
             "Built by `python -m wsp.data build` with seed 20261106 (docs/BRIEF_PHASE0.md §4). Each list is "
             "`<set>.jsonl` with `{key, text, image, doc, image_sha256}`; the crops live under "
             "`~/atr-cache/wsp/practice/<set>/` (git-ignored). The sha256 below is of the list file, which "
             "includes every crop's sha256, so it identifies the pixels as well as the selection. "
             "`<set>.meta.json` records source, candidate count and the document ids; "
             "`overlap.json` is the audit of `python -m wsp.audit overlap`.", "",
             "| set | lines | candidates | documents | list sha256 |", "|---|---:|---:|---:|---|"]
    metas = sorted(list_root.glob("*.meta.json"))
    for m in metas:
        d = json.loads(m.read_text())
        lines.append(f"| {d['set']} | {d['n']} | {d['candidates']} | {len(d['documents']) or '—'} | `{d['sha256']}` |")
    lines.append("")
    for m in metas:
        d = json.loads(m.read_text())
        lines += [f"## {d['set']}", "", SOURCE_NOTES.get(d["set"], ""), "",
                  f"Selection: `random.Random({d['seed']}).sample` over {d['candidates']} candidate lines in source order "
                  f"(pages sorted by name, lines in document order; arrow rows by index). Source: `{d['source']}`.", ""]
        if d["documents"]:
            lines += ["Documents: " + ", ".join(f"`{x}`" for x in d["documents"]), ""]
    out = list_root / "README.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample a practice set")
    b.add_argument("--set", required=True)
    b.add_argument("--source", required=True)
    b.add_argument("--n", type=int, default=300, help="lines to sample; 0 = every line (held-out sets)")
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--heldout", action="store_true",
                   help="write under data/heldout/ and ~/atr-cache/wsp/heldout/ instead of the practice roots")
    sub.add_parser("readme", help="write data/practice/README.md from the meta files")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        roots = {"list_root": HELDOUT_LIST_ROOT, "image_root": HELDOUT_ROOT} if a.heldout else {}
        lst = sample_lines(a.source, a.n, a.seed, a.set, **roots)
        meta = json.loads((lst.with_suffix(".meta.json")).read_text())
        print(lst, meta["sha256"], f"{meta['n']} of {meta['candidates']} lines, {len(meta['documents'])} documents")
    elif a.cmd == "readme":
        print(write_readme())


if __name__ == "__main__":
    main()
