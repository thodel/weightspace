"""Overlap audit: which documents a practice set draws from, against the held-out identities.

    python -m wsp.audit overlap --practice data/practice/<set>.jsonl [...] --out data/practice/overlap.json

Held-out identities (the sets that are never practice):

- BLB: the ``*_eval`` projects of ``lassberg-blb-gt`` — object (shelfmark) ids from
  ``export.json`` ``split.projects.<p>.eval_objects``;
- Hertwig: document 13922499, the eval pages of ``pipeline.py eval``
  (8, 17, 26, 39, 48, 55; docs/MODELS.md) = the HF project ``A_V_1443_Hertwig_test``;
- german-medieval-v1: the test (and validation) document ids of
  ``training-atr-models/config/heldout_eval_documents.json``.

A practice line from an arrow has no document id (``doc: null``); such a set is
reported as *not auditable by id* with the arrow's provenance instead.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BLB_EXPORT = Path("/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/export.json")
HELDOUT_CONFIG = Path.home() / "Repo/training-atr-models/config/heldout_eval_documents.json"
HERTWIG_DOC = "13922499"
HERTWIG_EVAL_PAGES = {8, 17, 26, 39, 48, 55}


def heldout_identities() -> dict:
    out: dict = {"blb_eval_objects": {}, "hertwig": {"doc": HERTWIG_DOC, "eval_pages": sorted(HERTWIG_EVAL_PAGES)},
                 "medieval": {}}
    if BLB_EXPORT.exists():
        split = json.loads(BLB_EXPORT.read_text())["split"]
        out["blb_eval_objects"] = {p: v["eval_objects"] for p, v in split["projects"].items()}
        out["blb_split"] = {k: split[k] for k in ("by", "fraction", "seed", "forced_eval")}
    if HELDOUT_CONFIG.exists():
        for s in json.loads(HELDOUT_CONFIG.read_text())["sets"]:
            out["medieval"][s["name"]] = {"test_documents": s.get("test_documents", []),
                                          "val_documents": s.get("val_documents", []),
                                          "page_digests": s.get("page_digests")}
    return out


def _hertwig_page(key: str) -> int | None:
    m = re.search(r"/p(\d{4})/", key)
    return int(m.group(1)) if m else None


def audit_set(lst: Path, ids: dict) -> dict:
    rows = [json.loads(l) for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta_path = lst.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    docs = sorted({r["doc"] for r in rows if r.get("doc")})
    report = {"set": lst.stem, "lines": len(rows), "sha256": meta.get("sha256"), "source": meta.get("source"),
              "documents": docs, "findings": [], "auditable_by_id": bool(docs)}
    blb_eval = {o for objs in ids["blb_eval_objects"].values() for o in objs}
    blb_eval_norm = {o.replace(" ", "") for o in blb_eval}
    hits = [d for d in docs if d in blb_eval or d.replace(" ", "") in blb_eval_norm]
    if hits:
        report["findings"].append({"kind": "blb_eval_object_in_practice", "documents": hits})
    if HERTWIG_DOC in docs:
        pages = sorted({p for p in (_hertwig_page(r["key"]) for r in rows) if p is not None})
        bad = sorted(set(pages) & HERTWIG_EVAL_PAGES)
        report["hertwig_pages"] = pages
        if bad:
            report["findings"].append({"kind": "hertwig_eval_page_in_practice", "pages": bad})
    for name, s in ids["medieval"].items():
        tdocs = set(s["test_documents"]); vdocs = set(s["val_documents"])
        t_hits = [d for d in docs if d in tdocs]
        v_hits = [d for d in docs if d in vdocs]
        if t_hits:
            report["findings"].append({"kind": f"{name}_test_document_in_practice", "documents": t_hits})
        if v_hits:
            report["findings"].append({"kind": f"{name}_val_document_in_practice", "documents": v_hits})
    if not docs:
        report["provenance"] = meta.get("source_info")
        report["note"] = ("no document ids in the source; audit by provenance: "
                          "the arrow's identity is its sha256 and the job that compiled it")
    report["clean"] = not report["findings"]
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overlap")
    o.add_argument("--practice", nargs="+", required=True)
    o.add_argument("--out", default="data/practice/overlap.json")
    a = ap.parse_args(argv)
    ids = heldout_identities()
    reports = [audit_set(Path(p), ids) for p in a.practice]
    result = {"heldout_identities": ids, "sets": reports}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for r in reports:
        status = "clean" if r["clean"] else "FINDINGS " + json.dumps(r["findings"])
        ident = f"{len(r['documents'])} documents" if r["auditable_by_id"] else "not auditable by id (arrow)"
        print(f"{r['set']}: {r['lines']} lines, {ident}: {status}")


if __name__ == "__main__":
    main()
