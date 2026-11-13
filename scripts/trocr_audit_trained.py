#!/usr/bin/env python3
"""Overlap audit of the BLB practice and held-out lines against what the TrOCR jobs were trained on (stage 0).

    python scripts/trocr_audit_trained.py --out docs/trocr/audit_trained_overlap.json

For each trainer job of the family (the job folders under training_folder/jobs hold the page XML of every training,
validation and reserved page and the line lists): (1) dataset level: the 20 datasets and project names of the job's
request, scanned for BLB / Lassberg / shelfmark names; (2) document level: every page XML scanned for a BLB shelfmark
(K 29xx..., K 31xx...), 'Lassberg', 'Badische Landesbibliothek'; (3) line level: exact match (NFD, whitespace
collapsed, at least 20 characters) of every practice and held-out line against every line of the job's pages.
The parents (trocr-kurrent, trocr-kurrent-XVI-XVII) were trained elsewhere and their corpora are described only in prose
on their cards; they cannot be audited here (stated in the output).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

JOBS = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs")
FAMILY = {"from-kurrent-v2": "20261026T064540Z-trocr-german-kurrent-xvi-xix-from-kurrent-v2",
          "v2": "20261026T071211Z-trocr-german-kurrent-xvi-xix-v2",
          "full-v1": "20261027T051612Z-trocr-german-kurrent-xvi-xix-full-v1"}
PAT = re.compile(r"K ?(?:29|31)\d\d[ ,\-]\s?\d+|La(?:ss|ß)berg|Badische Landesbibliothek|\bBLB\b", re.I)
UNI = re.compile(r"<Unicode>(.*?)</Unicode>", re.S)


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFD", s).split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/trocr/audit_trained_overlap.json")
    ap.add_argument("--practice", default="data/practice/trocr-blb-train-300.jsonl")
    ap.add_argument("--heldout", default="data/heldout/blb-eval-hand.jsonl")
    ap.add_argument("--minlen", type=int, default=20)
    a = ap.parse_args()
    sets = {}
    for name, p in (("practice", a.practice), ("heldout", a.heldout)):
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        sets[name] = {"rows": rows, "norm": [norm(r["text"]) for r in rows], "docs": sorted({r["doc"] for r in rows if r.get("doc")})}
    res: dict = {"min_line_chars": a.minlen, "pattern": PAT.pattern,
                 "sets": {k: {"lines": len(v["rows"]), "documents": len(v["docs"]),
                              "lines_ge_minlen": sum(len(t) >= a.minlen for t in v["norm"])} for k, v in sets.items()},
                 "jobs": {}, "not_auditable": ["dh-unibe/trocr-kurrent (292,997 lines, 'Kurrent dataset', many small collections)",
                                               "dh-unibe/trocr-kurrent-XVI-XVII (about 1.58 M words)",
                                               "microsoft/trocr-base-handwritten (IAM, English)"]}
    for tag, jid in FAMILY.items():
        job = JOBS / jid
        j = json.loads((job / "job.json").read_text())
        ds = j["request"]["datasets"]
        r: dict = {"job": jid, "datasets": len(ds), "hf_repos": [d["hf_repo"] for d in ds],
                   "dataset_level_hits": [d["hf_repo"] for d in ds if PAT.search(json.dumps(d, ensure_ascii=False))],
                   "train_samples": j.get("progress", {}).get("train_samples")}
        pages = sorted((job / "data" / "pages").glob("*.xml"))
        hits, lines = [], set()
        for p in pages:
            x = p.read_text(encoding="utf-8", errors="replace")
            texts = UNI.findall(x)
            for t in texts:
                lines.add(norm(t))
            m = PAT.search(x)
            if m:
                # a mention inside a transcribed line is text of the page (a printed book may talk about Lassberg), not an
                # identity of the page; anything outside the lines (metadata, comments, file names) would be identity
                in_text = bool(PAT.search("\n".join(texts)))
                hits.append({"page": p.name, "match": m.group(0), "in_transcribed_text": in_text,
                             "context": x[max(0, m.start() - 60):m.end() + 40].replace("\n", " ") if in_text else None})
        r["pages"] = len(pages)
        r["distinct_lines"] = len(lines)
        r["page_xml_metadata_hits"] = hits   # includes mentions inside transcribed text, marked
        for name, v in sets.items():
            m = [i for i, t in enumerate(v["norm"]) if len(t) >= a.minlen and t in lines]
            r[f"{name}_exact_line_matches"] = {"count": len(m), "keys": [v["rows"][i]["key"] for i in m][:20]}
        res["jobs"][tag] = r
        print(tag, r["pages"], "pages", r["distinct_lines"], "lines; meta hits", len(hits), "dataset hits", r["dataset_level_hits"],
              "practice matches", r["practice_exact_line_matches"]["count"], "heldout matches", r["heldout_exact_line_matches"]["count"])
    res["clean"] = all(not [h for h in r["page_xml_metadata_hits"] if not h["in_transcribed_text"]] and not r["dataset_level_hits"] and not r["practice_exact_line_matches"]["count"]
                       and not r["heldout_exact_line_matches"]["count"] for r in res["jobs"].values())
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("clean" if res["clean"] else "FINDINGS")
    return 0 if res["clean"] else 2


if __name__ == "__main__":
    sys.exit(main())
