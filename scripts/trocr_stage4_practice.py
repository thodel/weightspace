#!/usr/bin/env python3
"""Finish the stage-4 practice set (docs/TROCR_STAGE4_PREREG.md): add ``seen_by`` and check its overlap with the held-out set and the stage-3 set.

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/trocr_stage4_practice.py

Input: data/practice/trocr-blb-train-1200.jsonl (wsp.data build, seed 20261119, train side of the BLB hand only). Output:
data/practice/trocr-blb-train-1200-seen.jsonl (every line ``seen_by: []``: no model of the family is recorded as trained on it; the dataset-level audit of the
trainer jobs is scripts/trocr_audit_trained.py; the parents' corpora cannot be audited, an assumption stated in the preregistration) and
data/practice/overlap-trocr-blb-train-1200.json (documents, keys and exact lines of at least 20 characters, NFD, against blb-eval-hand and the 300-line set)."""
import hashlib
import json
import unicodedata
from pathlib import Path

P = Path("data/practice")
H = Path("data/heldout/blb-eval-hand.jsonl")
S3 = P / "trocr-blb-train-300-seen.jsonl"


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def norm(s):
    return " ".join(unicodedata.normalize("NFD", s).split())


src = rows(P / "trocr-blb-train-1200.jsonl")
seen = [{**r, "seen_by": []} for r in src]
body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in seen)
(P / "trocr-blb-train-1200-seen.jsonl").write_text(body, encoding="utf-8")
meta = json.loads((P / "trocr-blb-train-1200.meta.json").read_text())
held, s3 = rows(H), rows(S3)
hd, pd = {r["doc"] for r in held}, {r["doc"] for r in src}
ht = {norm(r["text"]) for r in held if len(norm(r["text"])) >= 20}
s3k = {r["key"] for r in s3}
ov = {"practice": "trocr-blb-train-1200", "lines": len(src), "documents": len(pd), "pages": len({r["key"].rsplit("/", 1)[0] for r in src}),
      "heldout_documents_shared": sorted(hd & pd), "heldout_keys_shared": len({r["key"] for r in src} & {r["key"] for r in held}),
      "heldout_exact_lines_ge20_shared": sum(1 for r in src if len(norm(r["text"])) >= 20 and norm(r["text"]) in ht),
      "stage3_set_keys_shared": len({r["key"] for r in src} & s3k), "stage3_set_lines": len(s3),
      "lines_taller_than_wide_or_ref_under_4_chars_kept": sum(1 for r in src if len(r["text"].strip()) < 4),
      "heldout_sha256": hashlib.sha256(H.read_bytes()).hexdigest()}
(P / "overlap-trocr-blb-train-1200.json").write_text(json.dumps(ov, indent=1))
meta.update({"set": "trocr-blb-train-1200-seen", "derived_from": "data/practice/trocr-blb-train-1200.jsonl", "seen_by_note":
             "every line seen_by=[]: audited for from-kurrent-v2 against its 20 datasets (docs/trocr/audit_trained_overlap_1200.json); NOT auditable for the parents' corpora "
             "(trocr-kurrent 293 k lines, XVI-XVII 1.58 M words) or the IAM base model: an assumption until the model authors answer",
             "sha256_of_this_file": hashlib.sha256(body.encode()).hexdigest()})
(P / "trocr-blb-train-1200-seen.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(json.dumps(ov, indent=1)); print("seen file sha256", meta["sha256_of_this_file"])
