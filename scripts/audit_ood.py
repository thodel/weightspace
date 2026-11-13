"""Audit of the out-of-domain sets of 08.11.2026 (docs/MODEL_FAMILY.md): practice / confirmation lines against the
held-out lines of the same source (no shared line, page or document), and every source against the training
records of the models it is used with.

    python scripts/audit_ood.py      # writes data/ood/audit.json

Training records read: the trainer's ``job.json`` of german-xix-v1, german-xix-blb-v1 and bern-hertwig-v2 (datasets
and projects trained on, benchmarks), and the Zenodo descriptions of cursive_inzigkofen (Johannes Jaeck, Berlin
Ms. germ. fol. 1045 / oct. 842), catmus_medieval (CATMuS Medieval: Old French, Latin, Spanish, Italian) and
bohemian_19th (19th-century Bohemian records); none names a source used here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from wsp.data import load, sha256_file  # noqa: E402

JOBS = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs")
TRAINING = {
    "kraken-german-xix-v1": JOBS / "20261007T131819Z-kraken-german-xix-v1/job.json",
    "kraken-german-xix-blb-v1": JOBS / "20261026T193646Z-kraken-german-xix-blb-v1/job.json",
    "kraken-bern-hertwig-v2": JOBS / "20261022T134133Z-kraken-bern-hertwig-v2/job.json",
}
#: source -> (HF repo it comes from, project rule, practice lists, held-out list, models it is used with)
SOURCES = {
    "federal-minutes": ("dh-unibe/image-text_federal-minutes-testset", "TEST_federal_minutes",
                        ["federal-minutes-300", "federal-minutes-conf-300"], "federal-minutes-heldout",
                        ["kraken-german-xix-v1", "kraken-german-xix-blb-v1"]),
    "basel-grundbuch": ("dh-unibe/image-text_historisches-grundbuch-basel_xix-xx_train", "all",
                        ["basel-grundbuch-300", "basel-grundbuch-conf-300"], "basel-grundbuch-heldout",
                        ["kraken-german-xix-v1", "kraken-german-xix-blb-v1"]),
    "schluss": ("dh-unibe/image-text_bern-disputation-1528", "A_V_1447_Schlussredaktion_train",
                ["schluss-300", "schluss-conf-300"], "schluss-heldout", ["kraken-bern-hertwig-v2"]),
}


def source_doc(key: str) -> str | None:
    """The source document a file name carries: ``docId_<n>`` (federal minutes) or the volume ``HGB_<a>_<b>_<c>``
    of a Basel page ``HGB_<a>_<b>_<c>_<page>``; None where the key carries none."""
    m = re.search(r"docId_(\d+)", key) or re.search(r"/(HGB_\d+_\d+_\d+)_\d+/", key)
    return m.group(1) if m else None


def page(key: str) -> str:
    m = re.search(r"page_(\d+)", key)
    return m.group(1) if m else "/".join(key.split("/")[:2])


def main():
    rep = {}
    for name, (repo, projects, prac, held, models) in SOURCES.items():
        H = load(REPO / "data" / "heldout" / f"{held}.jsonl")
        hk, hp, hd = {r["key"] for r in H}, {page(r["key"]) for r in H}, {r["doc"] for r in H}
        split = json.loads((REPO / "data" / "ood" / f"{name}.split.json").read_text())
        r = {"heldout": held, "heldout_sha256": sha256_file(REPO / "data" / "heldout" / f"{held}.jsonl"),
             "heldout_lines": len(H), "heldout_pages": len(hp), "heldout_documents": sorted(hd), "practice": {}, "training": {}}
        for p in prac:
            P = load(REPO / "data" / "practice" / f"{p}.jsonl")
            hs = {source_doc(x["key"]) for x in H} - {None}
            r["practice"][p] = {"lines": len(P), "shared_lines": len({x["key"] for x in P} & hk),
                                "shared_pages": len({page(x["key"]) for x in P} & hp),
                                "shared_source_documents": sorted(({source_doc(x["key"]) for x in P} - {None}) & hs)}
            if name != "schluss":       # schluss is one volume: the split is by page, the document is shared by design
                r["practice"][p]["shared_documents"] = sorted({x["doc"] for x in P} & hd)
        for m in models:
            job = json.loads(TRAINING[m].read_text())["request"]
            trained = [(d.get("hf_repo"), d.get("train_projects"), d.get("eval_projects")) for d in job["datasets"]]
            hit = [t for t in trained if t[0] == repo and (projects == "all" or projects in (t[1] or []) or projects in (t[2] or []))]
            r["training"][m] = {"datasets": [t[0] for t in trained], "source_in_training": bool(hit),
                                "benchmarks": [b.get("hf_repo") for b in job["params"].get("benchmarks") or []]}
        r["split"] = {k: split[k] for k in ("split",) if k in split}
        r["clean"] = (all(v["shared_lines"] == 0 and v["shared_pages"] == 0 and not v.get("shared_documents")
                              and not v["shared_source_documents"]
                          for v in r["practice"].values()) and not any(v["source_in_training"] for v in r["training"].values()))
        rep[name] = r
        print(name, "clean" if r["clean"] else "NOT CLEAN", {p: (v["shared_lines"], v["shared_pages"], v.get("shared_documents"), v["shared_source_documents"]) for p, v in r["practice"].items()},
              {m: v["source_in_training"] for m, v in r["training"].items()})
    (REPO / "data" / "ood" / "audit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
