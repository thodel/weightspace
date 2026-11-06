"""Put the correctly paired Hertwig pages back on dh-unibe/image-text_bern-disputation-1528.

The export of 25.10.2026 16:03 UTC (commit c2aa325b, now `main`) pairs 44 of the 50
Hertwig *train* pages and 5 of the 6 Hertwig *test* pages with the wrong page image;
the PAGE XML is identical to the 22.10 export (commit 28a16d55), whose images match
their text (docs/PHASE0_REPORT.md §3, docs/HERTWIG_REVISIONS.md). The Schlussredaktion
shards of 24.10 and 25.10 are byte-identical and are not touched.

What --apply does, as one commit on `main`:
  delete  data/train/A_V_1443_Hertwig_train/20261025_170131-0000.parquet
  delete  data/train/A_V_1443_Hertwig_test/20261025_170131-0000.parquet
  add     data/train/A_V_1443_Hertwig_train/20261022_114935-0000.parquet   (the 22.10 file, unchanged bytes)
  add     data/train/A_V_1443_Hertwig_test/20261022_114935-0000.parquet    (the 22.10 file, unchanged bytes)

Run from the weightspace checkout on asteraix, with a Hugging Face token that may write to
the dh-unibe organisation (`hf auth login`, or HF_TOKEN in the environment):

    .venv/bin/python scripts/fix_hertwig_hf.py            # dry run: downloads, checks, prints the plan
    .venv/bin/python scripts/fix_hertwig_hf.py --apply    # makes the commit, then verifies main

Nothing is written without --apply. The script never prints a token.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

REPO = "dh-unibe/image-text_bern-disputation-1528"
GOOD_REV = "28a16d55"          # export of 22.10.2026, the one hertwig-v1/v2 were trained on
BAD_REV = "c2aa325b"           # export of 25.10.2026, images swapped
GOOD = {s: f"data/train/A_V_1443_Hertwig_{s}/20261022_114935-0000.parquet" for s in ("train", "test")}
BAD = {s: f"data/train/A_V_1443_Hertwig_{s}/20261025_170131-0000.parquet" for s in ("train", "test")}
UNTOUCHED = ["data/train/A_V_1447_Schlussredaktion_train/20261025_170131-0000.parquet",
             "data/train/A_V_1447_Schlussredaktion_test/20261025_170131-0000.parquet"]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pages(path: str) -> dict[str, tuple[str, str]]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return {f: (hashlib.sha256(x.encode()).hexdigest(), hashlib.sha256(i["bytes"]).hexdigest())
            for f, x, i in zip(t.column("filename").to_pylist(), t.column("xml_content").to_pylist(),
                               t.column("image").to_pylist())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="make the commit on main (default: dry run)")
    a = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    api = HfApi()

    main_files = set(api.list_repo_files(REPO, repo_type="dataset", revision="main"))
    print("main holds:", *sorted(main_files), sep="\n  ")
    if all(GOOD[s] in main_files for s in GOOD) and not any(BAD[s] in main_files for s in BAD):
        print("\nmain already carries the 22.10 Hertwig shards; nothing to do.")
        return 0
    missing = [BAD[s] for s in BAD if BAD[s] not in main_files]
    if missing:
        print("\nmain does not look like the 25.10 export this script expects; stopping. Missing:", missing)
        return 2

    ok = True
    good_local = {}
    for s in ("train", "test"):
        g = hf_hub_download(REPO, GOOD[s], repo_type="dataset", revision=GOOD_REV)
        b = hf_hub_download(REPO, BAD[s], repo_type="dataset", revision="main")
        good_local[s] = g
        pg, pb = pages(g), pages(b)
        same_names = set(pg) == set(pb)
        same_xml = sum(pg[k][0] == pb[k][0] for k in pg if k in pb)
        same_img = sum(pg[k][1] == pb[k][1] for k in pg if k in pb)
        print(f"\nHertwig_{s}: {len(pg)} pages at 22.10, {len(pb)} on main; same names {same_names}; "
              f"identical XML {same_xml}/{len(pg)}; identical image {same_img}/{len(pg)}")
        print(f"  22.10 file sha256 {sha(g)}")
        if not same_names or same_xml != len(pg):
            print("  the XML differs between the exports: this is not the image swap the script was written for")
            ok = False
    if not ok:
        print("\nstopping without a commit.")
        return 2

    ops = [CommitOperationDelete(path_in_repo=BAD[s]) for s in ("train", "test")] + \
          [CommitOperationAdd(path_in_repo=GOOD[s], path_or_fileobj=good_local[s]) for s in ("train", "test")]
    print("\nplanned commit on main:")
    for op in ops:
        kind = "delete" if isinstance(op, CommitOperationDelete) else "add   "
        print(f"  {kind} {op.path_in_repo}")
    print("  untouched:", *UNTOUCHED, sep="\n    ")
    if not a.apply:
        print("\ndry run: nothing written. Re-run with --apply to commit.")
        return 0

    info = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=ops, revision="main",
        commit_message="Hertwig: restore the 22.10 export (images of 25.10 were paired with the wrong XML)",
        commit_description=(
            "The export of 25.10.2026 (c2aa325b) pairs 44 of 50 A_V_1443_Hertwig_train pages and 5 of 6 "
            "A_V_1443_Hertwig_test pages with another page's image; the PAGE XML is unchanged. This commit puts the "
            "22.10 shards (28a16d55, byte-identical) back. Evidence: kraken-bern-hertwig-v2 reads the test pages of "
            "22.10 at 23.6 % CER and those of 25.10 at 78.1 %; the one page whose image did not change reads the "
            "same in both. Schlussredaktion is unaffected. See thodel/weightspace docs/HERTWIG_REVISIONS.md."),
    )
    print("\ncommitted:", info.oid, info.commit_url)

    new_files = set(api.list_repo_files(REPO, repo_type="dataset", revision=info.oid))
    for s in ("train", "test"):
        p = hf_hub_download(REPO, GOOD[s], repo_type="dataset", revision=info.oid, force_download=True)
        print(f"verify Hertwig_{s}: on main {GOOD[s] in new_files}, old shard gone {BAD[s] not in new_files}, "
              f"bytes equal to 22.10 {sha(p) == sha(good_local[s])}")
    print("\nNext: training jobs should pin this commit (the trainer records revision: null today).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
