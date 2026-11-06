# The Hertwig image swap, the fix, and what target 1 is now (06.11.2026)

## What happened

`dh-unibe/image-text_bern-disputation-1528` has had three exports:

| commit | date (UTC) | Hertwig project | Schlussredaktion project |
|---|---|---|---|
| `28a16d55` | 22.10.2026 09:50 | train 50 pages, test 6 pages, images match their XML | — |
| `326bcd18` | 24.10.2026 20:02 | — (only Schlussredaktion exported) | train 118, test 8 |
| `c2aa325b` (= `main`) | 25.10.2026 16:03 | same page names and **identical XML**, but **44 of 50 train images and 5 of 6 test images replaced by another page's image** | byte-identical to 24.10 |

Measured on the six Hertwig test pages (287 lines, cut from the Transkribus polygons, `wsp.data`):

| model | 22.10 images | 25.10 images |
|---|---:|---:|
| `kraken-bern-hertwig-v2` | 23.6 % CER, 11 collapsed | 78.1 %, 128 collapsed |
| `kraken-bern-combined-v1` | 26.7 %, 24 | 85.3 %, 226 |
| `kraken-cursive_inzigkofen` (parent) | 73.1 %, 24 | 87.6 %, 144 |
| p0039, the one test page whose image did not change | 22 % (hertwig-v2) | 22 % |

On the train side (phase 0, `docs/PHASE0_REPORT.md` §3) the same holds: hertwig-v2 reads its own training page p0006 at 96.8 % character accuracy with the 22.10 image and 14.3 % with the 25.10 one. The 22.10 export is the correctly paired one; nothing in 25.10 is newer than 22.10 except the wrong images.

Who is affected: every job that downloaded the dataset after 25.10 16:03 UTC and used the Hertwig project. Known: `kraken-bern-combined-v1` (job `20261025T160407Z`, started 35 s after the push; the trainer records `revision: null`). `kraken-bern-schlussredaktion-v1` trained on the 24.10 Schlussredaktion export and is not affected. Still open: which export the trainer's held-out numbers in `docs/MODELS.md` (16.7 %, 28.8 %) were scored on; they match neither column above exactly.

**Schlussredaktion checked, 06.11.2026 evening.** The two Schlussredaktion shards on `main` (`20261025_170131`, from `c2aa325b`) were compared with those of the 24.10 export (`20261024_220124`, `326bcd18`): the files are byte-identical (train sha256 `19d7ddce…`, 118 pages; test `0f8f970d…`, 8 pages), so every page has the same XML and the same image. The 25.10 re-export only renamed them. The dataset on `main` (`dda578ea`) is now correctly paired for both projects, and `bern-eval`'s 8 Schlussredaktion test pages are the pages schlussredaktion-v1 was evaluated on.

## The fix (run by hand, not from a session)

`scripts/fix_hertwig_hf.py` replaces the two Hertwig shards on `main` by the 22.10 files (byte-identical, one commit, Schlussredaktion untouched) and verifies the result. It needs a Hugging Face token that may write to `dh-unibe`. From the weightspace checkout on asteraix:

```bash
cd ~/Repo/weightspace
.venv/bin/python scripts/fix_hertwig_hf.py            # dry run: checks the two exports and prints the commit
.venv/bin/python scripts/fix_hertwig_hf.py --apply    # writes the commit on main and verifies it
```

The dry run was checked on 06.11.2026: 50/50 and 6/6 identical XML, 6/50 and 1/6 identical images, 22.10 sha256 `566f5bd1…` (train) and `0397cddd…` (test).

**Applied 06.11.2026 ~22:05 CET by Tobias Hodel: commit `dda578ea` on `main`** ("Hertwig: restore the 22.10 export"). Verified independently afterwards: `main` lists the two 20261022 Hertwig shards and the two Schlussredaktion shards, and the downloaded Hertwig shards hash to the 22.10 values. Family-1 training jobs should pin `dda578ea` (the trainer records `revision: null` today).

## What the correct pages say about combined-v1 (measured after the decision)

On `bern-eval` (545 lines) combined-v1 reads 19.3 % CER overall, hertwig-v2 27.5 %, schlussredaktion-v1 20.4 %: combined-v1 is the best of the three; its noise damage is about 3 points on the Hertwig pages (26.7 % against the parent's 23.6 %). The first session on it (docs/PHASE3_REPORT.md, R3) found no edit that removes that damage on held-out pages.

## Target 1, kept and re-framed

PLAN.md target 1 was "a fine-tune that got worse: read the diff, the damage should be localised". It stays a target, with a sharper question: **combined-v1 was trained on 168 pages of which 44 had the wrong image. Can a weight-space edit undo training on label noise, and how close does it get to a retrain on the corrected data?**

- Subject `kraken-bern-combined-v1`, donor/parent `kraken-bern-hertwig-v2` (trained on the 22.10 images).
- Practice `data/practice/bern-hertwig-train-300.jsonl` (22.10 images, train side).
- Held-out `data/heldout/bern-eval.jsonl`: the 6 Hertwig test pages of 22.10 and the 8 Schlussredaktion test pages, both projects combined-v1 was trained on.
- Hypotheses from `docs/READINGS.md` (blank bias back to the parent's, output layer from hertwig-v2 with the new rows kept, per-block α toward the parent).
- Comparator C4: the same fine-tune retrained on the corrected data through the trainer. That is a trainer job on asteraix and needs an explicit OK; until it exists, hertwig-v2 itself (the state before the noisy fine-tune) and schlussredaktion-v1 (the same parent fine-tuned on clean Schlussredaktion pages) are the reference points.
