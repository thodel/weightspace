# Phase 0 report — asteraix, 06.11.2026

Executed on asteraix (dhserver03) on branch `phase0`, in this repo's own venv
(kraken 7.0.2, torch 2.10.0+cu128, `docs/ENV_ASTERAIX.txt`). Everything ran on
**CPU** (`CUDA_VISIBLE_DEVICES=""`): card 0 carried 42.6 GB of neighbours'
memory all day, card 1 is the trainer's and was never touched. Writes went
only under `~/Repo/weightspace/` and `~/atr-cache/wsp/`. No budgeted session on
a real target, no held-out score, no registry write, no upload, no change to
the trainer or the serving config.

## 1. What was verified (brief §1–§3, §5)

| item | result |
|---|---|
| venv, kraken pin | `kraken==7.0.2`, torch 2.10.0+cu128 resolved without the fallback; `pip check` clean; 19 engine-free tests pass (`tests/`), 3 asteraix tests pass (`tests/asteraix/`) |
| weights | seven in-house + parent copied, `models/DIGESTS.md`; the three digests the gateway knows **match**; the four kraken+/shard00 comparators added with their origin |
| parent | `cursive_inzigkofen` (Zenodo 10.5281/zenodo.18207767) = the file the trainer used as base (`bases/…/cursive_johannes_jaeck.mlmodel`, same sha256 `4ab19e7b…`) |
| kraken API | names as assumed, three corrections (`docs/KRAKEN_API_NOTES.md`): (a) `forward` returns `(N, classes, frames)`, not `(N, W, C)`; (b) **`save_model` on a CoreML-loaded model writes `model_type: "r"` and `load_any` refuses the file** — `KrakenSubject.save` stores the list form, after which the serving loader accepts the copy and the metadata is equal; (c) the re-saved file is not byte-identical (metadata key order), the tensors are |
| round-trip (§5, the gate) | `xix-blb-v1` on `blb-train-300` and `hertwig-v2` on `bern-hertwig-train-300`: tensor digest equal after save, 20 lines identical before/after, CER equal to the last digit |
| sandbox | `unshare -Un` works for this account; the runner's command line (`env -i … unshare -Un .venv/bin/python -I program.py`) was tested: `wsp` importable, network blocked, models readable |

## 2. Practice sets and audit (§4)

`data/practice/README.md` lists every set with its sha256 (the list now
carries each crop's sha256, so the hash identifies the pixels);
`data/practice/overlap.json` is the audit: **all sets clean** against the BLB
`_eval` objects (40 shelfmarks), the Hertwig eval pages (8/17/26/39/48/55 of
doc 13922499 — the Hertwig train project contains pages 3–60 minus exactly
those and p0008/p0015/p0017/p0025/p0026/p0039/p0048/p0055), and the
`german-medieval-v1` test/validation document ids. The two arrow-based sets
carry no document ids and are audited by provenance (the arrows' sha256 and
the job that compiled them).

Deviations from the brief:

- The compiled train arrows the brief expected under `~/atr-cache/arrows/`
  are only `german_val.arrow`, `german_test.arrow` and the two `sweep_train*`
  arrows; the Hertwig and BLB jobs' artefacts were purged. BLB lines were
  therefore cut from the PAGE XML tree on the share (source 2 of the brief,
  which also gives exact shelfmarks for the audit); Hertwig lines from the HF
  page shard (260 MB fetched once into `~/atr-cache/wsp/hf/`).
- `medieval-val-300` is from the validation arrow as the runbook allows; the
  brief's rule "none of the documents in `heldout_eval_documents.json`" cannot
  be checked by id for it, and by construction its documents *are* the 150
  validation documents of that file (not the 200 test documents). A train-side
  alternative `medieval-train-300` (from `sweep_train.arrow`) was built as
  well; it is the one to use for sessions.
- `huggingface_hub` was added to the venv (pinned `<1.0`, because 2.x pulls
  `click 8.5` and breaks kraken's pin) to fetch the Hertwig shard.

## 3. The finding that changes target 1: the Hertwig dataset was re-exported with the wrong images

The first C0 pass on the Hertwig set gave CER 0.76–0.86 for *every* family-1
model, and kraken's own `ketos test -f xml` gave 14 % character accuracy for
`hertwig-v2` on one of its own **training** pages. The venv was ruled out (the
trainer's venv, torch 2.8, gives the same output), the line cutting was ruled
out (`ketos compile` → `ketos test -f binary` gives the same 14 %; the arrow
crops of the medieval set are read sensibly by `xix-blb-v1`), the file was
ruled out (the share copy equals the trainer's `best_0.7678.mlmodel`).

The cause is in `dh-unibe/image-text_bern-disputation-1528`: it was
re-exported on **25.10.2026 16:03 UTC** (commit `c2aa325b`, "Add new data
export"; the Hertwig shard is now `20261025_170131-0000.parquet`). Against the
22.10 export (`28a16d55`, shard `20261022_114935-0000.parquet`, the one
`hertwig-v1`/`v2` were trained on):

| | 22.10 export | 25.10 export |
|---|---|---|
| pages in `A_V_1443_Hertwig_train` | 50 | 50 (same names) |
| PAGE XML | — | **identical** on every page |
| page images | — | **44 of 50 differ** (same pixel size, 7.1 MB → 4.3 MB on p0006, different content) |
| p0006: what the image shows | the text of its XML ("Gnad vnnd frid / von gott dem Herren…") | another page's text |
| `hertwig-v2`, ketos test on p0006 | **96.8 %** char. accuracy | 14.3 % |
| C0 CER on the 300 practice lines, hertwig-v2 | **0.083** | 0.757 |
| C0 CER, combined-v1 | 0.185 | 0.859 |

So the 25.10 export pairs most Hertwig training pages with the wrong page
image. **`kraken-bern-combined-v1` was trained from that export** (job
`20261025T160407Z`, started 35 s after the push; the trainer records
`revision: null`), i.e. largely on mismatched image/text pairs — which is a
sufficient explanation for its held-out 28.8 % against its parent's 16.7 %,
and the weight-space profile of that fine-tune (blank bias up, every rare row
pushed down hard; docs/READINGS.md) is the profile of training on label noise.
`schlussredaktion-v1` (24.10) was trained from the 24.10 export of the
Schlussredaktion project, which was not checked here. What the 25.10 images
are (another page of the same volume? the eval pages?) was not determined;
thumbnails match no page of the 22.10 train shard.

Consequences: (1) the practice set `bern-hertwig-train-300` is built from the
22.10 revision; the 25.10 version is kept as `…-export1025` only as evidence;
(2) target 1 of the plan ("combined-v1 is worse than its parent — read the
weights") is first a **data** defect; the retrained comparator (C4) on the
corrected data is the honest baseline, and the weight-space question becomes
whether a model trained on noise can be repaired by reverting to the parent
(hypotheses in docs/READINGS.md); (3) **this needs to go to the dataset's
owner / the trainer repo**: every run that downloaded the dataset after
25.10 16:03 UTC trained on these pairs, and the `pipeline.py eval` numbers
for family 1 depend on which revision the eval pages came from. It is a
finding, not fixed here.

## 4. Baselines (§5) — practice numbers, not results (`docs/BASELINES.md`)

| set | model | CER | WER | collapsed | length ratio |
|---|---|---:|---:|---:|---:|
| bern-hertwig-train-300 (22.10) | cursive_inzigkofen (parent) | 0.706 | 1.260 | 23 | 0.963 |
| | bern-hertwig-v1 | 0.177 | 0.484 | 19 | 0.915 |
| | **bern-hertwig-v2** | **0.083** | 0.205 | 12 | 0.951 |
| | bern-schlussredaktion-v1 | 0.214 | 0.631 | 17 | 0.895 |
| | bern-combined-v1 | 0.185 | 0.485 | 26 | 0.909 |
| blb-train-300 | german-xix-v1 (parent) | 0.304 | 0.734 | 23 | 0.867 |
| | **german-xix-blb-v1** | **0.065** | 0.171 | 10 | 0.969 |
| medieval-train-300 | medieval-shard00-std | 0.630 | 0.958 | 159 | 0.530 |
| | plus-shard00-cr1 | 0.638 | 0.956 | 132 | 0.573 |
| | medieval-shard00-v1 | 0.624 | 0.964 | 123 | 0.603 |
| | cursive_inzigkofen | 0.738 | 1.216 | 31 | 0.923 |
| | collapse-probe-131-cr255 | 0.985 | 1.000 | 298 | 0.019 |
| medieval-val-300 | medieval-shard00-std | 0.674 | 0.960 | 183 | 0.453 |
| | plus-shard00-cr1 | 0.721 | 0.975 | 167 | 0.443 |

These are train-side lines, so the fine-tunes' own numbers are optimistic by
construction (hertwig-v2 and xix-blb-v1 read their training lines at 0.08 and
0.06). The parents' numbers on the same lines (0.71 and 0.30) are C1 of the
protocol. Top confusions are deletions everywhere (` →∅`, `n→∅`, `e→∅`), which
is the pairing the readings use.

**Family 3 is not usable as it stands.** The three archived shard00/kraken+
models of 01.09.2026 read *neither* the validation arrow (0.67–0.72) *nor* the
train-side sweep arrow (0.62–0.64) of `german-medieval-v1`, with half the lines
collapsing to a few characters — while `xix-blb-v1`, a 19th-century Kurrent
model that never saw this material, reads the same lines at 0.55 with 59
collapses: the in-domain models are worse than an out-of-domain one. Their own arrows
(`shard_00.arrow`, `val_clean.arrow` on idhefix) were not archived, so their
recorded accuracy (val 0.79–0.82, test CER 0.13–0.17) cannot be reproduced
against the material they were measured on. Whether this is the same kind of
dataset-revision drift as §3 (the four medieval HF datasets were re-exported
after September) or a different compile is an open question for phase 1; until
it is settled, no session on these models. The `Cr255` collapse probe is a
collapsed run (CER 0.99) and serves only the rewrite.

## 5. The lossless rewrite (§6) — docs/RESULTS.md R1

`Cr255,1,85,1,1 → Cr1,1,85,1,1` on the only `Cr255` checkpoint left: 15,207,871
→ 4,153,791 parameters (−11,054,080, exactly 85 × 512 × 254), identical text on
all 300 lines of `medieval-val-300`, max |Δ softmax| 1.1 × 10⁻⁶, the new file
loads through `load_any`. The archived kraken+ run of 01.09 already has the
`Cr1` head, so nothing in the serving set needs this; the padding rows sit at
their initialisation norm (docs/READINGS.md).

## 6. The session runner (§7)

`wsp/run.py`: `new`, `freeze`, `heldout`, `export`; `open_session()` is the
programmer's entry point inside `env -i … unshare -Un python -I`. `keep()`
persists `current.mlmodel`, `freeze()` writes `final.mlmodel`, `diff.json`,
`summary.json` (digests, hashes, budget, baseline and last pass, controls slot,
deviations, the sandbox command line, `env.txt` = pip freeze). The budget
counter is in-process and in `budget.json`; a reopened session sees it spent.
Smoke sessions (budget 2, `scale` of `O_18.lin.bias` by 1.0, kept, frozen):
`runs/20261106-smoke-bern-hertwig-v2-b2-1` and
`runs/20261106-smoke-german-xix-blb-v1-b2-1` — `diff.json` is `{}`, start and
final digests equal, summaries complete. `export` writes the trainer's
`trained/<id>/metadata.json` shape plus a registry YAML stub with
`enabled: false`; nothing is copied to the share. `heldout` refuses to run
because no held-out set exists under `data/heldout/` — building those is phase
1 work (they must come from the gateway/`_eval` projects and the Hertwig eval
pages, with the revision question of §3 settled first).

## 7. Readings (§8) — docs/READINGS.md

Five lineage diffs with codec rows aligned by character, the output-layer
statistics of six models, the padding rows of the Cr255 head. The three
sentences: fine-tuning inflates the rows of rare characters (norm +30–110 %,
bias −0.2…−1.0) and barely moves the frequent ones; the blank row's bias rises
in every fine-tune (most in combined-v1 and xix-blb-v1) and deletions dominate
every confusion table, so a blank-bias edit is the first cheap hypothesis for
every target; `C_0` is generic and never moves, the BLB fine-tune re-tuned
`C_3`'s bias by a factor and the deeper convs by 0.26–0.33, the Hertwig
fine-tunes moved the recurrent stack and the classifier instead.

## 8. Open questions for phase 1

1. **Dataset revisions.** Which revision of `image-text_bern-disputation-1528`
   is correct, what the 25.10 images are, whether the Schlussredaktion export
   (24.10) and the four medieval datasets have the same problem, and which
   revision the `pipeline.py eval` numbers used. Until then, family-1 held-out
   numbers in `docs/MODELS.md` are suspect, and target 1 should be re-framed
   (data repair + C4 retrain first, then weight edits as the comparison).
2. **Family 3 material.** Find or rebuild the shard_00 arrows the kraken+
   runs trained on, or drop these models as targets and use the
   `KRAKEN_PLUS_SPEC` trainer runs of October on the current corpus instead.
3. **Held-out sets** for `wsp.run heldout`: build `data/heldout/<name>.jsonl`
   from the `_eval` projects (BLB), the Hertwig eval pages (which revision?),
   and `german_test.arrow`; they must never be built into `data/practice/`.
4. **Trainer follow-ups** (not for this repo to change): `KRAKEN_PLUS_SPEC`
   still says `Cr255`; kraken 7.0.2's `save_model` model_type bug affects any
   tool that re-saves a CoreML model; the trainer records `revision: null` for
   HF datasets, so a run's data identity is not pinned.
5. First session proposal: `xix-blb-v1` (family 2, clean data, clean audit,
   C0 0.065 on practice) with the blank-bias and rare-row hypotheses, budget
   10, held-out = the BLB `_eval` projects through the gateway.

## 9. Reproduce §5 and §6

```bash
ssh asteraix && cd ~/Repo/weightspace && git checkout phase0
export TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
.venv/bin/python -m pytest -q tests                                   # 19 passed, 1 skipped
WSP_MODEL=models/kraken/kraken-german-xix-blb-v1.mlmodel WSP_PRACTICE=data/practice/blb-train-300.jsonl \
  .venv/bin/python -m pytest -q tests/asteraix                        # round-trip + smoke session
.venv/bin/python -m wsp.baselines --practice data/practice/blb-train-300.jsonl --out docs/baselines/blb-train-300.json \
  --model models/kraken/kraken-german-xix-v1.mlmodel models/kraken/kraken-german-xix-blb-v1.mlmodel
.venv/bin/python -m wsp.rewrite --src models/kraken/kraken-collapse-probe-131-cr255.mlmodel \
  --dst models/kraken/kraken-collapse-probe-131-cr1.mlmodel --practice data/practice/medieval-val-300.jsonl
.venv/bin/python -m wsp.audit overlap --practice data/practice/*-300.jsonl --out data/practice/overlap.json
.venv/bin/python -m wsp.stats diff models/kraken/kraken-bern-hertwig-v2.mlmodel models/kraken/kraken-bern-combined-v1.mlmodel
```

The practice lists are committed (`data/practice/*.jsonl`); the crops are
rebuilt by `python -m wsp.data build` with the sources in
`data/practice/README.md` (the Hertwig shard needs `huggingface_hub` and the
22.10 revision `28a16d55`; HF offline afterwards).

Not done: h1 was not touched (no handoff line); nothing was pushed to the
share. A full `ketos test` of `medieval-shard00-std` over the whole validation
arrow (20,535 lines, single-threaded) was started as a cross-check and stopped
unfinished after 50 minutes; the three 300-line samples above make the same
point.
