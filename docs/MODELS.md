# Model inventory (state 06.11.2026)

Every kraken model the gateway on idhefix serves is listed in
`serving-atr-inference/config/models.yaml`. 42 of them are third-party Zenodo
models (CATMuS, TRIDIS, CREMMA, FoNDUE, OpenITI, …) and are *not* ours. The
in-house models are the seven below. Weights live on the research share
`/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/<id>/<id>.mlmodel`
(reachable from tei and asteraix, not from h1). All are CoreML `.mlmodel`,
~16 MB, loaded by the serving engine with `kraken.lib.models.load_any` under
kraken 7.0.2.

## Family 1 — Bern disputation 1528 (medieval/early modern German, hands of the Hertwig volume)

| id | base | trained on | reported CER | notes |
|---|---|---|---|---|
| `kraken-bern-hertwig-v1` | `kraken-cursive_inzigkofen` (Zenodo 10.5281/zenodo.18207767, itself a fine-tune of CATMuS Medieval 15030337) | `dh-unibe/image-text_bern-disputation-1528`, Hertwig project, 1,786 lines | trainer test 0.271 | lrate 1e-4, 60 epochs — under-trained (`kraken_train` tool doc) |
| `kraken-bern-hertwig-v2` | same | same | trainer test **0.233**; line-CER in `pipeline.py eval` 0.371 (41 of 287 GT lines got no text: segmentation) | lrate 5e-4, 150 planned, stopped at 91 |
| `kraken-bern-schlussredaktion-v1` | `trained:kraken-bern-hertwig-v2` | 1 project, 118 pages, 3,457 lines | held-out 16.7 % | |
| `kraken-bern-combined-v1` | `trained:kraken-bern-hertwig-v2` | 2 projects, 168 pages, 5,441 lines | held-out 28.8 % | worse than its parent on its own held-out — first candidate for a weight-space reading |

Scratch runs on UBELIX (`20261105T193111Z-kraken-bern-hertwig-scratch-ubelix-v2`, `KRAKEN_PLUS_SPEC` from scratch on the same 1,786 lines) exist as comparators; check `ubelix_record` for their CER.

## Family 2 — 19th-century German Kurrent (Laßberg correspondence, BLB)

| id | base | trained on | reported CER | notes |
|---|---|---|---|---|
| `kraken-german-xix-v1` | **to confirm** (job `20261007T131819Z-kraken-german-xix-v1`, 15,527 pages; ask the trainer's record) | Kurrent XIX corpora | 11.18 % (trainer, document-level held-out, `HELDOUT_BOUNDARY.md` says clean) | the production 19th-c. kraken model |
| `kraken-german-xix-blb-v1` | `trained:kraken-german-xix-v1` | `dh-unibe/image-text_lassberg-blb-gt`, 4 handwriting projects (lassberg, korrespondenten, unbestimmt, manuskripte; drucke left out), object-level `_eval` projects held out | held-out **8.7 %** | spec `agentic_historian/training/specs/blb-kraken-v1.json`: lrate 1e-4 cosine, 40 epochs early-stop, `resize: union` (codec may have grown — matters for merging), NFD |

Both xix models have already read the 276 hand-corrected Transkribus pages
(run `tk-gt-kraken-blb`) and the four `blb-eval-*` runs (105/22/23/7 pages),
so `score_ground_truth` gives a per-page baseline without a new batch.

## Family 3 — kraken+ (medieval German, from scratch)

| id | where | notes |
|---|---|---|
| `thodel/kraken-plus-medieval` (HF) | repo exists, **empty** (only `.gitattributes`) | |
| kraken+ runs of 2026-09-01 on asteraix | `serving-atr-inference/docs/KRAKEN_PLUS.md` | three runs on `shard_00`: kraken+ `Cr1,1,85` test CER 0.1655, kraken default h120 0.1335, kraken+ without last layer 0.1812; held-out 6,186 pages / 35 documents |

`KRAKEN_PLUS_SPEC` (`training-atr-models/src/atr_training/contracts.py`):

```
[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2
 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]
```

Known weight-space fact (measured in `KRAKEN_PLUS.md`): the trailing
`Cr255,1,85,1,1` sits on a one-row feature map, so 11,097,600 of 15,193,853
parameters are padding-only and never receive a gradient; only 43,520 (512 × 85)
act. `Cr1,1,85,1,1` is the same function with 11 M fewer weights. That is a
lossless weight-space edit and the natural first exercise for the harness.

## Ground truth and evaluation sets

| set | what | where | use |
|---|---|---|---|
| `lassberg-blb-gt` `_eval` projects | object-level held-out pages of the BLB set | `datasets:lassberg-blb-gt/<project>_eval` on tei; HF `dh-unibe/image-text_lassberg-blb-gt` (private) | **held-out** for family 2 |
| 276 Transkribus GT pages (`gt-keys.txt`) | hand-corrected Laßberg pages | `GT_ROOT` on tei; run `tk-gt-kraken-blb` | baseline + practice for family 2 (check overlap with blb-gt train side first) |
| `bern-disputation-1528` eval pages / 287 GT lines | Hertwig | HF (private); `pipeline.py eval --doc 13922499 --pages 8,17,26,39,48,55` | held-out for family 1 |
| `german-medieval-v1` test (695 pages, 200 documents) | medieval German, 4 corpora | `/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-medieval-v1/`, `~/atr-cache/arrows/german_test.arrow` on asteraix | held-out for family 3 and the CATMuS lineage |
| `dh-unibe/image-text_*-lines` (public parquet) | line crops + text | HF | cheap **practice** sets that run on CPU anywhere |
