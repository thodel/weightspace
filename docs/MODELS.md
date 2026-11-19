# Model inventory (state 06.11.2026)

Every kraken model the gateway on idhefix serves is listed in
`serving-atr-inference/config/models.yaml`. 37 of the kraken models are third-party Zenodo
models (CATMuS, TRIDIS, CREMMA, FoNDUE, OpenITI, …) and are *not* ours (`docs/zoo/gateway_zenodo.json`;
corrected 07.11.2026, the earlier count of 42 was not reproduced by the survey). The
in-house models are the six below (two families; the public parent `kraken-cursive_inzigkofen` of family 1 is third-party and is the seventh model in the
family's lineage, not an in-house one). The model family compared in the article is these six plus the 37 Zenodo models: 43 files (`docs/ZOO_SURVEY.md`, `docs/MODEL_FAMILY.md`).
Weights live on the research share
`/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/<id>/<id>.mlmodel`
(reachable from tei and asteraix, not from h1). All are CoreML `.mlmodel`,
~16 MB, loaded by the serving engine with `kraken.lib.models.load_any` under
kraken 7.0.2.

## Public identifiers (checked 19.11.2026)

The research-share path above is internal. Where a model is published, this is the identifier to cite.
"Same file" means the published weights have the sha256 of the copy used in this project.

| model | published as | public? | licence | note |
|---|---|---|---|---|
| `kraken-bern-hertwig-v1` | — | not published | — | weights only on the research share |
| `kraken-bern-hertwig-v2` | — | not published | — | |
| `kraken-bern-schlussredaktion-v1` | [`dh-unibe/kraken-bern-schlussredaktion-v1`](https://huggingface.co/dh-unibe/kraken-bern-schlussredaktion-v1) | public | MIT (model card) | same file, sha256 `d26eb02dfaf89f8f…` |
| `kraken-bern-combined-v1` | — | not published | — | trained on the mismatched 25.10 export (below) |
| `kraken-german-xix-v1` | [`dh-unibe/kraken-german-xix-v1`](https://huggingface.co/dh-unibe/kraken-german-xix-v1) | public | MIT (model card) | same file, sha256 `7af3d4a0eb1a778c…` |
| `kraken-german-xix-blb-v1` | `dh-unibe/kraken-german-xix-blb-v1` | private | MIT (model card) | |
| family 3 checkpoints, collapse probes | — | not published | — | `thodel/kraken-plus-medieval` is public but empty |

Third-party parents (not ours; each fine-tune above is a derivative and has to credit them):

| model | DOI | creators | licence |
|---|---|---|---|
| `kraken-cursive_inzigkofen` (parent of family 1) | [10.5281/zenodo.18207767](https://doi.org/10.5281/zenodo.18207767) | Nicole Eichenberger | CC BY 4.0 |
| CATMuS Medieval (parent of `cursive_inzigkofen`) | [10.5281/zenodo.15030337](https://doi.org/10.5281/zenodo.15030337) | Ariane Pinche, Thibault Clérice, Malamatenia Vlachou-Efstathiou et al. | CC BY 4.0 |
| `kraken-bohemian_19th` (parent of family 2) | [10.5281/zenodo.11673242](https://doi.org/10.5281/zenodo.11673242) | Daniel Baránek | CC BY 4.0 |

The 37 other third-party models of the model family carry their Zenodo DOIs in `docs/zoo/gateway_zenodo.json`
(field `zenodo_id`). The datasets the models were trained on are listed with their identifiers in the
`provenance` block of each `data/*/*.meta.json`.

## Family 1 — Bern disputation 1528 (medieval/early modern German, hands of the Hertwig volume)

| id | base | trained on | reported CER | notes |
|---|---|---|---|---|
| `kraken-bern-hertwig-v1` | `kraken-cursive_inzigkofen` (Zenodo 10.5281/zenodo.18207767, itself a fine-tune of CATMuS Medieval 15030337) | `dh-unibe/image-text_bern-disputation-1528`, Hertwig project, 1,786 lines | trainer test 0.271 | lrate 1e-4, 60 epochs — under-trained (`kraken_train` tool doc) |
| `kraken-bern-hertwig-v2` | same | same | trainer test **0.233**; line-CER in `pipeline.py eval` 0.371 (41 of 287 GT lines got no text: segmentation) | lrate 5e-4, 150 planned, stopped at 91 |
| `kraken-bern-schlussredaktion-v1` | `trained:kraken-bern-hertwig-v2` | 1 project, 118 pages, 3,457 lines | held-out 16.7 % | |
| `kraken-bern-combined-v1` | `trained:kraken-bern-hertwig-v2` | 2 projects, 168 pages, 5,441 lines, **from the 25.10 export: 44 of the 50 Hertwig pages carry another page's image** | held-out 28.8 % | worse than its parent on its own held-out. Target 1, re-framed as repairing a model trained on label noise (docs/HERTWIG_REVISIONS.md) |

Scratch runs on UBELIX (`20261105T193111Z-kraken-bern-hertwig-scratch-ubelix-v2`, `KRAKEN_PLUS_SPEC` from scratch on the same 1,786 lines) exist as comparators; check `ubelix_record` for their CER.

## Family 2 — 19th-century German Kurrent (Laßberg correspondence, BLB)

| id | base | trained on | reported CER | notes |
|---|---|---|---|---|
| `kraken-german-xix-v1` | `kraken-bohemian_19th` (Zenodo 10.5281/zenodo.11673242). Confirmed 13.11.2026 from the trainer's record, `jobs/20261007T131819Z-kraken-german-xix-v1/job.json` `request.base_model` (`resize: union`, lrate 1e-4, cosine, 15,527 pages); the weights are 0.70 from it (docs/ZOO_SURVEY.md) | Kurrent XIX corpora | 11.18 % (trainer, document-level held-out, `HELDOUT_BOUNDARY.md` says clean) | the production 19th-c. kraken model |
| `kraken-german-xix-blb-v1` | `trained:kraken-german-xix-v1` | `dh-unibe/image-text_lassberg-blb-gt`, 4 handwriting projects (lassberg, korrespondenten, unbestimmt, manuskripte; drucke left out), object-level `_eval` projects held out | held-out **8.7 %** | spec `agentic_historian/training/specs/blb-kraken-v1.json`: lrate 1e-4 cosine, 40 epochs early-stop, `resize: union` (codec may have grown — matters for merging), NFD |

Both xix models have already read the 276 hand-corrected Transkribus pages
(run `tk-gt-kraken-blb`) and the four `blb-eval-*` runs (105/22/23/7 pages),
so `score_ground_truth` gives a per-page baseline without a new batch.

## Family 3 — kraken+ (medieval German, from scratch) — **dropped as a target (06.11.2026)**

The three archived checkpoints of 01.09.2026 (`kraken-plus-shard00`, `kraken-medieval-shard00-v1`, `kraken-medieval-shard00-std`) cannot be measured against the data they were trained and validated on: their run folders under `archive/idhefix-2026-09/runs/` hold only `train_bin.lst` / `val_bin.lst`, which point at `~/atr-cache/arrows/shard_00.arrow` and `val_clean.arrow` on idhefix, and neither those arrows nor the page list they were compiled from were archived. On the arrows that exist (`german_val`, `sweep_train` of `german-medieval-v1`) they read at 0.62–0.72 CER, worse than the out-of-domain `xix-blb-v1`. Decision: no sessions on these checkpoints. What stays: the lossless `Cr255 → Cr1` rewrite (docs/RESULTS.md R1), which used the collapse-probe checkpoint only as a vehicle. A future family 3 would be a `KRAKEN_PLUS_SPEC` run of the trainer on the current corpus, with its arrows pinned.


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
| `blb-eval-hand` (`data/heldout/`) | the handwriting lines of the four `_eval` projects, 2,272 lines, chosen by data properties only | `scripts/build_blb_eval_hand.py` | **held-out for family 2 from 07.11.2026 on** |
| `lassberg-blb-gt` `_eval` projects | object-level held-out pages of the BLB set | `datasets:lassberg-blb-gt/<project>_eval` on tei; HF `dh-unibe/image-text_lassberg-blb-gt` (private) | **held-out** for family 2 |
| 276 Transkribus GT pages (`gt-keys.txt`) | hand-corrected Laßberg pages | `GT_ROOT` on tei; run `tk-gt-kraken-blb` | baseline + practice for family 2 (check overlap with blb-gt train side first) |
| `bern-disputation-1528` eval pages / 287 GT lines | Hertwig | HF (private); `pipeline.py eval --doc 13922499 --pages 8,17,26,39,48,55`; **only the 22.10 export (`28a16d55`) pairs them correctly** | held-out for family 1, with the 8 Schlussredaktion test pages: `data/heldout/bern-eval.jsonl` |
| `german-medieval-v1` test (695 pages, 200 documents) | medieval German, 4 corpora | `/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-medieval-v1/`, `~/atr-cache/arrows/german_test.arrow` on asteraix | held-out for family 3 and the CATMuS lineage |
| `dh-unibe/image-text_*-lines` (public parquet) | line crops + text | HF | cheap **practice** sets that run on CPU anywhere |

TrOCR models (transformer line recognisers on the hub, dh-unibe and third-party): `docs/MODELS_TROCR.md` (11.11.2026).
