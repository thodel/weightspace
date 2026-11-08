# C4 retrain of `kraken-bern-combined-v1` on the corrected data — preregistration

Written 08.11.2026 (CET) before any run and before any held-out score. Requested by the planning session with Tobias's OK ("go for c4 retrain");
the control is C4 of `docs/PROTOCOL.md` §4. Status of this file: **preregistration (part A)**; results are added below as part B after the runs, and
nothing in part A is changed after the first held-out score (a deviation goes into §B.6 with its reason).

## A.1 Question

Does a short continued training of `kraken-bern-combined-v1` on the **corrected** data (the Hertwig 22.10.2026 export instead of the 25.10 export, in which 44 of the 50
training pages carry another page's image, `docs/HERTWIG_REVISIONS.md`) give a held-out gain on `data/heldout/bern-eval.jsonl` that is larger, equal or smaller than

- the quarter-step interpolation edit toward `kraken-bern-schlussredaktion-v1` of `docs/CONTROLS_REPORT.md` §6: Δ −0.39 points, line interval [−0.69, −0.08], page-clustered [−0.84, +0.04];
- the seed null of `docs/SEED_NULL.md` (sd 0.18 points over seeds, all 545 lines)?

## A.2 Subject, data, validation, held-out

- **Subject:** `kraken-bern-combined-v1`, `models/kraken/kraken-bern-combined-v1.mlmodel`, sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`.
- **Training data:** the training side of the two projects of its job (`20261025T160407Z-kraken-bern-combined-v1`, `data/pages_train.lst`: 151 pages, 44 Hertwig `docId` 13922499 and 107 Schlussredaktion `docId` 13922579, identified by `docId` and `pageNr` read from the job's page XML),
  with the Hertwig pages taken from the **22.10 export** (`28a16d55`, parquet sha256 `566f5bd1…`, correct images) and the Schlussredaktion pages from the 25.10 export (byte-identical to 24.10; `docs/HERTWIG_REVISIONS.md`). Every line with a baseline and text is used; the count is recorded in B.1.
- **Validation:** the trainer's 17 validation pages of that job (`pages_val.lst`; 6 Hertwig from the 22.10 export, 11 Schlussredaktion) = `data/practice/bern-val.jsonl` of `docs/CONTROLS_REPORT.md` §3 (513 lines). Used by `ketos` for the per-epoch validation accuracy and, as the trainer does, for choosing the checkpoint (see A.3).
- **Held-out:** `data/heldout/bern-eval.jsonl` (545 lines, 14 pages: 6 Hertwig test, 8 Schlussredaktion test; sha256 `f7b024e0…`). **Each model is scored once**, on CPU, after training; no choice (epoch, learning rate, seed) is made on it.
- **Audit:** `python -m wsp.audit` on the training and validation line lists (`docs/c4/`): none of the 14 held-out pages, none of the 17 validation pages in training; the existing `overlap-bern-val.json` is the check for the validation pages against the held-out pages.

## A.3 Arms

| arm | learning rate | seeds | epochs | role |
|---|---:|---|---:|---|
| **A (C4 as in the protocol)** | 1e-5 | 1, 2, 3 | 5, fixed | primary |
| B (stated now, secondary) | 5e-5 | 1, 2, 3 | 5, fixed | run only after arm A is complete; reported beside A, never in place of it |

Everything else identical to the trainer's job of `combined-v1` (`job.json`): batch 16, AdamW (kraken default), cosine schedule with kraken's default `--cos-max`, augmentation on, NFD, whitespace normalised, `--resize union`, `--weights-format coreml`, kraken 7.0.2, starting from the subject; `--quit fixed --epochs 5`.
The weights kept per run are the **best-validation checkpoint** (`ketos`' `best_*.mlmodel`, the trainer's own rule); its epoch and the last epoch's validation accuracy are recorded. No other selection.

Route: `ketos train` from this repo's venv (not the trainer's job mechanism); **GPU card 0 only**, one run at a time, started only when `nvidia-smi` shows ≥ 12 GB free there (checked in `scripts/c4_train.sh`), card 1 never; `TMPDIR=~/atr-cache/tmp`; weights stay in `~/atr-cache/wsp/c4/` and are not exported, registered, uploaded or served; nothing is written to the research share, the registry, Hugging Face or the trainer's tree; the trainer's venvs are not touched.

## A.4 What is reported, and how the answer is read

For each model (3 seeds × arm A; 3 × arm B), against the unedited `combined-v1` (held-out 19.33 % CER, practice `bern-val` 13.86 %) **and** against the interpolation edit (rebuilt exactly as in `docs/CONTROLS_REPORT.md` §6: blank −0.3, the 32 non-output tensors α = 0.25 toward `schlussredaktion-v1`; re-scored once on CPU, deterministic):

- held-out CER over all 545 lines, over the 287 Hertwig and the 258 Schlussredaktion lines; collapsed lines (`wsp.metrics.is_collapsed`); WER;
- paired Δ CER with the line-level 95 % interval and the **page-clustered** 95 % interval (`wsp.metrics.bootstrap_delta` with clusters = page; 2,000 resamples, the repository's seed);
- practice (validation) CER on `bern-val`, per-epoch validation accuracy (from `train.log`), wall time, GPU card memory (card total, neighbours included, sampled every 30 s), card-0 free memory at the start.

**Reading rule (fixed now).** The effect of an arm is the mean Δ CER over its three seeds (all 545 lines). Seed noise: sd 0.18 (SEED_NULL), so the standard error of a three-seed mean is about 0.10. Then:
(1) a C4 gain is *detectable* if the mean Δ ≤ −0.20 (two standard errors) **and** it is negative for all three seeds;
(2) it is *larger than* the interpolation edit if the mean Δ ≤ −0.57 (−0.39 minus one seed sd), *equal* if between −0.57 and −0.21, *smaller* if > −0.21 (a gain, if any, below the interpolation's by more than one seed sd);
(3) a *loss* if the mean Δ ≥ +0.20.
The page-clustered interval of each model is shown but is not part of the rule: with 14 held-out pages it will be wide (the interpolation edit's was [−0.84, +0.04]), and a rule that required it to exclude 0 would decide nothing. The statement made in the report is the rule's, together with the intervals.

## A.5 Not claimed

One subject, one dataset, 3 seeds, 5 epochs at one or two learning rates. A gain here says continued gradient training on corrected data helps a model trained on noisy pairs, not that weight edits are worse. A null says the 5-epoch budget at 1e-5 is too small, not that retraining cannot repair the noise. The comparison with the interpolation edit is of point estimates against seed noise; the two were not run on the same random draw.
