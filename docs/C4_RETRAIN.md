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

---

# Part B — results (08.11.2026, CET)

Code: `scripts/c4_prepare.py`, `c4_train.sh`, `c4_all.sh`, `c4_eval.py`, `c4_tables.py`. Data: `docs/c4/` (`split.json`, `train.jsonl`, `val.jsonl`, `audit*.json`, `runs.json`, `results.json`, `results.hyp.json` with every hypothesis, `tables.md`). Weights in `~/atr-cache/wsp/c4/` only.
Part A above is unchanged since its commit (`2ea91a3`), made before any run.

## B.1 Setup as run

Training data: 151 pages (44 Hertwig from the 22.10 export, 107 Schlussredaktion), **4,927 lines**; validation: the 17 trainer pages, 514 lines in the `ketos` arrow (`bern-val` for scoring has 513: one degenerate line is skipped there). 4,927 + 514 = 5,441, the trainer's `lines_written` for the job.
Audit (`wsp.audit overlap`, `audit-validation.json`): training and validation lists clean; `bern-val` clean against the training pages and the held-out pages. Route as preregistered: `ketos` 7.0.2 from this repo's venv, GPU card 0, one run at a time, ≥ 12 GB free before each (34 GB each time), card 1 never touched, nothing outside `~/atr-cache/wsp/c4/` and `docs/c4/` written. Six runs (arm A lr 1e-5, arm B lr 5e-5, seeds 1–3), 5 epochs each, 2 to 5 minutes (the 2 min 41 s run found the card less busy).

### T1. Runs

| run | lrate | seed | epochs | val. acc. per epoch (1–5) | kept (best val.) | wall | card 0 free before | card 0 used, peak (all tenants) |
|---|---:|---:|---:|---|---|---:|---:|---:|
| A_seed1 | 1e-5 | 1 | 5 | 0.863 0.864 0.865 0.865 0.866 | 0.8657 (epoch 5) | 4 min 45 s | 34.3 GB | 20.8 GB |
| A_seed2 | 1e-5 | 2 | 5 | 0.863 0.864 0.865 0.866 0.866 | 0.8660 (epoch 5) | 2 min 41 s | 34.3 GB | 26.8 GB |
| A_seed3 | 1e-5 | 3 | 5 | 0.862 0.864 0.864 0.865 0.866 | 0.8657 (epoch 5) | 4 min 47 s | 34.4 GB | 23.4 GB |
| B_seed1 | 5e-5 | 1 | 5 | 0.867 0.870 0.872 0.873 0.874 | 0.8735 (epoch 5) | 4 min 47 s | 34.4 GB | 20.8 GB |
| B_seed2 | 5e-5 | 2 | 5 | 0.866 0.870 0.871 0.872 0.873 | 0.8726 (epoch 5) | 4 min 49 s | 34.3 GB | 26.8 GB |
| B_seed3 | 5e-5 | 3 | 5 | 0.867 0.869 0.871 0.871 0.872 | 0.8718 (epoch 5) | 4 min 49 s | 34.3 GB | 23.6 GB |

### T2. Held-out `bern-eval` (545 lines, 14 pages): each model scored once, CPU

Δ in points of CER; interval = line-level / page-clustered 95 %.

| model | CER all | CER Hertwig | CER Schluss. | collapsed | WER all | Δ vs combined-v1 | Δ vs interpolation edit | practice `bern-val` CER | collapsed (val) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| combined-v1 | 19.33 | 26.69 | 13.48 | 34 | 51.89 | — | — | 13.86 | 27 |
| interp | 18.94 | 26.19 | 13.18 | 26 | 51.43 | -0.39 [-0.69, -0.08] / [-0.84, +0.04] | — | 12.39 | 20 |
| A_seed1 | 18.79 | 25.44 | 13.50 | 28 | 51.08 | -0.54 [-0.77, -0.31] / [-1.06, -0.09] | -0.15 [-0.42, +0.10] / [-0.63, +0.29] | 13.27 | 23 |
| A_seed2 | 18.77 | 25.33 | 13.55 | 28 | 51.11 | -0.56 [-0.79, -0.35] / [-1.09, -0.09] | -0.18 [-0.44, +0.08] / [-0.72, +0.29] | 13.26 | 23 |
| A_seed3 | 18.77 | 25.39 | 13.52 | 26 | 50.88 | -0.56 [-0.79, -0.34] / [-1.06, -0.10] | -0.17 [-0.45, +0.09] / [-0.68, +0.28] | 13.27 | 23 |
| B_seed1 | 18.34 | 24.49 | 13.46 | 27 | 49.93 | -0.99 [-1.34, -0.66] / [-1.72, -0.30] | -0.60 [-0.90, -0.30] / [-1.32, +0.04] | 12.53 | 18 |
| B_seed2 | 18.31 | 24.50 | 13.40 | 26 | 49.84 | -1.02 [-1.37, -0.69] / [-1.74, -0.35] | -0.63 [-0.92, -0.34] / [-1.34, -0.01] | 12.55 | 19 |
| B_seed3 | 18.35 | 24.39 | 13.55 | 24 | 50.13 | -0.98 [-1.34, -0.65] / [-1.70, -0.29] | -0.59 [-0.90, -0.30] / [-1.33, +0.09] | 12.72 | 19 |

### T3. Arms: mean over three seeds (points)

| arm | mean Δ vs combined-v1 (all) | per seed | Hertwig Δ | Schluss. Δ | mean collapsed | mean practice CER | reading rule (A.4) |
|---|---:|---|---:|---:|---:|---:|---|
| A (1e-5) | -0.55 (sd 0.01) | -0.54 -0.56 -0.56 | -1.30 | +0.04 | 27.3 | 13.27 | detectable gain; equal to the interpolation edit (−0.39) |
| B (5e-5) | -0.99 (sd 0.02) | -0.99 -1.02 -0.98 | -2.22 | -0.01 | 25.7 | 12.60 | detectable gain; larger than the interpolation edit (−0.39) |

combined-v1: held-out 19.33 %, collapsed 34, practice 13.86 %. Interpolation edit rebuilt and re-scored: 18.94 % (Δ -0.39), collapsed 26, practice 12.39 %.

## B.2 Answer, by the reading rule of A.4

- **Arm A (C4 as in the protocol, lr 1e-5): a detectable gain, equal to the interpolation edit.** Mean Δ −0.55 points (−0.54, −0.56, −0.56) against `combined-v1`; −0.57 < −0.55 ≤ −0.21 so "equal" by the rule (the interpolation edit gave −0.39; the three runs differ from it by −0.15 to −0.18 points, line interval [−0.45, +0.10], page-clustered [−0.72, +0.29]: not distinguishable). All three seeds negative, and the page-clustered interval of each run against `combined-v1` excludes 0 ([−1.06, −0.09] to [−1.09, −0.09]), which the interpolation edit's did not ([−0.84, +0.04]).
- **Arm B (lr 5e-5, preregistered as secondary): larger than the interpolation edit.** Mean Δ −0.99 (−0.99, −1.02, −0.98); page-clustered interval against `combined-v1` [−1.74, −0.29]; against the interpolation edit −0.59 to −0.63 with page interval reaching +0.09 to −0.01 (touching 0).
- **Against the seed null (sd 0.18):** both arms are many seed standard deviations from 0 in the null's terms. But the sd *within* C4 is 0.01–0.02, an order of magnitude below the null's: here the seed changes only batch order, augmentation and dropout of a model that starts trained and moves by a small step, whereas the null re-trained from the parent and the rows new to the codec started from random values. The null's 0.18 is the right noise for retraining from the parent; for C4 the noise between seeds is far smaller and the preregistered thresholds were conservative.
- **Where the gain is.** Entirely on the Hertwig pages: −1.30 (A) and −2.22 (B) points on the 287 Hertwig lines, ±0.0 on the 258 Schlussredaktion lines (+0.04, −0.01). That is the part of the data that was corrected. The interpolation edit moved the other way round (Schlussredaktion toward its sibling).
- **Collapsed lines and practice.** Collapsed lines of the held-out set: 34 → 26–28 (A), 24–27 (B); the interpolation edit 26. Practice CER (`bern-val`): 13.86 → 13.27 (A), 12.60 (B) on average, the interpolation edit 12.39. Per-epoch validation accuracy rises monotonically over the 5 epochs in every run and is still rising at epoch 5 (T1); the kept checkpoint is epoch 5 each time.

## B.3 What this does and does not show

1. A short gradient retrain on the corrected pairs gives a held-out gain at least as large as the project's only weight-edit gain, at a lower cost of judgement (no programmer, no keep rule) and with a larger effect when the step is larger (arm B). It is a **negative result for the weight-edit programme as a repair method on this subject**: plain retraining on corrected data does what the interpolation edit does, and more.
2. **What it does not show:** (a) that the *correction* is what helps. No control continues training on the **uncorrected** 25.10 pairs, or on the corrected pages with a different learning-rate schedule; the gain on Hertwig lines is consistent with the correction, but more epochs at a small rate on any consistent labelling of these pages might also lower the CER (`combined-v1` was stopped by the trainer's early stopping on noisy validation pairs). That control is the next one to run if the claim "correction" is to be made. (b) That the gain would hold on other pages: 14 held-out pages, 6 of them Hertwig; the page-clustered intervals are wide (a width of about 1.0 to 1.4 points), and arm A's lower bounds are close to 0. (c) That 5 epochs is the optimum: validation accuracy is still rising, and arm B's larger step was larger in gain; no larger step was run. (d) Anything about other models or families.
3. The comparison with the interpolation edit is of point estimates; the two were not run on the same random draw, but the edit is deterministic and was re-scored here (18.94 %, same as in `docs/CONTROLS_REPORT.md` §6).

## B.4 For the manuscript (draft sentence)

*A short continued training of `combined-v1` on the corrected pairs (5 epochs, learning rate 1e-5, three seeds) lowered the held-out CER by 0.55 points on average (all three runs −0.54 to −0.56; page-clustered 95 % intervals about [−1.1, −0.1]), against 0.39 for the quarter-step interpolation toward `schlussredaktion-v1`, and by 0.99 points at a learning rate of 5e-5. The gain lies entirely on the Hertwig pages, the ones whose pairs had been corrected (−1.3 and −2.2 points); the Schlussredaktion pages do not change. The retraining therefore does at least what the one weight edit that survived our controls does, without any search over edits; whether the correction or only the additional training is responsible was not tested.*

## B.5 Deviations from the preregistration

None in design, data, arms, seeds, epochs or rule. Minor: (i) the validation arrow has 514 lines and `bern-val` 513 (one degenerate line skipped by the repository's cutter, as already recorded); (ii) the interpolation edit was re-scored, as announced; (iii) the wall times differ between runs by the card's load from neighbours (2 min 41 s for one run, 4 min 45–49 s for the others); (iv) card 0 memory in T1 is the whole card, neighbours included; this session's own footprint was not isolated.
