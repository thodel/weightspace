# C4 control on the uncorrected 25.10 pairs — preregistration

Written 14.11.2026 (CET) before any run and before any held-out score of these models. Requested by Tobias ("run the uncorrected 25.10 control"). It is the control named in
`docs/C4_RETRAIN.md` §B.3 (2a) and in `docs/PROJECT_STATE.md` (decision 5): is it the **correction** of the Hertwig image/text pairs that gives the gain of `docs/RESULTS.md` R11, or is it
extra training at a small learning rate on any labelling of these pages? Status of this file: **part A (preregistration)**; part B (results) is appended after the runs; nothing in part A is changed after the first held-out score (a deviation goes into §B.6 with its reason).

## A.1 Question

`combined-v1` was trained (job `20261025T160407Z`) on Hertwig pages whose image was another page's image for 44 of 50 training pages (`docs/HERTWIG_REVISIONS.md`). C4 continued it on the corrected 22.10 pairs: Δ −0.55 (arm A, 1e-5) and −0.99 points (arm B, 5e-5), all on Hertwig lines.
The control continues it with **the same recipe, seeds, pages and text, but with the Hertwig images as the job saw them (wrong for 44 of 50 pages)**. The only difference to C4 is the pairing of Hertwig line image and text in training.

## A.2 What is held fixed, what changes

Fixed (identical to C4, `docs/C4_RETRAIN.md` A.2–A.3): subject `kraken-bern-combined-v1` (sha256 `35e7eae2…`), the 151 training pages and 17 validation pages of the job, the Schlussredaktion training pages (byte-identical arrow content to C4), **the validation set (including its 6 Hertwig pages from the 22.10 export, correct images)**, `ketos` 7.0.2 from this repo's venv, batch 16, cosine schedule, augmentation, NFD, `--resize union`, 5 fixed epochs, best-validation checkpoint, seeds 1, 2, 3, held-out `data/heldout/bern-eval.jsonl` (545 lines, 14 pages, scored once per model, CPU), the same scoring code (`scripts/c4_eval.py` logic).
Changed: the 44 Hertwig training pages are taken from the job's own data directory (`…/jobs/20261025T160407Z-kraken-bern-combined-v1/data/pages/`, read only): XML identical to the 22.10 XML, images as the job trained on them. Before use, a check counts how many of the 44 images differ byte-wise from the 22.10 images and that the line texts equal those of `docs/c4/train.jsonl` (reported in B.1). If none of the 44 images differs from the 22.10 images, the control is not informative and ends there; if only some differ, it runs as is and the count is reported (the trainer's own figure for the 50-page training project is 44 of 50, of which the job used 44 Hertwig pages).

Arms (counterparts of C4's, run in this order, one at a time):

| arm | learning rate | seeds | epochs | counterpart |
|---|---:|---|---:|---|
| **U-A** | 1e-5 | 1, 2, 3 | 5 | C4 arm A |
| U-B | 5e-5 | 1, 2, 3 | 5 | C4 arm B |

Route and limits as in C4: GPU card 0 only, ≥ 12 GB free checked before each run, one run at a time, card 1 never; `TMPDIR=~/atr-cache/tmp`; weights only in `~/atr-cache/wsp/c4u/`, not exported, registered, uploaded or served; nothing written to the research share, the registry, Hugging Face or the trainer's tree (the job directory is read only); trainer venvs untouched.

## A.3 Reading rule (fixed now)

Per arm, the mean Δ CER over its three seeds against the unedited `combined-v1` (19.33 %, all 545 lines), and the same on the 287 Hertwig lines (the primary place of any effect, since the Schlussredaktion side is unchanged and C4's Schlussredaktion Δ was ±0.04). The comparison is paired by seed against C4's arm of the same learning rate (C4 numbers from `docs/c4/results.json`, not re-run). Seed noise: the sd within C4 was 0.01–0.02 (all lines), the sd of the full seed null 0.18; the margin used below is 0.20 points (two standard errors of a three-seed mean under the null's sd), the same as in C4's rule.

Let Δ_U be the mean Δ of the U arm and Δ_C that of the C4 arm of the same learning rate, all 545 lines:

1. **The correction is what helps** if Δ_U − Δ_C ≥ +0.20 (the uncorrected arm gains at least 0.20 points less) **and** every U seed is above (less negative than) every C4 seed of that arm.
2. **Extra training explains the gain** if Δ_U ≤ −0.20 **and** |Δ_U − Δ_C| < 0.20: the uncorrected pairs give a gain of the same size within the margin.
3. **Wrong images help more than correct ones** if Δ_U − Δ_C ≤ −0.20 (stated for completeness, not expected).
4. **Wrong images hurt** (independent of the above) if Δ_U ≥ +0.20.
5. Anything else (e.g. Δ_U between −0.20 and +0.20 with Δ_U − Δ_C < 0.20) is **inconclusive**; it is reported as such and no claim is made.

The rule is applied to arm U-A first (the protocol's C4) and to U-B beside it, never in place of it. Page-clustered intervals (`wsp.metrics.bootstrap_delta`, clusters = page, 2,000 resamples) are shown, including for U against the C4 arm, but are not part of the rule (14 held-out pages).
Also reported: Hertwig-line and Schlussredaktion-line Δ, collapsed lines, practice CER on `bern-val`, per-epoch validation accuracy (from `train.log`), wall time, GPU memory.

## A.4 Not claimed

One subject, one dataset, three seeds per arm, five epochs, two learning rates. The control separates "correct pairing" from "any continued training" for this subject and recipe only. The wrong images are not random noise: 44 of 50 Hertwig training pages carry another page's image, so the control is a *fully broken* pairing for most Hertwig pages, not a mildly noisy one; a null for U says that broken pairs do not give the gain, not that mild label noise would not. The Schlussredaktion side and the validation set are correct in both runs, so the control cannot say anything about them.

---

# Part B — results (14.11.2026, CET)

Code: `scripts/c4u_prepare.py`, `c4u_train.sh`, `c4u_all.sh`, `c4u_eval.py`, `c4u_tables.py`. Data and results: `docs/c4u/` (`split.json`, `runs.json`, `results.json`, `results.hyp.json`, `tables.md`). Weights in `~/atr-cache/wsp/c4u/` only. Part A above is unchanged since its commit (`d24d0ce`), made before any run.

## B.1 Setup as run

The 44 Hertwig training pages were read from the job directory (1,772 lines): the line texts equal the corrected pages' in all 44, the image differs byte-wise from the 22.10 image in **40 of 44** (`docs/c4u/split.json`). Schlussredaktion training pages (107) and the validation arrow (17 pages, 514 lines) are C4's files. Training lines: 1,772 + 3,155 = 4,927 as in C4. Route as preregistered: `ketos` 7.0.2 from this repo's venv, GPU card 0 only, one run at a time, ≥ 12 GB free before each (the runner waited for another tenant's job to leave room; the first two runs started with 13.8 and 14.8 GB free, the rest with 35 GB), card 1 never touched, nothing outside `~/atr-cache/wsp/c4u/` and `docs/c4u/` written; the job directory was only read. `combined-v1` was re-scored and its hypotheses are identical to C4's (asserted in `c4u_eval.py`); the C4 runs were not re-run, their saved hypotheses were used for the pairing.

### T1. Runs

| run | lrate | seed | epochs | val. acc. per epoch (1–5) | kept (best val.) | wall | card 0 free before | card 0 used, peak (all tenants) |
|---|---:|---:|---:|---|---|---:|---:|---:|
| UA_seed1 | 1e-5 | 1 | 5 | 0.860 0.860 0.860 0.861 0.860 | 0.8606 (epoch 4) | 8 min 42 s | 13.8 GB | 44.4 GB |
| UA_seed2 | 1e-5 | 2 | 5 | 0.861 0.861 0.860 0.860 0.860 | 0.8608 (epoch 2) | 6 min 48 s | 14.8 GB | 41.0 GB |
| UA_seed3 | 1e-5 | 3 | 5 | 0.860 0.859 0.860 0.860 0.860 | 0.8601 (epoch 5) | 6 min 27 s | 35.8 GB | 22.0 GB |
| UB_seed1 | 5e-5 | 1 | 5 | 0.859 0.859 0.860 0.862 0.861 | 0.8617 (epoch 4) | 6 min 54 s | 35.8 GB | 19.7 GB |
| UB_seed2 | 5e-5 | 2 | 5 | 0.860 0.860 0.862 0.860 0.859 | 0.8621 (epoch 3) | 6 min 22 s | 35.4 GB | 25.8 GB |
| UB_seed3 | 5e-5 | 3 | 5 | 0.860 0.858 0.862 0.861 0.861 | 0.8623 (epoch 3) | 8 min 18 s | 36.4 GB | 38.2 GB |

### T2. Held-out `bern-eval` (545 lines, 14 pages): each model scored once, CPU

Δ in points of CER; interval = line-level / page-clustered 95 %. The C4 rows are copied from `docs/c4/results.json`.

| model | CER all | CER Hertwig | CER Schluss. | collapsed | Δ vs combined-v1 | Δ vs C4 arm, same seed | practice `bern-val` CER |
|---|---:|---:|---:|---:|---|---|---:|
| combined-v1 | 19.33 | 26.69 | 13.48 | 34 | — | — | 13.86 |
| UA_seed1 | 19.28 | 26.58 | 13.48 | 33 | -0.05 [-0.16, +0.06] / [-0.21, +0.09] | +0.49 [+0.27, +0.71] / [+0.09, +0.96] | 13.80 |
| UA_seed2 | 19.20 | 26.58 | 13.34 | 31 | -0.13 [-0.22, -0.04] / [-0.25, -0.01] | +0.44 [+0.21, +0.66] / [-0.03, +0.94] | 13.80 |
| UA_seed3 | 19.29 | 26.60 | 13.47 | 33 | -0.04 [-0.17, +0.09] / [-0.21, +0.12] | +0.51 [+0.31, +0.73] / [+0.10, +0.98] | 13.88 |
| UB_seed1 | 19.23 | 26.60 | 13.37 | 34 | -0.10 [-0.27, +0.08] / [-0.32, +0.14] | +0.89 [+0.59, +1.23] / [+0.24, +1.60] | 13.67 |
| UB_seed2 | 19.26 | 26.63 | 13.40 | 34 | -0.07 [-0.24, +0.10] / [-0.30, +0.16] | +0.94 [+0.65, +1.27] / [+0.31, +1.60] | 13.69 |
| UB_seed3 | 19.40 | 26.84 | 13.49 | 27 | +0.07 [-0.13, +0.28] / [-0.21, +0.35] | +1.05 [+0.74, +1.41] / [+0.33, +1.80] | 13.67 |
| A_seed1 | 18.79 | 25.44 | 13.50 | 28 | -0.54 [-0.77, -0.31] / [-1.06, -0.09] | — | 13.27 |
| A_seed2 | 18.77 | 25.33 | 13.55 | 28 | -0.56 [-0.79, -0.35] / [-1.09, -0.09] | — | 13.26 |
| A_seed3 | 18.77 | 25.39 | 13.52 | 26 | -0.56 [-0.79, -0.34] / [-1.06, -0.10] | — | 13.27 |
| B_seed1 | 18.34 | 24.49 | 13.46 | 27 | -0.99 [-1.34, -0.66] / [-1.72, -0.30] | — | 12.53 |
| B_seed2 | 18.31 | 24.50 | 13.40 | 26 | -1.02 [-1.37, -0.69] / [-1.74, -0.35] | — | 12.55 |
| B_seed3 | 18.35 | 24.39 | 13.55 | 24 | -0.98 [-1.34, -0.65] / [-1.70, -0.29] | — | 12.72 |

### T3. Arms: mean over three seeds (points) and the reading rule (A.3 of `docs/C4_UNCORRECTED.md`)

| arm | mean Δ (all) | per seed | Hertwig Δ | Schluss. Δ | C4 arm mean Δ (all) | Δ_U − Δ_C | mean collapsed | mean practice CER | rule |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| UA (1e-5) | -0.07 (sd 0.05) | -0.05 -0.13 -0.04 | -0.10 | -0.05 | -0.55 | +0.48 | 32.3 | 13.83 | 1: the correction is what helps |
| UB (5e-5) | -0.03 (sd 0.09) | -0.10 -0.07 +0.07 | +0.00 | -0.06 | -0.99 | +0.96 | 31.7 | 13.68 | 1: the correction is what helps |

## B.2 Answer, by the reading rule of A.3

- **Arm U-A (1e-5): rule 1, the correction is what helps.** Mean Δ −0.07 points (−0.05, −0.13, −0.04), against −0.55 for C4's arm A: Δ_U − Δ_C = +0.48 ≥ 0.20, and every U seed is above every C4 seed. Rule 2 does not apply (mean Δ > −0.20). On the Hertwig lines −0.10 (C4: −1.30); Schlussredaktion −0.05.
- **Arm U-B (5e-5): rule 1 as well.** Mean Δ −0.03 (−0.10, −0.07, +0.07), against −0.99: Δ_U − Δ_C = +0.96. Hertwig lines ±0.00 (C4: −2.22).
- **No harm either:** no U arm has a mean Δ ≥ +0.20 (rule 4 not met); the one seed above 0 (UB seed 3, +0.07, interval [−0.13, +0.28]) is inside the seed null's spread (sd 0.18).
- **Pairing against C4, same seed:** U − C4 = +0.44 to +0.51 (arm A) and +0.89 to +1.05 (arm B); line-level intervals exclude 0 in all six; the page-clustered ones exclude 0 in five of six (UA seed 2: [−0.03, +0.94]).
- **Practice and validation:** `bern-val` CER 13.86 → 13.83 (U-A), 13.68 (U-B), against 13.27 and 12.60 for C4. Per-epoch validation accuracy stays flat, 0.858–0.862 over the five epochs in every U run, while it rose monotonically from 0.862–0.867 in C4 (`docs/C4_RETRAIN.md` T1). Collapsed held-out lines: 34 → 31–33 (U-A), 27–34 (U-B), against 24–28 in C4.

## B.3 What this does and does not show

1. For this subject and recipe, **continued training at a small learning rate on the same pages and texts with the wrong images gives no detectable gain** (−0.07 and −0.03 points, within the seed null's sd of 0.18 for a single run and below the 0.20 margin for the mean). The C4 gain therefore does not come from the extra epochs, the cosine schedule or the Schlussredaktion half of the data alone; it needs the correctly paired Hertwig lines, and it appears on the Hertwig lines only (C4 −1.30 and −2.22; U −0.10 and 0.00).
2. The U arms did not make the model worse either: the broken pairs are not learned (flat validation accuracy), and held-out CER stays where `combined-v1` was. That fits a model that had already been trained on these pairs: the wrong-image lines give a gradient it has mostly converged on.
3. **What it does not show:** (a) the pairing is *fully* broken for 40 of 44 pages, so the control separates correct from broken pairs, not correct from mildly noisy ones; (b) one subject, one dataset, three seeds, five epochs, two learning rates; (c) the Schlussredaktion half and the validation set are correct in both runs, so nothing is said about them beyond "unchanged"; (d) a control with another schedule (more epochs at 1e-5, or constant rate) on the **corrected** pages was not run, so "more training at a small rate cannot give the gain" is shown only for broken pairs; (e) the comparison with C4 is between two sets of runs with the same seeds but different data, not a re-run of both in one batch.
4. For the claim in the manuscript: the C4 gain on Hertwig lines can be attributed to the correction of the pairs (under (a) and (d)), not to training as such. That strengthens the statement of `docs/C4_RETRAIN.md` §B.4 and removes the open item (2a) of §B.3.

## B.4 For the manuscript (draft sentence)

*A control that repeated the retraining with the Hertwig images as the original job had them (wrong for 44 of 50 pages; the line texts unchanged) gave no gain (mean Δ −0.07 points at 1e-5, −0.03 at 5e-5, against −0.55 and −0.99 on the corrected pairs; three seeds each, the same seeds), and no Hertwig gain at all (−0.10 and 0.00 against −1.30 and −2.22). The gain of the retraining therefore depends on the correction of the pairs and not on the extra training alone.*

## B.5 Deviations from the preregistration

None in design, data, arms, seeds, epochs or rule. Minor: (i) the count of differing images is 40 of 44 (A.2 anticipated "some"); (ii) the first two runs started with 13.8 and 14.8 GB free and ran while another tenant was still on the card (wall 8 min 42 s and 6 min 48 s; peak card use 44.4 and 41.0 GB, the whole card including the neighbour); (iii) card 0 memory in T1 is the whole card, this session's own footprint was not isolated.
