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
