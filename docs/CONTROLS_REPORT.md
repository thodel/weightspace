# Controls report — planted defects and validation-based practice (items 2 and 3 of docs/BRIEF_ARTICLE_EXPERIMENTS.md)

Asteraix, 07.11.2026, branch `claude/article-controls-8803eb` on `origin/main` (07f55d7). CPU only, kraken 7.0.2, no retraining, nothing written to the share, registry, Hugging Face or the trainer. Run folders: `runs/20261107-planted-*` (32), `runs/20261107-val-*` (8), packs and programs in `runs/planted/` and `runs/validation/`. Aggregates: `docs/controls/*.json|md`. Tests: 44 passed, 2 skipped.

**In four sentences.** (1) The method *can* find and repair planted defects: of seven real corruptions of `xix-blb-v1`, the programmer with passes (a), the blind programmer (b) and a hand-written repair list (c) each brought the held-out CER back to within 0.22 points of the uncorrupted model on every corruption that did damage of 1 point or more (6 of 6; 19 of the 19 repaired cells, 10 of them to within 0.01 point), so "no defect found" on the real models carries information. (2) But these defects are of the kind the diff against the parent shows at once (a row with cosine −0.99, a block of zeros, a block at half the parent's norm), the hand-written list does as well as the programmer, and the blind programmer as well as both: on this family of defects the programmer adds nothing over (c), and the passes add nothing over (b). (3) A plain coordinate search over output biases (d) repairs only the corruption that lives in the output biases (blank +2) and partly the rest, and cannot touch a defect in a conv block. (4) On practice sets that are not training data, the BLB model again yields nothing (4 attempts, 0 kept edits), and `combined-v1` yields the first held-out gain of the project: all three independent programmer attempts converged on the same edit (blank bias −0.3, then interpolating the non-output tensors 25 % toward the clean sibling), −0.39 points on `bern-eval` (paired line interval [−0.69, −0.08], page-clustered [−0.72, −0.06], 14 pages), which is a model-merging gain, small, and not yet confirmed by a sham control.

## 1. Setup shared by both items

- Harness changes (all with tests in `tests/test_controls.py`): `Session.keep_unmeasured` (budget-0 programmer; refused when any budget exists); `Session.unseen_by(donor…)` and `Session.measure(…, rule_mask=…)` (the keep rule's paired interval is computed on the masked lines, metrics still on all); `wsp.audit validation` (practice vs the subject's training pages and vs the held-out list); `wsp.run.heldout` reuses a score for the same (digest, list), recorded in `summary.json` as `scored_from_cache`; the sandbox command line now carries `CUDA_VISIBLE_DEVICES=` (empty) and 4 threads and puts this checkout's `wsp` first (brief item 8, second bullet: the earlier sessions' `summary.json` say `CUDA_VISIBLE_DEVICES=0`; the sessions ran on the CPU path because the program passes `device="cpu"`, not confirmed beyond that here).
- Keep rule: PROTOCOL.md §3 (paired 95 % interval below 0, `min_effect` 0), 2,000 resamples, seed 20261106, line-level. **All intervals in this report resample lines, not pages; lines within a page are correlated, so they are too narrow.** Where a page-clustered interval was computed it is stated.
- Programmers. (a), (b), and the item-3 attempts `a1–a3`: fresh subagents, model `claude-sonnet-5-5`, default reasoning effort, tools Read/Write/Bash (only `py_compile`), shown only their reading pack (tensor statistics, diff to the parent with rows aligned by character, the C0 reading of the practice set) and `runs/*/PROGRAMMER_BRIEF.md`; told the model *may* be defective. They wrote their program in advance and never ran it. (c) and (d): fixed programs `runs/planted/arms/program_{c,d}.py`, written before any pass, identical for every model; (c) is mine (Claude Sonnet 5.5), thresholds taken from the family's healthy fine-tunes. The `replay` sessions rerun the programs of 06.11.2026 (Fable 5.1). The session's author planted the defects but did not write (a) and (b). The pack of a model contains the diff to its parent, which is where almost every planted defect shows; it does not contain the uncorrupted model.
- The subagent programmers are not independent in the item-3 attempts in practice: given the same pack, the three attempts on `combined-v1` reached the same final model (same tensor digest `c4cfb0500539…`).

## 2. Item 2 — planted defects (`kraken-german-xix-blb-v1`, practice `blb-train-300`, held-out `blb-eval-hand`)

Uncorrupted model: practice CER 6.46 %, held-out (2,272 lines, 29 objects) 9.55 %. The eight corruptions (`wsp/plant.py`; files in `~/atr-cache/wsp/planted/`, spec and digests in `plant.json` there): k1 output row `e` set to 0; k2 row `n` negated; k3 rows `e`,`n` swapped; k4 blank bias +2; k5 output weight ×2 (bias unchanged); k6 BiLSTM `L_16` ×0.5; k7 Gaussian noise on `L_14` (norm 25 % of the block); k8 conv `C_3` set to 0. A scale ×0.5 of a layer was not run separately (k6 is ×0.5 of a block).

Preregistered outcome classes (in each run's `preregistration.md`, before any pass): *repaired* = held-out CER recovers ≥ 80 % of the damage and the paired interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = below 20 %; damage under 0.5 points = *null corruption*.

### 2.1 Corruptions, before any session

| | what | practice CER | held-out CER | damage on held-out (points) [95 %] | |
|---|---|---:|---:|---|---|
| — | uncorrupted | 6.46 | 9.55 | | |
| k1 | zero row `e` | 16.23 | 19.66 | +10.10 [9.92, 10.27] | defect |
| k2 | negate row `n` | 74.56 | 70.24 | +60.69 [59.89, 61.48] | defect |
| k3 | swap rows `e`,`n` | 24.43 | 27.76 | +18.20 [17.93, 18.47] | defect |
| k4 | blank bias +2 | 7.89 | 10.85 | +1.30 [1.16, 1.45] | defect |
| k5 | output weight ×2 | 6.50 | 9.53 | −0.03 [−0.09, +0.03] | **null corruption**: no effect on the argmax decode |
| k6 | `L_16` ×0.5 | 19.93 | 21.17 | +11.62 [11.24, 12.04] | defect |
| k7 | noise on `L_14` | 7.26 | 10.04 | +0.49 [0.40, 0.59] | borderline (0.49 < 0.5: classed null by the preregistered rule, but the interval excludes 0) |
| k8 | zero conv `C_3` | 100.00 | 100.00 | +90.45 [89.88, 90.99] | defect (every line empty) |

### 2.2 The four arms: held-out CER, paired Δ against the corrupted model, class

Cell: held-out CER % · Δ (points) [95 % interval] · class. Passes / edits / kept in 2.3.

| corruption | (a) programmer, budget 10 | (b) blind, budget 0 | (c) hand-written list, budget 10 | (d) bias coordinate search, budget 10 |
|---|---|---|---|---|
| k1 zero row `e` (9.55) | 9.57 · −10.09 [−10.26, −9.91] · **repaired** | 9.76 · −9.90 [−10.07, −9.71] · repaired | 9.72 · −9.93 [−10.10, −9.75] · repaired | 12.56 · −7.10 [−7.32, −6.87] · partly |
| k2 negate `n` | 9.55 · −60.69 [−61.48, −59.88] · **repaired** | 9.55 · same · repaired | 9.62 · −60.62 [−61.41, −59.81] · repaired | 51.59 · −18.65 [−19.15, −18.14] · partly |
| k3 swap `e`,`n` | 9.55 · −18.20 [−18.47, −17.93] · **repaired** | 9.55 · same · repaired | 9.77 · −17.99 [−18.26, −17.70] · repaired | 27.68 · −0.07 [−0.10, −0.05] · **not repaired** |
| k4 blank +2 | 9.52 · −1.33 [−1.49, −1.18] · repaired | 9.55 · −1.30 [−1.45, −1.15] · repaired | 9.50 · −1.35 [−1.51, −1.19] · repaired | 9.55 · −1.30 [−1.45, −1.15] · repaired |
| k5 weight ×2 (null) | unchanged (no kept edit) | ×0.5 applied: 9.55 (Δ +0.03 [−0.03, +0.09]) | unchanged | unchanged |
| k6 `L_16` ×0.5 | 9.55 · −11.62 [−12.04, −11.24] · repaired | 9.55 · same · repaired | 9.55 · same · repaired | 12.94 · −8.23 [−8.56, −7.94] · partly |
| k7 noise `L_14` | unchanged, 10.04 | no edit, 10.04 | unchanged, 10.04 | unchanged, 10.04 |
| k8 zero `C_3` | 9.77 · −90.23 [−90.77, −89.64] · **repaired** | 9.77 · same · repaired | 9.77 · same · repaired | unchanged, 100.00 · **not repaired** |

There are 19 repaired cells (k1, k2, k3, k6, k8 × three arms, k4 × four arms). In all 19 the final CER is within 0.22 points of the uncorrupted 9.55 % (largest: k8 +0.22, k3 (c) +0.21, k1 (b) +0.20, k1 (c) +0.17); in 10 it is within 0.01 point (k1 (a), k2 (a)(b), k3 (a)(b), k4 (b)(d), k6 (a)(b)(c)). Per-arm detail (passes used / edits / kept):

| | (a) | (b) | (c) | (d) |
|---|---|---|---|---|
| k1 | 5/4/3: parent row for `e`, then ×1.15 and bias +0.5 of that row | 0/1/1: parent row | 2/1/1: parent row | 10/9/4: blank −1 ×2, `e` bias +1 ×2 |
| k2 | 3/2/1: negate row `n` back | 0/1/1: negate back | 2/1/1: parent row | 10/9/6: blank +1 ×3, space +1 ×3 |
| k3 | 2/1/1: swap back | 0/1/1: swap back | 2/1/1: parent rows | 10/9/1 |
| k4 | 4/3/1: blank −2.1 | 0/1/1: blank −2.0 | 2/1/1: blank gap to the parent's | 10/9/2: blank −1 ×2 |
| k5 | 5/4/0 (×0.5 refused) | 0/1/1: ×0.5 | 1/0/0 | 10/9/0 |
| k6 | 6/5/1: `L_16` ×2 | 0/1/1: ×2 | 2/1/1: rescale to the parent's norm | 10/9/4 |
| k7 | 6/5/0 | 0/0/0 (reading judged "nothing sure") | 3/2/0 | 10/9/0 |
| k8 | 7/6/1: parent's `C_3` | 0/1/1: parent's `C_3` | 2/1/1: transplant `C_3` | 10/9/0 |

### 2.3 What this shows and what it does not

Shows. (i) The positive control passes: for every corruption with damage ≥ 1 point, three arms find the defect from the readings and repair it; the harness's keep rule accepted the right edit each time it was tried and refused the wrong ones (e.g. k5 (a): `×0.5` refused, correctly, as the corruption has no effect). So the absence of a kept edit on the real models in sessions 1 and 2 is not explained by a method that cannot keep a true repair. (ii) The programmer (a) is not better than (c) or (b): the hand-written list (2 passes, one rule per defect type) repairs the same set; the blind reading alone does. The passes bought nothing here beyond confirmation (and the occasional refinement: k1 (a) 9.57 against (b) 9.76, +0.19 point, an unreplicated difference from one attempt). (iii) A plain bias coordinate search (d) is not a substitute: it repairs only what lives in the biases (k4; it also partly compensates k1, k2, k6 by blank/space shifts, which is the "trade deletions for insertions" mode seen in sessions 1 and 3), fails on k3 and on any non-output defect.

Does not show. (i) The defects are easy: each leaves a signature in the diff against the parent (cosine, norm ratio, zeros) and in the confusion table (k1 `e`→∅ 910, k2 `n` inserted 7,667 times, k3 `e`↔`n` 1,364/837); real damage from label noise or a bad fine-tune is spread (docs/PHASE3_REPORT.md) and leaves no such signature. The control shows the method is sensitive to localised, large, reference-visible defects, not that it would find diffuse ones. (ii) The repairs lean on a donor (the parent) that shares the lineage: rows and blocks copied from `xix-v1` fit the child almost exactly because the child is a 0.1-learning-rate fine-tune of it; an unrelated donor would not do this. (iii) k7 (noise at 25 % of the block, +0.49 points) was repaired by no arm, including (a), so a *diffuse* defect of about half a point is below what this protocol finds: the keep rule needs a gain whose line-level interval excludes 0, and nothing in the readings points at noise. (iv) k5 shows a corruption can be invisible to the metric. (v) One model, one hand, line-level intervals, 8 corruptions; the corruptions were chosen by the session's author, not sampled. (vi) The practice set is the model's own training lines, as in the earlier sessions; the corruption sessions do not need it to be otherwise (the defect is known), but the practice CER of the uncorrupted model there (6.46 %) is optimistic (item 3).

## 3. Item 3 — practice sets that are not training data

### 3.1 What the trainer's split allows (read from `training_folder/jobs/<job>/data/`)

| model | trainer's validation | consequence |
|---|---|---|
| `xix-blb-v1` (job 20261026T193646Z) | `pages_val.lst` = the **157 `_eval` pages** (the held-out sets of this repository); `train_lines` = `lines_written` = 38,360: every line of the 1,651 train pages was trained on | **No validation side exists inside the train projects**, and the trainer's checkpoint selection (`best_0.9127`) used the eval pages. The brief's first option (carve unused train pages) is not available. The earlier "held-out" `blb-eval` / `blb-eval-hand` were also the trainer's validation set: unseen for weights, used for early stopping (a mild selection effect, to be stated in the manuscript) |
| `combined-v1` (job 20261025T160407Z) | 17 pages (6 Hertwig, 11 Schlussredaktion) of 168, partition 0.9, page level, seed 42 | a real validation side, never trained on by the subject; its images are the 25.10 export's (Hertwig pairs wrong), so the pages are rebuilt from the 22.10 export (Hertwig) and the 25.10 export (Schlussredaktion, identical to 24.10) |

Donors: `hertwig-v2` (and v1) share one validation split (5 Hertwig pages), so **every one of the 6 Hertwig validation pages of `combined-v1` is a training page of `hertwig-v2`** (212 lines); `schlussredaktion-v1` has 12 validation pages, three of which (20, 143, 566) are in `combined-v1`'s validation set, so it trained on 8 of the 11 (216 lines). Also `combined-v1` was initialised from `hertwig-v2`: the Hertwig part of its validation set was seen by its *ancestor* (the Schlussredaktion part was not seen by any ancestor). That is why session 2 on training lines kept a transplant from `hertwig-v2` (docs/PHASE3_REPORT.md §3).

### 3.2 The sets (built by `scripts/build_validation_sets.py`, audited by `python -m wsp.audit validation`)

| set | lines | source | held-out for sessions on it | audit |
|---|---:|---|---|---|
| `bern-val` | 513 (1 line skipped: degenerate geometry) | the 17 validation pages of `combined-v1`; every line; `seen_by` per line | `bern-eval` (545 lines, 14 test pages) | clean: none of 17 pages in the 151 training pages, none in the held-out pages (`data/practice/overlap-bern-val.json`) |
| `blb-val-300` | 300 | **carved from the eval side** (the only unseen-by-training source): 7 whole objects (K2911,103 / 26 / 39 / 46 / 58, K2912,32, K2915,3) drawn with seed 20261107 from `blb-eval-hand` until 450 lines, 300 sampled | `blb-eval-hand-b` = `blb-eval-hand` minus those 7 objects: 1,718 lines, 22 objects | clean at object level (none of 7 in the 1,651 training pages parsed from the job's lists, none in the held-out set; `overlap-blb-val-300.json`) |

No validation source for BLB handwriting exists beyond the eval objects themselves; the other eval objects (printed book, typescript, printed form, music, envelope page) are excluded from `blb-eval-hand` as non-handwriting and are not a practice distribution for a Kurrent model, so none was used. The BLB held-out set therefore changes (`blb-eval-hand-b`, 1,718 lines; the unedited model reads it at 8.41 %); numbers are not comparable with the 9.55 % of `blb-eval-hand`.

What the practice-side change does to C0 (the review's second objection): `xix-blb-v1` reads its own training lines at 6.46 % (`blb-train-300`) and unseen handwriting at 13.83 % (`blb-val-300`, different objects, 7 objects only: a difference, not a clean measurement of memorisation); `combined-v1` 14.56 % on `bern-train-300`, 13.86 % on `bern-val`.

### 3.3 Sessions (budget 10, keep rule §3, donor rule applied through `rule_mask`)

Per model: `a1–a3` three programmer attempts and `replay` the 06.11.2026 program (BLB unchanged; Bern with masks added). Held-out scored once per session (identical final digests share one score).

**`xix-blb-v1`, practice `blb-val-300`, held-out `blb-eval-hand-b`** (before: practice 13.83 %, held-out 8.41 %)

| attempt | passes | edits | kept | practice C0 → last pass | held-out | kept edits |
|---|---:|---:|---:|---|---|---|
| a1 | 8 | 7 | 0 | 13.83 → 14.72 | 8.41 → 8.41 (Δ 0) | — |
| a2 | 8 | 7 | 0 | 13.83 → 13.72 | 8.41 → 8.41 (Δ 0) | — |
| a3 | 6 | 5 | 0 | 13.83 → 13.81 | 8.41 → 8.41 (Δ 0) | — |
| replay (H1–H8 of 06.11.) | 9 | 8 | 0 | 13.83 → 13.86 | 8.41 → 8.41 (Δ 0) | — |

On unseen practice lines the keep rule refuses every output-layer, blank, space and letter-row edit, again: 0 of 28 edits kept, final model equals the subject. (Session 1 on training lines had kept two noise-sized edits under the old rule.) This repeats the phase-1 finding on data that is not the model's training set.

**`combined-v1`, practice `bern-val`, held-out `bern-eval`** (before: practice 13.86 %, held-out 19.33 %)

| attempt | passes | edits | kept | practice C0 → last pass | held-out | Δ held-out (points) [line 95 %] | kept edits |
|---|---:|---:|---:|---|---|---|---|
| a1 | 9 | 8 | 2 | 13.86 → 12.38 | 19.33 → 18.94 | −0.387 [−0.693, −0.082] | blank bias −0.3; non-output tensors α = 0.25 toward `schlussredaktion-v1` |
| a2 | 8 | 7 | 2 | 13.86 → 12.35 | 19.33 → 18.94 | −0.387 [same] | the same two edits (same final digest) |
| a3 | 9 | 8 | 2 | 13.86 → 12.35 | 19.33 → 18.94 | −0.387 [same] | the same two edits |
| replay | 9 | 8 | 1 | 13.86 → 12.17 | 19.33 → 19.01 | −0.320 [−0.697, +0.071] | `L_12` transplanted from `schlussredaktion-v1` |

Per held-out subset of the a1–a3 model: Hertwig (287 lines) −0.49 [−1.10, +0.04]; Schlussredaktion (258) −0.30 [−0.58, −0.01]; page-clustered bootstrap over the 14 test pages −0.39 [−0.72, −0.06]. The replay's page-clustered interval is [−0.67, +0.04].

What happened to the 06.11. transplant: on training lines the output layer from `hertwig-v2` was kept (practice −0.42 points) and failed on held-out (+0.15). Under the donor rule the replay judges it on the 301 lines `hertwig-v2` did not train on and **refuses it** (`CER rose`, interval [+0.32, +0.83] points): the rule removes the artefact. The one edit the replay keeps instead, `L_12` from the sibling, is judged on the 297 lines the sibling did not train on (interval [−2.18, −0.66] points) and does not clearly transfer (held-out interval contains 0).

Reading. (i) The a1–a3 edit is a two-parameter-family model merge toward a sibling that reads the Schlussredaktion hand better (0.118 against 0.139 on `bern-val`) and the blank bias; the three programmers found it because the pack listed the sibling as a donor and said it read the practice lines better. The practice gain (−1.5 points) shrinks to −0.39 on held-out; the interval excludes 0 at line level and at page level, barely. It is not a "repair of label-noise damage" (the Hertwig part is not significant) and it has had no C2/C3 comparison (α = 0.5 reads better on practice and was refused by the rule, so the α-sweep would say where the optimum is), no sham (C5) and no second held-out set. (ii) With three attempts converging on one final digest the "at least three independent attempts" are three samples of one programmer given one pack, not three programmers. (iii) The BLB model, with a parent that is a worse model on every line, has no donor to merge toward, and nothing is found.

### 3.4 Proposed amendment to docs/PROTOCOL.md

Written into §2 of `docs/PROTOCOL.md`, marked **PROPOSED AMENDMENT (07.11.2026, not yet adopted)**: item 7, the donor rule (a donor must not have been trained on the lines the keep rule is computed on; donor edits are judged on lines unseen by that donor, `Session.unseen_by` / `rule_mask`; the preregistration lists each donor's training and validation pages); item 8, practice sets should be lines the subject was not trained on. The implementation and tests are in this branch; the amendment is for the author to adopt or change.

## 4. For Section 10 of the manuscript (paragraph)

> To check that a null result is informative, we planted defects in `kraken-german-xix-blb-v1` and ran the same session protocol on the corrupted copies. Eight corruptions were applied: a zeroed output row, a negated row, two swapped rows, a +2 offset of the blank bias, the output layer scaled by 2, the last recurrent block scaled by 0.5, noise on the middle recurrent block, and a zeroed convolution block. For the six that cost the model at least one point of held-out CER (1.3 to 90 points), a programmer with ten passes, a blind programmer without any pass, and a fixed list of repairs each restored 80–100 % of the damage on held-out lines (in 19 of the 19 repaired cases the held-out CER is within 0.22 points of the uncorrupted model, in 10 within 0.01 point); a coordinate search over the output biases repaired only the blank offset and partly three others. The control therefore shows that the method, and in particular the keep rule, can find and keep a true repair when the defect is localised and visible against the parent. It also shows its limits: the programmer did no better than the hand-written list or the blind reading, the repairs depend on a parent of the same lineage as donor, a diffuse defect of half a point (noise on one block) was repaired by no arm, and an output-layer scaling that does not change the decoded text is invisible to the metric. A defect of the kind we looked for in the real models, spread over the network and without a signature in the diff, is not covered by this control. When practice lines are not the model's training lines (the trainer's validation pages), the BLB model again yields no kept edit in four attempts, and `combined-v1` yields one small gain by merging 25 % toward its clean sibling together with a blank-bias shift (−0.39 points on 545 held-out lines, line-level interval [−0.69, −0.08]; page-clustered [−0.72, −0.06]), which we report as a merge rather than a repair and which has not been compared with an α-sweep or a sham.

## 5. Not done, failed, or open

- C4 (retrain) not touched. Item 3 held-out scores for sessions with kept edits were taken once; no second evaluation.
- Planted defects: line-level intervals only; no cluster bootstrap (item 4 of the brief); no repeated programmer attempts for (a)/(b) (one attempt per corruption and arm); one subject.
- k5 turned out to be a null corruption (its held-out CER is within 0.03 points of the original); k7's classification is "null" by the preregistered 0.5-point rule although its interval (0.40–0.59) excludes 0.
- A first launch of the 32 sessions ran nothing (an unexported variable) and an earlier one was killed for thrashing (the sandbox dropped the thread limit); both were deleted and relaunched before any result was read; no session was discarded after its result was seen. Two partial runs were removed the same way.
- `diff.json` is committed as `diff.json.gz` when under 5 MB; for 7 sessions (interpolations touch every weight) it is omitted (`DIFF_OMITTED.txt` in the folder); the edits are reproducible from `program.py` and the donor models.
- The three `a1–a3` combined-v1 programmers converged on one model, so the "three independent attempts per model" are not three independent solutions; more independent attempts would need different packs or programmers.
- The `replay` of session 1 on BLB used the original program unchanged; the 06.11. interval-free keep rule no longer applies, the current one does.
