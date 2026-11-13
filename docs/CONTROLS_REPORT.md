# Controls report — planted defects and validation-based practice (items 2 and 3 of docs/BRIEF_ARTICLE_EXPERIMENTS.md)

Asteraix, 07.11.2026, branch `claude/article-controls-8803eb` on `origin/main` (07f55d7). CPU only, kraken 7.0.2, no retraining, nothing written to the share, registry, Hugging Face or the trainer. Run folders: `runs/20261107-planted-*` (32), `runs/20261107-val-*` (8), packs and programs in `runs/planted/` and `runs/validation/`. Aggregates: `docs/controls/*.json|md`. Tests: 45 passed, 2 skipped (see §10 for the follow-up sections 6–10).

**In four sentences.** (1) The method *can* find and repair planted defects: of seven real corruptions of `xix-blb-v1`, the programmer with passes (a), the blind programmer (b) and a hand-written repair list (c) each brought the held-out CER back to within 0.22 points of the uncorrupted model on every corruption that did damage of 1 point or more (6 of 6; 19 of the 19 repaired cells, 10 of them to within 0.01 point), so "no defect found" on the real models carries information. (2) But these defects are of the kind the diff against the parent shows at once (a row with cosine −0.99, a block of zeros, a block at half the parent's norm), the hand-written list does as well as the programmer, and the blind programmer as well as both: on this family of defects the programmer adds nothing over (c), and the passes add nothing over (b). (3) A plain coordinate search over output biases (d) repairs only the corruption that lives in the output biases (blank +2) and partly the rest, and cannot touch a defect in a conv block. (4) On practice sets that are not training data, the BLB model again yields nothing (4 attempts, 0 kept edits), and `combined-v1` yields the first held-out gain of the project: all three independent programmer attempts converged on the same edit (blank bias −0.3, then interpolating the non-output tensors 25 % toward the clean sibling), −0.39 points on `bern-eval` (paired line interval [−0.69, −0.08]; **the page-clustered interval computed later, §9, is [−0.84, +0.04] and contains 0; the page-clustered figure [−0.72, −0.06] quoted in earlier drafts of this report came from a one-stage page resample and is withdrawn**), which is a model-merging gain, small; §6 shows it is distinguishable from norm-matched random edits but not from plain interpolation toward the sibling, and sits in a narrow α window.

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

Per held-out subset of the a1–a3 model: Hertwig (287 lines) −0.49 [−1.10, +0.04]; Schlussredaktion (258) −0.30 [−0.58, −0.01]; page-clustered (two-stage) bootstrap over the 14 test pages: see §9 (−0.39 [−0.84, +0.04]); the earlier one-stage figure was withdrawn.

What happened to the 06.11. transplant: on training lines the output layer from `hertwig-v2` was kept (practice −0.42 points) and failed on held-out (+0.15). Under the donor rule the replay judges it on the 301 lines `hertwig-v2` did not train on and **refuses it** (`CER rose`, interval [+0.32, +0.83] points): the rule removes the artefact. The one edit the replay keeps instead, `L_12` from the sibling, is judged on the 297 lines the sibling did not train on (interval [−2.18, −0.66] points) and does not clearly transfer (held-out interval contains 0).

Reading. (i) The a1–a3 edit is a two-parameter-family model merge toward a sibling that reads the Schlussredaktion hand better (0.118 against 0.139 on `bern-val`) and the blank bias; the three programmers found it because the pack listed the sibling as a donor and said it read the practice lines better. The practice gain (−1.5 points) shrinks to −0.39 on held-out; the line-level interval excludes 0; the two-stage page-clustered interval (§9) does not. It is not a "repair of label-noise damage" (the Hertwig part is not significant) and it has had no C2/C3 comparison (α = 0.5 reads better on practice and was refused by the rule, so the α-sweep would say where the optimum is), no sham (C5) and no second held-out set. (ii) With three attempts converging on one final digest the "at least three independent attempts" are three samples of one programmer given one pack, not three programmers. (iii) The BLB model, with a parent that is a worse model on every line, has no donor to merge toward, and nothing is found.

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


---

# Follow-up (07.11.2026, later): sections 6–10

Requested after the first report; CPU only, 4 threads per worker, nothing outside the repo and `~/atr-cache/wsp/`. New code: `wsp.metrics.bootstrap_delta(..., clusters=)` (two-stage cluster bootstrap: pages resampled, then lines within a drawn page; `ci95` unchanged), `wsp.metrics.page_of`, `scripts/sham_alpha_combined.py`, `scripts/global_interp.py`, `scripts/margins.py`, `scripts/recompute_intervals.py`, corruptions k9–k11 in `wsp/plant.py`. Aggregates in `docs/controls/`. Models for the programmers: Claude Sonnet 5.5 (`sonnet`), Claude Opus 5.5 (`opus`), Claude Haiku 5.5 (`haiku`) as fresh subagents given only their pack; Claude Fable 5.1 (`fable`) attempts `f1` were first launched on 07.11.2026 and failed with "out of usage credits"; they were rerun on 09.11.2026 and are in §9c. All intervals below are 95 %.

## 6. Is the `combined-v1` gain distinguishable from sham and from plain averaging? (`docs/controls/combined_controls.json`)

The edit: blank bias −0.3 and the 32 non-output tensors α = 0.25 toward `schlussredaktion-v1` (all three Sonnet attempts' final model). Everything below is scored on `bern-val` (practice, 513 lines) and `bern-eval` (held-out, 545 lines, 14 pages), against the unedited model (practice 13.86 %, held-out 19.33 %).

**(a) C5 sham, 20 draws.** Each non-output tensor receives Gaussian noise of the same Frobenius norm as the real α = 0.25 delta of that tensor, plus the same blank −0.3. Held-out Δ of the 20 shams: mean −0.013 points, sd 0.062, range [−0.199, +0.097]; practice CER of the shams 13.82 % on average (best 13.61). The real edit: practice 12.39 % and held-out −0.39. **0 of 20 shams reach the real edit's held-out gain** (the real gain lies about 6 sd below the sham mean). The keep rule of PROTOCOL §3, applied to each sham against the blank-−0.3 state, **kept 0 of 20** — on the 297 sibling-unseen lines (the rule used for the real edit) and on all 513. So the rule's false-keep rate against norm-matched noise is 0/20 here (rule-of-three upper 95 % bound 15 %), and the real gain is not what random edits of that size do.

**(b) C2 and (c) the α sweep** (non-output tensors α toward the sibling, `parent`-style axis; with and without the blank edit; page-clustered interval is the honest one):

| edit | practice CER % | held-out CER % | held-out Δ points [lines] | [pages] |
|---|---:|---:|---|---|
| blank −0.3 only | 13.71 | 19.34 | +0.01 [−0.09, +0.12] | [−0.15, +0.18] |
| α = 0.1 | 13.26 | 19.17 | −0.16 [−0.34, +0.01] | [−0.44, +0.10] |
| α = 0.25 | 12.51 | 19.01 | −0.31 [−0.61, −0.02] | [−0.77, +0.09] |
| α = 0.4 | 11.90 | 19.28 | −0.05 [−0.41, +0.31] | [−0.58, +0.43] |
| α = 0.5 | 11.66 | 19.35 | +0.02 [−0.37, +0.40] | [−0.55, +0.60] |
| α = 0.75 | 11.54 | 19.76 | +0.43 [−0.05, +0.89] | [−0.36, +1.29] |
| α = 1.0 | 11.83 | 20.37 | +1.04 [+0.52, +1.55] | [+0.05, +2.19] |
| α = 0.1 + blank | 13.19 | 19.14 | −0.19 [−0.40, +0.01] | [−0.51, +0.08] |
| **α = 0.25 + blank (the edit)** | 12.39 | 18.94 | **−0.39 [−0.69, −0.08]** | **[−0.84, +0.04]** |
| α = 0.4 + blank | 11.74 | 19.17 | −0.16 [−0.52, +0.22] | [−0.70, +0.36] |
| α = 0.5 + blank | 11.54 | 19.29 | −0.04 [−0.43, +0.34] | [−0.62, +0.56] |
| α = 0.75 + blank | 11.34 | 19.62 | +0.29 [−0.21, +0.75] | [−0.48, +1.13] |
| α = 1.0 + blank | 11.73 | 20.15 | +0.82 [+0.29, +1.31] | [−0.17, +1.95] |
| **C2** uniform average of subject and sibling, output rows aligned by character | 11.65 | 19.31 | −0.02 [−0.43, +0.38] | [−0.62, +0.60] |

Averages of the subject with `hertwig-v2` and of all three were not run.

State plainly: (1) The −0.39 is **distinguishable from random edits of the same size** (0/20 shams, 0/20 false keeps). (2) It is **not distinguishable from plain interpolation**: it *is* plain interpolation, the blank edit adds nothing alone (+0.01), and it adds −0.08 on top of α = 0.25 (inside every interval). The gain lives in a narrow window: held-out improves at α = 0.1–0.25 (−0.16 to −0.39), is zero at 0.4–0.5 (including the uniform average C2: −0.02) and the held-out loss is clear by α = 1. (3) **No interval that is page-clustered excludes 0 at any α.** (4) The practice set ranks the α wrongly: practice CER keeps falling to α ≈ 0.75 (11.3–11.5 %), where held-out is worse by +0.3 to +0.4; the keep rule's refusal of the larger steps (it refused α = 0.5 and 0.75 because the interval against the α = 0.25 state contained 0) saved the model from an overshoot, but only because the rule is conservative, not because the practice set measures the optimum. So the project's one candidate gain is a small, α-specific interpolation effect, with a page-level interval that includes 0 and a practice set that over-fits the interpolation; it should be reported as a model-merge observation, not as a repair.

## 7. Global interpolation and additivity, Kurrent pair (`docs/controls/global_interp.json`)

`parent + α (child − parent)` on `xix-v1` → `xix-blb-v1`; blocks C_0, C_3, C_6, C_9, L_12, L_14, L_16, O_18 (output rows aligned by character, the child's 72 new rows kept). α = 1 is the child, 0 the parent on the chosen blocks. Two sides, labelled: **practice** = `blb-val-300` (300 lines of 7 eval-side objects the model was not trained on; child 13.83 %) and **held-out** = `blb-eval-hand-b` (1,718 lines; child 8.41 %). All eight blocks jointly at α = 0…1.5 (step 0.25), each block alone at the same α, and all 28 block pairs at α = 0.5. "Additivity": joint cost minus the sum of the eight single-block costs at the same α (positive = the joint move is worse than the sum).

| α | practice: joint CER % | joint Δ | sum of singles | joint − sum [pages] | held-out: joint CER % | joint Δ | sum of singles | joint − sum [pages] |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 0.0 (parent) | 32.66 | +18.83 | +16.35 | +2.48 [−2.33, +6.68] | 21.70 | +13.29 | +9.49 | **+3.81 [+2.57, +4.98]** |
| 0.25 | 26.73 | +12.90 | +11.28 | +1.63 [−1.73, +4.91] | 17.02 | +8.61 | +6.38 | **+2.23 [+1.48, +3.02]** |
| 0.5 | 21.18 | +7.35 | +5.28 | +2.07 [−0.18, +4.26] | 13.24 | +4.84 | +2.65 | **+2.19 [+1.52, +2.95]** |
| 0.75 | 15.83 | +2.00 | +1.54 | +0.46 [−0.60, +1.64] | 9.23 | +0.82 | +0.44 | **+0.38 [+0.06, +0.70]** |
| 1.25 | 14.50 | +0.67 | +0.77 | −0.10 [−1.26, +1.00] | 8.69 | +0.29 | +0.06 | +0.22 [−0.10, +0.59] |
| 1.5 | 16.70 | +2.87 | +2.71 | +0.16 [−1.71, +1.89] | 10.46 | +2.05 | +1.02 | **+1.03 [+0.51, +1.59]** |

Reading. (i) **Nowhere on the line is there a gain**: moving any distance from the child toward the parent costs, and extrapolating costs (α = 1.25: +0.67 practice, +0.29 held-out; 1.5: +2.9, +2.1). (ii) **The joint cost is larger than the sum of the single-block costs on the held-out side** (+0.4 to +3.8 points; page interval excludes 0 at α ≤ 0.75 and at 1.5) and not distinguishable from additive on the practice side (300 lines, wide intervals, same sign). The blocks interact: the damage of moving all of them is worse than the sum of moving each alone. (iii) **Pairs at α = 0.5 are close to additive**: the sum of the two single-block costs predicts the pair's cost with correlation 0.98 (practice) and 0.97 (held-out); mean interaction +0.11 and +0.07 points; the largest are L_12+L_14 (+0.87, practice) and L_16+O_18 (+0.76, held-out). So the departure from additivity builds with the number of blocks moved, not within pairs. This qualifies the manuscript's statement that the sweep was "consistent with an additive loss surface": single blocks and pairs are close to additive, the whole network is not.

## 8. Margins at deleted frames (`docs/controls/margins.json`)

`xix-blb-v1` on `blb-eval-hand-b`: 696 deleted instances of space, `e` or `n` in 418 of 1,718 lines (44 lines skipped: unreadable crop or whitespace normalisation shifting positions). For each, the *window* is the frames strictly between the last frame of the previous emitted character and the first frame of the next (kraken's span end is inclusive). **127 of the 696 (18 %) have no frame between the neighbours**: no blank frame exists there, so no blank-bias edit can act on them. For the other 569: M1 = the smallest over the window of (blank − deleted character) log-probability difference (a softmax log-ratio equals the logit difference): how close the deleted character came to winning at its best frame; M2 = the smallest (blank − best non-blank class).

| | n | M1 min | p5 | p25 | median | p75 | p95 | share M1 < 2 | M2 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| space | 305 | 0.00 | 0.12 | 1.15 | 2.53 | 5.27 | 10.18 | 43 % | 1.77 |
| e | 180 | 0.07 | 0.44 | 1.72 | 3.94 | 6.98 | 10.49 | 30 % | 1.94 |
| n | 84 | 0.01 | 0.22 | 1.64 | 5.38 | 9.58 | 14.92 | 31 % | 1.62 |
| all | 569 | 0.00 | 0.20 | 1.35 | **3.17** | 6.60 | 11.57 | 37 % | 1.81 |

Effect of the blank-bias states of session 1 (each shifts every blank-minus-X margin by exactly the edit, so the margins after an edit d are the margins before plus d). Predicted share of the 569 recovered (M1 + d < 0): −0.113: 2.6 %; −0.5: 9.1 %; −1.0: 19.0 %. Measured by decoding the edited models on the 372 lines (the three characters, 696 instances incl. no-gap ones): deletions 696 → 677 (−2.7 %), 613 (−11.9 %), 544 (−21.8 %); insertions in those lines 252 → 263, 311, 385 (+11, +59, +133). So the recovered deletions are paid for with nearly as many insertions at −1.0 (−152 deletions, +133 insertions).

Reading: **the margins support the claim that the deletions are decided upstream of the output layer**: the median deleted character sits 3.2 nats (a factor of 24 in probability) below the blank at its best frame, 63 % are more than 2 nats away, a fifth have no blank frame to act on, and the three states of session 1 reach only about one deletion in 40, in 9 and in 5. The 37 % within 2 nats are reachable only by edits larger than session 1 tried, and the measured insertions say they trade one error for another. The measured and predicted recoveries agree in size; the bias edits are not the lever. (The first version of this script put the neighbours' own frames into the window and reported every margin as negative; it was corrected before any number was used.)

## 9. Cluster intervals

**Earlier results recomputed** (`docs/controls/cluster_intervals.md`; line-level numbers reproduce the originals exactly; practice-set lines are scattered over many pages, so practice intervals barely change):

| result | lines | pages | Δ points | line-level | page-clustered |
|---|---:|---:|---:|---|---|
| R2 session 1, held-out `blb-eval` | 5,628 | 157 | −0.010 | [−0.048, +0.026] | [−0.069, +0.049] |
| R2 session 1 on `blb-eval-hand` | 2,272 | 94 | −0.001 | [−0.055, +0.053] | [−0.082, +0.084] |
| R3 session 2, practice (pass 8) | 300 | 146 | −0.425 | [−0.850, −0.036] | [−0.926, +0.037] (**now contains 0**) |
| R3 session 2, held-out `bern-eval` | 545 | 14 | +0.151 | [−0.099, +0.421] | [−0.285, +0.597] |
| R3 held-out Hertwig / Schlussredaktion | 287 / 258 | 6 / 8 | −0.300 / +0.510 | [−0.721, +0.127] / [+0.206, +0.822] | [−0.825, +0.213] / **[−0.043, +1.076]** (the Schlussredaktion loss **is no longer significant**) |
| this report, `combined-v1` a1–a3, `bern-eval` | 545 | 14 | −0.387 | [−0.693, −0.082] | [−0.838, +0.035] |
| this report, replay | 545 | 14 | −0.320 | [−0.697, +0.071] | [−0.840, +0.189] |

Consequences for earlier text: the four per-project BLB intervals of PHASE1 §3 all widen but still contain 0; the "significant Schlussredaktion loss" of PHASE3 §3 and docs/RESULTS.md R3 and the "significant practice gain" of R3 are line-level findings that a page-clustered interval does not support; the conclusion "not a gain" stands, and the claim that the edit *harmed* Schlussredaktion does not. **The α sweep** (`docs/baselines/alpha-xix-blb-v1.cluster.json`, rerun, `docs/controls/alpha_cluster_table.md`): the 16 per-block rows match the old numbers; practice lines are spread over 263 pages, so page intervals differ from line intervals by under 0.02 points except at the edges (C_3 α = 0.5: line [+0.008, +0.302] → page [−0.008, +0.317]; C_9 α = 1.5: [+0.000, +0.355] → [−0.015, +0.365]; O_18 α = 1.5: [+0.000, +0.245] → [−0.016, +0.246]); the large costs (L_12, L_14, L_16 at α = 0.5: +1.2 to +2.3 points) are unchanged. The VLM page-level intervals of PHASE5 were already page-level.

**Planted-defect table with page-clustered intervals** (`docs/controls/planted_table.md`, `planted.json`): every cell of §2.2 now carries the page interval beside the line interval; classes are decided on the page interval. No class of §2 changes; the intervals widen by a factor of about 1.5–3 on the large effects (k1 (a): −10.09 [−10.52, −9.65] pages against [−10.26, −9.91] lines; k6 (a): −11.62 [−13.36, −10.07] against [−12.04, −11.24]).

## 9b. Planted defects: diffuse corruptions and an unrelated donor (new runs `runs/20261107-planted-k9…k11`, `…-u-…`)

Three new corruptions on the same model (held-out CER of the uncorrupted model 9.55 %): **k9** Gaussian noise on every weight tensor, 10 % of each tensor's norm (diffuse); **k10** noise on `L_14` at 60 % of the block's norm (k7 was 25 %); **k11** all three BiLSTMs ×0.9 (diffuse scale). And, for k1, k6 and k8, a variant in which the parent is not available and the only donor is `kraken-bern-hertwig-v2`, a model from another lineage (arms (a), (b), (c); (d) does not use a donor). Same protocol, same classes; one attempt per cell, Sonnet programmers.

| corruption | practice % | held-out % | damage (pts) | (a) | (b) | (c) | (d) |
|---|---:|---:|---:|---|---|---|---|
| k9 noise 10 % everywhere | 55.09 | 63.80 | +54.24 | **18.51 (83 %, repaired)**; kept: blank bias, `C_3` bias to the parent, an 8-tensor interpolation α = 0.25 toward the parent | no edit, 63.80 | no edit (no rule fired), 63.80 | 60.32 (6 %, not repaired) |
| k10 noise 60 % on `L_14` | 12.00 | 13.49 | +3.94 | 10.88 (66 %, partly); `L_14` 40 % toward the parent | 10.72 (70 %, partly); `L_14` 75 % toward the parent | 11.47 (51 %, partly) | 13.49 (0 %) |
| k11 all LSTM ×0.9 | 7.27 | 9.87 | +0.31 (null by the 0.5-point rule; interval excludes 0) | 9.55 (100 %) | 9.55 (100 %) | 9.55 (102 %) | 9.64 (74 %) |
| k1 zero row `e`, **unrelated donor** | 16.23 | 19.66 | +10.10 | no edit kept, 19.66 | no edit, 19.66 | 17.75 (19 %: donor rows copied) | — |
| k6 `L_16` ×0.5, **unrelated donor** | 19.93 | 21.17 | +11.62 | 9.55 (100 %, repaired: ×2 undone, no donor used) | 9.55 (100 %, repaired) | 16.33 (42 %, partly) | — |
| k8 zero `C_3`, **unrelated donor** | 100.00 | 100.00 | +90.45 | 99.52 (1 %): identity-like `C_3` guesses | 100.00, no edit | 100.00 | — |

Reading. (i) **A diffuse defect is found only by the programmer with passes, and only partly by the blind one**: k9 (noise on everything) is invisible to the diff against the parent at tensor level (no tensor stands out), the blind programmer correctly wrote "no localised defect" and did nothing, the fixed list of repairs fired no rule, and only (a) — which measured — found that pulling all non-output tensors 25 % toward the parent averages the noise away (63.8 → 18.5 %, 83 % of the damage). That is the one cell in the whole study where passes bought something that reading did not. It relies on the parent as an anchor (an interpolation toward a donor of the same lineage). (ii) k10 shows partial repair of block noise by interpolation toward the parent: 51–70 %; none restores the block. (iii) **The repairs of localised defects depend on the donor's lineage**: with an unrelated donor the zeroed row and the zeroed block cannot be rebuilt (k1-u: 0 % for (a), (b); 19 % for (c)'s blind row copy; k8-u: 0–1 %), whereas a scale defect (k6-u) is repaired without any donor (the block's norm is half its neighbours'). (iv) k11 (×0.9, +0.31) was repaired by three arms although it is below the 0.5-point damage threshold; k7 (noise 25 %, +0.49) was repaired by none: a scale defect leaves a norm signature, a noise defect does not.

## 9c. Different programmers on the validation sessions (new runs `runs/20261107-val-*-o1/h1`)

Opus 5.5 (`o1`) and Haiku 5.5 (`h1`) attempts on the same packs (7 attempts per model in total: three Sonnet, one Opus, one Haiku, one Fable 5.1 `f1` run on 09.11.2026, one replay of the 06.11. Fable program). `xix-blb-v1` (practice `blb-val-300`): **0 of 7 attempts kept any edit** (Opus 5 passes / 4 edits; Haiku 5 / 4; Fable 10 / 9); the final model is the subject in every case. `combined-v1` (practice `bern-val`, held-out `bern-eval`, line / page-clustered intervals):

| attempt | passes / edits / kept | practice C0 → last | held-out Δ points | line [95 %] | page [95 %] | kept |
|---|---|---|---:|---|---|---|
| a1 = a2 = a3 (Sonnet ×3, one final model) | 8–9 / 7–8 / 2 | 13.86 → 12.35–12.38 | −0.387 | [−0.693, −0.082] | [−0.838, +0.035] | blank −0.3; non-output α = 0.25 toward the sibling |
| o1 (Opus 5.5) | 9 / 8 / 3 | 13.86 → 12.92 | −0.193 | [−0.562, +0.188] | [−0.788, +0.369] | blank −0.25 (after −0.5 refused); α = 0.25; a further 0.2 toward the sibling |
| h1 (Haiku 5.5) | 7 / 6 / 1 | 13.86 → 12.51 | −0.314 | [−0.607, −0.024] | [−0.771, +0.086] | non-output α = 0.25 toward the sibling |
| f1 (Fable 5.1, 09.11.) | 6 / 5 / 1 | 13.86 → 11.60 | −0.018 | [−0.426, +0.382] | [−0.623, +0.603] | non-output tensors α = 0.5 toward the sibling (judged on the 297 sibling-unseen lines) |
| replay (Fable 5.1's 06.11. program with masks) | 9 / 8 / 1 | 13.86 → 12.17 | −0.320 | [−0.697, +0.071] | [−0.840, +0.189] | `L_12` from the sibling |

All six attempts that kept something moved the model toward the sibling; the held-out gains (−0.02 to −0.39) follow how far they went and none has a page interval excluding 0. Fable's `f1` went half-way (α = 0.5) in one step: practice falls the furthest of any attempt (13.86 → 11.60, −2.3 points) and held-out does not move (−0.02), exactly the α = 0.5 row of §6: the practice set rewards the overshoot that held-out does not. The attempts are more varied than the three Sonnet runs were, but still not independent of the pack, which names the sibling as a donor and gives its CER on the practice lines.

## 10. Not done in the follow-up

- The Fable run (two attempts) failed on 07.11.2026 for lack of usage credits and was run on 09.11.2026; see §9c.

- **The second independent attempt per planted-defect arm with a different pack ordering** was not run (it would be 16 further subagent sessions and the Fable credits had run out); the new programmers of §9b and §9c are separate attempts on new corruptions and new models, not repeats of the original eight cells. The variance of the original cells across attempts remains unmeasured.
- Averages with `hertwig-v2` and of three siblings (C2 variants), a seed-variance null, and C4.
- The sham of §6 is one family (Gaussian noise at the interpolation delta's per-tensor norm); a sham that keeps the *direction* of the delta but reshuffles it across tensors was not run.
- Everything is CPU, one subject per pair, one held-out set per model; the margins use the greedy reading and a window definition stated above.
- No session was discarded after its result was seen. Two errors (an unexported variable; a margin window that included neighbouring characters) were found and fixed before any number was used.

## 11. The planted-defect controls with Claude Fable 5.1 as programmer (11.11.2026)

Asteraix, 11.11.2026, branch `claude/planted-fable-1d89a1`. Kraken sessions on CPU 09:41–11:22 CET (logs 08:41–10:22 UTC); adapter p3 on card 0 only, 10:04–10:43 CET, while card 0 had ≥ 38 GB free (the adapter replicate held 7 GB); card 1 not used. Nothing registered, uploaded or served. Code: `scripts/run_planted_f1.py`, `scripts/run_planted_adapter_f1.py`, `scripts/aggregate_planted_f1.py`; data `docs/controls/planted_f1.{json,md}`; run folders `runs/20261111-planted-*-f1-*` (30).

**What was rerun.** Only the arms an LLM programmed: (a) the programmer with passes and (b) the blind budget-0 programmer, on every kraken corruption k1–k11 and the three `-u` (unrelated-donor) variants (28 runs), and on adapter p3 (2 runs; (a), (b) were run for p3 only on 07.11.). The fixed list (c) and the coordinate search (d) do not depend on the model and were not rerun. Same corrupted subjects (tensor digests in each `preregistration.md` match the 07.11. runs), practice sets (`blb-train-300`; `kurrent-val-32`), budgets (10 / 0; 6 / 0), keep rule, held-out sets (`blb-eval-hand`; score set `kurrent-val-holdout-64`), preregistration template and outcome classes. Run names: the original's with `-f1` after the arm letter (as `runs/20261109-val-*-f1-b10-1`).

**Programmers.** 30 fresh Claude Code subagents with `model: fable` (`claude-fable-5-1`), one per run, default effort, tools Read/Write/Bash (`py_compile` only). Each read only its brief (`runs/planted-f1/PROGRAMMER_BRIEF{,_UNRELATED}.md`, `runs/planted-adapter-f1/PROGRAMMER_BRIEF.md`: the Sonnet briefs with the model id and the read-restriction changed) and its pack, copied unchanged into an opaque folder (`runs/planted-f1/sNN/pack`, `runs/planted-adapter-f1/q01/pack`) so that the path does not name the defect; the map `runs/planted-f1/map.json` was written only after all 28 programs were in. None was told which defect was planted or saw the Sonnet runs, programs or results. As on 07.11., programs were written in full in advance and never run by the programmer (not the multi-round protocol of the adapter replicate `runs/20261110-kurrent-page-v2-f1-b10-1`, which would have changed the arm). Operator: a Claude Opus 5.5 session that knew the defects and the Sonnet results and wrote no hypothesis or edit.

### 11.1 Kraken, held-out `blb-eval-hand` (uncorrupted 9.55 %)

Cell: held-out CER % · recovered share of the damage · class (classes as §2, on the page-clustered interval of final vs corrupted). p/e/k = passes used / edits / kept.

| corruption | damage (pts) | arm | Sonnet 5.5 (07.11.) | p/e/k | Fable 5.1 (11.11.) | p/e/k |
|---|---:|---|---|---|---|---|
| k1 zero row `e` | +10.10 | (a) | 9.57 · 100 % · repaired | 5/4/3 | 9.56 · 100 % · repaired | 8/7/3 |
| | | (b) | 9.76 · 98 % · repaired | 0/1/1 | 9.76 · 98 % · repaired | 0/1/1 |
| k2 negate row `n` | +60.69 | (a) | 9.55 · 100 % · repaired | 3/2/1 | 9.55 · 100 % · repaired | 4/3/1 |
| | | (b) | 9.55 · 100 % · repaired | 0/1/1 | 9.55 · 100 % · repaired | 0/1/1 |
| k3 swap `e`,`n` | +18.20 | (a) | 9.55 · 100 % · repaired | 2/1/1 | 9.57 · 100 % · repaired | 6/5/1 |
| | | (b) | 9.55 · 100 % · repaired | 0/1/1 | 9.55 · 100 % · repaired | 0/1/1 |
| k4 blank bias +2 | +1.30 | (a) | 9.52 · 103 % · repaired | 4/3/1 | 9.53 · 102 % · repaired | 5/4/1 |
| | | (b) | 9.55 · 100 % · repaired | 0/1/1 | **10.86 · 0 % · not repaired** | 0/1/1 |
| k5 output weight ×2 | −0.03 | (a) | 9.53 · null | 5/4/0 | 9.53 · null | 5/4/0 |
| | | (b) | 9.55 · null (×0.5 applied) | 0/1/1 | 9.53 · null (no edit) | 0/0/0 |
| k6 `L_16` ×0.5 | +11.62 | (a) | 9.55 · 100 % · repaired | 6/5/1 | 9.55 · 100 % · repaired | 5/4/1 |
| | | (b) | 9.55 · 100 % · repaired | 0/1/1 | 9.54 · 100 % · repaired | 0/1/1 |
| k7 noise `L_14` 25 % | +0.49 | (a) | 10.04 · null | 6/5/0 | 10.04 · null | 5/4/0 |
| | | (b) | 10.04 · null | 0/0/0 | 10.04 · null | 0/1/1 |
| k8 zero `C_3` | +90.45 | (a) | 9.77 · 100 % · repaired | 7/6/1 | 9.77 · 100 % · repaired | 5/4/1 |
| | | (b) | 9.77 · 100 % · repaired | 0/1/1 | 9.77 · 100 % · repaired | 0/1/1 |
| k9 noise everywhere 10 % | +54.24 | (a) | 18.51 · 83 % · repaired | 8/7/3 | **50.99 · 24 % · partly** | 6/5/3 |
| | | (b) | 63.80 · 0 % · not repaired | 0/0/0 | 63.80 · 0 % · not repaired | 0/0/0 |
| k10 noise `L_14` 60 % | +3.94 | (a) | 10.88 · 66 % · partly | 5/4/1 | 10.88 · 66 % · partly | 4/3/1 |
| | | (b) | 10.72 · 70 % · partly | 0/1/1 | 10.84 · 67 % · partly | 0/1/1 |
| k11 all LSTM ×0.9 | +0.31 | (a) | 9.55 · 100 % · null | 5/4/1 | 9.67 · 64 % · null | 5/4/1 |
| | | (b) | 9.55 · 100 % · null | 0/1/1 | 9.87 · 0 % · null (no edit) | 0/0/0 |
| k1-u zero row `e`, unrelated donor | +10.10 | (a) | 19.66 · 0 % · not repaired | 3/2/0 | 19.66 · 0 % · not repaired | 5/4/0 |
| | | (b) | 19.66 · 0 % · not repaired | 0/0/0 | **15.28 · 43 % · partly** | 0/1/1 |
| k6-u `L_16` ×0.5, unrelated donor | +11.62 | (a) | 9.55 · 100 % · repaired | 6/5/1 | 9.55 · 100 % · repaired | 6/5/1 |
| | | (b) | 9.55 · 100 % · repaired | 0/1/1 | 9.55 · 100 % · repaired | 0/1/1 |
| k8-u zero `C_3`, unrelated donor | +90.45 | (a) | 99.52 · 1 % · not repaired | 7/6/2 | 86.07 · 15 % · not repaired | 8/7/4 |
| | | (b) | 100.00 · 0 % · not repaired | 0/0/0 | 100.00 · 0 % · not repaired | 0/0/0 |

**Adapter p3 (`lora_alpha` doubled; score set, 64 val pages, vLLM, not repeatable; clean 0.507, corrupted 3.764):** Sonnet (a) 0.517 · 100 % · 4/3/1, (b) 0.517 · 100 % · 0/1/1; **Fable (a) 0.517 · 100 % · 4/3/1, (b) 0.517 · 100 % · 0/1/1**. Both Fable programs halve every `lora_B` (back to the siblings' α); (a) then tries ×0.9 and ×1.1 around it, both refused. The final adapter is byte-identical to the Sonnet arms' (digest `f5adb9da…`), so both Fable scores are the cached score of that adapter (`scored_from_cache`), not new vLLM readings. Adapter defects p1, p2, p6 had no LLM arm on 07.11. and none here.

### 11.2 Reading

- **Same headline.** Over the 22 kraken cells with a defect (11 corruptions × 2 arms; k5, k7, k11 are null corruptions), Sonnet: 15 repaired, 2 partly, 5 not; Fable: 13 repaired, 4 partly, 5 not. On every localised defect against the parent (k1, k2, k3, k6, k8) and on k6-u both models repair in both arms, mostly to the same model (14 of 28 Fable finals have the Sonnet run's final digest). The blind programmer again does as well as the programmer with passes on localised defects; on adapter p3 all four LLM arms reach the same adapter.
- **Where Fable differs, four cells.** (i) **k4 (b), worse:** the blind Fable programmer judged the blank bias +2.11 above the parent to be "natural drift" (its threshold for a blank edit was 3.5) and instead reset two output-layer feature columns that its outlier test flagged; held-out unchanged (10.86 vs 10.85, page interval [−0.02, +0.02]). Sonnet's blind programmer set the blank back and repaired it. (ii) **k9 (a), worse:** Fable treated the diffuse noise as "blank out-votes the characters", kept two blank-bias steps (−2.5 in all) and a transplant of the parent's `C_3` bias, and never tried pulling the network toward the parent; 24 % of the damage against Sonnet's 83 % (an 8-tensor interpolation toward the parent). The one cell of §9b where the passes bought something is therefore not reproduced by a second model: it depended on the programmer thinking of the interpolation. (iii) **k1-u (b), better:** with only the unrelated donor, the blind Fable programmer did not copy the donor's row but raised the zeroed `e` row's bias to +1.5, so `e` wins where every other class is suppressed; 43 % of the damage, where every other arm got 0–19 %. (iv) **k8-u (a), better but not repaired:** a donor `C_3` graft, ×2 and two blank-bias steps, 100 → 86 % CER (15 %; Sonnet 1 %).
- **Null corruptions.** Fable's blind arm made no edit on k5 and k11 (Sonnet's blind arm applied ×0.5 and ×1/0.9, both harmless or helpful); on k7 it applied one unmeasured output-column reset with no held-out effect. On k11 (a) Fable kept a blank −0.5 (64 % of the 0.31-point damage) where Sonnet found the ×0.9 scale (100 %).
- **Passes.** Fable used 77 passes in the 14 (a) runs, Sonnet 72; Fable's programs were longer (many run-time outlier scans against the parent, guarded probes for the space bias, `C_3` bias and an `L_16` spike), and the keep rule refused every one of those probes except where noted above. Several Fable programmers read the most negative output bias (−7.85) as the space's and suspected a planted space defect. That row is the no-break space (U+00A0), which the pack's `describe.md` prints between backticks like a plain space; the space's own bias is +0.16, and both are the same in the uncorrupted model. This is a defect of the pack's rendering, present for the Sonnet programmers too. The rule refused every space-bias edit that was measured; the blind programs that planned a space edit (k10, k11) gated it on a comparison with the parent, and none fired.
- **What it means for R7.** The control's claim survives a second programmer model: localised, reference-visible defects are found and repaired, and blind as well as with passes. The one cell where passes alone made the difference (k9 (a)) is programmer-specific, so "the programmer who measures repairs diffuse damage" holds for one of two models, one attempt each. One attempt per cell and model; the variance across attempts is still not measured.

### 11.3 Deviations and failures

- The adapter pack's title named the defect (`# Reading pack — adapter p3-alpha2`); the Sonnet programmers saw it. For Fable it was replaced by `q01`; nothing else in the pack changed (it still prints `lora_alpha` 256 next to the siblings' 128).
- The six blind (b) kraken programmers launched first were told only after launch (by message) not to read the sibling `a/` folder of their `sNN`; the later ones had it in their task and the brief was tightened mid-way to "only your own `sNN/pack/` and your own `a/` or `b/` subfolder". Every (b) programmer reported that it did not read `a/`. Two (a) programmers (s05, s12) disclosed grepping the `Metrics` dataclass in `wsp/metrics.py` (not on the list) for field names.
- Not all 28 programmers ran at once (limit of 20 concurrent subagents); seven (b) programmers started as slots freed.
- First driver launch ran nothing: `scripts/run_planted_f1.py` imported the main checkout's `wsp` (no `wsp.plant` there); no run folder was created; relaunched with `PYTHONPATH` set to this worktree.
- Adapter p3 (a) was started twice before it ran: attempt 1 aborted before pass 1 (pass server not yet READY); attempt 2 queued its C0 pass on a server that then died on attempt 1's stale job (the first server launch had also failed, `ninja` not on `PATH`). Both run folders recorded 0 passes and no measurement and were deleted before any result was read; the third attempt is the run reported. No session was discarded after a result was seen.
- Held-out scores reuse the cache when the final digest equals an earlier one (`scored_from_cache`), as in §1.
- `diff.json` committed as `diff.json.gz` when under 5 MB, otherwise omitted with `DIFF_OMITTED.txt` (6 runs).
