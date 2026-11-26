# TrOCR stage 5: preregistration of the comparators (written 25.11.2026, before any run)

Status: **harness written and checked; no comparator run, no sibling-average or ceiling read, no sham, no confirmation read has happened.** Plan: `docs/TROCR_STAGE5_PLAN.md` (main, 850c852). Tobias's answers to the decisions of its section 7 ("Take proposed defaults", relayed by the planning session, 25.11.2026): (1) C4 runs directly on card 0 with the trainer's own script, nothing registered, as in stage 2; (2) the confirmation half is read once, for arm I and at most one other model per subject chosen by the rule of section 7 below (the read itself is **not yet approved**); (3) C4-big is included; (4) the sibling checkpoints are looked for first, read-only, and **not regenerated** if absent; (5) the ceiling comparators for B are included; (6) the parents' corpora question is not part of this stage.

**Not approved until Tobias says so:** any C4 training run (small or big), any sibling-average or ceiling-comparator reading, any sham, any read of the confirmation half. Registry, Hugging Face and trainer writes are never part of this stage.

## 1. Question

Stage 4 left one strong baseline: a single interpolation toward the donor (arm I: A α 0.6, discovery Δ −4.09; B α 0.3, −3.36) beats every programmer result (R18). Stage 5 asks what else reaches what arm I reaches: (a) continued training of the subject on the same practice data the programmers had (C4); (b) an average with same-recipe siblings instead of the parent; (c) does arm I replicate on lines nobody has chosen on (the confirmation half). Everything below is fixed before the first run.

## 2. Subjects, data, split

- **Subjects:** A = `dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2` (stage-4 subject A; parent `trocr-kurrent-XVI-XVII`), B = `dh-unibe/trocr-kurrent` (stage-4 subject B; donor of arm I `microsoft/trocr-base-handwritten`). Local copies as in stage 4 (`~/atr-cache/trocr/`).
- **Practice set** (the only training data of C4-small, and the only data any choice below may use): `data/practice/trocr-blb-train-1200-seen.jsonl`, 1,194 lines, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`; the trainer reads it directly (its `JsonlDataset` takes `image` and `text`, extra keys are ignored; image paths are absolute).
- **C4-big training data:** every candidate line of the BLB **train side** (`lassberg`, `korrespondenten`, `unbestimmt`, `manuskripte`; never `*_eval`; 38,724 candidates before the sampler's own skips), built once with `python -m wsp.data build --set trocr-blb-train-all --source pagexml:<the four directories> --n 0`, about 8 GB of crops in `~/atr-cache/wsp/practice/trocr-blb-train-all`. Before any C4-big run: (i) an assertion that its object set is disjoint from the objects of `blb-eval-hand` and of `blb-eval` (a failed assertion stops the arm); (ii) the overlap audit of `scripts/trocr_audit_trained.py` against the training datasets of both subjects' parents is run and its counts are reported with the result; overlapping lines are **not** removed (they are lines the parents have trained on; the point of C4-big is what a data person would do with all the material). The practice lines are inside this set.
- **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, split by object with seed 20261119 into the **discovery half** (17 objects, 1,147 lines) and the **confirmation half** (12 objects, 1,125 lines, never read). Every frozen model of this stage is read once on the discovery half; the confirmation half is read only under section 7.

## 3. Arms and fixed settings

| arm | what | subjects | seeds |
|---|---|---|---|
| **C4-small** | continued training of the subject on the 1,194 practice lines, **5 epochs** (150 steps each, 750 steps), lr **1e-5** (primary) and **5e-5** (secondary) | A, B | 1, 2, 3 |
| **C4-big** | continued training on the whole BLB train side, **1 epoch** (about 4,840 steps), lr **1e-5** | A, B | 1, 2, 3 |
| **Sibling average** | A only: with the six same-recipe siblings (stage-2 seed-null checkpoints `s1 … s5`, `s42`, `~/atr-cache/trocr/seed-null/`): `sibpair-<s>` = A + 0.6·(s − A) for each of the six; `sibmean` = A + 0.6·(mean of the six − A); `sibavg7` = uniform average of A and the six. α = 0.6 is arm I's α on A, so the mixing weight is equal to arm I's | A | – |
| **Ceiling comparators** | B only: B + α·(A − B) and B + α·(D − B), A = `from-kurrent-v2`, D = `trocr-kurrent-XVI-XVII`, α in the grid of arm I (0.1 … 0.9), practice first; only the best α per comparator (lowest practice CER, ties to the smaller α) is read on the discovery half. Each pair is checked for equal tensor names, shapes and tokenizer vocabulary first; a mismatch skips the comparator and is reported | B | – |
| **Arm I** | arm I as read in stage 4 on the discovery half, and (section 7) once on the confirmation half at the α chosen on practice in stage 4 (A 0.6, B 0.3) | A, B | – |
| **Sham floor** | five shams per sham-tested C4 final (section 8) | A, B | – |

lr, epoch count and seeds are fixed here. Nothing is tuned; no arm is added or dropped after seeing a result (a run that fails for a harness reason is rerun unchanged and logged; section 10).

**Recipe of C4** (the job's recipe of `docs/TROCR_NULLS_PREREG.md`, `engines/trocr_train_svc/train_trocr.py` of `/home/tobias/Repo/training-atr-models`, same code and venv `.venvs/trocr-train`, transformers 4.57.6): batch 8, no accumulation, cosine schedule, warmup ratio 0.1, weight decay 0, max grad norm 1, `adamw_torch`, bf16, gradient checkpointing, 8 loader workers, `set_seed(seed)`, aligned loss (`use_aligned_loss`), beam 4 / no-repeat-3 generation settings of the job. Run by `scripts/trocr_stage5_train.py`, which imports the trainer's own `JsonlDataset`, `TrocrDataCollator` and `use_aligned_loss` and builds the same `Seq2SeqTrainer`. **Starts from the subject's weights** (`--base-model` is the local directory of A or B), never from a hub parent.

## 4. Deviations from the recorded recipe, known in advance

1. **No validation set, no best-checkpoint selection**, no per-epoch evaluation: the final state after the fixed number of steps is the model (the job selected by eval loss on its own validation split; here a selection would be a choice on data other than practice). `save_strategy` is `no`; the final model is saved once.
2. The data is the practice set (or the whole train side) as cut by `wsp.data` (`cut_line`), not the job's compiled crops. These are the crops the readings of stage 4 use.
3. The script is a thin wrapper around the trainer's pieces, not the unmodified `train_trocr.py` (which requires a validation manifest and selects the best checkpoint). The training repository is not modified.
4. The subject is read by `wsp.trocr_subject.TrocrSubject` (float32, beam 4, no-repeat-3, 256 new tokens), as in stage 4, not by the trainer's evaluator.
5. Sources of non-determinism are as in the seed null (bf16, GPU reductions, gradient checkpointing, loader workers); the three seeds give the spread.

## 5. Readings and statistics

- Δ = CER(model) − CER(subject) in points, paired per line against the unedited subject (cached baselines of `docs/trocr/stage4_grid.jsonl` on practice and `docs/trocr/stage4_discovery.jsonl` on the discovery half), page-clustered bootstrap 95 % interval (`wsp.metrics.bootstrap_delta`; 1,000 resamples on practice, 2,000 on the discovery half). "vs arm I" is the paired difference CER(model) − CER(arm I) on the same lines with the same interval.
- Every C4 final, sibling and ceiling variant is read on **practice**; selection of what goes on to the discovery half: **all C4 finals (all seeds, both lrs) and all sibling variants are read on the discovery half; for the ceiling comparators only the best α per comparator.** No other selection exists.
- **Arm-level reading of a C4 setting** (subject × recipe): the per-line mean edit count over its three seeds (`wsp.stage5.seed_mean_edits`), with the page-clustered interval of that mean against the subject and against arm I. The class of the setting (section 6) is taken on this arm-level reading. The three seeds are also reported singly; the spread of the seed Δ is reported next to R16's (sd 0.14 points on held-out CER).
- Minimum effect 0.5 points (R16), as the keep rule of stage 4.

## 6. Classes of a comparator against arm I

Let own = (Δ against the subject, interval) and diff = (CER(comparator) − CER(arm I), interval), both on the discovery half, `m` = 0.005. Tested in this order (`wsp.stage5.classify`):

1. **harms**: the own interval lies entirely above +m (the comparator is worse than the unedited subject by more than the minimum effect).
2. **beats**: the diff interval lies entirely below 0 (better than arm I).
3. **below arm I**: the diff interval lies entirely above 0 (arm I is better). This includes comparators with a smaller gain than arm I and comparators with no gain at all.
4. **matches**: the diff interval contains 0, the own Δ is below −m and the own interval lies entirely below 0 (a gain that cannot be told from arm I's).
5. **unresolved**: the diff interval contains 0 and the comparator is not itself a gain by test 4.

The classes are descriptive. This definition is stage 5's; it adds nothing to section 7 of the stage-4 preregistration (the class of P1, P2 and R there, which the planning session and Tobias decide), and the stage-4 sessions are not re-classed by it.

**What each outcome means (written now):** C4 (either size) or a sibling variant **matches or beats** arm I: the stage-4 gain is a weight-averaging or data effect that a plain retrain or a seed average reaches, and it is reported as that; **below** for both: interpolation with a related model reaches something that the subject's own recipe and its siblings do not, and the donor is worth a second look (stage 6, not this stage); **harms** is reported as overfitting or damage of that setting (the 5e-5 arm is the one expected to show it); **unresolved** is reported as unresolved, not as a match.

## 7. Confirmation half: read once, with a rule written first

- **What is read:** per subject, arm I at its stage-4 α (A 0.6, B 0.3) and **at most one other model**, once each, on the 1,125 lines of the confirmation half, paired against the unedited subject read on the same lines, same statistics as section 5.
- **Which other model** (`wsp.stage5.select_other`, applied per subject after all discovery readings are in): among all models of section 3 other than arm I (each C4 seed final singly, each sibling variant, each ceiling comparator at its best α), those whose discovery interval lies entirely below −0.005 are eligible; the eligible model with the most negative discovery Δ is read (ties by name). If none is eligible, only arm I is read. A C4 seed final stands for itself here, not for its setting.
- **What the read decides:** whether arm I replicates. If arm I's Δ on the confirmation half has an interval above −0.5 points (upper end ≥ −0.005), the stage-4 baseline does not hold on unread lines and R18 gets a correction. The comparison of the other model with arm I on the confirmation half is reported with its class (section 6) as a replication of the discovery-half class, nothing else. No other result may be changed or chosen after this read; there is no second read.
- **Not part of this preregistration's approval:** the read itself. It is requested separately, after the discovery results are in.

## 8. Shams

For the **seed-1 final** of each (subject, setting) of C4-small (lr 1e-5 and 5e-5) and C4-big, six finals in all: five shams each (same tensors, same Frobenius norm of the net change final − subject, seed 20261119, `wsp.stage4.sham_deltas`), read on the **discovery half** only (30 reads, about 1 h). Seeds 2 and 3 are not sham-tested. A C4 final's Δ is compared with the largest of its five shams as in stage 4 (reported, not a gate here).

## 9. Siblings and the check that they exist

Checked read-only on 25.11.2026 on this host (dhserver03 = asteraix): `~/atr-cache/trocr/seed-null/{s1,s2,s3,s4,s5,s42}/model.safetensors` exist (1.3 GB each; recipe of job 20261026T064540Z, base `trocr-kurrent-XVI-XVII`, 3 epochs, seeds 1–5 and 42), config files identical to A's, vocabulary identical, all 478 tensor names and shapes identical to A's (`wsp.stage5.check_compatible`). **Nothing is regenerated.** The seed-42 repeat is a sibling of the same recipe and data, with the recorded seed; it is averaged like the others (stage 2 labels it "same seed"; here it is one of six siblings and is not pooled or counted differently). B has no same-recipe siblings.

## 10. Rules of running, stops

- **Cards:** card 0 only, and only when `wsp.card.ready` says at least 12 GB free and no trainer job waiting; card 1 is never used; `TMPDIR=~/atr-cache/tmp`. Training and readings wait for the card, they never take it. Stage 4's rule of the Landlock sandbox does not apply (no programmer).
- **Route:** the trainer's script's pieces directly, nothing registered, published or uploaded; weights stay in `~/atr-cache/trocr/stage5/`.
- **Disk:** about 1.3 GB per final; the host has little room (49 GB free on 25.11.2026). Run stops (and is reported) if less than 8 GB are free; after a model is read on practice and the discovery half and (if chosen) counted for the selection of section 7, finals that cannot be selected are removed only after the selection of section 7 has been made, and the digest of every removed final is kept in the log.
- **Stops:** non-finite loss, a crash, card taken, disk, a vocabulary/shape mismatch, or any read of data outside practice and the discovery half. A run that stops for a harness reason (not a result) is rerun unchanged and logged as an amendment; results of the first run are kept in the log but not used. Nothing in this stage may read the confirmation half before the go-ahead of section 7.
- **Checks before the first run** (done in this session, results in the report): (1) `scripts/trocr_stage5_train.py` with `--max-steps 0` reproduces the subject: every tensor equal, practice reading equal to the cached C0 of the subject; (2) a 3-step run gives finite loss, a model that differs from the subject and measures the cost per step; (3) `tests/test_stage5.py` and the stage-4 tests pass. Checks run on practice data only.

## 11. Cost (estimate; the 3-step run replaces it with a measured value)

C4-small: 12 runs of 750 steps (about 5 min each plus load) with practice and discovery reads (about 4 min): about 2 h of card 0. C4-big: 6 runs of about 4,840 steps (about 21 min each) with reads: about 3 h. Sibling and ceiling variants: 8 sibling variants and 18 + 18 ceiling grid readings on practice, plus 8 + 2 on the discovery half: about 1.5 h of card 0 (they are card readings of about 2 min, not CPU reads, unless the card is unavailable). Shams: about 1 h. Confirmation read: about 20 min. Total about 7 to 8 h of card 0.

## 12. Order

Harness checks (done before this preregistration is approved for running) → report → [go-ahead] → C4-small → sibling and ceiling readings → C4-big → shams → discovery summary and selection of section 7 → [go-ahead] → confirmation read → `docs/trocr/stage5_results.md`. The run stops after the harness check and again before the confirmation read.

## Amendment 1 (26.11.2026): the confirmation half of B has been read once already; selection rule adjusted (text only, no run, no read)

(1) **What happened elsewhere.** Another session (branch `claude/trocr-stage4-confirm`, commits 33dd289 and c249a3d, not merged here; amendment 6 of the stage-4 preregistration there, approved by Tobias in that session on 25.11.2026) read the confirmation half of `blb-eval-hand` once for subject B on 25–26.11.2026: the unedited `trocr-kurrent` (CER 29.83 %), the frozen edit `dec.NN.cross.o.{weight,bias}` × 1.15 (Δ −2.06 [−2.86, −1.31], best of its five shams +0.07) and arm I at α 0.3 (Δ −7.36 [−9.19, −5.47]); the edit is below arm I. Subject A was not read, and nothing else was. This preregistration said the confirmation half was unread (status line and section 1, written 25.11.2026); that was true when it was written and is no longer true for B. Results: `docs/trocr/stage4_B_confirmation.md` on that branch.

(2) **Consequences for stage 5, fixed now, before any run:**
- **B's arm I and B's `cross.o` ×1.15 edit already have their one confirmation read. Stage 5 does not read them again.** B's replication question of section 7 (does arm I's interval on the confirmation half reach above −0.5 points?) is answered by that read (upper end −5.47, it replicates) and is taken from it, not repeated. The read of the unedited B on that half is likewise reused as the paired baseline for any further B model; stage 5 does not read unedited B again. The per-line values are taken from that branch's result files, read only, without merging the branch.
- **Subject A:** arm I at α 0.6 and the unedited A are read once on the confirmation half, as in section 7 (the read itself is not approved yet).
- **The one "other model" per subject** (`wsp.stage5.select_other`) is selected as written in section 7 and for B only among the stage-5 models (C4 finals, ceiling comparators), never the `cross.o` edit, which is not a stage-5 comparator and has been read. For B this read, if any model is eligible, adds one model to the confirmation half of B, which is then read by two sessions in all (their one read each); the selection is made on practice and discovery as before and is not changed by what the other session found. If none is eligible, B gets no further read.
- **Nothing else changes:** the classes of section 6, the discovery-half readings, the shams, the arms, lr, epochs and seeds, the stops. The confirmation half of A remains unread; the rule "at most one other model per subject, read once" stands.

(3) **Record of the harness checks (26.11.2026, card 0, after it freed up; results in the report):** zero-step copies of A and B equal the subject in all 478 tensors (CPU) and read exactly as the cached C0 on practice (A 23.07 %, B 28.22 %, Δ 0.0 with interval [0, 0]); 3-step runs finished with finite loss (A 1.88, B 2.99), the models differ from the subject (practice Δ −0.23 [−0.44, −0.02] for A and −0.34 [−0.55, −0.12] for B, which is only a check that training moved the weights); full tests pass. The status line at the top of this file (no run has happened) refers to the comparator runs and still holds.

## Amendment 2 (26.11.2026, before any stage-5 reading): clarification of Amendment 1 (text only; no rule changed)

Relayed by the planning session (main 2c2db73): B's confirmation half has been read once in stage 4 for the unedited subject, the `cross.o` × 1.15 edit, its five shams and arm I (α 0.3, Δ −7.36 [−9.19, −5.47]); A's half is unread. Therefore: (1) the arm-I replay of this stage is **A only**; B's arm-I replay exists. (2) A further B model read on that half would be a **second use of a half that has already seen the edit and arm I**; it is counted as such: the selection rule of section 7 and Amendment 1 is unchanged (at most one other model for B, chosen by `select_other`), but any such read is reported as the second use of B's half, together with the three models it has then seen, and it is requested separately from Tobias like every confirmation read (none is approved now). (3) The confirmation read of A (unedited A, arm I α 0.6, at most one other model) is not approved either and starts only on Tobias's separate word. (4) Stage-4 amendment 1's statement that no same-recipe sibling exists on this host was wrong; the six checkpoints exist (checked 25.11.2026, section 9).

## Amendment 3 (26.11.2026): the first C4-big run crashed on a degenerate crop; manifest filtered, rerun

Before this amendment: C4-small (12 runs) and the sibling and ceiling readings had been run as preregistered (results in `docs/trocr/stage5_reads.jsonl`). The overlap audit and the disjointness assertion of section 2 were run for the C4-big set before its first run: the set `trocr-blb-train-all` has 38,474 lines (250 of the 38,724 candidates skipped by the sampler for degenerate geometry), 291 objects, none in `blb-eval` or `blb-eval-hand`, no shared key (audit: `docs/trocr/stage5_audit_big.json`).

**Defect.** The first C4-big run (`c4big-A-lr1e-5-s1`, 26.11.2026 18:26, step 3154 of 4,810) crashed with `TypeError: Cannot handle this data type: (1, 1, 172)` inside the ViT image processor: a crop 3 pixels high is read as channels-first (the processor infers the channel axis from the first image of a batch and applies it to the whole batch), so the batch cannot be processed. This is a data defect, not a result. The practice set and both held-out halves contain no crop of height 1 or 3 (checked), so the C4-small runs and every reading so far are unaffected.

**Fix (the only change).** The C4-big manifest drops the 11 crops of height 3 (none of height 1), giving 38,463 lines (4,808 steps per epoch), `data/practice/trocr-blb-train-all-f.jsonl` (sha256 `7dae64e43939b15ffd4867b6b7523f2cadb8ccacc64073e22d50633d02b2a8f0`; dropped keys in `data/practice/trocr-blb-train-all-f.dropped.json`). Everything else is as preregistered. The first run's partial output is discarded (its log is kept as `s5_c4big-A-lr1e-5-s1.train.FAILED1.log`); no reading was made of it.
