# TrOCR stage 5: the comparators, plan

> **Done 26.11.2026 on the discovery half; results in R19 (`docs/RESULTS.md`) and `docs/trocr/stage5_results.md`. The confirmation read was not run.** The text below is the plan as approved, kept unchanged.

Draft of 24.11.2026 (CET) for Tobias. Nothing in it runs until he has read it and given a go-ahead; the preregistration (`docs/TROCR_STAGE5_PREREG.md`) is written after his answers and before any run. Inputs: R16 (seed null), R17 (Gate 3), R18 (stage 4, Gate 4 not met); epic `docs/epics/EPIC_TROCR_TRACK.md` (stage 5: "C4 and the average of siblings, the same way as R11"); `docs/C4_RETRAIN.md` (the kraken C4 this copies).

## 0. Question

Stage 4 ended with one strong baseline: **a single interpolation toward the donor (arm I) beats every programmer result** (A −4.09, B −3.36 CER points on the discovery half; R18). The parent of A reads the held-out hand slightly worse than A itself (19.83 % against 19.74 %), so the gain is not "the donor is better": it is an average of two neighbouring models. Stage 5 asks what else does what arm I does:

1. **C4, retraining.** Does continued training of the subject on the same practice data the programmers had (gradients instead of edits) give a held-out gain that matches or beats arm I?
2. **Sibling average.** Does an average of same-recipe siblings (the stage-2 seed reruns of A) give the same gain as an average with the parent?
3. **Does arm I itself replicate** on held-out lines no choice has touched (the confirmation half)?

The answers fix what the article can claim: if retraining or sibling averaging gives the arm-I gain, "interpolation" is a weight-averaging effect, not something about the donor; if neither does, the donor carries something the subject's own recipe does not.

## 1. Facts this rests on

| Fact | Source |
|---|---|
| Arm I: A α 0.6 toward the parent, discovery Δ −4.09 [−5.13, −3.10] (19.12 to 15.03); B α 0.3 toward the IAM base, −3.36 [−4.51, −2.23] (23.26 to 19.90) | R18, `docs/trocr/stage4_discovery.md` |
| Retraining noise: seeds 1–5 of the recorded recipe, held-out CER 20.04 % mean, sd 0.14, range 19.93–20.26; edits under about 0.5 points are noise | R16 |
| Recorded recipe: parent `trocr-kurrent-XVI-XVII`, 3 epochs, batch 8, lr 5e-5 cosine, warmup 0.1, bf16, 20,181 steps, 86 min on card 0 | `docs/TROCR_NULLS_PREREG.md` |
| The practice set (1,194 lines, 759 pages, 216 documents) is drawn from the BLB train side, disjoint by object from the held-out set; the audit finds no BLB dataset among A's training datasets | `docs/trocr/audit_trained_overlap_1200.json` |
| The kraken C4 (R11, R14) trained 5 fixed epochs at 1e-5 (primary) and 5e-5 (secondary), seeds 1–3, with no choice made on the held-out set | `docs/C4_RETRAIN.md` |
| The confirmation half (12 objects, 1,125 lines) was read once, for subject B only (edit −2.06, arm I α 0.3 −7.36 [−9.19, −5.47], five shams ≤ +0.36); subject A's half is unread | R18 (corrected 26.11.2026), `docs/trocr/stage4_B_confirmation.md` |

## 2. Arms

Per subject A (`from-kurrent-v2`) and B (`trocr-kurrent`), every arm read the same way as in stage 4: practice first, one frozen model read once on the discovery half; Δ in CER points against the unedited subject, page-clustered 95 % interval, and "vs arm I" as the paired difference.

| arm | what | subjects |
|---|---|---|
| **C4-small** | continued training of the subject on the **practice set** (all 1,194 lines, labels as in the practice set), 5 fixed epochs (750 steps), lr 1e-5 (primary) and 5e-5 (secondary), seeds 1–3, the trainer's own script; the same data the programmers' keep rule saw | A, B |
| **C4-big** (optional, decision 3) | the same with the BLB train side as training data (38,724 candidate lines, disjoint from the held-out set by object, to be re-audited), 1 epoch, lr 1e-5, seeds 1–3: what a data person would do with all the material | A, B |
| **Sibling average** | average of the recorded `from-kurrent-v2` with the stage-2 reruns (seeds 1–5 and the seed-42 repeat), uniform weights, all subsets of 2 and the average of all 7; compared with arm I at equal mixing weight | A only (B has no siblings) |
| **Arm I replay** | arm I at the α chosen on practice, read once on the **confirmation half** | A only (B's replay was read in stage 4, amendment 6: −7.36, replicates) |
| **Ceiling comparators for B** (CPU only) | interpolation of B toward `from-kurrent-v2` and toward `XVI-XVII`, α grid as arm I, practice then discovery; excluded from stage 4 by amendment 2 | B |
| **Sham floor** | five shams per C4 final (same tensors, same Frobenius norm of the net change, as stage 4) | A, B |

Selection discipline as in stage 4: lr and epoch count are fixed before any read; nothing is chosen on held-out data; the confirmation half is read once, for arm I and for at most one other model per subject, chosen on practice and discovery by a rule written into the preregistration.

## 3. Reading rules (fixed in the preregistration, sketched here)

- Minimum effect 0.5 points (R16), page-clustered paired interval entirely below −0.005 as in stage 4.
- Each comparator is classed against arm I: **beats** (paired interval below 0), **matches** (interval contains 0 and the comparator's own Δ is itself below −0.5), **below** (interval above 0), **harms** (interval above +0.005 against the subject). Same wording as the "below arm I" class that section 7 of the stage-4 preregistration has not defined; the stage-5 preregistration defines it.
- What each outcome means, written now: C4 or the sibling average **matches or beats** arm I, then the stage-4 gain is a weight-averaging or data effect that a plain retrain or a seed average reaches, and the article reports it as that. Both **below**: interpolation with a related model reaches something the subject's own recipe does not, and the donor is worth a second look (stage 6, not stage 5).
- Arm I on the confirmation half: if its Δ there has an interval above −0.5, the stage-4 baseline does not hold on unread lines and R18 gets a correction. For B this was already read and holds (−7.36 [−9.19, −5.47]); the test is open for A only.

## 4. Harness work

1. A data adapter from `data/practice/trocr-blb-train-1200-seen.jsonl` (and the BLB train side for C4-big) to the input of `engines/trocr_train_svc/train_trocr.py` of the training repo, checked first by a no-op run (zero steps must reproduce C0 of the subject exactly, as the stage-4 no-op check did).
2. The training script started from the subject's weights, not from a hub parent; verified by a zero-step run and a 3-step run.
3. `wsp/stage5.py`: sibling averaging by tensor name, a check that the siblings' tokenizer, config and embedding shapes match, and the weight sums.
4. The stage-4 reading code is reused (paired bootstrap, shams, discovery split).
5. Tests for the averaging and the zero-step check; no change to the held-out code paths.

## 5. Cost (estimates, derived from stage 2 and stage 4 timings, to be replaced by the first measured run)

Training speed in stage 2 was about 0.26 s per step (86 min / 20,181 steps); a practice pass of the 1,194 lines takes about 2 min, a discovery read about the same.

- C4-small: 750 steps are about 4 to 5 minutes; with load, practice read and discovery read about 10 minutes per run. Subjects A and B × 2 learning rates × 3 seeds = 12 runs, **about 2 h of card 0**.
- C4-big (optional): 4,840 steps per run, about 21 minutes of training and 30 minutes per run with reads; 2 subjects × 3 seeds = 6 runs, **about 3 h**.
- Sibling average and ceiling comparators: CPU reads only, **under 1 h**, if the seed-rerun checkpoints still exist. If they were deleted (stage 4's amendment 1 says "no sibling model of the same recipe is on this host", which needs checking), regenerating them is 6 runs of 86 min, about 8.6 h, or three of them, about 4.3 h, for the average of the recorded model with three siblings.
- Shams and confirmation read: about 1 h.

Total, with C4-big and without regenerating siblings: about **6 to 7 h** of card 0, in place of the 15 GPU hours the epic planned (that figure was for retraining on the full data and for stage 2(b)). Rules as before: card 0 only with at least 12 GB free and no trainer job waiting, card 1 never; nothing registered, no HF or trainer-registry writes; TMPDIR on `~/atr-cache/tmp`.

## 6. Risks

- **The trainer route.** The epic's decision 2 said stage-5 retrains "go through the trainer"; stage 2 instead ran the trainer's script directly on card 0 with nothing registered. This plan uses the direct route again (decision 1).
- **C4-small may overfit or damage.** 1,194 lines are few; at 5e-5 for 5 epochs the model may lose general ability. That is a finding (the lr arm reads it), and the primary arm uses 1e-5.
- **Leak through the practice set.** Practice and held-out are disjoint by object and audited; C4 trains on the practice labels, which the programmers' keep rule also used, so the comparison is fair, but a C4 gain says nothing about lines outside the BLB hand.
- **Sibling checkpoints missing** (section 5).
- **B has no same-recipe siblings**, so the sibling comparison is A only; B's story rests on C4 and the ceiling comparators.
- **The confirmation half has 12 objects**; its intervals are wide. Reading arm I there tests replication, not a small difference. B's half is already read (stage 4, amendment 6), so another model of B read there would be a second use of a half that has seen two models; the preregistration has to count that.

## 7. Decisions for Tobias

1. **Route.** Run C4 directly on card 0 with the trainer's script, nothing registered (as stage 2), instead of through the trainer's job mechanism. Proposed: yes, direct.
2. **Confirmation half.** Read it once, for arm I and for at most one other model per subject. I advised against reading it for Gate 4; for stage 5 the reading is a replication of the baseline that carries the article's claim. Proposed: yes, with the selection rule written first.
3. **C4-big.** Include the larger retrain (about 3 h)? Proposed: yes, as it is the strongest data-side comparator; drop it if card time is tight.
4. **Sibling checkpoints.** First check whether the stage-2 checkpoints exist (a read-only look on asteraix); if they do not, may stage 5 regenerate three of them (about 4.3 h) for the sibling average? Proposed: ask again after the check.
5. **Ceiling comparators for B** toward `from-kurrent-v2` and `XVI-XVII`, CPU only. Proposed: yes.
6. **After stage 5:** stage 6, the write-up (`docs/TROCR_REPORT.md`) and the manuscript section; the parents' corpora question of `docs/trocr/STAGE0.md` §8, item 4 is still open and should be sent before the write-up.

Run order, once approved: preregistration and harness (a session); no-op and zero-step checks; C4-small; sibling average and ceiling comparators; C4-big; shams; confirmation read; report. Stops after the harness check and again before the confirmation read.
