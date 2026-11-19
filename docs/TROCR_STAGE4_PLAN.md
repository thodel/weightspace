# TrOCR stage 4: sessions on the Bern family. Plan (draft, 19.11.2026 CET)

Status: **draft for Tobias; nothing started, nothing preregistered.** Epic: `docs/epics/EPIC_TROCR_TRACK.md` (stage 4, answers of 11.11.2026: both `from-kurrent-v2` and `trocr-kurrent` are subjects). Inputs: R15 (lineage), R16 (seed null), R17 (planted-defect control, Gate 3 met). The preregistration (`docs/TROCR_STAGE4_PREREG.md`) is written only after the decisions of section 10 and before any session.

## 0. In one paragraph

Stage 3 showed that the harness, the keep rule and the programmers can repair a damaged model when the damage is localised and the parent is available, and mostly by looking the parent tensor up. Stage 4 asks the question that matters for the article: **on a healthy fine-tuned model, does a programmer who edits weights find an improvement that a one-parameter interpolation toward the donor, a sham edit of the same size, and retraining noise do not already give?** The kraken sessions answered "mostly no": the one held-out gain on a kraken model (session 2, −0.57) reduced to a 12.5 % step toward the parent, and the C4 retrain did at least as well (R11, R14). The TrOCR family differs in architecture (tied embedding, transformer encoder and decoder, a decoder that carries most of the movement), so the same question is open; a negative answer is a result and is reported as one.

## 1. What stages 0 to 3 settle, and what they leave open

| fact | source | consequence for stage 4 |
|---|---|---|
| Held-out CER differences under about 0.5 points are retraining noise (five seeds, sd 0.14, range 0.33) | R16 | minimum effect 0.5 points, as in stage 3 |
| Move directions do not reproduce across seeds (same-recipe cosine 0.36 encoder, 0.19 decoder); magnitudes do | R16 | the pack may offer magnitude traces as hypotheses and must not offer direction readings as evidence |
| The tied embedding is input and readout at once, moves most, and varies most | R15, R16 | embedding edits are a separate class; no output-bias analogue exists |
| With the donor, a repair is mostly a lookup of the parent tensor; the blind arm with no passes matched the arm with ten | R17 | a gain with the donor has to beat the interpolation toward that donor, not zero |
| Without the donor, 1 of 8 planted defects was repaired, and that rule was enforced by the brief only | R17 | the donor-free arm is the strong evidence; it needs a sandbox that does not contain the donor |
| Diffuse noise on every matrix (p10) was repaired by no arm; Fable 5.2 got 26 % | R17 | diffuse structure is where a programmer could add something |
| `revert` after `keep` crashed one session (p01) | R17 | harness fix before stage 4 (section 7) |
| C0 on `blb-eval-hand`: `from-kurrent-v2` 19.74 %, parent `XVI-XVII` 19.83 %, `trocr-kurrent` 26.41 % | STAGE0 | the first has no headroom against its parent; the second has headroom against its relatives (section 2) |

## 2. Subjects and donors

- **Subject A: `from-kurrent-v2`** (C0 19.74 %). Donor: its parent `trocr-kurrent-XVI-XVII` (19.83 %, practice 21.17 % against 21.45 %). The donor reads as well as the subject on the held-out set, so a gain from interpolation toward it is expected to be small; this subject tests whether a programmer finds something beyond the parent.
- **Subject B: `trocr-kurrent`** (C0 26.41 %). `from-kurrent-v2` and `XVI-XVII` read about 6.6 and 6.6 points better on the same held-out set (19.74 % and 19.83 % against 26.41 %), so **an interpolation toward them would "improve" it by construction**. That is a merge with a better model, not a finding about weights. Rule proposed: the donor of subject B is its documented ancestor only (the base model, per `docs/MODELS_TROCR.md`, to be checked in the preregistration); better relatives are not donors. The interpolation toward the best of them is run as a *ceiling comparator* and reported separately, never as the programmer's gain.
- **Donor rule (amendment A3, proposed, not adopted):** the paired interval is computed on the lines the named donor did not train on (`seen_by`). Stage 3 already applied it. The practice set's `seen_by` is empty for the subject (audited against its 20 datasets); for the parents' corpora it is an assumption (STAGE0 §8, item 4, still unanswered by the model authors). Proposal: adopt A3 for this track in the preregistration, and mask lines if the answer is yes.
- No third-party donors: `swe`, `stage1` have a different decoder; `lamop` and `ru` are not family members (R15).

## 3. Sets

- **Practice:** the stage-3 set (300 lines, 263 pages, 114 objects) is small; the BLB train side has 38,724 candidate lines. Proposal: draw a **larger practice set of 1,200 lines** (a different seed, objects disjoint from the held-out set, audited for both subjects and both donors) so that page-clustered intervals are tighter and a sham noise floor is measurable on it. Cost: a pass takes about 2.2 min on card 0 instead of 0.55. The 300-line set stays as the stage-3 comparison. Decision in section 10.
- **Held-out:** `blb-eval-hand` (2,272 lines, 29 objects), read once per frozen model, outside the sandbox. For the exchange grid and for the choice among kept edits it is **split by object into a discovery half and a confirmation half** (document-disjoint, split fixed by seed in the preregistration before any read). A cell or an edit is selected on practice, read on the discovery half, and only the best of them is read once on the confirmation half.
- **Out-of-domain control:** `bern-eval` (545 lines) read once for the finals that pass the held-out rule, as a control for damage elsewhere, after an overlap audit against both subjects' datasets (the Bern material may be in a parent's corpora).
- Generation settings pinned in every session as in stage 3: beam 4, 256 new tokens, no repeat of a trigram, length penalty 1.0, early stopping, float32, batch 8.

## 4. Arms (per subject)

| arm | who | donor | passes | what it tests |
|---|---|---|---:|---|
| P1 | Sonnet 5.5, fresh session | yes | 10 | a programmer with the donor |
| P2 | Fable 5.2, same conditions | yes | 10 | second family |
| P3 | the same two, **donor files absent from the sandbox** | no | 10 | a gain that is not a lookup |
| B | blind programmer (Sonnet 5.5), reads tensors against the donor, no pass | yes | 0 | whether the passes matter |
| R | replay: a second fresh session of the family with the better result, blind to the first | yes | 10 | reproducibility (the kraken adapter replicate was byte-identical) |
| I | global interpolation toward the donor, one parameter, line search on practice | yes | ≤ 10 | the comparator the kraken gain reduced to |
| S | five shams of the same norm as each kept edit (random direction, same tensor, same layer) | no | 0 | the noise floor of a kept edit (amendment A5, proposed) |
| M | margins: each kept edit at half and double size | no | 0 | whether the edit sits at an optimum |

Optional third family: a cross-vendor programmer using the tool layer of `docs/XVENDOR_PLAN.md` (PR #43), if it is ready; the epic asks for three programmers over at least two model families. Packs and briefs are built before the first session as in stage 3, with the non-breaking-space check, and the pack offers magnitude traces only (R16), no direction readings.

## 5. Exchange grid

For each (subject, donor) pair and for the reverse, on the practice set: whole encoder, whole decoder, each of the 12 encoder and 12 decoder layers, the tied embedding with its norm, and the layer norms, moved from one model to the other at full size and at half size (about 28 cells per pair and size). Shape and tokenizer identity are checked (R15). Cells are ranked on practice, the top six are read once on the discovery half, and the best is read once on the confirmation half. Preregistered readings: the exchange between two runs of the same recipe from one parent is a sanity case (R16 puts their weights 0.108 apart), while an exchange across lineages is expected to fail, as unrelated kraken models did. This stage does not need a programmer.

## 6. Keep rule, outcome classes, Gate 4

- **Keep rule as stage 3:** a pass since the edit, page-clustered paired 95 % interval of CER(after) − CER(last kept) entirely below −0.005 on the lines the donor did not train on; budget 10 passes; the harness refuses a keep otherwise.
- **Classes of a frozen model on the held-out set:** *gain* (the change is at most −0.5 points, the page-clustered interval is below 0, and the gain is confirmed on the confirmation half), *merge* (the gain is within the clustered interval of the interpolation arm I), *practice only* (a gain on practice, none held-out), *null* (nothing kept), *harm* (held-out worse by more than 0.5 with an interval above 0).
- **Gate 4 (a finding):** at least one frozen model is a *gain* that is also larger than the largest of its five shams and than arm I, and is reproduced by its replay R in direction. Anything else is reported as the result of the stage, with the classes of every session: practice-level gains that vanish held-out are the expected outcome of every earlier track (epic, risks).

## 7. Harness work before the preregistration

1. `Session.revert` after `keep` raises a clear refusal instead of a crash (the p01 failure); programs are not patched afterwards.
2. A sandbox mode without the donor files, checked by a test that lists what the session can see (stage 3 enforced the no-donor rule by the brief).
3. A sham generator (same tensor, same norm, seeded), a margin runner, an interpolation line search, and the object split of the held-out set.
4. Run `tests/` and a no-op session on both subjects, as in the stage-3 harness check.

## 8. Cost

On card 0 (observed: a 300-line pass 0.55 min, a held-out read about 3.3 min): per subject about 7 programmer sessions (P1, P2, P3 on two families, B, R) at 10 passes, or 14 for both subjects; at the 1,200-line practice set a pass is about 2.2 min, so the sessions need about 5 h of passes and the held-out reads (about 14 finals plus the grid's top cells, shams and margins) about 2 to 3 h; the grid on practice about 1 h. About 9 to 10 h of card 0 in all, with the usual rules (12 GB free, no trainer job waiting). Programmer sessions cost tokens in the order of the 32 sessions of stage 3 (about 14 here); not estimated. **No programmer session starts without Tobias's go-ahead** for the preregistered set.

## 9. Risks and stops

- The practice-to-held-out gap (gains of up to ten points on practice that vanish held-out) is the expected result; a null here is informative.
- Subject B's headroom invites "improvements" that are only merges with a better model; the donor rule of section 2 and the ceiling comparator are there for that.
- The parents' corpora may contain BLB material; until the authors answer, `seen_by` is an assumption and is stated as one.
- One held-out hand (`blb-eval-hand`) and one family: the second architecture family (large models, Swedish and CATMuS fine-tunes, answer of 11.11.) comes after this stage.
- Stop after the preregistration (cost, your OK), after the grid (before programmer sessions if the grid already shows a large exchange gain that the sessions would only rediscover), and after Gate 4.

## 10. Decisions for Tobias

1. **Both subjects** (as answered on 11.11.), with the donor of `trocr-kurrent` restricted to its ancestor (section 2)? Or `from-kurrent-v2` only first?
2. **Adopt A3 (donor rule) and A5 (sham noise floor)** for the TrOCR track in the preregistration?
3. **A larger practice set** (1,200 lines, audited) instead of the 300-line set?
4. **Programmers:** Sonnet 5.5 and Fable 5.2 only, or add a cross-vendor family (needs the PR #43 tool layer)?
5. **Donor-free arm P3** with a sandbox that does not contain the donor (needs a harness change, section 7)?
6. **Stage 5** (C4 retrain and the average of siblings through the trainer, about 15 GPU hours): later, after Gate 4, with its own OK.
7. The parents' corpora question to the model authors (STAGE0 §8, item 4): has it been sent?
