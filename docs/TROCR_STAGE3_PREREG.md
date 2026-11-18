# TrOCR stage 3: planted-defect positive control, preregistration (written before any programmer session)

Written 18.11.2026 (CET). Epic: `docs/epics/EPIC_TROCR_TRACK.md` (stage 3, Gate 3). Predecessors: `docs/TROCR_NULLS.md` (noise floor), `docs/CONTROLS_REPORT.md` (the kraken control this one follows). Code: `wsp/trocr_plant.py`, `runs/trocr-planted/arms/`, `scripts/run_trocr_planted.py`, harness changes in `wsp/run.py`, `wsp/session.py`, `wsp/trocr_subject.py`. **Nothing in this document has been run by a programmer (LLM) yet.** The non-LLM arms (c) and (d) are run as a validation of the harness before the programmer arms; their programs are fixed here.

## 1. Subject, sets, settings

- **Subject:** `dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2` (public; tensor digest in `docs/trocr/DIGESTS.json`). Parent donor: `dh-unibe/trocr-kurrent-XVI-XVII`. Unrelated donor (not used in the first round): `dh-unibe/trocr-medieval-escriptmask` (other lineage, same shapes).
- **Practice set:** `data/practice/trocr-blb-train-300-seen.jsonl` (300 lines, 263 pages, 114 documents, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`), the BLB train side; audited against the 20 datasets of the subject (`docs/trocr/audit_trained_overlap.json`). The `seen_by` field is `[]` on every line: **an assumption** for the parent donor (its corpora, 293 k lines and 1.58 M words, are not audited; STAGE0 §8, item 4). If the authors answer that BLB material is in them, the donor rule masks those lines and the with-donor results are recomputed on the rest.
- **Held-out set:** `data/heldout/blb-eval-hand.jsonl` (2,272 lines, 29 objects), read **once per frozen model**, outside the sandbox; the score of a model is cached by digest, so a session that ends unchanged does not read it again.
- **Generation settings pinned in every session** (`TrocrSubject.GENERATION`, written in `session.json` and in each run's preregistration stub): beam 4, 256 new tokens, no repeat of a trigram, length penalty 1.0, early stopping, no sampling. Batch 8, float32.
- **Keep rule** (harness, `Session.keep_allowed`): a pass since the edit, and the **page-clustered** paired 95 % interval of CER(after) − CER(last kept) entirely below −0.005 (**minimum effect 0.5 points**, the retraining noise floor of `docs/TROCR_NULLS.md`). Donor rule: the interval is computed on the lines the named donor did not train on (`seen_by`). Clusters are the pages of `wsp.metrics.page_of`; 300 lines fall on 263 pages, so the clustered interval is close to the line-level one on this set.
- **Passes:** 10 per session (the first is the baseline), 0 for the blind arm. One pass = the 300 practice lines on card 0 (about 30 s of generation); on CPU about 9 minutes.
- **Sandbox:** network cut (`unshare -Un`), `-I` interpreter of the TrOCR venv, card 0 visible only for TrOCR sessions; each pass first reads `wsp.card.ready` (at least 12 GB free, no trainer job waiting) from the state file written by `python -m wsp.card --watch` running outside; a refused pass is not counted.

**Harness check (done 18.11.2026, before this document):** a no-op session (8 practice lines, budget 2, scale by 1.0, donor mask, page-clustered rule) runs end to end in the sandbox on CPU (52 s) and on card 0 (18 s), gives the same CER (13.04 %), refuses the no-op keep ("interval reaches −0.005") and freezes; `wsp.run` now accepts a TrOCR directory (`final.trocr`, `current.trocr`), a held-out pass on the card the session ran on, and the cluster rule; the 128 tests of the repository pass.

## 2. The defects (calibrated on the practice set, not on the held-out set)

Each defect is a pure function of the subject's tensors and a seed (`wsp.trocr_plant`; the planted models and their digests are in `docs/trocr/plant_models.json`). **A defect counts** (for Gate 3) only if its held-out damage is at least **1.5 points** of CER and its page-clustered interval lies above 0 (three times the noise floor of the null); 0.5 to 1.5 points is a *weak defect* (reported, not used), below 0.5 a *null corruption*. Calibration (practice CER of the subject 21.45 %): `docs/trocr/plant_calibration.json`, all 37 size trials; the token rows hit are the most frequent ids of **what the subject writes** on the practice set (a first calibration on the reference text hit spaced punctuation the model never emits: no damage). **There is no output bias in this architecture** (the readout is the tied embedding), so the kraken blank-bias and output-weight analogues do not exist; a bias defect lives in the decoder embedding layer norm.

| id | defect | scope | practice damage, points [page-clustered 95 %] | donor repair | repair without a donor |
|---|---|---|---:|---|---|
| p01 | rows of the 6 most frequent emitted tokens zeroed in the tied embedding (input and readout) | localised, token rows | +3.63 [2.86, 4.68] | parent rows | no (rows are gone) |
| p02 | rows of the 2 most frequent emitted tokens (`.` and `,`) swapped | localised, token rows | +1.98 [1.74, 2.26] | parent rows | yes: swap back |
| p03 | decoder layer 5 cross-attention output projection ×0 | localised, one layer | +1.75 [0.84, 2.75] | parent block | no |
| p04 | 8 of 16 cross-attention heads of decoder layer 6 zeroed (columns of the output projection) | localised, heads | +3.23 [1.85, 4.90] | parent columns | no |
| p05 | encoder layer 6 attention output projection set to 0 | localised, one layer | +11.71 [10.32, 13.10] | parent block | no |
| p06 | encoder layer 3 FFN output projection ×0.5 | localised, one layer | +2.01 [1.00, 3.21] | parent block or ×2 | yes: ×2 |
| p07 | Gaussian noise on decoder layer 8 (self, cross, FFN), 60 % of the layer's norm | localised, no signature | +1.76 [0.87, 2.54] | parent block | no |
| p08 | a fixed random direction added to the decoder embedding layer-norm bias, norm = 1.0 × that of its weight | bias shift | +8.39 [6.82, 10.38] | parent bias | partly (a bias search) |
| p09 | decoder layer 11 FFN layer norm ×0 (weight and bias) | localised, catastrophic (garbled loops, length ratio 11) | +1,022 | parent norm | no |
| p10 | Gaussian noise on every encoder and decoder weight matrix, 20 % of each | diffuse | +2.81 [1.77, 3.93] | parent (whole model) | no |
| p11 | Gaussian noise on the tied embedding, 50 % of its norm | diffuse, vocabulary | +1.90 [0.97, 2.94] | parent embedding | no |
| p12 | every encoder FFN output projection ×0.7 | diffuse scale | +5.91 [4.52, 7.22] | parent or ×1/0.7 | yes |
| p13 | every decoder cross-attention output projection ×0.7 | diffuse scale | +7.76 [5.76, 9.81] | parent or ×1/0.7 | yes |
| p14 | null corruption: decoder layer 11 FFN norm bias ×1.0001 | null | +0.00 | none needed | none needed |

Calibration also showed three things worth stating. Scaling every decoder FFN output by 0.9 or 0.7 costs nothing (−0.1 points: not a defect, dropped). Zeroing the rows of the two most frequent *reference* tokens costs nothing. Negating the rows costs the same as zeroing them (+3.68 against +3.63 for 6 rows), so the negation defect is not run separately. The donor column says whether the parent can supply the answer; "no" means no repair is possible without information the model no longer holds, so a programmer without a donor can only *repair* p02, p06, p12 and p13 (and partly p08).

**Core set for the programmer arms (8):** p01, p03, p05, p06, p07, p08, p10, p12. They cover token rows, three localised layers (one weak, one large, one scale), a defect without signature, a bias shift, one diffuse noise and one diffuse scale. The baselines (c) and (d) run on all 14.

## 3. Arms

| arm | who | donor | passes | program |
|---|---|---|---:|---|
| (a) programmer | Claude Sonnet 5.5 (`claude-sonnet-5-5`) as a fresh subagent, default reasoning effort, tools Read, Write and Bash (`py_compile` only); reads only its pack and the brief; writes its program in advance, does not run it | with the parent donor, and **without** | 10 | its own |
| (a2) second family | Claude Fable 5.1 (`claude-fable-5-1`), same conditions | with the donor | 10 | its own |
| (b) blind programmer | Sonnet 5.5, same conditions, reads the model's tensors against the donor but no pass | with the donor | 0 (`keep_unmeasured`) | its own |
| (c) hand-written rule list | none; `runs/trocr-planted/arms/program_c.py` (rules R1 to R6, thresholds from the healthy step `xvi-xvii` → `from-kurrent-v2`) | parent | 10 | fixed, the same for every model |
| (d) scale/bias coordinate search | none; `runs/trocr-planted/arms/program_d.py` (scales ×0.8 / ×1.25 of seven part groups, then the embedding-norm bias) | none | 10 | fixed |

The pack and the brief of the programmer arms (the model description, tensor names and shapes, the API of `wsp.run`, the keep rule, the donor handle) are written and committed **before the first programmer session**, as `runs/trocr-planted/PROGRAMMER_BRIEF.md`; they are not in this commit. The kraken pack's non-breaking-space defect (`CONTROLS_REPORT` §11.2) is checked first. Every arm is a *new session* per planted model, with a preregistration stub written by `scripts/run_trocr_planted.py` from the digests before the first pass.

## 4. Outcome classes, Gate 3

- **Repaired:** the frozen model's held-out CER recovers at least 80 % of the defect's held-out damage and the page-clustered paired interval of final against corrupted lies below 0. **Partly:** 20 to 80 %. **Not repaired:** under 20 %, or CER above the corrupted model's. **Harmful keep:** a kept edit after which the held-out CER is worse than the corrupted model's by more than the minimum effect with the clustered interval above 0.
- For p14 (null): any kept edit is reported as a harm or neutral; there is nothing to repair.
- **Gate 3 passes** if, among the defects that count (held-out damage at least 1.5 points), at least one is classed *repaired* by an LLM programmer arm (a, a2 or b) **with** the donor, **and** at least one by arm (a) **without** the donor (p02, p06, p12, p13 are the candidates), and no more than one session in the whole set ends in a harmful keep. If the donor-free setting repairs none, stage 4 may run only with a donor available and that limit is part of the result. If neither setting passes, **the sessions of stage 4 are not run**, because a null there would mean nothing. The outcomes of (c) and (d) are reported with the same classes; the claim that an LLM programmer adds anything is judged against (c) and (d), not against zero.

## 5. Cost

| item | count | GPU time on card 0 | notes |
|---|---:|---:|---|
| planted models | 14 | 5 min | 1.3 GB each, `~/atr-cache/wsp/planted-trocr/` |
| baselines (c), (d), all 14 | 28 sessions, about 180 passes | 1.7 h | practice-level repair first; held-out for the core 8 and for changed finals (about 6 h worst) |
| programmer sessions, Sonnet 5.5: (a) with donor, (a) without donor, (b), on the core 8 | 24 | 24 × ≤ 10 passes × 0.55 min ≈ 2.2 h | plus the programmer's own time: about 15 to 30 min per session |
| programmer sessions, Fable 5.1: (a2) on the core 8 | 8 | 0.7 h | |
| held-out passes | 8 corrupted + at most 32 changed finals (the rest are cached) | 15 min each, at most 10 h; expected about 5 h | 2,272 lines at 0.39 s |
| **programmer sessions in all** | **32** (24 + 8), of which 8 with no pass | **about 3 h of passes + about 5 h of held-out, at most 13 h** | a pass costs 0.5 to 0.6 min on card 0, about 9 min on CPU; programmers' token cost not estimated here |

Programmer sessions need Tobias's go-ahead (tokens; stop gate after Gate 3's design). Nothing of arms (a), (a2) or (b) has been started.

## 6. Limits stated in advance

One subject; the practice set is BLB train side while the subject's parent corpora are unaudited (assumption on `seen_by`); the defects are calibrated on the practice set so their held-out damage can differ (a defect that does not count on held-out is dropped from Gate 3, not tuned); the programmer arms share one practice set with the baselines; two model families; the noise floor of 0.5 points rests on five seeds (`docs/TROCR_NULLS.md`); the tied embedding is both input and readout, so p01, p02 and p11 are not comparable with the kraken output-row defects.

## Baseline results (added 18.11.2026, after the baseline runs; the text above is unchanged)

The two non-LLM arms were run on all 14 planted models at practice level. Table and readings: `docs/trocr/baseline_results.md`. Held-out scoring of the finals follows, one read per frozen model, before any programmer session. Programmer sessions are not started.
