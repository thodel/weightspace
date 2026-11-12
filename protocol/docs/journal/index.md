# Journal

Dated entries in the order the work happened. Each entry says what was
planned, what was done, what was found and what it changes. Entries are not
edited after the fact except for typos and links; a correction is a new entry
that names the one it corrects.

| date | entry | in one line |
|---|---|---|
| 2026-11-06 | [Phase 0: harness, practice sets, baselines](2026-11-06-a-phase0.md) | Harness verified, a lossless 11 M-parameter rewrite, and a data defect that explains a bad fine-tune |
| 2026-11-06 | [Phase 1: first budgeted session](2026-11-06-b-phase1.md) | Output-layer edits do not move the model; no gain |
| 2026-11-06 | [Keep rule gets a minimum effect](2026-11-06-c-keep-rule.md) | Protocol amendment A1 |
| 2026-11-06 | [C3 per-block α-sweep](2026-11-06-d-alpha-sweep.md) | The fine-tune's gain is held jointly; only the last LSTM extrapolates |
| 2026-11-06 | [Phase 3: target 1 on correctly paired pages](2026-11-06-e-phase3.md) | The "broken" model is the best of its family; the kept transplant was a donor artefact |
| 2026-11-06 | [Phase 4: the collapsed lines are a data question](2026-11-06-f-phase4.md) | 0 of 428 collapsed lines are model failures; half the test set was a printed book |
| 2026-11-07 | [Phase 5: collapse detector and a handwriting held-out set](2026-11-07-a-phase5.md) | Amendment A2; `blb-eval-hand`, 2,272 lines |
| 2026-11-07 | [The served kraken family compared as networks](2026-11-07-b-model-family.md) | 43 models: front end generic, recurrent stack individual; a first exchange candidate |
| 2026-11-07 | [Weight survey of the kraken zoo](2026-11-07-c-zoo-survey.md) | Lineage readable from weights; names mislead, codecs don't |
| 2026-11-07 | [Seed-variance null](2026-11-07-d-seed-null.md) | Which fine-tuning traces survive five seeds |
| 2026-11-07 | [Controls for the article](2026-11-07-e-controls.md) | Planted defects repaired by everyone; the one gain is a merge |
| 2026-11-07 | [Where xix-blb-v1 finds and loses characters](2026-11-07-f-layer-probes.md) | Each LSTM has a job: find, distinguish, space |
| 2026-11-07 | [Which latin_incunabula candidate: neither](2026-11-07-g-candidate-decision.md) | Better on Fraktur, worse on abbreviations |
| 2026-11-07 | [Adapter track A0–A4](2026-11-07-h-adapter-track.md) | Looping is chaotic; −11.5 points held-out, in stopping, not reading |
| 2026-11-08 | [Phase 5 adapter session](2026-11-08-a-adapter-session-v2.md) | One attention direction; −10 points on practice |
| 2026-11-08 | [C4: retraining combined-v1 on corrected data](2026-11-08-b-c4-retrain.md) | The cheap gradient alternative beats the best edit |
| 2026-11-08 | [Phase 5 adapter session: no held-out gain](2026-11-08-c-adapter-heldout.md) | +0.24 points on 733 pages; the edit swaps which pages loop |
| 2026-11-08 | [latin_incunabula v3](2026-11-08-d-incunabula-v3.md) | Abbreviations spared, Fraktur gain kept; qualifies |
| 2026-11-08 | [Exchange on out-of-domain material and across lineages](2026-11-08-e-exchange-ood.md) | Real but small; unrelated models cannot exchange |
| 2026-11-09 | [A Fable programmer on the validation sessions](2026-11-09-a-fable-attempt.md) | Larger practice gain, no held-out gain |
| 2026-11-11 | [Phase 5 with a Fable programmer](2026-11-11-a-phase5-fable.md) | Same direction, same adapter, same held-out null; a random direction passes the keep rule |
| 2026-11-11 | [Planted defects with a Fable programmer](2026-11-11-c-planted-fable.md) | Localised defects repaired again, by both arms; the diffuse-noise repair does not reproduce |
| 2026-11-11 | [TrOCR track, stage 0](2026-11-11-d-trocr-stage0.md) | Seven models fetched, tokenizers identical, embedding tied; C0 deterministic, 19.7–26.4 % |
| 2026-11-11 | [TrOCR track, stage 1: the family as weights](2026-11-11-e-trocr-stage1.md) | Lineage readable for small fine-tunes; decoder moves, encoder hardly; data does not set the direction |
