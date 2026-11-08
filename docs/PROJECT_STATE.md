# Project state (7 November 2026, late evening CET)

One page for anyone who opens the repository cold. Details are in the files named.

## What this is

Weight-space programming (Takehi, Teoh, Zhang, Khattab, blog of 2 Nov 2026) tried on the in-house kraken text-recognition models of the Bern projects: read the weights, diagnose, edit, measure against held-out data and against ordinary merging. Planning on h1 (no GPU, no kraken), runs on asteraix. Article for JDMDH in `paper/`.

## Phases and tracks

| track | what | where | status |
|---|---|---|---|
| Phase 0-5 (A) | harness, readings, sessions on `xix-blb-v1` and `combined-v1`, collapse analysis, clean-up of held-out sets | `docs/PHASE0..5_REPORT.md`, `docs/RESULTS.md` R1-R3 | done |
| B | controls for the article: planted defects, unseen practice sets, sham, average, global interpolation, margins, seed null, zoo survey | `docs/CONTROLS_REPORT.md`, `docs/SEED_NULL.md`, `docs/ZOO_SURVEY.md`, `docs/RESULTS.md` R5-R8 | done, merged into main (8d7b577) |
| C | the family of 43 models: weights, lineage, function, CKA, exchange grid, candidate `latin_incunabula-wsp-v1` | `docs/MODEL_FAMILY.md`, `docs/BLOCK_STATS.md`, `docs/CERBERUS_NOTES.md`, `docs/family/`, `docs/RESULTS.md` R4 | done, merged |
| Adapters | sessions on a LoRA adapter of a vision-language model | branch `claude/adapters-4d972b`, `docs/ADAPTER_REPORT.md` | finished 7 Nov; **merged into main 8 Nov**; folded into manuscript 9.3 and RESULTS R10 |
| Article | JDMDH manuscript, draft 3 | `paper/manuscript.md`, `paper/STATUS.md`, Nextcloud `Documents/research/atr-research/atr-article-plan.md` | draft 3 complete except TODOs |
| Literature | search log, related-work report, reading notes, bib | `docs/SEARCH_LOG.md`, `docs/RELATED_WORK.md`, `docs/related_work/`, `refs/`, Zotero (user library) | done; OpenAlex Q4-Q10 incomplete |
| Epic: LLM playbook | instruction file so other LLMs can repair/improve a kraken model | `docs/epics/EPIC_LLM_PLAYBOOK.md`, issue thodel/weightspace#12 | proposed, not started |
| Epic: activation atlas | representation of the whole net, which units respond to which part of a line | `docs/epics/EPIC_ACTIVATION_ATLAS.md`, issue thodel/weightspace#13 | proposed, not started |

## Findings in one paragraph each

- Readings: traces of fine-tuning in the output layer; blank bias up relative to other biases and rare rows growing survive a seed null; the size of a change, the conv depth pattern and "rare rows move most" do not. Lineage can be read from weights for fine-tunes that did not move far (43 models). Front end generic, recurrent stack individual.
- Edits: no held-out gain on training-line practice (15 edits). On unseen practice lines one small gain for `combined-v1` (-0.19 to -0.39 points) that is a quarter-step interpolation toward a sibling; not matched by 20 shams; page-clustered intervals include 0; not distinguishable from plain interpolation.
- Positive control: localised planted defects are repaired by programmer, blind programmer and rule list alike; diffuse damage only by the programmer who measures and with a donor of the same lineage.
- Margins: 18 % of deleted frames have no blank frame; median margin 3.2 nats; blank-bias edits are not the lever.
- Interpolation of the Kurrent pair: single blocks and pairs additive, whole network not.
- Candidates: `kraken-latin_incunabula-wsp-v1` (v1: two group norms + output layer half-way) and v2 (norms only): both read held-out Fraktur better (v1 -2.04 NewsEye, -0.44 GT-Fraktur; v2 -1.08, -0.19) but v1 loses abbreviations (MUFI 532 to 731) and v2 slightly its own collection (+0.04); neither qualifies under the pre-registered rule (RESULTS R9); v1 registered and disabled, v2 not registered.
- Data findings: Hertwig export defect (44 of 50 pages, repaired); held-out set was half a printed book (replaced by `blb-eval-hand`); Reichenau incunable dataset: 39 of 59 validation pages identical to train pages.

## Open items

0. C4 retrain finished 8 Nov (branch `claude/c4-retrain`, merged 8 Nov): arm A -0.55, arm B -0.99 points on `bern-eval`, Hertwig lines only; RESULTS R11, manuscript 7.11. Next control if wanted: continue on the uncorrected pairs.

1. Not merged and not reviewed: `claude/layer-probe` (wsp.vizpage, probes; relevant to the atlas epic, issue #13). Also exists on the remote: `claude/layer-probe` (wsp.vizpage, probes; relevant to the atlas epic, issue #13), not reviewed here.
2. C4 retrain on corrected data needs the user's OK (GPU job on the trainer's card is not allowed for sessions).
3. 5,441 vs 5,243 line count of `combined-v1` (trainer to answer).
4. Donor-rule amendment in `docs/PROTOCOL.md` is proposed, not adopted.
5. Authors, availability, funding, AI statement, licences, public release (paper TODOs).
6. Leaked secrets to rotate by the user: Zotero API key; Nextcloud admin and database passwords (see handoff).
7. Branch hygiene: session branches `claude/zoo-survey-76f514`, `claude/seed-null-67d6e1-v2`, `claude/article-controls-8803eb` are merged; keep or delete on the remote at the user's decision.

## Rules that apply here

CPU unless stated; card 1 is the trainer's; research share read-only; nothing registered, uploaded or served by sessions; kraken pinned at 7.0.2; weights not committed; no children's first names or other personal data in public repositories; times in CET, labelled, logs are UTC.
