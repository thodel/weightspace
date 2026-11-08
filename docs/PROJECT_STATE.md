# Project state (8 November 2026, late morning CET)

One page for anyone who opens the repository cold. Details are in the files named. Main is at the merge of
thodel/weightspace#16; there are no open pull requests.

## What this is

Weight-space programming (Takehi, Teoh, Zhang, Khattab, blog of 2 Nov 2026) tried on the in-house kraken text-recognition models of the Bern projects: read the weights, diagnose, edit, measure against held-out data and against ordinary merging. Planning on h1 (no GPU, no kraken), runs on asteraix. Article for JDMDH in `paper/`.

## Phases and tracks

| track | what | where | status |
|---|---|---|---|
| Phase 0-5 (A) | harness, readings, sessions on `xix-blb-v1` and `combined-v1`, collapse analysis, clean-up of held-out sets | `docs/PHASE0..5_REPORT.md`, `docs/RESULTS.md` R1-R3 | done |
| B | controls for the article: planted defects, unseen practice sets, sham, average, global interpolation, margins, seed null, zoo survey | `docs/CONTROLS_REPORT.md`, `docs/SEED_NULL.md`, `docs/ZOO_SURVEY.md`, `docs/RESULTS.md` R5-R8 | done, merged |
| Block statistics | errors per Unicode block (after CERberus), additive and paired; exact session replays | `docs/BLOCK_STATS.md`, `docs/CERBERUS_NOTES.md`, `wsp/blocks.py`, `wsp/replay.py` | done, merged |
| C | the family of 43 served kraken models: weights, lineage, function (CKA), exchange grids, candidates for `latin_incunabula` | `docs/MODEL_FAMILY.md`, `docs/family/`, `docs/RESULTS.md` R4, R9 | done, merged; a v3 candidate on an unmerged branch (below) |
| C4 retrain | `combined-v1` retrained on the corrected Hertwig data | `docs/C4_RETRAIN.md`, `docs/RESULTS.md` R11 | done, merged 8 Nov |
| Adapters | sessions on a LoRA adapter of a vision-language model | `docs/ADAPTER_REPORT.md`, `docs/RESULTS.md` R10 | first track merged 8 Nov; **second session (v2 page adapter) running**, its commits only on the local `phase5` branch on asteraix (below) |
| Epic #3: visualise layers | frame map, layer probes, decision views, α overlay, output-layer geometry, confusion-to-frame index, pages for both lineages | `docs/viz/README.md`, `docs/viz/*.html`, `docs/READINGS.md` (07.11.2026) | **all eight sub-tasks done and merged** (#11, #16); issue still open |
| Article | JDMDH manuscript, draft 3 | `paper/manuscript.md`, `paper/STATUS.md`, Nextcloud `Documents/research/atr-research/atr-article-plan.md` | draft 3 complete except TODOs |
| Literature | search log, related-work report, reading notes, bib | `docs/SEARCH_LOG.md`, `docs/RELATED_WORK.md`, `docs/related_work/`, `refs/`, Zotero (user library) | done; OpenAlex Q4-Q10 incomplete |
| Epic #12: LLM playbook | instruction file so other LLMs can repair/improve a kraken model | `docs/epics/EPIC_LLM_PLAYBOOK.md` | proposed, not started |
| Epic #13: activation atlas | which units respond to which part of a line | `docs/epics/EPIC_ACTIVATION_ATLAS.md` | proposed, not started; can reuse the frame map and forced alignment of `wsp.layerview` |

## Findings in one paragraph each

- Readings: traces of fine-tuning in the output layer; blank bias up relative to other biases and rare rows growing survive a seed null; the size of a change, the conv depth pattern and "rare rows move most" do not. Lineage can be read from weights for fine-tunes that did not move far (43 models). Front end generic, recurrent stack individual.
- Edits: no held-out gain on training-line practice (15 edits). On unseen practice lines one small gain for `combined-v1` (-0.19 to -0.39 points) that is a quarter-step interpolation toward a sibling; not matched by 20 shams; page-clustered intervals include 0; not distinguishable from plain interpolation.
- Positive control: localised planted defects are repaired by programmer, blind programmer and rule list alike; diffuse damage only by the programmer who measures and with a donor of the same lineage.
- Margins: 18 % of deleted frames have no blank frame; median margin 3.2 nats; blank-bias edits are not the lever.
- Interpolation of the Kurrent pair: single blocks and pairs additive, whole network not. Same for exchange: every front-end block of `latin_incunabula` could be set back alone, the whole front end together could not.
- Candidates (R4, R9): `kraken-latin_incunabula-wsp-v1` (two group norms + output layer half-way) and v2 (norms only) both read held-out Fraktur better (v1 -2.04 NewsEye, -0.44 GT-Fraktur; v2 -1.08, -0.19); v1 loses abbreviations (MUFI 532 to 731), v2 slightly its own collection (+0.04); neither qualifies under the pre-registered rule; v1 registered on the share and disabled, v2 not registered.
- Layers (epic #3): letters are identified in the BiLSTMs (probe identity 0.36 at the last conv, 0.94 at `L_16` for `xix-blb-v1`); in both lineages the fine-tune grows the rare output rows and lowers their bias, and raises the blank's bias by about +0.11; about 72 % of deleted characters are decodable at no layer. The Kurrent fine-tune sits in all three LSTMs (4-6 points each when set back), the Bern fine-tune `hertwig-v2 → combined-v1` only in `L_16` (+1.04).
- C4 (R11): retraining `combined-v1` on the corrected data lowers `bern-eval` CER on the Hertwig lines (arm A -0.55, arm B -0.99 points).
- Data findings: Hertwig export defect (44 of 50 pages, repaired on HF, commit `dda578ea`); held-out set was half a printed book (replaced by `blb-eval-hand`); Reichenau incunable dataset: 39 of 59 validation pages identical to train pages, and dh-unibe's copy holds only the German and Latin-German sub-collections.

## Open items

**In progress or waiting for review**

1. **Adapter session (v2 page adapter).** Frozen with one kept edit (practice CER 45.2 % to 34.7 %); its held-out scoring and controls run in a queue on card 0. Seven commits (`d42f662` … `66fd4a9`) are on the local `phase5` branch on asteraix only: `phase5` was merged into main through #9 before them, so they need a new branch and PR once the held-out numbers are in.
2. **`claude/incunabula-v3-995321`** (pushed 8 Nov, no PR): a pre-registered third `latin_incunabula` candidate that moves the output layer without the abbreviation rows, and RESULTS R12 saying it qualifies. Not reviewed or merged; if it holds, it is the candidate to read through the serving path and enable.
3. **`claude/article-controls-8803eb`** (7 Nov, not merged): three small notes (results numbered R4-R6 there, a cluster-interval note in PROTOCOL, the Fable run recorded as not done). Probably superseded by main's R5-R8 and PROTOCOL; check, then close or merge the remainder.

**Decisions for the user**

4. **Rotate leaked secrets:** Zotero API key; Nextcloud admin and database passwords (see handoff).
5. **Close epic #3** (all sub-tasks done).
6. **Session design** before any below-output-layer session: page-clustered intervals (`docs/CONTROLS_REPORT.md` §9) and the practice-side amendment (PROTOCOL §2 item 8) are proposed, not adopted; the donor-rule amendment in `docs/PROTOCOL.md` likewise.
7. **Training jobs** need the user's OK (GPU on the trainer's card): a C4 continuation on the uncorrected pairs; a fine-tune with the first conv layer frozen (Track C).
8. 5,441 vs 5,243 line count of `combined-v1` (trainer to answer).
9. Authors, availability, funding, AI statement, licences, public release (paper TODOs).
10. **Branch hygiene.** Merged into main and safe to delete on the remote: `phase0`-`phase5`, `block-stats`, `model-family`, `candidate-v2`, `schluss-check`, `viz-epic3`, `claude/adapters-4d972b`, `claude/c4-retrain`, `claude/layer-probe`, `claude/seed-null-67d6e1-v2`, `claude/zoo-survey-76f514`, `protocol-current` (check its role first). Not `phase5` locally until item 1 is moved. On asteraix the worktrees `weightspace-blocks` and `weightspace-family` hold git-ignored data (exchange readings, replays); keep them until that data is archived.

**Not started**

11. Epic #12 (LLM playbook) and epic #13 (activation atlas).
12. Literature: OpenAlex Q4-Q10.
13. Track C follow-ups: exchange across lineages after matching units by activation; exchange that protects the abbreviation rows (now the v3 branch, item 2).

## Rules that apply here

CPU unless stated; card 1 is the trainer's; research share read-only, except the evaluation set `tei/datasets/reichenau-incunabula-wsp/` and the registration of `kraken-latin_incunabula-wsp-v1` (disabled), both with the user's OK on 7 Nov; nothing else registered, uploaded or served by sessions; kraken pinned at 7.0.2; weights not committed; no children's first names or other personal data in public repositories; times in CET, labelled, logs are UTC.
