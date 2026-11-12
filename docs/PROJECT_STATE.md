# Project state (11 November 2026, midday CET)

One page for anyone who opens the repository cold. Details are in the files it names.

- **Main:** includes the Fable replicate (#25), the manuscript update of 11 November (`9b4fa1f`) and the TrOCR preparation (`0106b90`).
- **Open pull requests:** none.

## What this is

Weight-space programming (Takehi, Teoh, Zhang, Khattab, blog of 2 Nov 2026) tried on the in-house kraken text-recognition models of the Bern projects, and on one LoRA adapter of a vision-language model. The method:

1. read the weights;
2. diagnose;
3. edit;
4. measure against held-out data and against ordinary merging and retraining.

Planning happens on h1, which has no GPU and no kraken; runs happen on asteraix. The article for JDMDH is in `paper/`. The protocol site (MkDocs, not yet published) is in `protocol/`.

## Phases and tracks

| track | what | where | status |
|---|---|---|---|
| Phase 0-4 (A) | harness, readings, sessions on `xix-blb-v1` and `combined-v1`, collapse analysis | `docs/PHASE0..4_REPORT.md`, `docs/RESULTS.md` R1-R3 | done |
| Phase 5 clean-up | stricter collapse detector, the handwriting held-out set `blb-eval-hand` | `docs/PHASE5_CLEANUP_REPORT.md` | done |
| B | controls for the article: planted defects, unseen practice sets with Sonnet, Opus, Haiku and Fable programmers, sham, average, global interpolation, margins, seed null, zoo survey | `docs/CONTROLS_REPORT.md`, `docs/SEED_NULL.md`, `docs/ZOO_SURVEY.md`, `docs/RESULTS.md` R5-R8 | done, merged (Fable attempts f1 on 9 Nov, #22) |
| Block statistics | errors per Unicode block (after CERberus), additive and paired; exact session replays | `docs/BLOCK_STATS.md`, `docs/CERBERUS_NOTES.md` | done, merged |
| C | the family of 43 served kraken models: weights, lineage, function, exchange grids; `latin_incunabula` candidates v1-v3; exchange on out-of-domain material and across lineages | `docs/MODEL_FAMILY.md`, `docs/family/`, `docs/RESULTS.md` R4, R9, R12, R13 | done, merged; v3 qualifies, registered and **disabled** |
| C4 retrain | `combined-v1` retrained on the corrected Hertwig data | `docs/C4_RETRAIN.md`, `docs/RESULTS.md` R11 | done, merged |
| Adapters | sessions on the LoRA adapter `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` of Qwen3.5-4B | `docs/ADAPTER_REPORT.md` (R10), `docs/PHASE5_PLAN.md`, `docs/PHASE5_REPORT.md` | first track, the phase-5 session (Opus 5.5) and its Fable 5.1 replicate (#25, `docs/PHASE5_FABLE_REPORT.md`, RESULTS R10b) merged |
| Epic #3: visualise layers | frame map, layer probes, decision views, α overlay, output-layer geometry, pages for both lineages | `docs/viz/README.md`, `protocol/docs/figures/` | done; checked on unseen practice lines (12 Nov): probe findings hold, block roles partly, part-block hypotheses withdrawn; issue closed |
| Protocol site | MkDocs Material, journal of every step, results R1-R13, amendments A1-A5 | `protocol/`, `protocol/PUBLISHING.md` | up to date; not published (private repo, Pages not set up) |
| Article | JDMDH manuscript, draft 3, with a second fact-check (10 Nov) and its fixes | `paper/manuscript.md`, `paper/factcheck_draft3.md`, `paper/STATUS.md` | draft 3; R13 and the phase-5 Opus session folded in on 11 Nov; the Fable replicate and its shams not yet in 9.3 |
| Literature | search log, related-work report, reading notes, bib | `docs/SEARCH_LOG.md`, `docs/RELATED_WORK.md`, `docs/related_work/`, `refs/` | done; OpenAlex Q4-Q10 incomplete |
| Epics #12 and #13 | LLM playbook; activation atlas | `docs/epics/` | proposed, not started |
| TrOCR track | the same experiments on the in-house TrOCR family (transformer encoder-decoder, 334 M, same tokenizer, six base-size models from one base) | `docs/epics/EPIC_TROCR_TRACK.md`, `docs/MODELS_TROCR.md`, `docs/BRIEF_TROCR_STAGE0.md`, `scripts/trocr_fetch.py` | prepared 11 Nov, not started; stage 0 needs asteraix (h1 has 5.5 GB free) and decisions 1-4 of the epic |

## Findings in one paragraph each

- **Readings:**
  - Some traces of fine-tuning survive a seed null: the blank's bias rising relative to the other biases, and the rare output rows growing.
  - Lineage can be read from the weights for fine-tunes that did not move far (43 models).
  - The front end is generic and the recurrent stack is individual.
  - Letters are identified in the BiLSTMs, on training and on unseen lines alike. About two thirds of deleted characters (66–72 %) are decodable at no layer. Which block's fine-tune share does what holds only partly on unseen lines, and the Bern block shares were confounded by the parent's training lines (docs/viz/README.md, 12 Nov).
- **Edits on kraken models:**
  - No held-out gain on practice lines from the training side (15 edits).
  - On unseen practice lines there is one small gain for `combined-v1` (−0.19 to −0.39 points). It is a quarter-step interpolation toward a sibling, and its page-clustered intervals include 0.
  - A Fable 5.1 programmer found a larger practice gain and no held-out gain (−0.02).
  - The C4 retrain on corrected data does at least as well as the best edit: −0.55 at learning rate 1e-5, −0.99 at 5e-5.
- **Positive control:** localised planted defects are repaired by the programmer, the blind programmer and the rule list alike. Diffuse damage is repaired only by the programmer who measures, and only with a donor of the same lineage.
- **Exchange:**
  - `latin_incunabula` v3 qualifies under the rule with minimum effects. It moves the output layer half-way without the abbreviation rows: NewsEye −1.66 [−2.02, −1.30], MUFI share +0.014. Through the serving path it is not worse (median page −0.09).
  - First-convolution exchange stabilises the Kurrent model on held-out out-of-domain documents (−0.35 to −1.07), but below the preregistered minimum.
  - Unrelated models cannot exchange units.
- **Adapter:**
  - The phase-5 session kept one edit, direction 0 of `layers.31.self_attn.o_proj`. Practice CER fell 45.2 → 34.7 %; on 733 held-out pages the change is +0.24 [−3.5, +3.9].
  - A Fable 5.1 programmer, blind to that session, kept the same edit: a byte-identical adapter and byte-identical held-out readings (#25, merged).
  - One of three seeded random directions passes the same keep rule on practice, which motivates the proposed amendment A5 (sham noise floor).
- **Data findings:**
  - the Hertwig export defect (44 of 50 pages; repaired on HF, commit `dda578ea`);
  - a held-out set that was half a printed book (replaced by `blb-eval-hand`);
  - in the Reichenau incunable dataset, 39 of 59 validation pages are identical to train pages;
  - shared source documents behind different container ids in the out-of-domain sets (audited out before any read).

## Open items

**Decisions for the user**

1. **TrOCR track:** decisions 1-4 of `docs/epics/EPIC_TROCR_TRACK.md` before stage 0 on asteraix.

2. **Offer `latin_incunabula` v3 for use?** It is registered and disabled. It is the better candidate for editions that keep abbreviations, while v1 has the larger Fraktur gain.
3. **Rotate leaked secrets:** the Zotero API key, and the Nextcloud admin and database passwords.
4. **Amendments:** A3 (donor rule and practice side; the 12 Nov visualisation rerun is one more case for it) and A5 (sham noise floor for page-level sessions) are proposed, not adopted. A4 (page-clustered intervals) is adopted.
5. **Training jobs** need your OK, because they use the GPU on the trainer's card:
   - a C4 continuation on the uncorrected pairs, to show whether the correction or the extra training causes R11's gain;
   - a fine-tune with the first conv layer frozen (Track C).
6. **Article:**
   - authors, availability, funding, AI statement, licences, public release (6 `[TODO]`);
   - length, about 18,000 words against the roughly 7,000 an editor expects;
   - whether Section 9.3 adds the Fable replicate and the sham result;
   - the open fact-check items listed in `paper/STATUS.md` (update of 11 Nov).
7. **Publishing the protocol:** see the checklist in `protocol/PUBLISHING.md` (repository visibility, licence, `site_url`).
8. **Branch hygiene.**
   - Every remote branch is merged into main and can be deleted.
   - On asteraix, keep the worktrees `weightspace-blocks` and `weightspace-family`, which hold git-ignored data, until that data is archived.
   - Remove the other worktrees once their sessions are archived. That includes `weightspace-protocol` and the `.claude/worktrees/*` of finished sessions.

**Settled since 8 November**

- The `combined-v1` line count: 5,441 is right. The other figure, 5,243, adds two quantities that are not like-for-like (`paper/factcheck_draft3.md` §12).
- `phase5` and the controls branch are merged (#21, #22). v3 is through the serving path (#20). The Fable replicate is merged (#25).

**Not started**

10. Epic #12 (LLM playbook) and epic #13 (activation atlas).
11. Literature: OpenAlex Q4-Q10.
12. A next adapter session designed under A5 (larger practice set, sham noise floor, teacher-forced stop margin).

## Rules that apply here

- **Compute:** CPU unless stated. Card 1 is the trainer's.
- **Research share:** read-only, with exceptions agreed with the user:
  - the evaluation set `tei/datasets/reichenau-incunabula-wsp/`;
  - the registrations of `kraken-latin_incunabula-wsp-v1` and `-v3`, both disabled.
- **Sessions:** nothing else is registered, uploaded or served by a session.
- **Software and weights:** kraken pinned at 7.0.2. Weights are not committed.
- **Privacy:** no children's first names or other personal data in public repositories.
- **Times:** CET and labelled; logs are UTC.
