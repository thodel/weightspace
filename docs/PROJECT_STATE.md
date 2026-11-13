# Project state (13 November 2026, CET)

One page for anyone who opens the repository cold. Details are in the files it names.

- **Main:** at the merge of thodel/weightspace#28 (`6d03a3f`, 13 Nov), plus this documentation update. thodel/weightspace#27, #28 and #30 were reviewed and merged on 12–13 Nov.
- **Open pull requests:** none.
- **Pushed on 13 Nov, but not in a PR and not merged:**
  - `claude/trocr-stage0-9ababb`: the TrOCR stage-2 plan and the preregistration of the seed null (direct route, card 0), written before any run.
  - `claude/zoo-survey-76f514`: the zoo survey completed, 37 of 37 models, with manuscript 6.4, READINGS and MODELS updated.
  - `claude/adapters-4d972b`: corrections to PLAN, DESIGN, PROTOCOL and the adapter report.
  - `claude/c4-retrain`: entries for R11, PROJECT_STATE and README.
  - Each needs a review before it is merged. The last three touch files this page also changes.
- **Running:**
  - A watcher from the planted-defect session (`~/atr-cache/wsp/spool/a3-f1/p3-attempt2.sh`) is waiting for card 0. Once card 0 has 18 GB free, it starts a pass server and runs a second Fable attempt on adapter p3 (`q02`). That attempt is not part of #28.
  - Both cards are held by a `vllm serve Qwen3.8-27B` that is not part of this project.

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
| Phase 0-4 (A) | harness, readings, sessions on `xix-blb-v1` and `combined-v1`, collapse analysis; session 2 replicated with a Fable 5.1 programmer | `docs/PHASE0..4_REPORT.md`, `docs/KRAKEN_S2_FABLE.md`, `docs/RESULTS.md` R1-R3 | done; Fable replicate merged 13 Nov (#27), its model registered and **disabled** |
| Phase 5 clean-up | stricter collapse detector, the handwriting held-out set `blb-eval-hand` | `docs/PHASE5_CLEANUP_REPORT.md` | done |
| B | controls for the article: planted defects, unseen practice sets with Sonnet, Opus, Haiku and Fable programmers, sham, average, global interpolation, margins, seed null, zoo survey | `docs/CONTROLS_REPORT.md`, `docs/SEED_NULL.md`, `docs/ZOO_SURVEY.md`, `docs/RESULTS.md` R5-R8 | done, merged (Fable attempts f1 on 9 Nov, #22; planted defects rerun with Fable, §11, merged 13 Nov, #28) |
| Block statistics | errors per Unicode block (after CERberus), additive and paired; exact session replays | `docs/BLOCK_STATS.md`, `docs/CERBERUS_NOTES.md` | done, merged |
| C | the family of 43 served kraken models: weights, lineage, function, exchange grids; `latin_incunabula` candidates v1-v3; exchange on out-of-domain material and across lineages | `docs/MODEL_FAMILY.md`, `docs/family/`, `docs/RESULTS.md` R4, R9, R12, R13 | done, merged; v3 qualifies, registered and **disabled** |
| C4 retrain | `combined-v1` retrained on the corrected Hertwig data | `docs/C4_RETRAIN.md`, `docs/RESULTS.md` R11 | done, merged |
| Adapters | sessions on the LoRA adapter `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` of Qwen3.5-4B | `docs/ADAPTER_REPORT.md` (R10), `docs/PHASE5_PLAN.md`, `docs/PHASE5_REPORT.md` | first track, the phase-5 session (Opus 5.5) and its Fable 5.1 replicate (#25, `docs/PHASE5_FABLE_REPORT.md`, RESULTS R10b) merged |
| Epic #3: visualise layers | frame map, layer probes, decision views, α overlay, output-layer geometry, pages for both lineages | `docs/viz/README.md`, `protocol/docs/figures/` | done; checked on unseen practice lines (12 Nov): probe findings hold, block roles partly, part-block hypotheses withdrawn; issue closed |
| Protocol site | MkDocs Material, journal of every step, results R1-R13, amendments A1-A5 | `protocol/`, `protocol/PUBLISHING.md` | up to date; not published (private repo, Pages not set up) |
| Article | JDMDH manuscript, draft 3, with a second fact-check (10 Nov) and its fixes | `paper/manuscript.md`, `paper/factcheck_draft3.md`, `paper/STATUS.md` | draft 3; R13 and the phase-5 Opus session folded in on 11 Nov; not yet in the text: the adapter Fable replicate (#25), kraken session 2 with Fable (#27), the planted defects with Fable (#28), the epic #3 corrections (#30) |
| Literature | search log, related-work report, reading notes, bib | `docs/SEARCH_LOG.md`, `docs/RELATED_WORK.md`, `docs/related_work/`, `refs/` | done; OpenAlex Q4-Q10 incomplete |
| Epic #12 | LLM playbook | `docs/epics/` | proposed, not started |
| Epic #13: activation atlas | L0 viewer and L2 occlusion (`wsp.atlas`) | `docs/ATLAS_REPORT.md`, `docs/atlas/` | L0 and L2 done (13 Nov); Gate 1 fails for both models (hit share 0.61 and 0.73 < 0.80), so viewers claim activation, not location; conv receptive field 140 px confirmed; L1, L3, L4 not started |
| TrOCR track | the same experiments on the in-house TrOCR family (transformer encoder-decoder, 334 M, same tokenizer, six base-size models from one base) | `docs/epics/EPIC_TROCR_TRACK.md`, `docs/MODELS_TROCR.md`, `docs/trocr/STAGE0.md`, `docs/TROCR_READINGS.md`, `wsp/trocr_subject.py`, `wsp/trocr_family.py` | **stage 0 (Gate 0) and stage 1 (Gate 1) passed and merged 11 Nov**; decisions 1-4 answered; stage 2 (seed null) preregistered on its branch 13 Nov, no run yet |

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
  - Session 2 (training-side practice) rerun with a Fable 5.1 programmer: 3 edits kept. Held-out −0.57 [−1.15, −0.05] by pages, almost all of it on Hertwig (−1.32). This is the first session on a kraken model whose held-out page interval excludes 0. Of the kept edits, the step of 12.5 % toward the parent `hertwig-v2` carries the gain (−0.53 alone), and the two blank steps carry none. Through the serving path: −2.16 [−3.32, −1.05], but on the other hand (Schlussredaktion). Every edited model shows that hand reversal, so it belongs to the measurement. The size and place match the C4 retrain (R11). It is a merge toward the parent, not a repair the retrain does not also buy.
  - The C4 retrain on corrected data does at least as well as the best edit: −0.55 at learning rate 1e-5, −0.99 at 5e-5.
- **Positive control:** localised planted defects are repaired by the programmer, the blind programmer and the rule list alike, with Sonnet 5.5 and with Fable 5.1 programmers (13 of 22 cells for Fable, 15 for Sonnet). Diffuse damage (k9) was repaired only by the Sonnet programmer who measured (83 %); the Fable programmer did not try the interpolation toward the parent (24 %). So that claim holds for one of two models, one attempt each.
- **Exchange:**
  - `latin_incunabula` v3 qualifies under the rule with minimum effects. It moves the output layer half-way without the abbreviation rows: NewsEye −1.66 [−2.02, −1.30], MUFI share +0.014. Through the serving path it is not worse (median page −0.09).
  - First-convolution exchange stabilises the Kurrent model on held-out out-of-domain documents (−0.35 to −1.07), but below the preregistered minimum.
  - Unrelated models cannot exchange units.
- **Adapter:**
  - The phase-5 session kept one edit, direction 0 of `layers.31.self_attn.o_proj`. Practice CER fell 45.2 → 34.7 %; on 733 held-out pages the change is +0.24 [−3.5, +3.9].
  - A Fable 5.1 programmer, blind to that session, kept the same edit: a byte-identical adapter and byte-identical held-out readings (#25, merged).
  - One of three seeded random directions passes the same keep rule on practice, which motivates the proposed amendment A5 (sham noise floor).
- **TrOCR (stages 0-1):**
  - Seven models share one tokenizer, and input embedding and output projection are one tied tensor.
  - Readings are deterministic. C0 on `blb-eval-hand`: 19.74 % (`from-kurrent-v2`), 19.83 % (parent `XVI-XVII`), 26.41 % (`trocr-kurrent`).
  - Lineage is recoverable for the small fine-tunes, and the ViT pooler is an exact lineage marker.
  - The decoder moves 3-5 times more than the encoder.
  - The same data from two origins moves the weights almost orthogonally.
  - All of these are hypotheses until the seed null of stage 2.
- **Data findings:**
  - the Hertwig export defect (44 of 50 pages; repaired on HF, commit `dda578ea`);
  - a held-out set that was half a printed book (replaced by `blb-eval-hand`);
  - in the Reichenau incunable dataset, 39 of 59 validation pages are identical to train pages;
  - shared source documents behind different container ids in the out-of-domain sets (audited out before any read).

## Open items

**Decisions for the user**

1. **TrOCR track:** when the stage-2 seeds go through the trainer (card and timing); whether to fetch the third-party fine-tunes for the comparison; and the parents' corpora question of `docs/trocr/STAGE0.md` §8, item 4 (Laßberg or BLB material in the Kurrent or XVI-XVII training data?).

2. **Offer `latin_incunabula` v3 for use?** It is registered and disabled. It is the better candidate for editions that keep abbreviations, while v1 has the larger Fraktur gain.
   **And `kraken-bern-combined-wsp-v1`?** The Fable replicate of kraken session 2, registered and disabled. Held-out −0.57 (harness, lines) and −2.16 [−3.32, −1.05] through the serving path. The interpolation part carries the gain on both measures. The hand reversal appears for every edited model, so it belongs to the measurement (`docs/KRAKEN_S2_FABLE.md` §6).
3. **Rotate leaked secrets:** the Zotero API key, and the Nextcloud admin and database passwords.
4. **Amendments:** A3 (donor rule and practice side; the 12 Nov visualisation rerun is one more case for it) and A5 (sham noise floor for page-level sessions) are proposed, not adopted. A4 (page-clustered intervals) is adopted.
5. **Training jobs** need your OK, because they use the GPU on the trainer's card:
   - a C4 continuation on the uncorrected pairs, to show whether the correction or the extra training causes R11's gain;
   - a fine-tune with the first conv layer frozen (Track C).
6. **Article:**
   - authors, availability, funding, AI statement, licences, public release (6 `[TODO]`);
   - length, about 18,000 words against the roughly 7,000 an editor expects;
   - whether and where the text takes up the Fable reruns: 9.3 (adapter replicate and shams, #25), 7.x (kraken session 2, #27, the first held-out page interval excluding 0), 7.6 (planted defects, #28), and the epic #3 corrections on unseen lines (#30);
   - the open fact-check items listed in `paper/STATUS.md` (update of 11 Nov).
7. **Publishing the protocol:** see the checklist in `protocol/PUBLISHING.md` (repository visibility, licence, `site_url`).
8. **Branch hygiene and loose ends.**
   - Every remote branch except the four under "Pushed on 13 Nov" is merged into main and can be deleted.
   - The pack's `describe.md` prints the no-break space like a plain space; several programmers took it for a planted defect (`docs/CONTROLS_REPORT.md` §11.2). Fix before the next pack is built.
   - Whether the waiting `q02` watcher (see Running) should still run.
   - On asteraix, keep the worktrees `weightspace-blocks` and `weightspace-family`, which hold git-ignored data, until that data is archived.
   - Remove the other worktrees once their sessions are archived. That includes `weightspace-protocol` and the `.claude/worktrees/*` of finished sessions.

**Settled since 8 November**

- The `combined-v1` line count: 5,441 is right. The other figure, 5,243, adds two quantities that are not like-for-like (`paper/factcheck_draft3.md` §12).
- `phase5` and the controls branch are merged (#21, #22). v3 is through the serving path (#20). The Fable replicates are merged (#25, #27, #28). Epic #3 is closed (#30).

**Not started**

9. Epic #12 (LLM playbook).
10. TrOCR stages 2-5 (seed null, planted defects, sessions, retrain comparator). Stage 2 is preregistered on its branch; no run yet.
11. Literature: OpenAlex Q4-Q10.
12. A next adapter session designed under A5 (larger practice set, sham noise floor, teacher-forced stop margin).

## Rules that apply here

- **Compute:** CPU unless stated. Card 1 is the trainer's.
- **Research share:** read-only, with exceptions agreed with the user:
  - the evaluation set `tei/datasets/reichenau-incunabula-wsp/`;
  - the registrations of `kraken-latin_incunabula-wsp-v1` and `-v3`, both disabled;
  - the registration of `kraken-bern-combined-wsp-v1` (Fable replicate of session 2), disabled, 11.11.2026;
  - the evaluation set `tei/datasets/bern-disputation-wsp/` (the 14 `bern-eval` pages), 11.11.2026;
  - the control registrations `kraken-bern-combined-wsp-v1-blank`, `-v1-interp` and `kraken-bern-combined-wsp-opus-s2`, disabled, 12.11.2026.
- **Sessions:** nothing else is registered, uploaded or served by a session.
- **Software and weights:** kraken pinned at 7.0.2. Weights are not committed.
- **Privacy:** no children's first names or other personal data in public repositories.
- **Times:** CET and labelled; logs are UTC.
