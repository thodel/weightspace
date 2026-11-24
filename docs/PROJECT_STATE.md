# Project state (24 November 2026, CET)

One page for anyone who opens the repository cold. Details are in the files it names.

- **Main:** at `ab52c1b` (20 Nov, #55), plus this documentation update.
- **Open pull requests:** none.
- **Merged since 14 Nov:**
  - kraken session 2 rerun with a Fable 5.2 programmer (#44);
  - cross-vendor replicates, E1 and E2, with Gemini 4 Argon and GPT-6.2 Astra as programmers (plan #43; runs and report #53, #54, #55; `docs/XVENDOR_REPORT.md`);
  - TrOCR stage 3, the planted-defect control (R17), and the stage-4 preregistration and harness (`0c5540e`);
  - the atlas synthesis and manuscript Section 6.6 (#42), the atlas explorer (#45, #48);
  - the public Quarto site (#46, #52), licences (#47, #49), dataset provenance (#50) and public model identifiers (#51).
- **Running:** nothing of this project.

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
| Phase 0-4 (A) | harness, readings, sessions on `xix-blb-v1` and `combined-v1`, collapse analysis; session 2 replicated with a Fable 5.1 programmer | `docs/PHASE0..4_REPORT.md`, `docs/KRAKEN_S2_FABLE.md`, `docs/RESULTS.md` R1-R3 | done; session 2 rerun by four more programmers: Fable 5.1 (#27, its model registered and **disabled**), Fable 5.2 (#44), Gemini 4 Argon and GPT-6.2 Astra (#53, `docs/XVENDOR_REPORT.md`) |
| Phase 5 clean-up | stricter collapse detector, the handwriting held-out set `blb-eval-hand` | `docs/PHASE5_CLEANUP_REPORT.md` | done |
| B | controls for the article: planted defects, unseen practice sets with Sonnet, Opus, Haiku and Fable programmers, sham, average, global interpolation, margins, seed null, zoo survey | `docs/CONTROLS_REPORT.md`, `docs/SEED_NULL.md`, `docs/ZOO_SURVEY.md`, `docs/RESULTS.md` R5-R8 | done, merged (Fable attempts f1 on 9 Nov, #22; planted defects rerun with Fable, §11, merged 13 Nov, #28) |
| Block statistics | errors per Unicode block (after CERberus), additive and paired; exact session replays | `docs/BLOCK_STATS.md`, `docs/CERBERUS_NOTES.md` | done, merged |
| C | the family of 43 served kraken models: weights, lineage, function, exchange grids; `latin_incunabula` candidates v1-v3; exchange on out-of-domain material and across lineages | `docs/MODEL_FAMILY.md`, `docs/family/`, `docs/RESULTS.md` R4, R9, R12, R13 | done, merged; v3 qualifies, registered and **disabled** |
| C4 retrain | `combined-v1` retrained on the corrected Hertwig data | `docs/C4_RETRAIN.md`, `docs/C4_UNCORRECTED.md`, `docs/RESULTS.md` R11, R14 | done, merged; control on the uncorrected pairs run 14 Nov (no gain: the correction is what helps) |
| Adapters | sessions on the LoRA adapter `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` of Qwen3.5-4B | `docs/ADAPTER_REPORT.md` (R10), `docs/PHASE5_PLAN.md`, `docs/PHASE5_REPORT.md` | first track, the phase-5 session (Opus 5.5) and its replicates with Fable 5.1 (#25, `docs/PHASE5_FABLE_REPORT.md`, RESULTS R10b), Gemini 4 Argon and GPT-6.2 Astra (#55, `docs/XVENDOR_REPORT.md` §6) merged |
| Epic #3: visualise layers | frame map, layer probes, decision views, α overlay, output-layer geometry, pages for both lineages | `docs/viz/README.md`, `protocol/docs/figures/` | done; checked on unseen practice lines (12 Nov): probe findings hold, block roles partly, part-block hypotheses withdrawn; issue closed |
| Protocol site | MkDocs Material, journal of every step, results R1-R13, amendments A1-A5; public Quarto site in the dhbern layout, which includes the protocol pages | `protocol/`, `protocol/PUBLISHING.md`, `site/` | not published: the site deploys from `main` once the repository is public (`.github/workflows/site.yml`) |
| Article | JDMDH manuscript, draft 3, with a second fact-check (10 Nov) and its fixes | `paper/manuscript.md`, `paper/factcheck_draft3.md`, `paper/STATUS.md` | draft 3; R13 and the phase-5 Opus session folded in on 11 Nov; medium and low fact-check findings fixed 14 Nov; not yet in the text: the adapter Fable replicate (#25), kraken session 2 with Fable (#27), the planted defects with Fable (#28), the epic #3 corrections (#30), the C4 control on uncorrected pairs (R14) |
| Literature | search log, related-work report, reading notes, bib | `docs/SEARCH_LOG.md`, `docs/RELATED_WORK.md`, `docs/related_work/`, `refs/` | done; OpenAlex Q4-Q10 incomplete |
| Epic #12 | LLM playbook | `docs/epics/` | proposed, not started |
| Epic #13: activation atlas | L0 viewer and L2 occlusion (`wsp.atlas`); L3 mean ablation (`wsp.atlas_ablate`); L1 selectivity (`wsp.atlas_select`) | `docs/ATLAS_REPORT.md`, `docs/atlas/` | L0 and L2 done and merged (13 Nov, #32); Gate 1 fails for both models (hit share 0.61 and 0.73 < 0.80), so viewers claim activation, not location; conv receptive field 140 px confirmed. L3 done (14 Nov, rule fixed first): no single unit moves CER ≥ 1 point except `hertwig-v2` `C_0 ch 3`; top-10 `C_0` groups silence both models; character circuits hold for 7 of 11 characters in `hertwig-v2` and only `t` in `xix-blb-v1`, almost all in conv blocks; no space circuit in either model; dead-filter prediction holds; fine-tune weight change does not predict importance (ρ ≈ 0). L1 done (14 Nov, rule fixed first): sparse probes code characters from `C_6` (xix) / `C_3` (hertwig) on, BiLSTM forward units carry the previous and reverse the next character; no single unit beats the random network's bar (claim S fails), so **Gate 2 fails for both models** and the atlas stays descriptive (L3 circuits shown as descriptions); seed runs reproduce the units almost exactly but mostly by inheritance from the shared parent. L4 not started (needs Gate 2). Closing synthesis (15 Nov) answers the epic's five questions in `docs/ATLAS_REPORT.md`; Section 6.6 of the manuscript |
| TrOCR track | the same experiments on the in-house TrOCR family (transformer encoder-decoder, 334 M, same tokenizer, six base-size models from one base) | `docs/epics/EPIC_TROCR_TRACK.md`, `docs/MODELS_TROCR.md`, `docs/trocr/STAGE0.md`, `docs/TROCR_READINGS.md`, `wsp/trocr_subject.py`, `wsp/trocr_family.py` | **Gates 0, 1 and 2 passed and merged** (stages 0-1 on 11 Nov, stage 2 on 15 Nov, R15, R16); decisions 1-4 answered; stage 3 (planted defects) next, not started |

## Findings in one paragraph each

- **Readings:**
  - Some traces of fine-tuning survive a seed null: the blank's bias rising relative to the other biases, and the rare output rows growing.
  - Lineage can be read from the weights for fine-tunes that did not move far (43 models).
  - The front end is generic and the recurrent stack is individual.
  - Letters are identified in the BiLSTMs, on training and on unseen lines alike. About two thirds of deleted characters (66–72 %) are decodable at no layer. Which block's fine-tune share does what holds only partly on unseen lines, and the Bern block shares were confounded by the parent's training lines (docs/viz/README.md, 12 Nov).
  - Activation atlas (epic #13, `docs/ATLAS_REPORT.md`, every level under a rule fixed before its first run): the viewers may show activation, not location (Gate 1 fails). No single unit matters much or is selective on its own; ten first-layer channels together carry the whole reading; probes read characters from the later blocks, and the BiLSTM directions carry the previous and next character. Gate 2 fails for both models, so the atlas stays descriptive and the character circuits of L3 count as descriptions. Neither model has a circuit for the deleted spaces.
- **Edits on kraken models:**
  - No held-out gain on practice lines from the training side (15 edits).
  - On unseen practice lines there is one small gain for `combined-v1` (−0.19 to −0.39 points). It is a quarter-step interpolation toward a sibling, and its page-clustered intervals include 0.
  - A Fable 5.1 programmer found a larger practice gain and no held-out gain (−0.02).
  - Session 2 (training-side practice) has been run by five programmers: Opus 5.5 (the original), Fable 5.1, Fable 5.2, Gemini 4 Argon and GPT-6.2 Astra, one run each (`docs/KRAKEN_S2_FABLE.md`, `docs/KRAKEN_S2_FABLE52.md`, `docs/XVENDOR_REPORT.md`).
    - The four later programmers all moved the whole network toward the parent `hertwig-v2` and gained about the same on held-out: −0.57 to −0.65 points. The gain lies on the Hertwig pages (−1.3 to −2.4); Opus, which transplanted the parent's output layer, gained nothing (+0.15).
    - **Only one of the five page-clustered intervals lies below 0**: Fable 5.1's, −0.57 [−1.15, −0.05], by 0.05. Gemini's reaches 0 ([−1.38, +0.00]); Fable 5.2's [−1.64, +0.24] and GPT's [−1.46, +0.24] include it. The larger steps toward the parent (Fable 5.2, GPT) cost the Schlussredaktion pages measurably (+0.69 and +0.67, intervals above 0).
    - So the finding is a direction, not a single result: every programmer finds the step toward the parent, worth about 0.6 points on held-out, but a single session does not reliably clear 0. Pairwise, the programmers do not differ (every difference under 0.1 point). With one run per model, run-to-run variation is unmeasured.
    - In every run the step toward the parent carries the gain and the blank-bias steps carry none. Its size and place match the C4 retrain (R11): a merge toward the parent, not a repair the retrain does not also buy.
    - Through the serving path (Fable 5.1 only): −2.16 [−3.32, −1.05], but on the other hand (Schlussredaktion). Every edited model shows that hand reversal, so it belongs to the measurement.
  - The C4 retrain on corrected data does at least as well as the best edit: −0.55 at learning rate 1e-5, −0.99 at 5e-5. A control with the uncorrected pairs gives no gain (−0.07, −0.03; R14), so the gain depends on the correction.
- **Positive control:** localised planted defects are repaired by the programmer, the blind programmer and the rule list alike, with Sonnet 5.5 and with Fable 5.1 programmers (13 of 22 cells for Fable, 15 for Sonnet). Diffuse damage (k9) was repaired only by the Sonnet programmer who measured (83 %); the Fable programmer did not try the interpolation toward the parent (24 %). So that claim holds for one of two models, one attempt each.
- **Exchange:**
  - `latin_incunabula` v3 qualifies under the rule with minimum effects. It moves the output layer half-way without the abbreviation rows: NewsEye −1.66 [−2.02, −1.30], MUFI share +0.014. Through the serving path it is not worse (median page −0.09).
  - First-convolution exchange stabilises the Kurrent model on held-out out-of-domain documents (−0.35 to −1.07), but below the preregistered minimum.
  - Unrelated models cannot exchange units.
- **Adapter:**
  - The phase-5 session kept one edit, direction 0 of `layers.31.self_attn.o_proj`. Practice CER fell 45.2 → 34.7 %; on 733 held-out pages the change is +0.24 [−3.5, +3.9].
  - A Fable 5.1 programmer, blind to that session, kept the same edit: a byte-identical adapter and byte-identical held-out readings (#25, merged).
  - Gemini 4 Argon and GPT-6.2 Astra both kept the same top reading direction, removed from layer 31's writers, and froze on byte-identical adapters (#55). Practice −7.8 points; held-out +1.13 [−3.0, +5.2], a null.
  - Random directions of the same size pass the keep rule on practice: one of three for the Fable shams, two of three for the E1 shams. This motivates the proposed amendment A5 (sham noise floor).
- **TrOCR (stages 0-3):**
  - Seven models share one tokenizer, and input embedding and output projection are one tied tensor.
  - Readings are deterministic. C0 on `blb-eval-hand`: 19.74 % (`from-kurrent-v2`), 19.83 % (parent `XVI-XVII`), 26.41 % (`trocr-kurrent`).
  - Lineage is recoverable for the small fine-tunes, and the ViT pooler is an exact lineage marker.
  - The decoder moves 3-5 times more than the encoder.
  - The same data from two origins moves the weights almost orthogonally.
  - Stage 2 (seed null, six reruns of the recorded recipe on card 0, R16): held-out CER 20.04 % mean over five seeds (sd 0.14; recorded 19.74 %), so edits under about 0.5 points are retraining noise. Magnitude traces (encoder move 0.039, hump, tied-embedding share 0.85, encoder/decoder 0.14) reproduce across seeds; move directions do not (same-recipe cosine 0.36 encoder, 0.19 decoder); a sibling run is farther (0.108) than the parent (0.088).
  - Stage 3 (planted-defect control, R17, 18-19 Nov): 14 planted models of `from-kurrent-v2`, 12 scored defects all count on held-out (1.54 to 13.42 points). Core eight: Sonnet 5.5 with donor repairs 7, without donor 1 (p06; p01 crashed), blind (no passes) 7, Fable 5.2 7 and one partly (p10 26 %); rule list 4, scale search 0. No harmful keep. Gate 3 met. With the donor most repairs are a lookup of the parent tensor; p10 (noise on every matrix) is repaired by no arm.
- **Data findings:**
  - the Hertwig export defect (44 of 50 pages; repaired on HF, commit `dda578ea`);
  - a held-out set that was half a printed book (replaced by `blb-eval-hand`);
  - in the Reichenau incunable dataset, 39 of 59 validation pages are identical to train pages;
  - shared source documents behind different container ids in the out-of-domain sets (audited out before any read).

## Open items

**Decisions for the user**

1. **TrOCR track:** stage 3 is done and merged (R17, Gate 3 met). The stage-4 preregistration and harness are merged (`docs/TROCR_STAGE4_PREREG.md`, 19 Nov), but the seven decisions of `docs/TROCR_STAGE4_PLAN.md` §10 are not yet answered one by one; the preregistration marks the choices that rest on them as proposed defaults. Also open: the parents' corpora question of `docs/trocr/STAGE0.md` §8, item 4 (Laßberg or BLB material in the Kurrent or XVI-XVII training data?).

2. **Offer `latin_incunabula` v3 for use?** It is registered and disabled. It is the better candidate for editions that keep abbreviations, while v1 has the larger Fraktur gain.
   **And `kraken-bern-combined-wsp-v1`?** The Fable replicate of kraken session 2, registered and disabled. Held-out −0.57 (harness, lines) and −2.16 [−3.32, −1.05] through the serving path. The interpolation part carries the gain on both measures. The hand reversal appears for every edited model, so it belongs to the measurement (`docs/KRAKEN_S2_FABLE.md` §6). Since then three more programmers found the same direction, but only this run's page interval excludes 0 (see the session-2 finding above); the case for offering it rests on the direction, which a short retrain on corrected data (R11) gives as well.
3. **Rotate leaked secrets:** the Zotero API key, and the Nextcloud admin and database passwords.
4. **Amendments:** A3 (donor rule and practice side; the 12 Nov visualisation rerun is one more case for it) and A5 (sham noise floor for page-level sessions) are proposed, not adopted. A4 (page-clustered intervals) is adopted.
5. **Training jobs** need your OK, because they use the GPU on the trainer's card:
   - ~~a C4 continuation on the uncorrected pairs~~ done 14 Nov (`docs/C4_UNCORRECTED.md`, R14): −0.07 and −0.03 points against −0.55 and −0.99; the correction, not the extra training, gives R11's gain;
   - a fine-tune with the first conv layer frozen (Track C).
6. **Article:**
   - authors, availability, funding, AI statement, licences, public release (6 `[TODO]`);
   - length, about 18,000 words against the roughly 7,000 an editor expects;
   - whether and where the text takes up the reruns: 9.3 (adapter replicates and shams: Fable #25, Gemini and GPT #55), 7.x (kraken session 2 by five programmers: #27, #44, #53; one page interval of five excludes 0, so the text should report a shared direction, not a single gain), 7.6 (planted defects, #28), and the epic #3 corrections on unseen lines (#30);
   - the open fact-check items listed in `paper/STATUS.md` (update of 11 Nov).
7. **Publishing the site:** see the checklist in `protocol/PUBLISHING.md`. Licences (#47), provenance (#50) and `site-url` are done; open are repository visibility, redaction of the repository if it goes public, telling the dataset owner of the defect, authorship and acknowledgements, and switching on Pages.
8. **Branch hygiene and loose ends.**
   - Fourteen remote branches are merged into `main` and can be deleted (24 Nov): `claude/atlas-explorer`, `claude/atlas-l1`, `claude/atlas-l3`, `claude/atlas-synthesis`, `claude/factcheck-addendum2`, `claude/manuscript-medium-fixes`, `claude/manuscript-table1-caption`, `claude/planted-fable-p3-attempt2`, `claude/site-explorer`, `claude/site-title`, `claude/xvendor-e2`, `claude/xvendor-plan`, `docs/status-14nov`, `site/quarto`. `claude/trocr-stage0-9ababb` (TrOCR stages 0-4) has commits not on `main`; keep it while its session works.
   - The pack's `describe.md` prints the no-break space like a plain space; several programmers took it for a planted defect (`docs/CONTROLS_REPORT.md` §11.2). Fix before the next pack is built.
   - On asteraix, keep the worktrees `weightspace-blocks` and `weightspace-family`, which hold git-ignored data, until that data is archived.
   - Remove the other worktrees once their sessions are archived. That includes `weightspace-protocol` and the `.claude/worktrees/*` of finished sessions.

**Settled since 8 November**

- The `combined-v1` line count: 5,441 is right. The other figure, 5,243, adds two quantities that are not like-for-like (`paper/factcheck_draft3.md` §12).
- `phase5` and the controls branch are merged (#21, #22). v3 is through the serving path (#20). The Fable replicates are merged (#25, #27, #28; the second p3 attempt, #38). Epic #3 is closed (#30). Atlas levels L0–L3 are done (#32, #39, #40); the report has a closing synthesis and the manuscript a Section 6.6 (15 Nov). The C4 control on the uncorrected pairs is run (R14).

**Not started**

9. Epic #12 (LLM playbook); level L4 of epic #13, which needs a model that passes Gate 2 (none does).
10. TrOCR stages 4-5 (sessions on the Bern family, retrain comparator). Stages 0-3 are done and merged (R15, R16, R17); the plan of stage 4 is a draft awaiting Tobias (`docs/TROCR_STAGE4_PLAN.md`).
11. Literature: OpenAlex Q4-Q10.
12. A next adapter session designed under A5 (larger practice set, sham noise floor, teacher-forced stop margin).
13. Closed 24.11: kraken session 2 run-to-run variation and step size. Repeats (docs/S2_REPEATS.md): four Fable 5.2 runs all found the parent step, held-out −0.47 to −0.67 (range 0.20 > 0.085 across the four parent-step programmers), so model differences are not interpretable; 0 of 4 pass A4. Fixed-size sweep (docs/S2_PARENT_SWEEP.md): the parent step alone at α 0.10–0.30 gives −0.49 to −0.70 (lowest at 0.20, range 0.21), practice is flat and keeps every α, and the page interval lies below 0 only at α 0.10 and 0.15. Step size accounts for the spread; practice cannot choose it.
14. E3 of the cross-vendor plan (planted defects with Gemini and GPT): waits for a go-ahead (`docs/XVENDOR_PLAN.md`).

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
