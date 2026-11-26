# Results

Every entry names its run folder or script, the practice/held-out identity
and the digests, as docs/PROTOCOL.md asks. Practice numbers are labelled as
such and are not results in the paper sense.

## R1 — lossless rewrite `Cr255,1,85,1,1 → Cr1,1,85,1,1` (06.11.2026)

A function-preserving edit of the kraken+ head: the trailing conv sits on a
one-row feature map, so 254 of its 255 kernel rows only ever multiply zero
padding. `wsp.rewrite` keeps kernel row 127, rebuilds the model from the
rewritten VGSL spec, copies every other tensor unchanged, and saves CoreML
(`models/kraken/kraken-collapse-probe-131-cr1.rewrite.json` holds the record).

| | Cr255 (source) | Cr1 (rewritten) |
|---|---|---|
| file | `kraken-collapse-probe-131-cr255.mlmodel`, sha256 `fa74476b…`, 60,813,907 B | `kraken-collapse-probe-131-cr1.mlmodel` |
| parameters | 15,207,871 | 4,153,791 (−11,054,080 = 85 × 512 × 254) |
| `C_12.co.weight` | (85, 512, 255, 1) | (85, 512, 1, 1) |
| practice set | `medieval-val-300` (300 lines, list sha256 in `data/practice/README.md`) | same |
| text identical | — | **300 / 300 lines** |
| max \|Δ softmax\| | — | 1.1 × 10⁻⁶ (float32 rounding) |
| CER (practice, C0) | 0.9900 | 0.9900 (by construction) |
| serving loader | — | `load_any` → `TorchSeqRecognizer`, model_type `recognition` |

Source checkpoint: the only `Cr255` kraken+ checkpoint still on disk,
`~/atr-cache/checkpoints/20261027T072036Z-kraken-collapse-probe-131/best_0.0083.mlmodel`
— a from-scratch run that collapsed (val accuracy 0.0083), which is why the
CER is 0.99 on both sides. The archived kraken+ run of 01.09.2026
(`kraken-plus-shard00`, val 0.7927) already has the `Cr1` head (4,148,545
parameters), so the rewrite was *not* needed for it. The padding rows of the
Cr255 checkpoint are at their initialisation norm (12.045 ± 0.027 against
12.38 for the live row; docs/READINGS.md), as expected for weights that never
receive a gradient. Reproduce:

```bash
cd ~/Repo/weightspace && export TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES=""
.venv/bin/python -m wsp.rewrite --src models/kraken/kraken-collapse-probe-131-cr255.mlmodel \
    --dst models/kraken/kraken-collapse-probe-131-cr1.mlmodel --practice data/practice/medieval-val-300.jsonl
```

What it is worth: a trained `Cr255` model saved by the trainer is 61 MB
instead of 17 MB and takes 11 M useless multiplies per frame; the same
function fits the serving box's 16 MB convention. `KRAKEN_PLUS_SPEC` in
`training-atr-models/src/atr_training/contracts.py` still says `Cr255`;
changing it to `Cr1` for future runs is a trainer decision, not a
weight-space one.

## R2 — session `20261106-german-xix-blb-v1-b10-1`: output-layer levers do not move this model (06.11.2026)

First budgeted session (docs/PHASE1_REPORT.md). Subject `kraken-german-xix-blb-v1` (tensor digest `f35e075a…`), practice `blb-train-300` (sha256 `4a803b49…`), held-out `blb-eval` (5,628 lines, sha256 `0729b93b…`, audit clean), budget 10, 8 passes used, 7 edits, 2 kept, one element changed (`O_18.lin.bias[0]` 1.2323 → 0.2323).

| | practice CER | held-out CER | collapsed (held-out) |
|---|---:|---:|---:|
| before (C0) | 0.06458 | 0.12335 | 428 |
| after (frozen) | 0.06435 | 0.12325 | 403 |
| Δ, 95 % paired interval | −0.023 pt [−0.34, +0.30] | **−0.010 pt [−0.048, +0.026]** | |
| parent `xix-v1` (C1) | 0.3039 | 0.3196 | 455 |

Not a gain. The blank-bias, space-bias and frequent-row hypotheses of docs/READINGS.md each reduce the deletions they target and add insertions or substitutions of the same size; the 12 typographic rows are inert (a measured null). The deletions are decided below the output layer.

## R3 — session `20261106-bern-combined-v1-b10-1`: label-noise damage is not removable by transplant (06.11.2026)

Target 1 re-framed (docs/HERTWIG_REVISIONS.md, docs/PHASE3_REPORT.md). Subject `kraken-bern-combined-v1` (trained on the 25.10 export with 44 of 50 Hertwig images swapped), donors `hertwig-v2` (parent) and `schlussredaktion-v1` (clean sibling), practice `bern-train-300` (sha256 `474414e3…`), held-out `bern-eval` (545 lines, sha256 `f7b024e0…`), budget 10, 9 passes, 8 edits, 1 kept (output layer from the parent).

| | practice CER | held-out CER | held-out Hertwig | held-out Schlussred. |
|---|---:|---:|---:|---:|
| before (C0) | 0.1456 | 0.1933 | 0.2669 | 0.1348 |
| after (frozen) | 0.1414 | 0.1948 | 0.2639 | 0.1399 |
| Δ, 95 % interval (points) | −0.42 [−0.85, −0.04] | **+0.15 [−0.10, +0.42]** | −0.30 [−0.72, +0.13] | +0.51 [+0.21, +0.82] |
| parent / sibling, held-out all | | 0.2749 / 0.2036 | | |

Not a gain: the kept edit's practice gain came from the parent's own training lines. On correctly paired pages the subject is already the best family-1 model; no block of the clean sibling improves it.

**Replicate with Claude Fable 5.1 as programmer (11.11.2026; `runs/20261111-bern-combined-v1-f1-b10-1`, docs/KRAKEN_S2_FABLE.md).** The subject, practice set, held-out set, budget and keep rule are the same; the programmer was a Fable subagent that did not see session 2. 9 passes, 8 edits, 3 kept: blank bias −0.20 twice, then every tensor 12.5 % toward the parent. The same direction as Opus (toward `hertwig-v2`; the sibling was refused in both), as a small whole-network step rather than an output-layer transplant. Practice 0.1456 → 0.1357. Held-out 0.1933 → **0.1875**, Δ −0.57, page-clustered [−1.15, −0.05]; Hertwig −1.32 [−2.36, −0.60], Schlussredaktion +0.02. Against Opus's frozen model, Δ −0.73 [−1.19, −0.35]. C5 (5 norm-matched shams): held-out −0.12 to +0.02 and none kept. The gain is the interpolation's (−0.53 alone); the blank steps alone give +0.04 on held-out. It matches C4 arm A (R11) in size and place. A first programmer attempt was discarded before any pass because the pre-registration it read named session 2's kept edit. At the user's request the frozen model is registered on the share as `kraken-bern-combined-wsp-v1` and was read through the serving path (14 `bern-eval` pages, both models in one run, then disabled again): character-weighted −2.16 [−3.32, −1.05], 11 of 14 pages better, but the gain lies on Schlussredaktion (−3.84 [−4.89, −2.91], 8 of 8) and not on Hertwig (−0.04 [−0.68, +0.56]), the reverse of the line-level split; the gateway segments the pages itself. Read the same way, the interpolation part alone gives −1.86 [−2.82, −1.03] and the blank part alone −0.41. Opus's frozen model gives −0.74 [−1.38, −0.17], also gained on Schlussredaktion, against +0.15 on lines. So the hand reversal belongs to the measurement, and Fable beats Opus on both measures (−1.41 through the gateway; KRAKEN_S2_FABLE §6).

**Second replicate with Claude Fable 5.2 as programmer (15.11.2026; `runs/20261115-bern-combined-v1-f2-b10-1`, docs/KRAKEN_S2_FABLE52.md).** The setup was the same, and the programmer did not see either earlier run. It was pinned to `claude-fable-5-2` and verified from its transcript; it ran as a headless Claude Code session, not a subagent. 8 passes, 7 edits, 2 kept: every tensor 25 % toward the parent, then the blank bias −0.30. The kept diff has cosine +1.000 with Fable 5.1's and twice its norm, so it lies on the same parent line at double the step. Practice 0.1456 → 0.1360. Held-out 0.1933 → **0.1868**, Δ −0.65, page-clustered **[−1.64, +0.24]**, so not a gain under A4. Hertwig −2.35 [−3.51, −1.40]; Schlussredaktion **+0.69 [+0.19, +1.33]**, a loss. Against Fable 5.1, Δ −0.08 [−0.61, +0.44]; against Opus, Δ −0.80 [−1.60, −0.09]. C5 (5 norm-matched shams): held-out +0.05 to +0.22, none kept. The interpolation alone gives −0.63 on held-out; the blank step alone gives +0.01. All three programmers moved toward `hertwig-v2`. None kept the sibling. Blank steps were kept on practice and never transferred. The step size varied: doubling it doubled the Hertwig gain and began to cost Schlussredaktion. One run per model cannot separate a model effect from run-to-run variation. Nothing was registered, so there are no serving-path numbers.

**Replicates with Gemini 4 Argon and GPT-6.2 Astra as programmers (19.11.2026; `runs/20261119-bern-combined-v1-{g1,o1}-b10-1`, docs/XVENDOR_REPORT.md §1–5).** Same setup and brief; both C0 baselines reproduced. Gemini kept two edits (`O_18` to the parent–sibling midpoint, then a 50/20/30 subject/parent/sibling blend), GPT one (a 35/30/35 blend). Held-out, page-clustered: Gemini −0.65 [−1.38, +0.00], GPT −0.57 [−1.46, +0.24]; Hertwig −1.76 and −2.13; Schlussredaktion +0.23 and **+0.67 [+0.18, +1.21]**. Pairwise against Fable 5.1, every difference is under 0.1 point. C5 (5 norm-matched shams each): all made held-out worse, none kept. **Across the five programmers of session 2**, every one gave the parent weight; the four whole-network steps gain −0.57 to −0.65 on held-out, and only Fable 5.1's page interval lies below 0. The finding is the direction, not a single session's gain; run-to-run variation is unmeasured.

**Repeats with one programmer model: three more Fable 5.2 runs (24.11.2026; `runs/20261124-bern-combined-v1-f2-b10-{2,3,4}`, docs/S2_REPEATS.md, pre-registered in docs/S2_REPEATS_PREREG.md).** Same setup as the first Fable 5.2 run; fresh programmers, `claude-fable-5-2` verified in every transcript, no run exposed, C0 reproduced in each. All three kept the same two moves as the first: a whole-network step toward the parent (α 0.25, 0.30, 0.25), then a blank step (−0.70, −0.40, −0.25). Held-out, page-clustered: −0.55 [−1.51, +0.35], −0.47 [−1.58, +0.57], −0.67 [−1.64, +0.22]; Hertwig −2.22 to −2.50, Schlussredaktion +0.67 to +1.14 (page intervals above 0). C5 (15 shams): held-out +0.05 to +0.33, none kept. Split: the parent step alone gives −0.63 at α 0.25 and −0.54 at α 0.30; the blank steps alone −0.03 to +0.04. Across the four Fable 5.2 runs the held-out Δ spans 0.20 points (SD 0.093). That is less than across the five programmers (0.80, all from Opus's single run) but more than across the four parent-step programmers (0.085). By the pre-registered rule their differences, including Fable 5.1's page interval below 0, are not interpretable as model effects. The direction is robust to the run (4 of 4); passing A4 is not (0 of 4).

**The parent step at fixed sizes, no programmer (24.11.2026; docs/S2_PARENT_SWEEP.md, pre-registered in docs/S2_PARENT_SWEEP_PREREG.md).** The programmers' parent step alone, at α 0.10–0.30 in steps of 0.05; α 0.25 and 0.30 reproduce the split digests and scores. Held-out Δ: −0.49, −0.59, −0.70, −0.63, −0.54 (line intervals all below 0), lowest at α 0.20; Hertwig falls monotonically (−1.21 to −2.57) and Schlussredaktion rises (+0.09 to +1.06). The range over α (0.21) matches the within-model range of the Fable 5.2 runs (0.20), so step size alone can account for the run-to-run spread. Practice keeps every α and is nearly flat (−0.65 to −0.82, lowest at 0.15), so it cannot choose the size. Page intervals below 0 at α 0.10 and 0.15 only (two of five looks on held-out, not claimed as a gain): the page interval widens with α as the Schlussredaktion cost grows, which is why the small step of Fable 5.1 (α 0.125) passed A4 and the α 0.25–0.30 runs did not.

### Note to R2 (07.11.2026)

Re-aggregated on `blb-eval-hand` (the handwriting held-out set chosen by data properties, docs/PHASE5_CLEANUP_REPORT.md; a re-aggregation of the one held-out reading, not a second evaluation): before 9.553 %, after 9.553 %, Δ −0.001 points [−0.055, +0.053]. The conclusion of R2 is unchanged.

## R4 — `kraken-latin_incunabula-wsp-v1`: a block exchange, held out and through the serving path (07.11.2026)

The first candidate from Track C (docs/MODEL_FAMILY.md, RQ3). No training: kraken-latin_incunabula
(Zenodo 10.5281/zenodo.11113737, file sha256 of the served model in `docs/family/weights.json`) with
its first two group norms `Gn_1`, `Gn_3` replaced by those of its base kraken-catmus_print_large and its
output layer `O_15` moved half-way to the base's (rows paired by character). Chosen on practice lines,
confirmed on disjoint lines (`docs/family/combine-latin_incunabula-from-catmus_print_large.json`), then
scored once. Run folder `runs/20261107-latin_incunabula-exchange-1` (session.json, summary.json, diff.json;
rebuilt from the combination with `wsp.exchange --candidate`, tensor digest `cc15de58…` equal to the
measured one); file sha256 `d8b9bc07…`; loads with `load_any`.

**Held-out set** `data/heldout/reichenau-heldout.jsonl`: 60 pages of the *train* shard of
dh-unibe/image-text_reichenau-inkunabeln (30 German, 30 Latin-German, seed 20261109), chosen among pages that
share neither name nor image with the validation shard the practice and confirmation lines came from; 4,374
lines on 58 of them; audit (`reichenau-heldout.audit.json`): 0 pages, 0 lines, 0 line crops shared with the
practice and confirmation sets. latin_incunabula was trained on the Latin sub-collection of the same Zenodo
ground truth, which dh-unibe's copy does not contain; these pages are the same printers and typefaces, not its
training pages.

| measurement | unedited | candidate | Δ (points) | 95 % interval |
|---|---:|---:|---:|---|
| line level, 4,374 held-out lines (`wsp.run heldout`) | 5.01 % | 4.99 % | −0.03 | [−0.07, +0.02] |
| serving path, 58 pages through the gateway, median page CER | 4.16 % | 4.05 % | −0.19 (median of page deltas) | [−0.27, −0.10] |
| serving path, character-weighted | 15.40 % | 15.17 % | −0.23 | [−0.34, −0.13] |

Serving path: registered on the share (`registry/trained/kraken-latin_incunabula-wsp-v1.yaml`, weights under
`training_folder/trained/`), enabled for the evaluation only, read with `start_batch` (run
`wsp-reichenau-heldout-v1`, source `datasets:reichenau-incunabula-wsp/reichenau_eval`, smoke run of 3 pages
first), scored with `score_ground_truth` by page name (report `score-gt-20261107T183547`), then **disabled
again**: the candidate is registered but not offered to agentic_historian until someone promotes it.
Per page: better on 40, worse on 17, equal on 1 (`runs/…/serving_path.json`; paired bootstrap over pages,
2,000 resamples). The weighted CER is far above the line-level one because the gateway segments the pages
itself and a few pages segment badly (worst 71 %), for both models alike.

**Reading.** On its own collection the candidate is at least as good: within noise on lines cut from the
ground-truth geometry, slightly but measurably better through the serving path. Its purpose, reading print it
was not trained for, was measured on practice and confirmation lines only (Fraktur −0.6 / −0.9, NewsEye
−2.4 / −1.8 points); no held-out Fraktur set was scored. **The block statistics show the price:** on the held-out
lines the abbreviation signs get worse — MUFI characters 532 → 731 errors (+0.11 points of CER, interval
excluding 0), combining marks +0.05, Latin Extended-D +0.006 — while lowercase letters get better (−0.08). Moving
the output layer towards a print model that does not use the Reichenau abbreviation conventions costs exactly
the characters this model is specialised in; for an edition that keeps abbreviations, that is a regression the
total CER hides (`runs/…/heldout_paired.json`).

**Data finding on the way:** in dh-unibe/image-text_reichenau-inkunabeln, 39 of the 59 validation pages are
byte-identical to train pages (the upstream Zenodo record lists them under both its Train Set and Validation Set
folders). Any training on this dataset that uses its validation split for model selection overfits the selection.

## R5 — weight survey of the kraken zoo (07.11.2026; docs/ZOO_SURVEY.md, data docs/zoo/)

30 of the gateway's 37 Zenodo kraken models were downloaded and read (seven not obtained), no forward pass. 21 of 30 (70 %) have exactly the stack of the in-house models; all 30 fall into three architecture families. Weights agree with the documented lineage in 11 of 13 documented pairs (one contradiction, one architecture mismatch); in a group of 28 models only three clusters are related at all; with a distance of about 1.41 as the null, lineage is recoverable for fine-tunes that did not move far (0.1–0.3) and visible but less clean for heavy retrainings (0.7–0.9); a shared name does not predict it (1 of 17 name-sharing pairs related), a near-identical codec does (23 of 23). Blank-bias rise holds relative to the other biases in 11 of 11 pairs (5 of 5 independent zoo pairs). Rare-row inflation holds in direction, not as "the rows that moved most" (all four criteria in 1 of 5 independent pairs). "Conv moves less than recurrent and output" is not supported (2 of 5); the first conv block moves least and movement grows with depth. Manuscript Section 6.4.

## R6 — seed-variance null (07.11.2026; docs/SEED_NULL.md, data docs/seed-null/)

Five runs of the `hertwig-v2` recipe differing only in seed (91 epochs, own 45/5-page split, GPU card 0, nothing registered). Seed against seed: output rows 0.12–0.15, pooled conv 0.24, pooled LSTM 0.28; parent → run: 0.41, 0.27, 0.31. Seed noise is about a third of the movement of the output rows and about 90 % of the pooled stacks; the sibling pair (0.16) is the size of seed noise; for the first conv block and the first LSTM block change cannot be told from noise. **Survives:** blank bias +0.53–0.55 relative (seed vs seed −0.02 to +0.01), rare rows grow and drop their bias in every parent → run pair (3 and 2 of 10 seed pairs at chance), recurrent change increasing with depth. **Does not survive:** the depth pattern of the conv stack (it appears between seeds too), "rare rows move most" (they move less than frequent letters between seeds), any claim from a size of change alone. Held-out `bern-eval`: seeds 27.20–27.65 % (sd 0.18), v2 27.49 %; every seed's paired interval against v2 contains 0 (about ±0.5 points); differences under about one point between two runs are not distinguishable from seed.

## R7 — planted defects and validation-based practice (07.11.2026; docs/CONTROLS_REPORT.md §2, §3, §9b, §9c)

Positive control on `xix-blb-v1` (held-out 9.55 %): of eight corruptions, the programmer with passes, the blind programmer and a hand-written repair list each restored the held-out CER to within 0.22 points for all six corruptions that cost at least one point (1.3 to 90 points; one corruption, output weight ×2, was a null). Absence of a found defect on the real models therefore carries information. But the list did as well as the programmer and the blind programmer as well as both; a coordinate search over output biases repaired little. Diffuse defects (noise on all weights, 10 %): only the programmer with passes found a repair (interpolation toward the parent, 83 % of the damage); block noise 51–70 % by interpolation; with an unrelated donor, zeroed rows or blocks were not rebuilt (0–19 %), a doubly-scaled block was undone without a donor. Validation-based practice (lines the subject was not trained on): `xix-blb-v1`, 0 of 6 attempts kept an edit; `combined-v1`, five attempts moved toward the sibling and gained −0.19 to −0.39 points held-out, none with a page-clustered interval excluding 0. A Fable 5.1 attempt (09.11.) kept 0 of 9 edits on `xix-blb-v1` (0 in 7 attempts in all) and went half-way toward the sibling on `combined-v1`: practice 13.86 → 11.60, held-out −0.02.

**Fable 5.1 as programmer (11.11.2026; docs/CONTROLS_REPORT.md §11).** The LLM arms were rerun with Claude Fable 5.1 programmers, one fresh subagent per run, never told the defect, with the same subjects, practice sets, budgets, keep rule and held-out sets. That is 30 runs: the programmer with passes and the blind programmer on k1–k11, on the three unrelated-donor variants and on adapter p3. Of the 22 kraken cells with a defect, Fable repaired 13, partly 4 and missed 5; Sonnet repaired 15, partly 2 and missed 5. Both models repaired every localised defect against the parent, in both arms. 14 of the 28 Fable final models are identical to Sonnet's, and on adapter p3 both Fable arms reach the Sonnet adapter (0.517 against clean 0.507). Four cells differ. On k9 (noise everywhere) Fable with passes recovered 24 % against Sonnet's 83 %: it shifted the blank bias and never tried interpolating toward the parent. So the one cell where passes alone made the difference does not reproduce across models. On k4 (b) the blind Fable programmer judged the blank shift natural and left the defect (0 % against 100 %). With only an unrelated donor, Fable did better on k1-u (b) (43 %, an `e`-bias edit, against 0 %) and on k8-u (a) (15 % against 1 %). One attempt per cell and model, except adapter p3, which got a second Fable attempt on 13.11. with fresh programmers: both arms again halve every `lora_B` and reach the same adapter (0.517), so on p3 the result does not vary across two attempts per arm.

## R8 — sham, global interpolation, margins (07.11.2026; docs/CONTROLS_REPORT.md §6–§8)

(a) Sham: 20 norm-matched random edits plus the same blank shift gave a held-out mean −0.013 (sd 0.062, range −0.199 to +0.097); the real `combined-v1` edit −0.39; 0 of 20 shams reach it and the keep rule kept 0 of 20. (b) Global interpolation parent → child, Kurrent pair: pairs of blocks are additive at α = 0.5, the whole network is not (joint cost exceeds the sum of single-block costs by +3.81 points [+2.57, +4.98] on the 1,718 held-out lines at α = 0). (c) Margins at deleted frames (696 deleted space, `e`, `n`; `xix-blb-v1`, `blb-eval-hand-b`): 18 % have no frame between the neighbours; of the rest the median deleted character is 3.2 nats below the blank at its best frame; blank-bias edits recover one deletion in 40, 9, 5 at −0.1, −0.5, −1.0 and pay with nearly as many insertions at −1.0 (−152 deletions, +133 insertions): the deletions are decided upstream of the output layer.

## R9 — which latin_incunabula candidate to enable: neither, by the pre-registered rule (07.11.2026)

Rule written and committed before any of these scores (`runs/20261107-latin_incunabula-exchange-2/preregistration.md`,
commit `0c8f4ff`): a candidate qualifies if it does not lose abbreviations (MUFI, Combining Diacritical Marks or Latin
Extended-D with a 95 % interval entirely above 0 on `reichenau-heldout`, as a whole or in either sub-collection), its
Reichenau CER is not worse, and its CER on neither held-out Fraktur set is worse.

**Candidates.** v1 = R4 (`Gn_1` + `Gn_3` from kraken-catmus_print_large, `O_15` half-way). v2 =
`runs/20261107-latin_incunabula-exchange-2` (`Gn_1` + `Gn_3` only, rebuilt from combination `gn13`, tensor digest
`1fe624af…`, loads with `load_any`).

**Held-out sets, each read once per model** (line level; audits in `data/heldout/*.audit.json`, all 0 shared pages,
lines and crops):

- `reichenau-heldout`: R4's 4,374 lines. The German sub-collection contributes only 163 of them, from 30 pages with
  few transcribed lines; the Latin-German one 4,211.
- `newseye-heldout`: 2,706 lines from 8 pages of the NewsEye ONB *train* shard; the practice and confirmation lines came
  from its validation shard, which shares no page name or image with it.
- `gt-fraktur-heldout`: 1,311 lines from the 35 GT-Fraktur pages no practice or confirmation line came from (same 10
  documents, other pages).

| candidate | `reichenau-heldout` CER (Δ, 95 % interval) | `newseye-heldout` CER (Δ, 95 % interval) | `gt-fraktur-heldout` CER (Δ, 95 % interval) | loses abbreviations | qualifies |
|---|---|---|---|---|---|
| unedited | 5.01 % | 28.56 % | 20.13 % | | |
| v1 | 4.99 % (-0.03 [-0.07, +0.02]) | 26.52 % (**-2.04** [-2.22, -1.86]) | 19.70 % (**-0.44** [-0.57, -0.31]) | yes | no |
| v2 | 5.06 % (**+0.04** [+0.01, +0.08]) | 27.48 % (**-1.08** [-1.26, -0.92]) | 19.94 % (**-0.19** [-0.30, -0.08]) | yes | no |

Abbreviation blocks on `reichenau-heldout`, change of the block's share of CER in points (95 % interval):

| candidate | scope | MUFI (CERberus list, overlay) | Combining Diacritical Marks | Latin Extended-D |
|---|---|---|---|---|
| v1 | all (4374 lines) | **+0.110** [+0.092, +0.127] (532→731) | **+0.048** [+0.024, +0.072] (2710→2797) | **+0.006** [+0.001, +0.010] (50→60) |
| v1 | German (163 lines) | +0.039 [-0.013, +0.101] (9→12) | **-0.221** [-0.380, -0.085] (143→126) | +0.000 [+0.000, +0.000] (3→3) |
| v1 | Latin-German (4211 lines) | **+0.113** [+0.095, +0.132] (523→719) | **+0.060** [+0.037, +0.083] (2567→2671) | **+0.006** [+0.001, +0.011] (47→57) |
| v2 | all (4374 lines) | **+0.010** [+0.003, +0.019] (532→551) | +0.010 [-0.004, +0.024] (2710→2728) | +0.000 [-0.003, +0.003] (50→50) |
| v2 | German (163 lines) | +0.013 [-0.027, +0.063] (9→10) | +0.039 [-0.054, +0.139] (143→146) | +0.000 [+0.000, +0.000] (3→3) |
| v2 | Latin-German (4211 lines) | **+0.010** [+0.002, +0.019] (523→541) | +0.009 [-0.004, +0.022] (2567→2582) | +0.000 [-0.003, +0.003] (47→47) |

**Choice by the pre-registered rule:** none.

(Numbers: `docs/family/candidates-latin_incunabula.json`, from `python -m wsp.candidates latin_incunabula`, which reads
the stored readings only.)

**Reading.**

- **The purpose of the exchange is confirmed on held-out print.** v1 reads Fraktur it was never trained on better:
  −2.04 points on NewsEye, −0.44 on GT-Fraktur; v2 half of that (−1.08, −0.19). Stabilisation by exchange is real
  beyond the lines it was chosen on.
- **Its price is the model's specialisation.** v1 loses abbreviations clearly on the Latin-German pages (MUFI signs
  523 → 719 errors). v2, without the output-layer move, keeps most of them but not all: MUFI 523 → 541, interval
  excluding 0, and its Reichenau CER rises by 0.04 points (interval excluding 0). Both changes are small; the rule set
  no minimum size, and it is not changed after the fact.
- **Decision: neither is enabled.** v1 stays registered on the share and disabled (R4); v2 was not registered. The
  serving-path read the rule reserved for a qualifying candidate was not run.

What would settle it differently: a rule with a minimum effect for the abbreviation blocks (as the session keep rule
has since PROTOCOL §3 of 06.11.2026), set *before* a new comparison; or a use case that wants Fraktur robustness more
than abbreviations, for which v1 is the better model — a decision for its users, not for this rule.

## R10 — adapter track: one QLoRA adapter of Qwen3.5-4B (07.11.2026; docs/ADAPTER_REPORT.md merged into main 08.11.2026)

Rank-64 update of 128 modules (Qwen3.5-4B, Kurrent pages). Readings: norms tight across three healthy siblings, no outlier; adapters of one base nearly orthogonal (cosine 0.10), so no donor operations. Positive control: of six planted defects two did damage beyond pass noise; the doubled scale (CER 3.76 → 0.517, clean 0.51) was repaired by programmer, blind programmer and rule list alike; eight late `down_proj` ×3 (+24 points) recognised by the rule list but its exact repair refused by the keep rule; four single-module defects did no damage. Real session (v2, budget 10; 7 passes, 6 edits, 1 kept: layer-27 `q_proj` ×0.8): held-out 134 pages CER 0.685 → 0.569, Δ −0.115 [−0.271, −0.001] by pages, sham +0.032; clipped CER unchanged (0.354 → 0.351), length ratio 1.33 → 1.22, 4 of 22 capped pages stop. Engines disagree (identical text on 10 of 61 pages; merged vs peft 3 of 10), so loop status is chaotic; the base alone loops more (12 of 32 pages at the ceiling vs 8). Manuscript Section 9.3.

## R10b — phase 5 adapter session, replicated with a Fable 5.1 programmer (10.–11.11.2026; docs/PHASE5_FABLE_REPORT.md)

Session `20261110-kurrent-page-v2-f1-b10-1`, the same subject (v2, `fb4f55ee…`), practice set (`kurrent-val-64`), held-out set (733 test pages, 70 documents), budget 10, engine (`evaluate_qlora`) and keep rule as the Opus 5.5 session `20261107-kurrent-page-v2-b10-1` (docs/PHASE5_REPORT.md). The programmer was Claude Fable 5.1, which did not see the Opus session. It used 5 passes and tried 4 edits, and kept 1: direction 0 of `layers.31.self_attn.o_proj` out of `B`. That is **the same edit as Opus, and the frozen adapter is byte-identical** (`3f68caa5…`). Practice CER 0.4515 → 0.3470, −10.45 points [−20.3, −2.4]. Held-out, scored once: 0.4012 → 0.4036, **+0.24 points [−3.5, +3.8] by pages, [−5.0, +6.7] by documents**; all 733 readings are byte-identical to the Opus session's. The refused follow-ups were directions 1–2 of the same module, late attention × 0.5, and the layer 31/27 `down_proj` pair, which Opus tried too. Two new C5 shams (random unit direction out of the same `B`): −5.68 [−11.8, −1.2] and −1.33 [−5.6, +2.6] against C0, with the phase-5 sham −3.87. **One of three random directions passes the keep rule**, and H1's interval against a sham excludes 0 for only one seed. The phase-5 null does not depend on the programmer model, and on 64 pages the keep rule does not separate the lens direction from a random one.

**Replicates with Gemini 4 Argon and GPT-6.2 Astra (19.–20.11.2026; docs/XVENDOR_REPORT.md §6).** Both kept the same single edit, the top reading direction removed from layer 31's writers, and froze on byte-identical adapters. Practice −7.78 points [−17.3, −0.4]; held-out **+1.13 [−3.0, +5.2]**, a null. Two of three random directions removed from the same tensors would also have been kept on practice, and H1 does not differ from any of them. Four programmer models, one family of edit, no held-out gain.

## R11 — C4: retraining `combined-v1` on the corrected data (08.11.2026; docs/C4_RETRAIN.md merged into main 08.11.2026)

Preregistered (commit 2ea91a3) before any run. Subject `kraken-bern-combined-v1`; 151 pages, 4,927 lines (Hertwig from the 22.10 export, correct images); validation the trainer's 17 pages; held-out `bern-eval` (545 lines, 14 pages), scored once per model on CPU; `ketos` 7.0.2 on card 0, nothing registered. Arm A (C4 as in PROTOCOL §4: 5 epochs, lr 1e-5, seeds 1-3): held-out 19.33 → 18.77-18.79 %, Δ -0.54/-0.56/-0.56 (page-clustered about [-1.1, -0.1]), equal to the interpolation edit (-0.39) by the rule. Arm B (lr 5e-5, secondary): Δ -0.98 to -1.02, larger than the interpolation edit (-0.6, page interval reaching 0). Seed sd within an arm 0.01-0.02. The gain lies on the Hertwig lines (-1.30 A, -2.22 B) and not on the Schlussredaktion lines. Not shown: that the correction, and not additional training, causes the gain (no control on uncorrected pairs). Manuscript Section 7.11.

## R12 — latin_incunabula v3: an exchange that spares the abbreviation rows qualifies under a rule with minimum effects, and holds through the serving path (08.11.2026)

Rule written, committed and pushed before any v3 held-out score (`runs/20261108-latin_incunabula-exchange-3/preregistration.md`,
commit `fdf25e2`). **Candidate v3**: `Gn_1` + `Gn_3` from kraken-catmus_print_large and `O_15` half-way, but only on the
rows of the 108 characters both codecs share that are not abbreviation characters (MUFI overlay, Combining Diacritical
Marks, Latin Extended-D), plus the blank row; the 13 shared abbreviation characters (U+0303, U+0304, U+0307, U+0313,
U+0328, U+0365, ꝑ ꝓ ꝗ ꝙ ꝟ ꝯ, ω) keep latin_incunabula's rows. Rows paired by character. Tensor digest `bd0f11ec…`,
file sha256 `11fffd8a…`, loads with `load_any`; weights not committed, nothing registered.

**Design, practice and confirmation lines only.** Four variants tried (all `Gn_1` + `Gn_3`: abbreviation rows spared
with the blank moved / kept; letters a–z A–Z only at β 0.5 / 0.25), next to v1 and v2 rebuilt bit-exactly as references
(`docs/family/combine-v3-latin_incunabula-from-catmus_print_large.json`). On 300-line sets the four were alike on the
abbreviation blocks (MUFI 32 → 33–34 and 28 → 29, v1 46 and 41); the one with the largest mean Fraktur gain went to the
held-out sets alone.

**Rule** (points of CER). *Loses abbreviations*: for MUFI, Combining Diacritical Marks or Latin Extended-D, on
`reichenau-heldout` or either sub-collection, the line-level interval of the change of the block's share of CER lies
entirely above +0.02. *Reichenau worse*: line-level interval of the total change entirely above +0.05. *Fraktur*: worse
on a set if its line-level interval lies entirely above 0; better if its **page-clustered** interval lies entirely below
0. Qualifies: no abbreviation loss, Reichenau not worse, no Fraktur set worse, at least one better. The minimum effects
were chosen from R9's numbers (v1 MUFI +0.110, v2 +0.010; v2 Reichenau +0.04); R9's decisions on v1 and v2 stand and are
not re-judged under this rule.

**Held-out sets, read once by v3** (as R9; unedited readings are R9's stored ones). Intervals: line level, then
page-clustered (two-stage bootstrap over pages, then lines; 2,000 resamples, seed 20261106).

| candidate | `reichenau-heldout` CER (Δ, 95 % intervals) | `newseye-heldout` CER (Δ, 95 % intervals) | `gt-fraktur-heldout` CER (Δ, 95 % intervals) | loses abbreviations | qualifies |
|---|---|---|---|---|---|
| unedited | 5.01 % | 28.56 % | 20.13 % | | |
| v3 | 4.99 % (−0.02 [−0.06, +0.01]; pages [−0.09, +0.04]) | 26.90 % (**−1.66** [−1.85, −1.49]; pages [−2.02, −1.30]) | 19.92 % (**−0.21** [−0.34, −0.08]; pages [−0.40, +0.01]) | no | **yes** |
| *R9 v1, for comparison* | *−0.03 [−0.07, +0.02]; pages [−0.12, +0.05]* | *−2.04 [−2.22, −1.86]; pages [−2.42, −1.65]* | *−0.44 [−0.57, −0.31]; pages [−0.65, −0.21]* | | |
| *R9 v2, for comparison* | *+0.04 [+0.01, +0.08]; pages [−0.01, +0.10]* | *−1.08 [−1.26, −0.92]; pages [−1.41, −0.73]* | *−0.19 [−0.30, −0.08]; pages [−0.37, −0.01]* | | |

Collapsed lines unedited → v3: 0 → 0, 75 → 65, 15 → 15.

Abbreviation blocks on `reichenau-heldout`, change of the block's share of CER in points (line level; pages):

| candidate | scope | MUFI (CERberus list, overlay) | Combining Diacritical Marks | Latin Extended-D |
|---|---|---|---|---|
| v3 | all (4374 lines) | +0.014 [+0.006, +0.024]; pages [+0.000, +0.030] (532→558) | +0.025 [+0.010, +0.041]; pages [+0.003, +0.049] (2710→2756) | +0.001 [−0.002, +0.004]; pages [−0.004, +0.007] (50→52) |
| v3 | German (163 lines) | +0.000 [−0.039, +0.038]; pages [−0.054, +0.052] (9→9) | −0.052 [−0.122, +0.013]; pages [−0.136, +0.024] (143→139) | +0.013 [+0.000, +0.040]; pages [+0.000, +0.055] (3→4) |
| v3 | Latin-German (4211 lines) | +0.015 [+0.006, +0.025]; pages [+0.000, +0.032] (523→549) | +0.029 [+0.014, +0.044]; pages [+0.007, +0.053] (2567→2617) | +0.001 [−0.003, +0.004]; pages [−0.005, +0.006] (47→48) |
| *R9 v1* | *all* | *+0.110 [+0.092, +0.127]; pages [+0.076, +0.144] (532→731)* | *+0.048 [+0.024, +0.072]; pages [+0.004, +0.089] (2710→2797)* | *+0.006 [+0.001, +0.010] (50→60)* |
| *R9 v2* | *all* | *+0.010 [+0.003, +0.019]; pages [−0.002, +0.024] (532→551)* | *+0.010 [−0.004, +0.024]; pages [−0.009, +0.031] (2710→2728)* | *+0.000 [−0.003, +0.003] (50→50)* |

Block statistics, all blocks that moved (share of CER, line level; pages):

| set | block | Δ share | errors |
|---|---|---|---|
| reichenau | Latin lowercase a–z | −0.052 [−0.085, −0.021]; pages [−0.112, +0.004] | 4377→4282 |
| reichenau | Latin-1 Supplement | +0.010 [+0.005, +0.016]; pages [+0.002, +0.021] | 669→688 |
| reichenau | Space | −0.010 [−0.021, +0.001] | 503→485 |
| reichenau | Latin uppercase A–Z | −0.008 [−0.015, −0.001] | 201→187 |
| reichenau | Latin Extended-A | +0.006 [−0.001, +0.012] | 258→268 |
| reichenau | General Punctuation (⁊ and others) | +0.003 [+0.001, +0.006] | 1→6 |
| newseye | Latin lowercase a–z | −1.341 [−1.486, −1.199]; pages [−1.628, −1.022] | 22001→20357 |
| newseye | uppercase / digits / ASCII punctuation | −0.096 / −0.081 / −0.055, all page intervals below 0 | |
| gt-fraktur | Latin lowercase a–z | −0.293 [−0.399, −0.185]; pages [−0.468, −0.112] | 6686→6473 |
| gt-fraktur | Latin Extended-A (ſ) | +0.107 [+0.075, +0.142]; pages [+0.057, +0.160] | 838→916 |
| gt-fraktur | Latin uppercase A–Z | +0.069 [+0.022, +0.117]; pages [+0.003, +0.138] | 2396→2446 |
| gt-fraktur | Combining Diacritical Marks | +0.048 [+0.026, +0.069]; pages [+0.019, +0.079] | 934→969 |

**Choice by the pre-registered rule: v3** (`docs/family/candidates-latin_incunabula-v3.json`, from
`python -m wsp.candidates latin_incunabula-v3`, which reads stored readings only).

**Reading.**

- **Sparing the abbreviation rows keeps most of the Fraktur gain and most of the abbreviations.** NewsEye −1.66 points
  (v1 −2.04, v2 −1.08), the only Fraktur gain the page-clustered interval supports; GT-Fraktur −0.21, line-level interval
  excluding 0, page interval reaching +0.01 (v2's size, half of v1's). MUFI on Reichenau +0.014 (532 → 558) against v1's
  +0.110 (532 → 731): v2's level.
- **The protection is not complete, and the rule lets that through by design.** Combining Diacritical Marks rise by
  +0.025 (2710 → 2756), between v2 (+0.010) and v1 (+0.048), although their rows were not moved: errors on a character also
  depend on the rows of its competitors and on the moved group norms (v2, no output move, had +0.010); which share is
  which was not tested. The point estimate lies above the +0.02 minimum; the line-level interval
  ([+0.010, +0.041]) does not lie entirely above it, so by the rule this is not a loss. Latin-1 Supplement (ß, ¬ and
  others, moved) +0.010. On GT-Fraktur the gain on lowercase letters is partly paid in ſ (Latin Extended-A +0.107), the
  combining marks and capitals.
- **Reichenau total unchanged** (−0.02, both intervals containing 0). Under the page-clustered interval, R9's "v2 worse
  by +0.04" would not have held (pages [−0.01, +0.10]); that decision stands as recorded.
- **Decision: v3 qualifies.** The three held-out sets have now been read by three exchange candidates of this model; any
  further candidate needs fresh held-out lines.

**Serving path (with the user's OK, 08.11.2026).** As R4: exported with `wsp.run export` (file sha256 `11fffd8a…`),
weights copied to `training_folder/trained/kraken-latin_incunabula-wsp-v3/` on the share, registered with
`atr_training.registration` (`registry/trained/kraken-latin_incunabula-wsp-v3.yaml`) disabled, enabled for the evaluation
only, read with `start_batch` (smoke run of 3 pages `wsp-reichenau-heldout-v3-smoke`, then run `wsp-reichenau-heldout-v3`,
source `datasets:reichenau-incunabula-wsp/reichenau_eval`, both models in the same run), scored with
`score_ground_truth` by page name (report `score-gt-20261108T203048`), then **disabled again**. The unedited model's
reading equals R4's on all 58 pages. Paired over pages (`wsp.serving_path`, 2,000 resamples, seed 20261106; on R4's
pages it reproduces R4's numbers):

| measurement | unedited | v3 | Δ (points) | 95 % interval | R4's v1 Δ |
|---|---:|---:|---:|---|---:|
| serving path, 58 pages, median page CER | 4.16 % | 4.14 % | −0.09 (median of page deltas) | [−0.16, −0.03] | −0.19 [−0.27, −0.10] |
| serving path, character-weighted | 15.40 % | 15.25 % | −0.14 | [−0.23, −0.07] | −0.23 [−0.34, −0.13] |

Per page: better on 36, worse on 15, equal on 7 (`runs/20261108-latin_incunabula-exchange-3/serving_path.json`). The
serving path is not worse; v3 reads its own collection through the gateway slightly better, by about half of v1's gain.
**v3 stays registered and disabled**, like v1: whether it is offered to agentic_historian is the user's decision. For an
edition that keeps abbreviations it is the better of the three candidates (MUFI +0.014 against v1's +0.110); its Fraktur
gain is about four fifths of v1's on NewsEye and half on GT-Fraktur.

## R13 — RQ3 on out-of-domain material: first-convolution exchange stabilises a little, below the pre-registered minimum; no exchange across lineages (08.11.2026; docs/MODEL_FAMILY.md, section of 08.11.2026)

Pre-registered (`docs/family/PREREG-ood-exchange.md`, commit `517606e`) before any held-out read. Subject `kraken-german-xix-blb-v1`, donor its parent `kraken-german-xix-v1`. Out-of-domain material the subject reads at 15–30 % and was never trained on: Swiss Federal Council minutes 1848–1903 (18.2 %; parent 11.3 %) and the Basel land register (28.2 %; parent 25.1 %), split by document before any reading; held-out `federal-minutes-heldout` (545 lines, 30 source documents), `basel-grundbuch-heldout` (1,888 lines, 111 pages), own `blb-eval-hand-b` (1,718 lines, 70 pages), each read once on CPU. Candidates from practice and line-disjoint confirmation: A = `C_0` half-way (confirmation: federal −2.31, own −0.02), B = `C_0` transplanted (−2.84, own +0.27). Rule: federal Δ ≤ −1.0 with page-clustered upper end ≤ −0.5; own and Basel upper ends ≤ +0.5.

Held-out, Δ CER in points (page-clustered 95 % interval): A federal −0.35 [−0.57, −0.14], own +0.05 [−0.02, +0.13], Basel −0.50 [−0.74, −0.24]; B −0.46 [−0.75, −0.17], +0.11 [−0.01, +0.25], −1.07 [−1.40, −0.74]. **Neither qualifies (minimum effect on the federal minutes not met); neither is registered.** The stabilisation holds on held-out documents at no own-material cost, but is a sixth of the practice estimate; post hoc, the practice gain sat in a few source documents the held-out side does not contain, so confirmation sets should be document-disjoint, not only line-disjoint. Combined conv moves (`C_0`+`C_6`, `C_0`+`C_3`+`C_6`, all four) add own-material cost (+0.28 to +0.87). Bern lineage (`bern-hertwig-v2` on the A V 1447 Schlussredaktion hand, 29.6 %): its parent reads that hand worse (65.5 %); no conv move helps, the suggested `C_3`+`C_9` half-way costs +1.41; towards catmus_medieval a one-for-one trade; no candidate, held-out unread. Cross-lineage (`wsp/permute.py`): unit permutation by activation correlation is exactly function-preserving (every line identical) and recovers the identity within a lineage, but unrelated pairs (blb-v1 and hertwig-v2 ← McCATMuS) match at unit correlation 0.2–0.4 and every front-end transplant breaks the specific model (+2.6 to +92 points); no candidate, nothing read held-out.

## R14 — C4 control on the uncorrected 25.10 pairs: the gain of R11 depends on the correction (14.11.2026; docs/C4_UNCORRECTED.md, data docs/c4u/)

Preregistered before any run (commit `d24d0ce`). The recipe of R11 (`combined-v1`, 5 fixed epochs, 1e-5 and 5e-5, seeds 1–3, `ketos` 7.0.2, GPU card 0, nothing registered) with one change: the 44 Hertwig training pages as the original job saw them, text unchanged but the image another page's for 40 of 44. Held-out `bern-eval` (545 lines): mean Δ CER −0.07 (−0.05, −0.13, −0.04) at 1e-5 and −0.03 (−0.10, −0.07, +0.07) at 5e-5, against −0.55 and −0.99 on the corrected pairs; Hertwig lines −0.10 and 0.00 (corrected: −1.30, −2.22). Reading rule, both arms: the correction is what helps (U − C4 = +0.48 and +0.96, every U seed above every C4 seed); neither arm harms. Validation accuracy stays flat (0.858–0.862) where it rose in R11. Limits: the pairing is fully broken, not mildly noisy; no corrected-pairs run with another schedule; one subject, three seeds.

## R15 — TrOCR stage 1: the lineage of the Bern family is recoverable from the weights; the encoder separates it best (11.11.2026; docs/TROCR_READINGS.md, tables docs/trocr/)

Weights only, no forward pass beyond C0. Seven models of the Bern family (one base, `microsoft/trocr-base-handwritten`) plus six third-party fine-tunes (13 in all), tensors paired by canonical name, distance ‖a−b‖/mean(‖a‖,‖b‖) pooled over tensors (1.41 is not the null here: every model shares a root). The minimum spanning tree of the all-tensor distances recovers 4 of 6 documented parent–child edges; on the weight matrices alone every small fine-tune sits nearest its documented parent, but two margins are tiny (`kurrent` 0.228 to `base` against 0.230 to `v2`). The unused ViT pooler is an exact lineage marker (identical across `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `full-v1` and the third-party `hanse`; absent in `base`). Decoder moves 3–5 times more than the encoder after 20 epochs (0.11 against 0.51–0.56) and the tied embedding moves most (0.83–0.92). Displacement cosines: consecutive steps about 0 in encoder and decoder, children of one parent 0.1–0.24, same data from two origins about 0.08 (encoder). Third-party: `hanse` is an undocumented fine-tune of `xvi-xvii` (0.018); `swe` descends from `stage1`, not from `base`, and the models without embedding norm and with sinusoidal positions differ by config flags, not by recipe; `lamop` is unrelated to the family (0.86–0.90 from every model). Gate 1 passes. Limits: the weights-only pooling was chosen after seeing the data (exploratory); documented lineage is from the model cards.

## R16 — TrOCR stage 2: the seed null. Magnitude traces reproduce across seeds, move directions do not (15.11.2026; docs/TROCR_NULLS.md, docs/TROCR_NULLS_PREREG.md, docs/TROCR_NULLS_SEED42.md, tables docs/trocr/nulls_tables.md)

Preregistered before the first run (amendment 14.11. for a mis-based stop limit). Six reruns of the recorded recipe of `from-kurrent-v2` (parent `trocr-kurrent-XVI-XVII`, 3 epochs, the trainer's own script on the job's compiled data, card 0, nothing registered): seeds 1–5 and a repeat of seed 42. **Quality reproduces:** held-out `blb-eval-hand` CER 20.04 % mean over seeds 1–5 (sd 0.14, range 19.93–20.26; seed 42 repeat 20.27 %; recorded 19.74 %); 500 job-eval lines 8.63–9.00 %. **Magnitudes reproduce almost exactly:** encoder move 0.039 (sd 0.000), encoder hump index 2.197, token-row movement decile 9 / decile 0 of id 1.36, tied-embedding share of the squared move 0.85, encoder/decoder 0.14. **Directions do not:** two runs of the same data, recipe and parent share a move cosine of 0.36 (encoder), 0.19 (decoder without embedding), 0.43–0.97 (tied embedding, two groups); their weights lie 0.108 apart, farther than the parent (0.088). The same-seed repeat behaves like one more seed (0.39, 0.23, 0.50 to the recorded run). The stage-1 consecutive-step cosine of 0.07 is below the seed-pair range, but the ceiling is 0.36, not 1. Gate 2: magnitude and depth traces are usable as hypotheses about the recipe; direction readings cannot separate lineages finer than same recipe or not. Held-out CER differences below about 0.5 points are within retraining noise. Limits: one recipe, one parent, five different-seed samples; the preregistered survive wording does not fit magnitude traces (the matched step is itself a null sample), stated in the report; sources of non-determinism not separated.

## R17 — TrOCR stage 3: the planted-defect control. With the parent donor most defects are repaired by a lookup; without it one of eight (18–19.11.2026; docs/TROCR_STAGE3_PREREG.md, results docs/trocr/baseline_results.md, classes docs/trocr/stage3_classes.md, calibration docs/trocr/plant_calibration.json)

Preregistered before any programmer session (commit `8b559a6`; dated amendment 18.11. for the model id of the second family and the test count). Fourteen planted models of `from-kurrent-v2` (13 defects and one null corruption), calibrated on the practice set (BLB train side, 300 lines), never tuned on the held-out set. Held-out `blb-eval-hand` (2,272 lines, each frozen model read once, subject 19.74 % CER): the 12 defects scored all count (damage 1.54 to 13.42 points; p09, a zeroed layer norm, is catastrophic at +863). Keep rule: page-clustered paired interval below −0.005 (minimum effect 0.5 points, the seed null of R16); donor rule on the lines the donor did not train on. **Core set of eight:** p01 (six token rows zeroed), p03 (one cross-attention output zeroed), p05 (one encoder attention output zeroed), p06 (one encoder FFN output ×0.5), p07 (noise on one decoder layer), p08 (embedding-norm bias shift), p10 (20 % noise on every matrix), p12 (every encoder FFN output ×0.7). The architecture has no output bias (tied embedding), so the kraken blank-bias and output-row analogues do not exist.

**Held-out classes** (repaired: at least 80 % of the damage recovered and a clustered interval below 0). Non-LLM baselines on the 12 scored defects: rule list with donor (c) repairs 7 (p01, p02, p03, p04, p05, p08, p09) and none of the diffuse ones; scale search without donor (d) repairs only p13 (98 %) and partly p04, p05, p08. On the core eight: **(a) Sonnet 5.5 with donor 7 repaired** (p01 99 %, p03 97 %, p05 101 %, p06, p07, p08, p12 100 %; p10 no edit kept); **(a) without donor 1 repaired** (p06, 100 %, 7 passes; p03, p05, p07, p08, p10, p12 kept nothing; p01 crashed, counted not repaired, program not patched); **(b) blind Sonnet 5.5 (no passes) 7 repaired** (p10 not); **(a2) Fable 5.2 7 repaired and p10 partly (26 %)**; rule list (c) 4 (p01, p03, p05, p08), scale search (d) 0. No harmful keep and no crash in the arms with donor. **Gate 3 is met** on the preregistered counts (7 repaired with donor, 1 without, 0 harmful keeps). p10, diffuse noise on every matrix, is the one defect no arm repaired fully.

**Reading.** With the parent as donor, the programmers, the blind programmer and the Fable programmer mostly found the same one-tensor edit (restore the zeroed tensor or the scaled one); the blind arm with no pass at all matches the arm with ten passes, so the donor, not the pass budget, does the work. The programmer adds something over the baselines only on p06, p07 and p12 (rule list and scale search fail there), and there it also copies the parent block or inverts the scale. Without the donor the same programmer repaired one defect (a ×2 scale on one encoder FFN output); noise-type and zeroing defects were not repaired. Limits: one subject and one held-out hand; the donor makes most repairs a lookup; the no-donor rule was enforced by the brief and the pack (no parent diff), not by the sandbox; the practice set's `seen_by` for the parent donor is assumed empty (the parent's corpora are not audited, STAGE0 §8, item 4); p07 recovers over 100 % in the blind and Fable arms (final CER below the unplanted subject's), which is plausible for noise but unexplained. A first tally of arm (c) on the core eight read 3 and was corrected to 4 (19.11.2026).

## R18 — TrOCR stage 4: programmer edits on a healthy model do not beat plain interpolation toward the donor (20–24.11.2026; docs/TROCR_STAGE4_PREREG.md with amendments 1–6, docs/TROCR_STAGE4_PLAN.md, tables docs/trocr/stage4_grid.md, stage4_discovery.md, stage4_A_results.md, stage4_B_results.md, stage4_B_confirmation.md)

Preregistered before any session, grid cell, sham, margin or held-out read (19.11.2026; the seven open decisions of the plan confirmed by Tobias on 20.11.2026, "Defaults OK, go ahead"). Two subjects, each with its one donor: A = `from-kurrent-v2` (donor: its parent `trocr-kurrent-XVI-XVII`) and B = `trocr-kurrent` (donor: its documented ancestor `microsoft/trocr-base-handwritten`, English IAM). Practice set: 1,194 lines of the BLB train side (759 pages, 216 documents). Held-out `blb-eval-hand` split by object into a discovery half (17 objects, 1,147 lines) and a confirmation half (12 objects, 1,125 lines); **the discovery half was read for every model; the confirmation half was read once afterwards, for subject B only (see "Confirmation read" below), and not for subject A**. Practice CER: A 23.07 %, B 28.22 %, parent 21.90 %, IAM base 64.41 %; discovery CER: A 19.12 %, B 23.26 %. All reads CPU, pinned generation, nothing registered, no HF or trainer writes.

**Exchange grid on practice** (4 directions × 28 cells × 2 sizes = 224 passes, 6.7 h on card 0): large gains, all of them half-way moves toward the other model. A ← parent: whole decoder at half size −2.41 [−2.82, −2.01], whole encoder at half size −1.90; 13 of 56 cells with an interval below 0 and Δ ≤ −0.5. B ← IAM base: whole decoder at half size −1.80 [−2.39, −1.19]; at full size the same exchange destroys B (whole decoder +61.9, whole encoder +49.7). Half-size cells beat full-size ones almost everywhere. **Arm I** (one global interpolation toward the donor, α 0.1 to 0.9, best α chosen on practice) beats every cell: A α 0.6 discovery −4.09 [−5.13, −3.10] (19.12 to 15.03; best cell −2.17), B α 0.3 discovery −3.36 [−4.51, −2.23] (23.26 to 19.90; best cell −1.30). B turns harmful above α 0.7. Practice and discovery agree in direction and rank. A 0.3 mix of the English base improves B by 3.4 points, although that base reads this hand at 64 % CER; its corpus cannot be audited here.

**Programmer sessions** (practice only for the programmers; each frozen model read once on the discovery half; Δ in CER points, page-clustered 95 % interval; "vs arm I" positive = worse than arm I):

| arm | A discovery Δ (vs arm I) | B discovery Δ (vs arm I) |
|---|---|---|
| P1 Sonnet 5.5, donor | −3.73 [−4.75, −2.75] (+0.37 [+0.12, +0.60]); 30 % then 60 % pull of all non-embedding tensors toward the donor | 0, nothing kept |
| P2 Fable 5.2, donor | −1.67 [−2.21, −1.14] (+2.42); encoder 30 % toward the donor | −0.70 [−1.16, −0.29] (+2.66 [+1.71, +3.65]); cross-attention output ×1.15 in all 12 layers |
| P3 donor absent (Landlock), Sonnet and Fable | 0, nothing kept (both) | Fable: the same ×1.15 edit, same digest, −0.70; Sonnet: crashed on its own program bug, counted 0, not rerun |
| B blind (no passes), Sonnet | 0, no edit | 0, no edit |
| R replay by a fresh programmer | −3.09 [−3.89, −2.28] (+1.01); 25 % pull twice | −0.70, the same ×1.15 edit (same digest) |

**Gate 4 is not met on either subject: no frozen model beats arm I.** On A every kept edit is a pull toward the donor, that is a lookup or interpolation; the donor-free sessions of both models and the blind session found nothing. On B the only kept edit is a gain scaling of cross-attention output, found by three independent sessions (one with no access to the donor, one a fresh replay), which needs no donor; it is real (five shams of the same norm: best +0.10 and −0.11, all intervals contain 0; margins 0.5× −0.56, 2× −0.45) and is about a fifth of arm I's gain. Shams and margins were taken on each session's net change.

**Reading.** On a healthy model with a related donor the question the programmer has to answer is not "does an edit help" but "does it beat a one-parameter interpolation", and in these runs it does not. The only edit that does not depend on the donor is small. This repeats R17 (the donor does the work) on models that are not damaged, and bears on the kraken result of session 2, whose gain reduces to the same comparison. Deviations (amendments 1–5): 1,000 bootstrap resamples in the grid, not 2,000; the "norms" cell is all 124 layer-norm tensors; no same-recipe sanity pair (amendment 1 says no sibling model of that recipe is on the host; the six seed-rerun checkpoints of R16 do exist on the host, so that statement was wrong and stage 5 uses them); harness defect, the card gate read the watcher's state file inside the Landlock sandbox, so the first runs of p3s, p1, p2 and p3f on A died with a PermissionError before anything was measured (first results discarded, same programs rerun after moving the gate into the runner, which no longer sees a trainer job that starts on card 0 in the middle of a session); the variants of shams and margins were read on the discovery half (28 reads). Limits: two subjects, one hand, one donor each; the confirmation half has only 12 objects, so a "gain" would be hard to confirm in any case (it was read once for B, below); the class "below arm I" (a gain with a paired interval above 0 against arm I) is not defined in section 7 of the preregistration and is reported as such, not as "merge"; B's donor independence cannot be audited (the parents' corpora question of STAGE0 §8, item 4 is open); one blind agent printed pack files by shell and two agents had a grep of `wsp/session.py` refused, none saw held-out data.

**Confirmation read (subject B only, 25.–26.11.2026, CET; amendment 6, PR #59; docs/trocr/stage4_B_confirmation.md).** After the stage-4 results were merged the seed-null session read the confirmation half once for B (12 objects, 1,125 lines; the approval for this read was given in that session and is not recorded in this one). Subject 29.83 % CER; the `cross.o` ×1.15 edit −2.06 [−2.86, −1.31], best of five shams +0.07, so the edit confirms; arm I (α 0.3) −7.36 [−9.19, −5.47], so arm I replicates on unread lines; edit minus arm I +5.30 [+3.81, +6.71], still "below arm I", Gate 4 still not met. Both gains are larger here than on discovery (−0.70 and −3.36) while the subject's baseline is higher (29.83 against 23.26), which shows how much objects differ. Subject A's confirmation half is unread. The class "below arm I" is still undefined in section 7 and left to Tobias.
