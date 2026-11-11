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

**Replicate with Claude Fable 5.1 as programmer (11.11.2026; `runs/20261111-bern-combined-v1-f1-b10-1`, docs/KRAKEN_S2_FABLE.md).** The subject, practice set, held-out set, budget and keep rule are the same; the programmer was a Fable subagent that did not see session 2. 9 passes, 8 edits, 3 kept: blank bias −0.20 twice, then every tensor 12.5 % toward the parent. The same direction as Opus (toward `hertwig-v2`; the sibling was refused in both), as a small whole-network step rather than an output-layer transplant. Practice 0.1456 → 0.1357. Held-out 0.1933 → **0.1875**, Δ −0.57, page-clustered [−1.15, −0.05]; Hertwig −1.32 [−2.36, −0.60], Schlussredaktion +0.02. Against Opus's frozen model, Δ −0.73 [−1.19, −0.35]. C5 (5 norm-matched shams): held-out −0.12 to +0.02 and none kept. The gain is the interpolation's (−0.53 alone); the blank steps alone give +0.04 on held-out. It matches C4 arm A (R11) in size and place. A first programmer attempt was discarded before any pass because the pre-registration it read named session 2's kept edit. At the user's request the frozen model is registered on the share as `kraken-bern-combined-wsp-v1` and was read through the serving path (14 `bern-eval` pages, both models in one run, then disabled again): character-weighted −2.16 [−3.32, −1.05], 11 of 14 pages better, but the gain lies on Schlussredaktion (−3.84 [−4.89, −2.91], 8 of 8) and not on Hertwig (−0.04 [−0.68, +0.56]), the reverse of the line-level split; the gateway segments the pages itself (KRAKEN_S2_FABLE §6).

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
