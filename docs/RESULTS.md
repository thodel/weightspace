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
