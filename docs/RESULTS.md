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

### Note to R2 (07.11.2026)

Re-aggregated on `blb-eval-hand` (the handwriting held-out set chosen by data properties, docs/PHASE5_REPORT.md; a re-aggregation of the one held-out reading, not a second evaluation): before 9.553 %, after 9.553 %, Δ −0.001 points [−0.055, +0.053]. The conclusion of R2 is unchanged.

### Note to R2 and R3 (07.11.2026, later): page-clustered intervals

Recomputed with the two-stage cluster bootstrap (`wsp.metrics.bootstrap_delta(..., clusters=)`, pages then lines; `docs/controls/cluster_intervals.md`). Line-level numbers above are unchanged. R2 held-out `blb-eval`: −0.010 [−0.069, +0.049]; on `blb-eval-hand` −0.001 [−0.082, +0.084]. R3 practice −0.425 [−0.926, +0.037] (now contains 0); held-out +0.151 [−0.285, +0.597]; Hertwig −0.300 [−0.825, +0.213]; Schlussredaktion +0.510 [−0.043, +1.076] (the "significant loss" is not supported at page level). Conclusions "not a gain" stand.

## R4 — planted defects, the positive control (07.11.2026; docs/CONTROLS_REPORT.md §2, §9b)

Eleven corruptions of `kraken-german-xix-blb-v1` (practice `blb-train-300`, held-out `blb-eval-hand`, uncorrupted 9.55 %), four arms each: programmer with ten passes (a), blind programmer (b), fixed hand-written repair list (c), coordinate search over output biases (d). For the six localised corruptions that cost at least a point (zero / negate / swap output rows, blank bias +2, last BiLSTM ×0.5, zeroed conv block) arms (a), (b), (c) each return the held-out CER to within 0.22 points of the uncorrupted model (19 of 19 repaired cells); (d) repairs only the bias defect. The programmer is not better than (c) or (b). With an unrelated donor the zeroed row and block are not repaired; a scale defect still is. A diffuse defect (noise on every weight) is repaired 83 % by (a) only (interpolation toward the parent); noise on one block at 25 % of its norm (+0.49 points) is repaired by no arm. Run folders `runs/20261107-planted-*`; aggregates `docs/controls/planted.json`, `planted_table.md`.

## R5 — validation-based practice sets (07.11.2026; docs/CONTROLS_REPORT.md §3, §6, §9c)

Practice sets that are not the subject's training lines: `bern-val` (the trainer's 17 validation pages of `combined-v1`, 513 lines) and `blb-val-300` (carved from seven eval objects; the trainer validated this model on the eval pages, so no validation side exists), audited by `wsp.audit validation`. `xix-blb-v1`: 0 edits kept in 7 attempts (Sonnet ×3, Opus, Haiku, Fable 5.1, replay). `combined-v1`: all attempts that kept anything interpolated toward the clean sibling; a Fable 5.1 attempt (α = 0.5, practice −2.3 points) gives held-out −0.02, the practice optimum that does not transfer; the common edit (blank −0.3, non-output tensors α = 0.25 toward `schlussredaktion-v1`) gives held-out −0.39 [−0.69, −0.08] lines, [−0.84, +0.04] pages; 0 of 20 norm-matched shams reach it and none is kept by the rule; the α sweep shows a narrow window (α ≈ 0.1–0.25) and a practice optimum at α ≈ 0.75 that is worse on held-out; the uniform average gives −0.02. A model-merge observation, not a repair. The donor rule (PROTOCOL §2, proposed) refuses the 06.11. transplant from `hertwig-v2`.

## R6 — additivity and margins (07.11.2026; docs/CONTROLS_REPORT.md §7, §8)

Kurrent pair `xix-v1` → `xix-blb-v1`: no α from 0 to 1.5 on all blocks gives a gain on either side; the joint cost exceeds the sum of single-block costs on held-out lines (by 0.4 to 3.8 points) while block pairs at α = 0.5 are close to additive (correlation 0.97–0.98). Margins at deleted space/`e`/`n` frames: median blank margin 3.2 nats, 18 % of deletions have no blank frame between neighbours, the three blank-bias states of session 1 reach about 3 %, 12 % and 22 % of them at the price of insertions. `docs/controls/global_interp.json`, `margins.json`.
