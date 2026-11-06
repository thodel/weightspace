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
