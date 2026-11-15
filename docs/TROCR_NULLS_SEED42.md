# TrOCR stage 2b: the seed-42 repeat (reported before the other runs, as preregistered)

14.11.2026 (CET). Run `scripts/trocr_seed_run.sh 42`, 08:57 to 10:58 CET, card 0 only, the trainer's script on the job's compiled data, 20,181 steps, 7,207 s of training. Check: `scripts/trocr_seed_check.py`, numbers in `docs/trocr/seed42_check.json`. The weights stay in `~/atr-cache/trocr/seed-null/s42/` (not registered, not uploaded). Checked first: the hub file of `from-kurrent-v2` and the trainer's own `trained/` model have the same sha256, so the reference is the model the job produced.

| | recorded `from-kurrent-v2` | seed-42 repeat |
|---|---:|---:|
| CER on the job's 500 eval lines (wsp.metrics, NFD) | 8.82 % (job's scorer: 8.89 %) | 8.66 % |
| final eval loss (4,289 lines) | 0.5649 | 0.5674 |
| train loss | 0.598 | 0.587 |
| distance to `from-kurrent-v2`, all tensors / weight matrices | | 0.162 / 0.105 |
| distance to the parent, all / weights | 0.162 / 0.088 | 0.149 / 0.088 |
| size of the move, encoder / decoder (excl. embedding) / embedding | 35.17 / 53.92 / 160.5 | 35.17 / 54.18 / 144.9 |
| cosine between the two moves, all / encoder / decoder / embedding | | 0.46 / 0.39 / 0.23 / 0.50 |

**Reading.** The recipe reproduces the quality (CER within 0.16 points, eval loss within 0.003) but **not the weights**: the repeat lies as far from the recorded model as that model lies from its parent, and the two moves share a cosine of only 0.46 (encoder 0.39, decoder 0.23). The size of the move is reproduced almost exactly (encoder 35.17 against 35.17). So with identical data, identical recipe and the same nominal seed, about two thirds of the squared displacement is not shared. Sources not separated here: GPU arithmetic (bf16, other card, atomics), data-loader worker order, a different hardware state. This is the first seed-null datum and bears directly on Gate 2: a trace that is a property of the movement direction (consecutive steps near-orthogonal, encoder hump shape) must be read against this, not against zero.

**Against the preregistered limit.** CER within 0.6 points: met. Distance to `from-kurrent-v2` at most 0.044: **not met** (0.162; also not met on weights only, 0.105). Note that the limit was set from the weights-only move (0.088) and applied to a measure that includes embeddings, where the move is 0.162; half of either is still far below what was observed. Per the preregistration the other runs wait for a decision. Not started: seeds 1 to 5.
