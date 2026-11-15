# TrOCR stage 2: nulls (seed null and same-data pair)

Written 15.11.2026 (CET). Preregistration: `docs/TROCR_NULLS_PREREG.md` (with the amendment of 14.11.). Code: `scripts/trocr_seed_run.sh`, `trocr_seed_chain.sh`, `trocr_seed_check.py`, `trocr_nulls_report.py`, `trocr_c0.py`. Tables (every number below is in them): `docs/trocr/nulls_tables.md` (N1 to N5), `nulls_summary.json`, `nulls_readings.json`; per-child C0 in `runs/trocr-seed-null/`. Weights stay in `~/atr-cache/trocr/seed-null/` (never registered, exported or uploaded).

## 1. What was run

Six extra runs of the recorded recipe of `from-kurrent-v2` (job `20261026T064540Z`: parent `dh-unibe/trocr-kurrent-XVI-XVII`, 3 epochs, batch 8, lr 5e-5 cosine, warmup 0.1, bf16, gradient checkpointing, 20,181 steps): seeds 1 to 5 and a repeat of the recorded seed 42. The trainer's own training script (identical to the job's commit) was run directly on the job's compiled manifests (53,812 train and 4,289 validation lines, read-only from the share), on card 0, one run at a time, 2.0 to 2.2 hours each (a kraken training shared the card from the second run on). Not run: the trainer's job mechanism (nothing registered or published). **Deviations from the recorded job:** card, hardware state and moment (GPU arithmetic is not made deterministic), the job's 500-line test replaced by the same 500 lines scored with `wsp.metrics`, checkpoints deleted after each run. Same-data pair (a): the stage-1 pair `from-kurrent-v2`–`v2` (T7 of `readings_tables.md`; parents differ, so it is a different, confounded null).

## 2. Quality is reproduced, the weights are not

| run | final eval loss | CER, job's 500 lines | CER `blb-eval-hand` (2,272 lines), WER, collapsed |
|---|---:|---:|---|
| recorded `from-kurrent-v2` | 0.5649 | 8.82 % (job's scorer 8.89 %) | 19.74 %, 60.4 %, 9 |
| s42 (same seed) | 0.5674 | 8.66 % | 20.27 %, 61.2 %, 14 |
| s1 | 0.5682 | 9.00 % | 20.09 %, 60.9 %, 15 |
| s2 | 0.5679 | 8.63 % | 20.26 %, 61.0 %, 11 |
| s3 | 0.5632 | 8.86 % | 19.93 %, 60.9 %, 11 |
| s4 | 0.5607 | 8.79 % | 19.93 %, 60.7 %, 12 |
| s5 | 0.5683 | 8.63 % | 20.00 %, 60.8 %, 11 |
| seeds 1 to 5 | mean 0.5657 | mean 8.78 %, sd 0.16, range 8.63 to 9.00 | mean 20.04 %, sd 0.14, range 19.93 to 20.26 |

Seed-to-seed spread of the held-out CER is about 0.3 points in range; the recorded run (evaluated with the same harness at stage 0) is 0.2 to 0.5 points better than every re-run on `blb-eval-hand`, a little more than the seed spread (one recorded run, and my re-runs differ from it in hardware state; not explained). Any effect of a weight edit on this set smaller than about 0.5 points cannot be told from retraining noise.

## 3. The readings against the null (N1 to N3)

**Magnitudes reproduce almost exactly.** Encoder move 0.039 (seed sd 0.000), decoder without embedding 0.111 (sd 0.000), encoder hump H = mean(layers 3-6)/mean(layers 10-11) 2.197 (range 2.192 to 2.204), ratio of token-row movement decile 9 to decile 0 of id 1.36 (range 1.35 to 1.38), share of the squared move in the tied embedding 0.85 (0.84 to 0.87), encoder/decoder 0.14 (0.13 to 0.14). Only the tied embedding itself varies (0.49 to 0.53). Weight-matrix distance of every re-run to the parent: 0.088 each, the same as the recorded run.

**Directions do not.** Two runs from the same parent with the same data and recipe differ in weights by 0.108 (weight matrices; all tensors 0.08 to 0.18), farther than either is from the parent (0.088). The cosine between their moves: encoder 0.36 (0.35 to 0.36 over 10 seed pairs), decoder without embedding 0.19 (0.17 to 0.20), tied embedding 0.75 (0.43 to 0.97). The same-seed repeat is no closer: 0.39, 0.23, 0.50 to the recorded run, and 0.36, 0.19 to the other seeds, i.e. it behaves like one more seed.

**The embedding move has two modes.** s1, s3, s4, s5 and the repeat agree with each other (cosine up to 0.97), `from-kurrent-v2` and s2 form a second group (0.94 between them); across groups the cosine is about 0.45 to 0.5. This is the only structure in the null beyond the tight magnitudes. Not explained here (candidates not tested: data order, handling of the three special tokens).

## 4. The five traces (N5) and Gate 2

Preregistered criterion: a documented value survives if outside the different-seed range and at least 2 sd from its mean. **That wording does not fit the magnitude traces, and I say so rather than bend it:** the matched step (`from-kurrent-v2`) is itself a run of the null's recipe, so its values lie inside the range by construction. The informative reading is the other one: the null shows how much the readings depend on the seed.

| trace of stage 1 | what the null shows | use as a hypothesis |
|---|---|---|
| encoder hump over depth | reproduced by every seed (H 2.197, range 0.012) | robust property of the recipe; says nothing about an individual run |
| rare token rows move most (as id proxy) | reproduced (ratio 1.36, range 1.35 to 1.38) | robust; the frequency version still needs token counts of the training text |
| tied embedding dominates | reproduced (0.85 of the squared move) | robust; the tied embedding is the largest and the only widely varying part |
| encoder moves far less than decoder | reproduced (0.14) | robust |
| consecutive steps near-orthogonal | documented cosine 0.07 (encoder) is **below** the seed-pair range 0.35 to 0.36 | survives, but read against a ceiling of about 0.36, not against 1: two runs on identical data share only about a third of the move |

For the other documented steps (N4) the "survives" verdicts only restate that those steps differ in recipe, parent or data (`v2`, the 20-epoch steps, `full-v1`); there is no null for those recipes, so they are not Gate-2 evidence. **Gate 2 outcome:** the readings of magnitude and depth profile can be used as hypotheses about what the recipe does, with the seed noise stated (1 % to 5 % of their size, except the tied embedding, about 4 %); the readings of direction cannot discriminate lineages finer than "same recipe or not".

## 5. What this means for the later stages

- **Lineage by weights** (stage 1, Gate 1) stays valid for model families that differ in data or recipe; for models trained by one recipe from one parent, a sibling is farther (0.108) than the parent (0.088) in weight matrices, so a nearest neighbour alone is not a parent proof.
- **Minimum effects** for sessions: held-out CER differences below about 0.5 points are within retraining noise (this is also the scale of the difference between the recorded run and its re-runs).
- **Same-seed does not buy reproducibility** on this hardware and recipe; a "retrain comparator" (stage 5) must be five-seed.
- A weight edit that changes the encoder by less than the seed spread of its direction (the encoder move shares only 0.36 cosine) cannot be called "the" direction of fine-tuning.

## 6. Sources of non-determinism (known, not separated)

bf16 arithmetic and non-associative GPU reductions; gradient checkpointing (recomputation); eight data-loader workers; dropout streams that depend on operation order; a different card, driver state and a second job on the card (speed, not values, in principle); kernel selection. The same-seed repeat shows they together suffice to change the move direction nearly as much as a new seed. None was tested separately (a deterministic-algorithms run with `CUBLAS_WORKSPACE_CONFIG` and fp32 would be the first test, at about three times the cost).

## 7. Limits

One recipe (3 epochs, 54 k samples) from one parent; five different-seed samples give a range, not a distribution (sd from n = 5); the 20-epoch steps have no null of their own; the verdict wording problem of section 4; the held-out comparison with the recorded run involves one recorded sample; the embedding modes rest on 7 runs.
