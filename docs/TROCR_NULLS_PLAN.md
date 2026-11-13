# TrOCR stage 2, nulls: plan (draft, nothing submitted)

Written 12.11.2026 (CET). Status: draft for Tobias's OK. No trainer job, no GPU use, no registry or hub write has happened.

## What stage 2 has to answer

Stage 1 found traces (encoder hump, tied embedding moves most, consecutive steps near-orthogonal, token-row trend, encoder/decoder ratio). Gate 2: only traces that survive a null become hypotheses in the sessions. A null must say how much of each reading appears between runs that differ only by chance.

## (a) Same data, different parent: done, no cost

`from-kurrent-v2` vs `v2` (parents `xvi-xvii` and `base`, same 53,812 samples, 3 epochs): all tensors 0.347, encoder 0.163 (against 0.039 to its own parent), decoder much larger. Displacement cosine of the two moves: about 0.08 encoder, 0.03 decoder. Source: `docs/trocr/readings_tables.md` T7 and `displacement.json`. This null is **confounded**: the parents differ, so it bounds a different quantity than a seed null.

## (b) Five seeds of the 3-epoch recipe from one parent: needs the trainer

- **Parent:** `dh-unibe/trocr-kurrent-XVI-XVII` (public; the clean parent of `from-kurrent-v2`). The five seeded children are compared with `from-kurrent-v2`, with each other (10 pairs) and with the parent.
- **Data and recipe:** the 20 public datasets, 53,812 train samples, lr 5e-5, cosine, bf16, 3 epochs, per the record of job `20261026T064540Z-...`. First step, read only: fetch that job's request body and `training_args.bin` from the trainer and compare with `docs/MODELS_TROCR.md`. Any difference is stated, not tuned away.
- **Seeds:** 1 to 5 (data order, dropout, augmentation; the split stays fixed). Seed 0 is not used because the recorded run's seed is unknown. Stated as a limit.
- **Route:** the trainer, as decided (answer 2). Need a spec `training/specs/trocr-seed-null-s{1..5}.json` with distinct `model_id`s and `revision` pinned. Weights stay on asteraix, are never registered, exported or uploaded (as in `docs/SEED_NULL.md`). To be checked: whether the trainer's route registers a model by default; if it does, this needs a switch or the kraken-style direct run instead.
- **Cost:** 1 to 1.5 GPU hours each, about 0.44 kWh each, so 5 to 7.5 GPU hours and about 2.2 kWh in all. Sequential on card 1 (the trainer's), about one day wall time; nothing on card 0.
- **Alternative to decide:** a direct run in the TrOCR venv (Seq2SeqTrainer) on card 0 with at least 12 GB free, as the kraken seed null did. Cheaper to control, but not the trainer's recipe byte for byte. Recommendation: trainer, unless it registers models.
- **Disk:** about 1.3 GB per child, 6.5 GB in `~/atr-cache/trocr/seed-null/`.
- **Held-out:** each child read once on `blb-eval-hand` (C0 on CPU or card 0, same harness as stage 0), to see whether the CER spread is above the noise of the metric.

## (c) Recompute the traces against the null

For each of the five documented traces compute the same reading on the 10 seed pairs and on the 5 child-to-parent pairs; a trace survives if the documented value lies outside the seed range by a stated margin (proposed: beyond the largest seed value and beyond range + 2x its spread).

| trace | reading | what the null must show |
|---|---|---|
| encoder hump over depth (T5) | per-layer relative L2, shape | same shape appears between seeds, or not |
| token-row trend by id decile (T9) | decile means of row movement | trend present in seed pairs or absent; the frequency proxy needs token counts of the training text, tokenised with the shared tokenizer |
| tied embedding dominates | T4 part ratio | seed ratio |
| encoder/decoder ratio | 0.11 vs 0.51 to 0.56 | seed ratio |
| orthogonal consecutive steps | displacement cosines | cosine between two seed moves from the same parent (expected well above the 0.1 to 0.24 seen between children of one parent only if the move is mostly shared) |

Code: reuse `wsp/trocr_family.py` (`read` with a models dict, `dist_matrix`) and `scripts/trocr_displacement.py`; one new script `scripts/trocr_nulls_report.py` writes `docs/trocr/nulls_tables.md`; report `docs/TROCR_NULLS.md`. Preregistered before the runs: the survive criterion above, the seed list, and the rule that the report states every trace whether it survives or not.

## Open points for Tobias

1. May the five runs go to the trainer, and when (card 1 is its own)? Or the direct route on card 0?
2. Parent `trocr-kurrent-XVI-XVII` for the seed pairs (recommended, matches `from-kurrent-v2`); the second subject `trocr-kurrent` (parent `base`, 20 epochs, 293 k lines) would be about 100 times costlier per seed and is not proposed here. The seed null for it would have to come from fewer seeds or a shorter recipe.
3. Is a seed null on one recipe (3 epochs, 54 k samples) accepted as the null for traces of the 20-epoch steps too? It is not an exact null for those; the report would say so.

## Before any submit

Read-only: pull the original job's request body, pin dataset revisions with `train_spec` action `request`, check `server_status` for free disk and the card state. Then ask for the go-ahead with the exact body.

## Update 12.11.2026: what the original job record shows (read only)

Found at `/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs/20261026T064540Z-trocr-german-kurrent-xvi-xix-from-kurrent-v2/job.json`. The trainer has a `trocr` engine, but the **specs of `train_spec` are in the gateway repo** (`/home/dh/agentic_historian`, `training/specs/`, three kraken/VLM specs only), so the five requests need to be added there by its owner or through a commit I am not authorised to make. A draft request body (revisions of all 20 datasets already pinned) is `docs/trocr/nulls_request/request_template.json`.

- **Recipe, as run:** `base_model` dh-unibe/trocr-kurrent-XVI-XVII, 3 epochs, batch 8, no accumulation, lr 5e-5 cosine, warmup 0.1, weight decay 0, max grad norm 1.0, adamw_torch, gradient checkpointing, bf16, **seed 42**, beam 4, length penalty 1.0, no_repeat_ngram 3, max_new_tokens 256, 500 eval samples, trainer commit 50f7c4fb, transformers 4.57.6. 53,008 train samples, 19,878 steps. Recorded CER 8.89 %.
- **Time:** wall 3 h 10 min: prepare 12 min, **compile 82 min** (CPU, same for every seed), train 86 min, test 9 min. Energy of the train stage 0.40 kWh (both cards counted), so five runs are about 2 kWh as estimated, but 5 x 82 min of compile is avoidable if the compiled data can be reused (to ask the trainer's owner).
- **Seed 42 was the recorded run's seed**, so my earlier remark that its seed is unknown was wrong. Proposal: seeds 1 to 5 plus a **seed-42 repeat** as a reproduction check (does the recipe give back `from-kurrent-v2` bit for bit, or only up to noise?). That repeat is also the cleanest null for "same recipe, same seed".
- **Side effects to switch off:** the original job was **published to the hub and registered** (`published: https://huggingface.co/dh-unibe/...`, registry yaml). The seed runs must do neither; whether the trainer has a switch is for its owner to say. The original used card 0 (`cuda:0`); card 1 is the trainer's, so the card is to be set in the request.
