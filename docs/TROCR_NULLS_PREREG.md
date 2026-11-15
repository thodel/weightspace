# TrOCR stage 2b: preregistration of the seed null (written before any run)

Written 13.11.2026 (CET). Route as instructed by the planning session (direct, card 0; not the trainer's job mechanism). Replaces the trainer route of `docs/TROCR_NULLS_PLAN.md`.

## Route and deviations from the recorded job

- **Code:** `engines/trocr_train_svc/train_trocr.py` of `/home/tobias/Repo/training-atr-models`, the same script the job `20261026T064540Z` ran. `git diff 50f7c4fb HEAD -- engines/trocr_train_svc` is empty, so the code is identical to the job's commit. Same venv (`.venvs/trocr-train`, transformers 4.57.6, torch 2.8.0+cu128). Run as a plain command, with the arguments of the job's `train.log`, plus `--seed S --device cuda:0 --output-dir ~/atr-cache/trocr/seed-null/sS`.
- **Data:** the **compiled manifests of the original job**, read only from the research share (`train.jsonl` 53,812 lines, `val.jsonl` 4,289, crops, 20 datasets at their pinned revisions). No new compile, so the data, split and crops are bit-identical for every seed.
- **Not used:** the job runner's prepare, compile, test, register and publish stages. So nothing is registered, published, exported or uploaded; weights stay in `~/atr-cache/trocr/seed-null/` and are never leaving it.
- **Deviations to state:** (i) card 0 on a shared host instead of the job's card, (ii) a different moment and driver state, so GPU non-determinism is not removed (bf16, atomics), (iii) the job's own 500-sample test is replaced by my C0 on the same 500 lines (`eval_report.raw.jsonl` of the job), (iv) the number of dataloader workers and the order of intermediate checkpoints are unchanged, (v) the checkpoint dir is deleted after each run (only the final model is kept).
- **Recipe (from the job):** base `dh-unibe/trocr-kurrent-XVI-XVII`, 3 epochs, batch 8, no accumulation, lr 5e-5 cosine, warmup 0.1, weight decay 0, max grad norm 1.0, adamw_torch, gradient checkpointing, bf16, workers 8, beam 4, length penalty 1.0, no_repeat_ngram 3, max_new_tokens 256. 20,181 steps.

## Runs and order

1. **Seed 42 repeat** (the recorded run's seed), the reproduction check against `from-kurrent-v2`. **Stop and report before the other runs** if either: its CER on the 500 job-eval lines differs from the recorded 8.89 % by more than 0.6 points (my scorer and the job's can differ a little), or its distance to `from-kurrent-v2` (all tensors the forward uses, as in stage 1) exceeds 0.044, half of the whole move of that step (0.088).
2. **Seeds 1 to 5**, one at a time, card 0 only.
3. C0 on `blb-eval-hand` for every child, same harness and generation settings as stage 0 (read once per frozen model).

## Stops

Stop and report if: disk under 3 GB free on my volume; card 0 is not free (`wsp.card.ready`: at least 12 GB free and no trainer job waiting; card 1 is never touched); the manifests or crops cannot be read; a run fails or its loss is not finite.

## What is read (pairs: 6 children, the parent `trocr-kurrent-XVI-XVII`, `from-kurrent-v2`)

The stage-1 readings (`wsp.trocr_family`) on all pairs among the 6 children, child to parent, child to `from-kurrent-v2`: pooled distance (all tensors, weights only), per part (T4), per layer (T5), token-row decile movement (T9), identical-value share, pooler identity, displacement cosines between the moves (parent to child). Plus CER on `blb-eval-hand` per child and on the 500 job-eval lines.

## Survive criterion (fixed now)

A documented trace of stage 1 **survives the null** if its documented value lies outside the seed range, i.e. beyond both the largest and the smallest seed value of the same reading and at least two seed standard deviations away from the seed mean. A trace inside the range is reported as **not distinguishable from seed noise**; one outside only by less than two standard deviations is reported as **borderline**. Reported for every trace, whichever way it goes.

Traces and the seed reading that corresponds to each:

| trace | seed reading |
|---|---|
| encoder hump over depth | per-layer profile of the seed-to-seed distance and of child-to-parent distance |
| rare token rows move most | decile means of row movement, seed pairs and child-to-parent |
| tied embedding dominates | embedding share of the move |
| encoder 0.11 vs decoder 0.51 to 0.56 | the same ratio, child-to-parent |
| consecutive steps near orthogonal | cosine between two seed moves from the same parent; cosine between a seed move and the documented `from-kurrent-v2` move |

Limits stated in advance: the null is for the 3-epoch recipe from one parent; it is not an exact null for the 20-epoch steps (`base`→`kurrent`, `kurrent`→`xvi-xvii`); five seeds give a range, not a distribution; the seed fixes the batch order and dropout but not the GPU's arithmetic.

## Cost

About 86 min of training per run plus evaluation, six runs in sequence, about 10 hours of card 0 in all. Card 0 is held at 44.5 GB of 46 GB by a vLLM server (2 workers, one on each card, up for 1 day 16 h) at the time of writing, so no run can start until it is released.

## Amendment 14.11.2026 (CET), after the seed-42 repeat and before seed 1

Added after the first run; the text above is unchanged.

1. **The stop limit was mis-based.** The distance limit of 0.044 for the seed-42 repeat was taken as half of 0.088, which is the *weight-matrix* distance of the step `xvi-xvii`→`from-kurrent-v2`, but it was applied to the *all-tensor* distance (which includes embeddings and norms), where that step is 0.162. The two are not comparable. The CER limit (0.6 points) was sound and was met (8.66 % against 8.82 %).
2. **The stop was honoured.** The repeat finished at 10:58 CET, breached the distance limit as measured (0.162 all tensors, 0.105 weights), and the other runs were held until the planning session decided (`docs/TROCR_NULLS_SEED42.md`).
3. **The seed-42 repeat counts as a sixth null sample, labelled "same seed".** It shares data, recipe and nominal seed with the recorded `from-kurrent-v2`; seeds 1 to 5 differ in seed. The report shows the same-seed pair separately and does not pool it with the five different-seed pairs when it computes the seed range and standard deviation; pooled values are given beside it.
4. **The survive criterion is unchanged** (outside the seed range and at least two seed standard deviations from the seed mean). The traces are compared with the different-seed range and, in a second column, with the range including the same-seed repeat.
5. **Seeds 1 to 5 run on, one at a time on card 0, under the same rules**; they stop only for a real failure (card 0 taken, disk under 3 GB, crash, non-finite loss).
6. **Sources of non-determinism, known and not tested here:** bf16 arithmetic and non-associative GPU reductions; gradient checkpointing (recomputation in the backward pass); data-loader workers (8, with their own random state for any augmentation); dropout (seeded by `set_seed`, but its stream depends on the order of operations); a different card and driver state from the recorded run; library and kernel selection that can differ between runs. None of these is separated by the repeat. Seeds 1 to 5 give the spread from all of them at once.
