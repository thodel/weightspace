# Plan — weight-space programming for ATR models

Written 06.11.2026 (CET), revised the same day. Status: proposal, nothing run yet.
**Where things run:** h1 (this box) holds the plan and the code; every run,
every forward pass and every weight file lives on **asteraix**
([docs/ASTERAIX_RUNBOOK.md](docs/ASTERAIX_RUNBOOK.md)).
**Scope:** the kraken models are the small pilot; the harness is built so the
same sessions can be run on QLoRA adapters of the Qwen-class page models
(Qwen3.5-4B, olmOCR-2-7B) afterwards. Everything is logged to the standard of
a reproducible paper ([docs/PROTOCOL.md](docs/PROTOCOL.md),
[docs/PAPER.md](docs/PAPER.md)).

## 0. What the method is, and what it is not

The Takehi et al. post (02.11.2026) describes an LLM that **reads a network's weights and
edits them in code, without gradient descent**, inside a sandbox with a small
budget of forward passes on practice data. Loop:

1. load the weights into a Python sandbox (no training data, no optimizer);
2. read: hypothesise what each component does, from the tensors;
3. spend a forward pass on practice inputs when a hypothesis needs checking
   (budget 0 / 10 / 25 / 50 passes per session);
4. write an edit in code;
5. keep the edit only if it wins on the practice set;
6. freeze, evaluate once on held-out data the programmer never saw.

Its best result (Qwen3.5-0.8B, MATH-500 7.9 → 48.1) came from one identified
defect: a looping direction written by two MLP layers, subtracted from 64 output
columns each. The authors' own caveats: edits **remove obstacles, they do not
add ability** (the Gemma control could not be improved); results are early; only
MLPs, one small transformer and two LLMs were tried — no CNN, no LSTM, no CTC,
no adapters.

It is **not** a merging method. The post names weight averaging as the merging
baseline it hopes to beat and lists "true model merging" as future work. Since
we have families of fine-tunes from one parent, merging and cross-model reading
are a natural second track here, but they are our extension, not the paper's.

Honest expectation: on a healthy fine-tune, a point or two of CER at most; a
real gain only where a model carries an identifiable defect (CTC collapse or
repetition, a dead or skewed output row, a layer the fine-tune wrecked, an
adapter direction that produces loops). The plan is built to find such defects
first and to measure everything against the cheapest alternatives (averaging,
a short retrain).

## 1. Material

### Pilot: seven in-house kraken models ([docs/MODELS.md](docs/MODELS.md))

- **Family 1, Bern 1528:** `bern-hertwig-v1` → `v2` → {`schlussredaktion-v1`,
  `combined-v1`}; parent `cursive_inzigkofen` (CATMuS Medieval lineage).
  `combined-v1` (28.8 %) is worse than its parent and sibling (16.7 %): the
  first thing to read.
- **Family 2, Kurrent XIX:** `german-xix-v1` (11.2 %) → `german-xix-blb-v1`
  (8.7 %). The production pair; ground truth and baseline runs exist.
- **Family 3, kraken+:** three from-scratch runs; the `Cr255,1,85` layer with
  11 M gradient-free weights is a known weight-space fact to start from.

Architecture: VGSL conv stack → 3 × BiLSTM → CTC over ~100–150 classes; ~4 M
live parameters; 16 MB CoreML files; one forward pass over a few hundred
lines takes seconds on CPU.

### Scale-up: QLoRA adapters of the page models

The trainer (training-atr-models) produces peft adapters (`adapter_model.safetensors`,
rank 16, language tower and, for Qwen3-VL, the vision tower) for
`Qwen3.5-4B`, `Qwen3-VL-4B/8B`, `olmOCR-2-7B` and Apertus. An adapter is a
few tens of MB of `lora_A` / `lora_B` pairs per target module — the same order
of size as a kraken model, and the natural unit of weight-space reading:
`ΔW = B·A` per module is low-rank by construction, so its directions are
enumerable, and a defect direction (looping, truncation, the token-limit hits
the trainer already counts) is a candidate for exactly the Exp-8 edit of Takehi et al..
The forward pass is then a vLLM batch on asteraix's free card and costs
minutes, which is why the budget is defined in passes, not seconds.

## 2. Two tracks

### Track A — WSP proper: read, diagnose, patch, one model at a time

Targets, in order:

1. `kraken-bern-combined-v1` vs its parent `hertwig-v2` — a fine-tune that got
   worse. Diff the two; the damage should be localised (output rows for the
   characters the second project adds, or the last LSTM). Edits: revert rows
   or blocks to the parent, rescale, interpolate per layer.
   *Re-framed 06.11.2026:* combined-v1 was trained on the 25.10 export, in
   which 44 of 50 Hertwig pages carry the wrong image. The question is now
   whether a weight-space edit can undo training on label noise, against a
   retrain on the corrected data as C4 (docs/HERTWIG_REVISIONS.md).
2. `kraken-bern-hertwig-v2` — 0.37 line-CER with known failure classes.
   Look for a blank-dominant direction in the output layer and the last LSTM
   (the CTC analogue of Qwen's loop direction).
3. `kraken-german-xix-blb-v1` — the production model. Expect little; do it
   because this is where an improvement would matter and the 276 GT pages give
   the best measurement.
4. kraken+ run: the lossless `Cr255 → Cr1` rewrite, then the 85-channel
   bottleneck (widen by weight surgery, see whether it needs training).
   *Dropped 06.11.2026* except for the rewrite (done, docs/RESULTS.md R1): the
   archived checkpoints' training and validation arrows were not kept, so the
   models cannot be measured on their own data (docs/MODELS.md, family 3).
5. **First adapter target:** the Qwen3.5-4B Kurrent page adapter, with
   truncation / repetition on long pages as the defect to look for; the
   trainer's `truncated_cer` and `finish_reason` counts are the symptom.

Edit classes, from safest to riskiest, each with a predicted effect written
down **before** the forward pass:

| class | what | kraken | adapter |
|---|---|---|---|
| output bias | shift blank vs. non-blank, or one class's bias | yes | lm_head is not adapted — n/a |
| row edits | revert, rescale or copy a row from parent/sibling | codec rows | per-module `B` rows |
| lossless rewrite | drop padding-only weights, merge chains | kraken+ `Cr255` | fold rank-deficient pairs |
| direction subtraction | project a direction out of a module's output | LSTM output / CTC input | `B` columns of the modules that write the loop |
| scaling | per-layer α on the fine-tune delta | `parent + α(child − parent)` | `lora_alpha` per module |
| transplant | one layer / module from parent or sibling | LSTM layer, conv block | one module's `A,B` from a sibling adapter |
| gate edits | BiLSTM gate biases | last resort | — |

Budget per session: 0 passes for pure reading, then 10, 25, 50. A "forward
pass" is one decode of the practice set at a fixed size. Every hypothesis,
prediction, edit, result and keep/revert decision goes to
`runs/<session>/log.jsonl`; the session ends with one held-out score.

### Track B — the family read across models, and merging as the comparator

1. **Lineage diffs** (reading only): per-layer L2 and cosine between parent
   and child, which rows moved, which codec classes were added. Exp 7 of Takehi et al.
   says weights alone can miss behavioural differences, so each diff is paired
   with the two models' confusion matrices on the same practice lines.
2. **Baselines the edits must beat:** uniform averaging of siblings and
   `parent + α·(child − parent)` for α ∈ {0.5, 0.75, 1.25, 1.5}. Codecs must
   align: `resize: union` in the BLB spec may have grown the codec, so rows are
   matched by character, never by index. For adapters: averaging and
   per-module α of sibling adapters of one base.
3. **Programmed merge:** per-layer choice of donor, justified from the
   readings — the merging Takehi et al. leave for future work, done by hand under the
   same budget rule.
4. **Retraining comparator:** one short fine-tune of the target (same data,
   5 epochs, lrate 1e-5) through `kraken_train` / `train_spec`, so a weight
   edit is compared with the cheapest gradient alternative.

### Track C — the served kraken family (added 07.11.2026)

Every kraken model the gateway serves for agentic_historian shares one network body
(`Cr` ×4, `Lbx200` ×3, output layer; VGSL of CATMuS/kraken's default), and only the
output layer's size differs with the alphabet. That makes the whole served family
comparable as networks, not only as readers:

- **RQ1 — identical parts.** For every pair, how many parameters are bit-identical or
  within float noise, per layer; across a lineage (base → fine-tune → fine-tune of it),
  which layers training never moved and could have been frozen.
- **RQ2 — where they differ.** Per layer, how far apart two models are: in weight space
  within a lineage (same initialisation, so element-wise distance means something), and
  functionally across lineages (what each layer computes on the same lines, compared with
  a permutation-invariant similarity, because independently trained networks may hold the
  same features in a different unit order).
- **RQ3 — exchange.** Whether replacing (or moving towards) a block of a specific model by the
  same block of a more general model of its lineage keeps the specific model's reading of its
  own material and improves it on material outside its training: stabilisation by
  exchange, measured with the paired, block-level statistics of docs/BLOCK_STATS.md.

Design, models and results: docs/MODEL_FAMILY.md.

## 3. Measurement discipline (details in docs/PROTOCOL.md)

- Practice set (seen by the programmer): a few hundred lines from the *train*
  side. Held-out set: one evaluation per session, at the end, never seen.
- Metrics: corpus CER (NFD), WER, collapsed-line count, length ratio, blank
  share (CTC) or `finish_reason` counts (VLM), top confusions; paired
  per-line bootstrap intervals for every difference.
- A win is CER down **and** collapses not up, on held-out, over parent,
  average, sham edit and the retrained comparator.
- Before the first number: the overlap audit between practice candidates and
  the `_eval` projects.

## 4. Infrastructure (details in docs/ASTERAIX_RUNBOOK.md)

| need | where |
|---|---|
| weights | research share `…/trained/<id>/<id>.mlmodel`, adapters under the jobs tree and `dh-unibe/*` on HF |
| sandbox, forward passes, scoring | asteraix, own venv with kraken **7.0.2** pinned (the serving loader reads CoreML only); card 0 for passes, card 1 stays the trainer's |
| practice data | compiled arrows in `~/atr-cache/arrows/`, line parquet sets on HF |
| held-out scoring | atr-mcp: `start_batch` → `score_ground_truth`; `ubelix_record` for the scratch comparators; `kraken_train` / `train_spec` for the retrain comparator |
| the programmer | one Claude Code session per `runs/<session>`, tools restricted to the sandbox runner; `python -I`, no network, budget enforced in-process |
| this box (h1) | plan, code, docs, tests that need no weights |

## 5. Phases and deliverables

| phase | work | done when |
|---|---|---|
| **0 Harness** | `wsp/`: subject interface, kraken subject, metrics, diff, session log with budget, sandbox runner; asteraix setup; practice sets | no-op round-trip through the serving engine gives identical text; `Cr255 → Cr1` scores identically; tests green on h1 |
| **1 Read** | per-model readings, lineage diffs paired with confusions | `docs/READINGS.md`: per model ≥3 hypotheses with predicted effect |
| **2 Patch** | WSP sessions on targets 1–3 at budget 10/25/50 | `runs/<session>/` with log, diff, practice and one held-out score |
| **3 Combine** | averaging, α-sweeps, programmed merge, retrain, sham control | one table, all candidates, same held-out sets |
| **4 Ship** | register winners on asteraix, read held-out through the gateway, `score_ground_truth`; publish diffs; weights to HF private | `docs/RESULTS.md` |
| **5 Adapters** | `LoraSubject`, vLLM forward pass, first adapter session on the Kurrent page model | the same deliverables as 1–4 for one adapter |
| **6 Family** | the kraken models agentic_historian serves (43 on the gateway, 07.11.2026), compared as networks: RQ1 how much is identical (freezable), RQ2 which layers differ, RQ3 whether exchanging parts from general to specific models stabilises them — Track C, docs/MODEL_FAMILY.md | `docs/MODEL_FAMILY.md` with the RQ1/RQ2 tables over every model whose weights are on asteraix, and a first RQ3 exchange grid with held-out-free practice numbers |

## 6. Risks

- The method's own evidence says gains come from removing obstacles. No
  obstacle found means a negative result; the harness, the lineage diffs and
  the merging baselines stay useful, and the `Cr255` fix is worth having.
- Convolutions are shared across positions and BiLSTM gates come in fixed
  blocks of four; a direction edit affects every position. Predictions
  written before the pass keep the budget from becoming blind search.
- Codec or module-name drift between parent and child breaks index-wise
  merging silently.
- Leakage: the 276 GT pages, the BLB `_eval` projects and the Hertwig pages
  have been used in several runs; the overlap audit comes first.
- On asteraix, card 1 belongs to the trainer; a session must never claim it.

## 7. Decisions taken / open

- Repo: `thodel/weightspace`, private (taken; remote to be created).
- Runs on asteraix; h1 for planning (taken, 06.11.).
- Pilot kraken, then QLoRA adapters (taken, 06.11.).
- Open: first target (proposal `bern-combined-v1`, one week, then xix);
  whether candidates may be registered on asteraix and read through the
  gateway; which adapter goes first in phase 5.
