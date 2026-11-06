# Plan — weight-space programming on the in-house kraken models

Written 06.11.2026 (CET). Status: proposal, nothing run yet.

## 0. What the method is, and what it is not

Tanaka's post (02.11.2026) describes an LLM that **reads a network's weights and
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
MLPs, one small transformer and two LLMs were tried — no CNN, no LSTM, no CTC.

It is **not** a merging method. The post names weight averaging as the merging
baseline it hopes to beat and lists "true model merging" as future work. Since
we have families of fine-tunes from one parent, merging and cross-model reading
are a natural second track here, but they are our extension, not the paper's.

So the honest expectation: on a healthy fine-tune, gains of a point or two of
CER at most; a real gain only where a model carries an identifiable defect
(CTC collapse / repetition, a dead or skewed output row, a layer the fine-tune
wrecked). The plan is built to find such defects first and to measure
everything against the cheapest alternatives (averaging, one more epoch of
fine-tuning).

## 1. Material

Seven in-house models in three families, listed in [docs/MODELS.md](docs/MODELS.md):

- **Family 1, Bern 1528:** `bern-hertwig-v1` → `v2` → {`schlussredaktion-v1`,
  `combined-v1`}; parent `cursive_inzigkofen` (CATMuS Medieval lineage).
  `combined-v1` (28.8 %) is worse than its parent and its sibling (16.7 %):
  the first thing to read.
- **Family 2, Kurrent XIX:** `german-xix-v1` (11.2 %) → `german-xix-blb-v1`
  (8.7 %). The production pair; ground truth and baseline runs exist.
- **Family 3, kraken+:** three from-scratch runs on the medieval corpus; the
  `Cr255,1,85` layer with 11 M gradient-free weights (`KRAKEN_PLUS.md`) is a
  known weight-space fact to start from.

Architecture everywhere: VGSL conv stack → 3 × BiLSTM (200 or 256) → CTC output
over a codec of ~100–150 classes. ~4 M live parameters, 16 MB CoreML files.

## 2. Two tracks

### Track A — WSP proper: read, diagnose, patch, one model at a time

Targets, in order:

1. `kraken-bern-combined-v1` vs its parent `hertwig-v2` — a fine-tune that got
   worse. Diff the two; the damage should be localised (output layer rows for
   characters the combined project adds, or the last LSTM). Edit classes:
   revert rows/blocks to the parent, rescale, interpolate per layer.
2. `kraken-bern-hertwig-v2` — 0.37 line-CER with known failure classes
   (collapsed lines, confusions from `pipeline.py eval`). Look for a
   blank-dominant direction in the output layer and the last LSTM (the CTC
   analogue of Qwen's loop direction).
3. `kraken-german-xix-blb-v1` — the production model. Expect little; do it
   because this is the one whose improvement would matter, and because the
   276 GT pages give the best measurement.
4. kraken+ run: the lossless `Cr255 → Cr1` rewrite, then the 85-channel
   bottleneck (widen by weight surgery — initialise the new channels from the
   parent's statistics — and see whether it needs training to help).

Edit classes, from safest to riskiest, each with a predicted effect written
down **before** the forward pass:

| class | what | where it helps |
|---|---|---|
| output bias | shift blank vs. non-blank, or one class's bias | over/under-segmentation of characters, systematic drops |
| output rows | revert, rescale or copy a class row from a sibling | a character the fine-tune unlearned; codec classes with ~zero rows |
| lossless rewrite | drop padding-only weights, merge linear chains | kraken+ `Cr255`; proves the harness round-trips |
| direction subtraction | project a direction out of LSTM output / CTC input columns | repetition / collapse, as in Tanaka Exp 8 |
| layer transplant | one LSTM layer or conv block from parent/sibling | damage localised to a layer |
| gate edits | forget/input gate bias of a BiLSTM | memory length; last resort |

Budget per session: 0 passes for pure reading, then 10, 25, 50. A "forward
pass" is one decode of the practice set (a few hundred lines, CPU, seconds).
Every edit, its hypothesis, prediction, result and keep/revert decision goes
into `runs/<session>/log.jsonl`; the session ends with one held-out score.

### Track B — the family read across models, and merging as the comparator

1. **Lineage diffs** (reading only): per-layer L2 and cosine between parent
   and child, which rows of the output layer moved, which codec classes were
   added. Tanaka's Exp 7 says weights alone can miss behavioural differences,
   so each diff is paired with the two models' confusion matrices on the same
   practice lines.
2. **Baselines the edits must beat:** uniform averaging of siblings
   (`hertwig-v2 + schlussredaktion + combined`; `xix-v1 + xix-blb-v1`), and
   `parent + α·(child − parent)` for α ∈ {0.5, 0.75, 1.25, 1.5}. Codecs must
   align: `resize: union` in the BLB spec may have grown the codec, so rows are
   matched by character, never by index.
3. **Programmed merge:** per-layer choice of donor, justified from the readings
   in 1 — the "merging" Tanaka leaves for future work, done by hand with the
   same budget rule.
4. **Retraining comparator:** `kraken_train` one more short fine-tune of the
   target (same data, 5 epochs, lrate 1e-5) so a weight edit is compared with
   the cheapest gradient alternative, not only with doing nothing.

## 3. Measurement discipline

- **Practice set** (the programmer sees it): a few hundred lines per family
  drawn from the *train* side — BLB train projects, the Hertwig train pages,
  or the public `*-lines` parquet sets. Never a held-out page.
- **Held-out set** (one evaluation per session, at the end): family 2 →
  `lassberg-blb-gt` `_eval` projects via `start_batch` + `score_ground_truth`
  (`datasets:lassberg-blb-gt/<project>_eval`, #752); family 1 → the Hertwig
  eval pages (`pipeline.py eval --doc 13922499 --pages 8,17,26,39,48,55`);
  family 3 → `german-medieval-v1` test on asteraix.
- Metrics: corpus CER (NFD, as `ketos test` counts), WER, **collapsed-line
  count**, length ratio, blank share of the CTC output, top confusions. A win
  is CER down **and** collapses not up, on held-out, over parent, average and
  the retrained comparator.
- Before the first number: check that the 276-GT practice candidates do not
  overlap the `_eval` projects (the HELDOUT_BOUNDARY audit in
  training-atr-models shows how).

## 4. Infrastructure

| need | where | notes |
|---|---|---|
| weights | research share, via tei or asteraix | not reachable from h1; copy the seven `.mlmodel` files once (112 MB) into `models/` on the sandbox host |
| sandbox + forward passes | **asteraix** (kraken-train venv, kraken 7.0.2, A40, `german_*.arrow` local) — recommended | h1 is possible for reading and CPU decodes (torch 2.14 cpu present, kraken 7.1.1 installable) but has 6.8 GB disk and 7 GB RAM; pin kraken **7.0.2** anywhere, because that is what `load_any` on the serving box reads (CoreML only) |
| practice data | HF `dh-unibe/*-lines` parquet (public), BLB train lines from the compiled corpus on asteraix | hundreds of lines, not thousands |
| held-out scoring | atr-mcp: `start_batch` → `score_ground_truth`; `kraken_train` for the comparator; `ubelix_record` for the scratch runs' CER | a candidate must first be registered on asteraix (`trained/<id>.yaml` + `weights_digest`) to be served — a deployment act, done by hand, named `kraken-<parent>-wsp-v<N>` |
| the programmer | Claude Code sessions, one per `runs/<session>` | the blog used codex with all tools off and a bubblewrap sandbox; we use `python -I`, no network, a budget counter in the runner that refuses the 51st pass |

Round-trip is the harness's first test: `.mlmodel` → `TorchVGSLModel` → edit →
`.mlmodel`, and the serving engine must read the result and produce the same
text as before the no-op edit.

## 5. Phases and deliverables

| phase | work | done when |
|---|---|---|
| **0 Harness** (week 1) | `wsp/`: loader, eval (CER/WER/collapse/confusions), diff, edit log with budget, sandbox runner; copy models; build practice sets | no-op round-trip through the serving engine is byte-identical in output; `Cr255 → Cr1` rewrite of a kraken+ checkpoint scores identically |
| **1 Read** (weeks 1–2) | per-model readings: architecture and codec dump, layer statistics, dead/near-zero rows, lineage diffs paired with confusion matrices | `docs/READINGS.md`: per model, ≥3 hypotheses each with a predicted CER effect |
| **2 Patch** (weeks 2–3) | WSP sessions on targets 1–3, budget 10/25/50 | per session a `runs/<session>/` with log, final diff, practice and one held-out score |
| **3 Combine** (weeks 3–4) | averaging and α-sweeps, programmed merge, retraining comparator | one table, all candidates, same held-out sets |
| **4 Ship** (week 4+) | register the winner(s) on asteraix, batch-read the held-out sets through the gateway, `score_ground_truth`; publish weight **diffs** against the parents in this repo, weights to HF private | `docs/RESULTS.md`; a candidate is promoted only on a held-out win with no collapse increase |

## 6. Risks

- The method's own evidence says gains come from removing obstacles. If the
  readings find no obstacle, the result is a negative one; it is still worth
  the week, because the harness, the lineage diffs and the merging baselines
  stay useful (and the `Cr255` fix alone is worth having).
- Convolutions are shared across positions and BiLSTM gates come in fixed
  blocks of four; a "direction" edit affects every position. Predictions must
  be written before the pass, otherwise the budget turns into blind search.
- Codec drift between parent and child breaks index-wise merging silently.
- Leakage: the 276 GT pages, the BLB `_eval` projects and the Hertwig pages
  have been used in several runs; the overlap audit comes before any number.
- Disk on h1 (96 % full) — keep weights and data on the sandbox host.

## 7. Decisions needed

1. Repo name and owner: proposal `thodel/kraken-weightspace`, private.
2. Sandbox host: asteraix (recommended, needs an account/ssh there for this
   session) or h1 (reading only, CPU decodes on small sets).
3. First target: `bern-combined-v1` (clearest defect) or `german-xix-blb-v1`
   (the model that matters)? Proposal: combined first, one week, then xix.
4. May a candidate be registered on asteraix and read through the gateway
   (uses the serving card for minutes, not hours)?
