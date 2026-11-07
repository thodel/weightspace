# Experimental protocol

Written so that every number in a future paper can be traced to a run folder,
a practice-set hash, a held-out-set hash and a log. Deviations are allowed but
must be written into the run's `summary.json` under `deviations`.

## 1. Units

- **Subject:** one model artefact with a digest (sha256 as saved).
- **Session:** one programmer, one subject, one practice set, one budget, one
  held-out evaluation at the end. Folder `runs/<YYYYMMDD>-<subject>-b<budget>-<n>/`.
- **Pass:** one forward pass over the whole practice set at the session's
  fixed size. Partial passes count as whole passes.

## 2. Pre-registration (before the first pass of a session)

The session's `preregistration.md` states:

1. subject id and digest; parent/sibling digests if donors are used;
2. practice set: source dataset, split side (**train side only**), selection
   rule, size, sha256 of the line list;
3. held-out set: identity (document ids or project names), size, sha256; the
   overlap audit result against the practice set and against every corpus the
   subject was trained on (0 shared documents, or the session is labelled
   *not held-out*);
4. budget (0 / 10 / 25 / 50);
5. the hypotheses the reading produced, each with a **predicted direction and
   magnitude** of CER change and the edit class that would test it;
6. the controls to be run (§4).

## 3. During the session

- Every hypothesis, edit, pass and decision is logged by the harness; the
  programmer's free text goes into `notes.md` with timestamps.
- **Keep rule (as of 06.11.2026, after phase 1):** an edit is kept only if
  the practice CER falls against the *last kept state* **and** the paired
  per-line bootstrap 95 % interval of that difference (2,000 resamples, seed
  20261106, §5) lies entirely below −`min_effect`; `min_effect` is
  preregistered per session and defaults to 0, i.e. the interval must exclude
  0. The harness enforces it: `Session.keep()` refuses an edit whose interval
  contains 0 (or that has no pass since the edit), undoes it and logs a
  `decision` with `kept: false` and the reason; `revert()` is unchanged. The
  collapsed-line count must still not rise (checked by the programmer and
  reported). Session `runs/20261106-german-xix-blb-v1-b10-1` ran under the
  earlier rule (CER falls and collapsed does not rise, ties revert) and kept
  two edits with intervals containing 0; under this rule it would have kept
  none (docs/PHASE1_REPORT.md §4).
- No held-out data is reachable from the sandbox. The held-out score is
  computed once after `freeze()`; a second held-out evaluation of the same
  session is recorded as such and is not a result.

## 4. Controls, per subject

| control | what | why |
|---|---|---|
| C0 no-op | load, save, score | proves the round-trip; the baseline number |
| C1 parent | the parent model on the same sets | is the fine-tune worth its delta |
| C2 average | uniform average of siblings (codec/module aligned) | the merging baseline the post names |
| C3 α-sweep | `parent + α(child − parent)`, α ∈ {0.5, 0.75, 1.25, 1.5} | the cheapest weight-space family |
| C4 retrain | 5 epochs, lrate 1e-5, same data, through the trainer | the cheapest gradient alternative |
| C5 sham | a random edit of the same tensors, same Frobenius norm as the kept edits, 5 seeds | does the *content* of the edit matter |
| C6 blind | the same programmer and prompt at budget 0 | how much the passes bought |

C2–C4 are run once per family, not per session. C5 and C6 are run for every
session whose edit is reported as a gain.

## 5. Metrics and statistics

- Corpus CER and WER (NFD, whitespace flattened, as `ketos test` and the
  trainer count), collapsed lines, length ratio, top-10 confusions; for CTC
  subjects the blank share; for VLM subjects `truncated_cer`, `finish_reason`
  counts and tokens.
- **Collapsed line (as of 07.11.2026):** hypothesis under half or over twice
  the reference length, or a repetition loop by `wsp.metrics.degenerate_repeat`
  (the most frequent character trigram occurs at least six times *and* at
  least twice as often as in the reference, plus two). Until 07.11.2026 any
  trigram occurring six times counted, which flags correct long lines
  (` . `, ` , `, `en `); every collapsed count reported before that date, and
  the collapsed checks of the sessions `20261106-german-xix-blb-v1-b10-1` and
  `20261106-bern-combined-v1-b10-1`, used that legacy test
  (`is_collapsed(..., legacy=True)` reproduces it). Those decisions stand as
  recorded. Counts on the committed baselines, legacy → current:

  | set | model | lines | legacy | current |
  |---|---|---:|---:|---:|
  | blb-eval | german-xix-blb-v1 | 5,628 | 428 | 286 |
  | blb-eval | german-xix-v1 | 5,628 | 455 | 436 |
  | blb-train-300 | german-xix-blb-v1 | 300 | 10 | 9 |
  | blb-train-300 | german-xix-v1 | 300 | 23 | 23 |
  | bern-eval, bern-train-300, bern-hertwig-train-300 | all family-1 models | | unchanged | unchanged |

  `python -m wsp.collapse` classifies the lines the legacy test flagged
  (docs/PHASE4_REPORT.md).
- Every reported difference carries a paired per-line bootstrap 95 %
  interval (2,000 resamples, seed 20261106). A gain whose interval includes 0
  is reported as such.
- All sessions are reported, including reverted and failed ones; the ratio of
  kept to attempted edits is a result.

## 6. Reproducibility artefacts per session

`preregistration.md`, `log.jsonl`, `notes.md`, `diff.safetensors` (sparse diff
against the starting digest), `summary.json` (digests, hashes, budget used,
practice and held-out metrics, controls, deviations), the exact package
versions (`pip freeze` of the sandbox venv), the sandbox command line, and
the programmer's model id and settings (model, reasoning effort, tools).
Weights themselves are not committed; diffs are.

## 7. Scaling to adapters

Identical, with: pass = one vLLM batch of fixed page count; sampling pinned
(temperature 0, or the trainer's eval settings, stated); held-out = the
trainer's own eval set for that adapter (`docs/EVAL_SETS.md` in
training-atr-models), scored by the same `wsp.metrics` code; C4 = one short
QLoRA continuation through `train_spec`.
