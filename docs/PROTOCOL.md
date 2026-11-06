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
- An edit is kept only if the practice CER falls **and** the collapsed-line
  count does not rise. Ties revert.
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
