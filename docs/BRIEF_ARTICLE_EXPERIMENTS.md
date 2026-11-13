# Brief for the asteraix session: experiments the article needs

Written 07.11.2026 on h1. Source: the internal review of the JDMDH manuscript
(`paper/review/internal_review_2026-11-07.md`) and the fact-check
(`paper/review/factcheck_2026-11-07.md`). Not started. Rules of docs/BRIEF_PHASE0.md section 0 apply
(CPU only, card 1 is the trainer's, nothing written outside the repo and `~/atr-cache/wsp/`, no change to the
share, registry, HF or trainer). Run in this order; each item is independent of the later ones.

## 1. Zoo survey (part 6 of the article)
See docs/BRIEF_ZOO_SURVEY.md. Add one requirement: report the blank bias **relative to the mean of the other
biases**, because the softmax is unchanged by a constant added to all of them. Also report it for the seven models
already read.

## 2. Positive control: planted defects
On `german-xix-blb-v1` (practice `blb-train-300`, held-out `blb-eval-hand`), corrupt the model in known ways:
zero a row; negate a row; scale a layer by 0.5 and by 2; add noise of fixed norm to one LSTM block; swap the rows
of two frequent characters; add +2 to the blank bias; zero one conv block. Record each planted edit's CER on practice
and held-out. Then run the same session protocol (preregistration, keep rule, budget 10) on each corrupted model,
with (a) a programmer, (b) a blind programmer (budget 0, readings only), (c) a hand-written list of the obvious
repairs, (d) a plain coordinate search over output biases. Question: does the method find and repair defects that
are known to exist? Without this the article cannot tell "no defect found" from "the method cannot find defects".

## 3. Practice set that is not training data
Build a validation-based practice set per family: lines from pages that are in neither the training nor the held-out
projects (BLB: carve pages from the unused part of the train projects only if the trainer's split allows it;
otherwise from `_eval` objects not used for the held-out set, audited). Repeat session 1 and session 2 with it, keep
rule of PROTOCOL.md section 3, at least three programmer runs per model. Record the programmer model and settings.
Update PROTOCOL.md section 2 with the donor rule (a donor must not have been trained on the practice lines).

## 4. Statistics
Add a cluster bootstrap (resample pages or objects, then lines) to `wsp.metrics.bootstrap_delta` and recompute every
interval in docs/RESULTS.md, the phase reports and the alpha sweep. Report the false-keep rate of the keep rule with
sham edits (random edits of the same norm, 20 per session) as the null.

## 5. Additivity and held-out interpolation
Interpolate all blocks of `german-xix-v1` → `german-xix-blb-v1` together (α from 0 to 1.5) and block pairs; compare the
sum of single-block costs with the joint cost at the same α. Score on the practice set and on `blb-eval-hand`.

## 6. Margins at deleted frames
For the lines where a space, `e` or `n` is deleted, record the blank logit minus the best character logit at the
deleted frames, before and after the blank-bias edits of session 1. This tests the claim that the deletions are decided
upstream.

## 7. Remaining protocol controls (C2, C5, C6)
Uniform average of siblings (aligned by character); sham edits with the norm of each kept edit, five seeds; blind
programmer at budget 0 for sessions 1 and 2.

## 8. Checks that the manuscript flags
- The value 9.55 % appears for two different sets and procedures (docs/PHASE4_REPORT.md section 4 and
  docs/PHASE5_CLEANUP_REPORT.md section 2). Re-run and confirm.
- The sandbox command line in `runs/*/summary.json` sets a GPU variable although the reports say CPU. Confirm.
- docs/READINGS.md states that the conv stack moves less than the recurrent stack; its own tables and
  `docs/readings/diff_*.json` say the deeper convs move as much or more. Correct the text.
- docs/READINGS.md and docs/PHASE0_REPORT.md call `hertwig-v1` → `hertwig-v2` a continuation; MODELS.md says two
  separate fine-tunes of `cursive_inzigkofen`. Settle which is true.
- The line count of `combined-v1` (5,441) is not 1,786 + 3,457. Ask the trainer.
- docs/MODELS.md: "seven in-house models"; the table lists six plus the public parent. Correct.

## Needs an OK first
C4, a retrain of `combined-v1` on the corrected 22.10 + 24.10 data, is a trainer job on a GPU; ask before starting.
The planted-defect and validation-set sessions use CPU only.
