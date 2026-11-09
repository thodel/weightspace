# Protocol amendments

Every change to [the rules](method.md), newest first. A session is judged by
the rules in force when it was preregistered; sessions affected by a change
are named.

## A4 — Intervals clustered by page { #a4 }

**Date:** 2026-11-07, adopted (in the protocol file since 2026-11-09). In use
for every result since the [controls](journal/2026-11-07-e-controls.md).

**Change:** every interval is also reported with a two-stage cluster bootstrap
by page. The keep rule keeps the line-level interval so that earlier decisions
stay reproducible; **a claim needs the page-level interval.**

**Why:** lines of one page share a hand, an image and a segmentation; treating
them as independent makes intervals too narrow. The best `combined-v1` edit
(−0.39 points) excludes 0 by lines and not by pages.

## A3 — Donor rule and practice side { #a3 }

**Date:** 2026-11-07. **Status:** *proposed, not adopted.*

**Proposed:**

1. A *donor* is any model whose weights an edit copies, transplants, averages
   or interpolates toward. It must not have been trained on the lines the keep
   rule is computed on: an edit that uses a donor is judged on the practice
   lines **unseen by that donor**; where that part is too small, the edit is
   not tried.
2. The practice set should be lines the subject was not trained on (validation
   pages, or evaluation-side pages disjoint from the held-out set). A practice
   set from the training side must be labelled so in the preregistration.

**Why:** in [phase 3](journal/2026-11-06-e-phase3.md) the keep rule accepted an
output-layer transplant from the parent because the practice lines were the
parent's own training lines; on held-out pages it was a loss. On training
lines a model has already fitted what an edit could still gain.

**Sessions affected:** sessions from 7 November onward use validation-side
practice sets and say so; earlier sessions used training-side lines.

## A2 — Stricter collapse detector { #a2 }

**Date:** 2026-11-07, adopted.

**Before:** a line counted as collapsed if its hypothesis was under half or
over twice the reference length, or if any character trigram occurred six
times.

**After:** the length test is unchanged; a repetition loop now needs the most
frequent trigram at least six times **and** at least twice as often as in the
reference, plus two.

**Why:** the old test flagged correct long lines (` . `, ` , `, `en `) — 142 of
the 428 "collapsed" held-out lines in [phase 4](journal/2026-11-06-f-phase4.md).

**Sessions affected:** the collapsed checks of the phase 1 and phase 3 sessions
used the old test; their decisions stand as recorded. Counts on the committed
baselines, old → new: `blb-eval`, `german-xix-blb-v1` 428 → 286; parent 455 →
436; `blb-train-300` 10 → 9; every Bern set unchanged.

## A1 — Keep rule requires a minimum effect { #a1 }

**Date:** 2026-11-06 (evening), after phase 1, before any further session.

**Before:** keep an edit if practice CER falls and the collapsed-line count
does not rise; ties revert.

**After:** additionally, the paired 95 % bootstrap interval of the CER
difference against the last kept state must lie entirely below
−*min_effect* (preregistered, default 0). Enforced by the harness.

**Why:** with a budget of 10 passes on 300 practice lines, the old rule kept
two edits whose effects (−0.016 and −0.008 CER points) were far inside their
intervals — it kept noise. See the [phase 1 entry](journal/2026-11-06-b-phase1.md).

**Sessions affected:** `20261106-german-xix-blb-v1-b10-1` ran under the old
rule and is reported as such; under A1 it would have kept nothing.
