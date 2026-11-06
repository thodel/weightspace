# Protocol amendments

Every change to [the rules](method.md), newest first. A session is judged by
the rules in force when it was preregistered; sessions affected by a change
are named.

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
