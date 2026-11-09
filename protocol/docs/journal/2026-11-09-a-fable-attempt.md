# 2026-11-09 · A Fable programmer on the validation sessions

**Type:** control (another programmer model) · **Source:** `docs/CONTROLS_REPORT.md`,
`runs/validation/*/f1` · **Result:** [R7](../results.md#r7)

## Plan

The [controls](2026-11-07-e-controls.md) ran the validation-practice sessions
with Sonnet, Opus and Haiku programmers. The planned run with a Claude Fable 5.1
programmer was postponed for lack of usage credits; it ran this morning, with
the same packs, budget and keep rule.

## Found

- **`german-xix-blb-v1`:** 0 of 9 edits kept, so 0 kept edits in 7 attempts by
  four programmer models.
- **`combined-v1`:** Fable went half-way toward the clean sibling in a single
  step. Practice CER fell further than in any attempt before (13.86 → 11.60),
  and held-out moved by **−0.02 points**: the practice optimum that does not
  transfer. The best edit of the earlier attempts (−0.39, a quarter-step) stays
  the best, and it is a merge, not a repair.

## Changes

A larger programmer model found a larger practice gain, and no held-out gain.
