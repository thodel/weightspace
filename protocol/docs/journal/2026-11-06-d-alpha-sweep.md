# 2026-11-06 · C3 per-block α-sweep

**Type:** control (outside any session budget) · **Passes:** 16 on practice ·
**Held-out touched:** no · **Source:** `docs/READINGS.md`, commit `4b6f567`

## Plan

Locate where the fine-tune `german-xix-v1 → german-xix-blb-v1` keeps its
24-point practice gain (parent CER 0.304, child 0.065). One block at a time is
set to parent + α(child − parent), the rest of the network left as the child;
output rows aligned by character, new rows kept. α ∈ {0.5, 1.5}.

## Found

Δ CER (points) against the unedited child, paired 95 % interval:

| block | α = 0.5 | α = 1.5 |
|---|---|---|
| C_0 (conv) | −0.02 [−0.13, +0.10] | +0.21 [+0.06, +0.37] |
| C_3 (conv) | +0.16 [+0.01, +0.30] | +0.27 [+0.14, +0.43] |
| C_6 (conv) | +0.33 [+0.14, +0.55] | +0.24 [+0.06, +0.42] |
| C_9 (conv) | +0.16 [−0.01, +0.32] | +0.17 [+0.00, +0.36] |
| L_12 (BiLSTM) | +1.90 [+1.32, +2.59] | +0.27 [+0.01, +0.55] |
| L_14 (BiLSTM) | +1.18 [+0.87, +1.51] | +0.37 [+0.09, +0.66] |
| L_16 (BiLSTM) | **+2.31** [+1.97, +2.67] | **+0.05** [−0.18, +0.28] |
| O_18 (output) | +0.23 [+0.10, +0.36] | +0.12 [+0.00, +0.25] |

1. **The gain is held jointly.** Halving any single block costs at most
   2.3 points of a 24-point gain; all eight halvings together about 6.5. Every
   block depends on the others having moved with it — the profile of a
   healthy fine-tune, not of a localised repair.
2. **The recurrent stack holds what is localisable.** The half-moved last
   LSTM keeps the letters but loses word boundaries (space deletions 103 → 263,
   WER 0.17 → 0.32): it is where the child learnt Kurrent spacing.
3. The conv bias that moved most in relative terms (rel. L2 1.09) is worth
   0.16 points: a large relative change in a 32-element bias is not a large
   function change.
4. **Only L_16 tolerates extrapolation** (α = 1.5 indistinguishable from the
   child); every other block gets worse.

## Changes

Preregistered for the next session (budget 10): target L_16, α ∈ {1.1, 1.25},
predicted change inside [−0.2, +0.2] points, expected to keep nothing under
A1 unless a sub-tensor extrapolates where the whole block does not. The
collapsed lines are likely the better next question.
