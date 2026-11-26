# Kraken session 2: the parent step at fixed sizes, no programmer — pre-registration

Written 24.11.2026, 21:55 CET, on asteraix, **before any sweep model is scored**. Operator: a
Claude Opus 5.5 session (`claude-opus-5-5`). Script: `scripts/s2_parent_sweep.py`, committed with
this file. PROJECT_STATE open item 13, second half; follows docs/S2_REPEATS.md.

## 1. Question

All four Fable 5.2 runs of session 2, and three other programmers, kept a whole-network step
toward the parent `hertwig-v2`. Three of the four Fable 5.2 runs chose exactly α 0.25. How does
held-out change with the size of that step, when no programmer chooses it?

## 2. The edit

For α in **0.10, 0.15, 0.20, 0.25, 0.30**, the subject `kraken-bern-combined-v1` is moved
`w ← (1 − α) w + α · parent` in every tensor. Output rows are paired by character; the blank
weight row and the blank bias pair with row 0; rows the parent lacks keep the subject's values.
This is exactly the edit the programmers kept as H1: at α 0.25 it has the digest of the
component-split `part1` of b10-2 and b10-4 (`343e03bd…`), at α 0.30 that of b10-3 (`f8968caf…`).
The script asserts both digests.

No other edit, no blank step, no keep decision, no programmer. Each model is scored once on
practice (`bern-train-300`) and once on held-out (`bern-eval`, 545 lines, 14 pages), CPU only,
5 workers with 4 OMP threads each.

## 3. Known before scoring

- α 0.25 and α 0.30 are the split scores already in docs/S2_REPEATS.md §4: held-out −0.63
  [page −1.62, +0.29] and −0.54 [page −1.66, +0.50]. They are rescored here and must agree.
- α 0 is C0; α 1 is the parent itself (held-out CER 0.2749 against C0 0.1933, +8.16 points).
- Fable 5.1 kept α 0.125 (not on this grid) with held-out −0.57.
- α 0.10, 0.15 and 0.20 have not been scored by anyone.

## 4. Quantities

Per α, against C0, in CER points (paired bootstrap, 2,000 resamples, seed 20261106; page
clusters `<project>/<page>`):
1. Practice Δ with line and page intervals, and whether the keep rule (line interval below 0)
   would keep it.
2. Held-out Δ with line and page intervals, for all lines, Hertwig and Schlussredaktion.
3. Collapsed counts on both sets.

## 5. Reading rule (fixed now)

1. **Shape.** The held-out Δ per α is reported as a curve with its intervals. Its range over
   α 0.10–0.30 is set against the within-model range of the four Fable 5.2 runs (0.20).
   - If the range over α is **below 0.20**, the step size within 0.10–0.30 moves held-out less
     than run-to-run variation did, and the size the programmer picks is not what decides a run.
   - If it is **0.20 or more**, the step size alone can account for the spread between runs.
2. **Per hand.** Hertwig and Schlussredaktion Δ per α are reported; the statement "a larger
   parent step buys Hertwig at a rising price on Schlussredaktion" (S2_REPEATS §5.4) holds if
   Hertwig Δ falls and Schlussredaktion Δ rises with α over the grid, and is reported as not
   holding otherwise.
3. **Practice against held-out.** The α with the lowest practice Δ and the α with the lowest
   held-out Δ are both named. If they differ, so does the step size that practice rewards from
   the one that transfers.
4. **A4.** The number of α (of 5) whose held-out page interval lies below 0. Held-out is looked
   at five times here and no α is chosen on it, so a page interval below 0 at one α is reported
   as one of five looks, **not** as a held-out gain of the parent step.
5. α 0.25 and 0.30 must reproduce the split scores (§3) exactly; if not, the sweep is reported
   as not reproducing and its other points are not read.

Nothing is registered, uploaded or served; no weights are committed.
