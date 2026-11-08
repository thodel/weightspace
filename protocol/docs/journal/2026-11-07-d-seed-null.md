# 2026-11-07 · Seed-variance null

**Type:** control (training runs, nothing registered) · **Source:**
`docs/SEED_NULL.md`, `docs/seed-null/` · **Result:** [R6](../results.md#r6)

## Plan

Every reading so far compared a parent with a child. How much of what they show
would also appear between two runs that differ *only by seed*?

## Done

Five runs of the `hertwig-v2` recipe (91 epochs, own 45/5-page split), seeds
differing, everything else fixed. Weight readings seed against seed and parent
against run; held-out CER of each on the Bern held-out set.

## Found

| relative change | output rows | pooled conv | pooled LSTM |
|---|---:|---:|---:|
| seed vs seed | 0.12–0.15 | 0.24 | 0.28 |
| parent → run | 0.41 | 0.27 | 0.31 |

Seed noise is about a third of the movement of the output rows and about 90 %
of the pooled stacks; the sibling pair (0.16) is the size of seed noise.

- **Survives the null:** the blank bias rises +0.53 to +0.55 relative to the
  other biases (between seeds −0.02 to +0.01); rare rows grow and drop their
  bias in every parent → run pair; recurrent change increases with depth.
- **Does not survive:** the depth pattern of the conv stack (it appears between
  seeds too), "rare rows move most" (between seeds they move less than frequent
  letters), and any claim from a size of change alone.
- **Held-out:** seeds 27.20–27.65 % (sd 0.18), `hertwig-v2` 27.49 %; every
  seed's interval against it contains 0. **Differences under about one point
  between two runs cannot be told from seed.**

## Changes

Readings in the manuscript are restated as what survives this null.
