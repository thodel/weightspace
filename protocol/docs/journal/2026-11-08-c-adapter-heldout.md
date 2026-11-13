# 2026-11-08 · Phase 5 adapter session: no held-out gain

**Phase:** 5 · **Session:** `20261107-kurrent-page-v2-b10-1` · **Held-out touched:** once ·
**Source:** `docs/PHASE5_REPORT.md` · Closes the [open entry](2026-11-08-a-adapter-session-v2.md)

## Found

The frozen adapter (direction 0 of layer 31 `o_proj` removed) and the unedited
one, scored on the same card with the same engine, paired over the 733 test
pages of 70 documents:

| 733 held-out pages | CER | clipped CER | length ratio | at cap | loops |
|---|---:|---:|---:|---:|---:|
| unedited | 0.4012 | 0.2430 | 1.177 | 44 | 63 |
| frozen (H1) | 0.4036 | 0.2460 | 1.175 | 43 | 61 |

**Δ +0.24 points [−3.5, +3.9]** by pages; 225 pages better, 233 worse, 275
identical. 37 pages run to the cap under both, 6 only with the edit, 7 only
without: **the edit swaps which pages loop more than it removes loops.**

Controls on the practice pages (outside the budget):

| practice, 64 pages | CER | clipped CER | at cap |
|---|---:|---:|---:|
| unedited | 0.4515 | 0.2564 | 9 |
| **H1 (session)** | **0.3470** | 0.2341 | **4** |
| sham: random direction out of the same module | 0.4128 | 0.2525 | 7 |
| whole adapter × 0.75 / 0.9 / 1.1 | 0.398 / 0.423 / 0.504 | | 5 / 7 / 6 |
| decoding with `repetition_penalty` 1.05, no edit | 0.3680 | **0.2287** | 4 |
| base model without the adapter | 1.6237 | 0.5581 | 23 |

## What it says

- **What held:** the budget-0 reading found the one direction with an outsized
  effect on stopping, and the edit did what the hypothesis said, with a dose
  curve, on the pages it was tuned on.
- **What failed:** the step from practice to held-out. Whether a long page loops
  under greedy decoding flips under small changes, so anything that lowers the
  end-of-turn logit — an edit or a random direction — changes a different
  handful of pages. A random direction alone moves practice CER by 3.9 points;
  the keep rule's interval, computed on one fixed set of readings, does not see
  that variance.
- A decoding guard with no weight edit nearly matches the edit on practice and
  reads better. It was not scored on held-out data (one held-out score per
  session).
- The base model loops on a third of the pages: stopping is something the
  adapter taught, and only partly.

## Changes

For a next attempt (not run): use the spread under several seeded sham
directions as the noise floor; a practice set with enough looping pages (about
200–300 pages for 15–20 loops); and measure stopping without decoding to the
end — the end-of-turn margin at the reference's true end, in one
teacher-forced pass. The retrain comparator (C4) for the adapter is not owed,
since no edit was kept on held-out evidence.
