# 2026-11-24 · Kraken session 2: the parent step at fixed sizes

**Type:** dose sweep (no programmer) · **Models:** `kraken-bern-combined-v1` moved α toward
`hertwig-v2`, α 0.10–0.30 · **Held-out touched:** once per α (5 looks, none selected on) ·
**Source:** `docs/S2_PARENT_SWEEP.md`, pre-registration `docs/S2_PARENT_SWEEP_PREREG.md` ·
**Result:** [R3](../results.md#r3)

## Plan

[The repeats](2026-11-24-a-kraken-s2-repeats.md) showed every parent-step run lands in one place,
with three of four Fable 5.2 runs choosing α 0.25. Open item 13 asked how held-out depends on that
size when nobody chooses it. The edit is exactly the programmers' kept step; the reading rule was
committed before scoring.

## Found

| α | practice Δ | held-out Δ, page 95 % | Hertwig | Schlussred. |
|---:|---:|---|---:|---:|
| 0.10 | −0.65 | −0.49 [−0.99, −0.06] | −1.21 | +0.09 |
| 0.15 | −0.82 | −0.59 [−1.23, −0.01] | −1.58 | +0.21 |
| 0.20 | −0.80 | −0.70 [−1.54, +0.06] | −2.09 | +0.41 |
| 0.25 | −0.76 | −0.63 [−1.62, +0.29] | −2.35 | +0.73 |
| 0.30 | −0.76 | −0.54 [−1.66, +0.50] | −2.57 | +1.06 |

- α 0.25 and 0.30 reproduce the repeats' split exactly (digest and score).
- Held-out is U-shaped, lowest at 0.20; range 0.21, the same as the four Fable 5.2 runs (0.20).
- Practice is flat and keeps every α.

## Changes

- **Step size explains the run-to-run spread,** and practice cannot pick it.
- **Passing A4 tracks step size:** small steps (0.10, 0.15, and Fable 5.1's 0.125) have narrow
  page intervals because the Schlussredaktion cost is small. Not claimed as a gain: five looks.
- **Open item 13 closed.** Nothing was registered.
