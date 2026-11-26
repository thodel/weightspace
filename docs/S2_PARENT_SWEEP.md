# Kraken session 2: the parent step at fixed sizes, no programmer

Pre-registration: docs/S2_PARENT_SWEEP_PREREG.md (commit c360782, 24.11.2026 21:55 CET, before any
model was scored). Script: `scripts/s2_parent_sweep.py`; numbers: `docs/s2_parent_sweep.json`.
Scored 24.11.2026, 21:50–21:52 UTC (22:50–22:52 CET), CPU, 5 workers × 4 threads.

## 1. The edit

`w ← (1 − α) w + α · hertwig-v2` in every tensor of `kraken-bern-combined-v1`, output rows by
character, blank row and blank bias with row 0, rows the parent lacks unchanged. This is the parent
step the programmers kept. **Reproduction:** α 0.25 and 0.30 have the digests of the split's
`part1` in b10-2/-4 and b10-3 (asserted) and give the same held-out Δ and intervals (−0.63 and
−0.54, S2_REPEATS §4). The other points are read.

## 2. Results (Δ vs C0, CER points; line and page-clustered 95 % intervals)

| α | practice Δ | keep rule | held-out Δ (all) | line | page | Hertwig | Schlussred. | collapsed, held-out (C0 34) |
|---:|---:|:---:|---:|---|---|---:|---:|---:|
| 0.10 | −0.65 | keep | −0.49 | [−0.75, −0.26] | **[−0.99, −0.06]** | −1.21 | +0.09 | 31 |
| 0.15 | **−0.82** | keep | −0.59 | [−0.89, −0.31] | **[−1.23, −0.01]** | −1.58 | +0.21 | 27 |
| 0.20 | −0.80 | keep | **−0.70** | [−1.06, −0.36] | [−1.54, +0.06] | −2.09 | +0.41 | 26 |
| 0.25 | −0.76 | keep | −0.63 | [−1.04, −0.26] | [−1.62, +0.29] | −2.35 | +0.73 | 24 |
| 0.30 | −0.76 | keep | −0.54 | [−0.96, −0.12] | [−1.66, +0.50] | −2.57 | +1.06 | 24 |

Per-hand page intervals: Hertwig lies below 0 at every α (upper bounds −0.61 to −1.59);
Schlussredaktion contains 0 up to α 0.20 and lies above 0 from α 0.25 (+0.24 and +0.54 lower bounds).

## 3. Read by the pre-registered rule

1. **Shape.** Held-out Δ is U-shaped in α, lowest at 0.20. Its range over α 0.10–0.30 is
   **0.21** (−0.49 to −0.70), at the Fable 5.2 within-model range (0.20). By the rule, **the step
   size alone can account for the spread between runs.** The repeats chose α 0.25 or 0.30, and
   their parent steps alone (−0.63, −0.54) lie on this curve.
2. **Per hand: holds.** Hertwig Δ falls monotonically with α (−1.21 → −2.57) and
   Schlussredaktion Δ rises monotonically (+0.09 → +1.06). The Schlussredaktion cost grows with each
   0.05 step (+0.12, +0.20, +0.32, +0.33), while the Hertwig gain per step shrinks after 0.20
   (−0.37, −0.51, −0.26, −0.22).
3. **Practice against held-out.** Practice is lowest at α 0.15 (−0.82), held-out at α 0.20
   (−0.70). Practice is nearly flat from 0.15 to 0.30 (−0.82 to −0.76), and the keep rule keeps
   every α. So practice does not tell the programmer which size to take; the programmers' α 0.25
   and 0.30 lie past the held-out minimum.
4. **A4.** 2 of 5 α (0.10 and 0.15) have a held-out page interval below 0, the larger steps do
   not. Held-out was looked at five times and no α was chosen on it, so this is **two of five
   looks, not a held-out gain of the parent step.** What it does show: the page interval widens
   with α (width 0.93 → 2.16), because the Schlussredaktion cost is concentrated on few pages. A
   smaller step trades some mean gain for much less page-to-page variance, which is how Fable 5.1
   (α 0.125) passed A4 and the α 0.25–0.30 runs did not.

## 4. What this means for session 2

- The parent step helps held-out at every size from 0.10 to 0.30 (line intervals all below 0).
- Which size a programmer takes moves the net held-out Δ by as much as everything that differed
  between the four Fable 5.2 runs, and decides whether the page interval excludes 0. Practice
  cannot choose the size: every α in the range is kept and the practice curve is flat.
- Together with S2_REPEATS: the differences between programmers on session 2 are differences in
  step size (and a blank step that does nothing), not in what they found.

## 5. Not done

Grid points outside 0.10–0.30 or finer than 0.05; a second step size chosen on a separate
validation set; the blank step at fixed sizes (it does nothing alone, S2_REPEATS §4).
