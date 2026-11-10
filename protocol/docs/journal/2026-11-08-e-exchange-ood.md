# 2026-11-08 · Exchange on out-of-domain material and across lineages

**Type:** preregistered candidates, then a feasibility test · **Source:** `docs/MODEL_FAMILY.md`
(section of 8 November), `docs/family/PREREG-ood-exchange.md` (commit `517606e`) ·
**Result:** [R13](../results.md#r13)

## Plan

The [first exchange grid](2026-11-07-b-model-family.md) measured
"stabilisation" on material every model read at 51–77 % CER. Repeat it where it
matters: on out-of-domain hands the specific model reads at roughly 15–30 %,
split by document before any reading. Then test whether exchange works across
*unrelated* models once their units are aligned.

## Done

- **Material:** for `german-xix-blb-v1`, Swiss Federal Council minutes
  1848–1903 (18.2 %; its parent 11.3 %) and the Basel land register (28.2 %;
  parent 25.1 %). For `bern-hertwig-v2`, the A V 1447 Schlussredaktion hand
  (29.6 %; parent 65.5 %). Three further corpora were too hard (51–64 %).
- **Audit:** it first found shared source documents behind different container
  ids; those lines were removed from the held-out side before any read.
- **Candidates**, from practice and line-disjoint confirmation: A = first
  convolution `C_0` half-way to the parent; B = `C_0` transplanted.
  Preregistered rule: on the federal minutes Δ ≤ −1.0 with the page-clustered
  upper end ≤ −0.5; own material and Basel upper ends ≤ +0.5.
- **Cross-lineage:** unit permutation by activation correlation (Hungarian
  matching per layer, applied consistently to conv channels, LSTM units and
  downstream inputs).

## Found

| held-out, Δ CER (points, page-clustered) | federal minutes | Basel | own material |
|---|---|---|---|
| A, `C_0` half-way | −0.35 [−0.57, −0.14] | −0.50 [−0.74, −0.24] | +0.05 [−0.02, +0.13] |
| B, `C_0` transplanted | −0.46 [−0.75, −0.17] | −1.07 [−1.40, −0.74] | +0.11 [−0.01, +0.25] |

- **Neither qualifies:** the stabilisation is real and holds on held-out
  documents at no own-material cost, but it is a sixth of the practice estimate
  and below the preregistered minimum. Post hoc, the practice gain sat in a few
  source documents the held-out side does not contain: **confirmation sets
  should be document-disjoint, not only line-disjoint.**
- Combining more conv blocks only adds own-material cost (+0.28 to +0.87).
- **Bern: null.** The parent reads the out-of-domain hand worse than the child,
  and no conv move helps.
- **Across lineages:** the permutation is exactly function-preserving (every
  line identical) and recovers the identity within a lineage. But units of
  unrelated models match only at correlation 0.2–0.4, and every front-end
  transplant breaks the specific model (+2.6 to +92 points). No candidate;
  nothing read held-out.

## Changes

Nothing registered. Exchange across unrelated models is not possible at the
unit level with this alignment.
