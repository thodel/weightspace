# 2026-11-08 · `latin_incunabula` v3: an exchange that spares the abbreviations

**Type:** preregistered candidate · **Source:** `runs/20261108-latin_incunabula-exchange-3/preregistration.md`
(commit `fdf25e2`, pushed before any v3 held-out score) · **Result:** [R12](../results.md#r12)

## Plan

[R9](2026-11-07-g-candidate-decision.md) found the price of the exchange: moving
the output layer toward a print model costs the abbreviation signs. v3 keeps the
two group norms from the base and moves the output layer half-way **only on the
rows of the 108 shared characters that are not abbreviation characters**, plus
the blank. The 13 shared abbreviation characters keep their own rows. Rows are
paired by character.

The new rule has minimum effects, chosen from R9's numbers before v3 was scored:

- **Loses abbreviations:** the change of a block's share of CER (MUFI, combining
  marks, Latin Extended-D) has a line-level interval entirely above +0.02 points.
- **Reichenau worse:** interval entirely above +0.05.
- **Fraktur better:** needs the page-clustered interval below 0.

R9's decisions on v1 and v2 are not re-judged under this rule.

## Done

Four variants on practice and confirmation lines only; the one with the largest
mean Fraktur gain went to the three held-out sets of R9, each read once.

## Found

| held-out | Δ CER (points) | page-clustered interval | *v1 (R9)* |
|---|---:|---|---:|
| Reichenau, own collection | −0.02 | [−0.09, +0.04] | *−0.03* |
| NewsEye (Fraktur) | **−1.66** | [−2.02, −1.30] | *−2.04* |
| GT-Fraktur | −0.21 | [−0.40, +0.01] | *−0.44* |
| MUFI share of CER | +0.014 | | *+0.110* |
| combining marks share | +0.025, line interval [+0.010, +0.041] | | *+0.048* |

- **v3 qualifies.** It keeps most of the Fraktur gain and most of the
  abbreviations. The combining marks are the closest call: the point estimate
  is above the minimum, but the interval is not entirely above it, so by the
  rule it is not a loss.
- **Serving path** (with the user's OK): registered for the evaluation only, 58
  Reichenau pages read through the project's gateway, then disabled again.
  Median page −0.09 points [−0.16, −0.03], character-weighted −0.14
  [−0.23, −0.07]; better on 36 pages, worse on 15. Not worse — about half of
  v1's serving-path gain.

## Changes

- v3 stays registered and disabled. Offering it for use is a decision for its
  users, not for this rule.
- The three held-out sets have now been read by three candidates of this model;
  any further candidate needs fresh held-out lines.
