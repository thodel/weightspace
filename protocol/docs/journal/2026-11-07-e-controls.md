# 2026-11-07 · Controls for the article

**Type:** controls · **Source:** `docs/CONTROLS_REPORT.md` ·
**Results:** [R7](../results.md#r7), [R8](../results.md#r8) ·
**Amendments:** [A3](../amendments.md#a3) (proposed), [A4](../amendments.md#a4)

## Plan

Two sessions without a gain say little unless we know the method *can* find a
gain. Questions: does it repair a defect we plant on purpose? Does practice on
lines the model was not trained on change the outcome? Is a kept edit better
than a random one of the same size? And why do output-layer levers not move
the Kurrent model?

## Done

- **Positive control:** eight planted corruptions of `german-xix-blb-v1`
  (held-out 9.55 %), each given to a programmer with passes, a blind
  programmer (budget 0) and a hand-written repair list; plus diffuse damage
  (noise on all weights or on one block) and donor-free variants.
- **Unseen practice lines:** sessions on both targets with practice drawn from
  lines the subject was not trained on, three programmer sizes.
- **Sham (C5):** 20 norm-matched random edits plus the same blank shift.
- **Global interpolation** parent → child on the Kurrent pair.
- **Margins** at 696 deleted characters.

## Found

- **Planted, localised defects are repaired — by everyone.** For all six
  corruptions that cost at least one point (1.3 to 90 points), the programmer,
  the blind programmer and the rule list each restored held-out CER to within
  0.22 points. So the *absence* of a found defect on the real models carries
  information. But the list did as well as the programmer, and the blind
  programmer as well as both.
- **Diffuse damage is repaired only by measuring:** with noise on all weights,
  only the programmer with passes found a repair (83 % of the damage, by
  interpolating toward the parent). With an unrelated donor, zeroed rows or
  blocks were not rebuilt (0–19 %).
- **Unseen practice:** the Kurrent model kept no edit in 6 attempts. For
  `combined-v1`, every attempt that kept anything moved toward the clean
  sibling, −0.19 to −0.39 points held-out. The best (−0.39) is reached by **0 of
  20 shams** (mean −0.01, sd 0.06) — but it equals plain interpolation (a
  quarter-step), is zero at half a step, and its page-clustered interval
  includes 0. **A merge, not a repair.**
- **Interpolation:** single blocks and pairs of blocks of the Kurrent fine-tune
  are close to additive; moving the whole network costs more than the sum
  (+3.81 points [+2.57, +4.98] at α = 0).
- **Margins:** 18 % of deleted characters have no frame in which the blank
  could be replaced; for the rest the median deleted character is 3.2 nats
  below the blank. Lowering the blank bias recovers one deletion in 40, 9 and 5
  at −0.1, −0.5 and −1.0, and pays with almost as many insertions at −1.0.
  **The deletions are decided upstream of the output layer** — which explains
  [phase 1](2026-11-06-b-phase1.md).

## Changes

- Intervals are now also reported clustered by page ([A4](../amendments.md#a4));
  a claim needs the page-level interval.
- Donor rule and practice side proposed ([A3](../amendments.md#a3)), not yet
  adopted.
- Not done: the same controls with a Fable programmer (usage credits); planned
  for 9 November.
