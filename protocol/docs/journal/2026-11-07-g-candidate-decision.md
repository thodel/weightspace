# 2026-11-07 · Which `latin_incunabula` candidate to enable: neither

**Type:** preregistered decision · **Source:** `runs/20261107-latin_incunabula-exchange-2/preregistration.md`
(commit `0c8f4ff`, written before any of the scores) · **Result:** [R9](../results.md#r9)

## Plan

Two candidates from the [block exchange](2026-11-07-b-model-family.md): **v1**
(two group norms from the base, output layer half-way) and **v2** (the two group
norms only). Rule fixed before scoring: a candidate qualifies if it does not
lose abbreviations (MUFI, combining marks or Latin Extended-D with an interval
entirely above 0), its Reichenau CER is not worse, and its CER on neither
held-out Fraktur set is worse.

## Done

Each candidate read once on three audited held-out sets: Reichenau (4,374
lines), NewsEye (2,706) and GT-Fraktur (1,311).

## Found

| | Reichenau | NewsEye | GT-Fraktur | loses abbreviations | qualifies |
|---|---|---|---|---|---|
| unedited | 5.01 % | 28.56 % | 20.13 % | | |
| v1 | −0.03 [−0.07, +0.02] | **−2.04** [−2.22, −1.86] | **−0.44** [−0.57, −0.31] | yes (MUFI 532 → 731) | no |
| v2 | **+0.04** [+0.01, +0.08] | **−1.08** [−1.26, −0.92] | **−0.19** [−0.30, −0.08] | yes (MUFI 532 → 551) | no |

- **The exchange's purpose is confirmed on held-out print:** both read Fraktur
  they were never trained on better.
- **Its price is the model's specialisation:** v1 clearly loses abbreviations;
  v2 keeps most but not all, and its own-collection CER rises by 0.04 points.
- **Decision: neither is enabled.** The rule set no minimum size, and it is not
  changed after the fact.

## Changes

A next comparison would need a minimum effect for the abbreviation criterion,
set *before* scoring — or a use case that wants Fraktur robustness more than
abbreviations, for which v1 is the better model.
