# 2026-11-11 · Kraken session 2 with a Fable programmer

**Type:** replicate (another programmer model) · **Session:** `runs/20261111-bern-combined-v1-f1-b10-1` ·
**Budget:** 10 (9 spent) · **Held-out touched:** once, after freeze ·
**Source:** `docs/KRAKEN_S2_FABLE.md` · **Result:** [R3](../results.md#r3)

## Plan

Repeat [phase 3](2026-11-06-e-phase3.md), the session on `combined-v1`, with Claude Fable 5.1
instead of Claude Opus 5.5 as the programmer. Everything else stays: the subject, the
training-side practice set, the held-out set, budget 10 and the keep rule. The programmer was a
subagent that read only a brief, a pack of phase-0 readings, the practice list and the harness
code. It did not see session 2. Its pre-registered hypotheses, in order:

- the blank bias in steps of −0.20, predicted −0.1 to −0.5 points;
- a half-way soup with the clean sibling, predicted −0.2 to −0.6;
- follow-ups: the space bias, the `"` row from the parent, and a quarter-step toward the
  parent. For the last it named the confound that the parent was trained on the Hertwig
  practice lines.

## Done

- **Attempt 0, discarded.** The first programmer subagent read a sentence in the
  pre-registration that named session 2's kept edit. Its files were set aside unrun, and a
  fresh subagent started again.
- **The session:** three programs in rounds, nine passes and eight edits. Three were kept:
  blank −0.20 twice, then every tensor 12.5 % toward the parent. The 25 % step had narrowly
  failed the rule just before.
- **After freeze:** held-out scored once, C5 with 5 seeds, and a split of the kept diff into
  its blank and interpolation parts.

## Found

| held-out CER (Δ points, page-clustered 95 %) | all | Hertwig | Schlussredaktion |
|---|---|---|---|
| Fable (this session) | 0.1875 (−0.57 [−1.15, −0.05]) | −1.32 [−2.36, −0.60] | +0.02 [−0.41, +0.52] |
| Opus (session 2) | 0.1948 (+0.15 [−0.29, +0.60]) | −0.30 [−0.82, +0.21] | +0.51 [−0.04, +1.08] |

- **A held-out gain.** Its page interval excludes 0, narrowly. The edit beats session 2's model
  by −0.73 [−1.19, −0.35].
- **The shams do nothing:** held-out −0.12 to +0.02, and none was kept on practice.
- **The gain is the interpolation's**, −0.53 alone. The two blank steps were kept on practice
  but give +0.04 on held-out.
- **Both programmers found the same direction**, toward the parent; both rejected the sibling.
  Opus took the parent's whole output layer and traded one hand for the other. Fable took a
  small step of the whole network and recovered Hertwig without cost to Schlussredaktion. That
  matches the size and place of retraining on the corrected data
  ([C4](2026-11-08-b-c4-retrain.md)).

## Changes

Phase 3's conclusion, that the label-noise damage is not removable by weight edits, now holds
only for transplants. A small parent-ward interpolation removes about what C4 removes. The
practice set still over-rewards the parent: the Hertwig practice gain (−2.73) was twice the
held-out one. So [A3](../amendments.md#a3) stands as a reason to move practice off the
training side. One replicate each, so this is a difference between two runs, not between two
models.

## Registration and serving path (15:08 and 22:14–22:39 CET)

At the user's request the frozen model was registered as `kraken-bern-combined-wsp-v1`, like
the `latin_incunabula` candidates: exported, weights copied to the trainer's folder on the
share, registry record written with `enabled: false`. Later the same evening it was enabled for
the evaluation only, the 14 held-out pages were read through the gateway with the unedited
model in the same run, and it was disabled again.

Through the gateway the candidate is better by 2.16 points (character-weighted, page interval
[−3.32, −1.05]), on 11 of 14 pages. The surprise is where: every Schlussredaktion page improves
and Hertwig does not move, the reverse of the line-level result. The gateway segments the pages
itself, so the two measurements read different line crops. Why the edit helps on those crops
is not yet tested.
