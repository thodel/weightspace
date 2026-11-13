# 2026-11-11 · Phase 5 with a Fable programmer

**Phase:** 5 (replicate) · **Session:** `20261110-kurrent-page-v2-f1-b10-1` · **Held-out touched:** once ·
**Source:** `docs/PHASE5_FABLE_REPORT.md` · Replicates the [Opus session](2026-11-08-c-adapter-heldout.md)

## Plan

The [phase 5 session](2026-11-08-a-adapter-session-v2.md) was repeated with Claude Fable 5.1
as programmer, so that its result is not tied to one model. Everything else stayed the same:
subject, 64 practice pages, 733 held-out pages, budget 10, engine and keep rule. The
programmer, a Fable subagent, read only a brief, the budget-0 reading, C0 and the harness
source. It did not read the Opus session. An Opus 5.5 operator ran its programs round by
round, then did the held-out score and the controls.

## Found

- **The same edit:** Fable kept one edit of four, direction 0 of layer 31 `o_proj`, removed.
  That is the edit Opus kept, and the frozen adapter is byte-identical.
- **The other edits were refused:**
  - the late `down_proj` pair, which Opus tried too;
  - the module's next two directions;
  - late attention × 0.5.
- **The budget:** Fable froze at 5 of 10 passes.
- **Held-out:** all 733 readings are byte-identical to the Opus session's. The result is
  +0.24 points [−3.5, +3.8] by pages and [−5.0, +6.7] by documents, so it is not a second
  data point. The engine is deterministic at full scale.
- **Two new seeded shams** (a random direction out of the same module) against C0:

  | sham seed | Δ against C0 | at the cap |
  |---|---|---|
  | 20261110 (new) | **−5.7 [−11.8, −1.2]** | 9 → 7 |
  | 20261111 (new) | −1.3 [−5.6, +2.6] | 9 → 9 |
  | 20261108 (phase 5) | −3.9 [−9.1, +0.1] | 9 → 7 |

  **One random direction passes the keep rule.** H1's interval against a sham excludes 0
  for one seed of three.

## What it says

The two programmers agree because both read the same lens table and its top row decides.
That makes the agreement a statement about the reading, not independent evidence for the
edit. The step that fails, from practice to held-out, is the same for both because the
artefact is the same. On 64 pages, the keep rule cannot tell the lens direction from a
random one.

## Changes

- Harness:
  - pass numbering continues across a session's programs;
  - the account's home is found from `atr-cache`, so a worktree runs.
- No change to the rules. The case for a noise floor of several seeded shams per pass, in
  page-unit sessions, is stronger.
