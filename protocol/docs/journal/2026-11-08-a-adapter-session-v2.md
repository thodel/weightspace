# 2026-11-08 · Phase 5 adapter session: one direction of one attention output

**Phase:** 5 · **Session:** `20261107-kurrent-page-v2-b10-1` ·
**Budget:** 10 (6 spent) · **Held-out touched:** scoring in progress ·
**Source:** the run folder's `preregistration.md` and `notes.md` (branch `phase5`, not yet pushed)

!!! info "Open entry"
    The session is frozen; its one held-out scoring was still running when this
    entry was written. The held-out result will be a new entry, not an edit of
    this one. **Outcome:** [no held-out gain](2026-11-08-c-adapter-heldout.md).

## Plan

Subject: the same adapter as the [adapter track](2026-11-07-h-adapter-track.md)
(`…-page-v2`). **Only the adapter is edited;** the base is read, never written.
Practice: 64 pages from the registered *validation* side, one per document
(a deliberate deviation from the train-side rule, see [A3](../amendments.md#a3)).
A pass is the trainer's own scorer on these 64 pages (greedy, cap 3,072
tokens). Hypotheses came from a logit-lens reading: which directions of the
adapter's late modules write *against* the end-of-turn token.

## Done

| pass | edit | practice CER | at cap | decision |
|---:|---|---:|---:|---|
| 1 | C0 (byte-identical to the earlier reading on 64/64 pages) | 0.4515 | 9 | baseline |
| 2 | **H1:** direction 0 of layer 31 `self_attn.o_proj` removed from `B` | **0.3470** | 4 | **kept**, [−20.3, −2.4] points |
| 3 | H1b: direction 0 of layers 31 and 27 `down_proj` also removed | | 6 | refused: +4.5 points vs H1 [−2.4, +15.8] |
| 4 | H2: the last five layers' 23 modules scaled to 0.75 | | 5 | refused: +5.6 points vs H1 [+0.3, +13.5] |
| 5 | H3: H1 at dose 1.5 | 0.3616 | 5 | refused |
| 6 | H3b: H1 at dose 0.5 | 0.3990 | 7 | refused |

H1's preregistered prediction: pages at the cap 9 → 5–7, CER −2 to −6
points, clipped CER within ±1 point. Frozen with H1, four passes unused.
Passes 3 and 4 ran in a new process and the harness compared them with the
wrong reference state; the log's deviation note recomputes them against H1
(the numbers above), and the harness now persists its reference states. The
decisions are unchanged.

## Found (practice only)

- H1 met its cap and length predictions; CER fell **far more** than predicted,
  because five pages that ran to the cap now stop (−11,000 of −11,309 edits).
  The clipped CER improved too (0.256 → 0.234), which was not predicted.
- **Dose-response:** CER falls monotonically up to full removal and turns up
  beyond it (0.4515 → 0.3990 → 0.3470 → 0.3616 at doses 0, 0.5, 1, 1.5). That
  argues for a mechanism rather than noise on five pages.
- Per page: 28 better, 16 worse, 20 identical. **A stopping gain resting on few
  pages**; given how chaotic looping is (previous entry), only the held-out set
  can carry it.

## Changes

Pending the held-out score: if the gain holds, the sham control (C5) is owed
and the kept state's pass should be repeated before the gain is reported.
