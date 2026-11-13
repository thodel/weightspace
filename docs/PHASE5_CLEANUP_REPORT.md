# Phase 5 report — clean-up: the collapse detector and a handwriting held-out set (asteraix, 07.11.2026)

> Restored 10.11.2026. This report was `docs/PHASE5_REPORT.md` until commit `003bbda` (08.11.) wrote the adapter report under the same name; that file is now the adapter report only.

Branch `phase5` from `main` after PR #7 (phase 4) was merged (`23e8dc5`). Two clean-ups that the phase-4 findings made necessary before any further family-2 number. CPU only, no forward pass (every number below re-aggregates readings already committed), nothing written outside this repo.

## 1. The collapse detector

`wsp.metrics.is_collapsed` now counts a repetition loop only when the most frequent character trigram occurs at least six times **and** at least twice as often as in the reference, plus two (`degenerate_repeat`). The old test (any trigram six times) flagged correct long lines; it remains available as `is_collapsed(..., legacy=True)`, and `wsp.collapse` keeps using it so its false-positive class stays measurable. PROTOCOL.md §5 records the change and the counts of every committed baseline under both tests:

| set | model | legacy | current |
|---|---|---:|---:|
| blb-eval | german-xix-blb-v1 | 428 | 286 |
| blb-eval | german-xix-v1 | 455 | 436 |
| blb-train-300 | german-xix-blb-v1 | 10 | 9 |
| every Bern set | every family-1 model | | unchanged |

Session decisions made under the legacy test (phases 1 and 3) stand as recorded. Tests: 40 pass (a correct long line is not collapsed; a real loop is; the legacy flag reproduces the old behaviour).

## 2. `blb-eval-hand`, family 2's held-out set from now on

`scripts/build_blb_eval_hand.py` keeps the handwriting part of `blb-eval` by data properties only; no rule uses a model's output, because choosing test lines by how a model reads them would bias every number measured on them. The object rules come from looking at one page of every one of the 33 objects:

| excluded | lines | why |
|---|---:|---|
| K 2913,17 | 2,825 | printed book (1853); its reference is the library's OCR of the print |
| K 3123,57 | 372 | typescript (1955) |
| K 2911,129 | 21 | printed government form with handwritten entries |
| K 2912,66 | 12 | music notation, words under the staves |
| K 2915-3 p. 10 | 12 | outside of a folded letter, addresses in several orientations |
| crop taller than wide | 103 | vertical lines (envelopes, margins) |
| reference of 1–3 characters | 9 | page numbers, signs |
| box over 2.5 × the page's median line height | 2 | boxes holding two lines |

Kept: **2,272 lines, 29 objects** (sha256 `a784ae08…`), audited clean against every practice list. It is a subset of `blb-eval` with the same crops, so numbers re-aggregate exactly (`docs/baselines/heldout-blb-eval-hand.json`):

| model | CER | WER | collapsed | lassberg | korrespondenten | unbestimmt | manuskripte |
|---|---:|---:|---:|---:|---:|---:|---:|
| `german-xix-blb-v1` | **9.55 %** | 28.4 % | 6 | 5.0 % | 15.2 % | 14.8 % | 8.3 % |
| `german-xix-v1` (parent) | 23.86 % | 66.3 % | 154 | 19.5 % | 25.6 % | 44.0 % | 21.5 % |

For comparison: on `blb-eval` as cut the subject reads 12.34 %; the trainer recorded 8.7 % on its own compile of the eval side. The phase-1 session's frozen artefact, re-aggregated on `blb-eval-hand` (a re-aggregation of its one held-out reading, not a second evaluation): 9.55 % → 9.55 %, Δ −0.001 points, interval [−0.055, +0.053]. Still not a gain.

## 3. Still open

- The data defects themselves (box heights, orientation, the printed book and the typescript in the handwriting split, the Transkribus outline/text mismatches) are for the dataset owners (docs/PHASE4_REPORT.md §6).
- The next step proposed on 07.11.2026 is PLAN.md phase 5, the adapters; it needs room on a GPU.
- PR #8 (block statistics) also edits `wsp/metrics.py` (`score`, `Metrics`), not `is_collapsed`; the two should merge without conflict.
