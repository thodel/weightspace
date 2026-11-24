# 2026-11-24 · Kraken session 2, repeated with one programmer model

**Type:** replicate (same programmer model, three more runs) · **Sessions:**
`runs/20261124-bern-combined-v1-f2-b10-{2,3,4}` · **Budget:** 10 each (7, 8, 8 spent) ·
**Held-out touched:** once per run, after freeze · **Source:** `docs/S2_REPEATS.md`,
pre-registration `docs/S2_REPEATS_PREREG.md` · **Result:** [R3](../results.md#r3)

## Plan

Five programmers had run session 2 once each. Four moved toward `hertwig-v2` and gained
−0.57 to −0.65 on held-out; Opus got +0.15; only Fable 5.1 passed A4. Without knowing how far two
runs of one model differ, none of that could be read as a model effect.
[The Fable 5.2 run](2026-11-15-a-kraken-s2-fable52.md) was repeated three times under its exact
conditions: brief, pack, pre-registration §1–4/6–7, budget, keep rule and the headless
`claude -p --model claude-fable-5-2` route, each with a new session id. The reading rule was
fixed and committed before any run was set up.

## Done

- **Three fresh programmers ran in parallel on the CPU** (24.11, 17:57–18:15 CET).
  - C0 reproduced exactly in each.
  - Every transcript carries only `claude-fable-5-2`.
  - No tool call left the brief's reading list.
- **Each run's frozen model was scored on held-out once.** Each also got 5 C5 shams and a split of
  its two kept edits, from a replay that reached the final digest.

## Found

| held-out Δ vs C0, points, page-clustered 95 % | kept | all | Hertwig | Schlussred. |
|---|---|---|---:|---:|
| b10-1 (15.11) | α 0.25 → parent, blank −0.30 | −0.65 [−1.64, +0.24] | −2.35 | +0.69 |
| b10-2 | α 0.25 → parent, blank −0.70 | −0.55 [−1.51, +0.35] | −2.22 | +0.78 |
| b10-3 | α 0.30 → parent, blank −0.40 | −0.47 [−1.58, +0.57] | −2.50 | +1.14 |
| b10-4 | α 0.25 → parent, blank −0.25 | −0.67 [−1.64, +0.22] | −2.36 | +0.67 |

- **Same moves every time.** All four runs kept a whole-network parent step and then a blank
  step. Three of them chose the same α 0.25. The blank steps do nothing on held-out alone
  (−0.03 to +0.04).
- **Spread.** Within the four Fable 5.2 runs: range 0.20, SD 0.093.
  - Across the five programmers: range 0.80, SD 0.34.
  - Across the four parent-step programmers: range 0.085, SD 0.047.
- **Passing A4.** 0 of 4 runs.

## Changes

- **The model comparison of session 2 reduces to one statement:** every programmer model that
  takes the parent step lands in the same place. Their differences are smaller than the
  run-to-run spread of one model. Fable 5.1 passing A4 is therefore not a model or version
  effect.
- **The direction is the robust finding.** Whether a run passes A4 is not.
- **Opus's output-layer transplant stays outside the Fable 5.2 spread,** but it is one run.
- **Open item 13 narrows to the parent step at fixed sizes**, with no programmer. Nothing was
  registered.
