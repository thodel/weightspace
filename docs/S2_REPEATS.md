# Kraken session 2, repeated: four runs of Claude Fable 5.2

Pre-registration: docs/S2_REPEATS_PREREG.md, commit 44396db, made before any run was set up.
Runs: `runs/20261115-bern-combined-v1-f2-b10-1` (b10-1, #44) and the repeats
`runs/20261124-bern-combined-v1-f2-b10-2`, `-3` and `-4`. Times are CET unless labelled UTC;
logs are in UTC.

## 1. Setup and what the operator knew (written 24.11.2026, before the first pass of any repeat)

**The operator** is a Claude Opus 5.5 session (`claude-opus-5-5`). Before the first pass it had
read, in full:
- b10-1's `summary.json`, `preregistration.md` (including the programmer's §5 and rounds),
  `PROGRAMMER_BRIEF.md` and `log.jsonl`;
- docs/KRAKEN_S2_FABLE52.md and docs/XVENDOR_REPORT.md §5;
- RESULTS R3 and PROJECT_STATE item 13;
- the round messages of b10-1, taken from its operator transcript.

It knew all five session-2 outcomes:
- Opus transplanted the parent's output layer (+0.15 on held-out).
- The Fable 5.1, Fable 5.2, Gemini and GPT runs each made a whole-network move with parent weight
  (−0.57 to −0.65).
- b10-1 kept α 0.25 toward the parent and then a blank step of −0.30.

Before the first pass it also computed the across-model quantities of the pre-registration with
`scripts/s2_repeats.py`.

**None of this is in any file a programmer may read.** The three run folders were built from
b10-1:
- **Brief:** b10-1's `PROGRAMMER_BRIEF.md`, with only the run name, the worktree path and the date
  changed.
- **Pre-registration:** b10-1's `preregistration.md` §1–4 and §6–7, as the b10-1 operator wrote
  them before that run's first pass. §5 is left empty and §7's model-verification line is reset to
  its placeholder. §7 describes the headless route that b10-1 actually used, not the subagent it
  had planned.
- **Pack:** a byte-for-byte copy of b10-1's pack, checked with `diff -r`.

The known exposures are the same as in f1 and b10-1, and are left as they are:
- the phase-0 readings in the pack list untested ideas for this subject;
- a `wsp/run.py` docstring names session 2's run id, for a harness bug.

**Harness version.** The harness is at `10450ae`, not b10-1's `2e40b56`. For a kraken line
session with `cluster_rule` false, the only change in behaviour is this: a `keep()` or `revert()`
with no pending edit is now logged as a refusal instead of raising. `session.json` gained the
fields `min_effect` 0.0 and `cluster_rule` false, which are b10-1's values.

**How the programmers were started.** Each run's programmer is a headless Claude Code session
started from the worktree root:

```
claude -p --model claude-fable-5-2 --session-id <new uuid> --permission-mode acceptEdits
  --allowedTools Read Write Edit Bash --disallowedTools Agent WebFetch WebSearch
  --append-system-prompt <b10-1's text> --output-format json <b10-1's round-0 message, run name changed>
```

There is no `--fallback-model`. Each session is resumed with `--resume <id>` between rounds. The
three runs went in parallel; each sandbox pass used 4 threads, as in b10-1.

**Model verification and blindness.** All three transcripts were checked before the first pass
and again after every round. Every assistant message has `message.model = claude-fable-5-2`, and
the `modelUsage` of every round lists only that id. Cumulative message counts were 23, 36, 43
(b10-2), 21, 29, 38, 41 (b10-3) and 23, 34, 40 (b10-4). Each programmer also gave
`claude-fable-5-2` as its own id.

Every tool call was audited, and none touched a file outside the brief's list: no other `runs/`
folder, nothing in `docs/`, `protocol/`, `.claude/` or `data/heldout/`, and no git history. Each
programmer confirmed in every round that nothing it read named an earlier edit, kept decision or
result. Each noted the same two near-mentions as b10-1: the `run.py` docstring and the phase-0
ideas. **No run is exposed.**

There were two small slips, both recorded in the runs' `summary.json` deviations:
- b10-2 wrote its read-only analysis scripts into its CLI scratchpad under `~/atr-cache/tmp`
  rather than its run folder.
- b10-3 first appended a round-2 paragraph after §7 and moved it into §5 at once. Its §1–4 and
  §6–7 were checked unchanged.

## 2. The sessions (24.11.2026, 17:57–18:15 CET)

C0 reproduced exactly in all three runs: CER 0.1456, 1,577 edits on 10,829 characters,
29 collapsed. No run crashed and none was restarted. Deltas are in CER points against the last
kept state, with the line interval of the keep rule.

**b10-2** (7 passes, 6 edits, 2 kept; passes 8–10 unspent)

| pass | edit | practice CER | Δ vs kept | decision |
|---:|---|---:|---|---|
| 2 | all tensors α 0.5 toward the sibling | 0.1445 | −0.11 [−0.73, +0.38] | refused |
| 3 | all tensors α 0.25 toward the parent | 0.1381 | −0.76 [−1.47, −0.15] | **kept** |
| 4 | blank bias −0.70 | 0.1346 | −0.34 [−0.67, −0.06] | **kept** |
| 5 | `L_16`, `O_18` back to the subject | 0.1330 | −0.17 [−0.45, +0.13] | refused |
| 6 | convs + `L_12` back to the subject | 0.1351 | +0.05 [−0.22, +0.35] | refused |
| 7 | late blocks to the subject, front to 0.45 parent | 0.1385 | +0.39 [+0.05, +0.74] | reverted (collapsed) |

**b10-3** (8 passes, 7 edits, 2 kept; passes 9–10 unspent)

| pass | edit | practice CER | Δ vs kept | decision |
|---:|---|---:|---|---|
| 2 | all tensors α 0.30 toward the parent | 0.1381 | −0.76 [−1.50, −0.09] | **kept** |
| 3 | to α 0.50 in total | 0.1490 | +1.10 [+0.61, +1.60] | reverted (collapsed) |
| 4 | `L_16`, `O_18` back to the subject | 0.1384 | +0.04 [−0.28, +0.36] | reverted (collapsed) |
| 5 | convs + `L_12` to 0.55 parent | 0.1444 | +0.64 [+0.30, +0.99] | reverted (collapsed) |
| 6 | blank bias −0.40 | 0.1359 | −0.21 [−0.40, −0.04] | **kept** |
| 7 | + 0.25 × (sibling − subject) | 0.1346 | −0.13 [−0.41, +0.14] | refused |
| 8 | blank bias −0.35 more | 0.1350 | −0.09 [−0.32, +0.09] | refused |

**b10-4** (8 passes, 7 edits, 2 kept; passes 9–10 unspent)

| pass | edit | practice CER | Δ vs kept | decision |
|---:|---|---:|---|---|
| 2 | all tensors α 0.25 toward the parent | 0.1381 | −0.76 [−1.47, −0.15] | **kept** |
| 3 | blank bias −0.25 | 0.1364 | −0.17 [−0.32, −0.02] | **kept** |
| 4 | all tensors α 0.30 toward the sibling | 0.1349 | −0.15 [−0.45, +0.17] | refused |
| 5 | a further α 0.20 toward the parent | 0.1418 | +0.54 [+0.18, +0.93] | reverted (collapsed) |
| 6 | convs + `L_12` a further α 0.25 | 0.1396 | +0.32 [+0.08, +0.58] | reverted (collapsed) |
| 7 | `L_14 L_16 O_18` a further α 0.25 | 0.1385 | +0.21 [−0.11, +0.55] | refused |
| 8 | blank bias −0.25 more | 0.1353 | −0.11 [−0.29, +0.05] | refused |

All four Fable 5.2 runs ended with the same two kept moves, in the same order of kind:
- first a whole-network step toward `hertwig-v2` (α 0.25, 0.25, 0.30, 0.25);
- then a blank-bias step (−0.30, −0.70, −0.40, −0.25).

Two of the repeats also opened with a sibling soup, as b10-1 did, and the rule refused it each
time. Every further parent step, whether whole-network or block-wise, raised CER or the
collapsed count. Every further sibling or blank step was refused.

Before held-out was scored, each programmer said the gain would be smaller on held-out. b10-3
predicted −0.6 to +0.3 points (centre −0.2) and b10-4 predicted −1.0 to +0.4. Both
measured values (−0.47 and −0.67) lie inside those ranges.

## 3. Held-out, scored once (24.11.2026, 18:10 and 18:15 CET)

`wsp.run heldout --set bern-eval`; no score came from the cache. Deltas are in CER points
against C0, with page-clustered 95 % intervals (A4; 2,000 resamples, seed 20261106). The
numbers come from `docs/s2_repeats.json` (`scripts/s2_repeats.py`). *a* is the parent
coefficient of the kept diff, blank excluded (pre-registration §4).

| run | kept | *a* | blank | practice Δ, page | held-out Δ, page | Hertwig | Schlussred. | page < 0 |
|---|---|---:|---:|---|---|---:|---:|---|
| b10-1 (15.11.) | α 0.25 → parent, blank | 0.250 | −0.33 | −0.96 [−1.91, −0.12] | −0.65 [−1.64, +0.24] | −2.35 | +0.69 [+0.19, +1.33] | no |
| b10-2 | α 0.25 → parent, blank | 0.250 | −0.73 | −1.10 [−2.11, −0.21] | −0.55 [−1.51, +0.35] | −2.22 | +0.78 [+0.28, +1.39] | no |
| b10-3 | α 0.30 → parent, blank | 0.300 | −0.44 | −0.97 [−1.99, −0.04] | −0.47 [−1.58, +0.57] | −2.50 | +1.14 [+0.64, +1.79] | no |
| b10-4 | α 0.25 → parent, blank | 0.250 | −0.28 | −0.92 [−1.88, −0.08] | −0.67 [−1.64, +0.22] | −2.36 | +0.67 [+0.19, +1.29] | no |

The blank column is the change of `O_18.lin.bias[0]`, including the interpolation's own share of
about −0.03 to −0.04. In every run the Hertwig page interval lies below 0 and the
Schlussredaktion page interval lies above 0.

## 4. Controls

Each control is scored against C0 (`runs/…-b10-{2,3,4}/controls.json`,
`scripts/sham_s2_repeats.py`).
- **C5 shams:** 5 seeds per run, norm-matched per tensor.
  - On practice: −0.09 to +0.44; the keep rule would keep 0 of 15.
  - On held-out: +0.05 to +0.33. One of 15 has a page interval above 0 (b10-3 seed 20261112,
    +0.33 [+0.04, +0.64]); none lies below 0.
- **Component split:** each kept edit alone, from a replay that reached the run's final digest.

  | run | parent step alone, held-out | blank step alone, held-out |
  |---|---|---|
  | b10-2 | −0.63 [−1.62, +0.29] | +0.00 [−0.33, +0.36] |
  | b10-3 | −0.54 [−1.66, +0.50] | +0.04 [−0.18, +0.25] |
  | b10-4 | −0.63 [−1.62, +0.29] | −0.03 [−0.18, +0.13] |
  | b10-1 (for comparison) | −0.63 [−1.62, +0.29] | +0.01 [−0.15, +0.18] |

  The whole held-out change is the parent step's. Three of the four runs chose the same α 0.25,
  which is the same edit and so gives the same −0.63. What differs between those three is only
  the blank step. It was kept on practice every time. Alone it moves held-out by −0.03 to
  +0.04. On top of the parent step it moves held-out by −0.04 to +0.08 against the parent step
  alone (finals −0.67, −0.65, −0.55).

## 5. Spread within and across models, read by the pre-registered rule

| | n | held-out Δ values | range | SD |
|---|---:|---|---:|---:|
| **Fable 5.2, four runs** | 4 | −0.65, −0.55, −0.47, −0.67 | **0.20** | **0.093** |
| five programmers, one run each | 5 | +0.15, −0.57, −0.65, −0.65, −0.57 | 0.80 | 0.344 |
| the four parent-step programmers | 4 | −0.57, −0.65, −0.65, −0.57 | 0.085 | 0.047 |

1. **Model comparison.**
   - *Against all five programmers:* the within-model range (0.20) and SD (0.093) are both smaller
     than across models (0.80, 0.34). So the across-model spread is larger than four runs of one
     model show. That spread is all Opus's +0.15 from a single run. By the rule it is not read as
     an Opus effect, and with four runs the reading is provisional.
   - *Among the four parent-step programmers* (Fable 5.1, Fable 5.2, Gemini 4 Argon, GPT-6.2
     Astra): the within-model range (0.20) is more than twice theirs (0.085), and the SD (0.093)
     is about twice theirs (0.047). **Their differences are not interpretable as model effects.**
     - Fable 5.1's −0.57 lies inside the Fable 5.2 range (−0.67 to −0.47).
     - Fable 5.1 alone passing A4 is therefore not a version or model effect either.
2. **Direction: robust to the run.** All 4 Fable 5.2 runs find the parent step (3 of 3
   repeats). Each has *a* of 0.25–0.30, R² 1.00 and no sibling component. None kept nothing, and
   none kept something else in its place. Every run also kept one blank step, and it never
   transferred.
3. **Passing A4.** 0 of 4 Fable 5.2 runs have a page interval below 0, against 1 of 5 across
   programmers. No run here shows a held-out gain under A4.
4. **Step size** (descriptive). *a* was 0.25 in three runs and 0.30 in one. The α 0.30 run lost
   most on Schlussredaktion (+1.14 against +0.67 to +0.78) and gained least overall (−0.47). Its
   parent step alone gave −0.54, against −0.63 at α 0.25. This fits Fable 5.1 (α 0.125:
   Schlussredaktion +0.02) on one line: a larger parent step buys Hertwig at a rising price on
   Schlussredaktion. Four runs do not support a fitted dose-response.

**What this means.**
- The finding of session 2 is the direction, and it reproduces in every run: give the parent
  weight across the whole network, gain about 2.2–2.5 points on held-out Hertwig, and lose 0.7–1.1
  on Schlussredaktion.
- The net held-out effect, about −0.5 to −0.7 points, varies between runs of one model by more
  than it varies between four different models.
- Whether a run passes A4 is decided by that run-to-run variation (step size, and a blank step
  that does nothing on held-out), not by the programmer model.

## 6. What was not done

- **C4** (trainer), the **serving path** (nothing was registered) and **C1/C6** (unchanged, from
  `docs/baselines/` and the phase-0 readings).
- The parent step at fixed sizes with no programmer (α 0.1–0.3) was done afterwards, in
  docs/S2_PARENT_SWEEP.md: the held-out Δ ranges over 0.21 with α, as much as the four runs here.
