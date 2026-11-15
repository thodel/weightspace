# Kraken session 2 with Claude Fable 5.2 as programmer

Run `runs/20261115-bern-combined-v1-f2-b10-1`, the third replicate of kraken session 2 on
`bern-combined-v1`, after `runs/20261106-bern-combined-v1-b10-1` (Claude Opus 5.5) and
`runs/20261111-bern-combined-v1-f1-b10-1` (Claude Fable 5.1, docs/KRAKEN_S2_FABLE.md). Times CET
unless labelled; logs are UTC.

## 1. Setup and what the operator knew (written 15.11.2026, 09:35 CET, before the first pass)

The operator is a Claude Opus 5.5 session (`claude-opus-5-5`). Before the first pass it had read
PROTOCOL.md, protocol/docs/amendments.md, the f1 run's `preregistration.md` and
`PROGRAMMER_BRIEF.md` (including the f1 programmer's §5 and rounds) and docs/KRAKEN_S2_FABLE.md
in full. So it knew both earlier outcomes: Opus kept an output-layer transplant from `hertwig-v2`
that gained nothing on held-out (+0.15 points); Fable 5.1 kept two blank-bias steps (−0.40 in all)
and a 12.5 % step of every tensor toward `hertwig-v2`, −0.57 points on held-out (page interval
−1.15 to −0.05), all of it from the parent step on Hertwig lines.

None of this is in any file the programmer may read. The run's `preregistration.md` §1–4 and §6–7
name no edit and no result of an earlier session; they say only that earlier sessions on this
subject exist, as the brief does. The brief restricts the programmer to named files of its run
folder (not every file there), so operator outputs (`stdout_*`, `summary.json`) are outside its
reading list; the stdout of its own programs is handed to it each round. The pack is a byte-for-byte
copy of the f1 pack (budget-0 readings of phase 0, describe/diff, C0/C1), the same material both
earlier programmers had. Two exposures are the same as in f1 and are left as they are: the phase-0
readings list three untested ideas for this subject (blank bias, output-layer transplant, α along
parent→subject), and a docstring in `wsp/run.py` names session 2's run id for a harness bug in
`keep()`'s return value, with no edit or result.

**Programmer and how its model was pinned.** The programmer is Claude Fable 5.2,
`claude-fable-5-2`. The planned route was a Claude Code subagent of type `wsp-programmer-f2`
(`.claude/agents/wsp-programmer-f2.md`, frontmatter `model: claude-fable-5-2`). The first spawn
was stopped by the permission check until the user allowed it. The second failed because agent
types are loaded when a session starts, and this file was written during the operator's
session. The programmer therefore ran as a headless Claude Code session with the same
instructions, brief and tools:
`claude -p --model claude-fable-5-2 --session-id d89d5bf5-… --allowedTools Read Write Edit Bash
--disallowedTools Agent WebFetch WebSearch`, with no `--fallback-model`. It was one session for
the whole run, continued between rounds with `--resume`.

**Verification.** In the session transcript (`~/.claude/projects/…kraken-s2-fable52-923f0b/d89d5bf5-….jsonl`,
not committed), every assistant message has `message.model = claude-fable-5-2`: 22 of 22 after
round 0, before the first pass, and 34 and 45 after rounds 1 and 2. The `modelUsage` of each
round lists only `claude-fable-5-2`, and the programmer gave that id itself. Its tool calls
touched only the brief, the pack, the harness source, the practice list, and its own run
files. In each round it confirmed that nothing it read named an edit, a kept decision or a
result of an earlier session. It noted two near-mentions, both the same as in f1: the `run.py`
docstring and the phase-0 ideas. This route is a deviation from "subagent" and is recorded in
`summary.json`.

## 2. The session (15.11.2026, 09:49–09:57 CET)

| pass | hypothesis | edit | practice CER | Δ vs kept, line 95 % | decision |
|---:|---|---|---:|---|---|
| 1 | C0 | — | 0.1456 | | baseline |
| 2 | H1 | all tensors α 0.5 toward the sibling | 0.1445 | −0.11 [−0.73, +0.38] | refused |
| 3 | H2 | all tensors α 0.25 toward the parent | 0.1381 | −0.76 [−1.47, −0.15] | **kept** |
| 4 | H3 | blank bias −0.30 | 0.1360 | −0.20 [−0.37, −0.04] | **kept** |
| 5 | H4 | second α 0.25 toward the parent, convs + `L_12` only | 0.1398 | +0.38 [+0.14, +0.64] | reverted (collapsed 21 → 22) |
| 6 | H5 | second α 0.25 toward the parent, `L_14 L_16 O_18` only | 0.1383 | +0.23 [−0.10, +0.58] | refused |
| 7 | H6 | all tensors α 0.2 toward the sibling | 0.1347 | −0.13 [−0.36, +0.10] | refused |
| 8 | H7 | blank bias −0.30 more | 0.1345 | −0.16 [−0.36, +0.01] | refused |

The session used 8 of 10 passes and made 7 edits, of which 2 were kept. Passes 9–10 were left
unspent on purpose. The programmer judged the remaining candidates either likely to be refused
or, if kept, not credible. The frozen model is a quarter step of every trained tensor toward
`hertwig-v2`, with output rows aligned by character and the rows `½ ƒ ̊` held. After that comes
a blank-bias step of −0.30 (+0.998 → +0.668; the interpolation itself moved the blank by
−0.030). The final tensor digest is `c8f3367d…` and the file sha256 `8db74d76…`.

What the programmer stated before seeing held-out:
- In §5 and its notes it named the confound: the parent was trained on the 103 Hertwig practice lines.
- It predicted that H3's gain would not transfer. 15 of H3's 22 saved edits are on the 42
  practice lines that are unreadable at C0.
- It called H3 the first candidate for the split control.

## 3. Held-out, scored once (15.11.2026, 09:57 CET)

All deltas are in CER points against C0. "line" is the paired per-line interval; "page" is the
two-stage page-clustered interval (A4). Each uses 2,000 resamples with seed 20261106.
`runs/…-f2-b10-1/comparison.json`, `scripts/compare_combined_f2.py`.

| | practice | held-out all | held-out Hertwig | held-out Schlussred. |
|---|---:|---:|---:|---:|
| C0 | 0.1456 | 0.1933 | 0.2669 | 0.1348 |
| **Fable 5.2 (this run)** | 0.1360 | **0.1868** | 0.2434 | 0.1418 |
| Δ, line | −0.96 [−1.74, −0.32] | −0.65 [−1.04, −0.27] | −2.35 [−3.07, −1.71] | +0.69 [+0.32, +1.09] |
| Δ, **page** | −0.96 [−1.91, −0.12] | **−0.65 [−1.64, +0.24]** | −2.35 [−3.51, −1.40] | +0.69 [+0.19, +1.33] |

Collapsed held-out lines: C0 34, this run 23.

**Under A4 this is not a held-out gain.** The point estimate (−0.65) is the largest of the three
runs, but the page interval includes 0. The run made two opposite changes, and each has a page
interval that excludes 0. Hertwig gained 2.35 points. Schlussredaktion lost 0.69 points.

## 4. Controls

`runs/…-f2-b10-1/controls.json`, `scripts/sham_combined_f2.py`, run on CPU after the freeze.
The setup is the same as f1 (§4 of docs/KRAKEN_S2_FABLE.md), and every control is scored
against C0.

- **C5 sham:** 5 seeds, Gaussian noise per tensor with the Frobenius norm of that tensor's
  kept diff (21.34 in total).
  - Practice: +0.06 to +0.44. The keep rule would keep 0 of 5.
  - Held-out: +0.05 to +0.22, and every page interval contains 0.
  - So the direction of the edit matters, not its size.
- **Split of the kept diff:**

  | part | practice, page | held-out, page |
  |---|---|---|
  | blank −0.30 alone | −0.30 [−0.56, −0.08] | **+0.01** [−0.15, +0.18] |
  | interpolation alone | −0.76 [−1.70, +0.04] | **−0.63** [−1.62, +0.29] |

  The whole held-out change comes from the step toward the parent. The blank step was kept on
  practice and does nothing on held-out, as the programmer predicted. It is the same pattern as
  f1's two blank steps (+0.04).
- **Not run:**
  - C0, C1 and C6: as in the pre-registration, from `docs/baselines/` and the phase-0 readings.
  - C4: needs a trainer job.
  - The serving path: it would need the frozen model registered on the share. Nothing was
    registered, so there are no serving-path numbers for this run.

## 5. Three programmers compared

The intervals are from `comparison.json`. Opus 5.5 is `runs/20261106-bern-combined-v1-b10-1`;
Fable 5.1 is `runs/20261111-bern-combined-v1-f1-b10-1`. Rerun here, their numbers equal those in
docs/KRAKEN_S2_FABLE.md exactly.

| Δ vs C0, page-clustered | Opus 5.5 | Fable 5.1 | Fable 5.2 |
|---|---|---|---|
| kept edits | `O_18` transplanted from the parent (new rows kept) | blank −0.20 ×2; all tensors α 0.125 → parent | all tensors α 0.25 → parent; blank −0.30 |
| passes / edits / kept | 9 / 8 / 1 | 9 / 8 / 3 | 8 / 7 / 2 |
| norm of the kept diff | 39.2 | 10.7 | 21.3 |
| practice all | −0.42 [−0.93, +0.04] | −1.00 [−1.80, −0.36] | −0.96 [−1.91, −0.12] |
| practice Hertwig | −1.53 [−2.49, −0.59] | −2.73 [−4.82, −1.39] | −4.05 [−6.46, −2.42] |
| practice Schlussred. | +0.08 [−0.49, +0.61] | −0.20 [−0.90, +0.37] | +0.46 [−0.43, +1.24] |
| held-out all | +0.15 [−0.29, +0.60] | **−0.57 [−1.15, −0.05]** | −0.65 [−1.64, +0.24] |
| held-out Hertwig | −0.30 [−0.82, +0.21] | −1.32 [−2.36, −0.60] | −2.35 [−3.51, −1.40] |
| held-out Schlussred. | +0.51 [−0.04, +1.08] | +0.02 [−0.41, +0.52] | **+0.69 [+0.19, +1.33]** |
| held-out collapsed (C0 34) | 23 | 23 | 23 |

Between runs on held-out:
- Fable 5.2 − Fable 5.1: −0.08 [−0.61, +0.44]. By hand, Hertwig −1.02 [−1.44, −0.63] and Schlussredaktion +0.67 [+0.31, +1.06].
- Fable 5.2 − Opus: −0.80 [−1.60, −0.09].

**Same direction.** All three programmers kept a move toward the parent.
- **The parent step.** Fable 5.2's kept diff has cosine **+1.000** with Fable 5.1's and twice its
  norm. The two kept models lie on the same line from the subject toward `hertwig-v2`, at
  α 0.25 against α 0.125, and the blank-bias parts are too small to change the angle. Each run's
  diff has cosine +0.46 with Opus's output-layer transplant. Fable 5.1 tried α 0.25 first and
  the rule refused it narrowly (−0.53 [−1.16, +0.02]), so it kept α 0.125. Fable 5.2 tried
  α 0.25 first and the rule kept it (−0.76 [−1.47, −0.15]), on another blank state: f1 had
  already lowered the blank by 0.40. It then found that a second quarter step costs more than it
  gains in either half of the network.
- **The sibling.** All three runs tried it and none kept it: half-way soups in both Fable runs,
  and α 0.2 in this one.
- **The blank bias.** Opus's single step to the parent's value was refused. Fable 5.1 kept
  −0.40 and Fable 5.2 kept −0.30. Neither transfers to held-out: +0.04 and +0.01 alone.
- **Ideas only Fable 5.1 had:** the space bias and the `"` row. Fable 5.2 had dropped
  single-row bias edits after reading its pass-1 confusions.

**Different dose, different trade.** On the line from the subject toward the parent, the
held-out effect of the dose looks like this:

| step toward the parent | Hertwig | Schlussredaktion | net |
|---|---|---|---|
| α 0.125 (Fable 5.1) | −1.3 | 0 | −0.57, page interval excludes 0 |
| α 0.25 (Fable 5.2) | −2.35 | +0.69 | −0.65, page interval includes 0 |
| whole output layer (Opus) | −0.30 | +0.51 | +0.15 |

So doubling the step doubles the Hertwig gain but begins to cost Schlussredaktion. The net is
about the same, and the page interval gets wider. The practice set rewards the larger dose on
Hertwig, where the gain is −4.05 on lines the parent was trained on, against −2.35 on held-out.
It does not show the Schlussredaktion cost: +0.46 on practice, with an interval that includes 0.
That is the training-side confound of A3 again. Both Fable runs named it in their
pre-registration.

**What one run per model cannot say.** These are three runs, one per programmer model. They
cannot separate a model effect from run-to-run variation. Two things in this run show that
variation directly:
- Fable 5.1 and 5.2 proposed the same α 0.25 step. One was refused and one was kept, a
  difference of a few hundredths of a point at the upper end of the line interval, with a
  different blank state underneath.
- Fable 5.1's α 0.125 is the follow-up its refusal led to.

The difference between the two Fable runs on held-out is −0.08 [−0.61, +0.44]. That is no
difference overall, and an opposite one on each hand. The stable finding across the three runs
is the direction: toward `hertwig-v2`, never the sibling, with blank steps that do not transfer.
The size of the step, and with it whether the result passes A4, varied between runs.

## 6. Limits

- One run per programmer model, and practice is on the training side.
- The held-out set is 14 pages from two documents.
- There are no serving-path numbers, because nothing was registered.
- The programmer was a headless Claude Code session, not a Task subagent (§1). It had the same
  brief and tools, and its model was verified from its transcript.
