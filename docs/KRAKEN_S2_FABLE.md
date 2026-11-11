# Kraken session 2 with Claude Fable 5.1 as programmer

Run `runs/20261111-bern-combined-v1-f1-b10-1`, replicating `runs/20261106-bern-combined-v1-b10-1`
(Claude Opus 5.5). Times CET unless labelled; logs are UTC.

## 1. Setup and what the operator knew (written 11.11.2026, before the first pass)

The operator is a Claude Opus 5.5 session. Before the first pass it had read session 2's
pre-registration, PROTOCOL.md and protocol/docs/amendments.md. From those it knows that
session 2 kept an output-layer transplant from `hertwig-v2` that lost on held-out (amendment A3).
It had not read session 2's log, notes, program or summary.

**Attempt 0, discarded.** The first version of the run's `preregistration.md` §7 stated that
knowledge. The programmer is allowed to read that file, and the first Fable subagent did read it
while writing its hypotheses, and said so. Its round-0 files (hypotheses and `program_01.py`)
were set aside before any pass. Section 7 was rewritten to point here; this file is outside
what the programmer may read. A fresh Fable subagent then started from the same brief.

## 2. The session (11.11.2026, 09:56–10:18 CET)

Same subject, practice set (`bern-train-300`, training side), held-out set (`bern-eval`), budget
(10) and keep rule as session 2. The programmer was Claude Fable 5.1, a Claude Code subagent
with `model: fable`. It read only `PROGRAMMER_BRIEF.md` and the files it lists: a pack of
phase-0 readings, describe/diff of subject, parent and sibling, C0/C1, the practice list and the
harness source. It wrote §5 of the pre-registration and three programs in rounds. In each round
it reported that nothing it had read named an edit or result of session 2. The operator ran each program in the sandbox.
One difference from session 2: there, one `program.py` was written in full in advance; here
later programs were written after seeing earlier results.

| pass | hypothesis | edit | practice CER | Δ vs kept, line 95 % | decision |
|---:|---|---|---:|---|---|
| 1 | C0 | — | 0.1456 | | baseline |
| 2 | H1 | blank bias −0.20 | 0.1444 | −0.12 [−0.23, −0.02] | **kept** |
| 3 | H2 | all tensors α 0.5 toward the sibling | 0.1429 | −0.16 [−0.76, +0.33] | refused |
| 4 | H1b | blank bias −0.20 more | 0.1410 | −0.34 [−0.55, −0.16] | **kept** |
| 5 | H3 | space bias +0.30 | 0.1426 | +0.16 [−0.03, +0.38] | refused |
| 6 | H6 | `"` output row from the parent | 0.1410 | 0.00 (identical output) | refused |
| 7 | H4 | all tensors α 0.25 toward the parent | 0.1357 | −0.53 [−1.16, +0.02] | refused |
| 8 | H1c | blank bias −0.20 more | 0.1397 | −0.13 [−0.33, +0.05] | refused |
| 9 | H4b | all tensors α 0.125 toward the parent | 0.1357 | −0.54 [−1.04, −0.13] | **kept** |

Nine of ten passes were used, with 8 edits and 3 kept. The frozen model is the subject with the
blank bias lowered by 0.40 (+0.998 → +0.598), then every trained tensor moved 12.5 % of the way
toward `hertwig-v2`. Output rows were aligned by character; the rows of `½ ƒ ̊` and the blank
bias were held. H4b is the pre-registered follow-up of H4, which narrowly failed the rule, so
two step sizes were tried on the parent axis.

## 3. Held-out, scored once (11.11.2026, 10:18 CET)

Intervals are paired, 2,000 resamples. "page" is the two-stage page-clustered interval (A4), in
CER points. C0 is the unedited subject.

| | practice | held-out all | held-out Hertwig | held-out Schlussred. |
|---|---:|---:|---:|---:|
| C0 | 0.1456 | 0.1933 | 0.2669 | 0.1348 |
| **Fable (this run)** | 0.1357 | **0.1875** | 0.2536 | 0.1350 |
| Δ, line | −0.99 [−1.65, −0.47] | −0.57 [−0.88, −0.29] | −1.32 [−1.91, −0.80] | +0.02 [−0.28, +0.31] |
| Δ, **page** | −0.99 [−1.80, −0.36] | **−0.57 [−1.15, −0.05]** | −1.32 [−2.36, −0.60] | +0.02 [−0.41, +0.52] |
| Opus (session 2) | 0.1414 | 0.1948 | 0.2639 | 0.1399 |
| Δ, page | −0.42 [−0.93, +0.04] | +0.15 [−0.29, +0.60] | −0.30 [−0.82, +0.21] | +0.51 [−0.04, +1.08] |
| Fable − Opus, page | −0.57 [−1.17, −0.05] | −0.73 [−1.19, −0.35] | | |

Collapsed held-out lines: C0 34, Fable 23, Opus 23 (current detector).

The Fable edit is a held-out gain whose page interval excludes 0. It lies wholly on the
Hertwig pages: −1.32 points, with Schlussredaktion unchanged. That is the size and place of the
C4 retrain on corrected data (R11 arm A: −0.54 to −0.56 all, −1.30 Hertwig).

## 4. Controls

`runs/…-f1-b10-1/controls.json`, `scripts/sham_combined_f1.py`, CPU. Each control is scored
against C0.
- **C5 sham:** 5 seeds. Every edited tensor gets Gaussian noise with the Frobenius norm of the
  kept diff of that tensor, 10.68 in all.
  - Practice: +0.03 to +0.14 points. The keep rule kept 0 of 5.
  - Held-out: −0.12 to +0.02. Every page interval contains 0.
  - So the direction of the edit matters, not its size.
- **Split of the kept diff:**
  - Blank −0.40 alone: practice −0.46 [page −0.81, −0.17], held-out **+0.04** [−0.18, +0.25].
  - Interpolation alone: practice −0.74 [−1.45, −0.20], held-out **−0.53** [−1.14, −0.01].
  - The held-out gain comes from the step toward the parent. The two blank steps, both kept on
    practice, do not transfer. This agrees with R8(c): the deletions are decided upstream of
    the output layer.
- **C0, C1:** `docs/baselines/` (unchanged since phase 0).
- **C6:** the phase-0 readings.
- **C4:** R11.

## 5. The two programmers compared

- **Same direction.** Both kept edits that move the subject toward the parent, `hertwig-v2`:
  - Opus transplanted the parent's whole output layer.
  - Fable moved every tensor 12.5 % of the way there.
  - Both tried the clean sibling, and in both sessions the keep rule refused it: Opus block by
    block, Fable as a half-way soup.
  - Both lowered the blank bias. Opus's single step to the parent's value was refused. Fable
    took two steps of −0.20, both kept, and they did nothing on held-out.
- **Different outcome.**
  - The full output-layer transplant traded Schlussredaktion for Hertwig: +0.51 and −0.30 on
    held-out, net +0.15, not a gain.
  - The small whole-network step kept Schlussredaktion and recovered 1.3 Hertwig points: net
    −0.57, page interval excluding 0. It beats Opus's frozen model by −0.73 [−1.19, −0.35].
- **Why the training-side practice set still misleads.**
  - Fable's practice gain on Hertwig lines (−2.73) is twice its held-out gain there (−1.32).
  - The parent was trained on those 103 lines, so a parent-ward edit is over-rewarded on
    practice, as in session 2 (amendment A3). Fable stated this confound in its
    pre-registration and notes, and chose a small step rather than a block transplant.
  - What survives to held-out is the part of the parent direction that undoes the label noise:
    the same Hertwig-only gain that retraining on corrected images buys (R11).
- **Limits.**
  - One replicate per programmer.
  - Fable worked in rounds and Opus in one program.
  - Fable tried two step sizes on the parent axis.
  - The page interval's upper end (−0.05) is close to 0.
  - On validation-side practice (R7), five attempts by other programmers moved toward the
    sibling and none gained by the page interval. No attempt there moved toward the parent.
    The parent was trained on every Hertwig validation line, so under the donor rule a
    parent-ward edit could be judged only on Schlussredaktion lines.
