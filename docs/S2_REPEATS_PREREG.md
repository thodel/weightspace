# Kraken session 2, repeated with one programmer model: pre-registration

Written 24.11.2026, 17:55 CET, on asteraix, **before any repeat run is set up or any programmer
is started**. Operator: a Claude Opus 5.5 session (`claude-opus-5-5`). Analysis script:
`scripts/s2_repeats.py`, committed with this file.

## 1. Question

Five programmers have run kraken session 2 (`kraken-bern-combined-v1`, training-side practice
`bern-train-300`, held-out `bern-eval`, 545 lines on 14 pages), one run each
(docs/XVENDOR_REPORT.md §5). Four of them gave the parent `hertwig-v2` weight across the whole
network and gained −0.57 to −0.65 points on held-out. Opus 5.5 transplanted the parent's output
layer and got +0.15. Only Fable 5.1's page interval lies below 0. Without knowing how much two
runs of the same model differ, none of these differences can be read as a model effect
(XVENDOR_REPORT §5 "Limits", PROJECT_STATE open item 13).

This study runs **three more independent sessions with Claude Fable 5.2** (`claude-fable-5-2`)
under the conditions of `runs/20261115-bern-combined-v1-f2-b10-1` (#44, "b10-1" below). With
b10-1, that makes four runs of one model. Their spread is compared with the spread across the five
programmers.

## 2. The runs

- `runs/20261124-bern-combined-v1-f2-b10-2`, `-3` and `-4`.
- Each is a **fresh programmer** with no memory of any other run: a new headless Claude Code
  session with a new session id, which is resumed only within its own run.
- The three run in parallel on the CPU. Each starts from the unedited subject.

## 3. Conditions (the same as b10-1)

- **Subject, donors, practice set, held-out set:** as in b10-1's `preregistration.md` §1–3.
  `bern-eval` is scored once per run after the freeze, with `wsp.run heldout --set bern-eval`.
  - If a frozen model has the same tensor digest as a model already scored, the harness reuses
    that score. Same weights on the same list give the same score, so this is not a second
    evaluation, and it is recorded.
- **Budget 10, pass 1 = C0; keep rule:** as in b10-1 §4. The paired per-line bootstrap 95 %
  interval, with 2,000 resamples and seed 20261106, must lie below 0. The programmer also
  reverts any edit that raises the collapsed count.
- **Each run folder** gets these files from b10-1:
  - b10-1's `preregistration.md` §1–4 and §6–7, with §5 empty and the §7 model-verification
    line reset to its placeholder;
  - b10-1's `PROGRAMMER_BRIEF.md`;
  - b10-1's `pack/`, byte for byte.

  In the brief and the pre-registration only the run name changes, plus what follows from it:
  the worktree path, and the date in the brief's title and §7.
- **Programmer start, as in b10-1** (summary.json → `programmer.how`, pre-registration §7):
  - `claude -p --model claude-fable-5-2 --session-id <new uuid> --permission-mode acceptEdits
    --allowedTools Read Write Edit Bash --disallowedTools Agent WebFetch WebSearch`;
  - the same `--append-system-prompt` and round messages as b10-1, with the run name and dates
    changed;
  - default effort, no `--fallback-model`;
  - resumed with `--resume <its id>` between rounds;
  - cwd is this worktree, as b10-1's cwd was its worktree.
- **Sandbox:** `wsp.run.sandbox_command`, unchanged. That means CPU only, no network, and
  `OMP_NUM_THREADS=4` as in b10-1, so three parallel runs use 12 of 48 cores. The thread count is
  kept at b10-1's value rather than raised to 12, so that C0 and every pass run in b10-1's
  numerical setting.
  - **C0 check:** pass 1 of each run must give CER 0.1456, i.e. 1,577 edits on 10,829
    characters, as in b10-1.
- **Model verification:** done from each run's transcript before the first pass and after every
  round. Every assistant message must have `message.model = claude-fable-5-2`, and `modelUsage`
  must list only that id. If anything else appears, that run is stopped and reported.
- **Blindness:**
  - The programmer reads only its brief and the files the brief lists.
  - Each round it confirms that nothing it read named an earlier edit, kept decision or result.
  - The operator audits every tool call in each transcript for paths outside that list, as for
    b10-1. Examples are another `runs/` folder (b10-1, f1, g1, o1, Opus's run, the other
    repeats), `docs/` (including this file, `docs/XVENDOR_*`, `docs/PROJECT_STATE.md`,
    RESULTS R3), `protocol/`, the README or `.claude/`.
  - A run that read such a file is reported as **exposed**. It is kept in the tables with the
    exposure stated, and is not replaced.
- **Controls per run:**
  - C0 is pass 1.
  - **C5 shams** for any kept edit: 5 norm-matched shams (Gaussian noise per tensor with that
    tensor's kept-diff Frobenius norm, seeds 20261111–15), scored on practice and held-out
    against C0, as in b10-1.
  - **Component split** if more than one edit is kept: each kept edit's increment, meaning the
    change between consecutive kept states, is applied alone to the subject and scored the same
    way. The increments are taken from a replay of the run's own programs in a scratch copy,
    outside the budget. The replay must reach the run's final digest.
- **No budget increases and no reruns** of a run that went badly. A run that crashes is recorded
  as crashed and not replaced. The one exception is a crash that is the operator's fault and
  happens before the first pass, for example a wrong path in the brief or a failed start of the
  CLI. Such a run is restarted with a new session id, and the restart is recorded.
- **Nothing is registered, uploaded or served.** No weights, no `diff.json` (its sha256 is
  recorded in `DIFF_OMITTED.txt`) and no programmer transcripts are committed.

## 4. Quantities

**Per run** (b10-1 recomputed by the same script):
1. **Held-out Δ vs C0** in CER points, with the paired line interval and the page-clustered
   interval (A4; 2,000 resamples, seed 20261106, clusters `<project>/<page>`). These are given
   for all 545 lines, for Hertwig and for Schlussredaktion.
2. **Whether the page interval for all lines lies below 0.**
3. **The kept edits:** pass, operation and tensors, with passes and edits used. Also practice Δ vs
   C0 (line and page).
4. **The step toward the parent.** Let d be the kept diff (final − subject, float64) over all
   tensors, with output rows aligned by character.
   - Rows that either donor lacks are masked out.
   - The blank bias element `O_18.lin.bias[0]` is masked out and reported alone as the blank
     step.
   - d is fitted by least squares on the parent direction (parent − subject) and the sibling
     direction (sibling − subject) in the same layout.
   - Reported: the parent coefficient *a*, the sibling coefficient *b*, R², cos(d, parent
     direction), ‖d‖ and the blank step.
   - **A run "finds the parent step" if *a* ≥ 0.05 and R² ≥ 0.5.** This is fixed here. On the five
     existing runs (computed 24.11.2026 with this script, so known when the thresholds were
     written; the thresholds were chosen to sort "kept a whole-network move with parent weight"
     from "kept something else"):

     | | Opus 5.5 | Fable 5.1 | Fable 5.2 b10-1 | Gemini 4 Argon | GPT-6.2 Astra |
     |---|---:|---:|---:|---:|---:|
     | *a* (parent) | 0.279 | 0.125 | 0.250 | 0.263 | 0.300 |
     | *b* (sibling) | −0.122 | 0.000 | 0.000 | 0.291 | 0.350 |
     | R² | 0.25 | 1.00 | 1.00 | 0.93 | 1.00 |
     | parent step | no (output layer only) | yes | yes | yes | yes |

**Across runs:**
5. **Within-model spread:** over the four Fable 5.2 runs (b10-1 to -4), the range and SD
   (n − 1) of held-out Δ (all lines). Both are also given per hand, as descriptive figures.
6. **Across-model spread:** over the five programmers, one run each with b10-1 as the Fable 5.2
   run, computed 24.11.2026 with this script:
   - **range 0.80, SD 0.34** (Δ = +0.15, −0.57, −0.65, −0.65, −0.57);
   - among the four that found the parent step: **range 0.085, SD 0.047**.
7. **Shares** over the three repeats and over all four Fable 5.2 runs:
   - runs that find the parent step;
   - runs that keep nothing;
   - runs that keep something else (blank-only steps, a sibling move, a partial transplant…);
   - runs whose page interval lies below 0.

## 5. Reading rule (fixed before any repeat run)

1. **Model comparison.**
   - If the within-model range is at least the across-model range (≥ 0.80), or the within SD is
     at least the across SD (≥ 0.34), the comparison of the five programmers is **not
     interpretable** as a model effect. One model's repeats spread as widely as five models do.
   - If both are smaller, the across-model spread is larger than what four runs of one model
     show. That is all it says. Opus's +0.15 is one run, so it is not read as an Opus effect.
   - Separately: if the within range is at least 0.085 (or the SD at least 0.047), the
     differences among the four parent-step programmers are **not interpretable**. That covers
     Fable 5.1 vs 5.2, and which of them passes A4. If both are smaller, those differences are
     at least as large as run-to-run variation.
   - If range and SD disagree, both are reported and the stricter reading (not interpretable)
     is the one stated.
2. **Direction.**
   - If every Fable 5.2 run finds the parent step (4 of 4), **the direction is robust to the
     run**.
   - Otherwise the share is reported (k of 4, and k of 3 among the repeats), with what each other
     run kept or that it kept nothing.
3. **Passing A4.** The share of the four Fable 5.2 runs whose page interval lies below 0 is
   reported next to 1 of 5 across programmers. No claim of a held-out gain is made from a run
   whose page interval contains 0.
4. **Step size.** *a* of the four runs and its range are reported, with each run's Schlussredaktion
   Δ. This is descriptive: four runs do not support a fitted dose-response.
5. **Incomplete runs.**
   - A crashed or stopped run, or one whose model check failed, counts in the shares as such.
   - It is left out of the spread, and the n is stated.
   - With fewer than three completed Fable 5.2 runs, b10-1 included, the spread is not read.
   - An exposed run (§3) is read with the others, and also once without it.

Four runs is a small sample. A range over four values is a lower bound on the model's run-to-run
spread, so a "not interpretable" reading is firm, and an "interpretable" one is provisional.
