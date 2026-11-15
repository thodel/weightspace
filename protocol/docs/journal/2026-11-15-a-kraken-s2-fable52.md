# 2026-11-15 · Kraken session 2 with a Fable 5.2 programmer

**Type:** replicate (another programmer model) · **Session:** `runs/20261115-bern-combined-v1-f2-b10-1` ·
**Budget:** 10 (8 spent) · **Held-out touched:** once, after freeze ·
**Source:** `docs/KRAKEN_S2_FABLE52.md` · **Result:** [R3](../results.md#r3)

## Plan

Repeat the session on `combined-v1` a third time, now with Claude Fable 5.2 as the programmer.
The two earlier runs are [phase 3](2026-11-06-e-phase3.md) (Opus 5.5) and the
[Fable 5.1 replicate](2026-11-11-b-kraken-s2-fable.md). Everything else is kept the same:
subject, donors, training-side practice set, held-out set, budget 10, keep rule. The model is
pinned to `claude-fable-5-2` and verified from the transcript, because the alias `fable`
resolves to either version.

The operator's knowledge of the two earlier runs is kept out of every file the programmer
reads. The brief now lists the files of the run folder one by one, and each round the
programmer confirms it saw no earlier edit or result.

## Done

- **The programmer could not run as a subagent.** The agent file pinning the model was written
  during the operator's session, and agent types load only when a session starts. The
  programmer ran instead as a headless Claude Code session (`claude -p --model
  claude-fable-5-2`), resumed between rounds, with the same brief and tools.
- **The model was verified.** Every assistant message in that session's transcript carries
  `claude-fable-5-2`. This was checked before the first pass and again after each round.
- **The session ran in three rounds, with 8 passes and 7 edits.** Its pre-registered order:
  - a half-way soup with the sibling;
  - a quarter step toward the parent, naming the confound that the parent was trained on the
    Hertwig practice lines;
  - a blank step of −0.30.

  The second and third edits were kept. The follow-ups were all refused: a second quarter step
  on either half of the network, a 20 % sibling step, and a second blank step. The programmer
  left two passes unspent and said why.
- **After the freeze:** held-out scored once, C5 shams with 5 seeds, and a split of the kept diff.

## Found

| held-out Δ vs C0, points, page-clustered 95 % | all | Hertwig | Schlussredaktion |
|---|---|---|---|
| Opus 5.5 (output layer from the parent) | +0.15 [−0.29, +0.60] | −0.30 | +0.51 |
| Fable 5.1 (α 0.125 → parent, blank −0.40) | −0.57 [−1.15, −0.05] | −1.32 | +0.02 |
| Fable 5.2 (α 0.25 → parent, blank −0.30) | −0.65 [−1.64, +0.24] | −2.35 [−3.51, −1.40] | +0.69 [+0.19, +1.33] |

- **Same line, twice the step.** Fable 5.2's kept diff has cosine +1.000 with Fable 5.1's and
  twice its norm. Fable 5.1 had also tried α 0.25 first, and the rule refused it narrowly;
  here the rule kept it.
- **Not a gain under A4.** The larger step doubles the Hertwig gain and costs Schlussredaktion
  0.69 points. The net is the same as Fable 5.1's, but the page interval includes 0.
- **The blank step does nothing on held-out** (+0.01 alone), as in f1. The programmer predicted
  this before the freeze: most of the step's practice gain sat on unreadable lines.
- **The shams do nothing** (+0.05 to +0.22 on held-out), and none was kept on practice.

## Changes

The direction holds across three programmers: toward `hertwig-v2`, never toward the sibling,
and with blank steps that do not transfer. The step size did not hold, and with it whether the
result passes A4. Practice on the training side rewards a bigger parent step on Hertwig
(−4.05) and does not show its Schlussredaktion cost, so this is again an argument for
[A3](../amendments.md#a3).

There is one run per model. Two runs of the same model family proposed the same α and got
opposite keep decisions, so a model effect cannot be told apart from run-to-run variation.
Nothing was registered.
