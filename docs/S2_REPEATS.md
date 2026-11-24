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
