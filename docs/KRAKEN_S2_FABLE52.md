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
