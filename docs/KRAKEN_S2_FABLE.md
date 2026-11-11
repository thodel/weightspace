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
