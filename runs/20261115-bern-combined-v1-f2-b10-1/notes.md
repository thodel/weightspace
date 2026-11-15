# Notes — 20261115-bern-combined-v1-f2-b10-1

## 2026-11-15 08:50 UTC — round 0 (before any pass)

**Programmer:** Claude Fable 5.2, exact model id `claude-fable-5-2` (as stated in my system
prompt). Settings: Claude Code agent, default reasoning effort, tools Read/Write/Edit/Bash; no
sampling settings are visible to me.

**Read:** `PROGRAMMER_BRIEF.md`; `preregistration.md`, `notes.md`, `session.json` of this run;
the pack (`readings_budget0.md`, `c0_c1.md`, `per_line_edits.json`, both `diff_*.md`,
`describe_bern-combined-v1.md`); the head of `data/practice/bern-train-300.jsonl` and its
`.meta.json`; `wsp/ops.py`, `session.py`, `run.py`, `kraken_subject.py`, `subject.py`,
`align.py`, `metrics.py`. Not read: `wsp/data.py`, the `describe_*.json`/`diff_*.json` files,
the line images. Nothing outside the brief's list. `ls` of the run folder showed `env.txt`; I
did not open it.

**Run (CPU, read-only, no forward):** two inline scripts.
- `per_line_edits.json` by hand: reproduces the table of `c0_c1.md`. Per-hand donor switch
  (parent on Hertwig, sibling on Schlussredaktion) 0.1131; per-line best of three 0.1014. The
  42 lines with ≥ 50 % errors hold 736 of 1,577 edits; on the 20 worst lines the subject has
  667 edits, the parent 589, the sibling 643.
- The three models' tensors: cos(subject − parent, sibling − parent) +0.24…+0.48 in convs and
  `L_12`, +0.42…+0.63 in `L_14`/`L_16` weights, +0.85 output weight, +0.96 output bias (rows
  by character). |subject − sibling| / |subject| 0.17–0.29 in conv/LSTM weights, 0.09 in the
  output weight. Blank bias subject +0.998, sibling +0.936, parent +0.879. Rows the sibling
  lacks: `ƒ`; the parent lacks `½ ƒ ̊`. All codec entries are single-label.

**Written:** section 5 of `preregistration.md` (H1 average with the sibling, H2 quarter step
toward the parent, H3 blank bias −0.3; plan for passes 5–10) and `program_01.py` (pass 1 C0
with a stop if it is not 0.1456 / 1,577 edits, then H1, H2, H3, each measured and decided; no
freeze). `py_compile` passes. Not executed.

**Earlier sessions:** nothing I read names an edit, a kept decision or a result of an earlier
session on this model. Two things come close and are recorded here:
- `wsp/run.py` (docstring of `RunSession.keep`) names the session
  `20261106-bern-combined-v1-b10-1` for a harness fault (the wrapper returned `None`, so that
  programmer saw every keep as refused). It gives no edit and no CER.
- `pack/readings_budget0.md` lists three phase-0 hypotheses for this subject, written before
  any session: blank bias back to the parent's value, output layer from the parent, per-layer
  α between parent and subject. They are proposals with predictions, not results.
