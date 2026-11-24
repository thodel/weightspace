# Notes — 20261124-bern-combined-v1-f2-b10-2


## Round 0 — 2026-11-24 17:05 UTC (programmer)

- **Model id:** `claude-fable-5-2` (Claude Fable 5.2), as stated in my system prompt. Headless
  Claude Code session, default reasoning settings as given by the operator; tools used: Read,
  Write, Edit, Bash. No sub-agents, no web.
- **Read:** `PROGRAMMER_BRIEF.md`; `preregistration.md`, `notes.md`, `session.json` of this
  folder (and its directory listing; `env.txt` was listed, not opened); the pack:
  `readings_budget0.md`, `c0_c1.md`, `per_line_edits.json`, `describe_bern-combined-v1.md`,
  both `diff_*.md` (the `describe_*`/`diff_*` JSON files and the two donor `describe_*.md` were
  not opened); `data/practice/bern-train-300.jsonl` and the head of `.meta.json`; harness source
  `wsp/ops.py`, `align.py`, `session.py`, `kraken_subject.py`, `subject.py`, `metrics.py`, and
  of `run.py` the outline and lines 196–272 (`RunSession`, `open_session`); of `data.py` only
  the `load` function and the first lines of `SOURCE_NOTES` after it (set descriptions). No line image was looked at.
- **Ran (CPU, read-only, scripts in the session scratchpad, no forward pass):**
  1. per-line analysis of `pack/per_line_edits.json` by hand;
  2. the three models' tensors: task vectors from the parent, their cosines, codec alignment.
  Also `py_compile` on `program_01.py`. No `program_*.py` was executed.
- **Findings** (details in section 5 of the pre-registration):
  - 40 lines with more than half the reference wrong hold 732 of the subject's 1,577 edits; 34
    of them fail for parent and sibling too. CER without them: 0.086.
  - Parent better than the subject on 81/103 Hertwig lines, worse on 174/197 Schlussredaktion
    lines. Sibling against subject on Schlussredaktion: better 58, worse 38.
  - Task vectors: ‖subject−parent‖ 85.4, ‖sibling−parent‖ 73.5, cosine 0.52 (L_12 0.24–0.28,
    O_18 weight 0.85, bias 0.97); ‖subject−sibling‖ 78.8.
  - The parent's codec maps to the subject's rows by identity; the sibling's does **not** (it
    lacks `ƒ`, so its later rows are shifted by one). `codec_map()` has no entry for the blank;
    the program adds row 0 → row 0 itself.
  - Practice references contain 22 no-break spaces, which `normalise` flattens.
- **Plan:** `program_01.py` = C0, then H1 (average with the sibling, α 0.5), H2 (quarter step
  toward the parent), H3 (blank bias, size by the D/I rule), each on the last kept state. It
  prints CER per hand, the paired interval per hand and the confusions for every pass. Passes
  5–10 are decided after its output. No freeze in program_01.
- **Earlier sessions:** the pre-registration (sections 1–4, 6, written by the operator) says
  that earlier sessions on this subject exist and that this run replicates their conditions; a
  code comment in `wsp/run.py` names a session id `20261106-bern-combined-v1-b10-1` in
  connection with a harness bug (`keep()` returned None). Neither names an edit, a kept
  decision or a result of such a session. `readings_budget0.md` lists phase-0 hypotheses for
  this model (blank bias to the parent's value, output-layer transplant, per-layer α); these
  are written before any session and are not results.
