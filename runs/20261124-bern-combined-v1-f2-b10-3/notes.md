# Notes — 20261124-bern-combined-v1-f2-b10-3


## 2026-11-24T17:02Z — round 0 (before any pass)

- **Programmer model id:** `claude-fable-5-2` (Claude Fable 5.2), as stated in my system prompt.
  Headless Claude Code session, default reasoning settings as given by the harness; I set nothing.
- **Read:** `PROGRAMMER_BRIEF.md`; `preregistration.md`, `notes.md`, `session.json` of this folder;
  the pack (`readings_budget0.md`, `c0_c1.md`, `describe_bern-combined-v1.md`, both `diff_*.md`,
  `per_line_edits.json`); `data/practice/bern-train-300.jsonl` and `.meta.json` (head);
  `wsp/ops.py`, `session.py`, `align.py`, `kraken_subject.py`, `subject.py`, `metrics.py`, and of
  `wsp/run.py` and `wsp/data.py` the `RunSession`/`open_session`/`load` parts. A listing of the run
  folder showed `env.txt`; I did not open it. No line image opened.
- **Ran (CPU, no forward):** two reading scripts. (1) per-hand and per-line statistics from
  `per_line_edits.json`; (2) the three models' tensors: task vectors child − parent, their cosines,
  output biases by character, reference character counts.
- **Findings** (details in preregistration §5): 736 of the subject's 1,577 edits are in 42 lines
  that all three models fail on; the parent is 311 edits better on Hertwig, the sibling 41 better
  on Schlussredaktion; subject and sibling task vectors have cosine 0.24–0.48 in conv/`L_12`,
  rising to 0.85/0.97 in the output layer; blank bias +0.998 against the parent's +0.879.
  `\xa0` occurs 22 times in the references and is not in the codec (normalisation flattens it).
- **Wrote:** preregistration §5 (H1–H6 and the plan); `program_01.py` = C0, H1 (all tensors 30 %
  toward the parent), H2 (branch: total 50 % if H1 kept; 50 % from C0 if H1 lowered CER but was
  refused; 12 % if H1 raised CER). No freeze. Compiled with `py_compile` only, not executed.
- **Earlier sessions:** the preregistration and two source comments say that earlier sessions on
  this subject exist (`wsp/run.py` names `20261106-bern-combined-v1-b10-1` for a harness bug:
  `keep()` returned None). None of what I read names an edit, a kept decision or a result of
  such a session. `readings_budget0.md` lists phase-0 hypotheses (blank bias to the parent's
  value, output layer from the parent, per-layer α), written before any session.
