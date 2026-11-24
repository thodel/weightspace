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

## 2026-11-24T17:04Z — round 1 (after passes 1–3, 7 left)

- **Results** (stdout of `program_01.py`, `readings/pass_001..003`):
  - C0: 0.1456, Hertwig 0.2050, Schlussredaktion 0.1184, 29 collapsed. Reproduced.
  - H1 (all tensors 30 % toward the parent): 0.1381, Δ −0.0076 [−0.0150, −0.0009], **kept**.
    Hertwig 0.1621 (−0.043), Schlussredaktion 0.1270 (+0.0086), collapsed 22. Predicted −0.008:
    met in total; per hand both the gain and the cost were larger than predicted.
  - H2a (total 50 %): 0.1490, Δ +0.0110 [+0.0061, +0.0160] against H1; Hertwig 0.1427,
    Schlussredaktion 0.1519; collapsed 23. **Reverted** (the rule would have refused it too).
    Predicted −0.002 with "more likely refused": direction wrong, outside my range.
- **Read in the readings** (script over my own passes, with the 42 lines all models fail on set apart):
  - Hertwig, pass 2 against pass 1: 62 lines better, 7 worse; −110 edits on readable lines, −36 on
    the failed ones (line 212 alone −22: a failed line that became readable).
  - Schlussredaktion: 26 lines better, 79 worse; +96 edits on readable lines, −32 on failed ones.
    What got worse: `D→d` +12, inserted space +10, deleted `/` +9, deleted `ͤ` +5, deleted
    space +5, inserted `/` +5, `ß→s` +4, `,→/` +4. These are spelling and punctuation habits of
    the parent's training text, not misread letters. At 50 % real misreadings appear (`l→b`,
    `a→o`, `v→u`).
  - Readable lines at the kept state: 345 deletions, 148 insertions, 334 substitutions.
  - The 7 lines that left the collapsed set in H1: 51, 116, 125, 197, 203, 212, 252.
- **Conclusion:** the whole-model line toward the parent has its optimum near 0.3; nothing more
  there. The open question is whether the trade between the hands differs by depth.
- **Wrote:** round-1 paragraph in preregistration §5; `program_02.py` = H3a (pass 4: `L_16` +
  `O_18` back to the unedited subject), H3b (pass 5: conv + `L_12` to 55 % parent), H5 (pass 6:
  blank bias −0.40). Each against the last kept state, keep or revert after each. No freeze.
  Compiled only.
- **Read this round:** the operator's message, `readings/pass_001–003.jsonl`, the pack's
  `per_line_edits.json`, the practice list. Nothing else.
