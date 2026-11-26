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

## 2026-11-24T17:08Z — round 2 (after passes 4–6, 4 left)

- **Results** (stdout of `program_02.py`, `readings/pass_004..006`), each against the kept H1 state:
  - H3a, `L_16` + `O_18` back to the unedited subject: 0.1384, Δ +0.0004 [−0.0028, +0.0036];
    Hertwig 0.1768, Schlussredaktion 0.1208; collapsed 24. Reverted (collapsed up; the rule would
    have refused it too).
  - H3b, conv + `L_12` to 55 % parent: 0.1444, Δ +0.0064 [+0.0030, +0.0099]; Hertwig 0.1542,
    Schlussredaktion 0.1400; collapsed 25. Reverted.
  - H5, blank bias −0.40 (now +0.562): 0.1359, Δ −0.0021 [−0.0040, −0.0004]; Hertwig 0.1586,
    Schlussredaktion 0.1255; collapsed 20. **Kept.**
- **Against my predictions:** H3a and H3b both predicted −0.002 and came out at or above 0. The
  per-hand signs were right, the sizes were not: giving the top back to the subject cost Hertwig
  three times what I expected, and the bottom alone carried most of the Schlussredaktion cost of
  the 50 % step. So the idea "conventions sit in the top, letter shapes in the bottom" is wrong
  for this model: each block trades one hand against the other. H5 was inside its range.
- **Breakdown of H5** (readable / failed lines, del-ins-sub): Hertwig readable 135-26-129 →
  105-33-146, failed 231-1-30 → 219-2-35; Schlussredaktion readable 210-122-205 → 190-133-211,
  failed 325-20-61 → 300-21-77. 34 lines better, 22 worse. The gain is spread over both hands
  and over readable and failed lines; no single line carries it (largest −4).
- **Observation:** practice line 252 (Schlussredaktion) is read at the kept state as a fluent
  sentence that has nothing to do with its reference: the crop and the text do not belong
  together. Some of the 42 failed lines are such pairs and no edit should be expected to fix them.
- **Wrote:** round-2 paragraph in preregistration §5; `program_03.py` = H4 (pass 7:
  + 0.25 × (sibling − unedited subject), all tensors) and H5b (pass 8: blank bias a further
  −0.35). No freeze; passes 9–10 are decided after these. Compiled only.
- **Read this round:** the operator's message, `readings/pass_001,002,004–006.jsonl`, the pack's
  `per_line_edits.json`, the practice list. Nothing else.

## 2026-11-24T17:10Z — round 3 (after passes 7–8, 2 left and left unspent)

- **Results** (stdout of `program_03.py`), each against the kept state (H1 + H5, 0.1359, 20 collapsed):
  - H4, + 0.25 × (sibling − unedited subject), all tensors: 0.1346, Δ −0.0013 [−0.0041, +0.0014];
    Hertwig 0.1533, Schlussredaktion 0.1261; collapsed 19. **Refused** (interval contains 0).
  - H5b, blank bias a further −0.35: 0.1350, Δ −0.0009 [−0.0032, +0.0009]; Hertwig 0.1562,
    Schlussredaktion 0.1253; collapsed 17. **Refused.**
- **Against my predictions:** H4 total as predicted (−0.0015), per hand reversed: I expected the
  sibling to help Schlussredaktion and cost Hertwig; it lowered Hertwig by 0.005 and left
  Schlussredaktion where it was. So what the average removed was more likely something the
  subject fitted on its mismatched Hertwig pages than a lack of Schlussredaktion training. That is
  a reading of one refused pass, not a finding. H5b as predicted: the return of the blank bias
  falls off after the first step.
- **Decision:** freeze now, 8 of 10 passes used. Round 2 fixed the use of passes 9–10 (a second
  step of an edit kept in passes 7–8 only). Neither was kept. Combining the two refused edits or
  enlarging H4 would be a choice made after seeing their point estimates, so I leave it.
- **Frozen state:** H1 (all 28 trained tensors 30 % toward `hertwig-v2`, output rows by character,
  rows `½ ƒ ̊` untouched) + H5 (blank bias −0.40, now +0.562). Practice 0.1456 → 0.1359; Hertwig
  0.2050 → 0.1586; Schlussredaktion 0.1184 → 0.1255 (worse); collapsed 29 → 20.
- **Session tally:** 7 edits, 2 kept (H1, H5), 3 reverted by me for a higher collapsed count (H2a,
  H3a, H3b; the rule would have refused all three as well), 2 refused by the rule (H4, H5b).
- **Caveats for whoever reads the held-out number:** the kept gain is all on Hertwig and comes
  from a donor trained on those very lines; Schlussredaktion got worse on practice. My written
  expectation for held-out is in preregistration §5, round 3.
- **Wrote:** round-3 paragraph in preregistration §5; `program_04.py` = no pass, `s.freeze()`.
  Compiled only.
- **Read this round:** the operator's message only.
