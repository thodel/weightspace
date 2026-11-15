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

## 2026-11-15 08:55 UTC — round 1 (after passes 1–4, before pass 5)

**Model id:** `claude-fable-5-2`, unchanged.

**Results of `program_01.py`** (operator's stdout, `log.jsonl`, `readings/pass_001…004`):

| pass | edit | CER | Hertwig | Schlussred. | collapsed | Δ vs kept (95 % line interval) | decision |
|---|---|---:|---:|---:|---:|---|---|
| 1 | C0 | 0.1456 | 0.2050 | 0.1184 | 29 | — | reproduces `c0_c1.md` |
| 2 | H1 sibling α 0.5 | 0.1445 | 0.2015 | 0.1184 | 23 | −0.0011 [−0.0073, +0.0038] | refused |
| 3 | H2 parent α 0.25 | 0.1381 | 0.1671 | 0.1247 | 22 | −0.0076 [−0.0147, −0.0015] | kept |
| 4 | H3 blank −0.3 | 0.1360 | 0.1645 | 0.1230 | 21 | −0.0020 [−0.0037, −0.0004] | kept |

**Against the predictions.**
- H1: predicted −0.3 to −1.0 pt, got −0.11 pt. Schlussredaktion did not move (predicted −0.3
  to −1.0 pt). Wrong on the hand it was meant for. The whole change is on the unreadable
  lines (−32 edits) and collapsed fell by 6; on the other lines Hertwig got worse (+21).
- H2: total −0.76 pt, inside the predicted range; Hertwig −3.8 pt and Schlussredaktion
  +0.6 pt, both inside. Line counts: Hertwig 57 better / 7 worse, Schlussredaktion 25 / 72.
- H3: −0.20 pt, at the lower edge. Deletions fell by 31 and insertions rose by 12 on the
  readable lines, both below the predicted ranges.

**Readings.**
- The 42 lines with ≥ 50 % errors at C0: 736 edits at C0, 657 in the kept state. Several
  hypotheses there are fluent text that is not the reference (e.g. reference `notturfft / das
  erhöuschen hatt …`, read as `laͤser vnnd zuͦhoͤer …`), on Schlussredaktion lines too. For
  the operator: these look like line images that do not belong to their text.
- Readable lines (258): Hertwig 400 → 301 edits, Schlussredaktion 441 → 515. Donors there:
  parent 145 on Hertwig, sibling 395 on Schlussredaktion.
- Kept state, readable lines: space 72 deleted / 63 inserted, `/` 46 / 42. The largest
  substitutions are conventions (`D`→`d` 22, `u`→`v` 16, `i`→`I` 15, `,`→`/` 16). No
  single-row bias edit is worth a pass; that candidate of the round-0 plan is dropped.

**Caveat on H2.** Its gain is Hertwig lines the parent was trained on with correct images;
its cost is Schlussredaktion lines the parent never saw. On held-out the gain may be smaller
and the cost the same.

**Next (`program_02.py`, passes 5–8, no freeze):** H4 parent α 0.25 in `C_*` + `L_12`; H5
parent α 0.25 in `L_14`, `L_16`, `O_18`; H6 sibling α 0.2 on all tensors; H7 blank −0.3 more.
The donor steps leave the blank bias as it is. Predictions are in section 5 of
`preregistration.md` (round 1) and in the hypotheses of the program. `py_compile` passes; not
executed. Passes 9–10 are decided afterwards.

**Read this round:** the stdout given by the operator, `log.jsonl` (decisions),
`readings/pass_001…004.jsonl`, the pack's `per_line_edits.json`, the practice list. An `ls` of
the run folder showed `stdout_program_01.log`, `sandbox_program_01.cmd` and `current.mlmodel`;
I did not open them. Nothing I read named an edit, a kept decision or a result of an earlier
session on this model.

## 2026-11-15 08:58 UTC — round 2 (after passes 5–8, before the freeze)

**Model id:** `claude-fable-5-2`, unchanged.

**Results of `program_02.py`** (operator's stdout, `log.jsonl`, `readings/pass_005…008`), each
against the kept state H2 + H3 (CER 0.1360, Hertwig 0.1645, Schlussredaktion 0.1230,
collapsed 21):

| pass | edit | CER | Hertwig | Schlussred. | collapsed | Δ vs kept (95 % line interval) | decision |
|---|---|---:|---:|---:|---:|---|---|
| 5 | H4 parent α 0.25, front | 0.1398 | 0.1624 | 0.1294 | 22 | +0.0038 [+0.0014, +0.0064] | reverted (collapsed up) |
| 6 | H5 parent α 0.25, back | 0.1383 | 0.1504 | 0.1328 | 21 | +0.0023 [−0.0010, +0.0058] | refused |
| 7 | H6 sibling α 0.2 | 0.1347 | 0.1645 | 0.1211 | 21 | −0.0013 [−0.0036, +0.0010] | refused |
| 8 | H7 blank −0.3 more | 0.1345 | 0.1612 | 0.1222 | 18 | −0.0016 [−0.0036, +0.0001] | refused |

**Against the predictions.**
- H4: total +0.38 pt, at the upper edge of the range. Hertwig −0.2 pt, less than the
  predicted −0.5 to −2; Schlussredaktion +0.6 pt, inside. The front blocks do not carry much
  of the parent's Hertwig reading at this dose.
- H5: total +0.23 pt, inside. Hertwig −1.4 pt, inside; Schlussredaktion +1.0 pt, inside.
- H4 and H5 together: the cut into front and back does not separate gain from cost. Readable
  lines: front Hertwig −12 / Schlussredaktion +47, back −38 / +72; the first whole-network
  quarter gave −95 / +77. The second quarter is worse than the first in both halves.
- H6: total −0.13 pt, inside. Schlussredaktion −0.2 pt, just short of the predicted −0.3 to
  −0.8; Hertwig unchanged (predicted +0.3 to +1.0). 47 lines better, 43 worse.
- H7: total −0.16 pt, inside; deletions −74 by the block table (846 → 772; predicted −15
  to −40), insertions +20 (174 → 194), inside, substitutions +37 (453 → 490). Collapsed fell to 18, not predicted. Refused by 0.0001 at the upper
  end of the interval.

**What H7 is made of.** All of its −17 edits are on the 42 lines that are unreadable at C0;
the 258 readable lines change by −1 (Hertwig) and +1 (Schlussredaktion). The same split for
the kept H3: −15 on the unreadable lines, −7 on the readable ones. So most of the blank gain,
the kept one included, comes from the model writing more on lines it cannot read. H3 met the
rule and stands as kept; for the operator, its gain should not be expected on held-out unless
held-out has the same kind of lines, and it is the first candidate for the split control.

**Decision: no pass 9 or 10.** Candidates considered and not run:
- H6 + H7 together: the sum of their per-line changes on the kept state gives −0.28 pt with
  a line interval of [−0.62, +0.04] (computed from the two readings; not a measurement). The
  predicted outcome is a refusal; a keep would be a chance result on a combination of two
  refused edits, half of it on unreadable lines.
- Sibling at a larger dose: same ratio of change to spread expected; H1 gave nothing.
- Taking back part of the parent step in the front blocks: predicted near 0.

**`program_03.py`** runs no pass. It checks that the restored state is H2 + H3 (blank bias and
one conv tensor against 0.75 subject + 0.25 parent), logs a note and calls `s.freeze()`.
`py_compile` passes; not executed.

**Session in one paragraph.** 8 of 10 passes used, 7 edits, 2 kept. Kept: a quarter step of
all tensors toward the parent (−0.76 pt) and blank bias −0.3 (−0.20 pt); practice CER 0.1456
→ 0.1360, collapsed 29 → 21. The parent step moves Hertwig by −3.8 pt on lines the parent was
trained on and Schlussredaktion by +0.6 pt on lines it never saw. Nothing localised was
found: no block, row or scalar that removes the Hertwig damage without the Schlussredaktion
cost.

**Read this round:** the operator's stdout, `log.jsonl` (decisions), `readings/pass_001…008`,
the practice list. Nothing I read named an edit, a kept decision or a result of an earlier
session on this model.
